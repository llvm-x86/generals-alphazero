"""
Behaviour-cloning warm start for az_selfplay: clone ExpanderAgent, then self-play fine-tune.

    python examples/bc_warmstart.py --bc-games 200 --bc-epochs 6 --size 6 --max-steps 150 --output bc.pt
    python examples/az_selfplay.py --resume --output bc.pt --size 6 --max-steps 150 --sims 4 --games 6   # fine-tune

Teacher data: Expander-vs-Expander games on make_env boards; odd games execute a uniformly random legal move
with prob EPS (the label stays the clean Expander action, so the data covers off-teacher states). Both seats'
observation histories (with memory planes) are recorded. Targets: teacher action (policy cross-entropy over
legal moves), final outcome (+-1, or the score proxy when truncated), belief label as in az_selfplay.train_step.
Games are split BY GAME (every 5th game is held out); held-out teacher-action accuracy is printed.
The checkpoint has az_selfplay's schema (empty replay, completed=0) so --resume loads it with the same
--size/--sims/--max-steps/--seed/--attn/--frames.
"""
import argparse
import importlib.util
import pathlib

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import torch

from generals import get_observation
from generals.agents import ExpanderAgent
from generals.core.action import DIRECTIONS
from generals.core.game import step as game_step

_spec = importlib.util.spec_from_file_location("az_selfplay", pathlib.Path(__file__).with_name("az_selfplay.py"))
az = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(az)
_spec = importlib.util.spec_from_file_location("eval_agents", pathlib.Path(__file__).with_name("eval_agents.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

EPS = 0.15
VAL_SEED, TEST_SEED = 5_000, 9_000  # eval map seeds (train maps use SEED_BASE+; --val/--test are disjoint from both)
SEED_BASE = 100_000  # teacher map seeds; eval_agents uses seeds from --seed (default 0), so they never overlap


def encode(action, h, w):
    """Inverse of az.decode: simulator action [pass, row, col, direction, split] -> policy index."""
    p, r, c, d, s = (int(v) for v in action)
    return h * w * 8 if p else (r * w + c) * 8 + d * 2 + s


def argmax_expander(obs, h, w):
    """Deterministic Expander: its move scores (see ExpanderAgent) but argmax with first-index ties, and when no
    capture exists, the legal full move of the largest army (first index) instead of a uniformly random one."""
    mask = az.legal_mask(obs, h, w)[:-1].reshape(h * w, 4, 2)[:, :, 0]  # full moves
    if not mask.any():
        return h * w * 8
    arm, own = np.asarray(obs.armies), np.asarray(obs.owned_cells)
    opp, score = np.asarray(obs.opponent_cells), np.zeros((h * w, 4))
    for d, (di, dj) in enumerate(np.asarray(DIRECTIONS)):
        for c in np.flatnonzero(mask[:, d]):
            r, q = divmod(int(c), w)
            tr, tc = min(max(r + di, 0), h - 1), min(max(q + dj, 0), w - 1)
            if arm[r, q] > arm[tr, tc] + 1:
                sc = float(arm[r, q])
                if not own[tr, tc]:
                    sc *= 10.0 * (2.0 if opp[tr, tc] else 1.0)
                score[c, d] = sc
    if score.max() <= 0:
        score = np.where(mask, arm.reshape(-1, 1), -1.0)
    c, d = divmod(int(score.argmax()), 4)
    return c * 8 + d * 2


def rollout(env, size, seed, max_steps, rng, eps, teacher="sample", nets=(None, None)):
    """One game -> samples (features, onehot teacher pi, mask, z, enemy general cell) from both seats.

    Labels are always the teacher's action (Expander sample, or argmax_expander per `teacher`). A seat with a net in
    `nets` executes that net's argmax (DAgger: student-visited states, teacher labels) instead of the teacher.
    With prob `eps` a teacher-executed move is a uniformly random legal one."""
    h = w = size
    state, key = env.init_state(jrandom.PRNGKey(seed)), jrandom.PRNGKey(seed + 1)
    agents, hist, rec = [ExpanderAgent(), ExpanderAgent()], [None, None], [[], []]
    while az.outcome(state, 0, max_steps) is None:
        acts = []
        for me in (0, 1):
            obs = get_observation(state, me)
            key, k = jrandom.split(key)
            mask = az.legal_mask(obs, h, w)
            if teacher != "sample":
                a = argmax_expander(obs, h, w)
            else:
                a = encode(np.asarray(agents[me].act(obs, k)), h, w)
            if not mask[a]:  # split of a 2-army cell is masked (equals full move)
                a = a ^ 1 if a < h * w * 8 and mask[a ^ 1] else h * w * 8
            hist[me] = az.stack_with(hist[me], az.features(obs, hist[me]))
            g = state.general_positions[1 - me]
            pi = np.zeros(az.num_actions(h, w), np.float32)
            pi[a] = 1
            rec[me].append((hist[me], pi, mask, int(g[0]) * w + int(g[1])))
            if nets[me] is not None:
                a = int(np.argmax(az.evaluate(nets[me], [hist[me]], [mask])[0][0]))
            elif rng.random() < eps:
                a = int(rng.choice(np.flatnonzero(mask)))
            acts.append(az.decode(a, h, w))
        state, _ = game_step(state, jnp.asarray(acts, dtype=jnp.int32), general_trade=True)
    return [(f, pi, m, az.outcome(state, me, max_steps), g) for me in (0, 1) for f, pi, m, g in rec[me]]


@torch.no_grad()
def accuracy(net, samples):
    hit = 0
    for i in range(0, len(samples), 256):
        s = samples[i:i + 256]
        logits = net(torch.from_numpy(np.stack([x[0] for x in s])))[0]
        logits = logits.masked_fill(~torch.from_numpy(np.stack([x[2] for x in s])), -1e9)
        hit += int((logits.argmax(1).numpy() == np.array([x[1].argmax() for x in s])).sum())
    return hit / len(samples)


def eval_vs_random(path, size, max_steps, seed, games):
    """Policy-only argmax checkpoint vs Random, side-swapped; returns real [wins, losses, draws]."""
    return ev.run(f"ckpt:{path}", "random", size, games, max_steps, seed)[0]


def fmt(real, n):
    lo, hi = ev.wilson(int(real[0]), n)
    return f"W/L/D {real[0]}/{real[1]}/{real[2]} of {n}, win rate {real[0] / n:.3f} Wilson95 [{lo:.3f}, {hi:.3f}]"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bc-games", type=int, default=100)
    ap.add_argument("--bc-epochs", type=int, default=5)
    ap.add_argument("--size", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--teacher", choices=("sample", "argmax"), default="sample",
                    help="sample = ExpanderAgent as is (noisy labels); argmax = deterministic Expander")
    ap.add_argument("--eps", type=float, default=EPS, help="random-move prob in odd teacher games (labels stay clean)")
    ap.add_argument("--dagger-rounds", type=int, default=0, help="rounds of student-rollout games labelled by the teacher")
    ap.add_argument("--dagger-games", type=int, default=40)
    ap.add_argument("--dagger-epochs", type=int, default=3)
    ap.add_argument("--eval-games", type=int, default=0, help="after training, evaluate argmax vs Random on --eval-seed")
    ap.add_argument("--eval-seed", type=int, default=VAL_SEED)
    ap.add_argument("--sims", type=int, default=32, help="recorded in the checkpoint; must match the later --resume run")
    ap.add_argument("--attn", choices=("dense", "pisa", "hybrid"), default="dense")
    ap.add_argument("--frames", type=int, default=az.FRAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="bc.pt")
    args = ap.parse_args(argv)
    if args.bc_games < 5 or args.bc_epochs < 1 or args.size < 4 or args.max_steps < 1 or args.dagger_rounds < 0:
        ap.error("need --bc-games >= 5, --bc-epochs >= 1, --size >= 4, --max-steps >= 1, --dagger-rounds >= 0")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    h = args.size
    env = az.make_env(h, args.max_steps)
    train, held = [], []
    for g in range(args.bc_games):
        (held if g % 5 == 4 else train).extend(
            rollout(env, h, SEED_BASE + 2 * g, args.max_steps, rng, args.eps if g % 2 else 0.0, args.teacher))
    print(f"teacher data: {len(train)} train samples from {args.bc_games - args.bc_games // 5} games, "
          f"{len(held)} held-out samples from {args.bc_games // 5} games", flush=True)
    net = az.Net(h, h, frames=args.frames, attn=args.attn)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for e in range(args.bc_epochs):
        metrics = az.train_step(net, opt, train, epochs=1)
        lp, lv = metrics["online"]["policy"], metrics["online"]["value"]
        print(f"epoch {e + 1}/{args.bc_epochs}: online policy loss {lp:.3f}, value loss {lv:.3f}, "
              f"held-out teacher-action accuracy {accuracy(net, held):.3f}", flush=True)
    for r in range(args.dagger_rounds):  # DAgger: student plays (seat alternates, teacher/student opponent), teacher labels
        net.eval()
        base = SEED_BASE + 1_000_000 * (r + 1)
        new = []
        for g in range(args.dagger_games):
            nets = (net, net) if g % 2 else (net, None)
            new.extend(rollout(env, h, base + 2 * g, args.max_steps, rng, 0.0, args.teacher, nets))
        net.train()
        train += new
        print(f"dagger round {r + 1}: +{len(new)} student-visited samples, "
              f"accuracy on them before retraining {accuracy(net, new):.3f}", flush=True)
        for _ in range(args.dagger_epochs):
            az.train_step(net, opt, train, epochs=1)
        print(f"  after {args.dagger_epochs} epochs: held-out accuracy {accuracy(net, held):.3f}", flush=True)
    net.eval()
    print(f"train-set teacher-action accuracy {accuracy(net, train[:2000]):.3f} (first 2000 samples)")
    ck = {"model": net.state_dict(), "optimizer": opt.state_dict(), "args": {**vars(args), "games": 0, "belief": "off"},
          "channels": az.CHANNELS, "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
          "jax_key": torch.from_numpy(np.asarray(jrandom.PRNGKey(args.seed)).copy()), "completed": 0, "replay": []}
    torch.save(ck, args.output)
    print("saved", args.output)
    if args.eval_games:
        real = eval_vs_random(args.output, h, args.max_steps, args.eval_seed, args.eval_games)
        print(f"policy argmax vs Random, seeds {args.eval_seed}..: {fmt(real, args.eval_games)}")


if __name__ == "__main__":
    main()
