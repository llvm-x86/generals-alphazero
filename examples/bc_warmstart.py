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
from generals.core.game import step as game_step

_spec = importlib.util.spec_from_file_location("az_selfplay", pathlib.Path(__file__).with_name("az_selfplay.py"))
az = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(az)

EPS = 0.15
SEED_BASE = 100_000  # teacher map seeds; eval_agents uses seeds from --seed (default 0), so they never overlap


def encode(action, h, w):
    """Inverse of az.decode: simulator action [pass, row, col, direction, split] -> policy index."""
    p, r, c, d, s = (int(v) for v in action)
    return h * w * 8 if p else (r * w + c) * 8 + d * 2 + s


def teacher_game(env, size, seed, max_steps, rng, eps):
    """One Expander-vs-Expander game -> samples (features, onehot pi, mask, z, enemy general cell)."""
    h = w = size
    state, key = env.init_state(jrandom.PRNGKey(seed)), jrandom.PRNGKey(seed + 1)
    agents, hist, rec = [ExpanderAgent(), ExpanderAgent()], [None, None], [[], []]
    while az.outcome(state, 0, max_steps) is None:
        acts = []
        for me in (0, 1):
            obs = get_observation(state, me)
            key, k = jrandom.split(key)
            mask = az.legal_mask(obs, h, w)
            a = encode(np.asarray(agents[me].act(obs, k)), h, w)
            if not mask[a]:  # split of a 2-army cell is masked (equals full move)
                a = a ^ 1 if a < h * w * 8 and mask[a ^ 1] else h * w * 8
            hist[me] = az.stack_with(hist[me], az.features(obs, hist[me]))
            g = state.general_positions[1 - me]
            pi = np.zeros(az.num_actions(h, w), np.float32)
            pi[a] = 1
            rec[me].append((hist[me], pi, mask, int(g[0]) * w + int(g[1])))
            if rng.random() < eps:
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bc-games", type=int, default=100)
    ap.add_argument("--bc-epochs", type=int, default=5)
    ap.add_argument("--size", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--sims", type=int, default=32, help="recorded in the checkpoint; must match the later --resume run")
    ap.add_argument("--attn", choices=("dense", "pisa", "hybrid"), default="dense")
    ap.add_argument("--frames", type=int, default=az.FRAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="bc.pt")
    args = ap.parse_args(argv)
    if args.bc_games < 5 or args.bc_epochs < 1 or args.size < 4 or args.max_steps < 1:
        ap.error("need --bc-games >= 5, --bc-epochs >= 1, --size >= 4, --max-steps >= 1")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    h = args.size
    env = az.make_env(h, args.max_steps)
    train, held = [], []
    for g in range(args.bc_games):
        (held if g % 5 == 4 else train).extend(teacher_game(env, h, SEED_BASE + 2 * g, args.max_steps, rng, EPS if g % 2 else 0.0))
    print(f"teacher data: {len(train)} train samples from {args.bc_games - args.bc_games // 5} games, "
          f"{len(held)} held-out samples from {args.bc_games // 5} games", flush=True)
    net = az.Net(h, h, frames=args.frames, attn=args.attn)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for e in range(args.bc_epochs):
        lp, lv = az.train_step(net, opt, train, epochs=1)
        print(f"epoch {e + 1}/{args.bc_epochs}: last-batch policy loss {lp:.3f}, value loss {lv:.3f}, "
              f"held-out teacher-action accuracy {accuracy(net, held):.3f}", flush=True)
    print(f"train-set teacher-action accuracy {accuracy(net, train[:2000]):.3f} (first 2000 samples)")
    ck = {"model": net.state_dict(), "optimizer": opt.state_dict(), "args": {**vars(args), "games": 0, "belief": "off"},
          "channels": az.CHANNELS, "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
          "jax_key": torch.from_numpy(np.asarray(jrandom.PRNGKey(args.seed)).copy()), "completed": 0, "replay": []}
    torch.save(ck, args.output)
    print("saved", args.output)


if __name__ == "__main__":
    main()
