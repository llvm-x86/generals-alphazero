"""
Play agent A against agent B on SxS boards, side-swapped, with fixed seeds.

    python examples/eval_agents.py expander random --size 8 --games 200 --max-steps 200
    python examples/eval_agents.py ckpt:az.pt expander --size 6 --games 20 --sims 8

Agents: random, expander, hunter, harvester, ckpt:<path> (az_selfplay checkpoint; policy-only argmax, or the
current source's Gumbel-root/PUCT-interior search with --sims N: a standardized engine applied to the checkpoint's
weights and recorded heur/truncation, not necessarily the original actor; legacy pre-schema checkpoints load eval-only). Game i uses map seed+i//2 and puts A in seat i%2, so each map is
played from both seats. A truncated game counts as a draw; a second table adjudicates draws by the
sign of az_selfplay.score (material proxy, not a real win). Intervals are Wilson 95% for game counts and
bootstrap 95% over paired maps for match score (win=1, draw=0.5, loss=0).
"""
import argparse
import importlib.util
import math
import pathlib

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import torch

from generals import get_observation
from generals.agents import ExpanderAgent, HunterAgent, RandomAgent
from generals.agents.harvester_agent import HarvesterAgent
from generals.core.game import step as game_step

_spec = importlib.util.spec_from_file_location("az_selfplay", pathlib.Path(__file__).with_name("az_selfplay.py"))
az = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(az)

BUILTIN = {"random": RandomAgent, "expander": ExpanderAgent, "hunter": HunterAgent, "harvester": HarvesterAgent}


class Builtin:
    def __init__(self, cls):
        self.agent = cls()

    def reset(self, seed=None):
        self.agent.reset()

    def act(self, state, me, key):
        return np.asarray(self.agent.act(get_observation(state, me), key))


class Ckpt:
    """az_selfplay checkpoint. sims=0: argmax of the masked policy; sims>0: Gumbel root search, best survivor."""

    def __init__(self, path, sims, max_steps, size, belief=False):
        ck = az.load_checkpoint(path)
        a = ck["args"]
        if a["size"] != size:
            raise SystemExit(f"checkpoint was trained for --size {a['size']}, not {size}")
        if ck.get("schema") is None:
            print(f"note: {path} is a legacy (pre-schema) checkpoint: eval-only; weights run under CURRENT source with "
                  f"heur={a.get('heur', 0.0)}, truncation={a.get('truncation', 'proxy')}, not the original actor")
        if a.get("max_steps", max_steps) != max_steps:
            print(f"note: checkpoint trained with --max-steps {a['max_steps']}, evaluating at {max_steps}")
        # search/value settings the checkpoint was trained with: per-agent, never shared module state
        # (legacy checkpoints without a record were trained with the proxy truncation value)
        self.cfg = az.SearchCfg(heur=a.get("heur", 0.0), truncation=a.get("truncation", "proxy"), belief=belief,
                                incremental=a.get("incremental", False))
        az.FAST = az.fastkernels.available()  # matches torch to ~3e-7 (tests/test_fastkernels.py)
        self.net = az.Net(size, size, frames=a["frames"], attn=a["attn"], stream=a.get("incremental", False),
                          order=a.get("hybrid_order"))
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.sims, self.size = sims, size
        self.rng = np.random.default_rng(0)
        self.searcher = az.Searcher(self.net, size, size, max_steps, self.rng, self.cfg)
        self.hist, self.mem = None, az.GeneralMemory()

    def reset(self, seed=None):
        self.hist, self.mem = None, az.GeneralMemory()
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.searcher.rng = self.rng

    def act(self, state, me, key):
        obs = get_observation(state, me)
        self.mem.update(obs)
        if self.sims:
            self.searcher.search(state, me, self.sims, self.hist, noise=False, mem=self.mem)
            a = self.searcher.best
        else:
            mask = az.legal_mask(obs, self.size, self.size)
            stack = az.stack_with(self.hist, az.features(obs, self.hist), self.net.frames)
            p, _ = az.evaluate(self.net, [stack], [mask])
            a = int(np.argmax(p[0]))
        self.hist = az.stack_with(self.hist, az.features(obs, self.hist), self.net.frames)
        return np.asarray(az.decode(a, self.size, self.size))


def make_agent(name, sims, max_steps, size, belief=False):
    if name.startswith("ckpt:"):
        return Ckpt(name[5:], sims, max_steps, size, belief)
    if name not in BUILTIN:
        raise SystemExit(f"unknown agent {name!r}; choose from {sorted(BUILTIN)} or ckpt:<path>")
    return Builtin(BUILTIN[name])


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - r), min(1.0, c + r)


def play(env, agents, state, key, max_steps):
    """One game; agents[i] sits in seat i. Returns (winner seat or -1, seat-0 score-proxy sign)."""
    for a in agents:
        a.reset(int(np.asarray(key)[1]))  # both seats on paired maps reuse the same search-randomness seed
    while int(state.winner) < 0 and int(state.time) < max_steps:
        key, *ks = jrandom.split(key, 3)
        acts = jnp.asarray(np.stack([agents[i].act(state, i, ks[i]) for i in (0, 1)]), dtype=jnp.int32)
        state, _ = game_step(state, acts, general_trade=True)
    win = int(state.winner)
    return win, float(az.score(state, 0))  # real material proxy, only consulted for draws


def run(name_a, name_b, size, games, max_steps, seed, sims=0, belief=False, return_pairs=False):
    env = az.make_env(size, max_steps)
    a, b = make_agent(name_a, sims, max_steps, size, belief), make_agent(name_b, sims, max_steps, size, belief)
    real, adj = np.zeros(3, int), np.zeros(3, int)  # A's [wins, losses, draws]
    pairs = []
    for g in range(games):
        a_seat = g % 2
        state = env.init_state(jrandom.PRNGKey(seed + g // 2))
        win, proxy = play(env, [a, b] if a_seat == 0 else [b, a], state, jrandom.PRNGKey(10_000 + seed + g // 2), max_steps)
        if win >= 0:
            i = 0 if win == a_seat else 1
            real[i] += 1
            adj[i] += 1
        else:
            real[2] += 1
            s = proxy if a_seat == 0 else -proxy
            adj[0 if s > 0 else 1 if s < 0 else 2] += 1
        points = 0.5 if win < 0 else float(win == a_seat)
        if g % 2:
            pairs.append((first_seat + points) / 2)
        else:
            first_seat = points
    return (real, adj, pairs) if return_pairs else (real, adj)


def paired_interval(scores, reps=5000):
    """Percentile bootstrap over MAP pairs, not independent seats. Seed fixed so reports reproduce."""
    scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(0)
    means = rng.choice(scores, size=(reps, len(scores)), replace=True).mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def report(title, name_a, name_b, counts, games):
    print(title)
    for label, name, k in zip(("wins", "losses", "draws"), (name_a,) * 3, counts):
        lo, hi = wilson(int(k), games)
        print(f"  {name_a} {label}: {k}/{games} = {k / games:.3f}  Wilson95 [{lo:.3f}, {hi:.3f}]")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent_a")
    ap.add_argument("agent_b")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sims", type=int, default=0, help="root-search simulations per ckpt move (0 = policy argmax)")
    ap.add_argument("--belief", action="store_true", help="ckpt agents sample determinizations from the belief head")
    args = ap.parse_args(argv)
    if args.games < 1 or args.size < 4 or args.max_steps < 1 or args.sims < 0:
        ap.error("need --games >= 1, --size >= 4, --max-steps >= 1, --sims >= 0")
    torch.set_num_threads(1)
    real, adj, pairs = run(args.agent_a, args.agent_b, args.size, args.games, args.max_steps, args.seed, args.sims, args.belief, True)
    print(f"{args.agent_a} (A) vs {args.agent_b} (B): {args.games} games on {args.size}x{args.size}, "
          f"max_steps {args.max_steps}, seeds {args.seed}..{args.seed + (args.games - 1) // 2}, sims {args.sims}")
    report("real outcomes (truncation = draw)", args.agent_a, args.agent_b, real, args.games)
    report("draws adjudicated by material proxy (sign of az_selfplay.score; NOT real wins)", args.agent_a, args.agent_b, adj, args.games)
    if len(pairs) > 1 and min(pairs) == max(pairs):
        print(f"paired map score: {pairs[0]:.3f} over {len(pairs)} maps; no observed variation, "
              "so a bootstrap interval would be degenerate and is not evidence of equivalence")
    elif len(pairs) > 1:
        lo, hi = paired_interval(pairs)
        print(f"paired map score (W=1,D=0.5,L=0): {np.mean(pairs):.3f}, bootstrap95 [{lo:.3f}, {hi:.3f}] "
              f"over {len(pairs)} complete two-seat maps"
              + ("; last unpaired game excluded" if args.games % 2 else ""))
    elif pairs:
        print(f"paired map score: {pairs[0]:.3f} over one map; too few pairs for an uncertainty interval")
    else:
        print("paired map score: unavailable (fewer than two games)")
    return real, adj


if __name__ == "__main__":
    main()
