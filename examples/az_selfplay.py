"""
AlphaZero-style self-play baseline for the JAX Generals.io simulator.

    python examples/az_selfplay.py --games 4 --sims 16 --size 6 --max-steps 60 --output az.pt

What it is
  * A conv policy/value network. Its input is ONLY the player's own fog-of-war
    observation (`get_observation`), so the network never sees hidden state.
  * PUCT tree search over one player's own actions (move cell x direction x
    split, plus pass). Policy targets are the root visit counts.
  * Both sides are played by the same net and searched every tick; the two
    actions are applied simultaneously through `generals.core.game.step`.
  * Value targets are the final outcome per player: +1 win, -1 loss, 0 for a
    draw (truncation at --max-steps). Terminal states inside the tree use the
    same values and are never evaluated by the network.

Perfect-information benchmark (explicit simplification)
  The search expands nodes with the *true* simulator state, so tree transitions
  know the real hidden board, and the opponent's simultaneous action inside the
  tree is sampled once per edge from the network's policy on the opponent's own
  fog observation (the true state is the model). This is a perfect-info-search
  baseline, not a fair imperfect-information agent: do not compare it with
  fog-limited agents as if it were one. Only the network input is fog-limited.
  Opponent sampling makes edges determinized (open-loop in the opponent), which
  is a known approximation to simultaneous-move games.
"""
import argparse
import math
import random

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from generals import GeneralsEnv, get_observation
from generals.core.action import compute_valid_move_mask_obs
from generals.core.game import step as game_step

C_PUCT = 1.5
CHANNELS = 11


def num_actions(h, w):
    return h * w * 8 + 1  # (cell, direction, split) ..., last index = pass


def decode(idx, h, w):
    """Policy index -> simulator action [pass, row, col, direction, split]."""
    if idx == h * w * 8:
        return [1, 0, 0, 0, 0]
    cell, rest = divmod(idx, 8)
    return [0, cell // w, cell % w, rest // 2, rest % 2]


def features(obs):
    """Fog observation -> (CHANNELS, H, W) float array. Own view only."""
    a = np.asarray(obs.armies, dtype=np.float32)
    planes = [np.log1p(a) / 5.0 * np.asarray(obs.owned_cells), np.log1p(a) / 5.0 * np.asarray(obs.opponent_cells),
              np.log1p(a) / 5.0 * np.asarray(obs.neutral_cells)]
    planes += [np.asarray(m, dtype=np.float32) for m in (
        obs.generals, obs.castles, obs.mountains, obs.owned_cells, obs.opponent_cells, obs.fog_cells,
        obs.structures_in_fog)]
    t = float(obs.timestep) / 500.0
    planes.append(np.full(a.shape, t, dtype=np.float32))
    return np.stack(planes).astype(np.float32)


def legal_mask(obs, h, w):
    m = np.asarray(compute_valid_move_mask_obs(obs))  # (H, W, 4)
    m = np.repeat(m.reshape(h * w, 4, 1), 2, axis=2).reshape(-1)  # split flag free
    return np.concatenate([m, [True]])


class Net(nn.Module):
    def __init__(self, h, w, width=32, blocks=2):
        super().__init__()
        self.inp = nn.Conv2d(CHANNELS, width, 3, padding=1)
        self.body = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in range(blocks)])
        self.pol = nn.Conv2d(width, 8, 1)  # per cell: 4 dirs x 2 splits
        self.pass_logit = nn.Linear(width, 1)
        self.val = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x):
        x = F.relu(self.inp(x))
        for c in self.body:
            x = F.relu(x + c(x))
        logits = self.pol(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1)  # (B, H*W*8), cell-major
        g = x.mean((2, 3))
        return torch.cat([logits, self.pass_logit(g)], 1), self.val(g).squeeze(1)


@torch.no_grad()
def evaluate(net, obs_list, masks):
    """Batch of fog observations -> masked priors (B, A) and values (B,)."""
    x = torch.from_numpy(np.stack([features(o) for o in obs_list]))
    logits, v = net(x)
    logits = logits.masked_fill(~torch.from_numpy(np.stack(masks)), -1e9)
    return F.softmax(logits, 1).numpy(), v.numpy()


class Node:
    """Search node for player `me`. Stores priors for both players (one batched eval)."""

    def __init__(self, state, prior, opp_prior, value):
        self.state, self.prior, self.opp_prior, self.value = state, prior, opp_prior, value
        self.n = np.zeros_like(prior)
        self.w = np.zeros_like(prior)
        self.children = {}  # action idx -> (Node | None, terminal value | None)


def outcome(state, me, max_steps):
    """Terminal value for `me`, or None if the game goes on. Draw = 0."""
    win = int(state.winner)
    if win >= 0:
        return 1.0 if win == me else -1.0
    return 0.0 if int(state.time) >= max_steps else None


class Searcher:
    def __init__(self, net, h, w, max_steps, rng):
        self.net, self.h, self.w, self.max_steps, self.rng = net, h, w, max_steps, rng

    def make_node(self, state, me):
        obs = [get_observation(state, me), get_observation(state, 1 - me)]
        masks = [legal_mask(o, self.h, self.w) for o in obs]
        p, v = evaluate(self.net, obs, masks)
        return Node(state, p[0], p[1], float(v[0]))

    def search(self, state, me, sims, noise=True):
        root = self.make_node(state, me)
        if noise:
            legal = np.flatnonzero(root.prior > 0)
            root.prior = root.prior.copy()
            root.prior[legal] = 0.75 * root.prior[legal] + 0.25 * np.random.dirichlet([0.3] * len(legal))
        for _ in range(sims):
            self.simulate(root, me)
        return root.n.copy()

    def simulate(self, root, me):
        path, node = [], root
        while True:
            total = node.n.sum()
            q = np.where(node.n > 0, node.w / np.maximum(node.n, 1), 0.0)
            u = q + C_PUCT * node.prior * math.sqrt(total + 1) / (1 + node.n)
            a = int(np.argmax(np.where(node.prior > 0, u, -1e9)))
            path.append((node, a))
            if a not in node.children:
                v = self.expand(node, a, me)
                break
            child, term = node.children[a]
            if child is None:
                v = term
                break
            node = child
        for nd, act in reversed(path):  # outcome stays from `me` view; no sign flip (simultaneous moves)
            nd.n[act] += 1
            nd.w[act] += v

    def expand(self, node, a, me):
        opp_a = int(self.rng.choice(len(node.opp_prior), p=node.opp_prior / node.opp_prior.sum()))
        acts = [None, None]
        acts[me], acts[1 - me] = decode(a, self.h, self.w), decode(opp_a, self.h, self.w)
        state, _ = game_step(node.state, jnp.asarray(acts, dtype=jnp.int32))
        term = outcome(state, me, self.max_steps)
        if term is not None:
            node.children[a] = (None, term)
            return term
        child = self.make_node(state, me)
        node.children[a] = (child, None)
        return child.value


def self_play(net, h, w, sims, max_steps, key, rng, env):
    """One game; returns list of (features, pi, mask, z) for both players, and result string."""
    state = env.init_state(key)
    searcher = Searcher(net, h, w, max_steps, rng)
    rec = [[], []]
    while True:
        res = outcome(state, 0, max_steps)
        if res is not None:
            break
        acts = [None, None]
        for me in (0, 1):
            obs = get_observation(state, me)
            counts = searcher.search(state, me, sims)
            pi = counts / counts.sum()
            temp = 1.0 if int(state.time) < 10 else 0.25
            p = pi ** (1 / temp)
            a = int(rng.choice(len(p), p=p / p.sum()))
            acts[me] = decode(a, h, w)
            rec[me].append((features(obs), pi.astype(np.float32), legal_mask(obs, h, w)))
        state, _ = game_step(state, jnp.asarray(acts, dtype=jnp.int32))
    samples = []
    for me in (0, 1):
        z = outcome(state, me, max_steps)
        samples += [(f, pi, m, z) for f, pi, m in rec[me]]
    return samples, ("draw" if res == 0.0 else f"player {int(state.winner)} wins")


def train_step(net, opt, samples, epochs=2, batch=64):
    x = torch.from_numpy(np.stack([s[0] for s in samples]))
    pi = torch.from_numpy(np.stack([s[1] for s in samples]))
    m = torch.from_numpy(np.stack([s[2] for s in samples]))
    z = torch.tensor([s[3] for s in samples], dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), batch):
            j = perm[i:i + batch]
            logits, v = net(x[j])
            logp = F.log_softmax(logits.masked_fill(~m[j], -1e9), 1)
            loss_p = -(pi[j] * logp).sum(1).mean()
            loss_v = F.mse_loss(v, z[j])
            opt.zero_grad()
            (loss_p + loss_v).backward()
            opt.step()
    return loss_p.item(), loss_v.item()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--size", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="alphazero.pt")
    args = ap.parse_args(argv)
    if args.games < 1 or args.sims < 1 or args.size < 4 or args.max_steps < 1:
        ap.error("--games, --sims, --max-steps must be positive and --size at least 4")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    h = w = args.size
    env = GeneralsEnv(grid_dims=(h, w), truncation=args.max_steps)
    net = Net(h, w)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    key = jrandom.PRNGKey(args.seed)
    buffer = []
    for g in range(args.games):
        key, k = jrandom.split(key)
        samples, result = self_play(net, h, w, args.sims, args.max_steps, k, rng, env)
        buffer = (buffer + samples)[-20000:]
        lp, lv = train_step(net, opt, buffer)
        print(f"game {g + 1}/{args.games}: {result}, {len(samples)} samples, policy loss {lp:.3f}, value loss {lv:.3f}")
    torch.save({"model": net.state_dict(), "args": vars(args)}, args.output)
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
