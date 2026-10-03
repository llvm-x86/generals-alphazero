"""
AlphaZero-style self-play baseline for the JAX Generals.io simulator.

    python examples/az_selfplay.py --games 4 --sims 16 --size 6 --max-steps 60 --output az.pt

What it is
  * The network sees only the player's fog-limited observation, including
    public army/land totals. A pixel-level spacetime transformer (a token per cell
    per frame over the last 4 own observations, no pooling) is trained
    from PUCT visits and final game outcomes.
  * Search samples hidden boards consistent with the player's observation
    and public totals. Four independent root samples reduce dependence on a
    single guess; opponents are sampled on every search visit.
  * Both sides use the same network and move simultaneously.
  * Unfinished games at --max-steps receive a bounded score-advantage proxy
    (not a win); games ending in a general capture use +1/-1.

Belief limitation: sampled fog tiles do not preserve remembered terrain or
past sightings; the opponent model is sampled rather than adversarial search.
This is not a ranked-ready agent. Evaluate separately before live play.
"""
import argparse
import math
import os
from collections import deque

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
CHANNELS = 15
FRAMES = 4  # past own-observation frames the net attends over (fog memory)


def stack_with(prev, frame):
    """Append a frame to a (FRAMES, C, H, W) history; first call repeats the frame."""
    if prev is None:
        return np.repeat(frame[None], FRAMES, 0)
    return np.concatenate([prev[1:], frame[None]])


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
    counts = (obs.owned_land_count, obs.owned_army_count, obs.opponent_land_count, obs.opponent_army_count)
    planes += [np.full(a.shape, np.log1p(float(v)) / 5.0, dtype=np.float32) for v in counts]
    planes.append(np.full(a.shape, float(obs.timestep) / 500.0, dtype=np.float32))
    return np.stack(planes)


def legal_mask(obs, h, w):
    m = np.asarray(compute_valid_move_mask_obs(obs))  # (H, W, 4)
    moves = np.repeat(m.reshape(h * w, 4, 1), 2, axis=2)
    moves[:, :, 1] &= np.asarray(obs.armies).reshape(-1, 1) > 2  # half of two equals full
    return np.concatenate([moves.reshape(-1), [True]])


def determinize(state, me, rng):
    """Sample a board from one player's observation, never from hidden tiles."""
    obs = get_observation(state, me)
    fog = np.asarray(obs.fog_cells | obs.structures_in_fog)
    armies = np.asarray(obs.armies).copy()
    own = np.asarray(obs.owned_cells).copy()
    enemy = np.asarray(obs.opponent_cells).copy()
    hidden = np.flatnonzero(fog)
    unseen_land = max(0, int(obs.opponent_land_count) - int(enemy.sum()))
    # ponytail: uniform hidden-land prior; use a learned belief model once replay data justifies it.
    chosen = rng.choice(hidden, size=min(unseen_land, len(hidden)), replace=False)
    enemy.flat[chosen] = True
    mountains = np.asarray(obs.mountains).copy()
    castles = np.asarray(obs.castles).copy()
    structures = np.asarray(obs.structures_in_fog).copy()
    hidden_castles = (rng.random(structures.shape) < 0.2) & structures
    hidden_castles |= structures & enemy
    castles |= hidden_castles
    mountains |= structures & ~hidden_castles
    armies[hidden_castles & ~enemy] = 40
    unseen_army = max(0, int(obs.opponent_army_count) - int((armies * enemy).sum()))
    if len(chosen):
        armies.flat[chosen] = rng.multinomial(unseen_army, np.full(len(chosen), 1 / len(chosen)))
    generals = np.asarray(obs.generals).copy()
    positions = np.asarray(state.general_positions).copy()
    own_general = np.argwhere(generals & own)
    enemy_general = np.argwhere(generals & enemy)
    if len(own_general):
        positions[me] = own_general[0]
    if len(enemy_general):
        positions[1 - me] = enemy_general[0]
    elif len(chosen):
        candidates = chosen[~structures.flat[chosen]]
        positions[1 - me] = np.unravel_index(rng.choice(candidates if len(candidates) else chosen), own.shape)
        generals[tuple(positions[1 - me])] = True
    ownership = np.stack((own, enemy) if me == 0 else (enemy, own))
    return state._replace(
        armies=jnp.asarray(armies),
        ownership=jnp.asarray(ownership),
        ownership_neutral=jnp.asarray(~(own | enemy)),
        generals=jnp.asarray(generals),
        castles=jnp.asarray(castles),
        mountains=jnp.asarray(mountains),
        passable=jnp.asarray(~mountains),
        general_positions=jnp.asarray(positions),
    )


class PisaLayer(nn.Module):
    """Pre-norm transformer layer with Pyramid Sparse Attention (PISA, arXiv 2609.31093).

    Cell tokens (token 0 is the global token) pick their `topk` key blocks of
    `block` tokens by coarse-to-fine search over a mean-pooled key pyramid,
    scoring candidates with LogSumExp, so selection costs O(N log N) rather than
    O(N^2 / block). Selection is shared by all heads (summed LSE) and is not
    differentiated; attention itself runs on the original keys/values. Every
    cell also always attends to the global token, which attends densely to all.
    Differences from the paper: bidirectional (no causal 'forced blocks'), no
    GQA, and plain PyTorch gathers instead of fused Triton kernels."""

    def __init__(self, width, heads, block=16, topk=4):
        super().__init__()
        self.h, self.block, self.topk = heads, block, topk
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):  # (B, 1 + N, width)
        b, n, width = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, self.h, width // self.h).permute(2, 0, 3, 1, 4)
        out_g = F.scaled_dot_product_attention(q[:, :, :1], k, v)
        out_c = self.sparse(q[:, :, 1:], k[:, :, 1:], v[:, :, 1:], k[:, :, :1], v[:, :, :1])
        x = x + self.proj(torch.cat([out_g, out_c], 2).transpose(1, 2).reshape(b, n, width))
        return x + self.mlp(self.n2(x))

    def sparse(self, q, k, v, kg, vg):
        B, H, N, d = q.shape
        C = self.block
        P = 1 << (-(-N // C) - 1).bit_length()  # leaf blocks padded to a power of two
        pad = P * C - N
        valid = F.pad(torch.ones(N, dtype=torch.bool), (0, pad))
        k, v = F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
        with torch.no_grad():
            sel = self.select(q * d ** -0.5, k, valid, P)  # (B, N, S) leaf-block ids
        S = sel.shape[-1]
        idx = sel.reshape(B, 1, N * S, 1, 1).expand(B, H, -1, C, d)
        ks = k.reshape(B, H, P, C, d).gather(2, idx).reshape(B, H, N, S * C, d)
        vs = v.reshape(B, H, P, C, d).gather(2, idx).reshape(B, H, N, S * C, d)
        ok = valid.reshape(P, C)[sel].reshape(B, N, S * C)
        ks = torch.cat([ks, kg.reshape(B, H, 1, 1, d).expand(B, H, N, 1, d)], 3)
        vs = torch.cat([vs, vg.reshape(B, H, 1, 1, d).expand(B, H, N, 1, d)], 3)
        ok = torch.cat([ok, torch.ones(B, N, 1, dtype=torch.bool)], -1)
        return F.scaled_dot_product_attention(q.unsqueeze(3), ks, vs, attn_mask=ok[:, None, :, None]).squeeze(3)

    def select(self, q, k, valid, P):
        """Pyramid top-K: returns (B, N, <=topk) leaf-block indices per query."""
        B, H, N, d = q.shape
        C, K = self.block, self.topk
        sums = [None, (k * valid[:, None]).reshape(B, H, P, C, d).sum(3)]
        cnts = [None, valid.reshape(P, C).sum(1)]
        while sums[-1].shape[2] > 1:  # fine-to-coarse mean pooling (as sum / valid count)
            sums.append(sums[-1].reshape(B, H, -1, 2, d).sum(3))
            cnts.append(cnts[-1].reshape(-1, 2).sum(1))
        means = [None] + [s / c.clamp(min=1)[:, None] for s, c in zip(sums[1:], cnts[1:])]

        def top(A, u):  # keep the K best-scoring candidates
            return A.gather(-1, u.topk(K, -1).indices)

        A = torch.zeros(B, N, 1, dtype=torch.long)
        for lvl in range(len(sums) - 1, 1, -1):  # coarsest level down to 2
            if A.shape[-1] > K:
                ch = torch.stack([2 * A, 2 * A + 1], -1)  # children at level lvl-1: (B, N, a, 2)
                g = means[lvl - 1].gather(2, ch.reshape(B, 1, -1, 1).expand(B, H, -1, d)).reshape(B, H, N, -1, 2, d)
                lg = torch.einsum("bhnd,bhnasd->bhnas", q, g).masked_fill(~(cnts[lvl - 1][ch] > 0)[:, None], -torch.inf)
                A = top(A, torch.logsumexp(lg, -1).sum(1))
            A = torch.stack([2 * A, 2 * A + 1], -1).flatten(-2)
        if A.shape[-1] > K:  # leaf blocks: exact LSE over their original keys
            g = k.reshape(B, H, P, C, d).gather(2, A.reshape(B, 1, -1, 1, 1).expand(B, H, -1, C, d)).reshape(B, H, N, -1, C, d)
            lg = torch.einsum("bhnd,bhnacd->bhnac", q, g).masked_fill(~valid.reshape(P, C)[A][:, None], -torch.inf)
            A = top(A, torch.logsumexp(lg, -1).sum(1))
        return A


class Net(nn.Module):
    """Pixel-level spacetime transformer: one token per cell per frame, no pooling.

    Input (B, FRAMES, C, H, W) is the player's own recent fog observations, so
    attention can recall earlier sightings. A learned global token feeds the
    value and pass heads; the policy comes from each current-frame cell token."""

    def __init__(self, h, w, width=64, layers=4, heads=4, frames=FRAMES, attn="dense", block=16, topk=4):
        super().__init__()
        self.cells = h * w
        self.inp = nn.Linear(CHANNELS, width)
        self.pos = nn.Parameter(torch.randn(1, 1, h * w, width) * 0.02)
        self.tpos = nn.Parameter(torch.randn(1, frames, 1, width) * 0.02)
        self.glob = nn.Parameter(torch.zeros(1, 1, width))
        if attn == "pisa":
            self.body = nn.Sequential(*[PisaLayer(width, heads, block, topk) for _ in range(layers)])
        else:
            layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.pol = nn.Linear(width, 8)  # per cell: 4 dirs x 2 splits
        self.pass_logit = nn.Linear(width, 1)
        self.val = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x):
        b, t = x.shape[:2]
        tok = self.inp(x.flatten(3).transpose(2, 3)) + self.pos + self.tpos  # (B, T, H*W, width)
        tok = tok.reshape(b, t * self.cells, -1)
        tok = self.norm(self.body(torch.cat([self.glob.expand(b, -1, -1), tok], 1)))
        g, cur = tok[:, 0], tok[:, -self.cells:]  # last frame's cells
        logits = self.pol(cur).reshape(b, -1)  # (B, H*W*8), cell-major
        return torch.cat([logits, self.pass_logit(g)], 1), self.val(g).squeeze(1)


@torch.no_grad()
def evaluate(net, stacks, masks):
    """Batch of observation histories -> masked priors (B, A) and values (B,)."""
    logits, v = net(torch.from_numpy(np.stack(stacks)))
    logits = logits.masked_fill(~torch.from_numpy(np.stack(masks)), -1e9)
    return F.softmax(logits, 1).numpy(), v.numpy()


class Node:
    """Search node for player `me`. Stores priors for both players (one batched eval)."""

    def __init__(self, state, prior, opp_prior, value, stack=None, opp_stack=None):
        self.state, self.prior, self.opp_prior, self.value = state, prior, opp_prior, value
        self.stack, self.opp_stack = stack, opp_stack  # observation histories incl. this node's frame
        self.n = np.zeros_like(prior)
        self.w = np.zeros_like(prior)
        self.children = {}  # (own action, sampled opponent action) -> child/terminal value


def outcome(state, me, max_steps):
    """Real win/loss, or bounded score proxy when a game is truncated."""
    win = int(state.winner)
    if win >= 0:
        return 1.0 if win == me else -1.0
    if int(state.time) < max_steps:
        return None
    land = np.asarray(state.ownership).sum((1, 2))
    army = np.asarray(state.armies * state.ownership[me]).sum()
    other_army = np.asarray(state.armies * state.ownership[1 - me]).sum()
    return float(0.5 * np.tanh((np.log1p(army + 5 * land[me]) -
                                np.log1p(other_army + 5 * land[1 - me])) / 3))


class Searcher:
    def __init__(self, net, h, w, max_steps, rng):
        self.net, self.h, self.w, self.max_steps, self.rng = net, h, w, max_steps, rng

    def make_node(self, state, me, prev=None, opp_prev=None):
        obs = [get_observation(state, me), get_observation(state, 1 - me)]
        masks = [legal_mask(o, self.h, self.w) for o in obs]
        # ponytail: the opponent's true history is unknown to us; it starts as a repeat of its current frame.
        stacks = [stack_with(prev, features(obs[0])), stack_with(opp_prev, features(obs[1]))]
        p, v = evaluate(self.net, stacks, masks)
        return Node(state, p[0], p[1], float(v[0]), *stacks)

    def search(self, state, me, sims, prev=None, noise=True):
        counts = np.zeros(num_actions(self.h, self.w), dtype=np.float32)
        roots = min(4, sims)
        for i in range(roots):
            root = self.make_node(determinize(state, me, self.rng), me, prev)
            if noise:
                legal = np.flatnonzero(root.prior > 0)
                root.prior = root.prior.copy()
                root.prior[legal] = 0.75 * root.prior[legal] + 0.25 * self.rng.dirichlet(
                    np.full(len(legal), 10 / len(legal)))
            for _ in range(i, sims, roots):
                self.simulate(root, me)
            counts += root.n
        return counts

    def simulate(self, root, me):
        path, node = [], root
        while True:
            total = node.n.sum()
            q = np.where(node.n > 0, node.w / np.maximum(node.n, 1), 0.0)
            u = q + C_PUCT * node.prior * math.sqrt(total + 1) / (1 + node.n)
            a = int(np.argmax(np.where(node.prior > 0, u, -1e9)))
            opp_a = int(self.rng.choice(len(node.opp_prior), p=node.opp_prior / node.opp_prior.sum()))
            edge = (a, opp_a)
            path.append((node, a))
            if edge not in node.children:
                v = self.expand(node, edge, me)
                break
            child, term = node.children[edge]
            if child is None:
                v = term
                break
            node = child
        for nd, act in reversed(path):  # outcome stays from `me` view; no sign flip (simultaneous moves)
            nd.n[act] += 1
            nd.w[act] += v

    def expand(self, node, edge, me):
        a, opp_a = edge
        acts = [None, None]
        acts[me], acts[1 - me] = decode(a, self.h, self.w), decode(opp_a, self.h, self.w)
        state, _ = game_step(node.state, jnp.asarray(acts, dtype=jnp.int32), general_trade=True)
        term = outcome(state, me, self.max_steps)
        if term is not None:
            node.children[edge] = (None, term)
            return term
        child = self.make_node(state, me, node.stack, node.opp_stack)
        node.children[edge] = (child, None)
        return child.value


def self_play(net, h, w, sims, max_steps, key, rng, env):
    """One game; returns list of (features, pi, mask, z) for both players, and result string."""
    state = env.init_state(key)
    searcher = Searcher(net, h, w, max_steps, rng)
    rec, hist = [[], []], [None, None]
    while True:
        res = outcome(state, 0, max_steps)
        if res is not None:
            break
        acts = [None, None]
        for me in (0, 1):
            obs = get_observation(state, me)
            counts = searcher.search(state, me, sims, hist[me])
            hist[me] = stack_with(hist[me], features(obs))
            pi = counts / counts.sum()
            temp = 1.0 if int(state.time) < 10 else 0.25
            p = pi ** (1 / temp)
            a = int(rng.choice(len(p), p=p / p.sum()))
            acts[me] = decode(a, h, w)
            rec[me].append((hist[me], pi.astype(np.float32), legal_mask(obs, h, w)))
        state, _ = game_step(state, jnp.asarray(acts, dtype=jnp.int32), general_trade=True)
    samples = []
    for me in (0, 1):
        z = outcome(state, me, max_steps)
        samples += [(f, pi, m, z) for f, pi, m in rec[me]]
    result = f"player {int(state.winner)} wins" if int(state.winner) >= 0 else f"score-adjudicated {res:+.3f}"
    return samples, result


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
    ap.add_argument("--attn", choices=("dense", "pisa"), default="dense",
                    help="pisa = pyramid block-sparse attention; only pays off on large boards")
    ap.add_argument("--size", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="alphazero.pt")
    ap.add_argument("--resume", action="store_true", help="resume from --output (trusted local checkpoint)")
    args = ap.parse_args(argv)
    if args.games < 1 or args.sims < 1 or args.size < 4 or args.max_steps < 1:
        ap.error("--games, --sims, --max-steps must be positive and --size at least 4")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    h = w = args.size
    env = GeneralsEnv(grid_dims=(h, w), truncation=args.max_steps, general_trade=True)
    net = Net(h, w, attn=args.attn)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    key = jrandom.PRNGKey(args.seed)
    buffer = deque(maxlen=4096)
    completed = 0
    if args.resume:
        checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
        if any(checkpoint["args"][name] != getattr(args, name) for name in ("size", "sims", "max_steps", "seed", "attn")):
            ap.error("resume requires matching --size, --sims, --max-steps, --seed and --attn")
        net.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        key = jnp.asarray(checkpoint["jax_key"].numpy(), dtype=jnp.uint32)
        buffer.extend((f.numpy(), pi.numpy(), m.numpy(), z) for f, pi, m, z in checkpoint["replay"])
        completed = checkpoint["completed"]
    for g in range(completed, args.games):
        key, k = jrandom.split(key)
        samples, result = self_play(net, h, w, args.sims, args.max_steps, k, rng, env)
        buffer.extend(samples)
        indices = rng.choice(len(buffer), size=min(512, len(buffer)), replace=False)
        lp, lv = train_step(net, opt, [buffer[int(i)] for i in indices])
        checkpoint = {
            "model": net.state_dict(), "optimizer": opt.state_dict(), "args": vars(args),
            "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "jax_key": torch.from_numpy(np.asarray(key).copy()), "completed": g + 1,
            "replay": [(torch.from_numpy(f), torch.from_numpy(pi), torch.from_numpy(m), z)
                       for f, pi, m, z in buffer],
        }
        torch.save(checkpoint, args.output + ".tmp")
        os.replace(args.output + ".tmp", args.output)
        print(f"game {g + 1}/{args.games}: {result}, {len(samples)} samples, "
              f"policy loss {lp:.3f}, value loss {lv:.3f}; saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
