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

Memory: each frame carries ever-seen / last-seen enemy / last-seen terrain / sticky general-sighting
planes. An auxiliary belief head predicts the enemy general's cell (trained only while it is unseen;
the true cell is a label, never an input); --belief on makes determinize() sample from it.
The opponent model is sampled rather than adversarial search.
This is not a ranked-ready agent. Evaluate separately before live play.
"""
import argparse
import math
import os
import sys
from collections import deque

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastkernels  # noqa: E402
from generals import GeneralsEnv, get_observation  # noqa: E402
from generals.core.action import compute_valid_move_mask_obs
from generals.core.game import step as game_step

C_PUCT = 1.5
FAST = False  # --fast: fused AVX-512 PISA attention in no-grad inference (examples/fastkernels), PyTorch otherwise
CHANNELS = 20  # 15 current-view planes + 5 memory planes (see features)
BELIEF_WEIGHT = 0.1  # auxiliary enemy-general cross-entropy weight in train_step
FRAMES = 4  # default number of past own-observation frames the net attends over (fog memory)


def stack_with(prev, frame, n=FRAMES):
    """Append a frame to an (n, C, H, W) history; first call repeats the frame."""
    if prev is None:
        return np.repeat(frame[None], n, 0)
    return np.concatenate([prev[1:], frame[None]])


def num_actions(h, w):
    return h * w * 8 + 1  # (cell, direction, split) ..., last index = pass


def decode(idx, h, w):
    """Policy index -> simulator action [pass, row, col, direction, split]."""
    if idx == h * w * 8:
        return [1, 0, 0, 0, 0]
    cell, rest = divmod(idx, 8)
    return [0, cell // w, cell % w, rest // 2, rest % 2]


def features(obs, prev=None):
    """Fog observation -> (CHANNELS, H, W) float array. Own view only.

    `prev` is this player's (n, C, H, W) history; its newest frame carries the memory planes
    (15 ever-seen, 16 last-seen enemy cells, 17 last-seen mountains, 18 last-seen castles,
    19 sticky enemy-general sighting), which are updated from the current view."""
    a = np.asarray(obs.armies, dtype=np.float32)
    planes = [np.log1p(a) / 5.0 * np.asarray(obs.owned_cells), np.log1p(a) / 5.0 * np.asarray(obs.opponent_cells),
              np.log1p(a) / 5.0 * np.asarray(obs.neutral_cells)]
    planes += [np.asarray(m, dtype=np.float32) for m in (
        obs.generals, obs.castles, obs.mountains, obs.owned_cells, obs.opponent_cells, obs.fog_cells,
        obs.structures_in_fog)]
    counts = (obs.owned_land_count, obs.owned_army_count, obs.opponent_land_count, obs.opponent_army_count)
    planes += [np.full(a.shape, np.log1p(float(v)) / 5.0, dtype=np.float32) for v in counts]
    planes.append(np.full(a.shape, float(obs.timestep) / 500.0, dtype=np.float32))
    old = np.zeros((5,) + a.shape, np.float32) if prev is None else prev[-1, 15:]
    vis = ~np.asarray(obs.fog_cells | obs.structures_in_fog)
    new = np.stack([vis, np.asarray(obs.opponent_cells), np.asarray(obs.mountains), np.asarray(obs.castles),
                    np.asarray(obs.generals & obs.opponent_cells)])
    planes += [np.maximum(old[0], vis), np.where(vis, new[1], old[1]), np.where(vis, new[2], old[2]),
               np.where(vis, new[3], old[3]), np.maximum(old[4], new[4])]
    return np.stack(planes).astype(np.float32)


def belief_support(x):
    """(B, C, H, W) newest frames -> (B, H*W) bool: cells the enemy general can still occupy
    (not own, never seen -- the general never moves -- and not a known mountain/castle)."""
    keep = (x[:, 6] < .5) & (x[:, 15] < .5) & (x[:, 17] < .5) & (x[:, 18] < .5) & (x[:, 5] < .5)
    return keep.flatten(1)


def legal_mask(obs, h, w):
    m = np.asarray(compute_valid_move_mask_obs(obs))  # (H, W, 4)
    moves = np.repeat(m.reshape(h * w, 4, 1), 2, axis=2)
    moves[:, :, 1] &= np.asarray(obs.armies).reshape(-1, 1) > 2  # half of two equals full
    return np.concatenate([moves.reshape(-1), [True]])


def make_env(size, max_steps):
    """Env with ~10 castles where the board allows (area // 6 on tiny boards) and a size-scaled
    minimum general distance. The generator carves castles out of mountains and caps all mountains
    at area // 4, so we ask for castles + the default 18% real terrain and let that cap bind.
    Generals stay connected over open ground (checked in tests)."""
    castles = min(10, size * size // 6)
    lo = 0.18 + castles / size**2
    return GeneralsEnv(grid_dims=(size, size), truncation=max_steps, general_trade=True,
                       min_generals_distance=max(3, size // 2), num_castles_range=(castles, castles),
                       mountain_density_range=(lo, lo + 0.08))


# Fraction of fogged structures that are castles; measured on make_env boards (see README); checked by tests/test_az_selfplay.py.
CASTLE_FRACTION = {6: 0.65, 8: 0.63}


def castle_fraction(size):
    return CASTLE_FRACTION[min(CASTLE_FRACTION, key=lambda k: abs(k - size))]


def determinize(state, me, rng, belief=None, seen=None):
    """Sample a board from one player's observation, never from hidden tiles.

    belief: optional (H*W,) probabilities over the enemy general's cell (from the net, own view only);
    seen: optional (H, W) ever-seen mask from this player's memory planes. With belief the general is
    drawn from it, hidden enemy land is biased toward it, and ever-seen cells leave the land support."""
    obs = get_observation(state, me)
    fog = np.asarray(obs.fog_cells | obs.structures_in_fog)
    armies = np.asarray(obs.armies).copy()
    own = np.asarray(obs.owned_cells).copy()
    enemy = np.asarray(obs.opponent_cells).copy()
    # Fogged structures are almost never enemy-owned castles, so enemy land goes on plain hidden tiles first.
    hidden = np.flatnonzero(fog & ~np.asarray(obs.structures_in_fog))
    unseen_land = max(0, int(obs.opponent_land_count) - int(enemy.sum()))
    gen_cell = None
    if belief is not None and not (np.asarray(obs.generals) & enemy).any() and len(hidden) and unseen_land:
        fresh = hidden[~np.asarray(seen).flat[hidden]] if seen is not None else hidden
        support = fresh if len(fresh) else hidden
        pg = np.asarray(belief, np.float64)[support]
        gen_cell = int(rng.choice(support, p=pg / pg.sum() if pg.sum() > 0 else None))
        rest = support[support != gen_cell]
        if len(rest) < unseen_land - 1:  # not enough never-seen cells: fall back to all hidden
            rest = hidden[hidden != gen_cell]
        near = np.abs(rest // enemy.shape[1] - gen_cell // enemy.shape[1]) + np.abs(rest % enemy.shape[1] - gen_cell % enemy.shape[1])
        wt = 1.0 / (1.0 + near)
        k = min(unseen_land - 1, len(rest))
        chosen = np.concatenate([[gen_cell], rng.choice(rest, size=k, replace=False, p=wt / wt.sum())]).astype(int)
    else:
        chosen = rng.choice(hidden, size=min(unseen_land, len(hidden)), replace=False)
    enemy.flat[chosen] = True
    mountains = np.asarray(obs.mountains).copy()
    castles = np.asarray(obs.castles).copy()
    structures = np.asarray(obs.structures_in_fog).copy()
    hidden_castles = (rng.random(structures.shape) < castle_fraction(structures.shape[0])) & structures
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
    elif gen_cell is not None:
        positions[1 - me] = np.unravel_index(gen_cell, own.shape)
        generals[tuple(positions[1 - me])] = True
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

    def step(self, x, cache, keep):
        """Streaming forward for the newest frame (x: (B, 1 + N, width)). `cache` = (k, v) of the
        cell tokens of earlier frames as computed at their own time, or None; only the newest
        frame's tokens are queries. Returns (out, cache trimmed to the last `keep` cell tokens)."""
        b, n, width = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, self.h, width // self.h).permute(2, 0, 3, 1, 4)
        kc, vc = (k[:, :, 1:], v[:, :, 1:]) if cache is None else (torch.cat([cache[0], k[:, :, 1:]], 2),
                                                                  torch.cat([cache[1], v[:, :, 1:]], 2))
        out_g = F.scaled_dot_product_attention(q[:, :, :1], torch.cat([k[:, :, :1], kc], 2), torch.cat([v[:, :, :1], vc], 2))
        out_c = self.sparse(q[:, :, 1:], kc, vc, k[:, :, :1], v[:, :, :1])
        x = x + self.proj(torch.cat([out_g, out_c], 2).transpose(1, 2).reshape(b, n, width))
        return x + self.mlp(self.n2(x)), (kc[:, :, kc.shape[2] - keep:], vc[:, :, vc.shape[2] - keep:])

    def sparse(self, q, k, v, kg, vg):
        B, H, N, d = q.shape
        C = self.block
        P = 1 << (-(-k.shape[2] // C) - 1).bit_length()  # leaf blocks padded to a power of two
        pad = P * C - k.shape[2]
        valid = F.pad(torch.ones(k.shape[2], dtype=torch.bool), (0, pad))
        k, v = F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
        with torch.no_grad():
            sel = self.select(q * d ** -0.5, k, valid, P)  # (B, N, S) leaf-block ids
        S = sel.shape[-1]
        if FAST and not torch.is_grad_enabled():
            nk = k.shape[2] - pad
            if k.stride() == v.stride() and kg.stride() == vg.stride():
                out = fastkernels.sparse_attn(q, k[:, :, :nk], v[:, :, :nk], kg, vg, sel, C)
                if out is not None:
                    return out
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

        def fast_lse(qq, X, ok, A, G):  # fused gather+LSE (no-grad inference with --fast), else None
            return fastkernels.group_lse(qq, X, ok, A, G) if FAST and not torch.is_grad_enabled() else None

        def top(A, u):  # keep the K best-scoring candidates
            return A.gather(-1, u.topk(K, -1).indices)

        A = torch.zeros(B, N, 1, dtype=torch.long)
        for lvl in range(len(sums) - 1, 1, -1):  # coarsest level down to 2
            if A.shape[-1] > K:
                u = fast_lse(q, means[lvl - 1], cnts[lvl - 1] > 0, A, 2)  # children at level lvl-1 of each a
                if u is None:
                    ch = torch.stack([2 * A, 2 * A + 1], -1)  # (B, N, a, 2)
                    g = means[lvl - 1].gather(2, ch.reshape(B, 1, -1, 1).expand(B, H, -1, d)).reshape(B, H, N, -1, 2, d)
                    lg = torch.einsum("bhnd,bhnasd->bhnas", q, g).masked_fill(~(cnts[lvl - 1][ch] > 0)[:, None], -torch.inf)
                    u = torch.logsumexp(lg, -1).sum(1)
                A = top(A, u)
            A = torch.stack([2 * A, 2 * A + 1], -1).flatten(-2)
        if A.shape[-1] > K:  # leaf blocks: exact LSE over their original keys
            u = fast_lse(q, k, valid, A, C)
            if u is None:
                g = k.reshape(B, H, P, C, d).gather(2, A.reshape(B, 1, -1, 1, 1).expand(B, H, -1, C, d)).reshape(B, H, N, -1, C, d)
                lg = torch.einsum("bhnd,bhnacd->bhnac", q, g).masked_fill(~valid.reshape(P, C)[A][:, None], -torch.inf)
                u = torch.logsumexp(lg, -1).sum(1)
            A = top(A, u)
        return A


def gdn2_step(S, q, k, v, alpha, b, w):
    """One time step of the GDN-2 recurrence; q, k, alpha, b: (B, H, dk), v, w: (B, H, dv), S: (B, H, dk, dv)."""
    S = alpha[..., None] * S
    r = torch.einsum("bhkv,bhk->bhv", S, b * k)
    S = S + torch.einsum("bhk,bhv->bhkv", k, w * v - r)
    return S, torch.einsum("bhkv,bhk->bhv", S, q)


def gdn2_scan(q, k, v, alpha, b, w):
    """Gated Delta Rule-2 (arXiv 2605.22791 eq. 9), sequential over time.

    q, k: (B, T, H, dk) (k unit-norm); v: (B, T, H, dv); alpha, b: (B, T, H, dk) in (0, 1];
    w: (B, T, H, dv). State S: (B, H, dk, dv):
        S_bar = diag(alpha) S;  r = S_bar^T (b * k);  S = S_bar + k (w * v - r)^T;  o = S^T q.
    Equal b/w scalars recover KDA; plus scalar alpha recovers Gated DeltaNet."""
    B, T, H, dk = q.shape
    S = q.new_zeros(B, H, dk, v.shape[-1])
    out = []
    for t in range(T):
        S, o = gdn2_step(S, q[:, t], k[:, t], v[:, t], alpha[:, t], b[:, t], w[:, t])
        out.append(o)
    return torch.stack(out, 1)


class GdnLayer(nn.Module):
    """Gated DeltaNet-2 over the frame axis, independently for every cell.

    Causal in time, so the last frame's cell token summarises the whole history
    in a fixed-size state. The global token is left to the attention layers.
    Simplifications vs the paper: no short convolution or output gate, and a
    plain PyTorch time loop instead of the chunkwise WY Triton kernels."""

    def __init__(self, width, heads, frames, cells):
        super().__init__()
        self.h, self.frames, self.cells = heads, frames, cells
        d = width // heads
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.alpha, self.erase, self.write = nn.Linear(width, width), nn.Linear(width, width), nn.Linear(width, width)
        self.log_a = nn.Parameter(torch.zeros(heads, d))  # decay rate scale: g = -exp(a) * softplus(W x + delta)
        self.delta = nn.Parameter(torch.zeros(heads, d))
        self.onorm, self.proj = nn.LayerNorm(width), nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):  # (B, 1 + frames*cells, width); token 0 is the global token
        B, _, width = x.shape
        T, N, H = self.frames, self.cells, self.h
        c = self.n1(x[:, 1:]).reshape(B, T, N, width).transpose(1, 2).reshape(B * N, T, width)
        q, k, v = (t.reshape(B * N, T, H, -1) for t in self.qkv(c).chunk(3, -1))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        g = -self.log_a.exp() * F.softplus(self.alpha(c).reshape(B * N, T, H, -1) + self.delta)
        b = torch.sigmoid(self.erase(c)).reshape(B * N, T, H, -1)
        w = torch.sigmoid(self.write(c)).reshape(B * N, T, H, -1)
        o = gdn2_scan(q, k, v, g.exp(), b, w).reshape(B * N, T, width)
        o = self.proj(self.onorm(o)).reshape(B, N, T, width).transpose(1, 2).reshape(B, T * N, width)
        x = torch.cat([x[:, :1], x[:, 1:] + o], 1)
        return x + self.mlp(self.n2(x))

    def step(self, x, state):
        """Advance one frame (x: (B, 1 + N, width)) from the per-cell state `(S,)` (None = zeros)."""
        B, _, width = x.shape
        N, H = self.cells, self.h
        c = self.n1(x[:, 1:]).reshape(B * N, width)
        q, k, v = (t.reshape(B * N, H, -1) for t in self.qkv(c).chunk(3, -1))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        g = -self.log_a.exp() * F.softplus(self.alpha(c).reshape(B * N, H, -1) + self.delta)
        b = torch.sigmoid(self.erase(c)).reshape(B * N, H, -1)
        w = torch.sigmoid(self.write(c)).reshape(B * N, H, -1)
        S = q.new_zeros(B * N, H, k.shape[-1], v.shape[-1]) if state is None else state[0]
        S, o = gdn2_step(S, q, k, v, g.exp(), b, w)
        x = torch.cat([x[:, :1], x[:, 1:] + self.proj(self.onorm(o.reshape(B * N, width))).reshape(B, N, width)], 1)
        return x + self.mlp(self.n2(x)), (S,)


def default_order(layers):
    return "".join("GP"[i % 2] for i in range(layers))


def check_order(order, layers):
    """Hybrid layer order: one char per layer, G = GDN-2, P = PISA, at least one of each."""
    if len(order) != layers or set(order) - set("GP") or len(set(order)) < 2:
        raise ValueError(f"hybrid order {order!r} must be {layers} chars of G/P with at least one of each")
    return order


def mix_target(z, q, lam):
    """Value target: lam * outcome (or truncation proxy) + (1 - lam) * root search value."""
    return lam * z + (1 - lam) * q


class Net(nn.Module):
    """Pixel-level spacetime transformer: one token per cell per frame, no pooling.

    Input (B, FRAMES, C, H, W) is the player's own recent fog observations, so
    attention can recall earlier sightings. A learned global token feeds the
    value and pass heads; the policy comes from each current-frame cell token."""

    def __init__(self, h, w, width=64, layers=4, heads=4, frames=FRAMES, attn="dense", block=16, topk=4, stream=False, order=None):
        super().__init__()
        self.cells, self.frames, self.stream = h * w, frames, stream
        self.block, self.topk = block, topk
        if stream and attn != "hybrid":
            raise ValueError("stream (incremental) mode needs attn='hybrid'")
        self.inp = nn.Linear(CHANNELS, width)
        self.pos = nn.Parameter(torch.randn(1, 1, h * w, width) * 0.02)
        self.tpos = nn.Parameter(torch.randn(1, frames, 1, width) * 0.02)
        self.glob = nn.Parameter(torch.zeros(1, 1, width))
        if attn == "pisa":
            self.body = nn.Sequential(*[PisaLayer(width, heads, block, topk) for _ in range(layers)])
        elif attn == "hybrid":  # GDN-2 (per-cell memory over frames) interleaved with PISA (space)
            order = check_order(order or default_order(layers), layers)
            self.body = nn.Sequential(*[GdnLayer(width, heads, frames, h * w) if c == "G"
                                        else PisaLayer(width, heads, block, topk) for c in order])
        else:
            layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.pol = nn.Linear(width, 8)  # per cell: 4 dirs x 2 splits
        self.pass_logit = nn.Linear(width, 1)
        self.bel = nn.Linear(width, 1)  # per cell: enemy-general location logit
        self.val = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

    def forward(self, x, with_belief=False):
        """(policy logits, value), plus masked (B, H*W) belief logits when with_belief.

        stream=True: the window is folded frame by frame from an empty state (== `step` applied to
        every frame), which is the same function the incremental path carries across turns."""
        if FAST and not self.stream and not torch.is_grad_enabled():  # whole forward as one C call
            r = fastkernels.net_forward(self, x, self.block, self.topk)
            if r is not None:
                return r if with_belief else r[:2]
        if self.stream:
            state = None
            for t in range(x.shape[1] - 1):
                state = self._trunk(x[:, t], state)[2]
            return self.step(x[:, -1], state, with_belief)[0]
        b, t = x.shape[:2]
        tok = self.inp(x.flatten(3).transpose(2, 3)) + self.pos + self.tpos  # (B, T, H*W, width)
        tok = tok.reshape(b, t * self.cells, -1)
        tok = self.norm(self.body(torch.cat([self.glob.expand(b, -1, -1), tok], 1)))
        return self._heads(tok[:, 0], tok[:, -self.cells:], x[:, -1], with_belief)

    def _heads(self, g, cur, frame, with_belief):
        logits = self.pol(cur).reshape(len(g), -1)  # (B, H*W*8), cell-major
        out = torch.cat([logits, self.pass_logit(g)], 1), self.val(g).squeeze(1)
        if not with_belief:
            return out
        bel = self.bel(cur).squeeze(2).masked_fill(~belief_support(frame), -1e9)
        return out + (bel,)

    def _trunk(self, frame, state):
        """One frame (B, C, H, W) + per-layer state list (None = empty) -> (global tok, cell toks, new state).
        GDN layers carry S (O(1) in history); PISA layers carry the last frames-1 frames' cell keys/values.
        All frames share one temporal embedding (the window slot of a cached key is not recomputable)."""
        b = frame.shape[0]
        tok = self.inp(frame.flatten(2).transpose(1, 2)) + self.pos[0] + self.tpos[0, -1]
        x = torch.cat([self.glob.expand(b, -1, -1), tok], 1)
        new = []
        for i, layer in enumerate(self.body):
            s = None if state is None else state[i]
            if isinstance(layer, GdnLayer):
                x, s = layer.step(x, s)
            else:
                x, s = layer.step(x, s, (self.frames - 1) * self.cells)
            new.append(s)
        x = self.norm(x)
        return x[:, 0], x[:, 1:], new

    def step(self, frame, state, with_belief=False):
        """Incremental inference for one new frame: (outputs as in forward, new state)."""
        g, cur, state = self._trunk(frame, state)
        return self._heads(g, cur, frame, with_belief), state


def priors(logits, v, masks):
    logits = logits.masked_fill(~torch.from_numpy(np.stack(masks)), -1e9)
    return F.softmax(logits, 1).numpy(), v.numpy()


@torch.no_grad()
def evaluate(net, stacks, masks):
    """Batch of observation histories -> masked priors (B, A) and values (B,)."""
    return priors(*net(torch.from_numpy(np.stack(stacks))), masks)


class Node:
    """Search node for player `me`. Stores priors for both players (one batched eval)."""

    def __init__(self, state, prior, opp_prior, value, stack=None, opp_stack=None, st=None):
        self.state, self.prior, self.opp_prior, self.value = state, prior, opp_prior, value
        self.stack, self.opp_stack = stack, opp_stack  # observation histories incl. this node's frame
        self.st = st  # incremental mode: batch-2 (me, opponent) recurrent net state after this node's frame
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
    def __init__(self, net, h, w, max_steps, rng, belief=False, incremental=False):
        self.net, self.h, self.w, self.max_steps, self.rng, self.belief = net, h, w, max_steps, rng, belief
        self.incremental = incremental  # carry GDN/PISA state in nodes: a child costs one net step
        if incremental and not net.stream:
            raise ValueError("incremental search needs Net(stream=True)")

    def make_node(self, state, me, prev=None, opp_prev=None, st=None):
        """`st`: root -> my batch-1 net state (opponent's starts empty); child -> parent's batch-2 state."""
        obs = [get_observation(state, me), get_observation(state, 1 - me)]
        masks = [legal_mask(o, self.h, self.w) for o in obs]
        # ponytail: the opponent's true history is unknown to us; it starts as a repeat of its current frame.
        n = self.net.frames
        stacks = [stack_with(prev, features(obs[0], prev), n), stack_with(opp_prev, features(obs[1], opp_prev), n)]
        if not self.incremental:
            p, v = evaluate(self.net, stacks, masks)
            return Node(state, p[0], p[1], float(v[0]), *stacks)
        fr = torch.from_numpy(np.stack([s[-1] for s in stacks]))
        with torch.no_grad():
            if opp_prev is None:  # root: two independent batch-1 steps, then batch
                (l0, v0), s0 = self.net.step(fr[:1], st)
                (l1, v1), s1 = self.net.step(fr[1:], None)
                (logits, v), st = (torch.cat([l0, l1]), torch.cat([v0, v1])), [tuple(torch.cat(t) for t in zip(a, b)) for a, b in zip(s0, s1)]
            else:
                (logits, v), st = self.net.step(fr, st)
        p, v = priors(logits, v, masks)
        return Node(state, p[0], p[1], float(v[0]), *stacks, st)

    def search(self, state, me, sims, prev=None, noise=True, prev_st=None):
        counts = np.zeros(num_actions(self.h, self.w), dtype=np.float32)
        wsum = nsum = 0.0
        roots = min(4, sims)
        bel = seen = None
        if self.belief:  # own fog view + memory only
            stack = stack_with(prev, features(get_observation(state, me), prev), self.net.frames)
            with torch.no_grad():
                if self.incremental:
                    logits = self.net.step(torch.from_numpy(stack[-1:]), prev_st, True)[0][2][0]
                else:
                    logits = self.net(torch.from_numpy(stack[None]), True)[2][0]
            bel, seen = F.softmax(logits, 0).numpy(), stack[-1, 15] > 0.5
        for i in range(roots):
            root = self.make_node(determinize(state, me, self.rng, bel, seen), me, prev, None, prev_st)
            if noise:
                legal = np.flatnonzero(root.prior > 0)
                root.prior = root.prior.copy()
                root.prior[legal] = 0.75 * root.prior[legal] + 0.25 * self.rng.dirichlet(
                    np.full(len(legal), 10 / len(legal)))
            for _ in range(i, sims, roots):
                self.simulate(root, me)
            counts += root.n
            wsum += root.w.sum()
            nsum += root.n.sum()
        self.root_q = float(wsum / max(nsum, 1))  # mean backed-up value at the roots, `me` view
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
        child = self.make_node(state, me, node.stack, node.opp_stack, node.st)
        node.children[edge] = (child, None)
        return child.value


def self_play(net, h, w, sims, max_steps, key, rng, env, belief=False, incremental=False, lam=1.0):
    """One game; returns list of (features, pi, mask, z, enemy general cell [label only]) for both players, and result string."""
    state = env.init_state(key)
    searcher = Searcher(net, h, w, max_steps, rng, belief, incremental)
    rec, hist, hist_st = [[], []], [None, None], [None, None]
    while True:
        res = outcome(state, 0, max_steps)
        if res is not None:
            break
        acts = [None, None]
        for me in (0, 1):
            obs = get_observation(state, me)
            counts = searcher.search(state, me, sims, hist[me], prev_st=hist_st[me])
            hist[me] = stack_with(hist[me], features(obs, hist[me]), net.frames)
            if incremental:  # real turn: advance my recurrent state by the one new frame
                with torch.no_grad():
                    hist_st[me] = net.step(torch.from_numpy(hist[me][-1:]), hist_st[me])[1]
            pi = counts / counts.sum()
            temp = 1.0 if int(state.time) < 10 else 0.25
            p = pi ** (1 / temp)
            a = int(rng.choice(len(p), p=p / p.sum()))
            acts[me] = decode(a, h, w)
            rec[me].append((hist[me], pi.astype(np.float32), legal_mask(obs, h, w), int(state.general_positions[1 - me][0]) * w + int(state.general_positions[1 - me][1]), searcher.root_q))
        state, _ = game_step(state, jnp.asarray(acts, dtype=jnp.int32), general_trade=True)
    samples = []
    for me in (0, 1):
        z = outcome(state, me, max_steps)
        samples += [(f, pi, m, mix_target(z, q, lam), g) for f, pi, m, g, q in rec[me]]
    result = f"player {int(state.winner)} wins" if int(state.winner) >= 0 else f"score-adjudicated {res:+.3f}"
    return samples, result


def train_step(net, opt, samples, epochs=2, batch=64):
    x = torch.from_numpy(np.stack([s[0] for s in samples]))
    pi = torch.from_numpy(np.stack([s[1] for s in samples]))
    m = torch.from_numpy(np.stack([s[2] for s in samples]))
    z = torch.tensor([s[3] for s in samples], dtype=torch.float32)
    gen = torch.tensor([s[4] for s in samples])
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), batch):
            j = perm[i:i + batch]
            logits, v, bel = net(x[j], True)
            logp = F.log_softmax(logits.masked_fill(~m[j], -1e9), 1)
            loss_p = -(pi[j] * logp).sum(1).mean()
            loss_v = F.mse_loss(v, z[j])
            unseen = x[j][:, -1, 19].flatten(1).sum(1) < .5  # label used only while the general is unseen
            ce = F.cross_entropy(bel, gen[j], reduction="none")
            loss_b = (ce * unseen).sum() / unseen.sum().clamp(min=1)
            opt.zero_grad()
            (loss_p + loss_v + BELIEF_WEIGHT * loss_b).backward()
            opt.step()
    return loss_p.item(), loss_v.item()


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("channels") != CHANNELS:
        raise ValueError(f"{path}: checkpoint has {ck.get('channels', 'no channels record (pre-memory planes)')} "
                         f"input channels, this code uses {CHANNELS}; retrain or use matching code")
    return ck


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--attn", choices=("dense", "pisa", "hybrid"), default="dense",
                    help="pisa = pyramid block-sparse attention; hybrid = GDN-2 over time + PISA over space")
    ap.add_argument("--frames", type=int, default=FRAMES, help="observation history length; long histories want --attn pisa")
    ap.add_argument("--belief", choices=("off", "on"), default="off", help="sample enemy general/land from the net's belief head")
    ap.add_argument("--hybrid-order", default=None, metavar="GPGP",
                    help="hybrid only: layer types, one char per layer (G = GDN-2, P = PISA); default alternates G,P")
    ap.add_argument("--value-target", choices=("outcome", "mix"), default="outcome",
                    help="mix = lambda * outcome-or-proxy + (1 - lambda) * root search value")
    ap.add_argument("--value-lambda", type=float, default=0.5, help="outcome weight for --value-target mix")
    ap.add_argument("--incremental", action="store_true",
                    help="hybrid only: streaming net (GDN state carried across turns, search nodes store it); trains on window rebuilds")
    ap.add_argument("--fast", action="store_true", help="fused AVX-512 PISA attention at inference (needs gcc + AVX-512, else PyTorch)")
    ap.add_argument("--size", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="alphazero.pt")
    ap.add_argument("--resume", action="store_true", help="resume from --output (trusted local checkpoint)")
    args = ap.parse_args(argv)
    if args.games < 1 or args.sims < 1 or args.size < 4 or args.max_steps < 1:
        ap.error("--games, --sims, --max-steps must be positive and --size at least 4")
    if args.incremental and args.attn != "hybrid":
        ap.error("--incremental needs --attn hybrid")

    if args.hybrid_order is not None and args.attn != "hybrid":
        ap.error("--hybrid-order needs --attn hybrid")
    if args.attn == "hybrid":
        try:
            args.hybrid_order = check_order(args.hybrid_order or default_order(4), 4)
        except ValueError as e:
            ap.error(str(e))
    if not 0 <= args.value_lambda <= 1:
        ap.error("--value-lambda must be in [0, 1]")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    h = w = args.size
    env = make_env(h, args.max_steps)
    net = Net(h, w, frames=args.frames, attn=args.attn, stream=args.incremental, order=args.hybrid_order)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    key = jrandom.PRNGKey(args.seed)
    buffer = deque(maxlen=4096)
    completed = 0
    if args.resume:
        checkpoint = load_checkpoint(args.output)
        old = {"hybrid_order": default_order(4) if args.attn == "hybrid" else None, "value_target": "outcome", "value_lambda": 0.5, **checkpoint["args"]}
        if any(old.get(name, False) != getattr(args, name) for name in ("size", "sims", "max_steps", "seed", "attn", "frames", "incremental", "hybrid_order", "value_target", "value_lambda")):
            ap.error("resume requires matching --size, --sims, --max-steps, --seed, --attn, --frames, --incremental, --hybrid-order, --value-target and --value-lambda")
        net.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        key = jnp.asarray(checkpoint["jax_key"].numpy(), dtype=jnp.uint32)
        buffer.extend((f.numpy(), pi.numpy(), m.numpy(), z, g) for f, pi, m, z, g in checkpoint["replay"])
        completed = checkpoint["completed"]
    for g in range(completed, args.games):
        key, k = jrandom.split(key)
        samples, result = self_play(net, h, w, args.sims, args.max_steps, k, rng, env, args.belief == "on", args.incremental,
                                   args.value_lambda if args.value_target == "mix" else 1.0)
        buffer.extend(samples)
        indices = rng.choice(len(buffer), size=min(512, len(buffer)), replace=False)
        lp, lv = train_step(net, opt, [buffer[int(i)] for i in indices])
        checkpoint = {
            "model": net.state_dict(), "optimizer": opt.state_dict(), "args": vars(args), "channels": CHANNELS,
            "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "jax_key": torch.from_numpy(np.asarray(key).copy()), "completed": g + 1,
            "replay": [(torch.from_numpy(f), torch.from_numpy(pi), torch.from_numpy(m), z, g)
                       for f, pi, m, z, g in buffer],
        }
        torch.save(checkpoint, args.output + ".tmp")
        os.replace(args.output + ".tmp", args.output)
        print(f"game {g + 1}/{args.games}: {result}, {len(samples)} samples, "
              f"policy loss {lp:.3f}, value loss {lv:.3f}; saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
