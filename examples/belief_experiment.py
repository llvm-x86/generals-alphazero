"""
Belief-head accuracy experiment: Expander-vs-Expander games on make_env boards, split BY GAME.

    python examples/belief_experiment.py gen 0 100 rows0.npz      # games seed 0..99
    python examples/belief_experiment.py train rows_*.npz         # lowest 300 game ids = test, next 100 = validation, rest = train

Rows are (own-view frame history, enemy general cell) sampled every 5 steps while the enemy general is
unseen. Reports mean P(true general) of the trained az_selfplay.Net belief head vs uniform over the same
plausible-cell support, on held-out games, at the epoch (1-3) chosen on the validation games.
"""
import importlib.util
import pathlib
import sys

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
import torch.nn.functional as F

from generals import get_observation
from generals.agents import ExpanderAgent
from generals.core.game import step

spec = importlib.util.spec_from_file_location("az", pathlib.Path(__file__).with_name("az_selfplay.py"))
az = importlib.util.module_from_spec(spec)
spec.loader.exec_module(az)
SIZE, STEPS, EVERY, LR = 8, 80, 5, 2e-3


def gen(lo, hi, out):
    env, agent = az.make_env(SIZE, STEPS), ExpanderAgent()
    X, Y, G = [], [], []
    for g in range(lo, hi):
        key, state = jr.PRNGKey(g), env.init_state(jr.PRNGKey(g))
        hist = [None, None]
        while int(state.winner) < 0 and int(state.time) < STEPS:
            obs = [get_observation(state, p) for p in (0, 1)]
            for p in (0, 1):
                hist[p] = az.stack_with(hist[p], az.features(obs[p], hist[p]))
                pos = np.asarray(state.general_positions[1 - p])
                if hist[p][-1, 19].sum() == 0 and int(state.time) % EVERY == 0:
                    X.append(hist[p].astype(np.float16)); Y.append(pos[0] * SIZE + pos[1]); G.append(g)
            key, k1, k2 = jr.split(key, 3)
            state, _ = step(state, jnp.stack([agent.act(obs[0], k1), agent.act(obs[1], k2)]), general_trade=True)
    np.savez(out, x=np.stack(X), y=np.array(Y), g=np.array(G))
    print(out, len(X), "rows from", hi - lo, "games")


def train(files, n_test=300, n_val=100, epochs=3):
    d = [np.load(f) for f in files]
    x, y, g = (np.concatenate([a[k] for a in d]) for k in "xyg")
    games = np.unique(g)
    test, val = games[:n_test], games[n_test:n_test + n_val]  # lowest game ids held out
    sets = {"train": ~np.isin(g, np.r_[test, val]), "val": np.isin(g, val), "test": np.isin(g, test)}
    print({k: (int(v.sum()), "rows") for k, v in sets.items()}, "games train/val/test:",
          len(games) - n_test - n_val, len(val), len(test))
    torch.manual_seed(0)
    net = az.Net(SIZE, SIZE, width=48, layers=2)
    opt = torch.optim.Adam(net.parameters(), LR, weight_decay=1e-4)
    tx, ty = torch.from_numpy(x[sets["train"]].astype(np.float32)), torch.from_numpy(y[sets["train"]]).long()

    def score(name):
        xs, ys = torch.from_numpy(x[sets[name]].astype(np.float32)), torch.from_numpy(y[sets[name]]).long()
        with torch.no_grad():
            logits = torch.cat([net(xs[i:i + 256], True)[2] for i in range(0, len(xs), 256)])
        ok = logits[torch.arange(len(ys)), ys] > -1e8  # true cell inside the support
        p = F.softmax(logits, 1)[torch.arange(len(ys)), ys]
        support = (logits > -1e8).sum(1).float()
        return float(p.mean()), float((1 / support).mean()), float(ok.float().mean())

    best = None
    for e in range(1, epochs + 1):
        perm = torch.randperm(len(tx))
        for i in range(0, len(tx), 64):
            j = perm[i:i + 64]
            loss = F.cross_entropy(net(tx[j], True)[2], ty[j])
            opt.zero_grad(); loss.backward(); opt.step()
        v = score("val")
        print(f"epoch {e}: val P(true) {v[0]:.4f} uniform {v[1]:.4f} support-covers-truth {v[2]:.4f}", flush=True)
        if best is None or v[0] > best[0]:
            best = (v[0], e, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[2])
    t = score("test")
    print(f"chosen epoch {best[1]} (on val); TEST P(true) learned {t[0]:.4f} uniform-over-plausible {t[1]:.4f} "
          f"ratio {t[0] / t[1]:.2f}x support-covers-truth {t[2]:.4f}")
    return best[1], t


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        train(sys.argv[2:])
