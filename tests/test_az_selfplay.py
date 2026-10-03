import importlib.util
import pathlib

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from generals.core import game

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("az", pathlib.Path(__file__).parents[1] / "examples" / "az_selfplay.py")
az = importlib.util.module_from_spec(spec)
spec.loader.exec_module(az)


def test_decode_and_search_targets(tmp_path):
    from generals import GeneralsEnv
    h = w = 5
    assert az.decode(h * w * 8, h, w)[0] == 1
    env = GeneralsEnv(grid_dims=(h, w), truncation=20)
    state = env.init_state(jr.PRNGKey(0))
    net = az.Net(h, w)
    s = az.Searcher(net, h, w, 20, np.random.default_rng(0))
    counts = s.search(state, 0, 8)
    assert counts.sum() == 8
    assert counts[: h * w * 8].sum() + counts[-1] == 8
    out = tmp_path / "c.pt"
    az.main(["--games", "1", "--sims", "2", "--size", "5", "--max-steps", "6", "--output", str(out)])
    assert out.exists()

def test_split_mask_excludes_equivalent_two_army_move():
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[4, 4].set(2)
    state = game.create_initial_state(grid)
    for army, expected in ((2, False), (3, True)):
        current = state._replace(armies=state.armies.at[0, 0].set(army))
        moves = az.legal_mask(game.get_observation(current, 0), 5, 5)[:-1].reshape(25, 4, 2)
        assert bool(moves[:, :, 1].any()) == expected


def test_search_cannot_read_hidden_general_position():
    a = jnp.zeros((8, 8), dtype=jnp.int32).at[0, 0].set(1).at[7, 7].set(2).at[5, 5].set(-2)
    b = jnp.zeros((8, 8), dtype=jnp.int32).at[0, 0].set(1).at[6, 7].set(2).at[5, 5].set(40)
    first, second = game.create_initial_state(a), game.create_initial_state(b)
    assert np.array_equal(az.features(game.get_observation(first, 0)),
                          az.features(game.get_observation(second, 0)))
    guessed = [az.determinize(s, 0, np.random.default_rng(3)) for s in (first, second)]
    for field in ("armies", "ownership", "generals", "general_positions", "passable"):
        assert np.array_equal(getattr(guessed[0], field), getattr(guessed[1], field))
    torch.manual_seed(4)
    net = az.Net(8, 8)
    visits = [az.Searcher(net, 8, 8, 12, np.random.default_rng(5)).search(s, 0, 8, noise=False)
              for s in (first, second)]
    assert np.array_equal(*visits)


def test_truncation_is_score_proxy_not_draw():
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[4, 4].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(40), time=jnp.int32(3))
    assert 0 < az.outcome(state, 0, 3) < 0.5
    assert az.outcome(state, 1, 3) == pytest.approx(-az.outcome(state, 0, 3))


def test_opponent_reply_resampled_each_visit():
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[4, 4].set(2)
    state = game.create_initial_state(grid)
    prior = np.zeros(az.num_actions(5, 5), dtype=np.float32)
    prior[-1] = 1
    opponent = prior.copy()
    opponent[0], opponent[-1] = 0.5, 0.5
    node = az.Node(state, prior, opponent, 0)

    class Alternating:
        i = 0

        def choice(self, *args, **kwargs):
            self.i += 1
            return 0 if self.i % 2 else len(prior) - 1

    search = az.Searcher(None, 5, 5, 1, Alternating())
    for _ in range(6):
        search.simulate(node, 0)
    assert set(node.children) == {(len(prior) - 1, 0), (len(prior) - 1, len(prior) - 1)}
    assert node.n[-1] == 6

def test_resume_matches_uninterrupted_training(tmp_path):
    full, resumed = tmp_path / "full.pt", tmp_path / "resumed.pt"
    common = ["--sims", "1", "--size", "4", "--max-steps", "4", "--seed", "13"]
    az.main([*common, "--games", "2", "--output", str(full)])
    az.main([*common, "--games", "1", "--output", str(resumed)])
    az.main([*common, "--games", "2", "--output", str(resumed), "--resume"])
    a = torch.load(full, weights_only=True)
    b = torch.load(resumed, weights_only=True)
    assert a["completed"] == b["completed"] == 2
    assert len(a["replay"]) == len(b["replay"]) == 16
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])

def test_net_is_spacetime_transformer_with_memory():
    h = w = 6
    net = az.Net(h, w).eval()
    assert not any(isinstance(m, (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d,
                                  torch.nn.AdaptiveAvgPool2d)) for m in net.modules())
    x = torch.randn(2, az.FRAMES, az.CHANNELS, h, w)
    logits, value = net(x)
    assert logits.shape == (2, az.num_actions(h, w)) and value.shape == (2,)
    y = x.clone()
    y[:, 0, :, 5, 5] += 3  # change only the OLDEST frame, far from cell (0, 0)
    out = net(y)
    assert not torch.allclose(out[0][:, :8], logits[:, :8])  # past frames reach current-cell logits
    assert not torch.allclose(out[1], value)

def _dense_reference(layer, q, k, v, kg, vg):
    """Exact attention over every cell plus the global token."""
    keys, vals = torch.cat([kg, k], 2), torch.cat([vg, v], 2)
    return torch.nn.functional.scaled_dot_product_attention(q, keys, vals)


def test_pisa_equals_dense_when_every_block_is_kept():
    torch.manual_seed(0)
    layer = az.PisaLayer(32, 2, block=4, topk=16)  # topk >= number of leaf blocks
    q, k, v = (torch.randn(2, 2, 21, 16) for _ in range(3))  # 21 cells: ragged last block + padded blocks
    kg, vg = torch.randn(2, 2, 1, 16), torch.randn(2, 2, 1, 16)
    assert torch.allclose(layer.sparse(q, k, v, kg, vg), _dense_reference(layer, q, k, v, kg, vg), atol=1e-5)


def test_pisa_pyramid_finds_planted_key():
    torch.manual_seed(1)
    layer = az.PisaLayer(32, 2, block=4, topk=2)
    n, d = 64, 16
    q = torch.randn(1, 2, n, d)
    k = torch.randn(1, 2, n, d) * 0.01
    for pos in (5, 37, 62):  # needle in different regions of the pyramid
        kk = k.clone()
        kk[:, :, pos] = 60 * q[:, :, 0] / q[:, :, 0].norm(dim=-1, keepdim=True)
        P = 16
        valid = torch.ones(P * 4, dtype=torch.bool)
        sel = layer.select(q[:, :, :1] * d ** -0.5, kk, valid, P)
        assert sel.shape[-1] == 2 and pos // 4 in sel[0, 0].tolist()


def test_pisa_net_trains_and_matches_shapes():
    torch.manual_seed(2)
    net = az.Net(6, 6, attn="pisa", block=4, topk=2)
    x = torch.randn(2, az.FRAMES, az.CHANNELS, 6, 6)
    logits, value = net(x)
    assert logits.shape == (2, az.num_actions(6, 6)) and torch.isfinite(logits).all()
    (logits.sum() + value.sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
