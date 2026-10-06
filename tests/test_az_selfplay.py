import importlib.util
import types
import pathlib

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
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
    on = [az.Searcher(net, 8, 8, 12, np.random.default_rng(5), az.SearchCfg(belief=True)).search(s, 0, 8, noise=False)
          for s in (first, second)]
    assert np.array_equal(*on)
    bel = np.random.default_rng(0).random(64)
    guessed = [az.determinize(s, 0, np.random.default_rng(3), bel) for s in (first, second)]
    for field in ("armies", "ownership", "generals", "general_positions", "passable"):
        assert np.array_equal(getattr(guessed[0], field), getattr(guessed[1], field))


def test_memory_planes_update_on_hand_built_sequence():
    grid = jnp.zeros((6, 6), dtype=jnp.int32).at[0, 0].set(1).at[5, 5].set(2).at[1, 1].set(-2)
    state = game.create_initial_state(grid)
    obs = game.get_observation(state, 0)
    f0 = az.features(obs)
    assert f0[15, 0, 0] == 1 and f0[15, 5, 5] == 0  # own corner seen, far corner not
    assert f0[17, 1, 1] == 1  # adjacent mountain remembered
    # the next observation has the mountain's cell and the origin visible again but the general far away:
    # memory must carry planes from the previous stack, and never lose a sighting.
    far = game.get_observation(state, 0)._replace(
        fog_cells=jnp.ones((6, 6), bool), structures_in_fog=jnp.zeros((6, 6), bool),
        mountains=jnp.zeros((6, 6), bool), opponent_cells=jnp.zeros((6, 6), bool))
    f1 = az.features(far, az.stack_with(None, f0, 2))
    assert f1[15, 0, 0] == 1 and f1[17, 1, 1] == 1  # remembered while fogged
    assert f1[8, 0, 0] == 1  # current view really is fogged
    # enemy general sighting is sticky and last-seen enemy cells follow visibility
    seen_enemy = game.get_observation(state, 0)._replace(
        opponent_cells=jnp.zeros((6, 6), bool).at[3, 3].set(True),
        generals=jnp.zeros((6, 6), bool).at[3, 3].set(True),
        fog_cells=jnp.zeros((6, 6), bool), structures_in_fog=jnp.zeros((6, 6), bool))
    f2 = az.features(seen_enemy, az.stack_with(None, f1, 2))
    assert f2[19].sum() == 1 and f2[19, 3, 3] == 1 and f2[16, 3, 3] == 1
    f3 = az.features(far, az.stack_with(None, f2, 2))
    assert f3[19, 3, 3] == 1 and f3[16, 3, 3] == 1  # still remembered once fogged again


def test_belief_is_masked_and_determinize_places_general_from_it():
    net = az.Net(6, 6)
    x = torch.zeros(1, az.FRAMES, az.CHANNELS, 6, 6)
    x[:, -1, 15, :2] = 1  # top two rows ever seen
    x[:, -1, 6, 5, 5] = 1  # own cell
    bel = net(x, True)[2][0].reshape(6, 6)
    assert (bel[:2] < -1e8).all() and bel[5, 5] < -1e8 and (bel[2:5] > -1e8).all()
    grid = jnp.zeros((6, 6), dtype=jnp.int32).at[0, 0].set(1).at[5, 5].set(2)
    state = game.create_initial_state(grid)
    point = np.zeros(36)
    point[3 * 6 + 4] = 1  # all belief mass on (3, 4)
    sample = az.determinize(state, 0, np.random.default_rng(0), point)
    assert tuple(np.asarray(sample.general_positions[1])) == (3, 4)
    assert bool(sample.ownership[1][3, 4])


def test_incompatible_checkpoint_rejected(tmp_path):
    out = tmp_path / "c.pt"
    az.main(["--games", "1", "--sims", "1", "--size", "4", "--max-steps", "3", "--output", str(out)])
    assert az.load_checkpoint(out)["channels"] == az.CHANNELS
    ck = torch.load(out, weights_only=True)
    del ck["channels"]  # what a pre-memory-planes checkpoint looks like
    torch.save(ck, out)
    with pytest.raises(ValueError, match="input channels"):
        az.load_checkpoint(out)
    with pytest.raises(ValueError, match="input channels"):
        az.main(["--games", "2", "--sims", "1", "--size", "4", "--max-steps", "3", "--output", str(out), "--resume"])


def test_truncation_is_score_proxy_not_draw():
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[4, 4].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(40), time=jnp.int32(3))
    assert az.outcome(state, 0, 3) == 0.0  # default: draw
    assert 0 < az.outcome(state, 0, 3, "proxy") < 0.5
    assert az.outcome(state, 1, 3, "proxy") == pytest.approx(-az.outcome(state, 0, 3, "proxy"))


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
    logits, value, bel = net(x, True)
    assert logits.shape == (2, az.num_actions(6, 6)) and torch.isfinite(logits).all()
    (logits.sum() + value.sum() + bel.clamp(min=-1e3).sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_history_length_is_configurable_end_to_end():
    first = az.stack_with(None, np.ones((az.CHANNELS, 6, 6), np.float32), 16)
    assert first.shape == (16, az.CHANNELS, 6, 6)
    nxt = az.stack_with(first, np.zeros((az.CHANNELS, 6, 6), np.float32), 16)
    assert nxt.shape == first.shape and nxt[-1].sum() == 0 and nxt[0].sum() > 0
    net = az.Net(6, 6, frames=16, attn="pisa", block=8, topk=2)
    logits, value = net(torch.from_numpy(nxt[None]))
    assert logits.shape == (1, az.num_actions(6, 6)) and torch.isfinite(value).all()


def _gdn_inputs(T=6, H=2, dk=8, dv=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    q, k, v = r(1, T, H, dk), torch.nn.functional.normalize(r(1, T, H, dk), dim=-1), r(1, T, H, dv)
    alpha, b, w = (torch.rand(1, T, H, d, generator=g) * 0.5 + 0.4 for d in (dk, dk, dv))
    return q, k, v, alpha, b, w


def test_gdn2_scan_matches_explicit_matrix_form():
    q, k, v, alpha, b, w = _gdn_inputs()
    got = az.gdn2_scan(q, k, v, alpha, b, w)
    for h in range(2):  # S_t = (I - k (b*k)^T) D S_{t-1} + k (w*v)^T ; o_t = S_t^T q_t (eq. 10)
        S = torch.zeros(8, 8)
        for t in range(6):
            kt = k[0, t, h]
            S = (torch.eye(8) - torch.outer(kt, b[0, t, h] * kt)) @ (alpha[0, t, h, :, None] * S) \
                + torch.outer(kt, w[0, t, h] * v[0, t, h])
            assert torch.allclose(got[0, t, h], S.T @ q[0, t, h], atol=1e-5)


def test_gdn2_overwrites_instead_of_accumulating_and_write_gate_is_selective():
    k = torch.nn.functional.normalize(torch.randn(1, 2, 1, 8), dim=-1)
    k = torch.cat([k[:, :1], k[:, :1]], 1)  # same key written twice
    v = torch.stack([torch.ones(1, 1, 8), -torch.ones(1, 1, 8)], 1).reshape(1, 2, 1, 8)
    ones = torch.ones(1, 2, 1, 8)
    out = az.gdn2_scan(k, k, v, ones, ones, ones)  # query = key, no decay, full erase/write
    assert torch.allclose(out[0, 1, 0], -torch.ones(8), atol=1e-5)  # latest value, not 1 + (-1)
    w = ones.clone()
    w[..., :4] = 0  # write gate off on the first four value channels
    out = az.gdn2_scan(k, k, v, ones, ones, w)
    assert torch.allclose(out[0, 1, 0, :4], torch.zeros(4), atol=1e-6)
    assert torch.allclose(out[0, 1, 0, 4:], -torch.ones(4), atol=1e-5)


def test_hybrid_net_remembers_oldest_frame_and_trains():
    torch.manual_seed(3)
    net = az.Net(6, 6, frames=16, attn="hybrid", block=8, topk=2).eval()
    x = torch.randn(1, 16, az.CHANNELS, 6, 6)
    logits, value = net(x)
    y = x.clone()
    y[:, 0, :, 0, 0] += 3  # oldest frame, cell (0, 0): reaches current-cell logits via the GDN state
    assert not torch.allclose(net(y)[0][:, :8], logits[:, :8])
    net.train()
    out, v, bel = net(x, True)
    (out.sum() + v.sum() + bel.clamp(min=-1e3).sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def _stream_net(frames, seed=5):
    torch.manual_seed(seed)
    return az.Net(6, 6, frames=frames, attn="hybrid", block=8, topk=2, stream=True).eval()


def _step_all(net, x, with_belief=True):
    state = None
    with torch.no_grad():
        for t in range(x.shape[1]):
            out, state = net.step(x[:, t], state, with_belief)
    return out


def test_incremental_equals_window_rebuild_when_window_covers_game():
    net, x = _stream_net(6), torch.randn(2, 6, az.CHANNELS, 6, 6)
    with torch.no_grad():
        want = net(x, True)
    for a, b in zip(_step_all(net, x), want):  # logits, value, belief
        assert torch.allclose(a, b, atol=1e-4)


def test_incremental_state_outlives_the_window():
    net, x = _stream_net(3), torch.randn(1, 6, az.CHANNELS, 6, 6)
    with torch.no_grad():
        rebuilt = net(x[:, -3:])  # the window only holds the last 3 of 6 frames
    inc = _step_all(net, x, False)
    assert not torch.allclose(inc[0], rebuilt[0], atol=1e-4)  # GDN state (and cached keys) still carry frames 0-2
    assert torch.allclose(_step_all(net, x[:, -3:], False)[0], rebuilt[0], atol=1e-4)  # same once the history fits


def test_incremental_search_child_costs_one_step_and_trains(tmp_path):
    net = _stream_net(4)
    state = az.make_env(6, 20).init_state(jr.PRNGKey(0))
    s = az.Searcher(net, 6, 6, 20, np.random.default_rng(0), az.SearchCfg(incremental=True))
    root = s.make_node(state, 0)
    assert root.st is not None
    s.simulate(root, 0)
    child = next(c for c, _ in root.children.values() if c is not None)
    x = torch.from_numpy(np.stack([root.stack[-1], child.stack[-1]]))[None]  # child == fold the two frames
    with torch.no_grad():
        assert abs(float(net(x)[1]) - child.value) < 1e-4
    out = tmp_path / "inc.pt"
    az.main(["--games", "1", "--sims", "2", "--size", "6", "--max-steps", "3", "--attn", "hybrid", "--frames", "4",
             "--incremental", "--belief", "on", "--output", str(out)])
    assert az.load_checkpoint(out)["args"]["incremental"]


def test_gdn2_scan_matches_symbolic_closed_form():
    """proofs/information_flow.py proves p_T = w sum_s lam^(T-s) v_s with lam = a(1-b) for unit k."""
    T, dk, dv = 7, 5, 3
    g = torch.Generator().manual_seed(4)
    k = torch.nn.functional.normalize(torch.randn(dk, generator=g), dim=0)
    v = torch.randn(T, dv, generator=g)
    a, b, w = 0.9, 0.35, 0.7
    full = lambda x, d: torch.full((1, T, 1, d), x)
    got = az.gdn2_scan(k.expand(1, T, 1, dk), k.expand(1, T, 1, dk), v.reshape(1, T, 1, dv),
                       full(a, dk), full(b, dk), full(w, dv))[0, :, 0]  # query = key reads p_t
    lam = a * (1 - b)
    for t in range(T):
        want = w * sum(lam ** (t - s) * v[s] for s in range(t + 1))
        assert torch.allclose(got[t], want, atol=1e-5)


def _fog_states(size, n, steps=60):
    from generals import get_observation
    from generals.agents import ExpanderAgent
    env, agent, key, out = az.make_env(size, steps), ExpanderAgent(), jr.PRNGKey(0), []
    while len(out) < n:
        key, k = jr.split(key)
        state = env.init_state(k)
        while int(state.time) < steps and int(state.winner) < 0:
            obs = [get_observation(state, p) for p in (0, 1)]
            if int(state.time) >= 4 and int(state.time) % 4 == 0:
                out += [(state, 0), (state, 1)]
            key, a, b = jr.split(key, 3)
            state, _ = game.step(state, jnp.stack([agent.act(obs[0], a), agent.act(obs[1], b)]), general_trade=True)
    return out[:n]


@pytest.mark.parametrize("size", [6, 8])
def test_make_env_boards(size):
    from generals.core.grid import bfs_distance_field
    env = az.make_env(size, 50)
    for i in range(10):
        s = env.init_state(jr.PRNGKey(i))
        assert int(np.asarray(s.castles).sum()) == min(10, size * size // 6)
        g = np.asarray(s.general_positions)
        d = int(bfs_distance_field(s.passable, tuple(g[0]))[tuple(g[1])])
        assert max(3, size // 2) <= d < size * size  # far enough apart and connected


@pytest.mark.parametrize("size", [6, 8])
def test_determinized_castle_fraction_matches_real(size):
    from generals import get_observation
    rng = np.random.default_rng(1)
    real = det = total = 0
    for state, me in _fog_states(size, 200):
        m = np.asarray(get_observation(state, me).structures_in_fog)
        d = az.determinize(state, me, rng)
        real += int((np.asarray(state.castles) & m).sum())
        det += int((np.asarray(d.castles) & m).sum())
        total += int(m.sum())
    assert total > 0
    assert abs(real - det) / total < 0.05


def test_mix_value_target_math():
    z, q = np.array([1.0, -1.0, 0.25]), np.array([0.2, 0.4, -0.5])
    assert np.allclose(az.mix_target(z, q, 1.0), z)
    assert np.allclose(az.mix_target(z, q, 0.0), q)
    assert np.allclose(az.mix_target(z, q, 0.5), [0.6, -0.3, -0.125])


def test_hybrid_order_validation_and_non_default_forward():
    for bad in ("GPG", "GGGG", "PPPP", "GPGX", "gpgp"):
        with pytest.raises(ValueError):
            az.check_order(bad, 4)
    assert az.default_order(4) == "GPGP"
    net = az.Net(5, 5, frames=3, attn="hybrid", order="PGPG").eval()
    assert [type(m).__name__ for m in net.body] == ["PisaLayer", "GdnLayer"] * 2
    x = torch.rand(2, 3, az.CHANNELS, 5, 5)
    with torch.no_grad():
        ref = net(x, True)
        assert all(torch.isfinite(t).all() for t in ref)
        if az.fastkernels.available():
            az.FAST = True
            try:
                out = net(x, True)
            finally:
                az.FAST = False
            assert all((a - b).abs().max() < 1e-4 for a, b in zip(ref, out))


def test_improved_policy_follows_q_and_keeps_prior_without_visits():
    prior = np.array([0.5, 0.3, 0.2, 0.0])
    n, w = np.zeros(4), np.zeros(4)
    assert np.allclose(az.improved_policy(prior, n, w, 0.0), prior)  # no visits: the prior
    n, w = np.array([0.0, 2.0, 0.0, 0.0]), np.array([0.0, 1.6, 0.0, 0.0])  # action 1: q = +0.8 vs root value 0
    p = az.improved_policy(prior, n, w, 0.0)
    assert p[1] > 0.3 and p[3] == 0 and p.sum() == pytest.approx(1, abs=1e-6)


def test_self_play_vs_scripted_opponent_records_only_the_net_side():
    from generals.agents import ExpanderAgent
    net = az.Net(h=5, w=5, width=16, layers=2, heads=2, frames=2, attn="dense").eval()
    env = az.make_env(5, 20)
    samples, _ = az.self_play(net, 5, 5, 2, 6, jr.PRNGKey(0), np.random.default_rng(0), env, opp=ExpanderAgent())
    assert len(samples) == 6  # one side only (a full self-play game records 12)


def test_pass_is_always_legal_and_truncation_is_a_draw_by_default():
    env = az.make_env(6, 5)
    state = env.init_state(jr.PRNGKey(0))
    assert az.legal_mask(az.get_observation(state, 0), 6, 6)[-1]  # a move exists AND pass is legal
    assert az.SearchCfg().truncation == "draw" and az.outcome(state._replace(time=jnp.int32(5)), 0, 5) == 0.0


def _sample(i, pi, z, term="capture", n=4, frames=2):
    m = np.ones(n * n * 8 + 1, bool)
    f = np.random.default_rng(i).random((frames, az.CHANNELS, n, n)).astype(np.float32)
    f[:, 19] = 0  # general never sighted: belief label is trained on
    return az.Sample(f, pi.astype(np.float32), m, z, int(i % (n * n)), 0, i, 0, z if term == "capture" else 0.0, term, 0.0, 0.0, 0.0)


def test_train_metrics_are_sample_weighted_and_match_a_direct_oracle():
    """Heterogeneous targets, a partial final minibatch (6 samples, batch 4) and lr 0: the reported online loss and the
    pre/post fit must equal the per-sample weighted means computed directly; mean-of-batch-means would differ."""
    torch.manual_seed(0)
    net = az.Net(h=4, w=4, width=16, layers=2, heads=2, frames=2, attn="dense")
    rng = np.random.default_rng(1)
    A = 129
    samples = []
    for i in range(6):
        pi = np.zeros(A)
        pi[rng.choice(A, 2 + 3 * i, replace=False)] = 1
        samples.append(_sample(i, pi / pi.sum(), [1.0, -1.0, 0.0, 0.0, 1.0, 0.0][i], "capture" if i in (0, 1, 4) else "time_limit"))
    with torch.no_grad():
        x = torch.from_numpy(np.stack([s.f for s in samples]))
        logits, v, bel = net(x, True)
        pi_t = torch.from_numpy(np.stack([s.pi for s in samples]))
        ce = -(pi_t * torch.log_softmax(logits, 1)).sum(1)
        ent = -(pi_t * pi_t.clamp_min(1e-12).log()).sum(1)
        z = torch.tensor([s.z for s in samples])
        se = (v - z) ** 2
    out = az.train_step(net, torch.optim.SGD(net.parameters(), lr=0.0), samples, epochs=1, batch=4)
    for stats in (out["pre"], out["post"]):
        assert stats["n"] == 6 and stats["n_capture"] == 3 and stats["n_time_limit"] == 3
        assert stats["ce"] == pytest.approx(ce.mean().item(), rel=1e-4) and stats["entropy"] == pytest.approx(ent.mean().item(), rel=1e-4)
        assert stats["kl"] == pytest.approx((ce - ent).mean().item(), rel=1e-3, abs=1e-5)
        assert stats["v_mse"] == pytest.approx(se.mean().item(), rel=1e-4)
        assert stats["v_mse_capture"] == pytest.approx(se[[0, 1, 4]].mean().item(), rel=1e-4)
        assert stats["v_mse_time_limit"] == pytest.approx(se[[2, 3, 5]].mean().item(), rel=1e-4)
        assert stats["v_base_zero"] == pytest.approx((z ** 2).mean().item()) and stats["v_base_mean"] == pytest.approx(z.var(unbiased=False).item())
    assert out["online"]["policy"] == pytest.approx(ce.mean().item(), rel=1e-4)  # weights 4/6 and 2/6, not 1/2 and 1/2
    batch_means = (ce[:4].mean() + ce[4:].mean()) / 2
    assert abs(batch_means.item() - ce.mean().item()) > 1e-3  # the data really distinguishes the two aggregations
    assert out["online"]["value"] == pytest.approx(se.mean().item(), rel=1e-4) and out["updates"] == 2


def test_training_reduces_the_probe_fit():
    torch.manual_seed(0)
    net = az.Net(h=4, w=4, width=16, layers=2, heads=2, frames=2, attn="dense")
    pi = np.zeros(129)
    pi[[3, 7]] = 0.5
    samples = [_sample(i, pi, 1.0) for i in range(8)]
    out = az.train_step(net, torch.optim.Adam(net.parameters(), lr=3e-3), samples, epochs=30, batch=4)
    assert out["post"]["kl"] < out["pre"]["kl"] * 0.5 and out["post"]["v_mse"] < out["pre"]["v_mse"]


def _tiny(tmp_path, name="c.pt", extra=(), games="1"):
    out = tmp_path / name
    az.main(["--games", games, "--sims", "2", "--size", "4", "--max-steps", "4", "--seed", "3", "--output", str(out), *extra])
    return out


def test_checkpoint_and_replay_carry_schema_outcome_and_episode_ids(tmp_path):
    out = _tiny(tmp_path, games="2", extra=["--snapshot-every", "1"])
    ck = az.load_checkpoint(out, resume=True)
    assert ck["schema"] == az.SCHEMA and ck["semantics"]["truncation"] == "draw" and ck["code"]["sha256"]
    assert ck["semantics"]["objective"] == "finite-horizon-4/draw/shape=0.0/heur=0.0"
    reps = [az.Sample.load(d) for d in ck["replay"]]
    assert {r.ep for r in reps} == {0, 1}  # one episode id per game
    assert all(r.term == "time_limit" and r.raw == 0.0 and r.z == 0.0 for r in reps)  # draw objective, raw outcome kept
    assert all(r.me in (0, 1) and 0 <= r.t < 4 for r in reps)
    snap = torch.load(f"{out}.g1", weights_only=True)  # snapshots keep their own immutable identity
    assert snap["semantics"] == ck["semantics"] and snap["code"] == ck["code"] and snap["args"]["sims"] == 2 and "optimizer" not in snap


def test_self_play_records_raw_capture_outcome_separately_from_the_trained_target():
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[0, 1].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(10).at[0, 1].set(1))

    class Env:
        def init_state(self, key):
            return state

    net = az.Net(h=5, w=5, width=16, layers=2, heads=2, frames=2, attn="dense").eval()
    samples, result = az.self_play(net, 5, 5, 4, 10, jr.PRNGKey(0), np.random.default_rng(0), Env(), shape=0.5, episode=7)
    assert result.startswith("player") and {s.ep for s in samples} == {7} and all(s.term == "capture" for s in samples)
    winner = int(result.split()[1])
    for s in samples:
        assert s.raw == (1.0 if s.me == winner else -1.0)
        assert s.z == pytest.approx(0.5 * s.raw + 0.5 * s.ph)  # trained target mixes shape; raw stays untouched


def test_legacy_checkpoints_are_eval_only_and_changed_semantics_cannot_resume(tmp_path):
    out = _tiny(tmp_path)
    ck = torch.load(out, weights_only=True)
    for key in ("schema", "semantics", "code", "code_lineage"):
        del ck[key]
    legacy = tmp_path / "legacy.pt"
    torch.save(ck, legacy)
    assert az.load_checkpoint(legacy)["model"]  # loadable for evaluation
    with pytest.raises(ValueError, match="evaluation-only"):
        az.load_checkpoint(legacy, resume=True)
    with pytest.raises(ValueError, match="evaluation-only"):
        az.main(["--games", "2", "--sims", "2", "--size", "4", "--max-steps", "4", "--seed", "3", "--output", str(legacy), "--resume"])
    for flag in (["--truncation", "proxy"], ["--heur", "0.3"], ["--shape", "0.2"], ["--belief", "on"], ["--opp-frac", "0.5"]):
        with pytest.raises(SystemExit):  # argparse error: incompatible semantics are never mixed silently
            az.main(["--games", "2", "--sims", "2", "--size", "4", "--max-steps", "4", "--seed", "3", "--output", str(out), "--resume", *flag])
    az.main(["--games", "2", "--sims", "2", "--size", "4", "--max-steps", "4", "--seed", "3", "--output", str(out), "--resume"])
    assert torch.load(out, weights_only=True)["completed"] == 2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_search_finds_forced_capture_that_prior_ignores(seed):
    """Tactical gate: capturing the adjacent general wins now (full or half stack). At 32 sims search must put
    >0.9 of its target on those moves, and play one, whatever the untrained prior thinks."""
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[0, 1].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(10).at[0, 1].set(1))
    torch.manual_seed(seed)
    net = az.Net(h=5, w=5, width=16, layers=2, heads=2, frames=2, attn="dense").eval()
    s = az.Searcher(net, 5, 5, 50, np.random.default_rng(seed))
    s.search(state, 0, 32)
    wins = [6, 7]  # cell 0, RIGHT, full / half
    assert s.target[wins].sum() > 0.9 and s.best in wins


def test_search_determinization_respects_memory_without_belief_head():
    """Belief head off: every sampled enemy general (no sighting yet) avoids ever-seen cells and is hidden."""
    env = az.make_env(6, 50)
    state = env.init_state(jr.PRNGKey(3))
    seen = az.features(az.get_observation(state, 0))[15] > 0.5
    net = az.Net(h=6, w=6, width=16, layers=2, heads=2, frames=2, attn="dense").eval()
    s = az.Searcher(net, 6, 6, 50, np.random.default_rng(0))
    orig, got = az.determinize, []
    az.determinize = lambda *a, **k: got.append(orig(*a, **k)) or got[-1]
    try:
        s.search(state, 0, 8)
    finally:
        az.determinize = orig
    assert len(got) == 4
    for x in got:
        r, c = np.asarray(x.general_positions[1])
        assert not seen[r, c] and bool(x.ownership[1][r, c]) and bool(x.generals[r, c])


# --- memory-consistent determinization -------------------------------------------------------------------

def _lost_scout():
    """Player 0's scout sees the enemy general (5,5) and a mountain (3,4), then is lost. No general swap occurred."""
    grid = jnp.zeros((6, 6), dtype=jnp.int32).at[0, 0].set(1).at[5, 5].set(2).at[3, 4].set(-2)
    s = game.create_initial_state(grid)
    scout = s._replace(ownership=s.ownership.at[0, 4, 4].set(True), ownership_neutral=s.ownership_neutral.at[4, 4].set(False),
                       armies=s.armies.at[4, 4].set(3))
    mem = az.GeneralMemory()
    obs = game.get_observation(scout, 0)
    mem.update(obs)
    prev = az.stack_with(None, az.features(obs), 2)
    lost = scout._replace(ownership=scout.ownership.at[0, 4, 4].set(False).at[1, 4, 4].set(True))
    obs = game.get_observation(lost, 0)
    mem.update(obs)
    return lost, az.features(obs, prev), mem


def _same_view(a, b):
    oa, ob = game.get_observation(a, 0), game.get_observation(b, 0)
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(oa, ob))


def test_known_general_and_mountain_survive_in_every_sample():
    """Reproduced failure: 0/32 samples kept the remembered general and 14/32 reclassified the remembered mountain."""
    lost, frame, mem = _lost_scout()
    assert mem.enemy == (5, 5) and frame[19, 5, 5] == 1 and frame[17, 3, 4] == 1
    obs = game.get_observation(lost, 0)
    assert bool(obs.fog_cells[5, 5]) and bool(obs.structures_in_fog[3, 4])  # both really are hidden now
    for belief in (None, np.ones(36)):
        for k in range(32):
            x = az.determinize(lost, 0, np.random.default_rng(k), belief, frame, mem.enemy)
            assert tuple(np.asarray(x.general_positions[1])) == (5, 5) and bool(x.generals[5, 5]) and bool(x.ownership[1][5, 5])
            assert bool(x.mountains[3, 4]) and not bool(x.castles[3, 4])
            assert _same_view(lost, x)  # current observation and public scoreboard are reproduced exactly
    assert int(np.asarray(x.ownership[1]).sum()) == int(obs.opponent_land_count) == 2


def test_enemy_territory_may_reoccupy_ever_seen_cells_but_general_may_not():
    lost, frame, _ = _lost_scout()
    seen = frame[15] > 0.5
    assert seen[4, 4] and bool(lost.ownership[1][4, 4])  # the enemy really took the cell we once saw
    land_on_seen, general_on_seen = 0, 0
    for k in range(96):
        x = az.determinize(lost, 0, np.random.default_rng(k), None, frame, None)  # no remembered general: sampled
        land_on_seen += bool(x.ownership[1][4, 4])
        gr, gc = np.asarray(x.general_positions[1])
        general_on_seen += bool(seen[gr, gc])
    assert land_on_seen > 0 and general_on_seen == 0


def test_general_swap_moves_the_remembered_site_and_clears_the_stale_sighting():
    grid = jnp.zeros((6, 6), dtype=jnp.int32).at[0, 0].set(1).at[0, 1].set(2)
    s = game.create_initial_state(grid)
    s = s._replace(armies=s.armies.at[0, 0].set(30).at[0, 1].set(30))
    mem, obs = az.GeneralMemory(), game.get_observation(s, 0)
    assert mem.update(obs) == (0, 1)
    prev = az.stack_with(None, az.features(obs), 2)
    acts = jnp.asarray([[0, 0, 0, 3, 0], [0, 0, 1, 2, 0]], dtype=jnp.int32)  # each general attacks the other
    after, _ = game.step(s, acts, general_trade=True)
    assert tuple(np.asarray(after.general_positions[0])) == (0, 1)  # generals really traded sites
    obs = game.get_observation(after, 0)
    frame = az.features(obs, prev)
    assert mem.update(obs) == (0, 0)
    assert frame[19, 0, 1] == 0 and frame[19, 0, 0] == 1  # stale sighting cleared, new one recorded
    for k in range(8):
        x = az.determinize(after, 0, np.random.default_rng(k), None, frame, mem.enemy)
        assert tuple(np.asarray(x.general_positions[1])) == (0, 0) and bool(x.generals[0, 0])
        assert tuple(np.asarray(x.general_positions[0])) == (0, 1)


def test_general_memory_swap_inference_when_new_site_is_hidden():
    """Own general changed cell and no enemy general is visible: it sits on our old site."""
    from types import SimpleNamespace
    z = np.zeros((4, 4), bool)

    def view(own_gen, fog_cell=None):
        g = z.copy(); g[own_gen] = True
        fog = z.copy()
        if fog_cell:
            fog[fog_cell] = True
        return SimpleNamespace(generals=g, owned_cells=g, opponent_cells=z, fog_cells=fog, structures_in_fog=z)

    mem = az.GeneralMemory()
    mem.update(view((0, 0)))
    assert mem.enemy is None
    assert mem.update(view((3, 3), fog_cell=(0, 0))) == (0, 0)
    assert mem.update(view((3, 3), fog_cell=(0, 0))) == (0, 0)  # remembered while fogged
    assert mem.update(view((3, 3))) is None  # visible without a general: dropped


def test_determinize_noninterference_with_inaccessible_true_state():
    """Two worlds with identical observation and memory but different hidden truth give identical samples."""
    lost, frame, mem = _lost_scout()
    other = lost._replace(  # the enemy's second cell (and its 3 armies) sit elsewhere, still fogged
        ownership=lost.ownership.at[1, 4, 4].set(False).at[1, 4, 3].set(True),
        ownership_neutral=lost.ownership_neutral.at[4, 4].set(True).at[4, 3].set(False),
        armies=lost.armies.at[4, 4].set(0).at[4, 3].set(3))
    assert _same_view(lost, other)
    for known in (mem.enemy, None):
        for belief in (None, np.random.default_rng(1).random(36)):
            for k in range(6):
                a, b = (az.determinize(w, 0, np.random.default_rng(k), belief, frame, known) for w in (lost, other))
                for field in a._fields:
                    assert np.array_equal(np.asarray(getattr(a, field)), np.asarray(getattr(b, field))), field


def test_determinize_never_falls_back_to_the_true_hidden_general():
    grid = jnp.zeros((6, 6), dtype=jnp.int32).at[0, 0].set(1).at[5, 5].set(2)
    s = game.create_initial_state(grid)
    # degenerate view: the only enemy land is a visible non-general cell, so no hidden cell can host the general
    s = s._replace(ownership=s.ownership.at[1, 5, 5].set(False).at[1, 0, 1].set(True),
                   ownership_neutral=s.ownership_neutral.at[5, 5].set(True).at[0, 1].set(False))
    x = az.determinize(s, 0, np.random.default_rng(0))
    assert tuple(np.asarray(x.general_positions[1])) == (0, 1)
    assert not bool(x.generals[5, 5])


# --- Gumbel root schedule vs the pinned mctx reference ---------------------------------------------------

def _mctx_sequence(max_num_considered_actions, num_simulations):
    """Independent copy of google-deepmind/mctx mctx/_src/seq_halving.py get_sequence_of_considered_visits
    (only commit of that file: 1232e22097c8bf84da68babd8326fdc65c64c2e0), kept here as the oracle."""
    import math
    if max_num_considered_actions <= 1:
        return tuple(range(num_simulations))
    log2max = int(math.ceil(math.log2(max_num_considered_actions)))
    sequence = []
    visits = [0] * max_num_considered_actions
    num_considered = max_num_considered_actions
    while len(sequence) < num_simulations:
        num_extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
        for _ in range(num_extra_visits):
            sequence.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        num_considered = max(2, num_considered // 2)
    return tuple(sequence[:num_simulations])


@pytest.mark.parametrize("m,sims", [(16, 32), (16, 100), (5, 60), (3, 60), (8, 10), (16, 20)])
def test_schedule_matches_mctx_reference(m, sims):
    assert az.considered_visits(m, sims) == _mctx_sequence(m, sims)
    assert len(az.considered_visits(m, sims)) == sims
    for k in range(1, 18):  # odd, tiny and ragged counts too
        assert az.considered_visits(k, sims) == _mctx_sequence(k, sims)


class _Bandit(az.Searcher):
    """Production Searcher.search with controlled leaf values: the reward of a simulation depends only on the root
    action, its visit number and the particle. Everything else (schedule, selection, targets) is production code."""

    def __init__(self, m, reward, cfg=az.SearchCfg()):
        super().__init__(types.SimpleNamespace(frames=2), 6, 6, 200, np.random.default_rng(0), cfg)
        self.m, self.reward, self.calls, self.seen_n = m, reward, [], np.zeros(289, int)

    def make_node(self, *args, **kwargs):
        p = np.zeros(289)
        p[:self.m] = 1 / self.m
        return types.SimpleNamespace(prior=p, value=0.0, pid=0)

    def simulate(self, root, me, first=None):
        a = int(first)
        v = self.reward(a, int(self.seen_n[a]), root)
        self.calls.append((a, int(self.seen_n[a]), root.pid, self._u, v))
        self.seen_n[a] += 1
        self._last = v


@pytest.fixture
def bandit_env(monkeypatch):
    state = game.create_initial_state(jnp.zeros((6, 6), dtype=jnp.int32).at[0, 0].set(1).at[5, 5].set(2))
    monkeypatch.setattr(az, "determinize", lambda state, *a, **k: state)

    def setup(m):
        monkeypatch.setattr(az, "legal_mask", lambda *a: np.arange(289) < m)
    return state, setup


def _run_bandit(bandit_env, m, sims, reward, seed=0, noise=False, cfg=az.SearchCfg()):
    state, setup = bandit_env
    setup(m)
    s = _Bandit(m, reward, cfg)
    s.rng = np.random.default_rng(seed)
    n = s.search(state, 0, sims, noise=noise)
    return s, n


def _budget(m, sims, particles=4, max_cand=16):
    """Independent restatement of the per-decision budget rule: (worlds R, blocks B, candidates)."""
    R = max(1, min(particles, sims // 2))
    return R, sims // R, min(m, max_cand, sims // R)


@pytest.mark.parametrize("m,sims", [(5, 32), (16, 32), (16, 100), (5, 60), (3, 60), (8, 10), (16, 20), (7, 3), (1, 9), (2, 1), (16, 1), (7, 7)])
def test_every_simulation_is_accounted_and_follows_the_schedule(bandit_env, m, sims):
    """One visit = a block of R simulations, one per world; the mctx schedule runs over blocks, so every considered
    candidate is measured in all R worlds before anything can be eliminated; leftovers go to finalists."""
    s, n = _run_bandit(bandit_env, m, sims, lambda a, k, r: 0.1 * a)
    R, B, mc = _budget(m, sims)
    assert n.sum() == sims == len(s.calls)
    assert s.plan["worlds"] == R and s.plan["blocks"] == B and s.plan["candidates"] == mc and s.plan["leftover"] == sims - R * B
    table = _mctx_sequence(mc, B)
    blocks = [s.calls[i * R:(i + 1) * R] for i in range(B)]
    for want, block in zip(table, blocks):
        assert len({c[0] for c in block}) == 1  # one action per block
        assert [c[1] for c in block] == [R * want + i for i in range(R)]  # its want-th visit, all R worlds in order
    assert n[m:].sum() == 0 and s.target[m:].sum() == 0 and abs(s.target.sum() - 1) < 1e-5
    assert n[n > 0].min() >= R  # nothing is ever judged on fewer worlds than declared
    assert bool(s.plan["limited"]) == (R < 4 or mc < min(m, 16))  # a reduced plan is always declared


def test_budget_plan_declares_what_it_cannot_cover(bandit_env):
    """32 sims, 16 candidates, 4 worlds would need 64: the plan must say so and keep 4 worlds on 8 candidates, not 16 on 1."""
    s, n = _run_bandit(bandit_env, 16, 32, lambda a, k, r: 0.0)
    assert (s.plan["worlds"], s.plan["candidates"], s.plan["blocks"]) == (4, 8, 8)
    assert len(s.plan["limited"]) == 1 and "8 candidates < 16" in s.plan["limited"][0]
    assert (n > 0).sum() == 8 and n[n > 0].min() == 4
    s, n = _run_bandit(bandit_env, 16, 256, lambda a, k, r: 0.0)  # enough budget: no limitation, all 16 get 4 worlds
    assert s.plan["limited"] == [] and (n > 0).sum() == 16 and n[n > 0].min() >= 4


def test_late_evidence_can_change_the_played_action():
    """5 actions, 32 sims = 8 blocks of 4 worlds. Action 0 looks great on its first block and then collapses; action 1 is a
    steady +0.3. Halving must keep testing both finalists, and the second block exposes action 0."""
    state = game.create_initial_state(jnp.zeros((6, 6), dtype=jnp.int32).at[0, 0].set(1).at[5, 5].set(2))
    reward = lambda a, k, r: (1.0 if k < 4 else -1.0) if a == 0 else 0.3 if a == 1 else -1.0
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(az, "determinize", lambda state, *a, **k: state)
        mp.setattr(az, "legal_mask", lambda *a: np.arange(289) < 5)
        s = _Bandit(5, reward)
        n = s.search(state, 0, 32, noise=False)
    finally:
        mp.undo()
    q = np.bincount([c[0] for c in s.calls], weights=[c[4] for c in s.calls], minlength=289) / np.maximum(n, 1)
    assert n.sum() == 32 and n[0] >= 8 and n[1] >= 12 and (n[2:5] == 4).all()
    assert q[0] < q[1] and q[1] == pytest.approx(0.3)
    assert s.best == 1 and int(s.target.argmax()) == 1


def _reference_root(prior_m, gumbel, reward, sims, cfg):
    """numpy oracle of mctx gumbel_muzero_root_action_selection + completed-Q (qtransform_completed_by_mix_value,
    rescale_values=True, value_scale=cfg.c_scale, maxvisit_init=cfg.c_visit) + final selection of policies.py
    (`considered_visit = max(visit_counts)`), written directly from the reference formulas, with the visit unit changed
    to a block of R worlds (R, B from _budget); `reward(a, k)` is the k-th sample of action a."""
    m = len(prior_m)
    logits = np.log(prior_m)
    R, B, mc = _budget(m, sims, cfg.particles, cfg.max_cand)
    n, w, nb = np.zeros(m), np.zeros(m), np.zeros(m)
    table = _mctx_sequence(mc, B)

    def completed():
        q = np.where(n > 0, w / np.maximum(n, 1), 0.0)
        pr = np.exp(logits) / np.exp(logits).sum()
        sum_n = n.sum()
        wq = (np.where(n > 0, pr * q, 0).sum() / pr[n > 0].sum()) if sum_n else 0.0
        mix = (0.0 + sum_n * wq) / (sum_n + 1)  # raw value 0
        c = np.where(n > 0, q, mix)
        c = (c - c.min()) / max(c.max() - c.min(), 1e-8)
        return (cfg.c_visit + n.max()) * cfg.c_scale * c

    def pick(considered):
        lg = logits - logits.max()
        score = np.maximum(-1e9, gumbel + lg + completed()) + np.where(nb == considered, 0, -np.inf)
        return int(np.argmax(score))

    for want in table:
        a = pick(want)
        for _ in range(R):
            w[a] += reward(a, int(n[a]))
            n[a] += 1
        nb[a] += 1
    for _ in range(sims - B * R):
        a = pick(nb.max())
        w[a] += reward(a, int(n[a]))
        n[a] += 1
    best = pick(nb.max())
    policy = np.exp(logits + completed())
    return n, best, policy / policy.sum()


@pytest.mark.parametrize("m,sims", [(5, 32), (16, 32), (16, 100), (8, 10), (3, 60), (16, 20)])
def test_production_root_matches_reference_oracle_with_specified_gumbel(bandit_env, m, sims):
    cfg = az.SearchCfg(qscale="mctx", c_scale=0.1)
    rewards = np.random.default_rng(7).uniform(-1, 1, size=(m, 3))
    reward = lambda a, k: rewards[a, min(k // 4, 2)]
    for seed in range(4):
        s, n = _run_bandit(bandit_env, m, sims, lambda a, k, r: reward(a, k), seed, noise=True, cfg=cfg)
        gumbel = np.random.default_rng(seed).gumbel(size=289)  # Searcher draws these right after the roots (stubs consume no rng)
        n_ref, best_ref, pol_ref = _reference_root(np.full(m, 1 / m), gumbel[:m], reward, sims, cfg)
        assert np.array_equal(n[:m], n_ref)
        assert s.best == best_ref
        assert np.allclose(s.target[:m], pol_ref, atol=1e-5)  # target is the clean improved policy


def test_gumbel_result_and_clean_target_are_distinct():
    """With a heavy Gumbel draw the played action can differ from the target argmax; the target is noise free."""
    state = game.create_initial_state(jnp.zeros((6, 6), dtype=jnp.int32).at[0, 0].set(1).at[5, 5].set(2))
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(az, "determinize", lambda state, *a, **k: state)
        mp.setattr(az, "legal_mask", lambda *a: np.arange(289) < 6)
        runs = []
        for seed in range(30):
            s = _Bandit(6, lambda a, k, r: 0.0)  # all actions look identical
            s.rng = np.random.default_rng(seed)
            s.search(state, 0, 6)
            runs.append((s.best, s.target.copy()))
        s0 = _Bandit(6, lambda a, k, r: 0.0)
        s0.search(state, 0, 6, noise=False)
    finally:
        mp.undo()
    assert len({b for b, _ in runs}) > 1  # exploration noise spreads the played action ...
    assert all(np.allclose(t, runs[0][1]) for _, t in runs)  # ... while the target stays the same noise-free policy


def test_particles_are_paired_and_cover_all_worlds():
    """Every considered candidate is run in the same 4 worlds (particle, stratified reply uniform), in every block."""
    state = game.create_initial_state(jnp.zeros((6, 6), dtype=jnp.int32).at[0, 0].set(1).at[5, 5].set(2))
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(az, "legal_mask", lambda *a: np.arange(289) < 16)
        mp.setattr(az, "determinize", lambda state, *a, **k: state)
        for seed in range(12):
            s = _Bandit(16, lambda a, k, r: 0.0)
            s.rng = np.random.default_rng(seed)
            made = []
            s.make_node = lambda *a, made=made, **k: (made.append(len(made)), types.SimpleNamespace(
                prior=np.where(np.arange(289) < 16, 1 / 16, 0.0), value=0.0, pid=made[-1]))[1]
            s.search(state, 0, 32, noise=False)
            worlds = {}
            for a, k, pid, u, v in s.calls:
                worlds.setdefault(a, []).append((pid, u))
            assert len(worlds) == 8 and all(len(w) == 4 for w in worlds.values())  # 8 candidates x 4 worlds
            assert all(w == worlds[next(iter(worlds))] for w in worlds.values())  # identical (particle, reply) pairs: paired
            assert sorted(pid for pid, _ in worlds[next(iter(worlds))]) == [0, 1, 2, 3]  # all particles used by each candidate
            assert sorted(int(u * 4) for _, u in worlds[next(iter(worlds))]) == [0, 1, 2, 3]  # one reply uniform per stratum
    finally:
        mp.undo()


def test_paired_root_reply_is_shared_and_valid():
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[4, 4].set(2)
    state = game.create_initial_state(grid)
    prior = np.zeros(az.num_actions(5, 5), np.float32)
    prior[-1] = 1
    opponent = np.zeros_like(prior)
    opponent[[0, 5, -1]] = [0.2, 0.3, 0.5]
    search = az.Searcher(None, 5, 5, 1, np.random.default_rng(0))
    node = az.Node(state, prior, opponent, 0)
    for u, want in ((0.0, 0), (0.19, 0), (0.21, 5), (0.49, 5), (0.51, len(prior) - 1), (0.999999, len(prior) - 1)):
        search._u = u
        search.simulate(node, 0, len(prior) - 1)
        assert (len(prior) - 1, want) in node.children


# --- tactical decisions with a controlled opponent (exact one-step action values) ----------------------------

def _scripted_opponent_search(monkeypatch, state, enemy_action, sims=32, seed=0):
    """Search where the net is replaced by: uniform own prior, value 0, and an opponent that always plays `enemy_action`."""
    def fake_eval(net, stacks, masks):
        mine = masks[0] / masks[0].sum()
        theirs = masks[1] / masks[1].sum()  # deeper positions: the scripted move may no longer exist
        if masks[1][enemy_action]:
            theirs = np.zeros(len(masks[1]))
            theirs[enemy_action] = 1.0
        return np.stack([mine, theirs]).astype(np.float32), np.zeros(2, np.float32)

    monkeypatch.setattr(az, "evaluate", fake_eval)
    s = az.Searcher(types.SimpleNamespace(frames=2), 5, 5, 50, np.random.default_rng(seed))
    s.search(state, 0, sims)
    return s


def _true_values(state, acts, enemy_action):
    """Exact one-step value for every root action vs the scripted reply: +-1 on a capture, 0 if the game goes on."""
    out = {}
    for a in acts:
        pair = jnp.asarray([az.decode(a, 5, 5), az.decode(enemy_action, 5, 5)], dtype=jnp.int32)
        nxt, _ = game.step(state, pair, general_trade=True)
        out[a] = 0.0 if int(nxt.winner) < 0 else (1.0 if int(nxt.winner) == 0 else -1.0)
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_search_defends_the_general_by_reinforcing_it(monkeypatch, seed):
    """Enemy stack (7) beside my general (3) always attacks with 6. Only moving my stack (1,0) up onto the general
    resolves first (defensive moves go first) and holds it; waiting and every other move loses the game."""
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[4, 4].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(3).at[0, 1].set(7).at[1, 0].set(10),
                           ownership=state.ownership.at[1, 0, 1].set(True).at[0, 1, 0].set(True),
                           ownership_neutral=state.ownership_neutral.at[0, 1].set(False).at[1, 0].set(False))
    enemy = 1 * 8 + 2 * 2 + 0  # cell (0, 1), LEFT, full stack
    s = _scripted_opponent_search(monkeypatch, state, enemy, seed=seed)
    legal = np.flatnonzero(s.target > 0)
    values = _true_values(state, legal, enemy)
    good = [a for a, v in values.items() if v == 0.0]
    assert sorted(good) == [40, 41] and len(legal) > 8  # (1,0) UP full / half hold the general; the rest lose
    assert values[az.num_actions(5, 5) - 1] == -1.0  # waiting loses
    prior_value = np.mean(list(values.values()))  # uniform prior's expected value
    target_value = sum(s.target[a] * v for a, v in values.items())
    assert s.target[good].sum() > 0.9 and s.best in good and target_value > prior_value + 0.5


def test_search_trades_generals_when_waiting_loses(monkeypatch):
    """Adjacent generals, enemy always attacks with 29 > my 28: waiting loses; attacking back is a general trade (value 0)."""
    grid = jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[0, 1].set(2)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(28).at[0, 1].set(30))
    enemy = 1 * 8 + 2 * 2 + 0
    s = _scripted_opponent_search(monkeypatch, state, enemy)
    legal = np.flatnonzero(s.target > 0)
    values = _true_values(state, legal, enemy)
    good = [a for a, v in values.items() if v == 0.0]
    assert sorted(good) == [6, 7]  # RIGHT full / half from my general: the trade; everything else loses the game
    prior_value = np.mean(list(values.values()))
    assert s.target[good].sum() > 0.9 and s.best in good
    assert sum(s.target[a] * v for a, v in values.items()) > prior_value + 0.5


# --- exact simultaneous-move toys: Monte Carlo root values vs exact expectation ------------------------------
# Fully visible 5x5 positions (every enemy cell is adjacent to one of mine), a controlled net (uniform own prior, value
# 0, scripted opponent distribution at the root) and a horizon that makes every child terminal, so the exact value of a
# root action is E_reply[outcome] by enumerating the real engine's joint step. No hidden state: determinize is the identity.

def _grid(*cells):
    g = jnp.zeros((5, 5), dtype=jnp.int32)
    for (r, c), v in cells:
        g = g.at[r, c].set(v)
    return game.create_initial_state(g)


def _toy_defense():
    """Enemy stack 7 beside my general 3 attacks with 6 (0.3) or waits (0.7): only reinforcing from (1,0) holds."""
    s = _grid(((0, 0), 1), ((0, 1), 2))
    return s._replace(armies=s.armies.at[0, 0].set(3).at[0, 1].set(7).at[1, 0].set(10),
                      ownership=s.ownership.at[0, 1, 0].set(True),
                      ownership_neutral=s.ownership_neutral.at[1, 0].set(False)), {1 * 8 + 2 * 2: 0.3, 200: 0.7}


def _toy_trade():
    """Adjacent generals 28 v 30. The enemy attacks (0.7) or waits (0.3): waiting then loses to an attack, attacking back trades."""
    s = _grid(((0, 0), 1), ((0, 1), 2))
    return s._replace(armies=s.armies.at[0, 0].set(28).at[0, 1].set(30)), {1 * 8 + 2 * 2: 0.7, 200: 0.3}


def _toy_two_turn():
    """Two-turn tactic: my 15 stack at (0,2) must first join (0,1) (which sees the enemy general at (0,0)), then capture.
    The enemy general only waits or shuffles. Horizon 2: nothing is terminal after one ply for most moves."""
    s = _grid(((4, 4), 1), ((0, 0), 2))
    s = s._replace(armies=s.armies.at[0, 2].set(15).at[0, 1].set(1).at[4, 4].set(1),
                   ownership=s.ownership.at[0, 0, 2].set(True).at[0, 0, 1].set(True),
                   ownership_neutral=s.ownership_neutral.at[0, 2].set(False).at[0, 1].set(False))
    return s, {200: 1.0}


def _controlled_eval(opp):
    """az.evaluate replacement: uniform own prior, value 0, the opponent plays `opp` ({action: p}) wherever those moves are legal."""
    def fake(net, stacks, masks):
        mine = masks[0] / masks[0].sum()
        theirs = masks[1] / masks[1].sum()
        if all(masks[1][a] for a in opp):
            theirs = np.zeros(len(masks[1]))
            for a, p in opp.items():
                theirs[a] = p
        return np.stack([mine, theirs]).astype(np.float32), np.zeros(2, np.float32)
    return fake


def _exact_q(state, opp, horizon=1, end=None):
    """Exact value of every legal root action: E_reply[outcome], the game cut `horizon` steps after the root (draw = 0).
    Beyond one step the continuation is the best own follow-up against the same scripted reply (a reference, not what
    PUCT's interior computes)."""
    end = int(state.time) + horizon if end is None else end
    mask = az.legal_mask(game.get_observation(state, 0), 5, 5)
    q = {}
    for a in np.flatnonzero(mask):
        tot = 0.0
        for o, p in opp.items():
            nxt, _ = game.step(state, jnp.asarray([az.decode(int(a), 5, 5), az.decode(o, 5, 5)], dtype=jnp.int32), general_trade=True)
            term = az.outcome(nxt, 0, end, "draw")
            tot += p * (term if term is not None else max(_exact_q(nxt, opp, end=end).values()))
        q[int(a)] = tot
    return q


def _toy_runs(monkeypatch, toy, seeds, sims=32, horizon=1, cfg=az.SearchCfg(), diag=True):
    state, opp = toy
    monkeypatch.setattr(az, "evaluate", _controlled_eval(opp))
    runs = []
    for seed in range(seeds):
        s = az.Searcher(types.SimpleNamespace(frames=2), 5, 5, int(state.time) + horizon, np.random.default_rng(seed), cfg)
        s.search(state, 0, sims, diag=diag)
        runs.append(s)
    return runs


def _stats(runs, ex):
    mc = {}
    for s in runs:
        for a, c in s.summary["candidates"].items():
            mc.setdefault(a, []).append(c["q"])
    return {a: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)), len(v)) for a, v in mc.items()}


@pytest.mark.parametrize("toy", [_toy_defense, _toy_trade])
def test_toy_monte_carlo_q_is_calibrated_and_the_action_is_exactly_optimal(monkeypatch, toy):
    """Exact E_reply[outcome] by enumeration vs search's Q over 120 fixed seeds: mean within 4 standard errors (across
    seeds), zero spread where the outcome is reply-independent, and the played action is exact-optimal on >= 97% of seeds."""
    t = toy()
    ex = _exact_q(*t)
    runs = _toy_runs(monkeypatch, t, 120)
    best = [a for a, v in ex.items() if v == max(ex.values())]
    assert len(set(round(v, 3) for v in ex.values())) == 2  # replies matter: two different consequences
    # the 8-candidate cap (11 legal at 32 sims) may leave every good move out on rare seeds: declared limit, bounded rate
    assert np.mean([s.best in best for s in runs]) >= 0.97
    for a, (mean, se, n) in _stats(runs, ex).items():
        assert n > 30 and abs(mean - ex[a]) <= 4 * se + 1e-9, (a, mean, ex[a], se)


def test_diag_summary_accounts_for_every_sample(monkeypatch):
    t = _toy_trade()
    (s,) = _toy_runs(monkeypatch, t, 1)
    d, plan = s.summary, s.plan
    cand = d["candidates"]
    assert sum(c["n"] for c in cand.values()) == plan["sims"] == int(s.root_n.sum())
    assert sum(d["depth"].values()) == plan["sims"]
    assert d["edges"][1]["new"] + d["edges"][1]["hit"] == plan["sims"]
    for a, c in cand.items():
        assert c["q"] == pytest.approx(s.root_w[a] / s.root_n[a]) and c["worlds"] <= plan["worlds"] and c["n"] >= plan["worlds"]
        if c["n"] > 1:
            assert c["se"] == pytest.approx(np.sqrt(c["var"] / c["n"]))
    assert abs(s.target.sum() - 1) < 1e-5 and (s.target[s.root_n == 0] >= 0).all()
    (s2,) = _toy_runs(monkeypatch, t, 1, diag=False)  # counters are off unless requested, and never change the decision
    assert s2.summary is None and s2.best == s.best and np.array_equal(s2.root_n, s.root_n)


def test_toy_two_turn_tactic_reports_depth_but_is_not_solved_at_32_sims(monkeypatch):
    """Measured limit, not a pass: with a value-blind net the win needs the 2nd visit of a root edge AND the right follow-up.
    Revisits do reach depth 2 (fixed worlds), but at 32 sims most candidates get one block, so the exact +1 is not found."""
    t = _toy_two_turn()
    ex = _exact_q(*t, horizon=2)
    assert sorted(a for a, v in ex.items() if v == 1.0) == [20, 21] and sum(v == 0 for v in ex.values()) == 5
    runs = _toy_runs(monkeypatch, t, 20, horizon=2)
    depth = {}
    for s in runs:
        for k, c in s.summary["depth"].items():
            depth[k] = depth.get(k, 0) + c
    assert depth.get(2, 0) > 0 and depth[1] > depth[2]
    assert all(c["q"] == 0.0 for s in runs for c in s.summary["candidates"].values())  # the +1 never reaches Q at this budget


def test_two_turn_search_with_exact_leaf_value(monkeypatch):
    """The 32-sim search finds a two-ply capture when the leaf value encodes its one-ply continuation.
    A value-blind net does not solve it at this budget (test above); this isolates search from value learning."""
    state, opp = _toy_two_turn()
    start = int(state.time)
    make_node = az.Searcher.make_node

    def exact_leaf(self, state, me, *args, **kwargs):
        node = make_node(self, state, me, *args, **kwargs)
        if int(state.time) > start and int(state.winner) < 0:
            node.value = max(_exact_q(state, opp, end=start + 2).values())
        return node

    monkeypatch.setattr(az.Searcher, "make_node", exact_leaf)
    runs = _toy_runs(monkeypatch, (state, opp), 10, horizon=2)
    assert all(s.best in {20, 21} for s in runs)
    assert all(any(c["q"] > 0 for c in s.summary["candidates"].values()) for s in runs)


def test_search_decision_depends_only_on_the_observation(monkeypatch):
    """Two states with the same observation and public totals but a different hidden enemy general site search identically."""
    def hidden(site):
        s = game.create_initial_state(jnp.zeros((6, 6), dtype=jnp.int32).at[0, 0].set(1).at[site].set(2))
        return s
    net = az.Net(h=6, w=6, width=16, layers=2, heads=2, frames=2, attn="dense").eval()
    out = []
    for site in ((5, 5), (4, 5)):
        s = az.Searcher(net, 6, 6, 50, np.random.default_rng(1))
        n = s.search(hidden(site), 0, 8)
        out.append((n, s.best, s.target))
    assert np.array_equal(out[0][0], out[1][0]) and out[0][1] == out[1][1] and np.allclose(out[0][2], out[1][2])
