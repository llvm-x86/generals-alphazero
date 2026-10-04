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
    on = [az.Searcher(net, 8, 8, 12, np.random.default_rng(5), belief=True).search(s, 0, 8, noise=False)
          for s in (first, second)]
    assert np.array_equal(*on)
    bel = np.random.default_rng(0).random(64)
    seen = np.zeros((8, 8), bool)
    guessed = [az.determinize(s, 0, np.random.default_rng(3), bel, seen) for s in (first, second)]
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
    sample = az.determinize(state, 0, np.random.default_rng(0), point, np.zeros((6, 6), bool))
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
    s = az.Searcher(net, 6, 6, 20, np.random.default_rng(0), incremental=True)
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
