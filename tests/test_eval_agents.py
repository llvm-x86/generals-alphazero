import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("ev", pathlib.Path(__file__).parents[1] / "examples" / "eval_agents.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_wilson_known_values():
    lo, hi = ev.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = ev.wilson(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-3)

def test_paired_score_uses_complete_maps_and_draws():
    real, adj, pairs = ev.run("random", "random", 5, 3, 2, 0, return_pairs=True)
    assert real.sum() == adj.sum() == 3
    assert pairs == [0.5]  # at 2 turns no general capture; third game is an incomplete map pair
    assert ev.paired_interval([0.0, 0.5, 1.0]) == pytest.approx((1 / 6, 5 / 6), abs=0.2)

def test_all_draw_maps_have_no_degenerate_bootstrap_claim(capsys):
    ev.main(["random", "random", "--size", "5", "--games", "4", "--max-steps", "2"])
    assert "no observed variation" in capsys.readouterr().out


def test_run_counts_and_ckpt_paths(tmp_path):
    real, adj = ev.run("expander", "random", 6, 4, 30, 0)
    assert real.sum() == adj.sum() == 4
    assert adj[2] <= real[2]  # adjudication only ever resolves draws
    ck = tmp_path / "u.pt"
    torch.save({"model": ev.az.Net(5, 5).state_dict(), "args": {"size": 5, "attn": "dense", "frames": 4}, "channels": ev.az.CHANNELS}, ck)
    for sims in (0, 2):  # policy argmax and PUCT
        real, adj = ev.run(f"ckpt:{ck}", "random", 5, 2, 8, 0, sims)
        assert real.sum() == adj.sum() == 2
    agent = ev.Ckpt(str(ck), 2, 8, 5)
    agent.reset(42)
    first = agent.rng.integers(1_000_000)
    agent.reset(42)
    assert agent.rng.integers(1_000_000) == first
    with pytest.raises(SystemExit):
        ev.run(f"ckpt:{ck}", "random", 6, 2, 8, 0)  # wrong board size
    with pytest.raises(SystemExit):
        ev.make_agent("nope", 0, 8, 5)


def test_loading_agent_b_cannot_change_agent_a(tmp_path):
    """Per-agent semantics: A (heur .5, proxy) is untouched by constructing B (heur 0, draw)."""
    import jax.numpy as jnp
    from generals.core import game
    torch.manual_seed(0)
    paths = {}
    for name, extra in (("a", {"heur": 0.5, "truncation": "proxy"}), ("b", {"heur": 0.0, "truncation": "draw"})):
        paths[name] = tmp_path / f"{name}.pt"
        torch.save({"model": ev.az.Net(5, 5).state_dict(), "args": {"size": 5, "attn": "dense", "frames": 4, **extra},
                    "channels": ev.az.CHANNELS}, paths[name])
    state = game.create_initial_state(jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[4, 4].set(2))
    state = state._replace(armies=state.armies.at[0, 0].set(30))  # material score is nonzero
    a = ev.Ckpt(str(paths["a"]), 2, 8, 5)
    before = a.searcher.make_node(state, 0).value
    b = ev.Ckpt(str(paths["b"]), 2, 8, 5)
    assert a.searcher.make_node(state, 0).value == before
    assert b.searcher.make_node(state, 0).value != before  # B really evaluates with different semantics
    assert (a.cfg.heur, a.cfg.truncation, b.cfg.heur, b.cfg.truncation) == (0.5, "proxy", 0.0, "draw")
    assert not hasattr(ev.az, "HEUR") and not hasattr(ev.az, "TRUNC")


def test_proxy_adjudication_uses_real_score_not_draw_label():
    import jax.numpy as jnp
    from generals.core import game
    state = game.create_initial_state(jnp.zeros((5, 5), dtype=jnp.int32).at[0, 0].set(1).at[4, 4].set(2))
    state = state._replace(armies=state.armies.at[0, 0].set(40), time=jnp.int32(3))
    win, proxy = ev.play(None, [], state, None, 3)  # already at the limit: no moves are played
    assert win == -1 and proxy > 0
