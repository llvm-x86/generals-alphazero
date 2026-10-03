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


def test_run_counts_and_ckpt_paths(tmp_path):
    real, adj = ev.run("expander", "random", 6, 4, 30, 0)
    assert real.sum() == adj.sum() == 4
    assert adj[2] <= real[2]  # adjudication only ever resolves draws
    ck = tmp_path / "u.pt"
    torch.save({"model": ev.az.Net(5, 5).state_dict(), "args": {"size": 5, "attn": "dense", "frames": 4}, "channels": ev.az.CHANNELS}, ck)
    for sims in (0, 2):  # policy argmax and PUCT
        real, adj = ev.run(f"ckpt:{ck}", "random", 5, 2, 8, 0, sims)
        assert real.sum() == adj.sum() == 2
    with pytest.raises(SystemExit):
        ev.run(f"ckpt:{ck}", "random", 6, 2, 8, 0)  # wrong board size
    with pytest.raises(SystemExit):
        ev.make_agent("nope", 0, 8, 5)
