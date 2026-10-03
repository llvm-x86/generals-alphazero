import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("az", pathlib.Path(__file__).parents[1] / "examples" / "az_selfplay.py")
az = importlib.util.module_from_spec(spec)
spec.loader.exec_module(az)


def test_decode_and_search_targets(tmp_path):
    import jax.random as jr
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
