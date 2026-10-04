import importlib.util
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ex = pathlib.Path(__file__).parents[1] / "examples"
spec = importlib.util.spec_from_file_location("bc", ex / "bc_warmstart.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)
az = bc.az


def test_encode_inverts_decode():
    h = w = 6
    for i in (0, 7, 8 * 13 + 5, h * w * 8):
        assert bc.encode(az.decode(i, h, w), h, w) == i


def test_teacher_samples_legal_and_bc_checkpoint_resumes(tmp_path):
    env = az.make_env(5, 12)
    s = bc.rollout(env, 5, 3, 12, np.random.default_rng(0), 0.5)
    assert s and all(m[p.argmax()] and p.sum() == 1 for _, p, m, _, _ in s)
    out = tmp_path / "bc.pt"
    bc.main(["--bc-games", "5", "--bc-epochs", "1", "--size", "5", "--max-steps", "12", "--sims", "2", "--output", str(out)])
    # self-play --resume must accept the BC checkpoint
    az.main(["--games", "1", "--sims", "2", "--size", "5", "--max-steps", "12", "--resume", "--output", str(out)])
    assert az.load_checkpoint(out)["completed"] == 1


def test_argmax_teacher_is_deterministic_legal_and_dagger_runs(tmp_path):
    env = az.make_env(5, 10)
    a = bc.rollout(env, 5, 3, 10, np.random.default_rng(0), 0.0, "argmax")
    b = bc.rollout(env, 5, 3, 10, np.random.default_rng(1), 0.0, "argmax")
    assert all(m[p.argmax()] for _, p, m, _, _ in a) and all((x[1] == y[1]).all() for x, y in zip(a, b))
    bc.main(["--bc-games", "5", "--bc-epochs", "1", "--size", "5", "--max-steps", "10", "--teacher", "argmax",
             "--dagger-rounds", "1", "--dagger-games", "2", "--dagger-epochs", "1", "--output", str(tmp_path / "d.pt")])
