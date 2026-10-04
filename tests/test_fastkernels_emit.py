import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")
root = pathlib.Path(__file__).parents[1] / "examples"
spec = importlib.util.spec_from_file_location("az", root / "az_selfplay.py")
az = importlib.util.module_from_spec(spec)
spec.loader.exec_module(az)
fk = az.fastkernels
pytestmark = pytest.mark.skipif(not fk.available(), reason="needs gcc + AVX-512")


@pytest.mark.parametrize("attn", ["dense", "pisa", "hybrid"])
@pytest.mark.parametrize("h,w", [(8, 8), (5, 7)])
def test_emitted_forward_matches_torch(attn, h, w):
    torch.manual_seed(0)
    net = az.Net(h, w, frames=4, attn=attn).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.1)
        x = torch.rand(3, 4, az.CHANNELS, h, w)
        ref = net(x, True)
        az.FAST = True
        try:
            out = net(x, True)
        finally:
            az.FAST = False
    assert fk.emit._libs, "emitted build was not used"
    assert any(v is not None for v in fk.emit._libs.values())
    assert all((a - b).abs().max() < 1e-4 for a, b in zip(ref, out))
