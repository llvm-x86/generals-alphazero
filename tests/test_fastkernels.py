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


@pytest.mark.parametrize("m,n,k", [(1, 8, 5), (7, 16, 32), (8, 16, 16), (13, 37, 29), (100, 5, 64), (64, 192, 64), (3, 100, 7)])
def test_mm_ragged_and_guard(m, n, k):
    a, b = torch.randn(m, k), torch.randn(k, n)
    buf = torch.full((m * n + 4096,), float("nan"))
    c = buf[: m * n].view(m, n)
    fk.mm(a, b, out=c)
    assert torch.allclose(c, a.double().mm(b.double()).float(), atol=1e-4, rtol=1e-5)
    assert torch.isnan(buf[m * n:]).all()  # nothing written past C


@pytest.mark.parametrize("n,s", [(64, 4), (37, 4), (5, 4), (100, 2), (17, 4)])
def test_sparse_attn_matches_pytorch(n, s):
    torch.manual_seed(n)
    layer = az.PisaLayer(64, 4, block=16, topk=s)
    q, k, v = torch.randn(3, 2, 4, n, 16).unbind(0)
    kg, vg = torch.randn(2, 2, 4, 1, 16).unbind(0)
    q, k, v = (t.transpose(1, 2).contiguous().transpose(1, 2) for t in (q, k, v))  # strided like qkv.permute
    with torch.no_grad():
        ref = layer.sparse(q, k, v, kg, vg)
        az.FAST = True
        try:
            out = layer.sparse(q, k, v, kg, vg)
        finally:
            az.FAST = False
    assert (out - ref).abs().max() < 1e-5


@pytest.mark.parametrize("n", [5, 37, 100, 300])
def test_select_matches_pytorch(n):
    torch.manual_seed(n)
    layer = az.PisaLayer(64, 4, block=16, topk=4)
    q, k = torch.randn(2, 2, 4, n, 16).unbind(0)
    P = 1 << (-(-n // 16) - 1).bit_length()
    valid = torch.nn.functional.pad(torch.ones(n, dtype=torch.bool), (0, P * 16 - n))
    k = torch.nn.functional.pad(k, (0, 0, 0, P * 16 - n))
    with torch.no_grad():
        ref = layer.select(q, k, valid, P)
        az.FAST = True
        try:
            out = layer.select(q, k, valid, P)
        finally:
            az.FAST = False
    assert torch.equal(out, ref)


@pytest.mark.parametrize("attn,size,frames,batch", [("dense", 5, 4, 2), ("pisa", 7, 3, 3), ("hybrid", 9, 4, 2), ("pisa", 1, 2, 2), ("hybrid", 3, 1, 2)])
def test_whole_net_matches_pytorch(attn, size, frames, batch):
    torch.manual_seed(0)
    net = az.Net(size, size, frames=frames, attn=attn).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.1)  # break the zero-initialised parameters
        x = torch.rand(batch, frames, az.CHANNELS, size, size)
        ref = net(x, True)
        az.FAST = True
        try:
            out = net(x, True)
            for p in net.parameters():  # a weight update must reach the C path
                p.add_(0.05)
            out2 = net(x, True)
            az.FAST = False
            ref2 = net(x, True)
            az.FAST = True
        finally:
            az.FAST = False
    assert all((a - b).abs().max() < 1e-4 for a, b in zip(ref, out))
    assert all((a - b).abs().max() < 1e-4 for a, b in zip(ref2, out2))


def test_stream_net_uses_fast_sparse_path():
    torch.manual_seed(0)
    net = az.Net(6, 6, frames=3, attn="hybrid", stream=True).eval()
    x = torch.rand(2, 3, az.CHANNELS, 6, 6)
    with torch.no_grad():
        ref = net(x)
        az.FAST = True
        try:
            out = net(x)
        finally:
            az.FAST = False
    assert all((a - b).abs().max() < 1e-4 for a, b in zip(ref, out))
