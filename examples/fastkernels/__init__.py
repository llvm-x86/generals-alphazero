"""ctypes loader for the fast AVX-512 kernels (see d2_nr16_u2.c / pisa_attn.c).

Builds on first use into a gitignored cache; if gcc or AVX-512 is missing, `available()` is False,
one notice is printed, and callers keep the PyTorch path. float32 contiguous tensors only."""
import ctypes
import hashlib
import shutil
import subprocess
from pathlib import Path

import torch

from . import emit as _emit

_HERE = Path(__file__).parent
_SRC = ["fastnet.c", "pisa_attn.c"]
_DEPS = ["d2_nr16_u2.c", "pieces.c"]  # #included by fastnet.c
EMIT = True  # use emit.py's specialised build when available
_lib = None
_tried = False


def _load():
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    try:
        if "avx512f" not in Path("/proc/cpuinfo").read_text():
            raise RuntimeError("no AVX-512")
        if shutil.which("gcc") is None:
            raise RuntimeError("no gcc")
        srcs = [_HERE / s for s in _SRC]
        tag = hashlib.sha1(b"".join(s.read_bytes() for s in srcs + [_HERE / f for f in _DEPS])).hexdigest()[:12]
        so = _HERE / ".cache" / f"libfast_{tag}.so"
        if not so.exists():
            so.parent.mkdir(exist_ok=True)
            tmp = so.with_suffix(f".{id(srcs)}.tmp")
            subprocess.run(["gcc", "-O3", "-march=native", "-mprefer-vector-width=512", "-funroll-loops", "-fPIC",
                            "-shared", *map(str, srcs), "-o", str(tmp), "-lm"], check=True, capture_output=True)
            tmp.replace(so)
        lib = ctypes.CDLL(str(so))
        P = ctypes.c_void_p
        i = ctypes.c_int
        lib.sgemm_direct2.argtypes = [i, i, i, P, i, P, i, P, i]
        lib.net_forward.argtypes = [i] * 7 + [P, i, i, P, P, P, P, P]
        lib.net_forward.restype = i
        lib.fk_prof = (ctypes.c_double * 8).in_dll(lib, "fk_prof")
        lib.pisa_sparse_attn.argtypes = [i] * 7 + [ctypes.c_float] + [P] * 10
        lib.pisa_sparse_attn.restype = i
        lib.pisa_group_lse.argtypes = [i] * 7 + [P] * 5
        lib.pisa_group_lse.restype = i
        _lib = lib
    except Exception as e:  # noqa: BLE001 - any failure degrades to PyTorch
        print(f"fastkernels unavailable ({e}); using the PyTorch path")
    return _lib


def available():
    return _load() is not None


def mm(a, b, out=None):
    """a (M,K) @ b (K,N), float32, unit inner stride (row strides arbitrary), via sgemm_direct2."""
    lib = _load()
    (m, k), n = a.shape, b.shape[1]
    assert a.dtype == b.dtype == torch.float32 and b.shape[0] == k and a.stride(1) == b.stride(1) == 1
    out = torch.empty(m, n) if out is None else out
    lib.sgemm_direct2(m, n, k, a.data_ptr(), a.stride(0), b.data_ptr(), b.stride(0), out.data_ptr(), n)
    return out


def sparse_attn(q, k, v, kg, vg, sel, block):
    """q (B,H,N,d); k,v (B,H,Nk,d) unpadded; kg,vg (B,H,1,d); sel (B,N,S) int64. Returns (B,H,N,d) or None if d>128."""
    lib = _load()
    B, H, N, d = q.shape
    for t in (q, k, v, kg, vg):
        assert t.dtype == torch.float32 and t.stride(3) == 1
    sel = sel.contiguous()
    out = torch.empty(B, H, N, d)
    L = lambda *x: (ctypes.c_long * len(x))(*x)  # noqa: E731
    r = lib.pisa_sparse_attn(B, H, N, d, k.shape[2], block, sel.shape[-1], d ** -0.5, q.data_ptr(),
                             L(q.stride(0), q.stride(1), q.stride(2)), k.data_ptr(), v.data_ptr(),
                             L(k.stride(0), k.stride(1), k.stride(2)), kg.data_ptr(), vg.data_ptr(),
                             L(kg.stride(0), kg.stride(1)), sel.data_ptr(), out.data_ptr())
    return out if r == 0 else None


def group_lse(q, X, valid, A, G):
    """Sum over heads of logsumexp of q.X over the G valid rows of group A[b,n,a]; q (B,H,N,d), X (B,H,R,d)
    contiguous, valid (R,) bool, A (B,N,na) int64. Returns (B,N,na), or None if unsupported."""
    lib = _load()
    B, H, N, d = q.shape
    q, X, A, valid = q.contiguous(), X.contiguous(), A.contiguous(), valid.to(torch.uint8).contiguous()
    out = torch.empty(B, N, A.shape[-1])
    r = lib.pisa_group_lse(B, H, N, d, X.shape[2], G, A.shape[-1], q.data_ptr(), X.data_ptr(), valid.data_ptr(),
                           A.data_ptr(), out.data_ptr())
    return out if r == 0 else None


def _pack(net):
    """Flat pointer list matching net_forward's weight order; linear weights are kept transposed (K,N)."""
    keep = []

    def t(lin):  # nn.Linear -> (W^T, bias)
        w = lin.weight.detach().t().contiguous()
        keep.append(w)
        return [w, lin.bias.detach()]

    def ln(m):
        return [m.weight.detach(), m.bias.detach()]

    P = [*t(net.inp), net.pos.detach(), net.tpos.detach(), net.glob.detach(), *ln(net.norm), *t(net.pol),
         net.pass_logit.weight.detach().reshape(-1), net.pass_logit.bias.detach(),
         net.bel.weight.detach().reshape(-1), net.bel.bias.detach(), *t(net.val[0]), net.val[2].weight.detach().reshape(-1),
         net.val[2].bias.detach()]
    kinds = []
    for layer in net.body.layers if hasattr(net.body, "layers") else net.body:
        if hasattr(layer, "self_attn"):  # nn.TransformerEncoderLayer, relu, norm_first
            kinds.append(0)
            a = layer.self_attn
            w = a.in_proj_weight.detach().t().contiguous()
            keep.append(w)
            P += [*ln(layer.norm1), w, a.in_proj_bias.detach(), *t(a.out_proj), *ln(layer.norm2), *t(layer.linear1), *t(layer.linear2)]
        elif hasattr(layer, "log_a"):  # GdnLayer
            kinds.append(2)
            w = torch.cat([layer.qkv.weight, layer.alpha.weight, layer.erase.weight, layer.write.weight], 0).detach().t().contiguous()
            b = torch.cat([layer.qkv.bias, layer.alpha.bias, layer.erase.bias, layer.write.bias]).detach().contiguous()
            keep += [w, b]
            P += [*ln(layer.n1), w, b, layer.log_a.detach(), layer.delta.detach(), *ln(layer.onorm), *t(layer.proj),
                  *ln(layer.n2), *t(layer.mlp[0]), *t(layer.mlp[2])]
        else:  # PisaLayer
            kinds.append(1)
            P += [*ln(layer.n1), *t(layer.qkv), *t(layer.proj), *ln(layer.n2), *t(layer.mlp[0]), *t(layer.mlp[2])]
    assert all(p.dtype == torch.float32 and p.is_contiguous() for p in P)
    arr = (ctypes.c_void_p * len(P))(*[p.data_ptr() for p in P])
    return arr, (ctypes.c_int * len(kinds))(*kinds), keep + P


def net_forward(net, x, block, topk):
    """Whole-net inference forward in one C call. x (B,T,CH,H,W) float32. Returns (logits, value, belief) or None."""
    lib = _load()
    ver = tuple(p._version for p in net.parameters())
    hit = net.__dict__.get("_fk")  # (param versions, pointers, kinds, tensors kept alive)
    if hit is None or hit[0] != ver:
        hit = net.__dict__["_fk"] = (ver, *_pack(net))
    _, arr, kinds_c, _keep = hit
    kinds = list(kinds_c)
    x = x.contiguous()
    B, T, CH, h, w = x.shape
    N = h * w
    logits, value, belief = torch.empty(B, N * 8 + 1), torch.empty(B), torch.empty(B, N)
    W = net.inp.out_features
    H = net.body[0].h if hasattr(net.body, "__getitem__") and hasattr(net.body[0], "h") else net.body.layers[0].self_attn.num_heads
    if EMIT:  # compile-time-shape build for exactly this config (emit.py); falls through to runtime-shape C
        spec = _emit.build(_emit.EmitCfg(tuple(kinds), W, H, T, h, w, CH, block, topk))
        if spec is not None:
            spec.net_forward_spec(B, x.data_ptr(), ctypes.cast(arr, ctypes.c_void_p), logits.data_ptr(), value.data_ptr(), belief.data_ptr())
            return logits, value, belief
    r = lib.net_forward(B, T, N, CH, W, H, len(kinds), kinds_c, block, topk, x.data_ptr(), ctypes.cast(arr, ctypes.c_void_p),
                        logits.data_ptr(), value.data_ptr(), belief.data_ptr())
    return (logits, value, belief) if r == 0 else None
