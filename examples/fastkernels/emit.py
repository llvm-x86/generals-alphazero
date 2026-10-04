"""Config-driven emitter: specialised C for the WHOLE net forward with compile-time shapes.

Why source strings and not aurora's Graph/uopcc (checked): aurora's Graph is a static-shape tensor graph for
*training* (gemm/ew/reduce/norm/gather; weights are graph buffers baked into an arena; it compiles via a
subprocess + numpy feeds). It could express the dense layer (matmul, norm, relu/gelu, softmax, residual), but
not PISA's top-k pyramid selection nor the GDN-2 sequential scan, takes weights by value rather than by
caller pointer (our weights change every training step), and its R5 epilogue fusion only knows ADD / GELU-bwd /
FMA, no bias. So this module emits C strings, in the style of aurora's `uopc/gpt.py build_c`.

What is GENERATED per config (cfg = EmitCfg): the layer sequence unrolled with every shape, weight offset and
layer kind a literal; one GEMM function per distinct (M,N,K,act,residual) whose micro-kernel is the 8x16
tile of aurora's d2_nr16_u2.c `mic1` (copied in structure, attribution below) with a **bias + relu/gelu +
residual store epilogue**, constant K/N/M so loops are unrolled/folded; heads; arena sized at compile time.
What stays HAND C (pieces.c, #included as templates and constant-propagated by gcc once inlined): layernorm,
dense attention (two d2 sgemm_direct2 + softmax), PISA select/online-softmax attention, GDN-2 scan, the input
transpose, and the value/belief heads' tiny scalar loops.

GEMM tile adapted from aurora lib/aurora/uopc/kernels/d2_nr16_u2.c (mic1 / uopc_epi_store512 / uopc_gelu_fwd512),
author's own code; the generalisation (masked N tail, R<8 row tail, bias operand, act+residual fused) is new.
"""
import ctypes
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).parent
_DEPS = ["d2_nr16_u2.c", "pieces.c"]
_libs = {}  # cfg -> CDLL | None


@dataclass(frozen=True)
class EmitCfg:
    kinds: tuple  # per layer: 0 dense(relu MLP) 1 pisa 2 gdn
    W: int
    H: int
    T: int
    h: int
    w: int
    CH: int
    block: int
    topk: int

    @property
    def N(self):
        return self.h * self.w

    def supported(self):
        d = self.W // self.H
        return d <= 64 and self.W <= 1024 and self.topk <= 16 and self.block <= 64 and self.W % self.H == 0


_TILE = r"""
/* 8x16 register tile, structure of d2 mic1: R rows (compile-time) x 16 cols, K-loop of broadcast-FMA,
 * then the epilogue on the accumulators before the single store: v = acc + bias; act; += residual. */
#define A_NONE 0
#define A_RELU 1
#define A_GELU 2
static inline __attribute__((always_inline)) void tile_(int R, int K, const float *A, const float *B, int ldb,
        __mmask16 mk, const float *bias, int act, int res, float *C, int ldc) {
    __m512 c[8];
    for (int r = 0; r < 8; r++) c[r] = _mm512_setzero_ps();
    for (int p = 0; p < K; p++) {
        __m512 b = _mm512_maskz_loadu_ps(mk, B + (size_t)p * ldb);
        for (int r = 0; r < R; r++) c[r] = _mm512_fmadd_ps(_mm512_set1_ps(A[(size_t)r * K + p]), b, c[r]);
    }
    __m512 bv = _mm512_maskz_loadu_ps(mk, bias);
    for (int r = 0; r < R; r++) {
        __m512 v = _mm512_add_ps(c[r], bv);
        if (act == A_GELU) { __m512 o, g; uopc_gelu_fwd512(v, &o, &g); v = o; }
        else if (act == A_RELU) v = _mm512_max_ps(v, _mm512_setzero_ps());
        float *cp = C + (size_t)r * ldc;
        if (res) v = _mm512_add_ps(v, _mm512_maskz_loadu_ps(mk, cp));  /* C aliases the residual stream */
        _mm512_mask_storeu_ps(cp, mk, v);
    }
}
"""


def _gemm(M, N, K, act, res):
    """C(M,N) [+]= act(A(M,K) @ Wt(K,N) + bias); A and C are dense (lda=K, ldc=N)."""
    name = f"g_{M}_{N}_{K}_{act}_{res}"
    nf, rt = N & ~15, M & 7
    tail = (1 << (N - nf)) - 1
    a = f"A_{('NONE', 'RELU', 'GELU')[act]}"
    body = [f"static void {name}(const float *A, const float *B, const float *bias, float *C) {{"]
    # row-outer keeps the 8 x K A-panel in L1 while B streams (d2's rule when B fits in cache)
    for j0, j1, mk in [(0, nf, "0xFFFF"), (nf, N, f"0x{tail:X}")]:
        if j0 == j1:
            continue
        body.append(f"    for (int j = {j0}; j < {j1}; j += 16) {{")
        if M >= 8:
            body.append(f"        for (int i = 0; i + 8 <= {M}; i += 8)")
            body.append(f"            tile_(8, {K}, A + (size_t)i * {K}, B + j, {N}, {mk}, bias + j, {a}, {res}, C + (size_t)i * {N} + j, {N});")
        if rt:
            body.append(f"        tile_({rt}, {K}, A + (size_t){M - rt} * {K}, B + j, {N}, {mk}, bias + j, {a}, {res}, C + (size_t){M - rt} * {N} + j, {N});")
        body.append("    }")
    body.append("}")
    return name, "\n".join(body)


def emit(cfg):
    W, H, T, N, CH, C, K = cfg.W, cfg.H, cfg.T, cfg.N, cfg.CH, cfg.block, cfg.topk
    n, TN, d, L = 1 + T * N, T * N, cfg.W // cfg.H, len(cfg.kinds)
    dense = 0 in cfg.kinds
    P = 1
    while P < -(-TN // C):
        P <<= 1
    gemms = {}

    def g(M, Nn, Kk, act, res):
        nm, src = _gemm(M, Nn, Kk, act, res)
        gemms[nm] = src
        return nm

    f = []  # body of one batch element
    ln = lambda m, x, y, wn, bn: f"layernorm_({m}, {W}, {x}, {wn}, {bn}, {y});"  # noqa: E731
    f += [f"const float *xb = x + (size_t)b * {T * CH * N};",
          f"for (int t = 0; t < {T}; t++) for (int c = 0; c < {N}; c++) for (int ch = 0; ch < {CH}; ch++)"
          f" Xin[(t * {N} + c) * {CH} + ch] = xb[((size_t)t * {CH} + ch) * {N} + c];",
          f"memcpy(X, glob, sizeof(float) * {W});",
          f"{g(TN, W, CH, 0, 0)}(Xin, inpWt, inpb, X + {W});",  # ldc=W: X+W rows are contiguous W-wide
          f"for (int t = 0; t < {T}; t++) for (int c = 0; c < {N}; c++) {{ float *xr = X + (size_t)(1 + t * {N} + c) * {W};"
          f" for (int j = 0; j < {W}; j++) xr[j] += pos[c * {W} + j] + tpos[t * {W} + j]; }}"]
    for l, kind in enumerate(cfg.kinds):
        f.append(f"/* layer {l}: {('dense', 'pisa', 'gdn')[kind]} */ {{")
        if kind == 2:
            f += [f"const float *n1w = NX, *n1b = NX, *qkvgWt = NX, *qkvgb = NX, *log_a = NX, *delta = NX, *onw = NX, *onb = NX, *pWt = NX, *pb = NX;",
                  ln(TN, f"X + {W}", "Y", "n1w", "n1b"),
                  f"{g(TN, 6 * W, W, 0, 0)}(Y, qkvgWt, qkvgb, QKV);",
                  f"gdn_scan_({T}, {N}, {W}, {H}, log_a, delta, QKV, O);",
                  ln(TN, "O", "Y", "onw", "onb"),
                  f"{g(TN, W, W, 0, 1)}(Y, pWt, pb, X + {W});"]  # += into the residual stream
        else:
            f += ["const float *n1w = NX, *n1b = NX, *qWt = NX, *qb = NX, *pWt = NX, *pb = NX;",
                  ln(n, "X", "Y", "n1w", "n1b"),
                  f"{g(n, 3 * W, W, 0, 0)}(Y, qWt, qb, QKV);",
                  f"pisa_attn_({n}, {W}, {H}, {C}, {K}, QKV, O, Pyr_);" if kind == 1 else f"dense_attn_({n}, {W}, {H}, QKV, O, Sc, Kt);",
                  f"{g(n, W, W, 0, 1)}(O, pWt, pb, X);"]
        f += ["const float *n2w = NX, *n2b = NX, *f1Wt = NX, *f1b = NX, *f2Wt = NX, *f2b = NX;",
              ln(n, "X", "Y", "n2w", "n2b"),
              f"{g(n, 4 * W, W, 1 if kind == 0 else 2, 0)}(Y, f1Wt, f1b, Hm);",
              f"{g(n, W, 4 * W, 0, 1)}(Hm, f2Wt, f2b, X);", "}"]
    f += [ln(n, "X", "Y", "normw", "normb"),
          f"float *lg = logits + (size_t)b * {N * 8 + 1};",
          f"const float *cur_ = Y + (size_t){n - N} * {W};",
          f"{g(N, 8, W, 0, 0)}(cur_, polWt, polb, lg);",
          f"lg[{N * 8}] = dot(Y, passW, {W}) + passb[0];",
          "float h1[32]; for (int j = 0; j < 32; j++) h1[j] = v1b[j];",
          f"for (int kk = 0; kk < {W}; kk++) for (int j = 0; j < 32; j++) h1[j] += Y[kk] * v1Wt[kk * 32 + j];",
          "float vv = v2b[0]; for (int j = 0; j < 32; j++) vv += (h1[j] > 0.f ? h1[j] : 0.f) * v2W[j];",
          "value[b] = tanhf(vv);",
          f"const float *fr = xb + (size_t){T - 1} * {CH * N};",
          f"for (int c = 0; c < {N}; c++) {{ int keep = fr[6 * {N} + c] < .5f && fr[15 * {N} + c] < .5f && fr[17 * {N} + c] < .5f"
          f" && fr[18 * {N} + c] < .5f && fr[5 * {N} + c] < .5f;"
          f" belief[(size_t)b * {N} + c] = keep ? dot(cur_ + (size_t)c * {W}, belW, {W}) + belb[0] : -1e9f; }}"]

    sizes = [n * W, n * W, n * 6 * W, n * W, n * 4 * W, TN * CH] + ([n * n, d * n] if dense else []) + [2 * P * H * d + 64 + 4 * P + 64]
    offs, tot = [], 0
    for s in sizes:
        offs.append(tot)
        tot += (s + 15) & ~15
    names = ["X", "Y", "QKV", "O", "Hm", "Xin"] + (["Sc", "Kt"] if dense else []) + ["Pyr_"]
    arena = "\n".join(f"float *const {nm} = arena_ + {o};" for nm, o in zip(names, offs))
    pre = ("const float *inpWt = *wp++, *inpb = *wp++, *pos = *wp++, *tpos = *wp++, *glob = *wp++, *normw = *wp++, *normb = *wp++,"
           " *polWt = *wp++, *polb = *wp++, *passW = *wp++, *passb = *wp++, *belW = *wp++, *belb = *wp++,"
           " *v1Wt = *wp++, *v1b = *wp++, *v2W = *wp++, *v2b = *wp++;")
    return "\n".join([
        f"/* emitted by examples/fastkernels/emit.py for {cfg} */", "#define FK_NOPROF 1", '#include "pieces.c"', _TILE,
        *gemms.values(),
        f"static float arena_[{tot}] __attribute__((aligned(64)));", "#define NX (*wp++)",
        "int net_forward_spec(int B, const float *x, const float *const *wp_in, float *logits, float *value, float *belief) {",
        arena, "for (int b = 0; b < B; b++) {", "const float *const *wp = wp_in;", pre, *f, "}", "return 0;", "}", ""])


def build(cfg):
    """CDLL exposing net_forward_spec for cfg (compiled once, cached by hash in .cache/), or None."""
    if cfg in _libs:
        return _libs[cfg]
    lib = None
    try:
        if cfg.supported():
            src = emit(cfg)
            tag = hashlib.sha1((src + "".join((_HERE / f).read_text() for f in _DEPS) + Path(__file__).read_text()).encode()).hexdigest()[:12]
            so = _HERE / ".cache" / f"emit_{tag}.so"
            if not so.exists():
                so.parent.mkdir(exist_ok=True)
                c = so.with_suffix(".c")
                c.write_text(src)
                tmp = so.with_suffix(f".{id(cfg)}.tmp")
                r = subprocess.run(["gcc", "-O3", "-march=native", "-mprefer-vector-width=512", "-funroll-loops", "-fPIC", "-shared",
                                f"-I{_HERE}", str(c), "-o", str(tmp), "-lm"], capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError(r.stderr[-1500:])
                tmp.replace(so)
            lib = ctypes.CDLL(str(so))
            P = ctypes.c_void_p
            lib.net_forward_spec.argtypes = [ctypes.c_int, P, P, P, P, P]
            lib.net_forward_spec.restype = ctypes.c_int
    except Exception as e:  # noqa: BLE001 - any failure degrades to the runtime-shape fastnet.c
        print(f"fastkernels emit unavailable ({e}); using runtime-shape C")
    _libs[cfg] = lib
    return lib
