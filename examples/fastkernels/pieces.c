/* Whole-net inference forward (dense | pisa | hybrid) of examples/az_selfplay.py Net as ONE C call.
 * Style/contract follows the author's aurora uopc program (lib/aurora/uopc/gpt.py build_c): a static
 * arena sized once, weights read through caller-owned pointers (no copies), aurora's sgemm_direct2 for
 * every GEMM, aurora's AVX-512 GELU/exp. Linear layers take W^T row-major (K,N); the Python side keeps
 * those transposes in sync with the parameters. Inference only. Compiled together with d2_nr16_u2.c.
 *
 * NOT fused (honest list): bias/GELU/residual are one extra pass over each GEMM output (aurora's store
 * epilogues have no bias kind); GDN-2 scan and PISA selection/attention are scalar-over-tokens with AVX-512
 * over the head dim; no threading. */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "d2_nr16_u2.c"

static float *arena;
static size_t cap, cur;
static int measuring;
static float *ws(size_t n) {
    n = (n + 15) & ~(size_t)15;
    float *p = measuring ? NULL : arena + cur;
    cur += n;
    return p;
}

#include <time.h>
/* per-stage seconds since last reset: 0 gemm+bias/act, 1 layernorm, 2 pisa select+attn, 3 dense attn, 4 gdn scan */
double fk_prof[8];
#ifdef FK_NOPROF  /* emitted build: no clocks in the hot path */
static inline double now(void) { return 0.0; }
#else
static inline double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
#endif
#define TIMED(i, stmt) do { double t0_ = now(); stmt; fk_prof[i] += now() - t0_; } while (0)

enum { ACT_NONE, ACT_GELU, ACT_RELU };

/* C = act(A@Wt + bias); if res, res += A@Wt + bias instead (C unused scratch). Wt is (K,N) row-major. */
static void lin_(int M, int N, int K, const float *A, int lda, const float *Wt, const float *bias, float *C, int act, float *res) {
    sgemm_direct2(M, N, K, A, lda, Wt, N, C, N);
    for (int i = 0; i < M; i++) {
        float *c = C + (size_t)i * N, *r = res ? res + (size_t)i * N : NULL;
        int j = 0;
        for (; j + 16 <= N; j += 16) {
            __m512 v = _mm512_add_ps(_mm512_loadu_ps(c + j), _mm512_loadu_ps(bias + j));
            if (act == ACT_GELU) { __m512 o, g; uopc_gelu_fwd512(v, &o, &g); v = o; }
            else if (act == ACT_RELU) v = _mm512_max_ps(v, _mm512_setzero_ps());
            if (r) _mm512_storeu_ps(r + j, _mm512_add_ps(_mm512_loadu_ps(r + j), v));
            else _mm512_storeu_ps(c + j, v);
        }
        for (; j < N; j++) {
            float v = c[j] + bias[j];
            if (act == ACT_GELU) v = 0.5f * v * (1.f + erff(v * 0.70710678f));
            else if (act == ACT_RELU) v = v > 0.f ? v : 0.f;
            if (r) r[j] += v; else c[j] = v;
        }
    }
}

static void layernorm_(int M, int W, const float *x, const float *w, const float *b, float *y) {
    for (int i = 0; i < M; i++) {
        const float *xi = x + (size_t)i * W;
        float *yi = y + (size_t)i * W, mu = 0.f, var = 0.f;
        for (int j = 0; j < W; j++) mu += xi[j];
        mu /= W;
        for (int j = 0; j < W; j++) { float t = xi[j] - mu; var += t * t; }
        float inv = 1.f / sqrtf(var / W + 1e-5f);
        for (int j = 0; j < W; j++) yi[j] = (xi[j] - mu) * inv * w[j] + b[j];
    }
}

static inline float dot(const float *a, const float *b, int d) {
    __m512 s = _mm512_setzero_ps();
    int j = 0;
    for (; j + 16 <= d; j += 16) s = _mm512_fmadd_ps(_mm512_loadu_ps(a + j), _mm512_loadu_ps(b + j), s);
    float r = _mm512_reduce_add_ps(s);
    for (; j < d; j++) r += a[j] * b[j];
    return r;
}

/* o += p * v (length d) */
static inline void axpy(float *o, float p, const float *v, int d) {
    __m512 pv = _mm512_set1_ps(p);
    int j = 0;
    for (; j + 16 <= d; j += 16) _mm512_storeu_ps(o + j, _mm512_fmadd_ps(pv, _mm512_loadu_ps(v + j), _mm512_loadu_ps(o + j)));
    for (; j < d; j++) o[j] += p * v[j];
}

/* ---- dense attention: all tokens, per head, via two GEMMs; qkv is (n,3W) with q|k|v column blocks ---- */
static void dense_attn_(int n, int W, int H, const float *qkv, float *O, float *sc, float *kt) {
    int d = W / H;
    float scale = 1.f / sqrtf((float)d);
    for (int h = 0; h < H; h++) {
        const float *q = qkv + h * d, *k = qkv + W + h * d, *v = qkv + 2 * W + h * d;
        for (int t = 0; t < n; t++)
            for (int j = 0; j < d; j++) kt[(size_t)j * n + t] = k[(size_t)t * 3 * W + j];
        sgemm_direct2(n, n, d, q, 3 * W, kt, n, sc, n);
        for (int i = 0; i < n; i++) {
            float *r = sc + (size_t)i * n, m = -INFINITY, z = 0.f;
            for (int j = 0; j < n; j++) { r[j] *= scale; if (r[j] > m) m = r[j]; }
            __m512 mv = _mm512_set1_ps(m), zv = _mm512_setzero_ps();
            int j = 0;
            for (; j + 16 <= n; j += 16) {
                __m512 e = uopc_expf_fast512(_mm512_sub_ps(_mm512_loadu_ps(r + j), mv));
                _mm512_storeu_ps(r + j, e); zv = _mm512_add_ps(zv, e);
            }
            z = _mm512_reduce_add_ps(zv);
            for (; j < n; j++) { r[j] = expf(r[j] - m); z += r[j]; }
            float iz = 1.f / z;
            for (j = 0; j < n; j++) r[j] *= iz;
        }
        sgemm_direct2(n, d, n, sc, n, v, 3 * W, O + h * d, W);
    }
}

/* ---- PISA: pyramid select (as PisaLayer.select) + fused online-softmax attention (no gather) ---- */
typedef struct { int Nk, C, K, H, d, W, Lmax; float *mean[24]; int *cnt[24]; const float *kbase; } Pyr;

/* sum over heads of LSE over rows g0..g0+G-1 of (level>=1: means, valid cnt>0; level 0: leaf tokens, valid r<Nk) */
static float group_score(const Pyr *p, int level, int64_t g0, int G, const float *qs /* H*d, pre-scaled */) {
    float tot = 0.f;
    for (int h = 0; h < p->H; h++) {
        float lg[64], m = -INFINITY;
        for (int g = 0; g < G; g++) {
            int64_t r = g0 + g;
            const float *x;
            if (level == 0) { if (r >= p->Nk) { lg[g] = -INFINITY; continue; } x = p->kbase + (size_t)r * 3 * p->W + h * p->d; }
            else { if (p->cnt[level][r] <= 0) { lg[g] = -INFINITY; continue; } x = p->mean[level] + (r * p->H + h) * p->d; }
            lg[g] = dot(qs + h * p->d, x, p->d);
            if (lg[g] > m) m = lg[g];
        }
        if (m == -INFINITY) return -INFINITY;
        float z = 0.f;
        if (G <= 16) {  /* one masked vector exp; -inf lanes give exp(-inf)=0 */
            __m512 e = uopc_expf_fast512(_mm512_sub_ps(_mm512_maskz_loadu_ps((__mmask16)((1u << G) - 1), lg), _mm512_set1_ps(m)));
            z = _mm512_mask_reduce_add_ps((__mmask16)((1u << G) - 1), e);
        } else for (int g = 0; g < G; g++) z += expf(lg[g] - m);
        tot += m + logf(z);
    }
    return tot;
}

static int topk_inplace(int *A, float *sc, int na, int K) { /* keep K best (stable on ties), returns K */
    for (int i = 0; i < K; i++) {
        int b = i;
        for (int j = i + 1; j < na; j++) if (sc[j] > sc[b]) b = j;
        float ts = sc[i]; sc[i] = sc[b]; sc[b] = ts;
        int ta = A[i]; A[i] = A[b]; A[b] = ta;
    }
    return K;
}

static void pisa_attn_(int n, int W, int H, int C, int K, const float *qkv, float *O, float *pyr_buf) {
    int d = W / H, Nk = n - 1;
    float scale = 1.f / sqrtf((float)d);
    double tp0_ = now();
    int nb = (Nk + C - 1) / C, P = 1;
    while (P < nb) P <<= 1;
    Pyr p = { Nk, C, K, H, d, W, 0, {0}, {0}, qkv + 3 * W + W };
    p.kbase = qkv + 3 * W + W;  /* key of cell token 0 (token 1), q at +0, k at +W within a row */
    int Lmax = 1;
    for (int t = P; t > 1; t >>= 1) Lmax++;
    p.Lmax = Lmax;
    /* levels 1..Lmax: sums -> means */
    float *fb = pyr_buf;
    int *ib = (int *)(pyr_buf + (size_t)2 * P * H * d + 64);
    for (int l = 1, R = P; l <= Lmax; l++, R >>= 1) {
        p.mean[l] = fb; fb += (size_t)R * H * d;
        p.cnt[l] = ib; ib += R;
    }
    for (int r = 0; r < P; r++) {
        int c = 0;
        float *m = p.mean[1] + (size_t)r * H * d;
        memset(m, 0, sizeof(float) * H * d);
        for (int g = 0; g < C; g++) {
            int t = r * C + g;
            if (t >= Nk) break;
            c++;
            for (int h = 0; h < H; h++) axpy(m + h * d, 1.f, p.kbase + (size_t)t * 3 * W + h * d, d);
        }
        p.cnt[1][r] = c;
    }
    for (int l = 2, R = P / 2; l <= Lmax; l++, R >>= 1)
        for (int r = 0; r < R; r++) {
            float *m = p.mean[l] + (size_t)r * H * d, *a = p.mean[l - 1] + (size_t)2 * r * H * d, *b = a + (size_t)H * d;
            for (int j = 0; j < H * d; j++) m[j] = a[j] + b[j];
            p.cnt[l][r] = p.cnt[l - 1][2 * r] + p.cnt[l - 1][2 * r + 1];
        }
    for (int l = 1, R = P; l <= Lmax; l++, R >>= 1)
        for (int r = 0; r < R; r++) {
            float s = 1.f / (p.cnt[l][r] > 1 ? p.cnt[l][r] : 1);
            for (int j = 0; j < H * d; j++) p.mean[l][(size_t)r * H * d + j] *= s;
        }

    /* global token (row 0) attends densely to all n tokens */
    for (int h = 0; h < H; h++) {
        const float *q = qkv + h * d;
        float m = -INFINITY, l = 0.f, acc[256];
        memset(acc, 0, sizeof(float) * d);
        for (int t = 0; t < n; t++) {
            float sc = dot(q, qkv + (size_t)t * 3 * W + W + h * d, d) * scale;
            if (sc > m) { float r = expf(m - sc); for (int j = 0; j < d; j++) acc[j] *= r; l *= r; m = sc; }
            float pw = expf(sc - m);
            axpy(acc, pw, qkv + (size_t)t * 3 * W + 2 * W + h * d, d);
            l += pw;
        }
        for (int j = 0; j < d; j++) O[h * d + j] = acc[j] / l;
    }
    fk_prof[6] += now() - tp0_;
    /* cell tokens */
    for (int i = 0; i < Nk; i++) {
        const float *qrow = qkv + (size_t)(1 + i) * 3 * W;
        float qs[1024];
        for (int j = 0; j < W; j++) qs[j] = qrow[j] * scale;
        int A[40], na = 1;
        double ts0_ = now();
        float sc[40];
        A[0] = 0;
        for (int lvl = Lmax; lvl >= 2; lvl--) {
            if (na > K) {
                for (int a = 0; a < na; a++) sc[a] = group_score(&p, lvl - 1, (int64_t)A[a] * 2, 2, qs);
                na = topk_inplace(A, sc, na, K);
            }
            for (int a = na - 1; a >= 0; a--) { A[2 * a + 1] = 2 * A[a] + 1; A[2 * a] = 2 * A[a]; }
            na *= 2;
        }
        if (na > K) {
            for (int a = 0; a < na; a++) sc[a] = group_score(&p, 0, (int64_t)A[a] * C, C, qs);
            na = topk_inplace(A, sc, na, K);
        }
        fk_prof[5] += now() - ts0_;
        float *out = O + (size_t)(1 + i) * W;
        for (int h = 0; h < H; h++) {
            const float *q = qrow + h * d;
            float m = -INFINITY, l = 0.f, acc[256];
            memset(acc, 0, sizeof(float) * d);
            for (int s = -1; s < na; s++) {
                int t0 = 0, t1 = 1;
                if (s >= 0) { t0 = A[s] * C; t1 = t0 + C; if (t1 > Nk) t1 = Nk; t0 += 1; t1 += 1; }
                int nt = t1 - t0;
                if (nt <= 0) continue;
                float sb[64] __attribute__((aligned(64))), bm = -INFINITY;
                for (int t = 0; t < nt; t++) {  /* scores of the block, one rescale per block, vector exp */
                    sb[t] = dot(q, qkv + (size_t)(t0 + t) * 3 * W + W + h * d, d) * scale;
                    if (sb[t] > bm) bm = sb[t];
                }
                if (bm > m) { float r = expf(m - bm); for (int j = 0; j < d; j++) acc[j] *= r; l *= r; m = bm; }
                __m512 mv = _mm512_set1_ps(m);
                for (int t = 0; t < nt; t += 16)
                    _mm512_store_ps(sb + t, uopc_expf_fast512(_mm512_sub_ps(_mm512_load_ps(sb + t), mv)));
                for (int t = 0; t < nt; t++) {
                    axpy(acc, sb[t], qkv + (size_t)(t0 + t) * 3 * W + 2 * W + h * d, d);
                    l += sb[t];
                }
            }
            for (int j = 0; j < d; j++) out[h * d + j] = acc[j] / l;
        }
    }
}

/* ---- GDN-2: per cell and head a sequential scan over T frames; rows of qkvg are t*N+cell, cols q|k|v|alpha|erase|write ---- */
static void gdn_scan_(int T, int N, int W, int H, const float *log_a, const float *delta, const float *g, float *O) {
    int d = W / H;
    float S[64 * 64], q[64], k[64], a[64], bb[64], ww[64], r[64];
    for (int c = 0; c < N; c++)
        for (int h = 0; h < H; h++) {
            memset(S, 0, sizeof(float) * d * d);
            for (int t = 0; t < T; t++) {
                const float *row = g + (size_t)(t * N + c) * 6 * W + h * d;
                float nq = 1e-24f, nk = 1e-24f;
                for (int j = 0; j < d; j++) { nq += row[j] * row[j]; nk += row[W + j] * row[W + j]; }
                nq = 1.f / fmaxf(sqrtf(nq), 1e-12f); nk = 1.f / fmaxf(sqrtf(nk), 1e-12f);
                for (int j = 0; j < d; j++) {
                    q[j] = row[j] * nq; k[j] = row[W + j] * nk;
                    float x = row[3 * W + j] + delta[h * d + j];
                    float sp = x > 20.f ? x : log1pf(expf(x));
                    a[j] = expf(-expf(log_a[h * d + j]) * sp);
                    bb[j] = 1.f / (1.f + expf(-row[4 * W + j]));
                    ww[j] = 1.f / (1.f + expf(-row[5 * W + j]));
                }
                const float *v = row + 2 * W;
                memset(r, 0, sizeof(float) * d);
                for (int kk = 0; kk < d; kk++) {
                    float *Sk = S + kk * d, ak = a[kk], bk = bb[kk] * k[kk];
                    for (int j = 0; j < d; j++) { Sk[j] *= ak; r[j] += Sk[j] * bk; }
                }
                float *o = O + (size_t)(t * N + c) * W + h * d;
                memset(o, 0, sizeof(float) * d);
                for (int kk = 0; kk < d; kk++) {
                    float *Sk = S + kk * d, kv = k[kk], qk = q[kk];
                    for (int j = 0; j < d; j++) { Sk[j] += kv * (ww[j] * v[j] - r[j]); o[j] += Sk[j] * qk; }
                }
            }
        }
}
