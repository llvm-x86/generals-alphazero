/* Fused PISA sparse attention, AVX-512, same build as d2_nr16_u2.c (author's own style).
 * For each (b,h,n): online-softmax attention over the S selected blocks of C key/value
 * tokens (tokens >= Nk are padding and skipped) plus one global key/value. No gather is
 * materialised. q/k/v are (B,H,*,d) with arbitrary b/h/token strides and unit d stride;
 * sel is (B,N,S) int64; out is contiguous (B,H,N,d); kg/vg are (B,H,d) strided (b,h). */
#include <immintrin.h>
#include <stdint.h>
#include <math.h>

#define PA_DMAX 128

int pisa_sparse_attn(int B, int H, int N, int d, int Nk, int C, int S, float scale,
                     const float *q, const long *qs, const float *k, const float *v, const long *kvs,
                     const float *kg, const float *vg, const long *gs,
                     const int64_t *sel, float *out) {
    if (d > PA_DMAX) return -1;
    int nv = (d + 15) / 16;
    __mmask16 mk[PA_DMAX / 16];
    for (int i = 0; i < nv; i++) {
        int r = d - 16 * i;
        mk[i] = r >= 16 ? 0xFFFF : (__mmask16)((1u << r) - 1);
    }
    for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++) {
        const float *kb = k + b * kvs[0] + h * kvs[1], *vb = v + b * kvs[0] + h * kvs[1];
        for (int n = 0; n < N; n++) {
            const float *qp = q + b * qs[0] + h * qs[1] + n * qs[2];
            __m512 qv[PA_DMAX / 16], acc[PA_DMAX / 16];
            for (int i = 0; i < nv; i++) { qv[i] = _mm512_maskz_loadu_ps(mk[i], qp + 16 * i); acc[i] = _mm512_setzero_ps(); }
            float m = -INFINITY, l = 0.f;
            const int64_t *sp = sel + ((int64_t)b * N + n) * S;
            for (int s = -1; s < S; s++) {
                int t0, t1;
                if (s < 0) t0 = 0, t1 = 1;
                else { t0 = (int)sp[s] * C; t1 = t0 + C; if (t1 > Nk) t1 = Nk; }
                for (int t = t0; t < t1; t++) {
                    const float *kp, *vp;
                    if (s < 0) { kp = kg + b * gs[0] + h * gs[1]; vp = vg + b * gs[0] + h * gs[1]; }
                    else { kp = kb + (int64_t)t * kvs[2]; vp = vb + (int64_t)t * kvs[2]; }
                    __m512 a = _mm512_setzero_ps();
                    for (int i = 0; i < nv; i++)
                        a = _mm512_fmadd_ps(qv[i], _mm512_maskz_loadu_ps(mk[i], kp + 16 * i), a);
                    float sc = _mm512_reduce_add_ps(a) * scale;
                    float p;
                    if (sc > m) {
                        float r = expf(m - sc);  /* m=-inf -> 0 */
                        __m512 rv = _mm512_set1_ps(r);
                        for (int i = 0; i < nv; i++) acc[i] = _mm512_mul_ps(acc[i], rv);
                        l *= r; m = sc; p = 1.f;
                    } else p = expf(sc - m);
                    __m512 pv = _mm512_set1_ps(p);
                    for (int i = 0; i < nv; i++)
                        acc[i] = _mm512_fmadd_ps(pv, _mm512_maskz_loadu_ps(mk[i], vp + 16 * i), acc[i]);
                    l += p;
                }
            }
            float *op = out + (((int64_t)b * H + h) * N + n) * d;
            __m512 il = _mm512_set1_ps(1.f / l);
            for (int i = 0; i < nv; i++) _mm512_mask_storeu_ps(op + 16 * i, mk[i], _mm512_mul_ps(acc[i], il));
        }
    }
    return 0;
}

/* PisaLayer.select's scoring step without the gather: out[b,n,a] = sum_h logsumexp_{g<G, row valid}
 * q[b,h,n].X[b,h,A[b,n,a]*G+g]. q (B,H,N,d) and X (B,H,R,d) are contiguous, valid is (R) bytes
 * (1 = row participates), A is (B,N,na) int64. A fully invalid group scores -inf (as in PyTorch). */
int pisa_group_lse(int B, int H, int N, int d, int R, int G, int na, const float *q, const float *X,
                   const unsigned char *valid, const int64_t *A, float *out) {
    if (d > PA_DMAX || G > 64) return -1;
    int nv = (d + 15) / 16;
    __mmask16 mk[PA_DMAX / 16];
    for (int i = 0; i < nv; i++) { int r = d - 16 * i; mk[i] = r >= 16 ? 0xFFFF : (__mmask16)((1u << r) - 1); }
    for (int b = 0; b < B; b++)
    for (int n = 0; n < N; n++)
    for (int a = 0; a < na; a++) {
        int64_t g0 = A[((int64_t)b * N + n) * na + a] * G;
        float tot = 0.f;
        for (int h = 0; h < H; h++) {
            const float *qp = q + (((int64_t)b * H + h) * N + n) * d;
            const float *xb = X + ((int64_t)b * H + h) * R * d;
            __m512 qv[PA_DMAX / 16];
            for (int i = 0; i < nv; i++) qv[i] = _mm512_maskz_loadu_ps(mk[i], qp + 16 * i);
            float lg[64], m = -INFINITY;
            for (int g = 0; g < G; g++) {
                if (!valid[g0 + g]) { lg[g] = -INFINITY; continue; }
                const float *xp = xb + (g0 + g) * d;
                __m512 s = _mm512_setzero_ps();
                for (int i = 0; i < nv; i++) s = _mm512_fmadd_ps(qv[i], _mm512_maskz_loadu_ps(mk[i], xp + 16 * i), s);
                lg[g] = _mm512_reduce_add_ps(s);
                if (lg[g] > m) m = lg[g];
            }
            if (m == -INFINITY) { tot = -INFINITY; continue; }
            float z = 0.f;
            for (int g = 0; g < G; g++) z += expf(lg[g] - m);
            tot += m + logf(z);
        }
        out[((int64_t)b * N + n) * na + a] = tot;
    }
    return 0;
}
