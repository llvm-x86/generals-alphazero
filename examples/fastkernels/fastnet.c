/* Whole-net forward, runtime-shape version: layer loop + weight walk over the shared pieces. emit.py emits the compile-time-shape twin. */
#include "pieces.c"

#define lin(...) TIMED(0, lin_(__VA_ARGS__))
#define layernorm(...) TIMED(1, layernorm_(__VA_ARGS__))
#define pisa_attn(...) TIMED(2, pisa_attn_(__VA_ARGS__))
#define dense_attn(...) TIMED(3, dense_attn_(__VA_ARGS__))
#define gdn_scan(...) TIMED(4, gdn_scan_(__VA_ARGS__))

/* Layer types in `kinds`: 0 dense (relu MLP), 1 pisa, 2 gdn. Weight pointers (consumed in order, Linear = W^T then bias):
 * global: inpWt inpb pos(N*W) tpos(T*W) glob(W) normw normb polWt polb passW(W) passb belW(W) belb val1Wt val1b val2W(32) val2b
 * dense: n1w n1b qkvWt qkvb projWt projb n2w n2b fc1Wt fc1b fc2Wt fc2b
 * pisa : same as dense
 * gdn  : n1w n1b qkvgWt(W,6W: q|k|v|alpha|erase|write) qkvgb log_a delta onormw onormb projWt projb n2w n2b fc1Wt fc1b fc2Wt fc2b */
int net_forward(int B, int T, int N, int CH, int W, int H, int L, const int *kinds, int C, int K,
                const float *x, const float *const *wp_in, float *logits, float *value, float *belief) {
    int n = 1 + T * N, TN = T * N, d = W / H;
    if (d > 64 || W > 1024 || K > 16 || C > 64) return -1;
    int dense = 0;
    for (int l = 0; l < L; l++) dense |= kinds[l] == 0;
    float *X, *Y, *QKV, *O, *Hm, *Xin, *Sc = 0, *Kt = 0, *Pyr_ = 0;
    int P = 1;
    for (int nb = (TN + C - 1) / C; P < nb; P <<= 1) ;
    for (int pass = 0; pass < 2; pass++) {
        measuring = pass == 0; cur = 0;
        X = ws((size_t)n * W); Y = ws((size_t)n * W); QKV = ws((size_t)n * 6 * W); O = ws((size_t)n * W);
        Hm = ws((size_t)n * 4 * W); Xin = ws((size_t)TN * CH);
        if (dense) { Sc = ws((size_t)n * n); Kt = ws((size_t)d * n); }
        Pyr_ = ws((size_t)2 * P * H * d + 64 + 4 * P + 64);
        if (pass == 0 && cap < cur) { free(arena); arena = aligned_alloc(64, cur * sizeof(float)); cap = cur; }
    }
    for (int b = 0; b < B; b++) {
        const float *const *wp = wp_in;
        const float *inpWt = *wp++, *inpb = *wp++, *pos = *wp++, *tpos = *wp++, *glob = *wp++, *normw = *wp++, *normb = *wp++;
        const float *polWt = *wp++, *polb = *wp++, *passW = *wp++, *passb = *wp++, *belW = *wp++, *belb = *wp++;
        const float *v1Wt = *wp++, *v1b = *wp++, *v2W = *wp++, *v2b = *wp++;
        const float *xb = x + (size_t)b * T * CH * N;
        for (int t = 0; t < T; t++)
            for (int c = 0; c < N; c++)
                for (int ch = 0; ch < CH; ch++) Xin[(size_t)(t * N + c) * CH + ch] = xb[((size_t)t * CH + ch) * N + c];
        memcpy(X, glob, sizeof(float) * W);
        lin(TN, W, CH, Xin, CH, inpWt, inpb, X + W, ACT_NONE, NULL);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < N; c++) {
                float *xr = X + (size_t)(1 + t * N + c) * W;
                for (int j = 0; j < W; j++) xr[j] += pos[c * W + j] + tpos[t * W + j];
            }
        for (int l = 0; l < L; l++) {
            if (kinds[l] == 2) {
                const float *n1w = *wp++, *n1b = *wp++, *qkvgWt = *wp++, *qkvgb = *wp++, *log_a = *wp++, *delta = *wp++;
                const float *onw = *wp++, *onb = *wp++, *pWt = *wp++, *pb = *wp++;
                layernorm(TN, W, X + W, n1w, n1b, Y);
                lin(TN, 6 * W, W, Y, W, qkvgWt, qkvgb, QKV, ACT_NONE, NULL);
                gdn_scan(T, N, W, H, log_a, delta, QKV, O);
                layernorm(TN, W, O, onw, onb, Y);
                lin(TN, W, W, Y, W, pWt, pb, O, ACT_NONE, X + W);
            } else {
                const float *n1w = *wp++, *n1b = *wp++, *qWt = *wp++, *qb = *wp++, *pWt = *wp++, *pb = *wp++;
                layernorm(n, W, X, n1w, n1b, Y);
                lin(n, 3 * W, W, Y, W, qWt, qb, QKV, ACT_NONE, NULL);
                if (kinds[l] == 1) pisa_attn(n, W, H, C, K, QKV, O, Pyr_);
                else dense_attn(n, W, H, QKV, O, Sc, Kt);
                lin(n, W, W, O, W, pWt, pb, Y, ACT_NONE, X);
            }
            const float *n2w = *wp++, *n2b = *wp++, *f1Wt = *wp++, *f1b = *wp++, *f2Wt = *wp++, *f2b = *wp++;
            layernorm(n, W, X, n2w, n2b, Y);
            lin(n, 4 * W, W, Y, W, f1Wt, f1b, Hm, kinds[l] == 0 ? ACT_RELU : ACT_GELU, NULL);
            lin(n, W, 4 * W, Hm, 4 * W, f2Wt, f2b, Y, ACT_NONE, X);
        }
        layernorm(n, W, X, normw, normb, Y);
        float *lg = logits + (size_t)b * (N * 8 + 1);
        const float *cur_ = Y + (size_t)(n - N) * W;
        sgemm_direct2(N, 8, W, cur_, W, polWt, 8, lg, 8);
        for (int i = 0; i < N * 8; i++) lg[i] += polb[i & 7];
        lg[N * 8] = dot(Y, passW, W) + passb[0];
        float h1[256];
        for (int j = 0; j < 32; j++) h1[j] = v1b[j];
        for (int kk = 0; kk < W; kk++) for (int j = 0; j < 32; j++) h1[j] += Y[kk] * v1Wt[kk * 32 + j];
        float vv = v2b[0];
        for (int j = 0; j < 32; j++) vv += (h1[j] > 0.f ? h1[j] : 0.f) * v2W[j];
        value[b] = tanhf(vv);
        const float *fr = xb + (size_t)(T - 1) * CH * N;
        for (int c = 0; c < N; c++) {
            int keep = fr[6 * N + c] < .5f && fr[15 * N + c] < .5f && fr[17 * N + c] < .5f && fr[18 * N + c] < .5f && fr[5 * N + c] < .5f;
            belief[(size_t)b * N + c] = keep ? dot(cur_ + (size_t)c * W, belW, W) + belb[0] : -1e9f;
        }
    }
    return 0;
}
