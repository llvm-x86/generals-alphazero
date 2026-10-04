/* Adapted from the author's own aurora library, lib/aurora/uopc/kernels/d2_nr16_u2.c
 * (copied unmodified below this header; only the sgemm_direct2 path is used here --
 * the packed kernel referenced elsewhere in aurora is not part of this copy). */
/* Direct SGEMM v3 (generated): max_nr=16, k-unroll=2 */

/* ===========================================================================
 * PERFORMANCE CONTRACT -- read before adding a kernel, activation, layernorm
 * or epilogue kind, or before "optimising" anything in this file.
 *
 * This library is the measured floor under lib/aurora/uopc/gpt.py.  Every
 * number here was taken on pinned, quiet cores with paired interleaved A/B.
 * A change that does not beat the baseline is a REGRESSION even when the
 * benchmark prints a bigger number.
 *
 * ---------------------------------------------------------------------------
 * 0. CURRENT BASELINE (GPT d=64 L=2 H=2 B=24 T=256 V=65, fwd+bwd+AdamW)
 *      1 thread : uopc ~96-106 ms/step vs numpy+OpenBLAS ~172-194  -> 1.78-1.82x
 *      3 threads: uopc ~44 ms/step                                 -> 3.42x
 *      gate     : max grad rel err 1.474e-06 (L2 relative norm, tolerance 1e-4)
 *    Reproduce:  PYTHONPATH=lib GPT_BENCH_THREADS=1 OMP_NUM_THREADS=1 \
 *                OPENBLAS_NUM_THREADS=1 taskset -c <core> python -m aurora.uopc.gpt
 *    Absolute ms drift 20%+ with machine load; only RATIOS and PAIRED deltas
 *    taken in the same session are comparable.  Never compare against a number
 *    from an earlier session.
 *
 * ---------------------------------------------------------------------------
 * 1. HOW TO MEASURE, or you will land a regression and believe it is a win
 *    a) A single run is NOT a measurement.  Use medians of >= 5.
 *    b) Interleave arms ROUND-ROBIN inside one process (A,B,A,B,...).  Running
 *       A seven times then B seven times gave the OPPOSITE ranking here,
 *       because each arm gets its own warm cache.  Across processes the same
 *       binary ranged 1.79-3.03 ms on an identical shape.
 *    c) Ratio-up-with-both-arms-slower is not a speedup: it means the machine
 *       changed, not your code.  Always report both arms' absolute times.
 *    d) OBSERVE YOUR OUTPUTS.  gcc deletes a loop whose results are never read
 *       and hands you 0.000 ms.  Checksum C and print it.
 *    e) Correctness is the L2 relative norm (rel_err in gpt.py), NOT the max
 *       element.  A max-element diff of 1.6e-04 on a K=6144 reduction with
 *       cancellation is normal and is not a gate failure.
 *    f) Bit-exactness is the real test for a refactor that should not change
 *       math: hash the whole C over ragged shapes (M%8, N%16, N<16, M<8) and
 *       compare old vs new, with a poisoned guard region after C to catch tail
 *       out-of-bounds writes.
 *
 * ---------------------------------------------------------------------------
 * 2. LOOP ORDER BEFORE MICROKERNEL.  The largest win in this file came from
 *    loop order, not from vector code.  The K=32 attention GEMMs were losing
 *    1.6x with a perfectly good microkernel: column-outermost blocking keeps a
 *    K x 16 panel of B resident but re-streams A and every row of C once per
 *    column block, and at ldc=256 floats that touches a new 4 KB page every 4
 *    rows.  When the whole B panel fits cache (K*N <= D2_ROWOUT_BMAX) go
 *    rows-outermost: same microkernel, same k order, bit-identical output.
 *      scores q@kT 256x256x32   2.707 -> 1.680 ms  (1.61x)
 *      dattn  dy@vT 256x256x32  2.554 -> 1.663 ms  (1.54x)
 *    Before writing a new microkernel, check the loop order and the number of
 *    passes over the LARGEST operand.  A 6 MB A read 16 times is the bug.
 *
 * ---------------------------------------------------------------------------
 * 3. ADDING AN ACTIVATION -> ADD AN EPILOGUE KIND, DO NOT ADD A PASS.
 *    An elementwise pass over a GEMM output costs a full re-read of a cold
 *    array.  The forward GELU cost 16.2 ms as its own loop and only ~4 ms
 *    fused, because the C tile is still in registers/L1 at the store.
 *    To add one:
 *      - define UOPC_EPI_<NAME> and handle it at the C-tile store below;
 *      - route it in sgemm_dispatch_epi (dispatch.c) for BOTH the direct and
 *        the packed path, or it will silently fall back to a separate pass;
 *      - NEVER K-split an epilogue-fused GEMM across threads: the epilogue
 *        would be applied to a partial sum.  Parallelise over M only.
 *      - if the activation needs its own derivative in backward, produce it
 *        from the SAME transcendental (see UOPC_EPI_GELU_FWD: A&S 7.1.26's
 *        own exp(-(z/sqrt2)^2) IS the exp(-z^2/2) that gelu' needs).
 *    Fusion is NOT automatically a win.  Fusing the backward dz = dgz*gelu'
 *    mask was bit-exact and removed a buffer, yet measured -1.1 ms at 1 thread
 *    and +0.9 ms at 3 threads, so it was rejected and deleted.  ALWAYS measure
 *    a fusion at 1 AND at 3 threads before landing it.
 *
 * ---------------------------------------------------------------------------
 * 4. MATCH THE REFERENCE'S MATH, DO NOT EXCEED IT.  The numpy reference
 *    (lib/aurora/numpy/gpt.py::_erf) computes A&S 7.1.26, max abs err ~1.5e-7.
 *    Calling libm erff() here bought precision the reference does not have,
 *    cannot reward, and is measurably slower.  Pick the approximation the
 *    gate actually scores you against, then spend the savings elsewhere.
 *
 * ---------------------------------------------------------------------------
 * 5. STRIDED WRITES ARE THE DEFAULT BUG.  dst[c][n] = src[n][c] walks the
 *    write side with a 24 KB stride at B*T=6144, so every store is an L1 dTLB
 *    miss.  Tiling helps but is not the answer: gcc emits the tiled nest as
 *    scalar vmovss (0 vector instructions).  Measured, 6144x256, 1 thread:
 *      naive 1.643 | tile32 1.165 | tile16 0.936 | clang+gather 0.903
 *      16x16 in-register transpose 0.409
 *    Contiguous loads AND contiguous stores beat gather/scatter.  Both
 *    compilers agree on the intrinsic form; they only disagree when left to
 *    choose.
 *
 * 6. THIS FILE IS UNCONDITIONALLY AVX-512 (__m512, no __AVX512F__ guards).
 *    That is pre-existing.  If you add intrinsics to the GENERATED C in
 *    gpt.py instead, guard them and provide a scalar fallback, then verify
 *    the fallback compiles and passes the gate:
 *      gcc -march=x86-64-v3 -mno-avx512f ...   (currently 1.463e-06, PASS)
 * ======================================================================== */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline int imin(int a, int b) { return a < b ? a : b; }

/* Row-outer blocking is used when the whole B panel fits in cache; see the
 * comment on sgemm_direct2 at the bottom of this file for the measurements. */
#define D2_ROWOUT_BMAX 65536   /* K*N floats, i.e. B <= 256 KB */

static inline void mic1(int kc, const float *A, int lda,
                             const float *B, int ldb, float *C, int ldc) {
    __m512 c00 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps();
    __m512 c60 = _mm512_setzero_ps();
    __m512 c70 = _mm512_setzero_ps();
    const float *a0 = A + (size_t)0 * lda;
    const float *a1 = A + (size_t)1 * lda;
    const float *a2 = A + (size_t)2 * lda;
    const float *a3 = A + (size_t)3 * lda;
    const float *a4 = A + (size_t)4 * lda;
    const float *a5 = A + (size_t)5 * lda;
    const float *a6 = A + (size_t)6 * lda;
    const float *a7 = A + (size_t)7 * lda;
    int p = 0;
    for (; p + 2 <= kc; p += 2) {
        const float *bp0 = B + (size_t)(p + 0) * ldb;
        __m512 b00 = _mm512_loadu_ps(bp0 + 0 + 0*16);
        const float *bp1 = B + (size_t)(p + 1) * ldb;
        __m512 b10 = _mm512_loadu_ps(bp1 + 0 + 0*16);
        __m512 a;
        a = _mm512_set1_ps(a0[p + 0]);
        c00 = _mm512_fmadd_ps(a, b00, c00);
        a = _mm512_set1_ps(a1[p + 0]);
        c10 = _mm512_fmadd_ps(a, b00, c10);
        a = _mm512_set1_ps(a2[p + 0]);
        c20 = _mm512_fmadd_ps(a, b00, c20);
        a = _mm512_set1_ps(a3[p + 0]);
        c30 = _mm512_fmadd_ps(a, b00, c30);
        a = _mm512_set1_ps(a4[p + 0]);
        c40 = _mm512_fmadd_ps(a, b00, c40);
        a = _mm512_set1_ps(a5[p + 0]);
        c50 = _mm512_fmadd_ps(a, b00, c50);
        a = _mm512_set1_ps(a6[p + 0]);
        c60 = _mm512_fmadd_ps(a, b00, c60);
        a = _mm512_set1_ps(a7[p + 0]);
        c70 = _mm512_fmadd_ps(a, b00, c70);
        a = _mm512_set1_ps(a0[p + 1]);
        c00 = _mm512_fmadd_ps(a, b10, c00);
        a = _mm512_set1_ps(a1[p + 1]);
        c10 = _mm512_fmadd_ps(a, b10, c10);
        a = _mm512_set1_ps(a2[p + 1]);
        c20 = _mm512_fmadd_ps(a, b10, c20);
        a = _mm512_set1_ps(a3[p + 1]);
        c30 = _mm512_fmadd_ps(a, b10, c30);
        a = _mm512_set1_ps(a4[p + 1]);
        c40 = _mm512_fmadd_ps(a, b10, c40);
        a = _mm512_set1_ps(a5[p + 1]);
        c50 = _mm512_fmadd_ps(a, b10, c50);
        a = _mm512_set1_ps(a6[p + 1]);
        c60 = _mm512_fmadd_ps(a, b10, c60);
        a = _mm512_set1_ps(a7[p + 1]);
        c70 = _mm512_fmadd_ps(a, b10, c70);
    }
    for (; p < kc; p++) {
        const float *bp = B + (size_t)p * ldb;
        __m512 b0 = _mm512_loadu_ps(bp + 0 + 0*16);
        __m512 a;
        a = _mm512_set1_ps(a0[p]);
        c00 = _mm512_fmadd_ps(a, b0, c00);
        a = _mm512_set1_ps(a1[p]);
        c10 = _mm512_fmadd_ps(a, b0, c10);
        a = _mm512_set1_ps(a2[p]);
        c20 = _mm512_fmadd_ps(a, b0, c20);
        a = _mm512_set1_ps(a3[p]);
        c30 = _mm512_fmadd_ps(a, b0, c30);
        a = _mm512_set1_ps(a4[p]);
        c40 = _mm512_fmadd_ps(a, b0, c40);
        a = _mm512_set1_ps(a5[p]);
        c50 = _mm512_fmadd_ps(a, b0, c50);
        a = _mm512_set1_ps(a6[p]);
        c60 = _mm512_fmadd_ps(a, b0, c60);
        a = _mm512_set1_ps(a7[p]);
        c70 = _mm512_fmadd_ps(a, b0, c70);
    }
    float *c0 = C + (size_t)0 * ldc;
    _mm512_storeu_ps(c0 + 0*16, c00);
    float *c1 = C + (size_t)1 * ldc;
    _mm512_storeu_ps(c1 + 0*16, c10);
    float *c2 = C + (size_t)2 * ldc;
    _mm512_storeu_ps(c2 + 0*16, c20);
    float *c3 = C + (size_t)3 * ldc;
    _mm512_storeu_ps(c3 + 0*16, c30);
    float *c4 = C + (size_t)4 * ldc;
    _mm512_storeu_ps(c4 + 0*16, c40);
    float *c5 = C + (size_t)5 * ldc;
    _mm512_storeu_ps(c5 + 0*16, c50);
    float *c6 = C + (size_t)6 * ldc;
    _mm512_storeu_ps(c6 + 0*16, c60);
    float *c7 = C + (size_t)7 * ldc;
    _mm512_storeu_ps(c7 + 0*16, c70);
}

static inline void mic2(int kc, const float *A, int lda,
                             const float *B, int ldb, float *C, int ldc) {
    __m512 c00 = _mm512_setzero_ps();
    __m512 c01 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps();
    __m512 c11 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps();
    __m512 c21 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps();
    __m512 c31 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps();
    __m512 c41 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps();
    __m512 c51 = _mm512_setzero_ps();
    __m512 c60 = _mm512_setzero_ps();
    __m512 c61 = _mm512_setzero_ps();
    __m512 c70 = _mm512_setzero_ps();
    __m512 c71 = _mm512_setzero_ps();
    const float *a0 = A + (size_t)0 * lda;
    const float *a1 = A + (size_t)1 * lda;
    const float *a2 = A + (size_t)2 * lda;
    const float *a3 = A + (size_t)3 * lda;
    const float *a4 = A + (size_t)4 * lda;
    const float *a5 = A + (size_t)5 * lda;
    const float *a6 = A + (size_t)6 * lda;
    const float *a7 = A + (size_t)7 * lda;
    int p = 0;
    for (; p + 2 <= kc; p += 2) {
        const float *bp0 = B + (size_t)(p + 0) * ldb;
        __m512 b00 = _mm512_loadu_ps(bp0 + 0 + 0*16);
        __m512 b01 = _mm512_loadu_ps(bp0 + 0 + 1*16);
        const float *bp1 = B + (size_t)(p + 1) * ldb;
        __m512 b10 = _mm512_loadu_ps(bp1 + 0 + 0*16);
        __m512 b11 = _mm512_loadu_ps(bp1 + 0 + 1*16);
        __m512 a;
        a = _mm512_set1_ps(a0[p + 0]);
        c00 = _mm512_fmadd_ps(a, b00, c00);
        c01 = _mm512_fmadd_ps(a, b01, c01);
        a = _mm512_set1_ps(a1[p + 0]);
        c10 = _mm512_fmadd_ps(a, b00, c10);
        c11 = _mm512_fmadd_ps(a, b01, c11);
        a = _mm512_set1_ps(a2[p + 0]);
        c20 = _mm512_fmadd_ps(a, b00, c20);
        c21 = _mm512_fmadd_ps(a, b01, c21);
        a = _mm512_set1_ps(a3[p + 0]);
        c30 = _mm512_fmadd_ps(a, b00, c30);
        c31 = _mm512_fmadd_ps(a, b01, c31);
        a = _mm512_set1_ps(a4[p + 0]);
        c40 = _mm512_fmadd_ps(a, b00, c40);
        c41 = _mm512_fmadd_ps(a, b01, c41);
        a = _mm512_set1_ps(a5[p + 0]);
        c50 = _mm512_fmadd_ps(a, b00, c50);
        c51 = _mm512_fmadd_ps(a, b01, c51);
        a = _mm512_set1_ps(a6[p + 0]);
        c60 = _mm512_fmadd_ps(a, b00, c60);
        c61 = _mm512_fmadd_ps(a, b01, c61);
        a = _mm512_set1_ps(a7[p + 0]);
        c70 = _mm512_fmadd_ps(a, b00, c70);
        c71 = _mm512_fmadd_ps(a, b01, c71);
        a = _mm512_set1_ps(a0[p + 1]);
        c00 = _mm512_fmadd_ps(a, b10, c00);
        c01 = _mm512_fmadd_ps(a, b11, c01);
        a = _mm512_set1_ps(a1[p + 1]);
        c10 = _mm512_fmadd_ps(a, b10, c10);
        c11 = _mm512_fmadd_ps(a, b11, c11);
        a = _mm512_set1_ps(a2[p + 1]);
        c20 = _mm512_fmadd_ps(a, b10, c20);
        c21 = _mm512_fmadd_ps(a, b11, c21);
        a = _mm512_set1_ps(a3[p + 1]);
        c30 = _mm512_fmadd_ps(a, b10, c30);
        c31 = _mm512_fmadd_ps(a, b11, c31);
        a = _mm512_set1_ps(a4[p + 1]);
        c40 = _mm512_fmadd_ps(a, b10, c40);
        c41 = _mm512_fmadd_ps(a, b11, c41);
        a = _mm512_set1_ps(a5[p + 1]);
        c50 = _mm512_fmadd_ps(a, b10, c50);
        c51 = _mm512_fmadd_ps(a, b11, c51);
        a = _mm512_set1_ps(a6[p + 1]);
        c60 = _mm512_fmadd_ps(a, b10, c60);
        c61 = _mm512_fmadd_ps(a, b11, c61);
        a = _mm512_set1_ps(a7[p + 1]);
        c70 = _mm512_fmadd_ps(a, b10, c70);
        c71 = _mm512_fmadd_ps(a, b11, c71);
    }
    for (; p < kc; p++) {
        const float *bp = B + (size_t)p * ldb;
        __m512 b0 = _mm512_loadu_ps(bp + 0 + 0*16);
        __m512 b1 = _mm512_loadu_ps(bp + 0 + 1*16);
        __m512 a;
        a = _mm512_set1_ps(a0[p]);
        c00 = _mm512_fmadd_ps(a, b0, c00);
        c01 = _mm512_fmadd_ps(a, b1, c01);
        a = _mm512_set1_ps(a1[p]);
        c10 = _mm512_fmadd_ps(a, b0, c10);
        c11 = _mm512_fmadd_ps(a, b1, c11);
        a = _mm512_set1_ps(a2[p]);
        c20 = _mm512_fmadd_ps(a, b0, c20);
        c21 = _mm512_fmadd_ps(a, b1, c21);
        a = _mm512_set1_ps(a3[p]);
        c30 = _mm512_fmadd_ps(a, b0, c30);
        c31 = _mm512_fmadd_ps(a, b1, c31);
        a = _mm512_set1_ps(a4[p]);
        c40 = _mm512_fmadd_ps(a, b0, c40);
        c41 = _mm512_fmadd_ps(a, b1, c41);
        a = _mm512_set1_ps(a5[p]);
        c50 = _mm512_fmadd_ps(a, b0, c50);
        c51 = _mm512_fmadd_ps(a, b1, c51);
        a = _mm512_set1_ps(a6[p]);
        c60 = _mm512_fmadd_ps(a, b0, c60);
        c61 = _mm512_fmadd_ps(a, b1, c61);
        a = _mm512_set1_ps(a7[p]);
        c70 = _mm512_fmadd_ps(a, b0, c70);
        c71 = _mm512_fmadd_ps(a, b1, c71);
    }
    float *c0 = C + (size_t)0 * ldc;
    _mm512_storeu_ps(c0 + 0*16, c00);
    _mm512_storeu_ps(c0 + 1*16, c01);
    float *c1 = C + (size_t)1 * ldc;
    _mm512_storeu_ps(c1 + 0*16, c10);
    _mm512_storeu_ps(c1 + 1*16, c11);
    float *c2 = C + (size_t)2 * ldc;
    _mm512_storeu_ps(c2 + 0*16, c20);
    _mm512_storeu_ps(c2 + 1*16, c21);
    float *c3 = C + (size_t)3 * ldc;
    _mm512_storeu_ps(c3 + 0*16, c30);
    _mm512_storeu_ps(c3 + 1*16, c31);
    float *c4 = C + (size_t)4 * ldc;
    _mm512_storeu_ps(c4 + 0*16, c40);
    _mm512_storeu_ps(c4 + 1*16, c41);
    float *c5 = C + (size_t)5 * ldc;
    _mm512_storeu_ps(c5 + 0*16, c50);
    _mm512_storeu_ps(c5 + 1*16, c51);
    float *c6 = C + (size_t)6 * ldc;
    _mm512_storeu_ps(c6 + 0*16, c60);
    _mm512_storeu_ps(c6 + 1*16, c61);
    float *c7 = C + (size_t)7 * ldc;
    _mm512_storeu_ps(c7 + 0*16, c70);
    _mm512_storeu_ps(c7 + 1*16, c71);
}

static inline void mic3(int kc, const float *A, int lda,
                             const float *B, int ldb, float *C, int ldc) {
    __m512 c00 = _mm512_setzero_ps();
    __m512 c01 = _mm512_setzero_ps();
    __m512 c02 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps();
    __m512 c11 = _mm512_setzero_ps();
    __m512 c12 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps();
    __m512 c21 = _mm512_setzero_ps();
    __m512 c22 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps();
    __m512 c31 = _mm512_setzero_ps();
    __m512 c32 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps();
    __m512 c41 = _mm512_setzero_ps();
    __m512 c42 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps();
    __m512 c51 = _mm512_setzero_ps();
    __m512 c52 = _mm512_setzero_ps();
    __m512 c60 = _mm512_setzero_ps();
    __m512 c61 = _mm512_setzero_ps();
    __m512 c62 = _mm512_setzero_ps();
    __m512 c70 = _mm512_setzero_ps();
    __m512 c71 = _mm512_setzero_ps();
    __m512 c72 = _mm512_setzero_ps();
    const float *a0 = A + (size_t)0 * lda;
    const float *a1 = A + (size_t)1 * lda;
    const float *a2 = A + (size_t)2 * lda;
    const float *a3 = A + (size_t)3 * lda;
    const float *a4 = A + (size_t)4 * lda;
    const float *a5 = A + (size_t)5 * lda;
    const float *a6 = A + (size_t)6 * lda;
    const float *a7 = A + (size_t)7 * lda;
    int p = 0;
    for (; p + 2 <= kc; p += 2) {
        const float *bp0 = B + (size_t)(p + 0) * ldb;
        __m512 b00 = _mm512_loadu_ps(bp0 + 0 + 0*16);
        __m512 b01 = _mm512_loadu_ps(bp0 + 0 + 1*16);
        __m512 b02 = _mm512_loadu_ps(bp0 + 0 + 2*16);
        const float *bp1 = B + (size_t)(p + 1) * ldb;
        __m512 b10 = _mm512_loadu_ps(bp1 + 0 + 0*16);
        __m512 b11 = _mm512_loadu_ps(bp1 + 0 + 1*16);
        __m512 b12 = _mm512_loadu_ps(bp1 + 0 + 2*16);
        __m512 a;
        a = _mm512_set1_ps(a0[p + 0]);
        c00 = _mm512_fmadd_ps(a, b00, c00);
        c01 = _mm512_fmadd_ps(a, b01, c01);
        c02 = _mm512_fmadd_ps(a, b02, c02);
        a = _mm512_set1_ps(a1[p + 0]);
        c10 = _mm512_fmadd_ps(a, b00, c10);
        c11 = _mm512_fmadd_ps(a, b01, c11);
        c12 = _mm512_fmadd_ps(a, b02, c12);
        a = _mm512_set1_ps(a2[p + 0]);
        c20 = _mm512_fmadd_ps(a, b00, c20);
        c21 = _mm512_fmadd_ps(a, b01, c21);
        c22 = _mm512_fmadd_ps(a, b02, c22);
        a = _mm512_set1_ps(a3[p + 0]);
        c30 = _mm512_fmadd_ps(a, b00, c30);
        c31 = _mm512_fmadd_ps(a, b01, c31);
        c32 = _mm512_fmadd_ps(a, b02, c32);
        a = _mm512_set1_ps(a4[p + 0]);
        c40 = _mm512_fmadd_ps(a, b00, c40);
        c41 = _mm512_fmadd_ps(a, b01, c41);
        c42 = _mm512_fmadd_ps(a, b02, c42);
        a = _mm512_set1_ps(a5[p + 0]);
        c50 = _mm512_fmadd_ps(a, b00, c50);
        c51 = _mm512_fmadd_ps(a, b01, c51);
        c52 = _mm512_fmadd_ps(a, b02, c52);
        a = _mm512_set1_ps(a6[p + 0]);
        c60 = _mm512_fmadd_ps(a, b00, c60);
        c61 = _mm512_fmadd_ps(a, b01, c61);
        c62 = _mm512_fmadd_ps(a, b02, c62);
        a = _mm512_set1_ps(a7[p + 0]);
        c70 = _mm512_fmadd_ps(a, b00, c70);
        c71 = _mm512_fmadd_ps(a, b01, c71);
        c72 = _mm512_fmadd_ps(a, b02, c72);
        a = _mm512_set1_ps(a0[p + 1]);
        c00 = _mm512_fmadd_ps(a, b10, c00);
        c01 = _mm512_fmadd_ps(a, b11, c01);
        c02 = _mm512_fmadd_ps(a, b12, c02);
        a = _mm512_set1_ps(a1[p + 1]);
        c10 = _mm512_fmadd_ps(a, b10, c10);
        c11 = _mm512_fmadd_ps(a, b11, c11);
        c12 = _mm512_fmadd_ps(a, b12, c12);
        a = _mm512_set1_ps(a2[p + 1]);
        c20 = _mm512_fmadd_ps(a, b10, c20);
        c21 = _mm512_fmadd_ps(a, b11, c21);
        c22 = _mm512_fmadd_ps(a, b12, c22);
        a = _mm512_set1_ps(a3[p + 1]);
        c30 = _mm512_fmadd_ps(a, b10, c30);
        c31 = _mm512_fmadd_ps(a, b11, c31);
        c32 = _mm512_fmadd_ps(a, b12, c32);
        a = _mm512_set1_ps(a4[p + 1]);
        c40 = _mm512_fmadd_ps(a, b10, c40);
        c41 = _mm512_fmadd_ps(a, b11, c41);
        c42 = _mm512_fmadd_ps(a, b12, c42);
        a = _mm512_set1_ps(a5[p + 1]);
        c50 = _mm512_fmadd_ps(a, b10, c50);
        c51 = _mm512_fmadd_ps(a, b11, c51);
        c52 = _mm512_fmadd_ps(a, b12, c52);
        a = _mm512_set1_ps(a6[p + 1]);
        c60 = _mm512_fmadd_ps(a, b10, c60);
        c61 = _mm512_fmadd_ps(a, b11, c61);
        c62 = _mm512_fmadd_ps(a, b12, c62);
        a = _mm512_set1_ps(a7[p + 1]);
        c70 = _mm512_fmadd_ps(a, b10, c70);
        c71 = _mm512_fmadd_ps(a, b11, c71);
        c72 = _mm512_fmadd_ps(a, b12, c72);
    }
    for (; p < kc; p++) {
        const float *bp = B + (size_t)p * ldb;
        __m512 b0 = _mm512_loadu_ps(bp + 0 + 0*16);
        __m512 b1 = _mm512_loadu_ps(bp + 0 + 1*16);
        __m512 b2 = _mm512_loadu_ps(bp + 0 + 2*16);
        __m512 a;
        a = _mm512_set1_ps(a0[p]);
        c00 = _mm512_fmadd_ps(a, b0, c00);
        c01 = _mm512_fmadd_ps(a, b1, c01);
        c02 = _mm512_fmadd_ps(a, b2, c02);
        a = _mm512_set1_ps(a1[p]);
        c10 = _mm512_fmadd_ps(a, b0, c10);
        c11 = _mm512_fmadd_ps(a, b1, c11);
        c12 = _mm512_fmadd_ps(a, b2, c12);
        a = _mm512_set1_ps(a2[p]);
        c20 = _mm512_fmadd_ps(a, b0, c20);
        c21 = _mm512_fmadd_ps(a, b1, c21);
        c22 = _mm512_fmadd_ps(a, b2, c22);
        a = _mm512_set1_ps(a3[p]);
        c30 = _mm512_fmadd_ps(a, b0, c30);
        c31 = _mm512_fmadd_ps(a, b1, c31);
        c32 = _mm512_fmadd_ps(a, b2, c32);
        a = _mm512_set1_ps(a4[p]);
        c40 = _mm512_fmadd_ps(a, b0, c40);
        c41 = _mm512_fmadd_ps(a, b1, c41);
        c42 = _mm512_fmadd_ps(a, b2, c42);
        a = _mm512_set1_ps(a5[p]);
        c50 = _mm512_fmadd_ps(a, b0, c50);
        c51 = _mm512_fmadd_ps(a, b1, c51);
        c52 = _mm512_fmadd_ps(a, b2, c52);
        a = _mm512_set1_ps(a6[p]);
        c60 = _mm512_fmadd_ps(a, b0, c60);
        c61 = _mm512_fmadd_ps(a, b1, c61);
        c62 = _mm512_fmadd_ps(a, b2, c62);
        a = _mm512_set1_ps(a7[p]);
        c70 = _mm512_fmadd_ps(a, b0, c70);
        c71 = _mm512_fmadd_ps(a, b1, c71);
        c72 = _mm512_fmadd_ps(a, b2, c72);
    }
    float *c0 = C + (size_t)0 * ldc;
    _mm512_storeu_ps(c0 + 0*16, c00);
    _mm512_storeu_ps(c0 + 1*16, c01);
    _mm512_storeu_ps(c0 + 2*16, c02);
    float *c1 = C + (size_t)1 * ldc;
    _mm512_storeu_ps(c1 + 0*16, c10);
    _mm512_storeu_ps(c1 + 1*16, c11);
    _mm512_storeu_ps(c1 + 2*16, c12);
    float *c2 = C + (size_t)2 * ldc;
    _mm512_storeu_ps(c2 + 0*16, c20);
    _mm512_storeu_ps(c2 + 1*16, c21);
    _mm512_storeu_ps(c2 + 2*16, c22);
    float *c3 = C + (size_t)3 * ldc;
    _mm512_storeu_ps(c3 + 0*16, c30);
    _mm512_storeu_ps(c3 + 1*16, c31);
    _mm512_storeu_ps(c3 + 2*16, c32);
    float *c4 = C + (size_t)4 * ldc;
    _mm512_storeu_ps(c4 + 0*16, c40);
    _mm512_storeu_ps(c4 + 1*16, c41);
    _mm512_storeu_ps(c4 + 2*16, c42);
    float *c5 = C + (size_t)5 * ldc;
    _mm512_storeu_ps(c5 + 0*16, c50);
    _mm512_storeu_ps(c5 + 1*16, c51);
    _mm512_storeu_ps(c5 + 2*16, c52);
    float *c6 = C + (size_t)6 * ldc;
    _mm512_storeu_ps(c6 + 0*16, c60);
    _mm512_storeu_ps(c6 + 1*16, c61);
    _mm512_storeu_ps(c6 + 2*16, c62);
    float *c7 = C + (size_t)7 * ldc;
    _mm512_storeu_ps(c7 + 0*16, c70);
    _mm512_storeu_ps(c7 + 1*16, c71);
    _mm512_storeu_ps(c7 + 2*16, c72);
}

static inline void mic1t(int kc, const float *A, int lda,
                             const float *B, int ldb, float *C, int ldc, int nb) {
    __m512 c00 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps();
    __m512 c60 = _mm512_setzero_ps();
    __m512 c70 = _mm512_setzero_ps();
    __mmask16 m0 = (nb - 0*16 >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (nb - 0*16)) - 1u);
    const float *a0 = A + (size_t)0 * lda;
    const float *a1 = A + (size_t)1 * lda;
    const float *a2 = A + (size_t)2 * lda;
    const float *a3 = A + (size_t)3 * lda;
    const float *a4 = A + (size_t)4 * lda;
    const float *a5 = A + (size_t)5 * lda;
    const float *a6 = A + (size_t)6 * lda;
    const float *a7 = A + (size_t)7 * lda;
    int p = 0;
    for (; p + 2 <= kc; p += 2) {
        const float *bp0 = B + (size_t)(p + 0) * ldb;
        __m512 b00 = _mm512_maskz_loadu_ps(m0, bp0 + 0 + 0*16);
        const float *bp1 = B + (size_t)(p + 1) * ldb;
        __m512 b10 = _mm512_maskz_loadu_ps(m0, bp1 + 0 + 0*16);
        __m512 a;
        a = _mm512_set1_ps(a0[p + 0]);
        c00 = _mm512_fmadd_ps(a, b00, c00);
        a = _mm512_set1_ps(a1[p + 0]);
        c10 = _mm512_fmadd_ps(a, b00, c10);
        a = _mm512_set1_ps(a2[p + 0]);
        c20 = _mm512_fmadd_ps(a, b00, c20);
        a = _mm512_set1_ps(a3[p + 0]);
        c30 = _mm512_fmadd_ps(a, b00, c30);
        a = _mm512_set1_ps(a4[p + 0]);
        c40 = _mm512_fmadd_ps(a, b00, c40);
        a = _mm512_set1_ps(a5[p + 0]);
        c50 = _mm512_fmadd_ps(a, b00, c50);
        a = _mm512_set1_ps(a6[p + 0]);
        c60 = _mm512_fmadd_ps(a, b00, c60);
        a = _mm512_set1_ps(a7[p + 0]);
        c70 = _mm512_fmadd_ps(a, b00, c70);
        a = _mm512_set1_ps(a0[p + 1]);
        c00 = _mm512_fmadd_ps(a, b10, c00);
        a = _mm512_set1_ps(a1[p + 1]);
        c10 = _mm512_fmadd_ps(a, b10, c10);
        a = _mm512_set1_ps(a2[p + 1]);
        c20 = _mm512_fmadd_ps(a, b10, c20);
        a = _mm512_set1_ps(a3[p + 1]);
        c30 = _mm512_fmadd_ps(a, b10, c30);
        a = _mm512_set1_ps(a4[p + 1]);
        c40 = _mm512_fmadd_ps(a, b10, c40);
        a = _mm512_set1_ps(a5[p + 1]);
        c50 = _mm512_fmadd_ps(a, b10, c50);
        a = _mm512_set1_ps(a6[p + 1]);
        c60 = _mm512_fmadd_ps(a, b10, c60);
        a = _mm512_set1_ps(a7[p + 1]);
        c70 = _mm512_fmadd_ps(a, b10, c70);
    }
    for (; p < kc; p++) {
        const float *bp = B + (size_t)p * ldb;
        __m512 b0 = _mm512_maskz_loadu_ps(m0, bp + 0 + 0*16);
        __m512 a;
        a = _mm512_set1_ps(a0[p]);
        c00 = _mm512_fmadd_ps(a, b0, c00);
        a = _mm512_set1_ps(a1[p]);
        c10 = _mm512_fmadd_ps(a, b0, c10);
        a = _mm512_set1_ps(a2[p]);
        c20 = _mm512_fmadd_ps(a, b0, c20);
        a = _mm512_set1_ps(a3[p]);
        c30 = _mm512_fmadd_ps(a, b0, c30);
        a = _mm512_set1_ps(a4[p]);
        c40 = _mm512_fmadd_ps(a, b0, c40);
        a = _mm512_set1_ps(a5[p]);
        c50 = _mm512_fmadd_ps(a, b0, c50);
        a = _mm512_set1_ps(a6[p]);
        c60 = _mm512_fmadd_ps(a, b0, c60);
        a = _mm512_set1_ps(a7[p]);
        c70 = _mm512_fmadd_ps(a, b0, c70);
    }
    float *c0 = C + (size_t)0 * ldc;
    _mm512_mask_storeu_ps(c0 + 0*16, m0, c00);
    float *c1 = C + (size_t)1 * ldc;
    _mm512_mask_storeu_ps(c1 + 0*16, m0, c10);
    float *c2 = C + (size_t)2 * ldc;
    _mm512_mask_storeu_ps(c2 + 0*16, m0, c20);
    float *c3 = C + (size_t)3 * ldc;
    _mm512_mask_storeu_ps(c3 + 0*16, m0, c30);
    float *c4 = C + (size_t)4 * ldc;
    _mm512_mask_storeu_ps(c4 + 0*16, m0, c40);
    float *c5 = C + (size_t)5 * ldc;
    _mm512_mask_storeu_ps(c5 + 0*16, m0, c50);
    float *c6 = C + (size_t)6 * ldc;
    _mm512_mask_storeu_ps(c6 + 0*16, m0, c60);
    float *c7 = C + (size_t)7 * ldc;
    _mm512_mask_storeu_ps(c7 + 0*16, m0, c70);
}

static inline void mic2t(int kc, const float *A, int lda,
                             const float *B, int ldb, float *C, int ldc, int nb) {
    __m512 c00 = _mm512_setzero_ps();
    __m512 c01 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps();
    __m512 c11 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps();
    __m512 c21 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps();
    __m512 c31 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps();
    __m512 c41 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps();
    __m512 c51 = _mm512_setzero_ps();
    __m512 c60 = _mm512_setzero_ps();
    __m512 c61 = _mm512_setzero_ps();
    __m512 c70 = _mm512_setzero_ps();
    __m512 c71 = _mm512_setzero_ps();
    __mmask16 m0 = (nb - 0*16 >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (nb - 0*16)) - 1u);
    __mmask16 m1 = (nb - 1*16 >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << (nb - 1*16)) - 1u);
    const float *a0 = A + (size_t)0 * lda;
    const float *a1 = A + (size_t)1 * lda;
    const float *a2 = A + (size_t)2 * lda;
    const float *a3 = A + (size_t)3 * lda;
    const float *a4 = A + (size_t)4 * lda;
    const float *a5 = A + (size_t)5 * lda;
    const float *a6 = A + (size_t)6 * lda;
    const float *a7 = A + (size_t)7 * lda;
    int p = 0;
    for (; p + 2 <= kc; p += 2) {
        const float *bp0 = B + (size_t)(p + 0) * ldb;
        __m512 b00 = _mm512_maskz_loadu_ps(m0, bp0 + 0 + 0*16);
        __m512 b01 = _mm512_maskz_loadu_ps(m1, bp0 + 0 + 1*16);
        const float *bp1 = B + (size_t)(p + 1) * ldb;
        __m512 b10 = _mm512_maskz_loadu_ps(m0, bp1 + 0 + 0*16);
        __m512 b11 = _mm512_maskz_loadu_ps(m1, bp1 + 0 + 1*16);
        __m512 a;
        a = _mm512_set1_ps(a0[p + 0]);
        c00 = _mm512_fmadd_ps(a, b00, c00);
        c01 = _mm512_fmadd_ps(a, b01, c01);
        a = _mm512_set1_ps(a1[p + 0]);
        c10 = _mm512_fmadd_ps(a, b00, c10);
        c11 = _mm512_fmadd_ps(a, b01, c11);
        a = _mm512_set1_ps(a2[p + 0]);
        c20 = _mm512_fmadd_ps(a, b00, c20);
        c21 = _mm512_fmadd_ps(a, b01, c21);
        a = _mm512_set1_ps(a3[p + 0]);
        c30 = _mm512_fmadd_ps(a, b00, c30);
        c31 = _mm512_fmadd_ps(a, b01, c31);
        a = _mm512_set1_ps(a4[p + 0]);
        c40 = _mm512_fmadd_ps(a, b00, c40);
        c41 = _mm512_fmadd_ps(a, b01, c41);
        a = _mm512_set1_ps(a5[p + 0]);
        c50 = _mm512_fmadd_ps(a, b00, c50);
        c51 = _mm512_fmadd_ps(a, b01, c51);
        a = _mm512_set1_ps(a6[p + 0]);
        c60 = _mm512_fmadd_ps(a, b00, c60);
        c61 = _mm512_fmadd_ps(a, b01, c61);
        a = _mm512_set1_ps(a7[p + 0]);
        c70 = _mm512_fmadd_ps(a, b00, c70);
        c71 = _mm512_fmadd_ps(a, b01, c71);
        a = _mm512_set1_ps(a0[p + 1]);
        c00 = _mm512_fmadd_ps(a, b10, c00);
        c01 = _mm512_fmadd_ps(a, b11, c01);
        a = _mm512_set1_ps(a1[p + 1]);
        c10 = _mm512_fmadd_ps(a, b10, c10);
        c11 = _mm512_fmadd_ps(a, b11, c11);
        a = _mm512_set1_ps(a2[p + 1]);
        c20 = _mm512_fmadd_ps(a, b10, c20);
        c21 = _mm512_fmadd_ps(a, b11, c21);
        a = _mm512_set1_ps(a3[p + 1]);
        c30 = _mm512_fmadd_ps(a, b10, c30);
        c31 = _mm512_fmadd_ps(a, b11, c31);
        a = _mm512_set1_ps(a4[p + 1]);
        c40 = _mm512_fmadd_ps(a, b10, c40);
        c41 = _mm512_fmadd_ps(a, b11, c41);
        a = _mm512_set1_ps(a5[p + 1]);
        c50 = _mm512_fmadd_ps(a, b10, c50);
        c51 = _mm512_fmadd_ps(a, b11, c51);
        a = _mm512_set1_ps(a6[p + 1]);
        c60 = _mm512_fmadd_ps(a, b10, c60);
        c61 = _mm512_fmadd_ps(a, b11, c61);
        a = _mm512_set1_ps(a7[p + 1]);
        c70 = _mm512_fmadd_ps(a, b10, c70);
        c71 = _mm512_fmadd_ps(a, b11, c71);
    }
    for (; p < kc; p++) {
        const float *bp = B + (size_t)p * ldb;
        __m512 b0 = _mm512_maskz_loadu_ps(m0, bp + 0 + 0*16);
        __m512 b1 = _mm512_maskz_loadu_ps(m1, bp + 0 + 1*16);
        __m512 a;
        a = _mm512_set1_ps(a0[p]);
        c00 = _mm512_fmadd_ps(a, b0, c00);
        c01 = _mm512_fmadd_ps(a, b1, c01);
        a = _mm512_set1_ps(a1[p]);
        c10 = _mm512_fmadd_ps(a, b0, c10);
        c11 = _mm512_fmadd_ps(a, b1, c11);
        a = _mm512_set1_ps(a2[p]);
        c20 = _mm512_fmadd_ps(a, b0, c20);
        c21 = _mm512_fmadd_ps(a, b1, c21);
        a = _mm512_set1_ps(a3[p]);
        c30 = _mm512_fmadd_ps(a, b0, c30);
        c31 = _mm512_fmadd_ps(a, b1, c31);
        a = _mm512_set1_ps(a4[p]);
        c40 = _mm512_fmadd_ps(a, b0, c40);
        c41 = _mm512_fmadd_ps(a, b1, c41);
        a = _mm512_set1_ps(a5[p]);
        c50 = _mm512_fmadd_ps(a, b0, c50);
        c51 = _mm512_fmadd_ps(a, b1, c51);
        a = _mm512_set1_ps(a6[p]);
        c60 = _mm512_fmadd_ps(a, b0, c60);
        c61 = _mm512_fmadd_ps(a, b1, c61);
        a = _mm512_set1_ps(a7[p]);
        c70 = _mm512_fmadd_ps(a, b0, c70);
        c71 = _mm512_fmadd_ps(a, b1, c71);
    }
    float *c0 = C + (size_t)0 * ldc;
    _mm512_mask_storeu_ps(c0 + 0*16, m0, c00);
    _mm512_mask_storeu_ps(c0 + 1*16, m1, c01);
    float *c1 = C + (size_t)1 * ldc;
    _mm512_mask_storeu_ps(c1 + 0*16, m0, c10);
    _mm512_mask_storeu_ps(c1 + 1*16, m1, c11);
    float *c2 = C + (size_t)2 * ldc;
    _mm512_mask_storeu_ps(c2 + 0*16, m0, c20);
    _mm512_mask_storeu_ps(c2 + 1*16, m1, c21);
    float *c3 = C + (size_t)3 * ldc;
    _mm512_mask_storeu_ps(c3 + 0*16, m0, c30);
    _mm512_mask_storeu_ps(c3 + 1*16, m1, c31);
    float *c4 = C + (size_t)4 * ldc;
    _mm512_mask_storeu_ps(c4 + 0*16, m0, c40);
    _mm512_mask_storeu_ps(c4 + 1*16, m1, c41);
    float *c5 = C + (size_t)5 * ldc;
    _mm512_mask_storeu_ps(c5 + 0*16, m0, c50);
    _mm512_mask_storeu_ps(c5 + 1*16, m1, c51);
    float *c6 = C + (size_t)6 * ldc;
    _mm512_mask_storeu_ps(c6 + 0*16, m0, c60);
    _mm512_mask_storeu_ps(c6 + 1*16, m1, c61);
    float *c7 = C + (size_t)7 * ldc;
    _mm512_mask_storeu_ps(c7 + 0*16, m0, c70);
    _mm512_mask_storeu_ps(c7 + 1*16, m1, c71);
}


static inline void d2_block16(int M, int K, const float *A, int lda,
                               const float *B, int ldb, float *C, int ldc,
                               int j0) {
    const float *b = B + j0;
    int mfull = M & ~7; /* rows covered by complete 8-row blocks */
    for (int i = 0; i < mfull; i += 8) {
        const float *a = A + (size_t)i * lda;
        float *c = C + (size_t)i * ldc + j0;
        mic1(K, a, lda, b, ldb, c, ldc);
    }
    if (M != mfull) { /* M % 8 != 0: tail rows exist */
        if (M >= 8) {
            /* Overlap M-tail: recompute rows [M-8, M) with the full
             * 8-row vector kernel (idempotent), so no OOB access ever
             * happens for M % 8 != 0. */
            int i0 = M - 8;
            const float *a = A + (size_t)i0 * lda;
            float *c = C + (size_t)i0 * ldc + j0;
            mic1(K, a, lda, b, ldb, c, ldc);
        } else {
            /* M < 8: scalar fallback (tiny, never in the hot path). */
            for (int ii = 0; ii < M; ii++) {
                const float *ar = A + (size_t)ii * lda;
                float *cr = C + (size_t)ii * ldc + j0;
                for (int jj = 0; jj < 16; jj++) {
                    float acc = 0.0f;
                    for (int kk = 0; kk < K; kk++)
                        acc += ar[kk] * b[(size_t)kk * ldb + jj];
                    cr[jj] = acc;
                }
            }
        }
    }
}

/* 32-column sibling of d2_block16: same blocking, mic2's 8x32 register
 * tile.  Wider tiles halve the number of passes over A, which pays only when A
 * is the streamed operand and K is large (see sgemm_direct2). */
static inline void d2_block32(int M, int K, const float *A, int lda,
                               const float *B, int ldb, float *C, int ldc,
                               int j0) {
    const float *b = B + j0;
    int mfull = M & ~7; /* rows covered by complete 8-row blocks */
    for (int i = 0; i < mfull; i += 8) {
        const float *a = A + (size_t)i * lda;
        float *c = C + (size_t)i * ldc + j0;
        mic2(K, a, lda, b, ldb, c, ldc);
    }
    if (M != mfull) { /* M % 8 != 0: tail rows exist */
        if (M >= 8) {
            /* Overlap M-tail: recompute rows [M-8, M) with the full
             * 8-row vector kernel (idempotent), so no OOB access ever
             * happens for M % 8 != 0. */
            int i0 = M - 8;
            const float *a = A + (size_t)i0 * lda;
            float *c = C + (size_t)i0 * ldc + j0;
            mic2(K, a, lda, b, ldb, c, ldc);
        } else {
            /* M < 8: scalar fallback (tiny, never in the hot path). */
            for (int ii = 0; ii < M; ii++) {
                const float *ar = A + (size_t)ii * lda;
                float *cr = C + (size_t)ii * ldc + j0;
                for (int jj = 0; jj < 32; jj++) {
                    float acc = 0.0f;
                    for (int kk = 0; kk < K; kk++)
                        acc += ar[kk] * b[(size_t)kk * ldb + jj];
                    cr[jj] = acc;
                }
            }
        }
    }
}

/* Fast dot product of two contiguous K-vectors (K-vectorized). */
static inline float d2_dot1(const float *ar, const float *bp, int K) {
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 64 <= K; k += 64) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(ar + k), _mm512_loadu_ps(bp + k), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(ar + k + 16), _mm512_loadu_ps(bp + k + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(ar + k + 32), _mm512_loadu_ps(bp + k + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(ar + k + 48), _mm512_loadu_ps(bp + k + 48), a3);
    }
    for (; k + 16 <= K; k += 16)
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(ar + k), _mm512_loadu_ps(bp + k), a0);
    __m512 acc = _mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3));
    float s = _mm512_reduce_add_ps(acc);
    for (; k < K; k++)
        s += ar[k] * bp[k];
    return s;
}

/* N-tail (nb = N - nfull < 16).  For the common nb==1 case compute the
 * single leftover column with a K-vectorized GEMV (contiguous packed B
 * column), which is far cheaper than a full 16-column masked block.  For
 * larger tails the masked 16-wide kernel is already about as fast as a
 * full block, so keep using it (no recompute overhead, no regression). */
static inline void d2_tail(int M, int K, const float *A, int lda,
                           const float *B, int ldb, float *C, int ldc,
                           int nfull, int nb) {
    if (nb == 1) {
        const float *bj = B + nfull;
        float *bp = (float *)malloc((size_t)K * sizeof(float));
        if (!bp) return;
        for (int k = 0; k < K; k++)
            bp[k] = bj[(size_t)k * ldb];
        for (int i = 0; i < M; i++) {
            const float *ar = A + (size_t)i * lda;
            C[(size_t)i * ldc + nfull] = d2_dot1(ar, bp, K);
        }
        free(bp);
        return;
    }
    /* nb in [2, 15]: masked 16-wide tail, same as before. */
    const float *b = B + nfull;
    int mfull = M & ~7;
    for (int i = 0; i < mfull; i += 8) {
        const float *a = A + (size_t)i * lda;
        float *c = C + (size_t)i * ldc + nfull;
        mic1t(K, a, lda, b, ldb, c, ldc, nb);
    }
    if (M != mfull) {
        if (M >= 8) {
            int i0 = M - 8;
            const float *a = A + (size_t)i0 * lda;
            float *c = C + (size_t)i0 * ldc + nfull;
            mic1t(K, a, lda, b, ldb, c, ldc, nb);
        } else {
            for (int ii = 0; ii < M; ii++) {
                const float *ar = A + (size_t)ii * lda;
                float *cr = C + (size_t)ii * ldc + nfull;
                for (int jj = 0; jj < nb; jj++) {
                    float acc = 0.0f;
                    for (int kk = 0; kk < K; kk++)
                        acc += ar[kk] * b[(size_t)kk * ldb + jj];
                    cr[jj] = acc;
                }
            }
        }
    }
}


/* 8-row x 8-col microkernel (256-bit, k-unroll 2).
 * For N==8 (thin attention GEMMs) the 16-wide kernel wastes half its FMAs on
 * a masked tail; this 8-wide kernel does exactly the useful work at full FMA
 * rate (measured ~1.7x faster on gemm(160,8,160)). */
static inline void mic8(int kc, const float *A, int lda,
                        const float *B, int ldb, float *C, int ldc) {
    __m256 c00 = _mm256_setzero_ps(), c10 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c30 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c50 = _mm256_setzero_ps();
    __m256 c60 = _mm256_setzero_ps(), c70 = _mm256_setzero_ps();
    const float *a0 = A + (size_t)0 * lda, *a1 = A + (size_t)1 * lda;
    const float *a2 = A + (size_t)2 * lda, *a3 = A + (size_t)3 * lda;
    const float *a4 = A + (size_t)4 * lda, *a5 = A + (size_t)5 * lda;
    const float *a6 = A + (size_t)6 * lda, *a7 = A + (size_t)7 * lda;
    int p = 0;
    for (; p + 2 <= kc; p += 2) {
        __m256 b0 = _mm256_loadu_ps(B + (size_t)(p + 0) * ldb);
        __m256 b1 = _mm256_loadu_ps(B + (size_t)(p + 1) * ldb);
        __m256 a;
        a = _mm256_set1_ps(a0[p + 0]); c00 = _mm256_fmadd_ps(a, b0, c00);
        a = _mm256_set1_ps(a1[p + 0]); c10 = _mm256_fmadd_ps(a, b0, c10);
        a = _mm256_set1_ps(a2[p + 0]); c20 = _mm256_fmadd_ps(a, b0, c20);
        a = _mm256_set1_ps(a3[p + 0]); c30 = _mm256_fmadd_ps(a, b0, c30);
        a = _mm256_set1_ps(a4[p + 0]); c40 = _mm256_fmadd_ps(a, b0, c40);
        a = _mm256_set1_ps(a5[p + 0]); c50 = _mm256_fmadd_ps(a, b0, c50);
        a = _mm256_set1_ps(a6[p + 0]); c60 = _mm256_fmadd_ps(a, b0, c60);
        a = _mm256_set1_ps(a7[p + 0]); c70 = _mm256_fmadd_ps(a, b0, c70);
        a = _mm256_set1_ps(a0[p + 1]); c00 = _mm256_fmadd_ps(a, b1, c00);
        a = _mm256_set1_ps(a1[p + 1]); c10 = _mm256_fmadd_ps(a, b1, c10);
        a = _mm256_set1_ps(a2[p + 1]); c20 = _mm256_fmadd_ps(a, b1, c20);
        a = _mm256_set1_ps(a3[p + 1]); c30 = _mm256_fmadd_ps(a, b1, c30);
        a = _mm256_set1_ps(a4[p + 1]); c40 = _mm256_fmadd_ps(a, b1, c40);
        a = _mm256_set1_ps(a5[p + 1]); c50 = _mm256_fmadd_ps(a, b1, c50);
        a = _mm256_set1_ps(a6[p + 1]); c60 = _mm256_fmadd_ps(a, b1, c60);
        a = _mm256_set1_ps(a7[p + 1]); c70 = _mm256_fmadd_ps(a, b1, c70);
    }
    for (; p < kc; p++) {
        __m256 b0 = _mm256_loadu_ps(B + (size_t)p * ldb);
        __m256 a;
        a = _mm256_set1_ps(a0[p]); c00 = _mm256_fmadd_ps(a, b0, c00);
        a = _mm256_set1_ps(a1[p]); c10 = _mm256_fmadd_ps(a, b0, c10);
        a = _mm256_set1_ps(a2[p]); c20 = _mm256_fmadd_ps(a, b0, c20);
        a = _mm256_set1_ps(a3[p]); c30 = _mm256_fmadd_ps(a, b0, c30);
        a = _mm256_set1_ps(a4[p]); c40 = _mm256_fmadd_ps(a, b0, c40);
        a = _mm256_set1_ps(a5[p]); c50 = _mm256_fmadd_ps(a, b0, c50);
        a = _mm256_set1_ps(a6[p]); c60 = _mm256_fmadd_ps(a, b0, c60);
        a = _mm256_set1_ps(a7[p]); c70 = _mm256_fmadd_ps(a, b0, c70);
    }
    _mm256_storeu_ps(C + (size_t)0 * ldc, c00);
    _mm256_storeu_ps(C + (size_t)1 * ldc, c10);
    _mm256_storeu_ps(C + (size_t)2 * ldc, c20);
    _mm256_storeu_ps(C + (size_t)3 * ldc, c30);
    _mm256_storeu_ps(C + (size_t)4 * ldc, c40);
    _mm256_storeu_ps(C + (size_t)5 * ldc, c50);
    _mm256_storeu_ps(C + (size_t)6 * ldc, c60);
    _mm256_storeu_ps(C + (size_t)7 * ldc, c70);
}

/* N==8 fast path: 8-wide microkernel (avoids the masked-16 tail's 2x FMA). */
static inline void d2_block8(int M, int K, const float *A, int lda,
                             const float *B, int ldb, float *C, int ldc) {
    int mfull = M & ~7;
    for (int i = 0; i < mfull; i += 8)
        mic8(K, A + (size_t)i * lda, lda, B, ldb, C + (size_t)i * ldc, ldc);
    if (M != mfull) {
        if (M >= 8) {
            int i0 = M - 8;
            mic8(K, A + (size_t)i0 * lda, lda, B, ldb, C + (size_t)i0 * ldc, ldc);
        } else {
            for (int ii = 0; ii < M; ii++) {
                const float *ar = A + (size_t)ii * lda;
                float *cr = C + (size_t)ii * ldc;
                for (int jj = 0; jj < 8; jj++) {
                    float acc = 0.0f;
                    for (int kk = 0; kk < K; kk++)
                        acc += ar[kk] * B[(size_t)kk * ldb + jj];
                    cr[jj] = acc;
                }
            }
        }
    }
}


/* ---------------------------------------------------------------------------
 * GEMM epilogue fusion (fused into the C-tile store).
 *   kind 1 (UOPC_EPI_ADD)  : C = C + E   -- bit-exact residual add: the same
 *                             two f32 operands as the unfused EW, so results
 *                             are bit-identical.
 *   kind 2 (UOPC_EPI_GELU) : C = C * gelu'(E), gelu'(x) = 0.5*(1+erf(x/sqrt2))
 *                             + x*exp(-x^2/2)/sqrt(2pi).  erf via A&S 7.1.26
 *                             (max abs err ~1.5e-7) + the fast exp2 poly
 *                             (~4e-6 rel), giving |gelu' err| <= ~6e-7 - well
 *                             inside the graph's 1e-6 loss gate.
 * ------------------------------------------------------------------------- */
#define UOPC_EPI_ADD  1
#define UOPC_EPI_GELU 2
#define UOPC_EPI_FMA  3   /* C = C + s*E: FMA fold of an intervening scale */
/* kind 4 (UOPC_EPI_GELU_FWD): C = gelu(C) AND *E* <- gelu'(C).  E is an
 * OUTPUT for this kind (the only kind that writes through it), because the
 * forward GELU needs both values and both come from the same erf/exp pair the
 * register tile already holds.  Traffic before fusion, per layer at DD=256,
 * N=6144: GEMM writes z (6.3 MB), the loop reads z, writes gelu(z) and
 * gelu'(z).  After: the GEMM stores gelu(z) and gelu'(z) straight out of the
 * accumulators and z never reaches memory at all. */
#define UOPC_EPI_GELU_FWD 4

static inline __m512 uopc_expf_fast512(__m512 x) {
    __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f));
    t = _mm512_max_ps(_mm512_set1_ps(-126.0f),
                      _mm512_min_ps(_mm512_set1_ps(127.0f), t));
    __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 f = _mm512_sub_ps(t, n);
    __m512 p = _mm512_fmadd_ps(f, _mm512_set1_ps(0.0013333558146428443f),
                                  _mm512_set1_ps(0.009618129107628477f));
    p = _mm512_fmadd_ps(f, p, _mm512_set1_ps(0.05550410866482158f));
    p = _mm512_fmadd_ps(f, p, _mm512_set1_ps(0.2402265069591007f));
    p = _mm512_fmadd_ps(f, p, _mm512_set1_ps(0.6931471805599453f));
    p = _mm512_fmadd_ps(f, p, _mm512_set1_ps(1.0f));
    __m512i ni = _mm512_cvttps_epi32(n);      /* n is integer-valued */
    __m512i bits = _mm512_slli_epi32(ni, 23);
    return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(p), bits));
}

/* erf(x) = sign(x) * (1 - poly(t)*exp(-x^2)), t = 1/(1 + p*|x|), A&S 7.1.26 */
static inline __m512 uopc_erf_fast512(__m512 x) {
    __m512 ax = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), x);        /* |x| */
    __m512 t = _mm512_div_ps(_mm512_set1_ps(1.0f),
                             _mm512_fmadd_ps(_mm512_set1_ps(0.3275911f), ax,
                                             _mm512_set1_ps(1.0f)));
    __m512 poly = _mm512_fmadd_ps(t, _mm512_set1_ps(1.061405429f),
                                     _mm512_set1_ps(-1.453152027f));
    poly = _mm512_fmadd_ps(t, poly, _mm512_set1_ps(1.421413741f));
    poly = _mm512_fmadd_ps(t, poly, _mm512_set1_ps(-0.284496736f));
    poly = _mm512_fmadd_ps(t, poly, _mm512_set1_ps(0.254829592f));
    poly = _mm512_mul_ps(poly, t);
    __m512 e = uopc_expf_fast512(_mm512_mul_ps(ax, _mm512_mul_ps(ax,
                                                    _mm512_set1_ps(-1.0f))));
    __m512 erf_abs = _mm512_fnmadd_ps(poly, e, _mm512_set1_ps(1.0f));
    return _mm512_xor_ps(erf_abs, _mm512_and_ps(x, _mm512_set1_ps(-0.0f)));
}

/* gelu'(x) = 0.5*(1+erf(x/sqrt2)) + x*exp(-x^2/2)/sqrt(2pi) */
static inline __m512 uopc_gelu_deriv512(__m512 x) {
    __m512 e = uopc_erf_fast512(_mm512_mul_ps(x, _mm512_set1_ps(0.7071067811865476f)));
    __m512 half = _mm512_mul_ps(_mm512_add_ps(e, _mm512_set1_ps(1.0f)),
                                _mm512_set1_ps(0.5f));
    __m512 g = uopc_expf_fast512(_mm512_mul_ps(x, _mm512_mul_ps(x,
                                                 _mm512_set1_ps(-0.5f))));
    g = _mm512_mul_ps(_mm512_mul_ps(x, g), _mm512_set1_ps(0.3989422804014327f));
    return _mm512_add_ps(half, g);
}

/* forward GELU, both halves from ONE erf/exp pair.
 *   gelu (z)  = 0.5*z*(1 + erf(z/sqrt2))
 *   gelu'(z)  = 0.5*(1 + erf(z/sqrt2)) + z*exp(-z^2/2)/sqrt(2pi)
 * erf's own A&S 7.1.26 evaluation already forms exp(-(z/sqrt2)^2) = exp(-z^2/2),
 * which is exactly the term gelu' needs, so the two results cost one exp and
 * one reciprocal between them -- the same collapse lib/aurora/uopc/gpt.py's
 * scalar as_erf_ex() makes, expressed on a 16-wide register tile. */
static inline void uopc_gelu_fwd512(__m512 z, __m512 *out, __m512 *deriv) {
    __m512 x = _mm512_mul_ps(z, _mm512_set1_ps(0.7071067811865476f));
    __m512 ax = _mm512_andnot_ps(_mm512_set1_ps(-0.0f), x);
    __m512 t = _mm512_div_ps(_mm512_set1_ps(1.0f),
                             _mm512_fmadd_ps(_mm512_set1_ps(0.3275911f), ax,
                                             _mm512_set1_ps(1.0f)));
    __m512 poly = _mm512_fmadd_ps(t, _mm512_set1_ps(1.061405429f),
                                     _mm512_set1_ps(-1.453152027f));
    poly = _mm512_fmadd_ps(t, poly, _mm512_set1_ps(1.421413741f));
    poly = _mm512_fmadd_ps(t, poly, _mm512_set1_ps(-0.284496736f));
    poly = _mm512_fmadd_ps(t, poly, _mm512_set1_ps(0.254829592f));
    poly = _mm512_mul_ps(poly, t);
    __m512 e = uopc_expf_fast512(_mm512_mul_ps(ax, _mm512_mul_ps(ax,
                                                    _mm512_set1_ps(-1.0f))));
    __m512 erf_abs = _mm512_fnmadd_ps(poly, e, _mm512_set1_ps(1.0f));
    __m512 er = _mm512_xor_ps(erf_abs, _mm512_and_ps(x, _mm512_set1_ps(-0.0f)));
    __m512 half = _mm512_mul_ps(_mm512_add_ps(er, _mm512_set1_ps(1.0f)),
                                _mm512_set1_ps(0.5f));
    *out = _mm512_mul_ps(z, half);
    *deriv = _mm512_fmadd_ps(_mm512_mul_ps(z, e),
                             _mm512_set1_ps(0.3989422804014327f), half);
}

static inline __m512 uopc_epi512(__m512 c, const float *e, int kind, float s) {
    if (kind == UOPC_EPI_ADD)
        return _mm512_add_ps(c, _mm512_loadu_ps(e));
    if (kind == UOPC_EPI_FMA) {
        /* separate mul + add (NOT fused) so C + s*E is bit-identical with the
         * unfused EW: round(s*E) then round(C + round(s*E)) */
        __m512 t = _mm512_mul_ps(_mm512_set1_ps(s), _mm512_loadu_ps(e));
        return _mm512_add_ps(c, t);
    }
    return _mm512_mul_ps(c, uopc_gelu_deriv512(_mm512_loadu_ps(e)));
}

/* store site: kind 4 writes TWO streams (C and E), every other kind one. */
static inline void uopc_epi_store512(__m512 c, const float *e, int kind,
                                     float s, float *cp) {
    if (kind == UOPC_EPI_GELU_FWD) {
        __m512 o, g;
        uopc_gelu_fwd512(c, &o, &g);
        _mm512_storeu_ps((float *)e, g);
        _mm512_storeu_ps(cp, o);
        return;
    }
    _mm512_storeu_ps(cp, uopc_epi512(c, e, kind, s));
}

static inline float uopc_gelu_deriv_scalar(float x) {
    __m512 v = uopc_gelu_deriv512(_mm512_set1_ps(x));
    return _mm512_cvtss_f32(v);
}

static inline void mic1_epi(int kc, const float *A, int lda,
                             const float *B, int ldb, float *C, int ldc,
                            int kind, const float *E, int lde, float s) {
    __m512 c00 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps();
    __m512 c60 = _mm512_setzero_ps();
    __m512 c70 = _mm512_setzero_ps();
    const float *a0 = A + (size_t)0 * lda;
    const float *a1 = A + (size_t)1 * lda;
    const float *a2 = A + (size_t)2 * lda;
    const float *a3 = A + (size_t)3 * lda;
    const float *a4 = A + (size_t)4 * lda;
    const float *a5 = A + (size_t)5 * lda;
    const float *a6 = A + (size_t)6 * lda;
    const float *a7 = A + (size_t)7 * lda;
    int p = 0;
    for (; p + 2 <= kc; p += 2) {
        const float *bp0 = B + (size_t)(p + 0) * ldb;
        __m512 b00 = _mm512_loadu_ps(bp0 + 0 + 0*16);
        const float *bp1 = B + (size_t)(p + 1) * ldb;
        __m512 b10 = _mm512_loadu_ps(bp1 + 0 + 0*16);
        __m512 a;
        a = _mm512_set1_ps(a0[p + 0]);
        c00 = _mm512_fmadd_ps(a, b00, c00);
        a = _mm512_set1_ps(a1[p + 0]);
        c10 = _mm512_fmadd_ps(a, b00, c10);
        a = _mm512_set1_ps(a2[p + 0]);
        c20 = _mm512_fmadd_ps(a, b00, c20);
        a = _mm512_set1_ps(a3[p + 0]);
        c30 = _mm512_fmadd_ps(a, b00, c30);
        a = _mm512_set1_ps(a4[p + 0]);
        c40 = _mm512_fmadd_ps(a, b00, c40);
        a = _mm512_set1_ps(a5[p + 0]);
        c50 = _mm512_fmadd_ps(a, b00, c50);
        a = _mm512_set1_ps(a6[p + 0]);
        c60 = _mm512_fmadd_ps(a, b00, c60);
        a = _mm512_set1_ps(a7[p + 0]);
        c70 = _mm512_fmadd_ps(a, b00, c70);
        a = _mm512_set1_ps(a0[p + 1]);
        c00 = _mm512_fmadd_ps(a, b10, c00);
        a = _mm512_set1_ps(a1[p + 1]);
        c10 = _mm512_fmadd_ps(a, b10, c10);
        a = _mm512_set1_ps(a2[p + 1]);
        c20 = _mm512_fmadd_ps(a, b10, c20);
        a = _mm512_set1_ps(a3[p + 1]);
        c30 = _mm512_fmadd_ps(a, b10, c30);
        a = _mm512_set1_ps(a4[p + 1]);
        c40 = _mm512_fmadd_ps(a, b10, c40);
        a = _mm512_set1_ps(a5[p + 1]);
        c50 = _mm512_fmadd_ps(a, b10, c50);
        a = _mm512_set1_ps(a6[p + 1]);
        c60 = _mm512_fmadd_ps(a, b10, c60);
        a = _mm512_set1_ps(a7[p + 1]);
        c70 = _mm512_fmadd_ps(a, b10, c70);
    }
    for (; p < kc; p++) {
        const float *bp = B + (size_t)p * ldb;
        __m512 b0 = _mm512_loadu_ps(bp + 0 + 0*16);
        __m512 a;
        a = _mm512_set1_ps(a0[p]);
        c00 = _mm512_fmadd_ps(a, b0, c00);
        a = _mm512_set1_ps(a1[p]);
        c10 = _mm512_fmadd_ps(a, b0, c10);
        a = _mm512_set1_ps(a2[p]);
        c20 = _mm512_fmadd_ps(a, b0, c20);
        a = _mm512_set1_ps(a3[p]);
        c30 = _mm512_fmadd_ps(a, b0, c30);
        a = _mm512_set1_ps(a4[p]);
        c40 = _mm512_fmadd_ps(a, b0, c40);
        a = _mm512_set1_ps(a5[p]);
        c50 = _mm512_fmadd_ps(a, b0, c50);
        a = _mm512_set1_ps(a6[p]);
        c60 = _mm512_fmadd_ps(a, b0, c60);
        a = _mm512_set1_ps(a7[p]);
        c70 = _mm512_fmadd_ps(a, b0, c70);
    }
    float *c0 = C + (size_t)0 * ldc;
    uopc_epi_store512(c00, E + (size_t)0 * lde + 0*16, kind, s, c0 + 0*16);
    float *c1 = C + (size_t)1 * ldc;
    uopc_epi_store512(c10, E + (size_t)1 * lde + 0*16, kind, s, c1 + 0*16);
    float *c2 = C + (size_t)2 * ldc;
    uopc_epi_store512(c20, E + (size_t)2 * lde + 0*16, kind, s, c2 + 0*16);
    float *c3 = C + (size_t)3 * ldc;
    uopc_epi_store512(c30, E + (size_t)3 * lde + 0*16, kind, s, c3 + 0*16);
    float *c4 = C + (size_t)4 * ldc;
    uopc_epi_store512(c40, E + (size_t)4 * lde + 0*16, kind, s, c4 + 0*16);
    float *c5 = C + (size_t)5 * ldc;
    uopc_epi_store512(c50, E + (size_t)5 * lde + 0*16, kind, s, c5 + 0*16);
    float *c6 = C + (size_t)6 * ldc;
    uopc_epi_store512(c60, E + (size_t)6 * lde + 0*16, kind, s, c6 + 0*16);
    float *c7 = C + (size_t)7 * ldc;
    uopc_epi_store512(c70, E + (size_t)7 * lde + 0*16, kind, s, c7 + 0*16);
}

static inline void d2_block16_epi(int M, int K, const float *A, int lda,
                                    const float *B, int ldb, float *C, int ldc,
                                    int j0, int kind, const float *E, int lde, float s) {
    /* In-place fold (C == B, e.g. the muon momentum update m = m + A@m):
     * every 8-row block re-reads ALL of B during its k-loop, so a store
     * into B would clobber rows later blocks still need.  Stage the 16-col
     * B block (K*16 floats, K <= 256 here) once and read from the copy
     * (row stride 16 inside the block, not ldb). */
    float btmp[16 * 256];
    const float *Buse = B; int bldb = ldb;
    const float *Euse = E; int elde = lde;
    const float *Bj = B + j0;
    if (C == B) {
        for (int k = 0; k < K; k++)
            memcpy(btmp + (size_t)k * 16, B + (size_t)k * ldb + j0, 16 * sizeof(float));
        Buse = btmp; bldb = 16; Bj = btmp;
        if (E == B) { Euse = btmp; elde = 16; }
    }
    int mfull = M & ~7;
    for (int i = 0; i < mfull; i += 8) {
        const float *a = A + (size_t)i * lda;
        float *c = C + (size_t)i * ldc + j0;
        const float *e = Euse + (size_t)i * elde + (Euse == btmp ? 0 : j0);
        mic1_epi(K, a, lda, Bj, bldb, c, ldc, kind, e, elde, s);
    }
}


/* apply an epilogue to an already-computed C (packed-path fallback) */
void sgemm_epilogue_apply(int M, int N, int kind, float *C, int ldc,
                          const float *E, int lde, float s) {
    for (int i = 0; i < M; i++) {
        const float *er = E + (size_t)i * lde;
        float *cr = C + (size_t)i * ldc;
        if (kind == UOPC_EPI_ADD) {
            for (int j = 0; j < N; j++) cr[j] += er[j];
        } else if (kind == UOPC_EPI_GELU) {
            for (int j = 0; j < N; j++) cr[j] *= uopc_gelu_deriv_scalar(er[j]);
        } else if (kind == UOPC_EPI_FMA) {
            for (int j = 0; j < N; j++) cr[j] += s * er[j];
        } else if (kind == UOPC_EPI_GELU_FWD) {
            /* E is the gelu'(z) OUTPUT here, so the row pointer is written,
             * not read.  Same 16-wide math as the fused store, one lane at a
             * time, so the two paths agree bit-for-bit. */
            float *gr = (float *)er;
            for (int j = 0; j < N; j++) {
                __m512 o, g;
                uopc_gelu_fwd512(_mm512_set1_ps(cr[j]), &o, &g);
                cr[j] = _mm512_cvtss_f32(o);
                gr[j] = _mm512_cvtss_f32(g);
            }
        }
    }
}

void sgemm_direct2(int M, int N, int K, const float *A, int lda,
                   const float *B, int ldb, float *C, int ldc);

void sgemm_direct2_epi(int M, int N, int K, const float *A, int lda,
                       const float *B, int ldb, float *C, int ldc,
                       int kind, const float *E, int lde, float s) {
    if (kind == 0) { sgemm_direct2(M, N, K, A, lda, B, ldb, C, ldc); return; }
    /* fast fused path: full 16-col blocks and 8-row blocks only, K bounded so
     * the in-place B staging fits the stack (this covers every 34k-graph shape
     * the fusion triggers on; the store-time epilogue runs on the register
     * tiles before the raw C ever spills to L2/DRAM). */
    if (((N & 15) == 0) && ((M & 7) == 0) && K <= 256) {
        /* same loop-order rule as sgemm_direct2 (see D2_ROWOUT_BMAX), minus the
         * in-place case, whose B staging is per column block.  The fused fc GEMM
         * is 6144x256x64, i.e. B = 64 KB, squarely in the row-outer regime. */
        if (C != B && N > 16 && (size_t)K * (size_t)N <= D2_ROWOUT_BMAX) {
            for (int i = 0; i < M; i += 8) {
                const float *a = A + (size_t)i * lda;
                float *c = C + (size_t)i * ldc;
                const float *e = E + (size_t)i * lde;
                for (int j = 0; j < N; j += 16)
                    mic1_epi(K, a, lda, B + j, ldb, c + j, ldc, kind, e + j, lde, s);
            }
            return;
        }
        for (int j = 0; j < N; j += 16)
            d2_block16_epi(M, K, A, lda, B, ldb, C, ldc, j, kind, E, lde, s);
        return;
    }
    /* fallback (correct for every shape, just not register-fused): stage B
     * when the store target aliases it, then plain gemm + epilogue. */
    int inplace = (C == B);
    const float *Buse = B, *Euse = E;
    float *Bcopy = NULL;
    if (inplace) {
        Bcopy = malloc((size_t)K * N * sizeof(float));
        memcpy(Bcopy, B, (size_t)K * N * sizeof(float));
        Buse = Bcopy; ldb = N;          /* staging is row-major N-wide */
        if (E == B) { Euse = Bcopy; lde = N; }
    }
    sgemm_direct2(M, N, K, A, lda, Buse, ldb, C, ldc);
    sgemm_epilogue_apply(M, N, kind, C, ldc, Euse, lde, s);
    free(Bcopy);
}


/* Loop order.  The two orders differ only in which operand they keep resident:
 *   column-outer (d2_block16 over j, rows inside): the B panel K x 16 stays in
 *     L1 while every row of A streams past it.  A is re-read once per column
 *     block, and C is walked top-to-bottom once per column block -- M row
 *     visits with a stride of ldc floats, so at ldc >= 256 every 4th visit is a
 *     new 4 KB page.
 *   row-outer (rows outside, columns inside): the A panel 8 x K stays in L1
 *     while all of B streams past it.  B is re-read once per row block, and
 *     each 8-row band of C is finished before the next is touched.
 * So the right order is decided by which operand is small: row-outer re-reads
 * B, which is K*N floats, and that is affordable exactly when B fits in cache.
 * Measured at 1 thread, medians of 15 round-robin interleaved rounds (so drift
 * hits both arms), col-outer -> row-outer:
 *     256x256x32  (attention scores/dattn, x48)  2.69 -> 1.79 ms   1.50x
 *     6144x64x256 (fc2)                          3.96 -> 3.08 ms   1.29x
 *     6144x256x64 (fc)                           3.20 -> 2.84 ms   1.13x
 *     6144x192x64 (qkv)                          2.26 -> 2.08 ms   1.09x
 *     64x256x6144 (grad_fc2w, B = 6 MB)          1.60 -> 1.61 ms   none
 * The last line is the boundary case the threshold protects: at K=6144 the B
 * panel is megabytes and re-reading it per row block buys nothing. */
void sgemm_direct2(int M, int N, int K, const float *A, int lda,
                   const float *B, int ldb, float *C, int ldc) {
    if (N == 8) { /* thin GEMM: 8-wide kernel beats the masked 16-wide tail */
        d2_block8(M, K, A, lda, B, ldb, C, ldc);
        return;
    }
    int nfull = N & ~15; /* columns covered by complete 16-col blocks */
    if (M >= 8 && nfull > 16 && (size_t)K * (size_t)N <= D2_ROWOUT_BMAX) {
        int mfull = M & ~7;
        for (int i = 0; i < mfull; i += 8) {
            const float *a = A + (size_t)i * lda;
            float *c = C + (size_t)i * ldc;
            for (int j = 0; j < nfull; j += 16)
                mic1(K, a, lda, B + j, ldb, c + j, ldc);
        }
        if (M != mfull) {          /* M % 8 != 0: overlap-recompute the last 8
                                    * rows, exactly as d2_block16 does */
            const float *a = A + (size_t)(M - 8) * lda;
            float *c = C + (size_t)(M - 8) * ldc;
            for (int j = 0; j < nfull; j += 16)
                mic1(K, a, lda, B + j, ldb, c + j, ldc);
        }
        if (N != nfull)
            d2_tail(M, K, A, lda, B, ldb, C, ldc, nfull, N - nfull);
        return;
    }
    /* Column-outer: A is the streamed operand, so a wider register tile halves
     * the number of passes over it.  mic2's 8x32 tile is measured better exactly
     * where B is big (the transposed gradient GEMMs, K = 6144), medians of 11
     * interleaved rounds: 64x256x6144 1.60 -> 1.41 ms, 256x64x6144 1.60 -> 1.39,
     * 192x64x6144 1.20 -> 1.03.  At small K it is worse (6144x256x64 3.20 ->
     * 3.50), but small K is the row-outer regime handled above. */
    int j = 0;
    for (; j + 32 <= nfull; j += 32)
        d2_block32(M, K, A, lda, B, ldb, C, ldc, j);
    for (; j < nfull; j += 16)
        d2_block16(M, K, A, lda, B, ldb, C, ldc, j);
    if (N != nfull) {
        /* N % 16 != 0: compute only the remaining nb = N - nfull columns
         * with a narrow K-vectorized tail kernel, instead of paying for a
         * full 16-column block (masked or overlapping). */
        d2_tail(M, K, A, lda, B, ldb, C, ldc, nfull, N - nfull);
    }
}
