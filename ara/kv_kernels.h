/* UAMM-2 KV kernels for a RISC-V Vector (RVV 1.0) target such as PULP Ara.
 *
 * Fused "QK on packed 4-bit keys":
 *   key element  K[t][c] = qv[t][c] * scale[c] + zero[c]      (KIVI per-channel keys)
 *   score        S[t]    = sum_c Q[c] * K[t][c]
 *                        = sum_c qv[t][c] * (Q[c]*scale[c])  +  sum_c Q[c]*zero[c]
 * The second term is the same for every token, so it is computed once ("bias") and the inner loop
 * is a pure integer->float widen + FMA over the packed nibbles. No dequantized K is materialized.
 *
 * Packing (must match uamm2/codecs.py pack_bits): row-major (T,d); element i of the flattened array
 * lives in byte i/2; even i in the LOW nibble, odd i in the HIGH nibble. d must be even.
 */
#ifndef UAMM2_KV_KERNELS_H
#define UAMM2_KV_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fills qs_even[d/2], qs_odd[d/2] with Q[c]*scale[c] for even/odd c; returns bias = sum Q[c]*zero[c]. */
float kv_qk_prepare(const float *q, const float *scale, const float *zero, int d,
                    float *qs_even, float *qs_odd);

/* Portable scalar reference. */
void kv_qk_dot_kivi4_scalar(const float *qs_even, const float *qs_odd, float bias,
                            const uint8_t *packed, int T, int d, float *out);

#if defined(__riscv_vector)
/* RVV 1.0 implementation (vector-length agnostic). */
void kv_qk_dot_kivi4_rvv(const float *qs_even, const float *qs_odd, float bias,
                         const uint8_t *packed, int T, int d, float *out);
#endif

/* Dispatches to RVV when compiled for it, otherwise to the scalar version. */
void kv_qk_dot_kivi4(const float *qs_even, const float *qs_odd, float bias,
                     const uint8_t *packed, int T, int d, float *out);

#ifdef __cplusplus
}
#endif
#endif
