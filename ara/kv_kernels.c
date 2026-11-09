#include "kv_kernels.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

float kv_qk_prepare(const float *q, const float *scale, const float *zero, int d,
                    float *qs_even, float *qs_odd) {
    float bias = 0.0f;
    for (int c = 0; c < d; c++) {
        bias += q[c] * zero[c];
        if ((c & 1) == 0) qs_even[c >> 1] = q[c] * scale[c];
        else              qs_odd[c >> 1]  = q[c] * scale[c];
    }
    return bias;
}

void kv_qk_dot_kivi4_scalar(const float *qs_even, const float *qs_odd, float bias,
                            const uint8_t *packed, int T, int d, float *out) {
    const int half = d / 2;
    for (int t = 0; t < T; t++) {
        const uint8_t *row = packed + (long)t * half;
        float acc = 0.0f;
        for (int j = 0; j < half; j++) {
            acc += (float)(row[j] & 0x0F) * qs_even[j];
            acc += (float)(row[j] >> 4)   * qs_odd[j];
        }
        out[t] = acc + bias;
    }
}

#if defined(__riscv_vector)
void kv_qk_dot_kivi4_rvv(const float *qs_even, const float *qs_odd, float bias,
                         const uint8_t *packed, int T, int d, float *out) {
    const int half = d / 2;
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    for (int t = 0; t < T; t++) {
        const uint8_t *row = packed + (long)t * half;
        vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        int j = 0;
        while (j < half) {
            size_t vl = __riscv_vsetvl_e32m1((size_t)(half - j));
            vuint8mf4_t b   = __riscv_vle8_v_u8mf4(row + j, vl);
            vuint8mf4_t lo8 = __riscv_vand_vx_u8mf4(b, 0x0F, vl);
            vuint8mf4_t hi8 = __riscv_vsrl_vx_u8mf4(b, 4, vl);
            vfloat32m1_t lo = __riscv_vfcvt_f_xu_v_f32m1(__riscv_vzext_vf4_u32m1(lo8, vl), vl);
            vfloat32m1_t hi = __riscv_vfcvt_f_xu_v_f32m1(__riscv_vzext_vf4_u32m1(hi8, vl), vl);
            vfloat32m1_t qe = __riscv_vle32_v_f32m1(qs_even + j, vl);
            vfloat32m1_t qo = __riscv_vle32_v_f32m1(qs_odd + j, vl);
            acc = __riscv_vfmacc_vv_f32m1_tu(acc, lo, qe, vl);   /* _tu: keep tail lanes of acc */
            acc = __riscv_vfmacc_vv_f32m1_tu(acc, hi, qo, vl);
            j += (int)vl;
        }
        vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vfloat32m1_t sum  = __riscv_vfredusum_vs_f32m1_f32m1(acc, zero, vlmax);
        out[t] = __riscv_vfmv_f_s_f32m1_f32(sum) + bias;
    }
}
#endif

void kv_qk_dot_kivi4(const float *qs_even, const float *qs_odd, float bias,
                     const uint8_t *packed, int T, int d, float *out) {
#if defined(__riscv_vector)
    kv_qk_dot_kivi4_rvv(qs_even, qs_odd, bias, packed, T, d, out);
#else
    kv_qk_dot_kivi4_scalar(qs_even, qs_odd, bias, packed, T, d, out);
#endif
}
