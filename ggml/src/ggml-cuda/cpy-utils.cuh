#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

// Verbatim port of ggml_q1_0_g128_lloyd_scale (ggml-quants.c): 4 seeds x <=8
// threshold/mean iterations, keep the best residual-reduction objective.
static __device__ float q1_t_g128_lloyd_scale_dev(const float * __restrict__ x, const int qk, const float mean) {
    if (mean <= 0.0f) {
        return 0.0f;
    }
    const float inits[4] = {0.5f, 0.7f, 0.9f, 1.1f};
    float best_a   = mean;
    float best_obj = -1.0f;
    for (int ci = 0; ci < 4; ci++) {
        float th = inits[ci] * mean;
        float s1 = 0.0f; int sw = 0;
        for (int j = 0; j < qk; j++) {
            const float w = fabsf(x[j]);
            if (w > th) { s1 += w; sw++; }
        }
        float a = sw > 0 ? s1/sw : 0.0f;
        for (int it = 0; it < 8 && a > 0.0f; it++) {
            th = 0.5f * a;
            s1 = 0.0f; sw = 0;
            for (int j = 0; j < qk; j++) {
                const float w = fabsf(x[j]);
                if (w > th) { s1 += w; sw++; }
            }
            const float na = sw > 0 ? s1/sw : 0.0f;
            if (fabsf(na - a) <= 1e-6f * fmaxf(na, a)) { a = na; break; }
            a = na;
        }
        const float obj = sw > 0 ? s1*s1/(float)sw : 0.0f;
        if (obj > best_obj) { best_obj = obj; best_a = a; }
    }
    return best_a > 0.0f ? best_a : mean;
}

template <bool lloyd>
static __device__ void quantize_f32_q1_t_g128_block_t(const float * __restrict__ x, block_q1_t_g128 * __restrict__ y) {
    float sum_abs = 0.0f;
#pragma unroll
    for (int j = 0; j < QK1_T_g128; ++j) {
        sum_abs += fabsf(x[j]);
    }
    float d = sum_abs / QK1_T_g128;
    if (lloyd) {
        d = q1_t_g128_lloyd_scale_dev(x, QK1_T_g128, d);
    }
    y->d = __float2half(d);
    const float dq = __half2float(y->d);
    const float id = dq > 0.0f ? 1.0f/dq : 0.0f;

#pragma unroll
    for (int b = 0; b < 26; ++b) {
        int byte_val = 0;
        int mul      = 1;
#pragma unroll
        for (int s = 0; s < 5; ++s) {
            const int j = b*5 + s;
            int digit = 0;
            if (j < QK1_T_g128) {
                int q = (int) roundf(x[j] * id);
                q = q < -1 ? -1 : (q > 1 ? 1 : q);
                digit = q + 1;
            }
            byte_val += digit * mul;
            mul      *= 3;
        }
        y->qs[b] = (uint8_t) byte_val;
    }
}

// CPU parity: quantize_row_q1_t_g128_ref applies Lloyd unless Q1_0_G128_LLOYD=0.
static inline bool ggml_cuda_q1_t_g128_lloyd_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("Q1_0_G128_LLOYD");
        cached = (v && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}
static __device__ void quantize_f32_q1_t_g128_block_lloyd(const float * __restrict__ x, block_q1_t_g128 * __restrict__ y) {
    quantize_f32_q1_t_g128_block_t<true>(x, y);
}

static __device__ void quantize_f32_q1_t_g128_block(const float * __restrict__ x, block_q1_t_g128 * __restrict__ y) {
    quantize_f32_q1_t_g128_block_t<false>(x, y);
}

static __device__ void cpy_blck_f32_q1_t_g128_lloyd(const char * cxi, char * cdsti) {
    quantize_f32_q1_t_g128_block_lloyd((const float *)cxi, (block_q1_t_g128 *)cdsti);
}

static __device__ void cpy_blck_f32_q1_t_g128(const char * cxi, char * cdsti) {
    quantize_f32_q1_t_g128_block((const float *)cxi, (block_q1_t_g128 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
