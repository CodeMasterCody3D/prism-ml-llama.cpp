#pragma once

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include "traits.h"
#include "ggml.h"

// GGML internal header

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

template <int K> constexpr int QK_0() {
    if constexpr (K == 4) {
        return QK4_0;
    }
    if constexpr (K == 8) {
        return QK8_0;
    }
    return -1;
}

template <int K, int N> struct block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

// control size
static_assert(sizeof(block<4, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 2, "wrong block<4,4> size/padding");
static_assert(sizeof(block<4, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<4,8> size/padding");
static_assert(sizeof(block<4, 16>) == 16 * sizeof(ggml_half) + QK8_0 * 8, "wrong block<4,16> size/padding");
static_assert(sizeof(block<8, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<8,4> size/padding");
static_assert(sizeof(block<8, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 8, "wrong block<8,8> size/padding");
static_assert(sizeof(block<8, 16>) == 16 * sizeof(ggml_half) + QK8_0 * 16, "wrong block<8,16> size/padding");

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q4_0x16 = block<4, 16>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;
using block_q8_0x16 = block<8, 16>;

struct block_q4_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[96];  // scales and mins, quantized with 6 bits
    uint8_t qs[1024];    // 4--bit quants
};

static_assert(sizeof(block_q4_Kx8) == sizeof(ggml_half) * 16 + K_SCALE_SIZE * 8 + QK_K * 4, "wrong q4_K block size/padding");
struct block_q4_Kx16 {
    ggml_half d[16];      // super-block scale for quantized scales
    ggml_half dmin[16];   // super-block scale for quantized mins
    uint8_t scales[192];  // scales and mins, quantized with 6 bits
    uint8_t qs[2048];    // 4--bit quants
};

static_assert(sizeof(block_q4_Kx16) == sizeof(ggml_half) * 32 + K_SCALE_SIZE * 16 + QK_K * 8, "wrong q4_K block size/padding");
struct block_q2_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[128];  // scales and mins, quantized with 4 bits
    uint8_t qs[512];    // 2--bit quants
};

static_assert(sizeof(block_q2_Kx8) == sizeof(ggml_half) * 16 + QK_K/2 + QK_K * 2, "wrong q2_K block size/padding");
struct block_q2_Kx16 {
    ggml_half d[16];       // Super-block scale for quantized scales
    ggml_half dmin[16];    // Super-block scale for quantized mins
    uint8_t   scales[256]; // Sub-block scales (16 cols * 16 sub-blocks)
    uint8_t   qs[1024];    // Data (16 cols * 64 bytes per block)
};
static_assert(sizeof(block_q2_Kx16) == sizeof(ggml_half) * 32 + QK_K + QK_K * 4, "wrong q2_K block size/padding");

struct block_q5_Kx8 {
    ggml_half d[8];              // super-block scale for quantized scales
    ggml_half dmin[8];           // super-block scale for quantized mins
    uint8_t   scales[96];        // scales and mins, quantized with 6 bits
    uint8_t   qh[QK_K * 8 / 8];  // high bits of 5-bit quants
    uint8_t   qs[QK_K * 8 / 2];  // low bits of 5-bit quants (in groups of 4)
};

static_assert(sizeof(block_q5_Kx8) == sizeof(ggml_half) * 16 + K_SCALE_SIZE * 8 + QK_K * 5,
              "wrong q5_K block size/padding");

struct block_q6_Kx8 {
    ggml_half d[8];
    int8_t    scales[QK_K / 16 * 8];
    uint8_t   ql[QK_K / 2 * 8];  // low bits of 6-bit quants (groups of 2)
    uint8_t   qh[QK_K / 4 * 8];  // high bits of 6-bit quants (groups of 4)
};

static_assert(sizeof(block_q6_Kx8) == sizeof(ggml_half) * 8 + QK_K / 16 * 8 + 3 * QK_K / 4 * 8,
              "wrong q6_K block size/padding");

struct block_q8_Kx4 {
    float d[4];              // delta
    int8_t qs[QK_K * 4];     // quants
    int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

static_assert(sizeof(block_q8_Kx4) == sizeof(float) * 4 + QK_K * 4 + (QK_K / 4) * sizeof(int16_t), "wrong q8_K block size/padding");

struct block_iq4_nlx4 {
    ggml_half d[4];            // deltas for 4 iq4_nl blocks
    uint8_t   qs[QK4_NL * 2];  // nibbles / quants for 4 iq4_nl blocks
};

static_assert(sizeof(block_iq4_nlx4) == 4 * sizeof(ggml_half) + QK4_NL * 2, "wrong iq4_nlx4 block size/padding");

struct block_iq4_nlx8 {
    ggml_half d[8];            // deltas for 8 iq4_nl blocks
    uint8_t   qs[QK4_NL * 4];  // nibbles / quants for 8 iq4_nl blocks
};

static_assert(sizeof(block_iq4_nlx8) == 8 * sizeof(ggml_half) + QK4_NL * 4, "wrong iq4_nlx8 block size/padding");

struct block_iq4_nlx16 {
    ggml_half d[16];            // deltas for 16 iq4_nl blocks
    uint8_t   qs[QK4_NL * 8];  // nibbles / quants for 16 iq4_nl blocks
};

static_assert(sizeof(block_iq4_nlx16) == 16 * sizeof(ggml_half) + QK4_NL * 8, "wrong iq4_nlx16 block size/padding");
struct block_mxfp4x4 {
    uint8_t e[4];
    uint8_t qs[QK_MXFP4 * 2];
};
static_assert(sizeof(block_mxfp4x4) == 4 + QK_MXFP4 * 2, "wrong mxfp4x4 block size/padding");

struct block_mxfp4x8 {
    uint8_t e[8];
    uint8_t qs[QK_MXFP4 * 4];
};
static_assert(sizeof(block_mxfp4x8) == 8 + QK_MXFP4 * 4, "wrong mxfp4x8 block size/padding");

// 8 interleaved block_q1_t_g128 rows: 8 fp16 scales, then qs bytes
// TRANSPOSED so byte position is the outer index and row is the inner index
// (qs[byte_idx*8 + row] = row's original qs[byte_idx]) -- the same pure
// byte-transpose already used by block_q8_0x16's blck_size_interleave==1
// path, just at NB_COLS=8. This layout is what makes the GEMV-level 243
// -entry lookup table amortization possible: for a fixed byte position, the
// 8 interleaved rows' bytes are contiguous, so one activation-derived table
// (built once per 128-weight block, shared across every row) turns each
// row's contribution into a single table read instead of a 5-trit unpack.
struct block_q1_t_g128x8 {
    ggml_half d[8];
    uint8_t   qs[26 * 8];
};
static_assert(sizeof(block_q1_t_g128x8) == 8 * sizeof(ggml_half) + 26 * 8, "wrong q1_t_g128x8 block size/padding");

// ----------------------------------------------------------------------
// Q1_T_g128 GEMV/GEMM: 243-value lookup table amortised over output rows.
// Shared between the portable generic implementation (repack.cpp) and the
// AVX2-vectorised one (arch/x86/repack.cpp) -- header-inline so both TUs
// get the same segment table without duplicating it.
//
// A q1_t_g128 byte packs 5 trits in base 3 (byte = d0 + 3*d1 + 9*d2 +
// 27*d3 + 81*d4, each digit in {0,1,2} meaning trit-1), so a byte has only
// 3^5 = 243 possible values. For a FIXED 5-element window of Q8_0
// activations, the partial dot product sum_k (d_k-1)*y[off+k] therefore
// also has only 243 possible values -- build them ONCE per activation
// block and every row's byte becomes a single table lookup instead of a
// 5-trit unpack.
//
// A 128-weight block's 26 bytes cover 130 trit slots for 128 real weights
// (the last byte's final 2 slots are always-zero padding). Weights
// dot-product against FOUR separate 32-wide Q8_0 sub-blocks, each with its
// own activation scale, so a byte's 5-weight window is only safe to fold
// into ONE table lookup if the whole window falls inside a single
// 32-weight sub-block. Bytes 6, 12, and 19 straddle a sub-block boundary
// -- verified against the AVX2 vec_dot kernel's ggml_q1_t_g128_shuffle_0..3
// mask tables (arch/x86/quants.c): byte 6 digits 0-1 feed sub-block 0 and
// digits 2-4 feed sub-block 1; byte 12 digits 0-3 feed sub-block 1 and
// digit 4 feeds sub-block 2; byte 19 digit 0 feeds sub-block 2 and digits
// 1-4 feed sub-block 3. Each straddling byte is split into two segments,
// one per side of the boundary. Byte 25 has only 3 real digits (the other
// two are the always-zero padding). Every other byte is a single 5-digit
// segment. 26 bytes therefore decompose into 29 (byte, sub-block)
// segments -- generated by an exhaustive weight-index walk (weight =
// 5*byte + digit, sub-block = weight/32) and cross-checked against the
// shuffle masks above by hand for all three straddle cases.
struct q1t_gemv_segment {
    uint8_t byte_idx;     // 0..25, which qs[] byte this segment reads
    uint8_t subblock;     // 0..3, which of the four Q8_0 sub-blocks it feeds
    uint8_t local_offset; // 0..31, activation index of the segment's first digit
    uint8_t digit_lo;     // 0..4, first digit (inclusive) this segment covers
    uint8_t digit_hi;     // 1..5, last digit (EXCLUSIVE) this segment covers
};

#define Q1T_GEMV_NUM_SEGMENTS 29

static const q1t_gemv_segment q1t_gemv_segments[Q1T_GEMV_NUM_SEGMENTS] = {
    {  0, 0,  0, 0, 5}, {  1, 0,  5, 0, 5}, {  2, 0, 10, 0, 5},
    {  3, 0, 15, 0, 5}, {  4, 0, 20, 0, 5}, {  5, 0, 25, 0, 5},
    {  6, 0, 30, 0, 2}, {  6, 1,  0, 2, 5},
    {  7, 1,  3, 0, 5}, {  8, 1,  8, 0, 5}, {  9, 1, 13, 0, 5},
    { 10, 1, 18, 0, 5}, { 11, 1, 23, 0, 5},
    { 12, 1, 28, 0, 4}, { 12, 2,  0, 4, 5},
    { 13, 2,  1, 0, 5}, { 14, 2,  6, 0, 5}, { 15, 2, 11, 0, 5},
    { 16, 2, 16, 0, 5}, { 17, 2, 21, 0, 5}, { 18, 2, 26, 0, 5},
    { 19, 2, 31, 0, 1}, { 19, 3,  0, 1, 5},
    { 20, 3,  4, 0, 5}, { 21, 3,  9, 0, 5}, { 22, 3, 14, 0, 5},
    { 23, 3, 19, 0, 5}, { 24, 3, 24, 0, 5},
    { 25, 3, 29, 0, 3},
};

// Incremental base-3 table build: the d4 partial sum is computed once and
// reused by all 81 entries beneath it, d3's once per 27, ... down to d0
// which is added once per final entry -- 3+9+27+81+243 = 363 additions
// total to fill all 243 entries, instead of 243*5 independent 5-term sums.
// `yvals[k - digit_lo]` is the activation value for digit k, so `yvals`
// itself is just `&sub_block.qs[local_offset]`. int32_t (not int16_t)
// entries so the AVX2 path can feed a table row straight into
// _mm256_set_epi32 without a widening step.
//
// Split into a branch-free fast path for the 23/29 "clean" segments
// (digit_lo=0, digit_hi=5, i.e. all 5 digits real) and a branchy slow path
// for the 6 partial ones (3 straddling bytes' two halves + byte 25's
// 3-digit tail). MEASURED reason this split exists: the single generic
// version (branches on digit_lo/digit_hi at every nesting level, even
// though they are loop-invariant) compiled to a genuinely bad branchy scalar
// mess -- confirmed via objdump and an isolated microbenchmark -- costing
// ~5.4us per 29-segment build at -O3 on this machine. Hoisting the common,
// branch-free case into its own function (letting the compiler see
// digit_lo/digit_hi as compile-time 0/5 rather than runtime parameters)
// cut that to ~2.0us, a 2.7x reduction, with ZERO change in output (both
// paths compute the identical incremental sum; only the slow path still
// needs the digit-range guards for the true partial segments).
static inline void q1t_build_segment_table_full(int32_t table[243], const int8_t * yvals) {
    for (int d4 = 0; d4 < 3; d4++) {
        const int s4 = (d4 - 1) * (int) yvals[4];
        for (int d3 = 0; d3 < 3; d3++) {
            const int s3 = s4 + (d3 - 1) * (int) yvals[3];
            for (int d2 = 0; d2 < 3; d2++) {
                const int s2 = s3 + (d2 - 1) * (int) yvals[2];
                for (int d1 = 0; d1 < 3; d1++) {
                    const int s1 = s2 + (d1 - 1) * (int) yvals[1];
                    for (int d0 = 0; d0 < 3; d0++) {
                        table[d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4] = s1 + (d0 - 1) * (int) yvals[0];
                    }
                }
            }
        }
    }
}

static inline void q1t_build_segment_table_partial(int32_t table[243], int digit_lo, int digit_hi, const int8_t * yvals) {
    for (int d4 = 0; d4 < 3; d4++) {
        const int s4 = (digit_lo <= 4 && 4 < digit_hi) ? (d4 - 1) * (int) yvals[4 - digit_lo] : 0;
        for (int d3 = 0; d3 < 3; d3++) {
            const int s3 = s4 + ((digit_lo <= 3 && 3 < digit_hi) ? (d3 - 1) * (int) yvals[3 - digit_lo] : 0);
            for (int d2 = 0; d2 < 3; d2++) {
                const int s2 = s3 + ((digit_lo <= 2 && 2 < digit_hi) ? (d2 - 1) * (int) yvals[2 - digit_lo] : 0);
                for (int d1 = 0; d1 < 3; d1++) {
                    const int s1 = s2 + ((digit_lo <= 1 && 1 < digit_hi) ? (d1 - 1) * (int) yvals[1 - digit_lo] : 0);
                    for (int d0 = 0; d0 < 3; d0++) {
                        const int s0 = s1 + ((digit_lo <= 0 && 0 < digit_hi) ? (d0 - 1) * (int) yvals[0 - digit_lo] : 0);
                        const int byte = d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4;
                        table[byte] = s0;
                    }
                }
            }
        }
    }
}

// Build all 29 segment tables from the four Q8_0 sub-blocks belonging to
// one 128-weight Q1_T_g128 block. Called once per activation block and
// reused across every output row -- this is the whole amortisation.
static inline void q1t_build_all_tables(int32_t tables[Q1T_GEMV_NUM_SEGMENTS][243], const block_q8_0 y4[4]) {
    for (int s = 0; s < Q1T_GEMV_NUM_SEGMENTS; s++) {
        const q1t_gemv_segment * seg = &q1t_gemv_segments[s];
        const int8_t * yvals = &y4[seg->subblock].qs[seg->local_offset];
        if (seg->digit_lo == 0 && seg->digit_hi == 5) {
            q1t_build_segment_table_full(tables[s], yvals);
        } else {
            q1t_build_segment_table_partial(tables[s], seg->digit_lo, seg->digit_hi, yvals);
        }
    }
}
// ----------------------------------------------------------------------

#if defined(__cplusplus)
extern "C" {
#endif

void ggml_quantize_mat_q8_0_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemv_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q2_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q5_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q5_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q6_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q6_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_mxfp4_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_mxfp4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q2_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q5_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q5_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q6_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q6_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_mxfp4_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_mxfp4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q1_t_g128_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q1_t_g128_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
#if defined __riscv_zvfh
void ggml_quantize_mat_q8_0_4x1(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x1(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemv_q4_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q2_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q2_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
#endif

// Native implementations
void ggml_quantize_mat_q8_0_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_0_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemv_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q2_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q5_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q5_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q6_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q6_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_mxfp4_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_mxfp4_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q2_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q5_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q5_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q6_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q6_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_mxfp4_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_mxfp4_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q1_t_g128_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q1_t_g128_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
#if defined __riscv_zvfh
void ggml_quantize_mat_q8_0_4x1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemv_q4_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
#endif

#if defined(__cplusplus)
} // extern "C"
#endif
