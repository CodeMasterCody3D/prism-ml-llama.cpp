// Golden-vector test for the forge block-Hadamard rotation.
//
// The failure mode this exists for: a half-applied rotation, a double-applied
// rotation, and a sequency-ordered (Walsh) transform all land on rel err ~1.408
// against the truth — indistinguishable from each other by any downstream
// metric. So the rotation is pinned here with EXACT integer goldens, before a
// single model tensor is involved:
//   - the x[i]=i+1 vector at b=256 produces only exact powers of two, so the
//     asserts are bit-equality, not tolerances. Sequency order is a permutation
//     and moves y[1]=-8 elsewhere -> instant failure.
//   - R(e0)[0] is the one probe that separates x1 vs x1/sqrt(b) vs x1/b
//     normalization.
//   - the 5120/128 and 17408/256 full-width checks prove blocks transform
//     independently (the ffn_down case is 256 and must not be skipped).
// The graph-side checks run the SHIPPED generator (llama_gen_hadamard) and the
// SHIPPED wrapper (llama_mul_mat_hadamard) through a real ggml graph, so what
// is asserted here is what the model will actually execute.

#include "../src/llama-impl.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do { \
        if (!(cond)) { g_fail++; printf("FAIL %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
    } while (0)

// natural-order (Sylvester) FWHT with the 1/sqrt(block) normalization —
// matches rotation.apply_rotation in the forge, verified against float64.
static void fwht_ref(float * v, int n, int block) {
    for (int off = 0; off < n; off += block) {
        float * p = v + off;
        for (int h = 1; h < block; h <<= 1) {
            for (int i = 0; i < block; i += 2*h) {
                for (int j = i; j < i + h; ++j) {
                    const float a = p[j], b = p[j+h];
                    p[j] = a + b; p[j+h] = a - b;
                }
            }
        }
        const float s = 1.0f / sqrtf((float) block);
        for (int i = 0; i < block; ++i) p[i] *= s;
    }
}

// deterministic values in [-1, 1) — no rand(), no seed drift between runs
static float lcg_next(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return (float) ((state >> 8) / 8388608.0 - 1.0);
}

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    ggml_cplan plan = ggml_graph_plan(graph, n_threads, nullptr);
    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }
    ggml_graph_compute(graph, &plan);
}

// run x [n, cols] through the SHIPPED wrapper with a SHIPPED-generator H [b, b]
static std::vector<float> graph_rotate(const std::vector<float> & x, int64_t n, int64_t cols, int b) {
    ggml_init_params params = { 256u*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * H = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b, b);
    llama_gen_hadamard(H);

    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, cols);
    memcpy(t->data, x.data(), x.size()*sizeof(float));

    ggml_tensor * y = llama_mul_mat_hadamard(ctx, t, H);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    std::vector<uint8_t> work;
    ggml_graph_compute_helper(work, gf, 4);

    std::vector<float> out(n*cols);
    memcpy(out.data(), y->data, out.size()*sizeof(float));
    ggml_free(ctx);
    return out;
}

int main(void) {
    // ---------- A: b=256, x[i]=i+1 — EXACT integer goldens, bit equality ----
    {
        const int b = 256;
        std::vector<float> v(b);
        for (int i = 0; i < b; i++) v[i] = (float)(i + 1);
        fwht_ref(v.data(), b, b);
        struct { int idx; float want; } gold[] = {
            {0, 2056.0f}, {1, -8.0f}, {2, -16.0f}, {4, -32.0f}, {8, -64.0f},
            {16, -128.0f}, {32, -256.0f}, {64, -512.0f}, {128, -1024.0f},
        };
        bool is_gold[256] = {false};
        for (auto & g : gold) {
            is_gold[g.idx] = true;
            CHECK(v[g.idx] == g.want, "A: y[%d] = %.9g, want %.9g exactly", g.idx, v[g.idx], g.want);
        }
        for (int i = 0; i < b; i++) {
            if (!is_gold[i]) CHECK(v[i] == 0.0f, "A: y[%d] = %.9g, want exact 0", i, v[i]);
        }
        printf("A  b=256 exact-integer goldens          %s\n", g_fail ? "FAIL" : "ok");
    }

    // ---------- B: b=128 — separates ORDERING failure from NORMALIZATION ----
    {
        const int b = 128; const int before = g_fail;
        std::vector<float> v(b);
        for (int i = 0; i < b; i++) v[i] = (float)(i + 1);
        fwht_ref(v.data(), b, b);
        const float s = sqrtf(128.0f);
        struct { int idx; float want; } gold[] = {
            {0, 8256.0f}, {1, -64.0f}, {2, -128.0f}, {4, -256.0f},
            {8, -512.0f}, {16, -1024.0f}, {32, -2048.0f}, {64, -4096.0f},
        };
        bool is_gold[128] = {false};
        for (auto & g : gold) {
            is_gold[g.idx] = true;
            CHECK(fabsf(v[g.idx]*s - g.want) < 1e-3f*fabsf(g.want) + 1e-3f,
                  "B: y[%d]*sqrt(128) = %.6g, want %.6g", g.idx, v[g.idx]*s, g.want);
        }
        for (int i = 0; i < b; i++) {
            if (!is_gold[i]) CHECK(fabsf(v[i]*s) < 1e-3f, "B: y[%d]*sqrt(128) = %.6g, want ~0", i, v[i]*s);
        }
        printf("B  b=128 ordering vs normalization      %s\n", g_fail > before ? "FAIL" : "ok");
    }

    // ---------- C: R(e0) — the ONLY probe separating x1 / x1/sqrt(b) / x1/b --
    {
        const int before = g_fail;
        {   // b=128
            std::vector<float> v(128, 0.0f); v[0] = 1.0f;
            fwht_ref(v.data(), 128, 128);
            CHECK(fabsf(v[0] - 0.08838834764831845f) < 1e-9f, "C: R(e0)[0] b=128 = %.12g", v[0]);
        }
        {   // b=256 — 1/16 is exact in binary, assert bit equality
            std::vector<float> v(256, 0.0f); v[0] = 1.0f;
            fwht_ref(v.data(), 256, 256);
            CHECK(v[0] == 0.0625f, "C: R(e0)[0] b=256 = %.12g, want exactly 0.0625", v[0]);
        }
        printf("C  R(e0) normalization probe            %s\n", g_fail > before ? "FAIL" : "ok");
    }

    // ---------- D: involution R(R(x)) == x — catches butterfly sign errors ---
    {
        const int before = g_fail;
        uint32_t st = 42;
        std::vector<float> x(512), v(512);
        for (auto & f : x) f = lcg_next(st);
        v = x;
        fwht_ref(v.data(), 512, 128);
        fwht_ref(v.data(), 512, 128);
        for (int i = 0; i < 512; i++) {
            CHECK(fabsf(v[i] - x[i]) <= 1e-6f * (1.0f + fabsf(x[i])), "D: R(R(x))[%d] drifted", i);
        }
        printf("D  involution                           %s\n", g_fail > before ? "FAIL" : "ok");
    }

    // ---------- E+F+G: the SHIPPED generator + wrapper through a real graph --
    // n=5120 b=128 (body width) and n=17408 b=256 (the ffn_down case).
    // Reference = per-block fwht_ref on the same data; agreement proves both
    // that llama_gen_hadamard is the same basis AND that blocks transform
    // independently on the path the model will actually run.
    {
        struct { int64_t n; int b; int64_t cols; } cases[] = {
            { 128,   128, 1 },   // single block, the direct comparison
            { 256,   256, 1 },
            { 5120,  128, 3 },   // full body width, multiple tokens
            { 17408, 256, 2 },   // ffn_down — do not skip
        };
        for (auto & c : cases) {
            const int before = g_fail;
            uint32_t st = 1234 + (uint32_t)c.n;
            std::vector<float> x(c.n * c.cols);
            for (auto & f : x) f = lcg_next(st);

            std::vector<float> ref = x;
            for (int64_t col = 0; col < c.cols; col++) {
                fwht_ref(ref.data() + col*c.n, (int)c.n, c.b);
            }
            std::vector<float> got = graph_rotate(x, c.n, c.cols, c.b);

            double max_rel = 0.0;
            for (size_t i = 0; i < ref.size(); i++) {
                const double rel = fabs((double)got[i] - ref[i]) / (1.0 + fabs((double)ref[i]));
                if (rel > max_rel) max_rel = rel;
            }
            CHECK(max_rel < 1e-5, "G: n=%" PRId64 " b=%d cols=%" PRId64 " max rel %.3g",
                  c.n, c.b, c.cols, max_rel);
            printf("G  shipped H + wrapper  n=%-6" PRId64 " b=%-4d %s (max rel %.2e)\n",
                   c.n, c.b, g_fail > before ? "FAIL" : "ok", max_rel);
        }
    }

    printf("%s\n", g_fail ? "========== FAILED ==========" : "========== ALL PASS ==========");
    return g_fail != 0;
}
