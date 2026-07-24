#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifndef GGML_SOURCE_DIR
#error "GGML_SOURCE_DIR must point at the source tree"
#endif

static std::string read_source(const char * relative) {
    const std::string path = std::string(GGML_SOURCE_DIR) + "/" + relative;
    std::ifstream in(path);
    assert(in && "shader source must exist");
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

static void require_contains(const std::string & text, const std::string & needle) {
    if (text.find(needle) == std::string::npos) {
        std::cerr << "missing required production expression: " << needle << '\n';
        std::abort();
    }
}

static void require_absent(const std::string & text, const std::string & needle) {
    if (text.find(needle) != std::string::npos) {
        std::cerr << "forbidden stale production expression: " << needle << '\n';
        std::abort();
    }
}

static void test_standalone_dequant_contract() {
    const std::string shader = read_source("ggml/src/ggml-vulkan/vulkan-shaders/dequant_tq.comp");
    const std::string host   = read_source("ggml/src/ggml-vulkan/ggml-vulkan.cpp");

    // ggml_vk_mul_mat_q_f16 pushes {M,K,stride_a,stride_b,nel}; the shader must read word 4.
    require_contains(shader, "#include \"dequant_head.glsl\"");
    require_absent(shader, "uint ne;");
    require_contains(shader, "p.nel");

    // 256 invocations x four scalar outputs = 1024 elements per workgroup.
    require_contains(host,
        "pipeline_dequant[GGML_TYPE_TQ2_0], \"dequant_tq2_0\", dequant_tq2_0_len, dequant_tq2_0_data, \"main\", 2, 5 * sizeof(uint32_t), {256 * 4, 1, 1}");
    require_contains(host,
        "pipeline_dequant[GGML_TYPE_TQ1_0], \"dequant_tq1_0\", dequant_tq1_0_len, dequant_tq1_0_data, \"main\", 2, 5 * sizeof(uint32_t), {256 * 4, 1, 1}");

    for (uint32_t nel : {256u, 1024u, 1280u, 4096u, 4352u}) {
        const uint32_t workgroups = (nel + 1023u) / 1024u;
        uint32_t covered = 0;
        for (uint32_t wg = 0; wg < workgroups; ++wg) {
            for (uint32_t lane = 0; lane < 256; ++lane) {
                const uint32_t i = (wg * 256u + lane) * 4u;
                for (uint32_t j = 0; j < 4 && i + j < nel; ++j) {
                    ++covered;
                }
            }
        }
        assert(covered == nel);
    }
}

static void test_cm2_tq1_signed_trit() {
    const std::string shader = read_source("ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs_cm2.glsl");
    require_contains(shader, "float16_t(int(val) - 1)");
    require_absent(shader, "float16_t(val - 1)");

    for (uint32_t val = 0; val <= 2; ++val) {
        const int decoded = int(val) - 1;
        assert(decoded >= -1 && decoded <= 1);
        assert(decoded == static_cast<int>(val) - 1);
    }
}

static void test_integer_dot_offsets() {
    const std::string shader = read_source("ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_tq_q.comp");
    require_contains(shader,
        "const uint ib0 = a_offset + (first_row + n) * num_blocks_per_row;");
    require_absent(shader,
        "const uint ib0 = a_offset / QUANT_K + (first_row + n) * num_blocks_per_row;");
    require_contains(shader,
        "const uint b0_block_idx = (b_offset + b_base + b0_idx) / QUANT_K_Q8_1;");
    require_contains(shader,
        "const uint b1_block_idx = (b_offset + b_base + b1_idx) / QUANT_K_Q8_1;");

    constexpr uint32_t quant_k = 256;
    constexpr uint32_t q8_k = 32;
    for (uint32_t batch_a : {0u, 1u, 3u}) {
        for (uint32_t batch_b : {0u, 1u, 5u}) {
            const uint32_t batch_stride_a = 7u * quant_k;
            const uint32_t batch_stride_b = 11u * q8_k;
            const uint32_t a_offset_blocks = batch_a * (batch_stride_a / quant_k);
            const uint32_t b_offset_elements = batch_b * batch_stride_b;
            for (uint32_t row : {0u, 2u, 9u}) {
                const uint32_t ib0 = a_offset_blocks + row * 7u;
                assert(ib0 == batch_a * 7u + row * 7u);
            }
            for (uint32_t element : {0u, 32u, 224u}) {
                const uint32_t block = (b_offset_elements + element) / q8_k;
                assert(block == batch_b * 11u + element / q8_k);
            }
        }
    }
}

int main(int argc, char ** argv) {
    const std::string which = argc > 1 ? argv[1] : "all";
    if (which == "dequant" || which == "all") {
        test_standalone_dequant_contract();
    }
    if (which == "cm2" || which == "all") {
        test_cm2_tq1_signed_trit();
    }
    if (which == "offsets" || which == "all") {
        test_integer_dot_offsets();
    }
    std::cout << "TQ Vulkan regression contract passed: " << which << '\n';
    return 0;
}
