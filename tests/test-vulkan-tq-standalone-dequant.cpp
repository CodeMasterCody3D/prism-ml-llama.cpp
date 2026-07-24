#undef NDEBUG
#include <ggml.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static ggml_backend_dev_t find_vulkan_device() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        const char * desc = ggml_backend_dev_description(dev);
        if ((name && std::strstr(name, "Vulkan")) || (desc && std::strstr(desc, "Vulkan"))) {
            return dev;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = ggml_backend_reg_name(reg);
        if (reg_name && std::strstr(reg_name, "Vulkan")) {
            return dev;
        }
    }
    return nullptr;
}

static void run_case(ggml_backend_t backend, ggml_type type) {
    constexpr int64_t K = 2048; // two 1024-element dequant dispatch groups per weight row
    constexpr int64_t M = 16;
    constexpr int64_t N = K;

    std::vector<float> weights(M * K);
    std::vector<float> activations(N * K);
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = std::sin(float(i) * 0.017f) * (0.5f + 0.5f * std::cos(float(i) * 0.003f));
    }
    for (int64_t n = 0; n < N; ++n) {
        activations[n * K + n] = 1.0f;
    }

    const size_t quant_bytes = ggml_row_size(type, K) * M;
    std::vector<uint8_t> quantized(quant_bytes);
    const size_t written = ggml_quantize_chunk(type, weights.data(), quantized.data(), 0, M, K, nullptr);
    assert(written == quant_bytes);

    std::vector<ggml_fp16_t> activations_f16(activations.size());
    ggml_fp32_to_fp16_row(activations.data(), activations_f16.data(), activations.size());

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    assert(traits && traits->to_float);
    std::vector<float> dequantized(weights.size());
    const size_t row_bytes = ggml_row_size(type, K);
    for (int64_t row = 0; row < M; ++row) {
        traits->to_float(quantized.data() + row * row_bytes, dequantized.data() + row * K, K);
    }

    std::vector<float> reference(M * N);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            reference[n * M + m] = dequantized[m * K + n];
        }
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    assert(ggml_backend_supports_op(backend, out));

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);
    ggml_backend_tensor_set(a, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(b, activations_f16.data(), 0, activations_f16.size() * sizeof(ggml_fp16_t));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(reference.size());
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));

    double sq_error = 0.0;
    double sq_ref = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(std::isfinite(actual[i]));
        const double delta = double(actual[i]) - double(reference[i]);
        sq_error += delta * delta;
        sq_ref += double(reference[i]) * double(reference[i]);
    }
    const double nmse = sq_ref > 0.0 ? sq_error / sq_ref : sq_error;
    std::printf("%s standalone TQ->F16 dequant MUL_MAT NMSE %.9g\n", ggml_type_name(type), nmse);
    assert(nmse < 1e-7);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static bool mul_mat_id_supported(ggml_backend_t backend, ggml_type type_a, ggml_type type_b) {
    constexpr int64_t K = 256;
    constexpr int64_t M = 64;

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx);

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, type_a, K, M, 1);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, type_b, K, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    const bool supported = ggml_backend_supports_op(backend, out);
    ggml_free(ctx);
    return supported;
}

int main() {
    ggml_backend_dev_t dev = find_vulkan_device();
    if (!dev) {
        std::puts("SKIP: no Vulkan backend device");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    assert(backend);
    run_case(backend, GGML_TYPE_TQ1_0);
    run_case(backend, GGML_TYPE_TQ2_0);

    for (ggml_type type : {GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0}) {
        const bool f32_supported = mul_mat_id_supported(backend, type, GGML_TYPE_F32);
        const bool f16_supported = mul_mat_id_supported(backend, type, GGML_TYPE_F16);
        const bool q8_supported  = mul_mat_id_supported(backend, type, GGML_TYPE_Q8_1);
        std::printf("%s MUL_MAT_ID support: f32=%d f16=%d q8_1=%d\n",
                    ggml_type_name(type), f32_supported, f16_supported, q8_supported);
        assert(f32_supported);
        assert(!f16_supported);
        assert(!q8_supported);
    }

    ggml_backend_free(backend);
    std::puts("TQ standalone dequant and support-query hardware tests passed");
    return 0;
}
