# Native Vulkan validation for legacy TQ1_0/TQ2_0

Validated on 2026-07-24 with AMD Radeon Graphics (RADV Renoir), subgroup size 64, fp16 enabled, no integer-dot-product extension, and no cooperative matrix cores.

This work concerns legacy llama.cpp `TQ1_0` and `TQ2_0` tensors with group size 256. It does not conflate them with Prism's official `Q1_0`/`Q2_0` group-128 formats, upstream `Q2_0` group 64, or Prism `PQ2_0`.

## Source provenance

The dense TQ Vulkan matvec/matmul implementation was ported from `infinitalo/llama.cpp`, branch `italo/llama-bitnet`, commits `782d22ca3..ef80e915e`, then integrated with Prism's newer official Q1/Q2, IQ4_NL, shader generator, and cooperative-matrix paths.

Additional local work adds:

- TQ quantized `GET_ROWS` and f32-output `GET_ROWS`.
- TQ-to-f32 copy/dequantization.
- Exact f32-to-TQ1_0 and f32-to-TQ2_0 packers matching the CPU references.
- TQ `SET_ROWS` for generated i32/i64 index shaders.
- TQ `MUL_MAT_ID` through the GPU vector-ID implementation.
- Realistic multi-block decode and tiled-matmul differential cases.
- Standalone TQ-to-f16 dequantization with the standard five-word push-constant ABI and 1024-element dispatch denominator.
- Signed TQ1 trit conversion in the cooperative-matrix-2 decoder.
- Correct quant-block/scalar-element batch offsets in the integer-dot TQ matvec shader.

Tiled TQ `MUL_MAT_ID` shaders are deliberately not generated or registered. Their ID gather path failed hardware differential tests. All TQ expert shapes therefore use the correct GPU vector-ID path. This is a performance fallback, not a CPU fallback.

## Native configuration and build

No Docker is required.

```bash
cd /media/cody/32897c2b-d8cf-416d-9bab-5271d50be99b1/LLMworkspace/prism-vulkan-native

cmake -S . -B build-vulkan-native -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGGML_VULKAN=ON \
  -DGGML_CUDA=OFF \
  -DGGML_HIP=OFF \
  -DBUILD_TESTING=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=ON

cmake --build build-vulkan-native -j6
```

The validated clean build regenerated all shaders and completed 557 Ninja steps.

## Focused hardware differential test

```bash
./build-vulkan-native/bin/test-backend-ops test \
  -b Vulkan0 \
  -p 'tq1_0|tq2_0' \
  -j 1
```

Result on RADV Renoir: `86/86` supported TQ cases pass after adding realistic `m=512,n=1,k=4096` decode, batched/broadcast dense vector, multi-block four-expert ID, explicit i32 SET_ROWS, and raw byte-for-byte CPU-oracle packing cases. Earlier preserved log files may show `76/76`, `78/78`, `80/80`, or `84/84` before those additional cases.

Covered operation classes:

- `GET_ROWS` for TQ1_0 and TQ2_0.
- f32-to-TQ and TQ-to-f32 `CPY`. The dedicated 2048-element packer cases compare raw CPU/Vulkan bytes and cover all 243 valid TQ1 five-trit combinations, all 81 valid TQ2 four-trit combinations, high-trit lanes, and positive/negative half-away ties.
- Exact TQ packing through `SET_ROWS`.
- Dense vector and tiled `MUL_MAT`, including one-block and multi-block shapes.
- `MUL_MAT_ID` with one/four experts, one/two selected experts, and one/32 tokens.

Expected unsupported probes are not counted as failures:

- `GET_ROWS_BACK` on quantized source types.
- Direct f16/bf16-to-TQ packing.
- TQ `MUL_MAT_ID` with a non-f32 right-hand operand; vector-ID currently has only f32 RHS pipelines, so the support query rejects f16 and q8_1 rather than advertising paths that would abort.
- Arbitrarily permuted quant-to-same-quant copies.

These gaps are shared with other quantized formats and are not part of immutable inference-weight execution.

## Review-regression tests

```bash
ctest --test-dir build-vulkan-native --output-on-failure \
  -R '^test-vulkan-tq-(regressions|standalone-dequant)$'
```

`test-vulkan-tq-regressions` enforces three shader contracts that Renoir cannot execute directly: cooperative-matrix-2 signed TQ1 trits, integer-dot batch offsets, and standalone dequant push/dispatch geometry. The standalone dequant variants compile during the native build; CM2 and integer-dot remain source-contract and arithmetic-simulation checks with this toolchain.

`test-vulkan-tq-standalone-dequant` executes the standalone TQ-to-f16 path on Vulkan with an f16 identity matrix. It checks 32,768 dequantized weights per format across multiple workgroups against the CPU TQ decoder. Both TQ1_0 and TQ2_0 measured NMSE exactly `0` on Renoir.

The current glslc does not support `GL_NV_cooperative_matrix2` or `GL_EXT_integer_dot_product`, and Renoir exposes neither feature. Those two feature-specific paths therefore have compile/source-contract and arithmetic-simulation coverage here, not a claim of local hardware execution.

## Project test suite

```bash
ctest --test-dir build-vulkan-native --output-on-failure -j6
```

Initial result: 57/59 passed.

- `test-jinja-py` failed only because system Python lacked `jinja2`. It passes in an isolated environment:

  ```bash
  uv venv /tmp/prism-ctest-venv
  uv pip install --python /tmp/prism-ctest-venv/bin/python jinja2
  PATH=/tmp/prism-ctest-venv/bin:$PATH \
    ctest --test-dir build-vulkan-native --output-on-failure -R '^test-jinja-py$'
  ```

- `test-quant-type-selection` has pre-existing Qwen3.5-397B and Qwen3.5-27B snapshot mismatches. The TQ Vulkan diff modifies none of the quant-selection implementation or snapshots. Do not regenerate those snapshots as part of this work.

The complete `test-backend-ops` CTest target passed after 497 seconds.

## Model-level validation

Models:

- `/home/cody/onebit-forge/_tq1_fresh.gguf` — qwen3vl 8B, TQ1_0, 1.69 bpw.
- `/home/cody/onebit-forge/_tq2_fresh.gguf` — qwen3vl 8B, TQ2_0, 2.06 bpw.
- `/home/cody/onebit-forge/_gptq_fixed.gguf` — qwen3vl 8B, mixed/corrected TQ1_0 artifact.
- `/home/cody/onebit-forge/_q8_ref.gguf` — Q8 reference control.

Fixed corpus: `/home/cody/onebit-forge/ppl_corpus/chunks1k/c285.txt`.

Example batched CPU-logit baseline:

```bash
./build-vulkan-native/bin/llama-perplexity \
  -m /home/cody/onebit-forge/_tq1_fresh.gguf \
  -f /home/cody/onebit-forge/ppl_corpus/chunks1k/c285.txt \
  -ngl 0 -c 128 -b 128 -ub 128 --chunks 1 --no-warmup \
  --save-all-logits /tmp/tq1-cpu.logits
```

Vulkan KL comparison:

```bash
./build-vulkan-native/bin/llama-perplexity \
  -m /home/cody/onebit-forge/_tq1_fresh.gguf \
  -f /home/cody/onebit-forge/ppl_corpus/chunks1k/c285.txt \
  -ngl 99 -c 128 -b 128 -ub 128 --chunks 1 --no-warmup \
  --kl-divergence --kl-divergence-base /tmp/tq1-cpu.logits
```

Batched CPU/Vulkan results for all three TQ artifacts:

- Same top token: 100%.
- Probability RMS difference: rounds to 0.000%.
- KL divergence: approximately `-1e-5` to `-3e-5`, numerical noise.

Perplexity on the fixed chunk:

| Model | CPU PPL, batch 128 |
|---|---:|
| TQ1 fresh | 3,974,589.67 |
| TQ2 fresh | 3,974,589.67 |
| GPTQ fixed | 37,530,599.78 |
| Q8 reference | 6.6735 |

The huge TQ perplexities are model-artifact failures. The Q8 control proves the corpus/evaluator is sane.

### Decode-path numerical caveat

Batch-1 CPU and Vulkan generation diverges on these malformed TQ models despite operation-level differential tests passing. For TQ1, batch-1 KL against CPU was `3.56892`, with the same top token on only `7.94%` of evaluated positions. The CPU model itself is numerically unstable: TQ1 PPL changes from about `3.97M` at batch 128 to `8.10M` at batch 1. Different valid accumulation paths therefore produce different garbage output.

This caveat must not be presented as successful semantic generation. A stable corrected TQ artifact is required for meaningful model-level decode agreement. Dense and expert kernels remain covered by CPU-vs-Vulkan operation tests, and batched full-model logits agree.

## Benchmark

```bash
./build-vulkan-native/bin/llama-bench \
  -m /home/cody/onebit-forge/_tq1_fresh.gguf \
  -ngl 0,99 -p 128 -n 16 -r 1 -o json
```

RADV Renoir results in tokens/second:

| Model | CPU pp128 | Vulkan pp128 | CPU tg16 | Vulkan tg16 |
|---|---:|---:|---:|---:|
| TQ1 fresh | 31.10 | 37.09 | 6.77 | 2.13 |
| TQ2 fresh | 35.60 | 46.27 | 6.29 | 2.64 |
| GPTQ fixed | 19.35 | 25.71 | 3.88 | 1.92 |

On this integrated Renoir GPU, Vulkan improves batched prompt processing by about 1.19–1.33× but is 2.0–3.2× slower for token decode than the optimized AVX2 CPU path.

## Scope exclusions

- TQ model weights do not imply TQ KV cache. Use supported f16/q8/q4 KV-cache formats.
- No TQ flash-attention K/V kernels are claimed.
- Official Prism Q1_0/Q2_0, upstream Q2_0 G64, Prism PQ2_0, and legacy TQ formats remain distinct wire formats and shader paths.
