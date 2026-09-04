# TAARDIS fork of llama.cpp

**Ternary Adaptive Alignment & Rotation for Dense Integer Stacking** — the
runtime for [TAARDIS-27B](https://huggingface.co/CodeMasterCody3D/taardis-27b-full-ternary),
a 27B model whose weights, head, embeddings, norms, group scales **and KV cache**
are ternary integers. By Cody Dixon, 2026.

**Try it in one click** (free Colab GPU, 1M-token ternary context, chat UI): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CodeMasterCody3D/taardis-llama.cpp/blob/q1_0_g128-port/notebooks/TAARDIS_chat.ipynb)


## Lineage (read this if the GitHub "forked from" badge confuses you)

1. This repo **began as a fork of PrismML's llama.cpp** (via Mintplex-Labs'
   mirror) so Ternary-Bonsai could be run and benchmarked head-to-head. That
   is why GitHub still shows the fork badge and why a `prism` branch exists.
2. When upstream `ggml-org/llama.cpp` gained Bonsai support itself, TAARDIS
   was **ported onto fresh upstream main** — branch `q1_0_g128-port`, based on
   upstream `b356fa2` (2026-09-01). It contains **no PrismML code**: upstream
   commits plus TAARDIS commits only. Verify: compare the branch against
   `ggml-org/llama.cpp:master` — ahead only by the TAARDIS commits.
3. The repo was renamed from `prism-ml-llama.cpp` to `taardis-llama.cpp` on
   2026-09-03 to stop implying otherwise. Old URLs redirect.

## What this branch adds on top of upstream

**Formats (GGUF tensor types)**
- `Q1_0_g128` (enum 43) — ternary `{-1,0,+1}`, 2-bit pack, one fp16-class scale
  per 128 weights, 2.125 bpw. Byte-layout-compatible with Bonsai's Q2_0 g128.
- `Q1_T_g128` (enum 44) — the same ternary states in a **base-3 five-trit
  pack**, 1.75 bpw. Lossless repack of Q1_0_g128 (`q1t_transcode.py`).
- `Q1_R_g128` (enum 45) — rotated-trit k1, KV-cache use.

**CPU**
- AVX2 `vec_dot` kernels for `Q1_0_g128` and `Q1_T_g128` (reciprocal-multiply
  base-3 unpack), generic C fallbacks, quantize/dequantize with Lloyd scale
  refinement (`Q1_0_G128_LLOYD=0` disables).
- Ternary KV cache: `-ctk q1_t_g128 -ctv q1_t_g128` (also `q1_0_g128`).

**CUDA**
- Weight matmuls for enums 43/44 via dequant + cuBLAS GEMM (mmvq/fusion
  excluded by design).
- **Ternary KV cache on GPU**: `set_rows` write kernel (Lloyd-parity with the
  CPU quantizer) + flash-attention vec kernels (K dot on packed trits via
  dp4a, V dequant), matched-pair instances for head sizes 64/128/256.
  Validated: GPU-written cache scores 13.29 vs 13.28 CPU-written; a 1M-token
  ternary cache allocated on-GPU with the model decoding through it.
- `fwht.cu`: block-Hadamard rotation kernel (the weights live in a rotated
  basis; `LLAMA_FORGE_ROT_DISABLE=1` for bisecting).
- Fix for upstream's `fattn-vec` ncols=1 path: the q8_1 `Q_ds` scratch read is
  done with 32-bit loads (a fused 64-bit shared load faulted on Blackwell).

**Runtime**
- Rotation hook in `build_lora_mm`: per-tensor block-Hadamard from GGUF
  metadata (`forge.rotation.*`), asserted at load (497 tensors or refuse).
- "The Doctors" (cross-layer ternary corrections) load as a **standard LoRA
  GGUF** with the rotation folded in offline — no runtime change needed.
- `llama-quantize` guards against silently promoting head/embed off the
  ternary grid. `test-backend-ops` covers the new types (MUL_MAT, SET_ROWS,
  FLASH_ATTN_EXT incl. decode shapes).

**PrismML Bonsai**: upstream carries their `Q2_0`; for same-kernel comparisons
Bonsai GGUFs can also be retagged (type 42→43, byte-identical layout) and run
through the TAARDIS kernels.

## Build

```bash
git clone https://github.com/CodeMasterCody3D/taardis-llama.cpp llama.cpp && cd llama.cpp
cmake -B build -DLLAMA_CURL=OFF && cmake --build build -j --target llama-cli llama-perplexity   # CPU (AVX2)
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF  # CUDA (75=T4, 80=A100, 89=L4/40xx, 120=Blackwell)
```

License: upstream llama.cpp is MIT; TAARDIS additions are MIT. Models are
separate (see the model card for Apache-2.0 attribution to Qwen).
