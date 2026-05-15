# API Guide

## Two APIs, One Library `[Essentials]`

hipBLASLt provides two API surfaces:

- **C API** (`hipblaslt.h`) — Mirrors cuBLASLt for portability. Uses opaque handles and attribute setters. Best for projects that need to support both CUDA and ROCm with minimal code changes.
- **C++ Extension API** (`hipblaslt-ext.hpp`) — More ergonomic, ROCm-native API. Uses classes like `hipblaslt_ext::Gemm`, `hipblaslt_ext::GroupedGemm`, and `hipblaslt_ext::GemmPreference`. Recommended for new code.

A third surface exists for standalone operations:

- **Extension Operations API** (`hipblaslt-ext-op.h`) — C API for non-GEMM operations: softmax, layer normalization, and absolute max (amax).

**Recommendation:** Use the C++ extension API for new projects. It's cleaner and exposes features that the C API does not (e.g., `getAllAlgos`).

## Core API Pattern `[Essentials]`

Regardless of which API surface you use, every GEMM follows the same sequence:

```
┌─────────────────────────┐
│  1. Create handle       │  hipblasLtCreate() / hipblasLtHandle_t
├─────────────────────────┤
│  2. Describe matrices   │  hipblasLtMatrixLayout_t for A, B, C, D
├─────────────────────────┤
│  3. Set up matmul desc  │  hipblasLtMatmulDesc_t (compute type, transpose, epilogue)
├─────────────────────────┤
│  4. Configure prefs     │  hipblasLtMatmulPreference_t (max workspace size)
├─────────────────────────┤
│  5. Find algorithms     │  hipblasLtMatmulAlgoGetHeuristic() or getAllAlgos()
├─────────────────────────┤
│  6. Execute GEMM        │  hipblasLtMatmul() with chosen algorithm
├─────────────────────────┤
│  7. Cleanup             │  Destroy handles, descriptors, free memory
└─────────────────────────┘
```

With the C++ extension API, steps 2-5 are simplified. You create a `hipblaslt_ext::Gemm` instance, set problem parameters, and call `algoGetHeuristic()` or `getAllAlgos()` directly on the instance.

## Data Types `[Essentials]`

hipBLASLt supports a wide range of data types for mixed-precision GEMM:

| hipDataType | Description | Notes |
|---|---|---|
| `HIP_R_4F_E2M1` | 4-bit float4 (FP4) | Input only |
| `HIP_R_6F_E2M3` | 6-bit float6 (FP6) | Input only |
| `HIP_R_6F_E3M2` | 6-bit bfloat6 (BF6) | Input only |
| `HIP_R_8I` | 8-bit signed integer (INT8) | |
| `HIP_R_8F_E4M3_FNUZ` | 8-bit float8 (FP8 FNUZ) | gfx942 only |
| `HIP_R_8F_E5M2_FNUZ` | 8-bit bfloat8 (BF8 FNUZ) | gfx942 only |
| `HIP_R_8F_E4M3` | 8-bit float8 (FP8 OCP) | gfx950, gfx12 |
| `HIP_R_8F_E5M2` | 8-bit bfloat8 (BF8 OCP) | gfx950, gfx12 |
| `HIP_R_16F` | 16-bit half precision (FP16) | |
| `HIP_R_16BF` | 16-bit bfloat16 (BF16) | |
| `HIP_R_32F` | 32-bit single precision (FP32) | |
| `HIP_R_32I` | 32-bit signed integer (INT32) | Output/accumulate |

**Compute types** control the precision of the internal accumulation:

| Compute Type | Description |
|---|---|
| `HIPBLAS_COMPUTE_16F` | 16-bit half precision compute |
| `HIPBLAS_COMPUTE_32F` | 32-bit single precision compute |
| `HIPBLAS_COMPUTE_32I` | 32-bit integer compute |
| `HIPBLAS_COMPUTE_64F` | 64-bit double precision compute |
| `HIPBLAS_COMPUTE_32F_FAST_16F` | Tensor Cores with FP16 down-conversion |
| `HIPBLAS_COMPUTE_32F_FAST_16BF` | Tensor Cores with BF16 down-conversion |
| `HIPBLAS_COMPUTE_32F_FAST_TF32` | Tensor Cores with TF32 compute |

A/B/C/D types can be mixed — for example, FP16 inputs with FP32 output. See the [data type support reference](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/data-type-support.html) for all valid combinations.

## Fused Operations `[Essentials]`

hipBLASLt can fuse post-GEMM operations into the kernel, avoiding extra memory round-trips. Set the epilogue via `hipblasLtMatmulDescSetAttribute()` with `HIPBLASLT_MATMUL_DESC_EPILOGUE`.

### Activation functions

Apply an activation to the GEMM output:

```cpp
// ReLU: D = max(alpha * A * B + beta * C, 0)
hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_RELU;
hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
    &epilogue, sizeof(epilogue));

// GELU: D = GELU(alpha * A * B + beta * C)
// Use: HIPBLASLT_EPILOGUE_GELU

// Swish: D = Swish(alpha * A * B + beta * C)
// Use: HIPBLASLT_EPILOGUE_SWISH_EXT

// Sigmoid: D = Sigmoid(alpha * A * B + beta * C)
// Use: HIPBLASLT_EPILOGUE_SIGMOID

// Clamp: D = clamp(alpha * A * B + beta * C, activation_arg1, activation_arg2)
// (these are separate from the GEMM scaling factors alpha/beta)
// Use: HIPBLASLT_EPILOGUE_CLAMP_EXT
```

### Bias

Add a broadcast bias vector to the GEMM result:

```cpp
// D = alpha * A * B + beta * C + bias
// Use: HIPBLASLT_EPILOGUE_BIAS

// Activation + bias combinations:
// HIPBLASLT_EPILOGUE_RELU_BIAS    — bias then ReLU
// HIPBLASLT_EPILOGUE_GELU_BIAS    — bias then GELU
// HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT — bias then Swish
// HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT — bias then Clamp
```

Set the bias pointer via `HIPBLASLT_MATMUL_DESC_BIAS_POINTER` and its data type via `HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`.

### Auxiliary output

Save intermediate results before activation for backward pass:

```cpp
// Save pre-activation output for GELU backward
// Use: HIPBLASLT_EPILOGUE_GELU_AUX or HIPBLASLT_EPILOGUE_GELU_AUX_BIAS
// Use: HIPBLASLT_EPILOGUE_RELU_AUX or HIPBLASLT_EPILOGUE_RELU_AUX_BIAS
// Use: HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT or HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT
```

### Gradient operations

Compute activation gradients for backward pass:

```cpp
// HIPBLASLT_EPILOGUE_DGELU       — gradient GELU
// HIPBLASLT_EPILOGUE_DGELU_BGRAD — gradient GELU + bias gradient
// HIPBLASLT_EPILOGUE_DRELU       — gradient ReLU
// HIPBLASLT_EPILOGUE_DRELU_BGRAD — gradient ReLU + bias gradient
// HIPBLASLT_EPILOGUE_BGRADA      — bias gradient on A
// HIPBLASLT_EPILOGUE_BGRADB      — bias gradient on B
```

### Amax output

Compute the absolute maximum of the output for FP8 scaling:

See samples `09_hipblaslt_gemm_amax/` and `10_hipblaslt_gemm_amax_with_scale/`.

## Grouped GEMM `[Essentials]`

Grouped GEMM batches multiple independent GEMM problems of potentially different sizes into a single kernel launch. This is useful for:

- Transformer models with variable sequence lengths
- Multi-head attention with different head sizes
- Any workload with many small GEMMs

Using the C++ extension API:

```cpp
hipblaslt_ext::GroupedGemm groupedGemm(handle,
    hipblasOperation_t::HIPBLAS_OP_N, hipblasOperation_t::HIPBLAS_OP_N,
    HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F);

// Each problem has its own M, N, K, and data pointers
// std::vector<int64_t>                     m_vec, n_vec, k_vec, batch_vec;
// std::vector<hipblaslt_ext::GemmEpilogue> epilogues;
// std::vector<hipblaslt_ext::GemmInputs>   inputs;
// ... populate vectors (see sample 16 for the full pattern)
groupedGemm.setProblem(m_vec, n_vec, k_vec, batch_vec, epilogues, inputs);

// Get algorithms and execute
groupedGemm.algoGetHeuristic(requestedAlgoCount, pref, heuristicResults);
groupedGemm.initialize(heuristicResults[0].algo, workspace);
groupedGemm.run(stream);
```

See samples `16_hipblaslt_groupedgemm_ext/`, `17_hipblaslt_groupedgemm_fixed_mk_ext/`, and `18_hipblaslt_groupedgemm_get_all_algos_ext/`.

## Extension Operations `[Essentials]`

Standalone operations available via `hipblaslt-ext-op.h`:

- **Softmax** — `hipblasltExtSoftmax()`: Compute softmax along a specified dimension of a 2D tensor. Supports FP32 only. Only `dim=1` is supported. Max second dimension (`n`) is 256.
- **Layer Normalization** — `hipblasltExtLayerNorm()`: Compute layer normalization on a 2D tensor with optional gamma/beta. Supports FP32 only. Max first dimension is 4096.
- **Absolute Maximum** — `hipblasltExtAMax()`: Compute the absolute maximum of a 2D tensor. Supports FP32 and FP16 input.

These operations use precompiled device kernels from `device-library/extops/`.

## Algorithm Selection and Tuning `[Deep Dive]`

### getHeuristic vs. getAllAlgos

- **`algoGetHeuristic()`** — Returns a sorted list of recommended algorithms for your problem. Fast, good default choice. May not return all possible algorithms.
- **`getAllAlgos()`** — Returns every available algorithm. Use this when tuning for maximum performance — benchmark each algorithm and pick the fastest.

### What algorithm parameters mean

Each `hipblasLtMatmulAlgo_t` returned encodes:

- **`algoId`** — Identifies the kernel solution (maps to a TensileLite logic file entry)
- **`solutionIndex`** — Specific kernel variant within the solution family

### Tuning parameters

When using the C++ extension API, you can set tuning parameters:

- **Split-K** — Splits the K dimension across multiple workgroups for better parallelism on small M/N problems. Set via the ext API `GemmTuning::setSplitK()` (C++ extension only).
- **Workgroup Mapping (WGM)** — Controls how workgroups are mapped to output tiles. Different mappings optimize for different memory access patterns.

See sample `03_hipblaslt_gemm_tuning_splitk_ext/` for split-K tuning and `14_hipblaslt_gemm_tuning_wgm_ext/` for WGM tuning.

## Workspace Management `[Deep Dive]`

Some algorithms require temporary workspace memory for intermediate results (e.g., split-K reduction buffers).

1. **Set maximum workspace in preferences:**
   ```cpp
   hipblaslt_ext::GemmPreference pref;
   pref.setMaxWorkspaceBytes(32 * 1024 * 1024); // 32 MB
   ```

2. **Allocate workspace on device:**
   ```cpp
   void* workspace;
   hipMalloc(&workspace, pref.getMaxWorkspaceBytes());
   ```

3. **Pass workspace to the GEMM call.** The library selects algorithms that fit within your workspace budget. Larger workspace allows more algorithms to be considered.

4. **Check actual workspace needed:** After selecting an algorithm, call `matmulIsAlgoSupported()` which returns `workspaceSizeInBytes` — the actual workspace the algorithm needs.

## Samples Index `[Essentials]`

| # | Sample | Feature |
|---|---|---|
| 01 | `01_hipblaslt_gemm` / `01_hipblaslt_gemm_ext` | Basic GEMM (C API / C++ ext) |
| 02 | `02_hipblaslt_gemm_batched` / `02_hipblaslt_gemm_batched_ext` | Batched (strided) GEMM |
| 03 | `03_hipblaslt_gemm_tuning_splitk_ext` | Split-K tuning |
| 04 | `04_hipblaslt_gemm_bias` / `04_hipblaslt_gemm_bias_ext` | GEMM with bias vector |
| 05 | `05_hipblaslt_gemm_get_all_algos` / `05_hipblaslt_gemm_get_all_algos_ext` | Enumerate all algorithms |
| 06 | `06_hipblaslt_gemm_get_algo_by_index_ext` | Select algorithm by index |
| 07 | `07_hipblaslt_gemm_alphavec_ext` | Alpha vector scaling |
| 08 | `08_hipblaslt_gemm_gelu_aux_bias` / `08_hipblaslt_gemm_gelu_aux_bias_ext` | GELU + aux output + bias |
| 09 | `09_hipblaslt_gemm_amax` / `09_hipblaslt_gemm_amax_ext` | Amax output |
| 10 | `10_hipblaslt_gemm_amax_with_scale` / `10_hipblaslt_gemm_amax_with_scale_ext` | Amax with scale |
| 11 | `11_hipblaslt_gemm_bgradb` / `11_hipblaslt_gemm_ext_bgradb` | Bias gradient on B |
| 12 | `12_hipblaslt_gemm_dgelu_bgrad` / `12_hipblaslt_gemm_dgelu_bgrad_ext` | Gradient GELU + bias gradient |
| 12 | `12_hipblaslt_gemm_drelu_bgrad` / `12_hipblaslt_gemm_drelu_bgrad_ext` | Gradient ReLU + bias gradient |
| 13 | `13_hipblaslt_gemm_is_tuned_ext` | Check if algorithm is tuned |
| 14 | `14_hipblaslt_gemm_tuning_wgm_ext` | Workgroup mapping tuning |
| 15 | `15_hipblaslt_gemm_with_scale_a_b` / `15_hipblaslt_gemm_with_scale_a_b_ext` | Scale A and B |
| 15 | `15_hipblaslt_gemm_with_scale_a_b_vector` | Scale A and B (vector) |
| 16 | `16_hipblaslt_groupedgemm_ext` | Grouped GEMM |
| 17 | `17_hipblaslt_groupedgemm_fixed_mk_ext` | Grouped GEMM (fixed M, K) |
| 18 | `18_hipblaslt_groupedgemm_get_all_algos_ext` | Grouped GEMM algorithm enumeration |
| 19 | `19_hipblaslt_gemm_mix_precision` / `19_hipblaslt_gemm_mix_precision_ext` | Mixed precision GEMM |
| 20 | `20_hipblaslt_gemm_mix_precision_with_amax_ext` | Mixed precision + amax |
| 21 | `21_hipblaslt_gemm_attr_tciA_tciB` | Tensor core index attributes |
| 22 | `22_hipblaslt_ext_op_layernorm` | Extension op: layer normalization |
| 23 | `23_hipblaslt_ext_op_amax` | Extension op: absolute maximum |
| 24 | `24_hipblaslt_gemm_with_TF32` | TF32 compute |
| 25 | `25_hipblaslt_gemm_bias_swizzle_a_ext` | Bias + swizzle A |
| 25 | `25_hipblaslt_gemm_swizzle_a` / `25_hipblaslt_gemm_swizzle_b` | Matrix swizzle |
| 25 | `25_hipblaslt_weight_swizzle_padding` | Weight swizzle with padding |
| 26 | `26_hipblaslt_gemm_swish_bias` | Swish activation + bias |
| 27 | `27_hipblaslt_gemm_clamp_bias` | Clamp activation + bias |

**Recommended reading order for juniors:** 01 (ext) → 02 (ext) → 04 (ext) → 05 (ext) → 08 (ext) → 16.
