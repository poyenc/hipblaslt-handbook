# Chapter 1: What is hipBLASLt?

## What is GEMM? `[Essentials]`

GEMM stands for **GE**neral **M**atrix **M**ultiplication. At its core, it multiplies two matrices together and adds a third:

```
C = alpha * A * B + beta * C
```

where `A`, `B`, and `C` are matrices and `alpha` and `beta` are scalar values that control scaling.

Matrix multiplication is one of the most computationally important operations in modern computing. Training a single layer of a neural network boils down to multiplying weight matrices by activation matrices -- often thousands of times per second. Scientific simulations, signal processing, and recommendation systems all rely on the same fundamental operation.

GPUs are well-suited for GEMM because matrix multiplication is inherently parallel: every element of the output matrix can be computed independently. A modern AMD GPU has thousands of compute units that can perform these multiplications simultaneously, achieving throughput orders of magnitude higher than a CPU. Specialized matrix hardware (such as AMD's MFMA -- Matrix Fused Multiply-Add -- units) accelerates this further by computing small matrix tiles in a single instruction.

A **BLAS** (Basic Linear Algebra Subprograms) library provides standardized, optimized implementations of operations like GEMM. Rather than writing GPU kernels from scratch, application developers call into a BLAS library and get hardware-tuned performance automatically.

## What hipBLASLt does `[Essentials]`

hipBLASLt is AMD's extended GEMM library. Unlike standard BLAS GEMM (which overwrites C in place), hipBLASLt writes the result to a separate output matrix D and adds fused post-processing:

```
D = Activation(alpha * op(A) * op(B) + beta * C + bias)
```

Here is what each term means:

- **A, B** -- Input matrices. These are the two matrices being multiplied.
- **C** -- Accumulation matrix. Scaled by `beta` and added to the product before post-processing. Unlike A and B, C is not transposed — it uses its own layout descriptor directly.
- **D** -- Output matrix. The final result after all operations.
- **alpha, beta** -- Scalar values that scale the matrix product and the accumulation matrix, respectively.
- **op()** -- An in-place transformation applied to A or B, such as transpose or non-transpose.
- **bias** -- A vector added to the result before activation (forward pass). It is broadcast across all columns of D, so its length must match the number of rows in D.
- **Activation** -- A pointwise function applied element-by-element to the result.

The key idea behind hipBLASLt is **fusion**. Without fusion, you would need to run the matrix multiply, write the result to GPU memory, then launch separate kernels for bias addition and activation. Each of those steps pays the cost of reading and writing large matrices to global memory. hipBLASLt fuses these operations into a single kernel launch: the bias addition and activation happen in registers before the result is ever written out, saving memory bandwidth and kernel launch overhead.

The primary entry point is the `hipblasLtMatmul` API. You describe the operation once (matrix layouts, data types, epilogue options), then reuse that description across different inputs.

## Supported features `[Essentials]`

- **Mixed precision** -- Input matrices can use lower-precision types (such as FP8 or FP16) while the computation accumulates in higher precision (such as FP32), trading off precision for throughput.

- **Activation functions** -- Fused pointwise transforms applied to the GEMM output. Supported activations from the epilogue enumeration include:
  - **ReLU** (`HIPBLASLT_EPILOGUE_RELU`) -- `x := max(x, 0)`
  - **GELU** (`HIPBLASLT_EPILOGUE_GELU`) -- Gaussian Error Linear Unit
  - **Swish** (`HIPBLASLT_EPILOGUE_SWISH_EXT`) -- `x := Swish(x, 1)`, also known as SiLU
  - **Sigmoid** (`HIPBLASLT_EPILOGUE_SIGMOID`) -- Sigmoid activation function
  - **Clamp** (`HIPBLASLT_EPILOGUE_CLAMP_EXT`) -- `x := max(activation_arg1, min(x, activation_arg2))` — these are separate from the GEMM scalars `alpha`/`beta`, even though the header comment confusingly uses the same names
  - **AUX variants** -- ReLU, GELU, and Clamp have AUX variants that save the pre-activation result to a separate buffer (e.g., `HIPBLASLT_EPILOGUE_GELU_AUX_BIAS`), useful for training workflows
  - **Gradient variants** -- `DRELU`, `DGELU` for backward-pass activation gradients, with optional fused bias gradient computation (`DRELU_BGRAD`, `DGELU_BGRAD`)

- **Bias vectors** -- A broadcast bias vector added to the GEMM result before the activation function (`HIPBLASLT_EPILOGUE_BIAS` for standalone bias). Bias can be combined with most activations (`RELU_BIAS`, `GELU_BIAS`, `SWISH_BIAS_EXT`, `CLAMP_BIAS_EXT`). Note that Sigmoid does not have a fused bias variant.

- **Bias gradient** -- Compute the gradient of the bias in the backward pass, fused with the GEMM (`HIPBLASLT_EPILOGUE_BGRADA`, `HIPBLASLT_EPILOGUE_BGRADB`).

- **Grouped GEMM** -- Execute multiple independent GEMM problems in a single kernel launch. Useful for batched inference, mixture-of-experts models, and other workloads that would otherwise require many small kernel launches.

- **Amax output** -- Compute the absolute maximum value of the output matrix D during the GEMM (`HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER`). This is essential for dynamic quantization in FP8 training workflows.

- **Matrix scaling modes** -- Per-tensor, per-row-vector, and block-scaled scaling of input matrices. Relevant for FP8 and MX (Microscaling) data formats, where MX is a block-scaled numeric format that groups elements and shares a common scale factor per block.

- **Matrix layout swizzling** -- Reorder matrix data into tiled memory layouts (such as `HIPBLASLT_ORDER_COL16_4R32`) for improved memory access patterns on specific hardware.

- **Auxiliary output** -- Save the pre-activation GEMM result to a separate buffer while also writing the post-activation result to D. Used for training workflows that need both values.

- **Alpha vector** -- Per-row scaling of the matrix product using a device-side vector instead of a single scalar.

- **ExtOp (Extended Operation) kernels** -- Standalone fused kernels for common operations outside of GEMM: layernorm, softmax, and amax reduction.

- **Matrix transform** -- A helper operation (`hipblasLtMatrixTransform`) for converting between matrix memory layouts and scaling values.

## Where hipBLASLt fits in the ROCm stack `[Essentials]`

hipBLASLt is one of several BLAS-level libraries in the ROCm ecosystem. Here is how it relates to other components:

**rocBLAS** is the foundational BLAS library for ROCm. It implements the full BLAS specification (Level 1, 2, and 3 routines) including a standard GEMM. hipBLASLt is narrower in scope -- it only does GEMM and related operations -- but goes deeper: it provides fused epilogues, mixed-precision support, FP8 data types, grouped GEMM, and algorithm selection APIs that rocBLAS does not expose.

**When to choose hipBLASLt over rocBLAS:**

- You need fused bias, activation, or quantization in the GEMM kernel
- You are working with FP8, BF6, F6, F4, or MX-format data types
- You want to tune algorithm selection or use grouped GEMM
- Your workload is deep learning training or inference

**When to use rocBLAS instead:**

- You need non-GEMM BLAS operations (AXPY, DOT, TRSM, etc.)
- You need fully optimized double-precision (FP64) GEMM — rocBLAS has broader FP64 coverage
- You want a standard BLAS interface with minimal configuration

**MIOpen** is AMD's deep learning primitives library (convolutions, RNNs, batch normalization, etc.). MIOpen and hipBLASLt are complementary: MIOpen handles convolution and other DNN-specific operations, while hipBLASLt handles the GEMM operations that make up the bulk of transformer and MLP workloads.

**PyTorch and JAX** on ROCm can use hipBLASLt under the hood for GEMM dispatch. When you call `torch.mm()` or a JAX `dot_general` on an AMD GPU, the framework's backend can route the operation to hipBLASLt to take advantage of fused epilogues and FP8 support. Application developers typically interact with hipBLASLt indirectly through these frameworks rather than calling the C API directly.

## Supported hardware `[Essentials]`

hipBLASLt supports the following AMD GPU architectures, as defined in `cmake/tensilelite_supported_architectures.cmake`:

| Architecture | GPU family | Notes |
|---|---|---|
| `gfx908` | AMD Instinct MI100 | CDNA 1 |
| `gfx90a` | AMD Instinct MI200 series (MI210, MI250, MI250X) | CDNA 2 |
| `gfx942` | AMD Instinct MI300 series (MI300A, MI300X) | CDNA 3. Supports FP8 FNUZ (Finite, NaN, Unsigned Zero — an AMD-specific FP8 encoding). |
| `gfx950` | AMD Instinct MI350 series | CDNA 4. Supports FP8 OCP (Open Compute Project — the industry-standard FP8 encoding), BF6/F6, F4, MX formats. |
| `gfx1100` | AMD Radeon RX 7900 series | RDNA 3 |
| `gfx1101` | AMD Radeon RX 7700/7800 series | RDNA 3 |
| `gfx1102` | AMD Radeon RX 7600 series | RDNA 3. Valid for `GPU_TARGETS` but not included in the default "all" build — must be explicitly specified. |
| `gfx1103` | AMD Radeon iGPU (780M, etc.) | RDNA 3 integrated |
| `gfx1150` | AMD Radeon | RDNA 3.5 |
| `gfx1151` | AMD Radeon | RDNA 3.5 |
| `gfx1152` | AMD Radeon | RDNA 3.5 |
| `gfx1153` | AMD Radeon | RDNA 3.5 |
| `gfx1200` | AMD Radeon RX 9070 series | RDNA 4 |
| `gfx1201` | AMD Radeon | RDNA 4 |
| `gfx1250` | AMD Radeon | RDNA 4+ |

The architecture name follows a pattern: `gfx` + generation + variant. The `gfx9xx` series (CDNA) are data center GPUs optimized for compute workloads. The `gfx1xxx` series (RDNA) are consumer and workstation GPUs. Higher numbers within a generation indicate different product SKUs.

Some architectures support an `:xnack` suffix that controls Unified Shared Memory (XNACK) replay mode. The `:xnack+` variant enables XNACK (required for address sanitizer builds): `gfx908:xnack+`, `gfx90a:xnack+`, `gfx942:xnack+`, `gfx950:xnack+`, and `gfx1250:xnack+`. The `:xnack-` variant explicitly disables it (e.g., `gfx908:xnack-`, `gfx90a:xnack-`). When no suffix is specified, the default XNACK behavior for that architecture is used.

When you build hipBLASLt with `GPU_TARGETS=all` (or without specifying a target), the default build includes: `gfx908`, `gfx90a`, `gfx942`, `gfx950`, `gfx1100`, `gfx1101`, `gfx1103`, `gfx1150`, `gfx1151`, `gfx1152`, `gfx1153`, `gfx1200`, `gfx1201`, and `gfx1250`. To build for a single architecture and reduce build time, pass it explicitly:

```bash
cmake -DGPU_TARGETS=gfx942 ...
```
