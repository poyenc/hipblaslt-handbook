# Chapter 4: Your First GEMM

## What We're Building

This chapter walks through a complete FP16 GEMM using the hipBLASLt extension API. The computation is:

```
D = alpha * A * B + beta * C
```

where A, B, C, and D are half-precision (FP16) matrices and the arithmetic is performed in FP32.

We will annotate every section of the sample at `clients/samples/01_hipblaslt_gemm_ext/sample_hipblaslt_gemm_ext.cpp`. This sample uses the C++ extension API (`hipblaslt_ext::Gemm`) rather than the lower-level C API, which is the recommended path for new code.

## Prerequisites

You should have a working build environment before proceeding. See [Chapter 2: Environment Setup](02-environment-setup.md). In particular, you need a build that includes client programs:

```bash
inv build --architecture <your_gfx> --clients
```

where `<your_gfx>` is your GPU target (e.g., `gfx942`, `gfx950`).

## Step-by-Step Walkthrough

The sample consists of two parts: a `main()` function that uses a `Runner` helper to manage memory and a HIP stream, and a `simpleGemmExt()` function that contains the actual hipBLASLt API calls. We will focus on the API calls in `simpleGemmExt()` and explain how `Runner` sets up the environment around them.

### Step 1: Create a hipBLASLt Handle

Before any hipBLASLt call, you need a library handle. The `Runner` constructor creates it:

```cpp
CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
```

The handle is an opaque `hipblasLtHandle_t` that the library uses to manage internal state. You create one per application (or per thread), and destroy it when done. The `Runner` destructor calls:

```cpp
CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
```

**Why:** The handle is the entry point to the library. Every subsequent API call takes it as the first parameter. Creating and destroying handles is cheap, but you only need one.

### Step 2: Set Up the Problem Type (Gemm Constructor)

The `simpleGemmExt()` function begins by creating a `hipblaslt_ext::Gemm` object that describes the data types and transpose operations:

```cpp
hipblaslt_ext::Gemm gemm(
    handle, trans_a, trans_b, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F);
```

The constructor parameters are:

| Parameter | Value in the sample | Meaning |
|-----------|-------------------|---------|
| `handle` | The hipBLASLt handle | Library context |
| `trans_a` | `HIPBLAS_OP_N` | A is not transposed |
| `trans_b` | `HIPBLAS_OP_N` | B is not transposed |
| `typeA` | `HIP_R_16F` | A is FP16 |
| `typeB` | `HIP_R_16F` | B is FP16 |
| `typeC` | `HIP_R_16F` | C is FP16 |
| `typeD` | `HIP_R_16F` | D is FP16 |
| `typeCompute` | `HIPBLAS_COMPUTE_32F` | Accumulate in FP32 |

**Why:** This tells the library what kind of problem to solve before specifying dimensions. The data types drive kernel selection -- FP16 input with FP32 compute is the most common configuration for deep learning workloads.

### Step 3: Configure the Epilogue and Input Pointers

Next, the sample sets up the epilogue (post-GEMM operation) and binds the GPU memory pointers:

```cpp
hipblaslt_ext::GemmEpilogue
    epilogue; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
hipblaslt_ext::GemmInputs inputs;
inputs.setA(d_a);
inputs.setB(d_b);
inputs.setC(d_c);
inputs.setD(d_d);
inputs.setAlpha(&alpha);
inputs.setBeta(&beta);
```

The `GemmEpilogue` object controls what happens after the matrix multiply -- bias addition, activation functions (GELU, ReLU), etc. These operations are fused into the GEMM kernel itself, not launched as separate kernels. Here it uses the default: no extra operations. We will add epilogue operations in the exercises below.

The `GemmInputs` object binds device pointers for each matrix and the scalar values alpha and beta. Note that `d_a`, `d_b`, `d_c`, `d_d` are device pointers (GPU memory) allocated by the `Runner` helper via `hipMalloc()` (see `helper.h` for the full memory setup), while `alpha` and `beta` are host-side `float` values passed by pointer.

**Why:** Separating epilogue configuration from input pointers lets you reuse the same problem setup with different data. You can update just the pointers (e.g., for a new batch) without rebuilding the entire problem.

### Step 4: Define Problem Dimensions

With inputs bound, the sample sets the matrix dimensions:

```cpp
gemm.setProblem(m, n, k, batch_count, epilogue, inputs);
```

In `main()`, these are set to:

```cpp
Runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float> runner(
    1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);
```

So: m=1024, n=512, k=1024, batch_count=1, alpha=1.0, beta=1.0, and 32 MB of workspace.

The matrix layout for `HIPBLAS_OP_N` (no transpose) is column-major:
- A is (m x k) = (1024 x 1024), leading dimension = m
- B is (k x n) = (1024 x 512), leading dimension = k
- C and D are (m x n) = (1024 x 512), leading dimension = m

**Why:** `setProblem()` combines dimensions, epilogue, and inputs into a fully-specified GEMM problem. The library needs all of this to select the right kernel. The simple overload shown here computes leading dimensions and strides automatically from m, n, k, and the transpose operations.

### Step 5: Find an Algorithm

hipBLASLt uses a heuristic system to select the best kernel for your problem:

```cpp
hipblaslt_ext::GemmPreference gemmPref;
gemmPref.setMaxWorkspaceBytes(max_workspace_size);

const int                                     request_solutions = 1;
std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
CHECK_HIPBLASLT_ERROR(gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResult));

if(heuristicResult.empty())
{
    std::cerr << "No valid solution found!" << std::endl;
    return;
}
```

Note: In the actual sample code, `GemmPreference` is created on line 100, before the `Gemm` object. We present it in Step 5 because it is only consumed by `algoGetHeuristic()` -- the creation order is flexible.

`GemmPreference` tells the heuristic how much workspace memory is available. More workspace can unlock faster algorithms that use extra scratch memory.

`algoGetHeuristic()` returns up to `request_solutions` algorithms, sorted by estimated performance (fastest first). Each result contains an `algo` field (the algorithm handle) and a `workspaceSize` field (how much workspace that algorithm needs).

**Why:** Different problem sizes and data types favor different kernels. The heuristic searches through available precompiled solutions (from TensileLite logic files) and returns the best match. In the hipBLASLt API, each solution is called an **algorithm** -- the two terms refer to the same thing. Always check that the result is non-empty -- some exotic configurations may not have a matching kernel.

### Step 6: Initialize and Execute the GEMM

With an algorithm selected, the sample initializes the kernel arguments and runs it:

```cpp
// Make sure to initialize every time when algo changes
gemm.setMaxWorkspaceBytes(max_workspace_size);
CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, d_workspace));
CHECK_HIPBLASLT_ERROR(gemm.run(stream));
```

`initialize()` takes the chosen algorithm and a device workspace pointer. It prepares all kernel arguments. You must call this again if you change the algorithm or workspace pointer.

`run()` enqueues the kernel on the given HIP stream. It is asynchronous -- the function returns immediately and the GPU executes in the background.

**Why:** The two-step initialize/run pattern lets you amortize setup cost. In a training loop you would call `initialize()` once per shape change and `run()` every iteration.

### Step 7: Transfer Data and Synchronize

The `Runner::run()` method wraps the GEMM call with data transfers:

```cpp
void run(const std::function<void()>& func)
{
    hostToDevice();

    static_cast<void>(func());

    deviceToHost();
    static_cast<void>(hipStreamSynchronize(stream));
}
```

`hostToDevice()` copies A, B, and C from host (CPU) memory to device (GPU) memory using `hipMemcpyAsync`. `deviceToHost()` copies D back. `hipStreamSynchronize(stream)` waits for all GPU work on the stream to finish before the program continues.

**Why:** GPU memory is separate from CPU memory. Data must be explicitly transferred. In a real application, your data likely lives on the GPU already (e.g., the output of a previous layer), so these copies are often unnecessary.

### Step 8: Cleanup

The `Runner` destructor handles all cleanup:

```cpp
~Runner()
{
    CHECK_HIP_ERROR(hipFree(d_workspace));
    CHECK_HIP_ERROR(hipFree(a));
    // ... (all host and device allocations)
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(handle));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}
```

Every `hipMalloc` / `hipHostMalloc` is paired with `hipFree`. The hipBLASLt handle is destroyed. The HIP stream is destroyed. The `hipblaslt_ext::Gemm` object is destroyed automatically when it goes out of scope -- it manages any internal GPU resources via RAII, so no manual cleanup is needed.

**Why:** Leaking GPU memory will exhaust VRAM. In production code, use RAII wrappers or smart pointers.

## Build and Run It

The sample is built automatically when you include `--clients` in the build:

```bash
inv build --architecture <your_gfx> --clients
```

Replace `<your_gfx>` with your GPU target (e.g., `gfx942`, `gfx950`).

After the build completes, run the sample:

```bash
# If you used invoke:
./build/release/clients/samples/01_hipblaslt_gemm_ext/sample_hipblaslt_gemm_ext

# If you used CMake presets or CMake directly:
./build/clients/samples/01_hipblaslt_gemm_ext/sample_hipblaslt_gemm_ext

# Note: The `hipblaslt-clients` preset does not build samples
# (HIPBLASLT_ENABLE_SAMPLES=OFF). Use `cmake --preset default:release`
# or add -DHIPBLASLT_ENABLE_SAMPLES=ON to your configuration.
```

Expected output: the program runs silently and exits with code 0. There is no output on success. If your GPU does not support the requested configuration, you will see `"No valid solution found!"` on stderr.

## Exercises

### 1. Change the Matrix Dimensions

In `main()`, modify the `Runner` constructor to use different sizes:

```cpp
Runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float> runner(
    1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);
```

Try changing `1024, 512, 1024` (m, n, k) to `2048, 2048, 2048` for a square matrix multiply, or to a non-power-of-two like `1000, 500, 768`. Rebuild and re-run. The heuristic will automatically select a different kernel suited to the new sizes.

### 2. Switch to FP32

Change the data types from FP16 to FP32. Two changes are needed:

In `main()`, change the `Runner` template parameters and type constants:

```cpp
Runner<float, float, float, float, float> runner(
    1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);
```

In `simpleGemmExt()`, change the `Gemm` constructor data types:

```cpp
hipblaslt_ext::Gemm gemm(
    handle, trans_a, trans_b, HIP_R_32F, HIP_R_32F, HIP_R_32F, HIP_R_32F, HIPBLAS_COMPUTE_32F);
```

Note that the compute type stays `HIPBLAS_COMPUTE_32F` -- for FP32 inputs, the accumulation is naturally FP32.

### 3. Add a Bias Vector

To see how epilogue operations work, look at `clients/samples/04_hipblaslt_gemm_bias_ext/sample_hipblaslt_gemm_bias_ext.cpp`. That sample demonstrates adding a bias vector to the GEMM output:

```
D = alpha * A * B + beta * C + bias
```

It uses `GemmEpilogue::setMode(HIPBLASLT_EPILOGUE_BIAS)` and `GemmInputs::setBias()` to enable and configure the bias. Start by reading that sample and comparing it with the one you just walked through. Key differences include the epilogue mode, the extra bias pointer (`GemmInputs::setBias()`), a bias data type setting (`GemmEpilogue::setBiasDataType()`), and bias configuration in the `Runner` helper.

## Where to Go Next

- [Chapter 5: API Guide](05-api-guide.md) -- comprehensive coverage of all API types, data types, epilogue modes, and advanced features like grouped GEMM and FP8 scaling.

After this chapter, a good reading order through the samples is:

1. `01_hipblaslt_gemm_ext` -- you just read this one
2. `02_hipblaslt_gemm_batched_ext` -- batched GEMM (multiple GEMMs in one call)
3. `04_hipblaslt_gemm_bias_ext` -- adding a bias vector
4. `08_hipblaslt_gemm_gelu_aux_bias_ext` -- fused GELU activation with bias
5. `05_hipblaslt_gemm_get_all_algos_ext` -- enumerating and benchmarking all available algorithms
6. `16_hipblaslt_groupedgemm_ext` -- grouped GEMM for variable-size problems
