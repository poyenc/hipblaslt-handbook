# Chapter 7 -- Host Library Guide

## 1. What the host library does `[Essentials]`

The host library is the C++ code that sits between the public API and the GPU. Its job is to translate API calls into contraction problems, select the best kernel solution from the library tree, and launch the kernel on a HIP stream. For definitions of these terms (Problem, Contraction Problem, Solution, Library), see [Chapter 3 Section 1: Core Concepts](03-architecture.md#1-core-concepts-essentials).

---

## 2. Request lifecycle `[Essentials]`

There are two call paths into the host library: the C API path, which
selects a kernel on every call, and the C++ extension API path, which
caches the selection for repeated dispatch. Both ultimately reach
`tensile_host.cpp` to launch the kernel on the GPU.

### C API path

```
App          hipblaslt.cpp    rocblaslt_mat.cpp   tensile_host.cpp    GPU
 |                |                 |                   |               |
 | hipblasLtMatmul()               |                   |               |
 |───────────────>|                |                   |               |
 |                | rocblaslt_matmul()                 |               |
 |                |────────────────>|                   |               |
 |                |                 | runContractionProblem()           |
 |                |                 |──────────────────>|               |
 |                |                 |                   | launchKernels |
 |                |                 |                   |──────────────>|
```

**hipblaslt.cpp** -- The public C API entry point. `hipblasLtMatmul()` casts the opaque hipBLASLt handle types (e.g., `hipblasLtHandle_t`, `hipblasLtMatmulDesc_t`) to their internal rocblaslt equivalents and delegates to `rocblaslt_matmul()`. Status codes are translated back to `hipblasStatus_t` on the way out.

**rocblaslt_mat.cpp** -- The validation and problem-construction layer. `rocblaslt_matmul()` checks arguments, extracts dimensions and data-type information from the descriptor structs, assembles a `RocblasltContractionProblem`, and passes it to `runContractionProblem()`.

**tensile_host.cpp** -- The backend dispatch layer. `runContractionProblem()` translates the `RocblasltContractionProblem` into a TensileLite `ContractionProblemGemm`, looks up a solution through the library tree, solves the problem to produce kernel invocations, and calls `adapter->launchKernels()` to enqueue the HIP kernel.

**GPU** -- The `SolutionAdapter` (the runtime object that manages code-object
loading and kernel submission) calls `hipModuleLaunchKernel()` to submit the precompiled code object kernel on the specified HIP stream.

### C++ extension API path

The ext API separates problem setup (`initialize`) from execution (`run`)
so that repeated dispatches of the same problem shape skip the
solution-selection step entirely.

```
App          Gemm              rocblaslt_mat.cpp   tensile_host.cpp    GPU
 |            |                      |                   |               |
 | initialize()                     |                   |               |
 |───────────>|                      |                   |               |
 |            | rocblaslt_makeArgument_cpp()             |               |
 |            |─────────────────────>|                   |               |
 | run()      |                      |                   |               |
 |───────────>|                      |                   |               |
 |            | rocblaslt_run_cpp()   |                   |               |
 |            |─────────────────────>|                   |               |
 |            |                      | runKernelFromInvocation()         |
 |            |                      |──────────────────>|               |
 |            |                      |                   | launchKernels |
 |            |                      |                   |──────────────>|
```

**Gemm** -- The `hipblaslt_ext::GemmInstance` class in `hipblaslt-ext.cpp`. The ext API separates problem setup from execution: `initialize()` selects a solution and caches the kernel invocation; `run()` replays the cached invocation without re-solving.

**rocblaslt_mat.cpp** -- `rocblaslt_makeArgument_cpp()` translates the problem into a kernel invocation and caches it. `rocblaslt_run_cpp()` delegates to `runKernelFromInvocation()` in `tensile_host.cpp`.

**tensile_host.cpp** -- `runKernelFromInvocation()` takes the pre-solved kernel invocation and calls `adapter->launchKernels()` directly, skipping the solution-lookup step that the C API path performs on every call.

**GPU** -- Same as the C API path: `hipModuleLaunchKernel()` submits the kernel.

---

## 3. Algorithm selection `[Essentials]`

The three user-facing entry points for algorithm selection are described in [Chapter 3: How Solutions Are Selected](03-architecture.md#4-how-solutions-are-selected-essentials).

### Priority cascade

The library tree narrows candidate solutions through two levels of priority.

At the **hardware level**, the tree first looks for solutions tuned for the exact product SKU, identified by PCI device ID (e.g., `gfx950_id75a3` = MI355X). If no exact-chip entry exists, it falls back to solutions tuned for the generic architecture (e.g., `gfx950`). An exact chip-ID match is always preferred, and the generic-architecture node serves as a fallback when chip-specific tuning is not available.

At the **strategy level** within each hardware node, EqualityMatching is checked before GridBasedMatching. If EqualityMatching has a benchmark-derived entry for the exact M, N, K dimensions of your problem, that solution wins -- it is the most precisely tuned result. Otherwise, GridBasedMatching provides a heuristic-ranked solution by interpolating from nearby data points. Additional strategies (Range, FreeSize) exist as lower-priority fallbacks; see Chapter 3 Section 4 for the full priority tree.

### User-driven tuning override

When the environment variable `HIPBLASLT_TUNING_OVERRIDE_FILE` is set, the heuristic path injects a user-specified solution index ahead of the heuristic results. This lets users lock a specific kernel for a workload after manual benchmarking. The override file includes a git version header that is validated against the library build to prevent stale overrides from silently producing suboptimal results.

---

## 4. Lazy loading `[Essentials]`

Lazy loading defers code object loading from startup to first use, reducing memory and startup time.

| Aspect       | Lazy ON (default)         | Lazy OFF                  |
|--------------|---------------------------|---------------------------|
| Startup      | Loads metadata only       | Loads all `.co` files     |
| First call   | Loads `.co` on first use  | No extra latency          |
| Memory       | Grows as kernels are used | Peak memory at startup    |
| Error timing | Missing `.co` at dispatch | Missing `.co` at init     |
| Library file | `TensileLibrary_lazy_<arch>.dat` | `TensileLibrary_<arch>.dat` |

### Debugging implications

First-call latency is normal with lazy loading and is not a performance bug -- each kernel's code object is loaded exactly once on its first dispatch, and subsequent calls incur no loading overhead. If you suspect a missing or misconfigured library path, set `HIPBLASLT_TENSILE_LIBPATH` to explicitly point to the directory containing the `.dat` and `.co` files; when info logging is enabled, the host will log the resolved path at initialization.

---

## 5. Adding a new API feature `[Deep Dive]`

This checklist traces the layers you must touch when adding a new feature (e.g., a new epilogue mode, a new data type attribute, or a new matmul descriptor field).

### Step 1: Public header

Add the new enum value, struct field, or function prototype to the appropriate header in `library/include/hipblaslt/`:

| Header | Content |
|---|---|
| `hipblaslt.h` | C API functions, opaque handle types, enums for epilogue/attributes |
| `hipblaslt-ext.hpp` | C++ extension classes (`Gemm`, `GroupedGemm`, `GemmEpilogue`, etc.) |
| `hipblaslt-ext-op.h` | ExtOp kernels (softmax, layernorm, amax) |
| `hipblaslt-types.h` | Shared type definitions, `hipblasLtEpilogue_t`, pointer modes |

### Step 2: hipblaslt implementation

Wire the new feature through the API implementation layer:

- **C API**: `library/src/amd_detail/hipblaslt.cpp` -- add the new function or modify an existing `hipblasLtMatmulDescSetAttribute` case to forward the new attribute.
- **C++ ext API**: `library/src/amd_detail/hipblaslt-ext.cpp` -- add getter/setter to the relevant pimpl class (`GemmEpilogueImpl`, `GemmProblemTypeImpl`, etc.) and update `GemmInstance` methods.

### Step 3: rocblaslt backend types

Update the internal types that carry the new information through the backend:

- `library/src/amd_detail/rocblaslt/include/rocblaslt-types.h` -- add the field to `rocblaslt_matmul_desc_struct` or the relevant internal struct.
- `library/src/amd_detail/rocblaslt/include/rocblaslt-auxiliary.h` -- declare the new rocblaslt-level function if adding a C API entry point.
- `library/src/amd_detail/rocblaslt/src/rocblaslt_auxiliary.cpp` -- implement attribute get/set and `construct_rocblaslt_problem()` plumbing.

### Step 4: Problem representation

Update `RocblasltContractionProblem` (defined in `rocblaslt-types.h`) to carry the new field. Then update:

- `updateTensileProblem()` -- translate the field into the Tensile problem representation.
- `GetTensileInputs()` -- if the feature adds new input pointers.

### Step 5: Argument validation

Add validation for the new feature in:

- `rocblaslt_matmul_valid_args()` (referenced from `rocblaslt_mat.cpp`)
- `validateMatmulArgs()` (for the ext API path in `rocblaslt_mat.cpp`)

### Step 6: Tests

1. Add test case definitions to `clients/tests/data/hipblaslt_gtest.yaml` or create a new YAML file.
2. Test data YAML files are processed by `clients/tests/hipblaslt_gentest.py` to generate C++ test sources at build time.
3. Run the relevant tests:
   ```bash
   ./build/clients/hipblaslt-test --gtest_filter=*your_feature*
   ```

### Step 7: Samples (optional)

Add or update a sample program in `clients/samples/`. The samples are numbered and each demonstrates a specific API usage pattern (basic GEMM, batched, tuning, bias, get-all-algos, etc.).

### Step 8: Bench support

If the feature adds new command-line-visible parameters, update `clients/bench/` to expose them through `hipblaslt-bench`.
