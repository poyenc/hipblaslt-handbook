# Chapter 7 -- Host Library Guide

This chapter traces the host-side code path from a public API call down to GPU kernel dispatch. Every function name and file path references the actual source tree under `projects/hipblaslt/`.

---

## 1. Request Lifecycle [Deep Dive]

A `hipblasLtMatmul()` call passes through four layers before a GPU kernel is launched. The two main dispatch backends -- TensileLite (precompiled) and RocRoller (JIT) -- diverge at the `runContractionProblem` entry point.

### Layer 1: Public API entry (`library/src/amd_detail/hipblaslt.cpp`)

```
hipblasLtMatmul()          -- line 476
  |-- Debug marker: rocblaslt::Debug::Instance().markerStart("hipblasLtMatmul")
  |-- Cast all hipblasLt opaque types to rocblaslt types
  \-- rocblaslt_matmul()   -- delegates to rocblaslt backend
```

The function casts each opaque handle (`hipblasLtHandle_t` to `rocblaslt_handle`, `hipblasLtMatmulDesc_t` to `rocblaslt_matmul_desc`, etc.) and calls `rocblaslt_matmul()`. Status codes are translated back through `RocBlasLtStatusToHIPStatus()` (line 81).

### Layer 2: rocblaslt validation (`library/src/amd_detail/rocblaslt/src/rocblaslt_mat.cpp`)

```
rocblaslt_matmul()         -- line 683
  |-- null-pointer and type-mismatch checks
  |-- logging (log_api / log_trace)
  \-- rocblaslt_matmul_impl()  -- line 43, same file
```

`rocblaslt_matmul_impl()` does the heavy lifting at this layer:

1. **Argument validation** -- calls `rocblaslt_matmul_valid_args()` which extracts dimensions (m, n, k), leading dimensions, batch strides, data types, epilogue parameters, and bias/scale pointers from the descriptor structs.
2. **scaleAlphaVec handling** -- when a per-column alpha vector is set, the scalar alpha is forced to 1.0 and the vector is passed to the kernel instead.
3. **Problem construction** -- builds a `RocblasltContractionProblem` struct (defined in `tensile_host.cpp`, line 79) that captures all GEMM parameters in a single object.
4. **Dispatch** -- calls `runContractionProblem(handle, algo, problem, gemmData)` (line 225).

### Layer 3: Backend dispatch (`library/src/amd_detail/rocblaslt/src/tensile_host.cpp`)

```
runContractionProblem()    -- line 2868
  |-- #ifdef HIPBLASLT_USE_ROCROLLER
  |     |-- useRocRoller(handle, prob) ?
  |     \-- YES: runRocRollerContractionProblem(handle, algo, prob)   [see Section 4]
  |
  |-- get_library_and_adapter(&library, &deviceProp, &hardware)
  |     \-- initializes TensileHost singleton on first call (line 2624)
  |
  |-- if algo == nullptr:
  |     \-- getBestSolutions(prob, ..., 1, &heuristicResult, ...)
  |         \-- library->findTopSolutions(tensileProblem, hardware, count)
  |
  |-- updateTensileProblem(prob, data->problem)   -- line 1846, translates RocblasltContractionProblem to TensileLite::ContractionProblemGemm
  |-- data->inputs = GetTensileInputs(prob)       -- line 2110, maps pointers/scalars to Tensile input struct
  |
  |-- library->getSolutionByIndex(data->problem, hardware, solutionIndex)
  |-- workspace size check
  |-- solution->solve(data->problem, inputs, hardware)   -- returns KernelInvocation vector
  \-- adapter->launchKernels(kernels)  -- enqueues HIP kernel launches
```

Key helper functions at this layer:

| Function | Line | Purpose |
|---|---|---|
| `updateTensileProblem()` | 1846 | Translates matrix dimensions, strides, types, and epilogue settings from `RocblasltContractionProblem` into `TensileLite::ContractionProblemGemm` |
| `GetTensileInputs()` | 2110 | Maps data pointers (A, B, C, D, bias, scales) and alpha/beta values into `TensileLite::ContractionInputs` |
| `get_library_and_adapter()` | 2613 | Returns the TensileLite library and per-device `SolutionAdapter`, initializing the `TensileHost` singleton on first use |

### Layer 4: Kernel launch

TensileLite's `SolutionAdapter` calls `hipModuleLaunchKernel()` to submit the precompiled code object kernel. Lazy loading (see Section 5) may trigger `hipModuleLoadData()` at this point if the code object was not yet loaded.

### The ext API path

The C++ extension API (`hipblaslt-ext.hpp`) provides an alternative entry point through `hipblaslt_ext::Gemm` and `hipblaslt_ext::GroupedGemm`. The call chain is:

```
Gemm::algoGetHeuristic()     -- hipblaslt-ext.cpp, line 737
  \-- rocblaslt_algo_get_heuristic_cpp()

Gemm::initialize(algo, workspace, ...)  -- line 815
  \-- rocblaslt_makeArgument_cpp()
      \-- makeArgument()               -- tensile_host.cpp

Gemm::run(stream)                       -- line 882
  \-- rocblaslt_run_cpp()
      \-- runKernelFromInvocation()     -- tensile_host.cpp, line 3442
```

The ext API separates problem setup (`initialize`) from execution (`run`), allowing the kernel invocation to be cached and replayed without re-solving. `rocblaslt_run_cpp()` (in `rocblaslt_mat.cpp`, line 1426) delegates to `runKernelFromInvocation()`.

---

## 2. Algorithm Selection Internals [Deep Dive]

### Entry points

There are two public APIs for algorithm selection:

1. **C API**: `hipblasLtMatmulAlgoGetHeuristic()` (in `hipblaslt.cpp`, line 431)
   -> `rocblaslt_matmul_algo_get_heuristic()` (in `rocblaslt_auxiliary.cpp`, line 1876)
   -> constructs a `RocblasltContractionProblem` via `construct_rocblaslt_problem()`
   -> calls `getBestSolutions()`

2. **C++ ext API**: `GemmInstance::algoGetHeuristic()` (in `hipblaslt-ext.cpp`, line 737)
   -> `rocblaslt_algo_get_heuristic_cpp()` (in `rocblaslt_auxiliary.cpp`, line 2176)

Both paths converge on `getBestSolutions()` in `tensile_host.cpp` (line 3910).

### How getBestSolutions() works

```
getBestSolutions()  -- tensile_host.cpp, line 3910
  |-- if HIPBLASLT_USE_ROCROLLER && useRocRoller(handle, prob):
  |     \-- getRocRollerBestSolutions()   [see Section 4]
  |
  |-- get_library_and_adapter(&library, &deviceProp, &hardware)
  |-- updateTensileProblem(prob, data->problem)
  |-- getSolutions()  -- line 3858, template wrapper
  |     \-- library->findTopSolutions(tensileProblem, hardware, requestedAlgoCount)
  |
  |-- xf32 fallback: if no solutions found and compute_type == f32_fast_xf32,
  |     retry with plain f32 via data->problem.setF32XdlMathOp(rocisa::DataType::Float)
  |
  \-- _convertToHeuristicResultArray()  -- line 3830
        |-- packs solution index into algo.data[0..3]
        |-- records requiredWorkspaceSize per solution
        \-- sets algo.max_workspace_bytes and state
```

### Logic file lookup

`library->findTopSolutions()` walks the `MasterSolutionLibrary` tree, which is populated from logic files at library initialization time. The logic files live in:

```
library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/<arch>/
```

Each YAML file describes solution kernels and the problem predicates (data types, transpose modes, sizes, epilogue) that select them. The library is serialized to msgpack (`.dat`) format for production use, loaded as either `TensileLibrary_lazy_<arch>.dat` or `TensileLibrary_<arch>.dat` depending on lazy loading mode.

### Solution filtering

`findTopSolutions()` filters the solution candidates by matching problem properties:

- Data types (A, B, C, D) and compute type
- Transpose modes (op_a, op_b)
- Epilogue type (bias, activation, etc.)
- Batch mode (strided, grouped)
- Scale format (scalar, vector, block scaling)

Solutions are ranked by the library's internal heuristic (typically based on predicted performance for the problem dimensions), and the top `requestedAlgoCount` results are returned.

### Workspace requirements

Each `ContractionSolution` reports its required workspace via `solution->requiredWorkspaceSize(problem, hardware)`. This is stored in `heuristicResult.workspaceSize` and later validated at dispatch time: if the user-provided workspace is smaller than the solution requires, `runContractionProblem()` returns `rocblaslt_status_invalid_value` (line 2970).

### User-driven tuning override

The heuristic path supports a file-based override mechanism. When the environment variable `HIPBLASLT_TUNING_OVERRIDE_FILE` is set, `rocblaslt_matmul_algo_get_heuristic()` calls `problem_override_from_file()` (in `rocblaslt_auxiliary.cpp`) to inject a user-specified solution index ahead of the heuristic results. The file's git version header is validated against the library build via `override_path_compare_git_version()` (in `hipblaslt.cpp`, line 45) to prevent stale overrides.

---

## 3. Adding a New API Feature [Deep Dive]

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

Update `RocblasltContractionProblem` (constructor in `tensile_host.cpp`, line 79) to carry the new field. Then update:

- `updateTensileProblem()` (line 1846) -- translate the field into the Tensile problem representation.
- `GetTensileInputs()` (line 2110) -- if the feature adds new input pointers.

### Step 5: Argument validation

Add validation for the new feature in:

- `rocblaslt_matmul_valid_args()` (referenced from `rocblaslt_mat.cpp`, line 72)
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

---

## 4. RocRoller JIT Path [Deep Dive]

### When RocRoller is used

RocRoller provides JIT-compiled GEMM kernels as an alternative to TensileLite's precompiled code objects. The decision is made by `useRocRoller()` in `tensile_host.cpp` (line 2845):

```cpp
bool useRocRoller(rocblaslt_handle handle, const RocblasltContractionProblem& prob)
{
    // Excluded: FP4 A + FP4 B with pre-swizzled (shuffled) scale layout
    if(isFp4A && isFp4B && isShuffledScale)
        return false;

    return handle->useRocRoller == 1
           || (handle->useRocRoller == -1
               && (isBlockScaling(prob.scaleAType) || isBlockScaling(prob.scaleBType)));
}
```

In practice, RocRoller is used when:

1. The handle's `useRocRoller` flag is explicitly set to `1`, OR
2. The flag is `-1` (auto-detect, the default) AND the problem uses **block scaling** (MX format scales), excluding the specific case of FP4-A + FP4-B with pre-swizzled scale layouts.

The `isBlockScaling()` helper is defined in `library/src/amd_detail/rocblaslt/src/include/utility.hpp` (line 418).

The feature is gated by the CMake option `HIPBLASLT_ENABLE_ROCROLLER` (default `ON`, `CMakeLists.txt` line 86), which defines the `HIPBLASLT_USE_ROCROLLER` compile-time macro.

### RocRoller dispatch flow

Source files live in `library/src/amd_detail/rocblaslt/src/rocroller/`. The dispatch flow (from `rocroller_host.cpp`):

```
runRocRollerContractionProblem()         -- line 686
  |-- if algo == nullptr:
  |     \-- getRocRollerBestSolutions()  -- line 475
  |
  |-- getKernelFromAlgo()                -- line 625
  |     |-- genKernelType(prob)          -- line 386, maps problem to KernelType
  |     |-- indexToParameters(solutionIndex) -> SolutionIndexParameters
  |     |-- cache.getKernel(kernelType, params, dims)
  |     \-- if miss: genKernelFromSolutionIndexParameters() -> JIT compile
  |
  \-- kernel->run(prob)                  -- launches the HIP kernel
```

### Solution selection with Origami

When finding best solutions, `getRocRollerBestSolutions()` calls `chooseSolutionIndexParameters()` (in `solution_selection.cpp`, line 211). This function:

1. Generates a list of candidate tile configurations (`origami::config_t`) via `generateTileList()` -- each config specifies macro-tile dimensions (MT_M, MT_N, MT_K), matrix instruction size, and cache hint variants.
2. Calls `origami::rank_configs()` with the problem dimensions and hardware description. Origami is an analytical performance model that predicts kernel throughput for each tile configuration.
3. Returns `SolutionIndexParameters` in ranked order (best predicted performance first).

Tile sizes are defined as compile-time arrays in `solution_selection.cpp`: `possibleTileSizes` (35 entries, line 28) for standard GEMM and `possibleSwizzleTileSizes` (37 entries, line 39) for swizzled-data GEMM. For each tile, three nontemporal-hint variants are generated (both off, A-only, B-only).

### Kernel caching

The `SolutionCache` (defined in `solution_cache.hpp`) stores generated kernels keyed by `(KernelType, SolutionIndexParameters, ProblemDims)`. On a cache hit, the pre-compiled kernel is returned directly. On a miss, `genKernelFromSolutionIndexParameters()` invokes RocRoller to JIT-compile a new kernel, which is then stored in the cache for future reuse.

The cache is owned by `RocRollerHandle` (created in `rocroller_create_handle()`, line 46), which is attached to each `rocblaslt_handle`. Custom precompiled kernels can be pre-loaded into the cache via `preloadCustomKernels()` (line 53) unless `HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL` is set.

### Key source files

| File | Purpose |
|---|---|
| `rocroller_host.cpp` | Entry points: `runRocRollerContractionProblem`, `getRocRollerBestSolutions`, handle create/destroy |
| `solution_selection.cpp` | `chooseSolutionIndexParameters()`, tile list generation, Origami integration |
| `parameter_selection.cpp` | Selects full `SolutionParameters` from `SolutionIndexParameters` and `KernelType` |
| `runtime_args_selection.cpp` | Selects runtime parameters (e.g., StreamK grid sizing) before kernel launch |
| `gemm.cpp` | GEMM kernel generation via RocRoller |
| `solution_cache.cpp` | `SolutionCache` implementation |
| `custom_kernels.cpp` | Pre-compiled custom kernels loaded as alternatives to JIT |

### Adding/modifying JIT kernel selection

To change which tile sizes or parameters RocRoller considers:

1. Edit the `possibleTileSizes` or `possibleSwizzleTileSizes` arrays in `solution_selection.cpp`.
2. To add a new parameter dimension to `SolutionIndexParameters`, update `include/solution_selection.hpp` and the `parametersToIndex()`/`indexToParameters()` encoding functions (in `solution_selection.cpp`, lines 372 and 403).
3. To change how parameters are selected for a given tile, modify `parameter_selection.cpp`.
4. To change runtime arguments (grid size, StreamK partitioning), modify `runtime_args_selection.cpp`.

---

## 5. Lazy Loading [Deep Dive]

### What it does

Lazy loading defers the loading of GPU code objects (`.co` files) from library initialization time until the moment a specific kernel is first needed. This reduces initial memory consumption and startup time, which matters because hipBLASLt ships hundreds of precompiled kernels per GPU architecture.

### CMake configuration

The feature is controlled by a single CMake option in `CMakeLists.txt`:

```cmake
option(HIPBLASLT_ENABLE_LAZY_LOAD
    "Enable lazy loading of runtime code object files to reduce ram usage."
    ON)                                                         # line 79
```

This sets the compile definition:

```cmake
ROCBLASLT_TENSILE_LAZY_LOAD=$<BOOL:${HIPBLASLT_ENABLE_LAZY_LOAD}>   # line 349
```

Throughout `tensile_host.cpp`, `#if ROCBLASLT_TENSILE_LAZY_LOAD` guards differentiate the two modes.

### Initialization differences

The `TensileHost::initialize()` function (in `tensile_host.cpp`, line 2420) behaves differently based on the flag:

**Lazy loading ON (`ROCBLASLT_TENSILE_LAZY_LOAD=1`)**:

- Loads the library metadata from `TensileLibrary_lazy_<arch>.dat` (line 2523). This file contains the solution library tree and problem predicates but does **not** embed the kernel binaries.
- Code object files are **not** loaded at startup. The `SolutionAdapter` is initialized with `adapter.initializeLazyLoading(processor, path)` (line 2602), which records the path for later on-demand loading.
- The `TensileHost` maintains per-architecture maps (`m_devicePropMap`, `m_hardwareMap`, `m_deviceSet` at lines 2335-2337) to support multi-GPU systems where different architectures may be present.
- When a kernel is first dispatched, `SolutionAdapter` calls `hipModuleLoadData()` to load just that specific code object into GPU memory.

**Lazy loading OFF (`ROCBLASLT_TENSILE_LAZY_LOAD=0`)**:

- Loads all `.co` files matching the current architecture from the library directory at startup (lines 2482-2503). Each file is loaded via `adapter.loadCodeObjectFile()`.
- Loads the full library metadata from `TensileLibrary_<arch>.dat` (no `_lazy_` prefix, line 2535).
- Uses a single `m_deviceProp` and `m_hardware` (lines 2339-2340) since all devices must be the same architecture.

### Memory and startup impact

With lazy loading ON:
- Startup allocates only the library metadata tree (solution predicates and indices).
- Each kernel's code object is loaded on first use, adding a one-time latency to the first dispatch of that kernel.
- Resident GPU memory grows incrementally as different kernels are used.

With lazy loading OFF:
- All code objects for the current architecture are loaded into GPU memory at startup.
- First dispatch of any kernel has no additional loading latency.
- Peak memory is reached immediately at initialization.

### Implications for debugging

When debugging kernel dispatch issues with lazy loading enabled:

1. **First-call latency** -- The first invocation of a new kernel type will be slower due to code object loading. This is normal and not a performance bug. Subsequent calls with the same kernel will not incur this overhead.

2. **Missing code objects** -- If a `.co` file is missing from the library path, the error manifests at dispatch time rather than at initialization. Set `HIPBLASLT_TENSILE_LIBPATH` to verify the library path, and check that the architecture-specific code objects are present.

3. **Multi-GPU considerations** -- With lazy loading, the `TensileHost` initializes device property maps for all detected GPUs. If debugging on a multi-GPU system, verify that the correct architecture is being resolved via `rocblaslt_internal_get_arch_name()`.

4. **Disabling for diagnosis** -- To rule out lazy-loading-related issues, rebuild with `-DHIPBLASLT_ENABLE_LAZY_LOAD=OFF` to force eager loading of all code objects at startup.

5. **Environment variable** -- `HIPBLASLT_TENSILE_LIBPATH` overrides the default library search path. When set, the host logs `Using HIPBLASLT_TENSILE_LIBPATH=<path>` (if info logging is enabled). When unset, the path is resolved relative to `librocblaslt.so` or falls back to `/opt/rocm/lib`.
