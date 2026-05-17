# Chapter 11 -- Reference Appendix

> **Note:** Line numbers reference the codebase at time of writing and may drift. Use function names as the primary search target.

This appendix collects source-level implementation detail for developers
actively working in the hipBLASLt codebase. For conceptual understanding, see
Chapters [3](03-architecture.md), [6](06-tensilelite-guide.md), and
[7](07-host-library-guide.md).

---

## 1. Host Library File Reference

> For a conceptual overview, see [Chapter 3, Section 2](03-architecture.md#2-how-a-gemm-call-becomes-a-kernel-essentials).

### Top-level files

Source lives in `library/src/amd_detail/`.

| File | Role |
|------|------|
| `hipblaslt.cpp` | Implements every `hipblasLt*` C function. Each function casts hipBLAS types to rocblaslt types and delegates. Status codes are translated by `RocBlasLtStatusToHIPStatus()`. |
| `hipblaslt-ext.cpp` | Implements the `hipblaslt_ext` C++ classes. `Gemm::run()` calls `rocblaslt_run_cpp()`. `GemmInstance::algoGetHeuristic()` calls `rocblaslt_algo_get_heuristic_cpp()`. |
| `hipblaslt-ext-op.cpp` | Implements ExtOp functions. Loads `.dat` code object bundles via `ExtOpMasterLibrary` and dispatches precompiled kernels for softmax, layernorm, and amax. |

### rocblaslt backend

Located in `library/src/amd_detail/rocblaslt/src/`.

| File | Role |
|------|------|
| `rocblaslt_mat.cpp` | Implements `rocblaslt_matmul()`, which constructs a `RocblasltContractionProblem` and calls `runContractionProblem()`. |
| `rocblaslt_auxiliary.cpp` | Handle create/destroy, descriptor management. |
| `rocblaslt_transform.cpp` | Matrix transform operations. |
| `tensile_host.cpp` | The TensileLite interface layer (described in Section 2 below). |
| `handle.cpp` | Internal handle state. |
| `UserDrivenTuningParser.cpp` | Parses user-driven tuning override files. |

The public header `rocblaslt.h` lives in `rocblaslt/include/`, while internal headers (`tensile_host.hpp`, `handle.h`, `rocblaslt_mat_utils.hpp`, etc.) live in `rocblaslt/src/include/`.

---

## 2. Request Lifecycle -- Detailed Call Chains

> For a conceptual overview, see [Chapter 3, Section 2](03-architecture.md#2-how-a-gemm-call-becomes-a-kernel-essentials).

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
3. **Problem construction** -- builds a `RocblasltContractionProblem` struct (defined in `rocblaslt-types.h`, line 467; constructor in `tensile_host.cpp`, line 79) that captures all GEMM parameters in a single object.
4. **Dispatch** -- calls `runContractionProblem(handle, algo, problem, gemmData)` (line 225).

### Layer 3: Backend dispatch (`library/src/amd_detail/rocblaslt/src/tensile_host.cpp`)

```
runContractionProblem()    -- line 2868
  |-- #ifdef HIPBLASLT_USE_ROCROLLER
  |     |-- useRocRoller(handle, prob) ?
  |     \-- YES: runRocRollerContractionProblem(handle, algo, prob)   [see Section 5: RocRoller Dispatch Details]
  |
  |-- get_library_and_adapter(&library, &deviceProp, &hardware)
  |     \-- initializes TensileHost singleton on first call (line 2623)
  |
  |-- if algo == nullptr:
  |     \-- getBestSolutions(prob, ..., 1, &heuristicResult, ...)
  |         \-- see Section 3: Algorithm Selection Internals for the full algorithm-selection flow
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

TensileLite's `SolutionAdapter` calls `hipExtModuleLaunchKernel()` to submit the precompiled code object kernel. Lazy loading (see Section 7) may trigger `hipModuleLoadData()` at this point if the code object was not yet loaded.

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

## 3. Algorithm Selection Internals

> For a conceptual overview, see [Chapter 3, Section 4](03-architecture.md#4-how-solutions-are-selected-essentials).

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
  |     \-- getRocRollerBestSolutions()   [see Section 5: RocRoller Dispatch Details]
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

## 4. TensileLite Solution Library Class Hierarchy

> For a conceptual overview, see [Chapter 3, Section 4](03-architecture.md#4-how-solutions-are-selected-essentials) and [Chapter 6, Section 1](06-tensilelite-guide.md#1-what-tensilelite-does-essentials).

The TensileLite solution library is implemented as a tree of template classes
that progressively narrow the solution search space. The key headers live in
`tensilelite/include/Tensile/`.

### Class tree

```
SolutionLibrary<MyProblem, MySolution>              (SolutionLibrary.hpp)
  Abstract base. Defines the virtual interface:
    - findBestSolution(problem, hardware)
    - findTopSolutions(problem, hardware, count)
    - findAllSolutions(problem, hardware)
    - getSolutionByIndex(problem, hardware, index)
  |
  +-- ExactLogicLibrary<MyProblem, MySolution, MyPredicate>   (ExactLogicLibrary.hpp)
  |     Ordered list of (predicate, sub-library) rows.
  |     Iterates rows in order; the first predicate match wins.
  |     |
  |     +-- HardwareSelectionLibrary                          (ExactLogicLibrary.hpp)
  |     |     Predicate type: HardwarePredicate
  |     |     Matches GPU architecture (AMDGPU processor, chip ID).
  |     |     Supports fallback matching for compatible chip IDs.
  |     |     type() returns "Hardware"
  |     |
  |     +-- ProblemSelectionLibrary                           (ExactLogicLibrary.hpp)
  |           Predicate type: ProblemPredicate<MyProblem>
  |           Matches problem properties (data types, transpose, epilogue, etc.).
  |           type() returns "Problem"
  |
  +-- MasterSolutionLibrary<MyProblem, MySolution>            (MasterSolutionLibrary.hpp)
        Root of the tree. Owns the solution map and delegates to
        a child `library` (typically a HardwareSelectionLibrary).
        Handles serialization, lazy library loading, and solution
        index lookup.
        type() returns "Master"
```

### A typical runtime tree

At runtime, the library loaded from `TensileLibrary_lazy_<arch>.dat` forms this structure:

```
MasterSolutionLibrary  (root -- owns the flat solution map)
  \-- HardwareSelectionLibrary  (rows keyed by GPU arch predicate)
        \-- ProblemSelectionLibrary  (rows keyed by problem-type predicate)
              \-- Matching libraries (EqualityMatching, GridBasedMatching,
                   RangeMatching, FreeSizeMatching, PredictionMatching, etc.)
                    \-- Individual ContractionSolution objects
```

### findBestSolution() priority logic

`ExactLogicLibrary::findBestSolution()` (ExactLogicLibrary.hpp) implements a
two-pass priority scheme:

1. **Exact match pass** -- Iterates rows in order. For each row whose predicate
   matches and is **not** a fallback (i.e., the GPU's chip ID is in the
   predicate's target set), calls `row.second->findBestSolution()`. Returns
   immediately on the first successful result.

2. **Fallback pass** -- If the predicate matches but `isFallbackMatch()` is true
   (the GPU's chip ID differs from the predicate's target), the result is saved
   as `fallbackRv` but iteration continues to look for an exact match. If no
   exact match is found, the first fallback result is returned.

This allows architecture-specific solutions to take priority while still falling
back to compatible solutions from a related chip when needed.

The `ExperimentalStreamK` row type is skipped unless `Debug::Instance().useExperimentalSelection() == 2`.

### findTopSolutions() behavior

`ExactLogicLibrary::findTopSolutions()` collects solutions across rows until
`numSolutions` results are accumulated:

1. Iterates rows in order (skipping `ExperimentalStreamK` unless enabled, and
   skipping `EqualityMatching`/`RangeMatching` when `usePredictionLibrary()` is
   active).
2. For each matching row, requests `numSolutions - rv.size()` from the
   sub-library.
3. Appends results, tagging each solution with a `MatchingTag` (`Equal`,
   `GridBased`, `Range`, `FreeSize`, `Prediction`) based on the predicate type.
4. Returns early once the requested count is reached, or sets
   `lastFindTopRetAll = true` if fewer solutions exist than requested.

### MasterSolutionLibrary lazy library loading

When lazy loading is enabled, `MasterSolutionLibrary::getSolutionByIndex(hardware, index)`
calls `loadLibrary(index)` before looking up the solution. `loadLibrary()`:

Note: the mapping file uses the `TensileLite` prefix while the main library file uses `Tensile` (without `Lite`) — this naming inconsistency exists in the build output.

1. Uses `libraryMapping` (loaded from `TensileLiteLibrary_lazy_<arch>_Mapping.dat`)
   to map the solution index to a shard filename.
2. Loads the shard file (e.g., `<prefix>.dat`) via `LoadLibraryFile()`.
3. Merges the shard's solutions into the master map and sets each solution's
   `codeObjectFilename` to `<prefix>.co`.
4. Caches the loaded sub-library in `indexLoadedLibraries` so it is not
   reloaded.

All operations are guarded by `solutionsGuard` (a `std::mutex`) for thread safety.

---

## 5. RocRoller Dispatch Details

> For a conceptual overview, see [Chapter 3, Section 3](03-architecture.md#3-the-two-dispatch-paths-essentials).

Source files live in `library/src/amd_detail/rocblaslt/src/rocroller/`.

### When RocRoller is used

The decision is made by `useRocRoller()` in `tensile_host.cpp` (line 2845):

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

---

## 6. Device Libraries

> Device libraries are listed in [Chapter 3, Section 6: Key Directory Map](03-architecture.md#6-key-directory-map-essentials). This section provides additional build-level detail.

Located at `device-library/`.

```
device-library/
  CMakeLists.txt
  extops/                Precompiled ExtOp kernels
    CMakeLists.txt         (softmax, layernorm, amax)
  matrix-transform/      Precompiled matrix transform kernels
    CMakeLists.txt
    matrix_transform.cpp
    matrix_transform.h
```

- **ExtOps** -- Built into `hipblasltExtOpLibrary.dat` code object bundles
  (one per architecture plus a combined bundle).  Loaded at runtime by
  `hipblaslt-ext-op.cpp` via `ExtOpMasterLibrary`.

- **Matrix transform** -- Provides layout conversion kernels (e.g., column-
  major to tile-swizzled layouts).

- **GEMM code objects** -- The main GEMM kernels are not stored here; they
  are built by the TensileLite toolchain and installed alongside the logic
  files.  The `device-library/` tree holds only the non-GEMM precompiled
  kernels.

The CMake option `HIPBLASLT_ENABLE_DEVICE=ON` (default) controls whether
device libraries are built.

---

## 7. Lazy Loading Internals

> For a conceptual overview, see [Chapter 7, Section 4](07-host-library-guide.md#4-lazy-loading-essentials) (Senior path chapter).

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

### TensileHost::initialize() differences

The `TensileHost::initialize()` function (in `tensile_host.cpp`, line 2420) behaves differently based on the flag:

**Lazy loading ON (`ROCBLASLT_TENSILE_LAZY_LOAD=1`)**:

- Loads the library metadata from `TensileLibrary_lazy_<arch>.dat` (line 2523). This file contains the solution library tree and problem predicates but does **not** embed the kernel binaries.
- Code object files are **not** loaded at startup. The path is recorded for later on-demand loading.
- The `TensileHost` maintains per-architecture maps (`m_devicePropMap`, `m_hardwareMap`, `m_deviceSet` at lines 2335-2337) to support multi-GPU systems where different architectures may be present.
- When a kernel is first dispatched, `SolutionAdapter` calls `hipModuleLoadData()` to load just that specific code object into GPU memory.

**Lazy loading OFF (`ROCBLASLT_TENSILE_LAZY_LOAD=0`)**:

- Loads all `.co` files matching the current architecture from the library directory at startup (lines 2482-2503). Each file is loaded via `adapter.loadCodeObjectFile()`.
- Loads the full library metadata from `TensileLibrary_<arch>.dat` (no `_lazy_` prefix, line 2535).
- Uses a single `m_deviceProp` and `m_hardware` (lines 2339-2340) since all devices must be the same architecture.

**In both modes**, `adapter.initializeLazyLoading(processor, path)` (line 2602) is called unconditionally -- it runs outside the `#if ROCBLASLT_TENSILE_LAZY_LOAD` guards. The call records the library path on the adapter regardless of whether lazy loading will actually defer code object loading.

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
