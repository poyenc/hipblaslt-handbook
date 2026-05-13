# Chapter 3: Architecture

This chapter describes how hipBLASLt is structured -- from the public API headers
down to the GPU kernel code objects that execute on hardware.  Understanding
this stack is the fastest way to figure out where a given behavior lives and
what you need to rebuild after a change.


## 1. Layer overview `[Essentials]`

```
 User application
       |
       v
 +-------------------------------+
 | Public API                    |   library/include/hipblaslt/
 |  hipblaslt.h        (C)      |     hipblasLtMatmul, hipblasLtCreate, ...
 |  hipblaslt-ext.hpp  (C++)    |     hipblaslt_ext::Gemm, GroupedGemm
 |  hipblaslt-ext-op.h (C)      |     hipblasltExtSoftmax, hipblasltExtLayerNorm, ...
 +-------------------------------+
       |
       v
 +-------------------------------+
 | Host library                  |   library/src/amd_detail/
 |  hipblaslt.cpp                |     Casts hipblasLt types to rocblaslt types,
 |  hipblaslt-ext.cpp            |     delegates every call to the rocblaslt backend
 |  hipblaslt-ext-op.cpp         |     ExtOp kernels load from .dat code object bundles
 +-------------------------------+
       |  rocblaslt_matmul()
       |  rocblaslt_run_cpp()
       v
 +-------------------------------+
 | rocblaslt backend             |   library/src/amd_detail/rocblaslt/src/
 |  rocblaslt_mat.cpp            |     Matmul execution: builds contraction problem,
 |  tensile_host.cpp             |     TensileLite dispatch: solution lookup,
 |                               |       code object loading, kernel launch
 +-------------------------------+
       |                    |
       |  (default path)    |  (block-scaling / env override)
       v                    v
 +-----------------+  +--------------------+
 | TensileLite     |  | RocRoller          |   library/src/amd_detail/rocblaslt/
 | Host Runtime    |  | Integration        |     src/rocroller/
 | tensilelite/    |  |  rocroller_host.cpp |
 |  src/           |  |  gemm.cpp          |   JIT-compiles kernels at runtime
 |  include/       |  |  solution_cache.cpp |   via the RocRoller compiler
 +-----------------+  +--------------------+
       |
       |  loadCodeObjectFile() / initializeLazyLoading()
       v
 +-------------------------------+
 | Logic files + Device libs     |
 |  Logic/asm_full/<arch>/       |   YAML: problem-size -> solution mapping
 |  device-library/              |   Precompiled .co / .dat code objects
 +-------------------------------+
       |
       v
 +-------------------------------+
 | TensileLite Python + rocisa   |   tensilelite/Tensile/   (offline toolchain)
 |  KernelWriter*.py             |   tensilelite/rocisa/
 |  Generates assembly kernels   |   ISA assembler (nanobind C++ extension)
 +-------------------------------+
```

### How an API call flows to a kernel launch

1. The application calls `hipblasLtMatmul()`.
2. `hipblaslt.cpp` casts every handle/descriptor to its `rocblaslt_*` counterpart
   and calls `rocblaslt_matmul()`.
3. `rocblaslt_matmul()` (in `rocblaslt_mat.cpp`) builds a
   `RocblasltContractionProblem` and calls `runContractionProblem()`.
4. `runContractionProblem()` (in `tensile_host.cpp`) decides the dispatch path:
   - If RocRoller is enabled and `useRocRoller()` returns true, it calls
     `runRocRollerContractionProblem()` for JIT kernel generation.
   - Otherwise it uses `get_library_and_adapter()` to obtain the TensileLite
     `MasterSolutionLibrary` and `SolutionAdapter`, looks up the best solution
     via `findTopSolutions()`, and launches the precompiled kernel.
5. The C++ extension API (`hipblaslt_ext::Gemm::run()`) follows the same path
   but through `rocblaslt_run_cpp()`.


## 2. Public API layer `[Essentials]`

hipBLASLt exposes three API surfaces.  All live under
`library/include/hipblaslt/`.

### C API -- `hipblaslt.h`

The primary API.  Modeled after NVIDIA's cublasLt, it provides:

- **Handle management**: `hipblasLtCreate`, `hipblasLtDestroy`
- **Descriptors**: `hipblasLtMatrixLayoutCreate`, `hipblasLtMatmulDescCreate`,
  `hipblasLtMatmulPreferenceCreate` and their Set/Get/Destroy variants
- **Algorithm selection**: `hipblasLtMatmulAlgoGetHeuristic`
- **Execution**: `hipblasLtMatmul`
- **Matrix transform**: `hipblasLtMatrixTransform`

Status codes, epilogue enums (`hipblasLtEpilogue_t`), pointer modes, and
scale modes are also defined here.

### C++ extension API -- `hipblaslt-ext.hpp`

A higher-level C++ interface in the `hipblaslt_ext` namespace.  It wraps
the C API workflow into classes:

- `GemmPreference` -- workspace size limits
- `GemmProblemType` -- operation, data types, layout order
- `GemmEpilogue` -- activation, bias, scaling configuration
- `Gemm` / `GroupedGemm` -- end-to-end GEMM objects with
  `algoGetHeuristic()`, `initialize()`, and `run()` methods
- Utility functions: `getAllAlgos()`, `getAlgosFromIndex()`,
  `matmulIsAlgoSupported()`

### Extension operations -- `hipblaslt-ext-op.h`

Standalone C functions for non-GEMM operations:

- `hipblasltExtSoftmax` -- 2-D softmax
- `hipblasltExtLayerNorm` -- 2-D layer normalization
- `hipblasltExtAMax` -- absolute maximum reduction

These do not go through TensileLite.  They load precompiled kernels from
`hipblasltExtOpLibrary.dat` code object bundles under the device-library
install path.

> **Essentials-only readers:** Skip ahead to [Section 10 (Key directory map)](#10-key-directory-map-essentials) for a quick reference of all directories, then return to the Deep Dive sections as needed.


## 3. Host library `[Deep Dive]`

Source lives in `library/src/amd_detail/`.

### Top-level files

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
| `tensile_host.cpp` | The TensileLite interface layer (described in the next section). |
| `handle.cpp` | Internal handle state. |
| `UserDrivenTuningParser.cpp` | Parses user-driven tuning override files. |

The `include/` subdirectory holds internal headers: `rocblaslt.h`, `tensile_host.hpp`,
`handle.h`, `rocblaslt_mat_utils.hpp`, etc.


## 4. TensileLite runtime `[Deep Dive]`

### C++ host library

Source and headers live at:

- `tensilelite/src/` -- C++ implementation files (`ContractionProblem.cpp`,
  `ContractionSolution.cpp`, `Tensile.cpp`, `EmbeddedLibrary.cpp`, HIP
  adapter code in `src/hip/`, etc.)
- `tensilelite/include/Tensile/` -- Public headers consumed by the rocblaslt
  backend (`MasterSolutionLibrary.hpp`, `ContractionProblem.hpp`,
  `hip/HipSolutionAdapter.hpp`, `PlaceholderLibrary.hpp`, etc.)

This library is built as the CMake target `tensilelite::tensilelite-host`.

### How tensile_host.cpp uses it

`tensile_host.cpp` is the only file in the rocblaslt backend that includes
TensileLite headers.  Key interactions:

1. **Library initialization** -- `RocblasltTensileHost::initialize()` loads
   the `MasterSolutionLibrary` from YAML logic files or msgpack bundles and
   calls `adapter.initializeLazyLoading()` to set up on-demand code object
   loading.

2. **Solution lookup** -- `getBestSolutions()` calls
   `library->findTopSolutions()` (or `findAllSolutions()` /
   `findAllSolutionsGroupedGemm()`) to query the solution library for the
   best kernels matching a `ContractionProblemGemm`.

3. **Kernel dispatch** -- `runContractionProblem()` obtains a
   `SolutionAdapter`, prepares `ContractionInputs`, and launches the
   selected kernel on the HIP stream.

4. **Lazy loading** -- With `HIPBLASLT_ENABLE_LAZY_LOAD=ON` (default), code
   objects are loaded on first use via `PlaceholderLibrary`, reducing startup
   memory.

The host maintains a per-device adapter array (`m_adapters`) protected by
thread-safe locking so multiple threads safely share the same library.


## 5. TensileLite Python toolchain `[Deep Dive]`

Located at `tensilelite/Tensile/`.  This is the offline toolchain that
generates optimized GEMM kernels.  It is not part of the runtime library.

Key subsystems:

| Directory / file | Purpose |
|---|---|
| `KernelWriter.py` | Main entry point: dispatches to specialized kernel writers |
| `KernelWriterAssembly.py` | Emits GPU ISA assembly (CDNA and RDNA) for GEMM kernels |
| `KernelWriterBase.py` | Base class for kernel writers |
| `KernelWriterModules.py` | Composable instruction-sequence modules |
| `Components/` | Pluggable components: MAC units, local read, global write batch, GSU, etc. |
| `Common/` | Shared constants, architecture capabilities, data types, global parameters |
| `Contractions.py` | Contraction problem definition |
| `BenchmarkStructs.py` | Benchmark configuration parsing |
| `ClientWriter.py` | Generates C++ client code for validation |
| `CustomKernels/` | Hand-written kernel templates |
| `bin/Tensile` | Entry point script for running kernel generation |
| `Tests/` | pytest test suites (common, unit) |

The toolchain outputs:

- Assembly source files (.s)
- Compiled code objects (.co)
- Logic files (YAML) mapping problem configurations to solutions


## 6. rocisa `[Deep Dive]`

Located at `tensilelite/rocisa/`.

rocisa is a Python/C++ ISA code generator built with nanobind.  It provides
Python bindings to ROCm ISA primitives -- registers, instructions, data types
-- so the TensileLite kernel writers can construct assembly programs from
Python without string manipulation.

Key points:

- Built with `invoke rocisa` from the tensilelite root.
- Uses scikit-build-core for cmake-based compilation.
- Exports a `rocisa` Python module with submodules under
  `tensilelite/rocisa/rocisa/`.
- Also contains the stinkytofu C++ code (typed enums for DPP/MFMA modifier
  fields, etc.).  Changes to stinkytofu C++ require `invoke rocisa` to
  rebuild.
- An `ImportError` with a stale-source hint is raised if the compiled
  extension is out of date.


## 7. Logic files `[Deep Dive]`

Located at `library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/`.

These YAML files define the mapping from problem descriptions (data types,
transpose modes, sizes) to specific kernel solutions.  They are organized
by GPU architecture:

```
asm_full/
  aldebaran/        (gfx90a -- MI210/MI250)
  aquavanjaram/     (gfx942 -- MI300)
  arcturus/         (gfx908 -- MI100)
  gfx950/
  gfx1103/
  gfx1150/
  gfx1151/
  gfx1152/
  gfx1153/
  gfx1200/
  gfx1201/
  gfx1250/
  navi31/           (gfx1100)
  navi32/           (gfx1101)
  navi33/           (gfx1102)
```

Older architectures use internal codenames (e.g. `aldebaran` for gfx90a); newer ones use the `gfx` identifier directly.

Each architecture directory contains one or more YAML files describing
solution libraries.  At build time these are converted to a binary format
(msgpack) that the TensileLite runtime loads.  At runtime, the
`MasterSolutionLibrary` uses matching predicates (data type, transpose,
tile size) to select the best solution for a given problem.


## 8. Device libraries `[Deep Dive]`

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


## 9. RocRoller integration `[Deep Dive]`

Located at `library/src/amd_detail/rocblaslt/src/rocroller/`.

RocRoller is a JIT kernel compiler.  When enabled
(`HIPBLASLT_ENABLE_ROCROLLER=ON`, default), it provides an alternative
dispatch path for GEMM kernels that are generated at runtime rather than
loaded from precompiled code objects.

### When RocRoller is used

The decision is made by `useRocRoller()` in `tensile_host.cpp`:

- `handle->useRocRoller == 1` -- forced on (via environment variable)
- `handle->useRocRoller == -1` (auto) and the problem uses block scaling
  (`ScalingFormat::Block_*`) -- RocRoller is used for block-scaled GEMM
- Exception: FP4 A (`HIP_R_4F_E2M1`) + FP4 B with `Block_32_UE8M0_32_8_EXT`
  scale format (pre-swizzled 32x8 layout) on both A and B falls back to
  TensileLite even when RocRoller would otherwise be selected

### Key files

| File | Role |
|------|------|
| `rocroller_host.cpp` | Entry point: `rocroller_create_handle()`, `runRocRollerContractionProblem()` |
| `gemm.cpp` | Translates a `RocblasltContractionProblem` into RocRoller kernel generation calls |
| `parameter_selection.cpp` | Selects `SolutionParameters` (tile size, etc.) using Origami heuristics |
| `solution_selection.cpp` | Chooses among candidate `SolutionIndexParameters` |
| `solution_cache.cpp` | Caches compiled kernels (`SolutionCache`) to avoid regeneration |
| `runtime_args_selection.cpp` | Selects runtime parameters (e.g., StreamK grid size) |
| `custom_kernels.cpp` | Preloaded hand-optimized kernels used in place of JIT when available |
| `include/` | Internal headers: `gemm.hpp`, `kernel_type.hpp`, `solution_cache.hpp`, etc. |

The RocRoller compiler itself lives outside hipBLASLt at
`../../shared/rocroller` in the monorepo.


## 10. Key directory map `[Essentials]`

| Directory | Purpose |
|---|---|
| `library/include/hipblaslt/` | Public API headers (C, C++, type headers) |
| `library/src/amd_detail/` | Top-level API implementation files |
| `library/src/amd_detail/rocblaslt/src/` | rocblaslt backend (handle, matmul, TensileLite dispatch) |
| `library/src/amd_detail/rocblaslt/src/include/` | Internal backend headers |
| `library/src/amd_detail/rocblaslt/src/rocroller/` | RocRoller JIT integration |
| `library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/` | Logic files (YAML) per GPU architecture |
| `tensilelite/src/` | TensileLite C++ host library source |
| `tensilelite/include/Tensile/` | TensileLite C++ host library headers |
| `tensilelite/Tensile/` | TensileLite Python toolchain (kernel generation) |
| `tensilelite/Tensile/Components/` | Pluggable kernel-writer components |
| `tensilelite/Tensile/Common/` | Shared Python utilities and constants |
| `tensilelite/Tensile/CustomKernels/` | Hand-written kernel templates |
| `tensilelite/rocisa/` | rocisa Python/C++ ISA code generator |
| `device-library/extops/` | Precompiled ExtOp kernels (softmax, layernorm, amax) |
| `device-library/matrix-transform/` | Precompiled matrix transform kernels |
| `clients/tests/` | gtest suite (`hipblaslt-test`) |
| `clients/bench/` | Benchmarking tool (`hipblaslt-bench`) |
| `clients/samples/` | API usage examples |
