# Chapter 3: Architecture

This chapter defines the core concepts used throughout hipBLASLt and traces
the path a GEMM call takes from the public API down to a GPU kernel launch.
If you need a refresher on what a GEMM operation is, see
[Chapter 1: What is hipBLASLt?](01-what-is-hipblaslt.md).


## 1. Core concepts `[Essentials]`

The table below defines six terms that appear throughout the codebase and
this handbook.  Later sections reference them by name.

| Concept | Definition |
|---------|------------|
| Problem | Abstract GEMM configuration: data types, transpose modes, and features (bias, activation, scaling). Does *not* include dimensions or data pointers. |
| Contraction Problem | A Problem plus concrete runtime parameters: M, N, K dimensions, leading dimensions, strides, batch count, and data pointers. |
| Solution | A kernel implementation that can execute a Problem. Defined by tuning parameters: tile size, unroll depth, prefetch strategy, matrix instruction. One Problem type can have many Solutions. |
| Logic File | A YAML file mapping a Problem type to its Solutions. Contains the problem descriptor, solution definitions, and a size-to-solution mapping table. Organized by GPU architecture. |
| Code Object | A compiled `.co` (ELF) file containing one or more GPU kernels for a specific ISA target (e.g., `gfx950`). |
| Library | A tree of selection nodes built from logic files at build time. At runtime, the tree narrows from hardware to problem type to strategy to individual solution. |

### How the concepts relate

```
 Problem                      Contraction Problem                Kernel launch
 (types, transpose, features) (Problem + M,N,K + pointers)       (on GPU)
 ┌──────────┐    add dims     ┌───────────────────┐  runtime     ┌──────────┐
 │ Problem  │───& pointers───>│ContractionProblem  │──selection──>│  Kernel  │
 └──────────┘                 └───────────────────┘              └──────────┘
      │                                                               ^
      │ has many                                                      │
      v                                                               │
 ┌──────────┐                                                         │
 │ Solution │  ──────── compiled into ──────>  Code Object ───────────┘
 │ Solution │
 │ Solution │    Logic File groups a Problem
 │   ...    │    with its Solutions and maps
 └──────────┘    sizes to the best one.
```

A **Problem** is a static description -- it captures *what kind* of GEMM is
needed but says nothing about the actual matrix sizes or memory locations.
When the application calls `hipblasLtMatmul()`, the library combines that
Problem with the runtime arguments (dimensions, pointers, batch count) to form
a **Contraction Problem**.  The **Library** tree then walks its selection nodes
-- hardware, problem type, size heuristics -- to pick the best **Solution**.
That Solution was compiled ahead of time into a **Code Object**, which the HIP
runtime loads and launches on the GPU.


## 2. How a GEMM call becomes a kernel `[Essentials]`

Every GEMM call passes through four layers before a GPU kernel executes.

```
App            hipblaslt.cpp     rocblaslt_mat.cpp    tensile_host.cpp     GPU
               (Public API)      (rocblaslt backend)  (TensileLite dispatch)
 │                  │                  │                    │                │
 │ hipblasLtMatmul()│                  │                    │                │
 │─────────────────>│                  │                    │                │
 │                  │ rocblaslt_matmul()│                    │                │
 │                  │─────────────────>│                    │                │
 │                  │                  │ runContractionProblem()              │
 │                  │                  │───────────────────>│                │
 │                  │                  │                    │ launchKernels()│
 │                  │                  │                    │───────────────>│
```

**Public API (`hipblaslt.cpp`).** `hipblasLtMatmul()` casts every
`hipblasLt*` handle and descriptor to its `rocblaslt_*` counterpart and
calls `rocblaslt_matmul()`.  No computation happens here -- the function is
a thin translation layer between the public types and the internal types.

**rocblaslt backend (`rocblaslt_mat.cpp`).** `rocblaslt_matmul()` validates
arguments (null pointers, type mismatches, workspace size -- the workspace
is a temporary GPU buffer the application allocates for intermediate results) and delegates to
`rocblaslt_matmul_impl()`.  That function extracts dimensions, data types,
and epilogue settings (post-GEMM operations such as bias, activation,
and scaling) from the descriptors, packs them into a
`RocblasltContractionProblem`, and calls `runContractionProblem()`.

**TensileLite dispatch (`tensile_host.cpp`).** `runContractionProblem()`
obtains the `MasterSolutionLibrary` and a `SolutionAdapter` via
`get_library_and_adapter()`.  The `MasterSolutionLibrary` is the in-memory
library tree described in Section 1; the `SolutionAdapter` manages
code-object loading and kernel launch.  The function translates the
`RocblasltContractionProblem`
into a TensileLite `ContractionProblemGemm`, looks up the best solution
(via `getBestSolutions()` which calls `library->findTopSolutions()`), and
calls `solution->solve()` to produce a list of kernel invocations.

**Kernel launch.** The `SolutionAdapter` loads the code object if it has not
been loaded already (lazy loading is the default) and submits the kernel to
the HIP stream via `adapter->launchKernels()`.


## 3. The two dispatch paths `[Essentials]`

hipBLASLt has two backends for kernel dispatch.

| Aspect        | TensileLite              | RocRoller                |
|---------------|--------------------------|--------------------------|
| Kernels       | Precompiled `.co` files  | JIT-compiled at runtime  |
| Default usage | All standard GEMM        | Block-scaled GEMM        |
| Selection     | Logic file lookup        | Origami analytical cost model   |
| First-call    | Loads code object on use | JIT compiles kernel      |
| Caching       | Loaded once, reused      | Cached after first JIT   |

TensileLite is the default path for all GEMM operations.  RocRoller activates
automatically when the problem uses block scaling (per-tile scale factors
rather than one per tensor or per row, indicated by
`ScalingFormat::Block_32_UE8M0` or `Block_32_UE8M0_32_8_EXT`
on the A or B scale type), or when the application forces it on via a handle
flag (`handle->useRocRoller == 1`).  The decision is made by `useRocRoller()`
in `tensile_host.cpp`.  One exception: when both A and B are FP4 (4-bit floating-point) with a
pre-swizzled block-scale layout (data reordered in memory to match the
GPU's expected access pattern), TensileLite is used instead because it has
hand-optimized kernels for that format.  For RocRoller internals, see
[Chapter 11, Section 5](11-reference-appendix.md#5-rocroller-dispatch-details).


## 4. How solutions are selected `[Essentials]`

At build time, logic files are compiled into binary `.dat` bundles and
organized into a library tree.

```
 Logic files (.yaml)          .dat bundles           Library tree
 ┌──────────────────┐  build  ┌─────────────┐ load  ┌──────────────┐
 │ per-arch YAML    │────────>│ msgpack data │──────>│ selection    │
 │ solution mappings│         │ code objects │       │ nodes        │
 └──────────────────┘         └─────────────┘       └──────────────┘
```

Each `.dat` bundle is a [msgpack](https://msgpack.org/)-serialized
(a compact binary format similar to JSON) file containing solution metadata
and references to the compiled code objects.

At runtime, the library tree narrows the candidate set through a priority
cascade:

```
Library tree
└── Hardware layer (which GPU?)
    ├── gfx950_id75a3 (architecture + product SKU)   ← preferred
    └── gfx950 (generic architecture)               ← fallback
        └── Problem type (data types, transpose, features)
            ├── Equality       ← exact dimension match
            ├── GridBased      ← heuristic interpolation
            ├── Range          ← range-based lookup
            └── FreeSize       ← any size (last resort)
```

The selection API maps user-facing calls to internal library lookups:

| User-facing API       | Internal method        | Returns               |
|-----------------------|------------------------|-----------------------|
| `algoGetHeuristic()`  | `findTopSolutions()`   | Ranked top N solutions|
| `getAllAlgos()`        | `findAllSolutions()`   | Every compatible solution|
| `hipblasLtMatmul()` without pre-selected algo | `getBestSolutions()`   | Single best solution  |

The priority cascade means the library first tries to find a solution tuned
for the exact product SKU (identified by PCI device ID, e.g., `75a3` =
MI355X), then falls back to the
generic architecture.  Within each architecture node, Equality entries
(benchmark-derived decisions for specific M/N/K values) are checked first.
If no exact match exists, GridBased heuristics interpolate from nearby data
points.  Range entries match dimension ranges.  FreeSize solutions are the
last resort -- they work for any dimensions but are not tuned for any
specific size.

In practice, most users never need to tune manually.  `algoGetHeuristic()`
returns solutions ranked by expected performance, and the top result is
usually the best choice.  Manual selection via `getAllAlgos()` is useful for
exhaustive benchmarking or when the default heuristic misranks a solution for
a particular workload.


## 5. The three API surfaces `[Essentials]`

| Surface               | Header              | Style         | Recommended for        |
|-----------------------|---------------------|---------------|------------------------|
| C API                 | `hipblaslt.h`       | Opaque handles| cuBLASLt portability   |
| C++ Extension API     | `hipblaslt-ext.hpp` | Classes       | New ROCm-native code   |
| Extension Operations  | `hipblaslt-ext-op.h`| C functions   | Non-GEMM ops (softmax, layernorm, amax) |

Use the **C API** when porting code from cuBLASLt -- the function signatures
and workflow map one-to-one.  Use the **C++ Extension API** for new code that
targets ROCm exclusively -- it wraps the descriptor boilerplate into classes
and provides grouped-GEMM support.  Use **Extension Operations** for
standalone non-GEMM kernels that do not go through TensileLite.  See
[Chapter 5: API Guide](05-api-guide.md) for full API details and usage
examples.


## 6. Key directory map `[Essentials]`

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
