# Chapter 6 -- TensileLite Guide

## 1. What TensileLite Does `[Essentials]`

> For terminology used in this chapter (Problem, Solution, Logic File, etc.), see [Chapter 3: Core Concepts](03-architecture.md#1-core-concepts-essentials).

TensileLite is the kernel generation and selection engine inside hipBLASLt. It
produces optimized assembly GEMM kernels for AMD GPUs and packages them as
loadable code objects (`.co` files). At runtime the host library selects the
best kernel for a given problem and dispatches it.

**Pipeline at a glance:**

```
Problem description (YAML)
        |
        v
  Code generation (Python)          -- Tensile/ Python modules
        |
        v
  Assembly IR (in-memory)           -- rocisa register/instruction primitives
        |
        v
  Optimized assembly (.s)           -- stinkytofu (optional, ScheduleIterAlg=4)
        |
        v
  Assembled + linked code objects   -- amdclang assembler + linker
        |
        v
  Logic files (YAML)                -- map problem shapes to solutions
        |
        v
  Host library (C++)                -- tensilelite/src/, tensilelite/include/
        |
        v
  hipBLASLt runtime dispatch        -- loads .co, launches kernels
```

---

## 2. TensileLite-Specific Concepts `[Essentials]`

This section defines concepts specific to TensileLite that build on the
[core glossary in Chapter 3](03-architecture.md#1-core-concepts-essentials).

| Concept | What it is | Where it lives |
|---------|------------|----------------|
| Custom Kernel | A hand-written assembly kernel (`.s` file) referenced by name in a logic file's solution entry. | `CustomKernels/` |
| rocisa | Python/C++ ISA code generation module built with nanobind (a lightweight Python/C++ binding library). Provides register/instruction primitives for kernel writers. Includes the stinkytofu optimizer (see below). | `rocisa/` |
| stinkytofu | LLVM-inspired pass-based IR optimizer compiled into `_rocisa.so`. When activated (`ScheduleIterAlg=4` on supported architectures), converts rocisa IR to its own IR for DAG scheduling, wait-count insertion, and dead-code elimination. | `shared/stinkytofu/` |
| Selection Strategy | How a logic file's size-to-solution mapping works. Options: Equality (exact match), GridBased (heuristic), Range (range-based), FreeSize (any size), Prediction (see Origami below). | Logic file element 11 |
| Origami | A shared library (`shared/origami/`) that analytically predicts optimal GEMM tile configurations (tile size, matrix instruction, etc.) without benchmark data. Logic files under `Origami/` directories use the element 11 strategy value `Prediction`. Unlike Equality and GridBased which rely on offline benchmarking, Origami predicts performance from hardware parameters, making it useful for new architectures or untested problem shapes. | `shared/origami/` |
| Code Generation Pipeline | The offline Python pipeline that turns problem descriptions into rocisa IR, optionally optimizes it via stinkytofu, and assembles the result into compiled code objects. | `Tensile/` |

### Code generation lifecycle

```
┌────────────────────────┐
│ Problem description    │
│ (YAML)                 │
└───────────┬────────────┘
            │ KernelWriter builds rocisa IR
            v
┌────────────────────────┐
│ In-memory assembly IR  │
│ (rocisa Module)        │
└───────────┬────────────┘
            │ rocIsaPass + optional stinkytofu
            v
┌────────────────────────┐
│ Assembly source (.s)   │
└───────────┬────────────┘
            │ amdclang assembles + links
            v
┌────────────────────────┐
│ Code objects (.co)     │
└───────────┬────────────┘
            │ TensileCreateLibrary packages with logic
            v
┌────────────────────────┐
│ Logic files (YAML)     │
└───────────┬────────────┘
            │ serialized to .dat at build time
            v
┌────────────────────────┐
│ .dat bundles (msgpack) │
└────────────────────────┘
```

### rocisa

rocisa is a Python/C++ ISA code generator built with nanobind. It provides
Python bindings to ROCm ISA primitives -- registers, instructions, data
types -- so kernel writers can construct assembly programs from Python without
string manipulation. It also contains the stinkytofu C++ submodule, an
LLVM-inspired pass-based IR optimizer that post-processes the
rocisa-generated assembly at build time — reordering instructions for
better scheduling, inserting wait counts, and eliminating dead code.
Build or rebuild rocisa from the
tensilelite root with:

```bash
invoke rocisa
```

If the compiled extension is out of date, `import rocisa` raises an
`ImportError` with a hint listing the modified source files that triggered
the mismatch.

---

## 3. Directory Layout `[Essentials]`

```
tensilelite/
  tasks.py                  invoke tasks: rocisa, build-client, get-gpu-arch
  tox.ini                   tox test environments
  Makefile                  rebuild .co from modified assembly
  requirements.txt          Python dependencies
  CMakeLists.txt            CMake build for host lib + client

  rocisa/                   ISA code generation module (Python/C++ via nanobind)
    rocisa/                 Python package source
    test/                   rocisa unit tests
    docs/                   rocisa documentation
    CMakeLists.txt
    pyproject.toml

  Tensile/                  Python code-generation toolchain
    Tensile.py              top-level orchestrator (entry point logic)
    bin/                    CLI entry points
      Tensile               main entry point for running tests / benchmarks
      TensileCreateLibrary  build code objects + logic from solutions
      TensileLogic          logic file operations (validation, custom kernels)
      TensileMergeLibrary   merge multiple logic files
      TensileUpdateLibrary  update existing logic files
      TensileRetuneLibrary  re-benchmark and update perf data
      TensileBenchmarkCluster  cluster benchmarking utilities
      TensileClientConfig   generate client configuration
      TensileGenerateSummations  generate summation configs
      TensileLibLogicToYaml convert logic to YAML

    Common/                 shared constants, data types, utilities
      GlobalParameters.py   global config knobs
      Architectures.py      GPU architecture definitions
      DataType.py           data type handling
      ValidParameters.py    valid parameter ranges and defaults

    Components/             modular assembly-generation components
      MAC_F16.py, MAC_F32.py, ...   multiply-accumulate for each type
      LocalRead.py          LDS read logic
      GlobalWriteBatch.py   global memory write batching
      StreamK.py            stream-K partitioning
      PersistentLoop.py     persistent kernel loop
      Signature.py          kernel argument signature
      GSU.py                global split-U logic
      ...

    SolutionStructs/        solution data structures
      Solution.py           solution parameter management
      Problem.py            problem type definitions
      Naming.py             solution naming conventions
      Utilities.py          helper functions
      Validators/           parameter validation
        MatrixInstruction.py
        WorkGroup.py

    CustomKernels/          hand-written assembly kernels (.s files)
    Toolchain/              assembler / compiler abstraction
      Assembly.py           assembly toolchain driver
      Source.py             source compilation driver
      Validators.py         toolchain validation

    TensileCreateLibrary/   library-build orchestration
      Run.py                main build logic
      ParseArguments.py     CLI argument parsing

    TensileLogic/           logic file management
      Run.py                logic operations driver
      HandleCustomKernel.py custom kernel handling
      KnownBugs.py          known-bug workarounds
      ValidMatrixInstruction.py
      ValidWorkGroup.py
      ValidWorkGroupMappingXCC.py

    KernelWriterAssembly.py assembly kernel code generator (uses rocisa)
    KernelWriterModules.py  modular kernel building blocks
    KernelWriter.py         base kernel writer
    KernelWriterBase.py     abstract base class
    KernelWriterBetaOnly.py beta-scaling-only kernel
    KernelWriterReduction.py reduction kernel
    KernelWriterConversion.py type-conversion kernel
    KernelWriterActivationFunction.py activation kernel code gen
    KernelWriterActivationEnumHeader.py activation enum header gen

    ClientWriter.py         generates client test harness code
    ClientExecutable.py     client binary management
    LibraryLogic.py         logic file read/write
    LibraryIO.py            serialization / deserialization
    Contractions.py         contraction problem definitions
    SolutionLibrary.py      solution collection management
    SolutionSelectionLibrary.py  runtime solution selection
    Properties.py           problem properties
    BenchmarkStructs.py     benchmark configuration
    Configuration.py        configuration management

    Tests/                  test suite
      common/               end-to-end tests (require GPU + client)
        gemm/               GEMM test YAMLs
        exception/          exception handling tests
        streamk/            stream-K tests
        groupedgemm/        grouped GEMM tests
        gsu/                global split-U tests
        ...
      unit/                 Python unit tests (no GPU required)
      build_client.yaml     client build config for tests
      conftest.py           pytest fixtures

    Utilities/              misc helpers
      merge.py              logic file merging
      stats.py              statistics
      Decorators/           Python decorators
      tensile_generator/    generator utilities

  src/                      C++ host library source
    ContractionProblem.cpp  problem representation
    ContractionSolution.cpp solution management
    DataTypes.cpp           data type support
    Tensile.cpp             main library entry
    AMDGPU.cpp              GPU detection
    EmbeddedData.cpp        embedded data handling
    hip/                    HIP backend
    msgpack/                msgpack serialization
    ...

  include/                  C++ host library headers
    Tensile/
      ContractionProblem.hpp
      ContractionSolution.hpp
      AMDGPU.hpp
      CachingLibrary.hpp
      ...

  client/                   tensilelite-client C++ source
  tests/                    C++ host library tests
```

---

## 4. Logic File Anatomy `[Deep Dive]`

Logic files are YAML documents that map problem types to kernel solutions.
They are stored under
`library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/<arch>/`.

Files are organized by GPU architecture and matching strategy. For example:

```
gfx950/
  gfx950/
    Equality/       solutions selected by exact dimension match
    GridBased/      solutions selected by grid-based heuristics
    Origami/        analytical cost model (element 11 value = `Prediction`)
    Range/          solutions selected by range-based matching
  gfx950_id75a3/
    Equality/       solutions for a specific device variant
  gfx950_id75a8/
    Equality/       solutions for another device variant
```

Each logic file is a YAML list with a fixed element structure:

| Element | Contents | What it holds | Example |
|---------|----------|---------------|---------|
| 0 | Version header | Minimum logic file format version required to parse this file. | `{MinimumRequiredVersion: 5.0.0}` |
| 1 | Scheduling model | GPU architecture name, used to select the instruction scheduling model. | `gfx950` |
| 2 | Architecture | Target GPU architecture. Usually same as Element 1. | `gfx950` |
| 3 | Device ID filter | PCI device IDs this file applies to. Limits solutions to specific product SKUs. | `[Device 75a0]` |
| 4 | Problem type | Full GEMM variant specification: data types (`DataType`, `DestDataType`, `ComputeDataType`), transpose modes (`TransposeA/B`), index assignments (`IndexAssignmentsA/B`, `IndicesFree/Summation/Batch`), and feature flags (`UseBias`, `Activation`, `UseScaleAlphaVec`, etc.). | See walkthrough below |
| 5 | Solution list | Array of kernel definitions. Each solution specifies tuning parameters: tile size (`MacroTile0/1`), unroll depth (`DepthU`), matrix instruction (`MatrixInstruction`), work-group shape (`WorkGroup`), prefetch depths, GSU settings, etc. Each has a unique `SolutionIndex` (local to this file). | See walkthrough below |
| 6 | Index order | Dimension traversal order for the Element 7 lookup tree. Lists dimension indices (defined in Element 4) in the order they are checked. | `[2, 3, 0, 1]` = Batch, K, M, N |
| 7 | Size-to-solution mapping | Maps problem dimensions to solutions. Each entry is a pair: dimension bounds (8-element tuple of sizes/strides) and `[SolutionIndex, efficiency]`. Multiple entries can map to the same solution. | `[[4607, 1335, ...], [0, 19321.7]]` |
| 8-9 | Reserved | Unused, always null. | `null` |
| 10 | Performance metric | How solution quality is measured. | `DeviceEfficiency` |
| 11 | Selection strategy | How the size-to-solution mapping is queried at runtime. | `GridBased`, `Equality`, `Range`, `Prediction` |

### Filename convention

Logic file names encode the contraction pattern and features. For example,
`gfx950_Cijk_Ailk_Bjlk_HHS_BH_Bias_Aux_AH_SAV.yaml`:

- `Cijk` = C tensor indices
- `Ailk` / `Bjlk` = A and B index layouts (see TransposeA/B below)
- `HHS` = half/half/single (input/dest/compute data types)
- `BH` = high-precision accumulate
- `Bias` = bias enabled, `AH` = activation hipblaslt, `SAV` = scale-alpha-vec

### Element 4 — Problem type fields

Element 4 is a large mapping that fully specifies the GEMM variant.
Key fields (values from the example file above):

| Field | Example value | Meaning |
|-------|---------------|---------|
| `OperationType` | `GEMM` | Always GEMM for matrix multiply |
| `DataType` | `4` | Input data type (4 = half) |
| `DestDataType` | `4` | Output data type (4 = half) |
| `ComputeDataType` | `0` | Accumulation data type (0 = single; see `Common/DataType.py` for full mapping) |
| `HighPrecisionAccumulate` | `true` | Use higher precision in MAC |
| `TransposeA` | `0` | A is not transposed (0 = N, 1 = T) |
| `TransposeB` | `1` | B is transposed (0 = N, 1 = T) |
| `IndexAssignmentsA` | `[0, 3, 2]` | A's dimensions in index order: M(0), K(3), Batch(2) |
| `IndexAssignmentsB` | `[1, 3, 2]` | B's dimensions in index order: N(1), K(3), Batch(2) |
| `IndicesFree` | `[0, 1]` | Free (output) indices: index 0 = M, index 1 = N |
| `IndicesSummation` | `[3]` | Summation (contraction) index: index 3 = K |
| `IndicesBatch` | `[2]` | Batch index: index 2 = batch dimension |
| `UseBias` | `1` | Bias vector enabled |
| `UseScaleAlphaVec` | `1` | Per-element alpha scaling |
| `Activation` | `true` | Fused activation |
| `ActivationType` | `hipblaslt_all` | Activation function type |
| `Batched` | `true` | Batched GEMM |
| `StridedBatched` | `true` | Strided batched mode |

The index numbering scheme: `IndicesFree`, `IndicesSummation`, and
`IndicesBatch` assign a numeric index to each GEMM dimension.  In this
file, index 0 = M (rows of C), index 1 = N (columns of C), index 2 =
batch, and index 3 = K (contraction).  `IndexAssignmentsA/B` then list
which indices each input tensor uses, defining its memory layout.

`TransposeA/B` and `IndexAssignmentsA/B` encode the same information.
When `TransposeA = 0` (not transposed), M comes first in A's layout:
`IndexAssignmentsA = [0, 3, 2]` (M, K, Batch).  When `TransposeA = 1`
(transposed), K comes first: `IndexAssignmentsA = [3, 0, 2]` (K, M,
Batch).  The filename encodes this too: `Ailk` = A indices are i(M),
l(K), k(Batch); `Alik` = l(K), i(M), k(Batch).

### Element 5 — Solution tuning parameters

Each solution in the Element 5 array defines a kernel's tuning
parameters.  Key fields:

| Field | Example value | Purpose |
|-------|---------------|---------|
| `SolutionIndex` | `0` | Unique ID within this file |
| `SolutionNameMin` | `Cijk_Ailk_Bjlk_HHS_BH_Bias_Aux_AH_SAV_MT128x256x32_MI32x32x8x1_SN_...` | Human-readable name |
| `KernelLanguage` | `Assembly` | Assembly (not source) |
| `ISA` | `[9, 5, 0]` | Target ISA = gfx950 |
| `MacroTile0` / `MacroTile1` | `128` / `256` | Work-group tile dimensions |
| `DepthU` | `32` | Unroll depth along K |
| `MatrixInstruction` | `[32, 32, 8, 1]` | MFMA instruction: M, N, K, blocks |
| `WorkGroup` | `[64, 4, 1]` | Thread group dimensions |
| `NumThreads` | `256` | Threads per work-group |
| `PrefetchGlobalRead` | `2` | Prefetch pipeline depth |
| `PrefetchLocalRead` | `1` | LDS prefetch depth |
| `GlobalSplitU` | `1` | K-dimension parallelism |
| `GlobalSplitUAlgorithm` | `MultipleBuffer` | GSU strategy |
| `BufferLoad` / `BufferStore` | `true` / `true` | Use buffer instructions |
| `StaggerU` | `32` | Stagger unroll to reduce bank conflicts |
| `WavefrontSize` | `64` | Wavefront width |
| `VectorWidth` | `2` | Output vector width |
| `CustomKernelName` | `''` | Empty = auto-generated kernel |
| `LdsNumElements` | `12288` | LDS usage in elements |
| `WorkGroupMapping` | `8` | Work-group mapping strategy |
| `SourceSwap` | `true` | Swap A/B sources |
| `DirectToLds` | `0` | Direct global-to-LDS transfer |
| `DirectToVgprA` / `DirectToVgprB` | `false` | Direct-to-VGPR bypassing LDS |

### Element 7 — Size-to-solution mapping

Each entry in the Element 7 array maps a range of problem dimensions to
a solution.  Example from the file above:

```yaml
- - - [4607, 1335, 1, 320, 4607, 4607, 4607, 1335]
    - [1, 19321.7]
```

The first sub-list is an 8-element tuple of dimension and stride bounds
(traversed in the order specified by Element 6).  The second sub-list is
`[SolutionIndex, efficiency]` — the index into Element 5 and the
measured performance used for heuristic ranking.  Multiple entries can
map to the same SolutionIndex when one kernel is optimal across several
size ranges.

---

## 5. Code Generation Pipeline `[Deep Dive]`

TensileLite transforms a problem description into a GPU kernel through
several stages. The main Python modules involved:

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

### Stage 1: Problem parsing

`Tensile/Contractions.py` and `SolutionStructs/Problem.py` parse the YAML
problem description and build an internal `ContractionProblem` object. Index
assignments, data types, and feature flags are validated here.

### Stage 2: Solution parameter resolution

`SolutionStructs/Solution.py` resolves derived parameters from the base
solution definition. Validators in `SolutionStructs/Validators/` check that
parameters like `MatrixInstruction` and `WorkGroup` are valid for the target
architecture. `Common/ValidParameters.py` defines the allowed ranges.

### Stage 3: Kernel code generation

`KernelWriterAssembly.py` is the unified interface for all kernel source.
Its `getSourceFileString()` method either generates assembly via rocisa or
reads a hand-written `.s` file from `CustomKernels/` -- the caller
(`Run.py`) sees no difference. For custom kernels (when `CustomKernelName`
is non-empty), the entire code generation engine is skipped; for
auto-generated kernels, it uses `rocisa` to build an in-memory IR tree of
GPU ISA instructions (CDNA and RDNA). Currently 119 custom kernels exist
(gfx942 and gfx950 only), covering ~0.06% of shipped solutions --
mainly FP8 GroupedGemm and GSU variants where hand-tuned assembly
outperforms the generator.

The auto-generated path is modular:

- **Components/** contains reusable assembly-generation modules. Each
  component handles one aspect of the kernel:
  - `MAC_F16.py`, `MAC_F32.py`, etc. -- multiply-accumulate for each data type
  - `LocalRead.py` -- reading tiles from LDS
  - `GlobalWriteBatch.py` -- writing results to global memory, including
    fused epilogue (bias, activation, scaling) applied in-register
    before the store. In some multi-kernel configurations the epilogue
    is instead handled by a separate HIP conversion kernel
    (`KernelWriterConversion.py`).
  - `GSU.py` -- global split-U reduction
  - `StreamK.py` -- stream-K work partitioning
  - `PersistentLoop.py` -- persistent kernel loops
  - `Signature.py` -- kernel argument layout

- `KernelWriterModules.py` provides helper functions (e.g., `wait()`,
  `tdmWait()`) used during IR construction.

- `KernelWriter.kernelBody()` finalizes the IR tree by running two
  post-processing stages:
  1. **rocIsaPass** -- a lightweight rocisa-native pass (duplicate removal,
     delay-ALU insertion, cycle counting).
  2. **stinkytofu** (optional) -- when `ScheduleIterAlg=4` and the target
     architecture is supported, converts the rocisa IR to stinkytofu's own
     IR, runs DAG scheduling, wait-count optimization, and dead-code
     elimination, then emits the final assembly text.

- `Toolchain/Component.py` (`Assembler`) invokes amdclang to convert `.s`
  files into `.o` object files. `Toolchain/Assembly.py` links `.o` files
  into `.co` code objects and compresses them. Both steps are orchestrated
  by `TensileCreateLibrary/Run.py`.

#### Instruction scheduling (`ScheduleIterAlg`)

Instruction scheduling -- interleaving operations between MFMAs to hide
memory latency -- happens *during* IR construction, not as a separate
pass (except for stinkytofu). The `ScheduleIterAlg` solution parameter
(SIA) selects the strategy. The default is SIA=3.

| SIA | Strategy | What it does |
|-----|----------|--------------|
| 0 | Sequential | No scheduling. Global reads, local reads, local writes, MACs emitted in fixed order. |
| 1 | Half-read | Simple interleaving: emits half the local reads, then global reads, then the rest. |
| 2 | Two-workgroup | Priority-based. While WG0 computes, WG1 fetches and vice-versa. Requires `CUOccupancy >= 2`. |
| 3 | Full MFMA-interleaved | Interleaves global reads, local writes, local reads, and pack instructions between individual MFMAs. Controlled by `GlobalReadPerMfma` and `LocalWritePerMfma`. This is the default. |
| 4 | stinkytofu | Remapped to SIA=0 (no scheduling during generation), then the entire IR is handed to stinkytofu's DAG scheduler for optimization. |

SIA scheduling is **MFMA-centric**: it decides what goes between each
pair of MFMAs. The scheduled instruction types are:

- **Memory operations** -- global reads, local reads, local writes
  (the primary targets for latency hiding).
- **Pack VALUs** -- type-conversion instructions (e.g.,
  `v_cvt_pk_f32_bf16`) that feed MFMA operands. These are the only
  VALU instructions that receive per-instruction placement between
  MFMAs.
- **Pointer updates, sync, waitcnt** -- placed at fixed MFMA indices
  as opaque blocks.

General VALU instructions (address arithmetic, other ALU work) are
**not** individually scheduled -- they are embedded inside larger
instruction modules and emitted as-is. Only stinkytofu (SIA=4)
performs full DAG-based scheduling across all instruction types.

SIA 0--3 are implemented as `Components/SIA.py` classes (SIA0, SIA1,
SIA2, SIA3) and called via `makeSchedule()` during IR construction in
`KernelWriter.py`. SIA=4 is remapped at solution setup time
(`SolutionStructs/Solution.py`): the kernel is generated with SIA=0,
and the `_StinkyTofuOptLevel` flag triggers stinkytofu after
`rocIsaPass`.

**In practice:** SIA is tuned per-solution and stored in the shipped
logic YAML files. SIA=3 is used by ~99% of shipped solutions across
all architectures. SIA=1 appears as a minority for specific cases
(e.g., DGEMM, RDNA). SIA=4 (stinkytofu) is not yet shipped in any
production logic file -- it currently only has a backend for gfx1250
and appears only in test/benchmark configs.

Note: `rocIsaPass` does **no** instruction scheduling -- only
delay-ALU insertion, duplicate removal, and cycle estimation.

A separate **subtile path** (`UseSubtileImpl=True`) bypasses the SIA
system entirely and uses its own `LogicalScheduler` +
`InstructionScheduler` (`Components/Subtile/`) for constraint-based
slot placement between MFMAs. It schedules ds_read, buffer_load,
waitcnt, and M0 updates, but not pack or general VALU instructions.

### Stage 4: Library creation

`TensileCreateLibrary/Run.py` orchestrates the full build:

1. Read logic files
2. Generate assembly source for each solution
3. Assemble and link into code objects
4. Write the solution metadata for the host library

Entry point: `Tensile/bin/TensileCreateLibrary`

### Stage 5: Logic file management

`TensileLogic/` handles logic file operations:

- `Run.py` -- main driver for logic operations
- `HandleCustomKernel.py` -- resolves custom kernel references
- `ValidMatrixInstruction.py` -- validates MI parameters in logic files
- `ValidWorkGroup.py`, `ValidWorkGroupMappingXCC.py` -- work-group validation

`LibraryLogic.py` and `LibraryIO.py` handle reading and writing logic files.

### Key entry points in `Tensile/bin/`

| Script | Purpose |
|--------|---------|
| `Tensile` | Run benchmarks / test YAMLs |
| `TensileCreateLibrary` | Build code objects + logic from solutions |
| `TensileLogic` | Logic file validation and custom kernel checks |
| `TensileMergeLibrary` | Merge multiple logic files |
| `TensileUpdateLibrary` | Update existing logic files |
| `TensileRetuneLibrary` | Re-benchmark and update performance data |
| `TensileBenchmarkCluster` | Cluster benchmarking utilities |
| `TensileClientConfig` | Generate client configuration |
| `TensileGenerateSummations` | Generate summation configs |
| `TensileLibLogicToYaml` | Convert logic to YAML format |

---

## 6. Building and Testing TensileLite `[Deep Dive]`

### Prerequisites

All commands run from the `tensilelite/` directory with the project venv
activated (see [Chapter 2](02-environment-setup.md#building-hipblaslt-essentials)
for venv setup).

```bash
source <venv>/bin/activate
cd <repo>/projects/hipblaslt/tensilelite
```

### Building rocisa

rocisa must be built once after cloning (and again after editing rocisa or
stinkytofu C++ source):

```bash
invoke rocisa
```

This runs `pip install -e rocisa/` with the correct CMake flags for your
ROCm installation. After installation, `import rocisa` works from anywhere
in the venv.

If rocisa bindings are stale, importing it raises an `ImportError` with a
clear rebuild hint listing the modified source files.

### Building the client

The tensilelite-client is a C++ executable used to run benchmark and test
YAMLs on the GPU.

```bash
# Default build (auto-detects GPU)
invoke build-client

# Specify architecture and ROCm path
invoke build-client --gpu-targets gfx950 --rocm-path /opt/rocm-7.3.0

# Generate compile_commands.json for IDE integration
invoke build-client --export-compile-commands

# Clean rebuild
invoke build-client --clean
```

The client binary is placed in `build_tmp/tensilelite/client/tensilelite-client`
by default. The `--build-dir` flag overrides this location.

Alternatively, use CMake directly:

```bash
cd ..   # run from the project root (projects/hipblaslt)
cmake --preset tensilelite -B my-build
cmake --build my-build --parallel
```

### Tox environments

The `tox.ini` defines these environments:

| Environment | Purpose |
|-------------|---------|
| `py3` | Full test suite: build client + run rocisa tests + common tests |
| `unit` | Python unit tests only (no client build, no GPU needed) |
| `rocisa` | rocisa tests only |
| `lint` | flake8 linting |
| `format` | Auto-format code with black |
| `isort` | Sort imports with isort |
| `pre_commit` | Pre-commit checks (lint + unit tests) |
| `coverage` | Full coverage report (unit + common tests) |
| `coverage-unit` | Coverage for unit tests only |
| `coverage-common` | Coverage for common tests only (requires prebuilt client) |

Common usage:

```bash
# Full test suite
tox -e py3 -- Tensile/Tests -m common

# Unit tests only (fast, no GPU)
tox -e unit -- Tensile/Tests/unit

# Coverage report
tox -e coverage

# Unit-only coverage
tox -e coverage-unit
```

### Running individual tests

After building the client, run test YAMLs directly:

```bash
Tensile/bin/Tensile Tensile/Tests/common/gemm/<test>.yaml tensile-out
```

With a custom-built client:

```bash
Tensile/bin/Tensile Tensile/Tests/common/exception/<test>.yaml tensile-out \
    --prebuilt-client=my-build/tensilelite/client/tensilelite-client
```

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TENSILE_NUM_PYTEST_WORKERS` | `4` | Parallel pytest workers in tox |
| `TENSILELITE_CLIENT_ARGS` | (empty) | Extra args passed to `invoke build-client` during tox |

### Rebuilding assembly to code objects

During tuning, you can modify `.s` files and reassemble just the code
object without re-running the full Tensile pipeline. The hand-written
`Makefile` in `tensilelite/` is a thin wrapper around `amdclang++` for
this purpose (it is not CMake-generated):

```bash
# After editing an .s file in the tensile-out directory:
make co TENSILE_OUT=tensile-out

# Specify architecture (with xnack suffix) and wavefront size:
make co TENSILE_OUT=tensile-out ARCH="gfx942:xnack-" WAVE=64

# For RDNA (wavefront size 32):
make co TENSILE_OUT=tensile-out ARCH="gfx1100" WAVE=32
```

The `ARCH` value is auto-detected from `.co` filenames when omitted.
The `Makefile` also accepts `ASM_ARGS` and `LINK_ARGS` for additional
assembler and linker flags.

### Rebuild cheat sheet

| What you changed | Command |
|------------------|---------|
| rocisa C++ sources | `invoke rocisa` |
| stinkytofu C++ sources | `invoke rocisa` |
| tensilelite-client C++ sources | `invoke build-client` |
| rocisa `pyproject.toml` or `CMakeLists.txt` | `invoke rocisa` |
| TensileLite Python code | No rebuild needed |
| Assembly `.s` files (tuning) | `make co TENSILE_OUT=<dir>` |

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `invoke: command not found` | venv not activated | Activate your project venv (see [Chapter 2](02-environment-setup.md#building-hipblaslt-essentials)) |
| `ImportError: cannot import name 'rocIsa' from 'rocisa'` | rocisa not built or stale `_rocisa.so` | `invoke rocisa` |
| `tox -e rocisa` fails with `Cannot import 'scikit_build_core.build'` | tox venv missing build deps | Run `invoke rocisa` first to build in the project venv, then use `tox -e rocisa` |

---

## 7. How to Add a New Kernel Solution `[Deep Dive]`

This section walks through adding a new auto-generated kernel solution to an
existing logic file.

### Step 1: Identify the target logic file

Logic files are under
`library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/<arch>/`. Find
the file matching your problem type. The filename encodes the contraction
pattern and features:

- `Cijk` -- tensor index pattern
- `Alik` / `Bjlk` / `Ailk` / `Bljk` -- A and B layouts (transpose modes)
- `HSS`, `HHS`, `BBS`, `F8BS`, ... -- type abbreviations (input/compute/output)
- `BH` -- high-precision accumulate (HighPrecisionAccumulate)
- `Bias`, `SAV`, `SAB` -- optional features
- `UserArgs` -- user-arguments variant

### Step 2: Define the solution parameters

Add a new solution mapping to Element 5 (the solution list) of the YAML
file. At minimum, specify these parameters (using values appropriate for
your target architecture):

```yaml
- SolutionIndex: <next available index>
  KernelLanguage: Assembly
  ISA: [9, 5, 0]                    # gfx950
  MacroTile0: 128
  MacroTile1: 128
  DepthU: 64
  MatrixInstruction: [16, 16, 16, 1]
  WorkGroup: [64, 4, 1]
  NumThreads: 256
  PrefetchGlobalRead: 2
  PrefetchLocalRead: 1
  GlobalSplitU: 1
  GlobalSplitUAlgorithm: MultipleBuffer
  WavefrontSize: 64
  # ... additional parameters as needed
```

Most derived parameters (fields prefixed with `_`, and computed values like
`LdsNumElements`, `NumLoadsA`, `ThreadTile0`, etc.) are filled in
automatically when `AssignedDerivedParameters: true` is set. However, for
logic files that ship with precomputed values, all fields must be present.

### Step 3: Update the size-to-solution mapping

In Element 7, add entries that map problem-size ranges to your new
`SolutionIndex`. For `Equality`-type files, add exact size tuples. For
`GridBased` files, the mapping uses size ranges with efficiency values.

### Step 4: Validate the solution

Use `TensileLogic` to validate:

```bash
Tensile/bin/TensileLogic --check-only-custom-kernels <path-to-logic-file>
```

For non-custom kernels, build the library to verify the solution compiles:

```bash
Tensile/bin/TensileCreateLibrary <logic-dir> <output-dir> HIP
```

### Step 5: Test the kernel

Write or reuse a benchmark YAML and run it through the client:

```bash
Tensile/bin/Tensile Tensile/Tests/common/gemm/<test>.yaml tensile-out
```

Or run a targeted hipblaslt-bench invocation to exercise the new kernel via
the full hipBLASLt stack (after rebuilding the library):

```bash
./build/clients/hipblaslt-bench -m <M> -n <N> -k <K> --precision <type> -v
```

### Adding a custom (hand-written) kernel

For hand-tuned assembly kernels:

1. Place the `.s` file in `Tensile/CustomKernels/`. Follow the existing
   naming convention:
   `Custom_<contraction>_<types>_<features>_<tile>_<shortname>_<arch>.s`

2. In the logic file solution entry, set `CustomKernelName` to the
   filename stem (without `.s`).

3. Optionally add a `custom.config` section in the assembly file to
   override metadata fields from the logic file.

4. Validate with:
   ```bash
   Tensile/bin/TensileLogic --check-only-custom-kernels <path-to-logic-file>
   ```
