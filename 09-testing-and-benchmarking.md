# Chapter 9: Testing and Benchmarking

This chapter covers every test and benchmark tool in hipBLASLt: how the test
data pipeline works, how to run and filter tests, the full `hipblaslt-bench`
CLI, the `rtest.py` convenience script, and the separate TensileLite test
infrastructure.


## 1. Test infrastructure overview `[Essentials]`

hipBLASLt test cases are defined in YAML, converted to a binary data file at
build time, and consumed by a gtest binary at runtime.

```
 hipblaslt_gtest.yaml               Master include file
       |
       |--- include: smoke_gtest.yaml
       |       |--- include: hipblaslt_common.yaml   (precision anchors, defaults)
       |       |--- include: matmul_common.yaml      (matrix size ranges)
       |       '--- include: known_bugs.yaml
       |
       |--- include: matmul_gtest.yaml
       |       |--- include: hipblaslt_common.yaml
       |       |--- include: matmul_common.yaml
       |       '--- include: known_bugs.yaml
       |
       |--- include: auxiliary_gtest.yaml
       |       |--- include: hipblaslt_common.yaml
       |       |--- include: matmul_common.yaml
       |       '--- include: known_bugs.yaml
       |
       '--- include: rocroller_gtest.yaml
               |--- include: hipblaslt_common.yaml
               '--- include: matmul_common.yaml
       |
       v
 hipblaslt_gentest.py               Python generator (needs PyYAML)
       |
       v
 build/clients/hipblaslt_gtest.data Binary test argument records
       |
       v
 build/clients/hipblaslt-test       gtest binary (reads .data at startup)
```

> **Note:** `hipblaslt_template.yaml` also exists in the data directory.
> It is used for processing YAML from log files, not for regular test runs.

The pipeline:

1. **YAML definitions** in `clients/tests/data/` declare test parameters:
   matrix sizes, precisions, transpose modes, alpha/beta values, activation
   types, bias vectors, and which GPU architectures each test applies to.

2. **`hipblaslt_gtest.yaml`** is the root file.  It `include:`s four test
   files: `smoke_gtest.yaml`, `matmul_gtest.yaml`, `auxiliary_gtest.yaml`,
   and `rocroller_gtest.yaml`.  Those sub-files in turn include shared
   anchors from `hipblaslt_common.yaml` and `matmul_common.yaml` (precision
   lists, size ranges, default parameters) and platform-specific expected
   failures from `known_bugs.yaml`.

3. **`hipblaslt_gentest.py`** reads the YAML, expands every combinatorial
   parameter (precisions x sizes x transposes x ...), and writes compact
   binary `Arguments` records to `hipblaslt_gtest.data`.

4. **`hipblaslt-test`** loads the `.data` file at startup.  Each record
   becomes a parameterized gtest instance whose name encodes the category
   and test name.

The CMake target `hipblaslt-test-data` runs the generator.  Building
`hipblaslt-test` automatically depends on it, so the data file is always
regenerated when YAML changes.

> **Note:** `hipblaslt-test` looks for `hipblaslt_gtest.data` in the same
> directory as the binary (resolved via `/proc/self/exe`).  The build system
> places both in `build/clients/`, so no special working-directory setup is
> needed — you can run the binary from anywhere.


## 2. hipblaslt-test (gtest) `[Essentials]`

### Building

```bash
cmake --build build --target hipblaslt-test
```

This builds both the test binary and generates `hipblaslt_gtest.data`.

### Running

```bash
cd build/clients
./hipblaslt-test                           # full suite
./hipblaslt-test --gtest_filter=<pattern>  # filtered
```

### Filter syntax

The gtest filter uses `*` wildcards and `:` to separate multiple patterns.
A leading `-` excludes patterns.

| Goal | Filter |
|------|--------|
| All smoke tests | `--gtest_filter=*smoke*` |
| All quick tests | `--gtest_filter=*quick*` |
| DRelu gradient matmul tests | `--gtest_filter=*drelu*` |
| Pre-checkin tests | `--gtest_filter=*pre_checkin*` |
| Nightly tests | `--gtest_filter=*nightly*` |
| Specific test by name | `--gtest_filter=*matmul_bias_gelu_smoke*` |
| Multiple patterns | `--gtest_filter=*smoke*:*quick*` |
| Exclude a pattern | `--gtest_filter=*quick*:-*grouped*` |

### Test categories

Each test entry in the YAML has a `category` field.  The categories control
which test tier a case belongs to:

| Category | Purpose | Typical run time |
|----------|---------|------------------|
| `smoke` | Minimal sanity (small sizes, all precisions) | seconds |
| `quick` | Broader coverage, moderate sizes | minutes |
| `pre_checkin` | Extended coverage before merge | tens of minutes |
| `nightly` | Large sizes, exhaustive combinations | hours |

### GPU architecture filtering

Tests can declare `gpu_arch` as a regex pattern (e.g., `'942'`,
`'(120[0-1]|1250)'`).  At runtime, the test harness matches the pattern
against the current GPU's architecture string.  Tests whose `gpu_arch`
does not match are skipped.

### Known bugs

`clients/tests/data/known_bugs.yaml` lists test cases that are expected to
fail on specific platforms.  Each entry specifies parameter values that
identify the failing case and a `known_bug_platforms` field listing affected
GPU architectures (e.g., `"gfx908"`).  Matching tests are reclassified as
`known_bug` and skipped.

### Reading test output

Standard gtest output.  Each line shows `[ RUN ]`, `[  OK  ]`, or
`[ FAIL ]` with the full parameterized test name.  At the end, a summary
reports passed, failed, and skipped counts.


## 3. hipblaslt-bench `[Essentials]`

`hipblaslt-bench` is the standalone benchmarking tool for measuring GEMM
performance.

### Building

```bash
cmake --build build --target hipblaslt-bench
```

### Basic usage

```bash
cd build/clients
./hipblaslt-bench --help

# FP16 GEMM (default)
./hipblaslt-bench -m 4096 -n 4096 -k 4096

# FP32 GEMM with CPU validation
./hipblaslt-bench --precision f32_r -v

# BF16 GEMM, transposed A
./hipblaslt-bench --transA T -m 4096 -n 4096 -k 4096 \
    --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r \
    --compute_type f32_r
```

### Matrix memory layout

All matrices are **column-major**: columns are contiguous in memory.

Example: a 4×3 matrix (M=4, N=3) with `ld = 4`:

```
Logical view:              Memory layout (column-major, ld=4):

  col0  col1  col2         addr:  0  1  2  3 | 4  5  6  7 | 8  9 10 11
┌─────┬─────┬─────┐              ├──col 0──┤ ├──col 1──┤ ├──col 2──┤
│ a00 │ a01 │ a02 │  row 0       a00 a10 a20 a30 a01 a11 a21 a31 a02 a12 a22 a32
│ a10 │ a11 │ a12 │  row 1
│ a20 │ a21 │ a22 │  row 2       Element (row, col) is at offset: row + col * ld
│ a30 │ a31 │ a32 │  row 3
└─────┴─────┴─────┘
```

The **leading dimension** (ld) is the distance in elements between the
start of two consecutive columns. It must be ≥ the number of rows. When
`ld > rows`, the extra elements are padding (useful for alignment).

**Transpose and leading dimensions.** The GEMM computes
`D = α·op(A)·op(B) + β·C` where `op(X) = X` (N) or `op(X) = Xᵀ` (T).
The `transA`/`transB` flags do **not** change how data is stored — they
tell the kernel how to *read* the data you already have:

```
transA=N: your A is already (M×K).     transA=T: your A is (K×M) and
The kernel reads it as-is.             the kernel transposes it on the fly
                                       to get the (M×K) operand it needs.

  A in memory (M×K)                      A in memory (K×M)
  ┌─── K cols ───┐                       ┌─── M cols ───┐
  │              │ M rows                │              │ K rows
  │    a[r,c]    │                       │    a[r,c]    │
  └──────────────┘                       └──────────────┘
  lda = M (rows in stored matrix)        lda = K (rows in stored matrix)
```

The leading dimension is always the number of rows in the **stored**
matrix. C, D, and E are always (M×N) regardless of transpose settings,
so `ldc = ldd = lde = M`.

**Batching.** With `batch_count > 1`, each batch element is a separate
matrix. The **batch stride** is the offset in elements from one batch
element to the next:

```
batch 0             batch 1             batch 2
├── ld × cols ──┤   ├── ld × cols ──┤
[  matrix data  ]   [  matrix data  ]   [  matrix data  ]
^                   ^                   ^
base              base + stride       base + 2*stride
```

Default strides: `stride = ld × number_of_columns` (= total elements per
matrix). For example, with transA=N: `stride_a = lda × K`.

In both batching modes, all batch elements must have the **same shape**
(M, N, K) and the same leading dimensions. Only the data differs.

**Strided Batched** (batch_mode=0, default): all batch elements live in
one contiguous allocation, spaced by the stride.
**General Batched** (batch_mode=1): each batch element can be a separate
allocation; the API receives a device array of pointers.

If you need **different sizes per problem**, use Grouped GEMM
(`hipblaslt_ext::GroupedGemm`) instead — see [Chapter 5](05-api-guide.md#grouped-gemm-essentials).

**Matrix E** is the auxiliary (AUX) matrix used with certain epilogues.
In the forward pass (`--use_e` without `--gradient`), E stores
pre-activation values (e.g., pre-GELU output). In the backward pass
(`--use_e --gradient`), E is an input providing saved activations for
gradient computation. Same shape as D: (M rows × N cols).

### Full CLI reference

**Matrix dimensions:**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--sizem` | `-m` | 128 | Number of rows in C/D (and rows of op(A)) |
| `--sizen` | `-n` | 128 | Number of columns in C/D (and columns of op(B)) |
| `--sizek` | `-k` | 128 | Number of columns of op(A) and rows of op(B) |
| `--lda` | | auto | Leading dimension of A (default: M if transA=N, K if transA=T) |
| `--ldb` | | auto | Leading dimension of B (default: K if transB=N, N if transB=T) |
| `--ldc` | | auto | Leading dimension of C (default: M) |
| `--ldd` | | auto | Leading dimension of D (default: M) |
| `--lde` | | auto | Leading dimension of E / AUX matrix (default: M) |
| `--any_stride` | | off | Pass user-specified strides through without clamping to the minimum (ld × columns). Without this flag, strides smaller than the minimum are silently raised. |
| `--stride_a` | | auto | Batch stride for A in elements (default: lda × K if transA=N, lda × M if transA=T) |
| `--stride_b` | | auto | Batch stride for B in elements (default: ldb × N if transB=N, ldb × K if transB=T) |
| `--stride_c` | | auto | Batch stride for C in elements (default: ldc × N) |
| `--stride_d` | | auto | Batch stride for D in elements (default: ldd × N) |
| `--stride_e` | | auto | Batch stride for E in elements (default: lde × N) |

**Scalar parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--alpha` | 1 | Scalar multiplier for op(A)*op(B) in D = alpha*op(A)*op(B) + beta*C |
| `--beta` | 0 | Scalar multiplier for C. 0 = C is not read. |

**Data types:**

| Flag | Short | Default | Description | Options |
|------|-------|---------|-------------|---------|
| `--precision` | `-r` | f16_r | Set A/B/C/D types at once. Override per-matrix with `--a_type` etc. `_r` = real. | f32_r, f16_r, bf16_r, f64_r, i32_r, i8_r |
| `--a_type` | | (precision) | Data type of matrix A (overrides `--precision`) | f32_r, f16_r, bf16_r, f8_r, bf8_r, f8_fnuz_r, bf8_fnuz_r, i8_r |
| `--b_type` | | (precision) | Data type of matrix B (overrides `--precision`) | f32_r, f16_r, bf16_r, f8_r, bf8_r, f8_fnuz_r, bf8_fnuz_r, i8_r |
| `--c_type` | | (precision) | Data type of matrix C (overrides `--precision`) | f32_r, f16_r, bf16_r, i8_r |
| `--d_type` | | (precision) | Data type of matrix D (overrides `--precision`) | f32_r, f16_r, bf16_r, i8_r |
| `--compute_type` | | f32_r | Accumulator precision for the Matrix Fused Multiply-Add (MFMA) hardware instructions. `s` = f32_r; `x` = xf32_r (inputs truncated to ~10 mantissa bits for higher throughput); `f32_bf16_r` = FP32 accumulation with BF16 input down-conversion. | s, f32_r, x, xf32_r, f64_r, i32_r, f32_bf16_r |
| `--compute_input_typeA` | | INVALID | Cast A to this type before MFMA. INVALID = use storage type. For mixed-precision FP8 workflows. | f32_r, f16_r, bf16_r, f8_r, bf8_r, f8_fnuz_r, bf8_fnuz_r |
| `--compute_input_typeB` | | INVALID | Cast B to this type before MFMA. INVALID = use storage type. | f32_r, f16_r, bf16_r, f8_r, bf8_r, f8_fnuz_r, bf8_fnuz_r |
| `--scale_type` | | | Data type of scalar scale factors (scaleA, scaleB, etc.) | f16_r, bf16_r |

**Transpose and batching:**

| Flag | Default | Description |
|------|---------|-------------|
| `--transA` | N | N = no transpose, T = transpose |
| `--transB` | N | N = no transpose, T = transpose |
| `--batch_count` | 1 | Number of matrices (batched/strided_batched) |
| `--batch_mode` | 0 | 0 = Strided Batched, 1 = General Batched |

**Fusion and epilogue:**

| Flag | Default | Description |
|------|---------|-------------|
| `--activation_type` | none | Fused activation function: none, gelu, relu, swish, clamp |
| `--activation_arg1` | 0 | Activation parameter 1. For clamp: lower bound. |
| `--activation_arg2` | inf | Activation parameter 2. For clamp: upper bound. |
| `--bias_vector` | off | Add a 1-D bias vector to each column of D. See `--bias_type` and `--bias_source`. |
| `--bias_type` | (d_type) | Data type of the bias vector. `default` = match `--d_type`. Requires `--bias_vector`. Options: f16_r, bf16_r, f32_r, default |
| `--bias_source` | d | Bias vector dimension. d = M rows (forward); a = M (gradient w.r.t. A); b = N (gradient w.r.t. B). Requires `--bias_vector`. |
| `--scaleA` | 0 | Scale mode for A. 0=None, 1=scalar (one per tensor), 2=vector (one per row). MX (Microscaling) block-scale formats share a single scale exponent per block: 3=B32E8 (32-element block, E8M0 = 8-bit exponent-only scale), 4=B16E8, 5=B32E4M3, 6=B16E4M3, 7=B32E5M3, 8=B16E5M3. 1001=scale data pre-arranged into hardware 32×8 tile layout. Requires FP8/FP6/FP4 input types. |
| `--scaleB` | 0 | Scale mode for B (same values as `--scaleA`) |
| `--scaleC` | 0 | Scale mode for C (0=None, 1=scalar) |
| `--scaleD` | 0 | Scale mode for D (0=None, 1=scalar) |
| `--scaleAlpha_vector` | off | Per-column alpha vector (length N) instead of scalar `--alpha`. Forces alpha to 1.0. |
| `--amaxScaleA` | off | Compute amax(A) and use it to scale A (for FP8 dynamic quantization) |
| `--amaxScaleB` | off | Compute amax(B) and use it to scale B (for FP8 dynamic quantization) |
| `--amaxD` | off | Compute and output the absolute maximum of D (for FP8 output scaling) |
| `--use_e` | off | Enable matrix E. Forward: stores pre-activation output. With `--gradient`: reads saved activations as input. |
| `--aux_type` | (d_type) | Data type of matrix E. Requires `--use_e`. |
| `--gradient` | off | Switch epilogue to backward-pass mode (activations become derivatives). Combine with `--use_e`. |
| `--swizzleA` | off | Reorder A into a tiled layout matching MFMA access patterns, avoiding runtime gather. Only supported for transA=T, transB=N with FP16, BF16, FP8 (FNUZ), or FP4 types. |
| `--swizzleB` | off | Reorder B into a tiled layout matching MFMA access patterns. Same type/transpose constraints as `--swizzleA`. |

**Benchmark control:**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--function` | `-f` | matmul | BLASLt function to test |
| `--verify` | `-v` | off | Validate GPU results with CPU |
| `--iters` | `-i` | 10 | Number of timed iterations (the "hot loop"). Reported `us` is the average across all iterations. |
| `--cold_iters` | `-j` | 2 | Untimed warm-up iterations before the timing loop. Warms GPU clocks, caches, and HIP runtime. Results are discarded. |
| `--initialization` | | hpl | How to fill input matrices. hpl = random in [-0.5, 0.5) following the High Performance Linpack (HPL) convention, well-conditioned and avoids overflow; rand_int = small integers; trig_float = sin/cos; zero = all zeros; norm_dist = normal distribution; uniform_01 = uniform [0,1); special = NaN/Inf/denorm edge cases; integer_exact = exact integers for reproducibility; fp16_accumulator_probe = adversarial values that stress FP16 accumulation precision (diagnostic) |
| `--rotating` | | 0 | Allocate multiple copies of all matrix buffers and cycle through them each iteration, defeating L2 cache. Value is total memory budget in MB; the tool divides it by the per-iteration data footprint to determine the copy count. 0 = off. Use 256-512 for problems that fit in L2. |
| `--flush` | | off | Flush GPU instruction cache (`s_icache_inv`) between timed iterations so each kernel dispatch starts cold. Flush overhead is measured separately and subtracted from the reported time. |
| `--use_gpu_timer` | | false | Measure kernel time via GPU events (hipEventElapsedTime) instead of host wall clock. GPU timer excludes launch overhead and is more stable. |

**Algorithm selection:**

| Flag | Default | Description |
|------|---------|-------------|
| `--algo_method` | heuristic | heuristic = library picks best kernel; all = benchmark every available solution; index = run a specific solution (set `--solution_index`) |
| `--solution_index` | -1 | Run a specific solution by index. Requires `--algo_method index`. Use `--algo_method all --print_kernel_info` to discover valid indices. -1 = none. |
| `--requested_solution` | 1 | How many solutions the heuristic returns (each benchmarked separately). -1 = all. With `--algo_method heuristic` only. |

**Tuning parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--splitk` | 0 | Split K dimension across workgroups for better utilization on small-M/N problems (0 = solution default). Requires `--api_method` cpp or mix. See [Ch10 glossary](10-troubleshooting-and-reference.md#4-glossary-essentials). |
| `--wgm` | 0 | Override workgroup-to-tile mapping strategy (0 = solution default). Affects cache locality. Requires `--api_method` cpp or mix. See [Ch10 glossary](10-troubleshooting-and-reference.md#4-glossary-essentials). |

**API and output:**

| Flag | Default | Description |
|------|---------|-------------|
| `--api_method` | c | Which hipBLASLt API to use. c = C API (`hipblasLtMatmul`): full setup + dispatch each call, no caching. cpp = C++ ext API (`hipblaslt_ext::Gemm`): caches kernel selection after initialize(), minimal dispatch overhead. mix = C API for setup, ext API for cached dispatch. |
| `--grouped_gemm` | off | Launch multiple GEMMs with different sizes in one kernel. See [Ch5 Grouped GEMM](05-api-guide.md#grouped-gemm-essentials). |
| `--use_user_args` | off | Pack grouped GEMM arguments into a device-side buffer to reduce launch overhead. Requires `--grouped_gemm`. |
| `--c_equal_d` | off | C and D share the same device allocation (in-place update). C is overwritten. |
| `--workspace` | 128 MB | Max GPU device memory (bytes) for internal use by kernel algorithms (e.g., split-K partial-sum buffers). The heuristic excludes solutions that need more than this budget. Default 128 MB. See [Ch5 Workspace](05-api-guide.md#workspace-management-deep-dive). |
| `--device` | 0 | GPU device index |
| `--HMM` | off | Use HIP managed memory (HMM) where the runtime handles host/device data migration |
| `--log_function_name` | off | Prepend the BLASLt function name to each output row |
| `--function_filter` | | Only run functions whose name contains this substring |
| `--print_kernel_info` | off | Print solution name, kernel name, and index for each result |
| `--dump_matrix` | off | Dump input/output matrices to file for debugging |
| `--skip_slow_solution_ratio` | 0 | With `--algo_method all`: skip solutions whose cold-iteration time × ratio exceeds the best seen so far. Range 0-1: 0 = disabled; 0.5 = skip if >2× slower; 1.0 = skip anything slower than current best. |
| `--version` | | Print version number and exit |
| `--data` | | Read test parameters from a `.data` file |
| `--yaml` | | Read test parameters from a YAML file |

### Output columns

The default output is CSV.  Key columns:

| Column | Meaning |
|--------|---------|
| `hipblaslt-Gflops` | Achieved GFLOPS |
| `hipblaslt-GB/s` | Achieved bandwidth (GB/s) |
| `us` | Elapsed time in microseconds |
| `CPU-Gflops` | CPU reference GFLOPS (with `-v`) |
| `CPU-us` | CPU reference time (with `-v`) |
| `norm_error` | Numerical error vs CPU (with `-v`) |
| `atol` / `rtol` | Absolute / relative tolerance (with `-v`) |

### Environment variables

| Variable | Effect |
|----------|--------|
| `HIPBLASLT_BENCH_FREQ=1` | Include GPU core and memory clock frequencies in output |
| `HIPBLASLT_BENCH_FREQ_ALL=1` | Include per-XCD frequency columns |
| `HIPBLASLT_BENCH_PERF=1` | Include efficiency, CU count, tile granularity, and memory byte counts |

### Tuning mode defaults

When `HIPBLASLT_TUNING_FILE` is set, several flags change their defaults
to values suitable for solution sweeping:

| Flag | Normal default | Tuning default |
|------|---------------|----------------|
| `--iters` | 10 | 1000 |
| `--cold_iters` | 2 | 1000 |
| `--requested_solution` | 1 | -1 (all) |
| `--rotating` | 0 | 512 |
| `--flush` | off | on |
| `--workspace` | 128 MB | from `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` |

These defaults prioritize measurement accuracy over speed: high iteration
counts stabilize GPU clocks, buffer rotation prevents L2 cache bias, and
I-cache flushing ensures fair comparison between solutions.

### Common benchmarking patterns

```bash
# Quick smoke test: default sizes, FP16
./hipblaslt-bench

# Large GEMM with extended iterations and GPU timer
./hipblaslt-bench -m 4096 -n 4096 -k 4096 -i 100 -j 10 --use_gpu_timer

# FP8 GEMM with scalar scaling
./hipblaslt-bench --a_type f8_r --b_type f8_r --d_type bf16_r \
    --scaleA 1 --scaleB 1 -m 4096 -n 4096 -k 4096

# Sweep all solutions for a problem
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --algo_method all

# Benchmark a specific solution index
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --algo_method index --solution_index 3

# Print kernel info for debugging
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --print_kernel_info

# Grouped GEMM
./hipblaslt-bench --grouped_gemm -m 1024 -n 1024 -k 1024 --batch_count 4

# Fused GEMM with bias + GELU
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --bias_vector --activation_type gelu

# Rotating buffer to reduce cache effects
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --rotating 256

# Efficiency analysis
HIPBLASLT_BENCH_PERF=1 ./hipblaslt-bench -m 4096 -n 4096 -k 4096 --use_gpu_timer
```


## 4. rtest.py `[Essentials]`

`rtest.py` is a thin wrapper around `hipblaslt-test` that selects predefined
test tiers via a single flag.  It lives in the project root.

### Usage

```bash
cd build/clients
python3 ../../rtest.py -e <mode>
```

Or, pointing to the build directory:

```bash
python3 rtest.py -e <mode> -i build/clients
```

### Modes

| Mode | gtest filter applied | What it runs |
|------|---------------------|--------------|
| `smoke` | `--gtest_filter=*smoke*` | Minimal sanity checks (small sizes, all precisions) |
| `regression` | `--gtest_filter=*quick*` | Broader coverage: quick-category tests |
| `extended` | `--gtest_filter=*pre_checkin*:*nightly*` | Pre-checkin and nightly tests combined |

### Options

| Flag | Description |
|------|-------------|
| `-e`, `--emulation` | Required.  One of: `smoke`, `regression`, `extended` |
| `-i`, `--install_dir` | Directory containing `hipblaslt-test` (default: current directory) |
| `-o`, `--output` | Test output format passed to `--gtest_output=` (default: `xml`) |

### When to use rtest.py vs gtest directly

- **Use `rtest.py`** for CI-style "run a tier" with a single command.
- **Use `hipblaslt-test --gtest_filter=` directly** when you need custom
  filter patterns, want to combine includes and excludes, or need to pass
  additional gtest flags (e.g., `--gtest_repeat`, `--gtest_shuffle`).


## 5. TensileLite tests `[Deep Dive]`

TensileLite has its own test infrastructure using `tox` and `pytest`, separate
from the hipBLASLt gtest suite.  These tests exercise the kernel code
generation, assembly, and solution matching -- they do not test the hipBLASLt
host API.

### Tox environments

All commands below run from the `tensilelite/` directory.

| Environment | Command | Description |
|-------------|---------|-------------|
| `py3` | `tox -e py3 -- Tensile/Tests -m common` | Full test suite: builds the client, runs rocisa tests, then common tests |
| `unit` | `tox -e unit -- Tensile/Tests/unit` | Python unit tests only (skips client build, fast). Assumes rocisa has been built previously (`invoke rocisa`) |
| `rocisa` | `tox -e rocisa` | rocisa module tests only |
| `lint` | `tox -e lint` | flake8 linting on `Tensile/` |
| `format` | `tox -e format` | Auto-format code with black |
| `isort` | `tox -e isort` | Sort import statements with isort |
| `pre_commit` | `tox -e pre_commit` | Lint + unit tests (quick pre-commit check) |
| `coverage` | `tox -e coverage` | Full coverage: unit + common tests with HTML/XML/JSON reports |
| `coverage-unit` | `tox -e coverage-unit` | Coverage for unit tests only (fast) |
| `coverage-common` | `tox -e coverage-common` | Coverage for common tests only |

### Pytest markers

| Marker | Meaning |
|--------|---------|
| `common` | End-to-end tests that build and run kernels on GPU |
| `unit` | Pure Python unit tests (no GPU required) |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TENSILE_NUM_PYTEST_WORKERS` | 4 | Number of parallel pytest-xdist workers |
| `TENSILELITE_CLIENT_ARGS` | (empty) | Extra args passed to `invoke build-client` during tox |

### Running individual test YAMLs

After building the TensileLite client, you can run a single test YAML directly
without tox or pytest:

```bash
cd tensilelite

# Build client (once)
invoke rocisa
invoke build-client

# Run a single test
Tensile/bin/Tensile Tensile/Tests/common/exception/<test>.yaml tensile-out
```

With a custom client build location, pass `--prebuilt-client`:

```bash
Tensile/bin/Tensile Tensile/Tests/common/gemm/<test>.yaml tensile-out \
    --prebuilt-client=my-build/tensilelite/client/tensilelite-client
```

### Custom build flags via tox

```bash
# Debug build targeting a specific GPU
TENSILELITE_CLIENT_ARGS="--build-type Debug --gpu-targets gfx90a --clean" \
    tox -e py3 -- Tensile/Tests -m common

# Single pytest worker (useful for debugging)
TENSILE_NUM_PYTEST_WORKERS=1 tox -e py3 -- Tensile/Tests -m common
```


## 6. Adding test cases `[Deep Dive]`

### Step 1: Choose the right YAML file

| File | Contents |
|------|----------|
| `smoke_gtest.yaml` | Smoke tests (small sizes, fast) |
| `matmul_gtest.yaml` | Main matmul tests (quick, pre_checkin, nightly) |
| `auxiliary_gtest.yaml` | Auxiliary operation tests |
| `rocroller_gtest.yaml` | RocRoller JIT kernel tests |
| `hipblaslt_common.yaml` | Shared anchors: precision lists, default parameters |
| `matmul_common.yaml` | Shared anchors: matrix size ranges |
| `known_bugs.yaml` | Platform-specific expected failures |

### Step 2: Write the YAML entry

A test entry needs at minimum: `name`, `category`, `function` (with
precisions), and `matrix_size` or explicit `M`/`N`/`K`.

Example -- adding a smoke test for BF16 with bias and ReLU:

```yaml
- name: matmul_bf16_bias_relu_custom_smoke
  category: smoke
  function:
    matmul: *real_precisions_2b    # expands to bf16_r + f16_r (defined in hipblaslt_common.yaml)
  matrix_size:
    - { M: 256, N: 256, K: 256 }
  transA_transB: *transA_transB_range
  alpha: 1
  beta: [ 0.0, 2.0 ]
  activation_type: relu
  bias_vector: 1
  unit_check: 1
```

Key fields:

| Field | Purpose |
|-------|---------|
| `name` | Unique test name (appears in gtest output) |
| `category` | smoke, quick, pre_checkin, or nightly |
| `function` | `matmul:` followed by a precision anchor or explicit type dict |
| `matrix_size` | List of `{M, N, K}` dicts (or use an anchor like `*smoke_matrix_size_range`) |
| `transA_transB` | Transpose combinations (usually `*transA_transB_range`, defined locally in each test YAML file, not in the shared common files) |
| `alpha` / `beta` | Scalar values or lists to expand |
| `gpu_arch` | Regex matching GPU architecture strings (optional) |
| `unit_check` | 1 = exact comparison, 0 = skip |
| `norm_check` | 1 = norm-based comparison (for approximate results like GELU) |

### Step 3: Regenerate test data

```bash
cmake --build build --target hipblaslt-test-data
```

Or simply rebuild `hipblaslt-test`, which depends on the data target:

```bash
cmake --build build --target hipblaslt-test
```

### Step 4: Verify the new test appears

```bash
cd build/clients
./hipblaslt-test --gtest_filter=*bf16_bias_relu_custom_smoke* --gtest_list_tests
```

`--gtest_list_tests` prints matching test names without running them.

### Step 5: Run and validate

```bash
./hipblaslt-test --gtest_filter=*bf16_bias_relu_custom_smoke*
```

### Marking a known bug

If a test is expected to fail on specific platforms, add an entry in
`known_bugs.yaml`:

```yaml
- { function: matmul, a_type: bf16_r, b_type: bf16_r, M: 256, N: 256,
    known_bug_platforms: "gfx908" }
```

The generator matches all specified fields against each test.  If every field
matches, the test is reclassified as `known_bug` on those platforms and
skipped.


## 7. Performance regression testing `[Deep Dive]`

hipBLASLt does not ship a built-in regression harness with stored baselines.
Performance regression detection is done by comparing `hipblaslt-bench`
output across builds or commits.

### Collecting a baseline

```bash
# Record baseline (save the full CSV output)
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
    -i 100 -j 20 --use_gpu_timer > baseline.csv

# After a code change, collect the same run
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
    -i 100 -j 20 --use_gpu_timer > current.csv
```

### Best practices for reliable comparisons

- **Use `--use_gpu_timer`** to measure kernel time via HIP events, removing
  host-side jitter.
- **Increase iterations** (`-i 100` or more) to reduce variance.
- **Use cold iterations** (`-j 20`) to warm up caches and clock boosting.
- **Use `--rotating <MB>`** to avoid unrealistic cache hit rates on small
  problems.
- **Fix the GPU clock** (via `rocm-smi --setperflevel high`) to eliminate
  frequency variation.
- **Run on an idle GPU** with no other workloads.
- **Compare the `us` column** (microseconds) as the primary metric.
  `hipblaslt-Gflops` is derived from it.

### What constitutes a regression

There is no universal threshold.  Guidelines:

| Problem size | Concern threshold |
|-------------|-------------------|
| Large GEMMs (M,N,K >= 4096) | > 2% slower consistently |
| Medium GEMMs (512-4096) | > 5% slower consistently |
| Small GEMMs (< 512) | High variance expected; focus on large regressions |

Always run multiple times and compare medians rather than single-run
results.  GPU frequency fluctuations and thermal throttling can cause
5-10% variation in individual runs.

### Sweeping all solutions

To check if a regression is in the kernel or in the solution selection
heuristic:

```bash
# Before change
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --algo_method all --print_kernel_info > before_all.csv

# After change
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --algo_method all --print_kernel_info > after_all.csv
```

If individual solution performance is unchanged but the default pick is slower,
the regression is in the heuristic (logic file selection), not in the kernels.
