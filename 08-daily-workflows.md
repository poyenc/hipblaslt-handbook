# Chapter 8 -- Daily Workflows

This chapter is a recipe book. Each section addresses a concrete task you will face
during hipBLASLt development, with exact commands and file paths.

---

## 1. "I need to build for a specific GPU"

### Using invoke (preferred)

The `inv build` task wraps CMake configuration and build in a single command.
The `--architecture` flag maps directly to `-DGPU_TARGETS` in CMake.

```bash
# Single architecture
inv build --architecture gfx950

# Multiple architectures (semicolon-separated, quoted)
inv build --architecture "gfx90a;gfx942;gfx950"

# All supported architectures (default)
inv build --architecture all
```

Common flag combinations:

```bash
# Library + tests + benchmarks
inv build --architecture gfx950 --clients

# Debug build
inv build --debug --architecture gfx950

# Debug build with clients
inv build --debug --architecture gfx950 --clients

# Clean rebuild (removes previous build directory first)
inv build --architecture gfx950 --clean

# RelWithDebInfo (keeps debug symbols, release optimization)
inv build --relwithdebinfo --architecture gfx950

# Skip the RocRoller JIT backend
inv build --architecture gfx950 --skip-rocroller

# Use a subset of logic files
inv build --architecture gfx950 --logic-filter "gfx950/Equality/*"
```

The build output goes to `build/release/` (release), `build/debug/` (debug), or
`build/release-debug/` (RelWithDebInfo).

### Using CMake directly

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx950

cmake --build build --parallel
```

For multiple targets: `-DGPU_TARGETS="gfx90a;gfx942;gfx950"`.

The full list of supported architecture names is defined in
`cmake/tensilelite_supported_architectures.cmake`.

> **Cross-reference:** See [Chapter 2 -- Environment Setup](02-environment-setup.md)
> for prerequisite toolchain installation and venv setup.

---

## 2. "I need to add a kernel solution"

Adding a new kernel solution involves the TensileLite Python toolchain and the
logic YAML files that control which solutions are selected at runtime.

### Step-by-step flow

1. **Identify the problem type.** Determine the precision combination
   (e.g., bf16 I/O + f32 compute), transpose modes, and target architecture.

2. **Write or modify solution parameters.** Logic files live at:

   ```
   library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/<arch>/
   ```

   Architectures present include: `gfx950`, `aquavanjaram` (gfx942), `arcturus`
   (gfx908), `aldebaran` (gfx90a), `navi31`/`navi32`/`navi33`, `gfx1200`,
   `gfx1201`, `gfx1250`, and others.

   Each YAML file in the architecture directory describes a set of solutions
   (kernel configurations) and the matching rules that map problem sizes to
   those solutions.

3. **Build device libraries.** After modifying logic files, rebuild the Tensile
   device libraries so the new solution is compiled into code objects:

   ```bash
   cmake --build build --target tensilelite-device-libraries
   ```

   Or, with invoke:

   ```bash
   inv build --architecture gfx950
   ```

4. **Validate the new solution.** Run benchmarks to confirm the solution loads
   and produces correct results:

   ```bash
   cd build/release/clients
   # Check which solution the heuristic dispatches (--print_kernel_info)
   ./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r -v --print_kernel_info

   # Force a specific solution by index to test it directly
   ./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r -v \
       --algo_method index --solution_index <your_index>
   ```

   The `-v` flag enables CPU validation. `--print_kernel_info` prints the
   solution name and index so you can confirm your new solution is being
   dispatched. Use `--algo_method index --solution_index` to bypass the
   heuristic and test a specific solution directly.

   Solution names are auto-generated from tuning parameters (tile size, MI,
   ISA, etc.) by `Naming.py` — you don't set them. To find your solution,
   use `--algo_method all --print_kernel_info` to list all candidates and
   match by tile/MI combination (e.g., `MT256x256x64_MI16x16x1`), or use
   the `SolutionIndex` from your logic file entry.

5. **Run the test suite** to check for regressions:

   ```bash
   cd build/release/clients
   ./hipblaslt-test --gtest_filter=*quick*
   ```

> **Cross-reference:** See [Chapter 3 -- Architecture](03-architecture.md) for
> details on the TensileLite code generation pipeline and the logic file format.

---

## 3. "I need to tune a kernel"

hipBLASLt provides two tuning workflows: a **tuning utility** (`find_exact.py`)
for exact-match logic generation, and an **offline tuning script** (QuickTune)
for model-level GEMM tuning.

### Tuning utility (find_exact.py)

Located at `utilities/find_exact.py`. This tool benchmarks all available kernel
solutions for a given problem size and generates exact-match logic YAML files.

1. **Edit the template.** Copy and modify `utilities/template.yaml`:

   ```yaml
   # Two steps: comment out Bench or CreateLogic to disable either.
   Bench:
     ProblemType:
       ComputeDataType: s
       ComputeInputDataType: s
       DataTypeA: s
       DataTypeB: s
       DataTypeC: s
       DataTypeD: s
       TransposeA: 0
       TransposeB: 0
       UseBias: False
     TestConfig:
       ColdIter: 20
       Iter: 100
       AlgoMethod: "all"
       RotatingBuffer: 512
     TuningParameters:
       # SplitK: [0, 4, 8]
     ProblemSizes:
     - [128, 128, 1, 128]   # [M, N, batch_count, K]
   CreateLogic: {}
   ```

2. **Run the tuning:**

   ```bash
   python3 utilities/find_exact.py my_tuning.yaml build/release output_dir
   ```

   The script reads `build/release/device-library/MatchTable.yaml`, benchmarks
   all candidate solutions, and writes exact-match logic to `output_dir`.

3. **Integrate the result.** Copy the generated logic files into the appropriate
   architecture directory under
   `library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/<arch>/`.

### QuickTune (offline model-level tuning)

Located at `utilities/QuickTune/`. This tool tunes GEMMs extracted from a
real model workload.

1. **Extract GEMM calls from your model.** `HIPBLASLT_LOG_MASK=32` enables
   bench-format logging, which records every GEMM call as a reproducible
   `hipblaslt-bench` command line (see [Section 6](#use-the-log-mask-for-selective-output)
   for the full bit table):

   ```bash
   export HIPBLASLT_LOG_MASK=32
   export HIPBLASLT_LOG_FILE=model_gemms.log
   # Run your model inference/training
   ```

2. **Run the tuning script:**

   ```bash
   python utilities/QuickTune/gemm_tuning.py \
     --input_file model_gemms.log \
     --output_path tuning_output \
     --requested_solution 128
   ```

   Key options:
   - `--requested_solution 128` -- how many candidate solutions the
     heuristic returns for each GEMM shape. The script benchmarks all of
     them and picks the fastest. Higher values explore more solutions but
     take longer. Use `-1` to try every available solution.
   - `--gpu_id 0` -- target GPU
   - `--stablize_gpu` -- lock GPU frequency for consistent results

3. **Apply tuning results at runtime:**

   ```bash
   export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning_output/tuning.txt
   # Run your model -- tuned kernels are selected automatically
   ```

4. **Analyze results:**

   ```bash
   python utilities/QuickTune/tuning_analysis.py \
     --input_log model_gemms.log \
     --input_csv tuning_output/tuning_result.csv \
     --output_csv analysis.csv
   ```

> **Tip:** For stable tuning, lock the GPU frequency before benchmarking:
>
> ```bash
> export HIP_FORCE_DEV_KERNARG=1
> rocm-smi --setperfdeterminism 1900 -d 0
>
> # After tuning, reset:
> unset HIP_FORCE_DEV_KERNARG
> rocm-smi -r -d 0
> ```

---

## 4. "I need to benchmark a GEMM"

The `hipblaslt-bench` binary is built when you pass `--clients` to the invoke
build task. It lives at `build/release/clients/hipblaslt-bench`.

### Common flag combinations

```bash
# Basic FP16 GEMM benchmark
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r

# FP32 GEMM with CPU validation
./hipblaslt-bench --precision f32_r -v

# BF16 with transposed A, specific iteration counts
./hipblaslt-bench -m 16 -n 16 -k 4096 \
  --transA T --transB N \
  --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r \
  --compute_type f32_r \
  --cold_iters 20 --iters 100

# Test all available algorithm solutions
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
  --algo_method all

# Benchmark a specific solution by index
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
  --algo_method index --solution_index 42

# Print kernel name and solution index
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
  --print_kernel_info

# GEMM with fused activation and bias
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
  --activation_type gelu --bias_vector --bias_source d

# Grouped GEMM
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
  --grouped_gemm

# Use GPU timer for more accurate kernel timing
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
  --use_gpu_timer

# Rotating buffer to measure cold-cache performance
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
  --rotating 512
```

### Environment variables for bench output

```bash
# Show GPU frequency alongside results
HIPBLASLT_BENCH_FREQ=1 ./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r

# Show per-XCD frequencies (multi-die GPUs)
HIPBLASLT_BENCH_FREQ_ALL=1 ./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r

# Show efficiency and detailed performance metrics
HIPBLASLT_BENCH_PERF=1 ./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision bf16_r \
  --use_gpu_timer
```

### Tuning parameters in bench

```bash
# Override split-K value (requires C++ API mode)
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
  --api_method cpp --splitk 4

# Override workgroup mapping
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
  --api_method cpp --wgm 2
```

> **Cross-reference:** See [Chapter 4 -- Your First GEMM](04-your-first-gemm.md)
> for introductory usage. Full flag reference is in `clients/bench/README.md`.

---

## 5. "I need to add a test case"

Test cases are defined in YAML files under `clients/tests/data/`. At build time,
the script `clients/tests/hipblaslt_gentest.py` expands these YAML definitions
into the binary file `build/release/clients/hipblaslt_gtest.data`, which the
`hipblaslt-test` binary reads at runtime.

Note: `hipblaslt-test` exercises the **heuristic path** — the library picks
whichever solution it considers best, so a specific new kernel may not be
selected. To validate a specific solution, use `hipblaslt-bench` with
`--algo_method index --solution_index <N> -v` (see
[Section 2, Step 4](#2-i-need-to-add-a-kernel-solution)). Use gtests for
regression testing across the library, and `hipblaslt-bench -v` for
targeted kernel validation.

### YAML files in `clients/tests/data/`

| File | Purpose |
|------|---------|
| `hipblaslt_common.yaml` | Shared data types, enums, defaults, and argument structure |
| `matmul_common.yaml` | Common matrix size ranges and precision lists |
| `matmul_gtest.yaml` | Main matmul test cases (quick, pre_checkin, nightly) |
| `smoke_gtest.yaml` | Lightweight smoke tests |
| `auxiliary_gtest.yaml` | Tests for auxiliary operations (ExtOps) |
| `rocroller_gtest.yaml` | Tests for RocRoller JIT kernels |
| `hipblaslt_gtest.yaml` | Top-level file that includes others |
| `hipblaslt_template.yaml` | Template defining the Arguments structure |
| `known_bugs.yaml` | Known failures to skip on specific platforms |

### Adding a new test case

1. **Choose the right YAML file.** For matmul tests, edit `matmul_gtest.yaml`.
   For smoke tests, edit `smoke_gtest.yaml`.

2. **Write the test entry.** Test entries use YAML anchors and combinatorial
   expansion. Here is an example that tests a specific size with multiple
   precisions and transpose modes:

   ```yaml
   - name: matmul_my_feature
     category: quick
     function:
       matmul: *real_precisions
     matrix_size: *small_matrix_size_range
     transA_transB: *transA_transB_range
     alpha: 1.0
     beta: [ 0.0, 1.0 ]
   ```

   Key fields:
   - `name` -- test name (appears in gtest filter)
   - `category` -- `quick`, `pre_checkin`, `nightly`, or `smoke`
   - `function` -- maps the test to a function with precision combinations
   - `matrix_size` -- YAML anchor referencing a list of `{M, N, K}` dicts
   - `transA_transB` -- YAML anchor for transpose combinations
   - Values given as lists are expanded combinatorially

3. **Rebuild the test data.** After editing YAML, rebuild to regenerate
   `hipblaslt_gtest.data`:

   ```bash
   cmake --build build --target hipblaslt-test-data
   ```

   Or rebuild the full test binary (which also regenerates the data):

   ```bash
   cmake --build build --target hipblaslt-test
   ```

4. **Run your new test:**

   ```bash
   cd build/release/clients
   ./hipblaslt-test --gtest_filter=*my_feature*
   ```

### Adding a known bug entry

If a test fails on a specific platform and you want to mark it as a known issue,
add an entry to `clients/tests/data/known_bugs.yaml`:

```yaml
- { function: matmul, a_type: bf16_r, b_type: bf16_r, transA: N, transB: N,
    M: 512, N: 512, K: 512, known_bug_platforms: gfx908 }
```

The test will be skipped on the listed platforms. If `known_bug_platforms` is
omitted, the test is skipped on all platforms.

### How the test generator works

`hipblaslt_gentest.py` reads the YAML files (processing `include:` directives),
expands all combinatorial parameters (list values, integer ranges like
`1..10..2`, dictionary list expansions), applies defaults from
`hipblaslt_common.yaml`, matches known bugs, and writes each unique test case
as a binary record to `hipblaslt_gtest.data`.

> **Note:** `hipblaslt-test` looks for `hipblaslt_gtest.data` in the same
> directory as the binary (resolved via `/proc/self/exe`).  The build system
> places both together, so no special working-directory setup is needed.

---

## 6. "I need to debug a failing kernel"

### Enable logging

```bash
# Error-level logging
HIPBLASLT_LOG_LEVEL=1 ./hipblaslt-test --gtest_filter=*failing_test*

# Trace logging (shows kernel launch parameters)
HIPBLASLT_LOG_LEVEL=2 ./hipblaslt-test --gtest_filter=*failing_test*

# Full API trace
HIPBLASLT_LOG_LEVEL=5 ./hipblaslt-test --gtest_filter=*failing_test*

# Log to file (with PID substitution)
HIPBLASLT_LOG_LEVEL=5 HIPBLASLT_LOG_FILE=debug_%i.log \
  ./hipblaslt-test --gtest_filter=*failing_test*
```

### Use the log mask for selective output

The log mask allows combining multiple log channels via bitwise OR:

| Bit | Value | Channel |
|-----|-------|---------|
| 0 | 1 | Error |
| 1 | 2 | Trace |
| 2 | 4 | Hints |
| 3 | 8 | Info |
| 4 | 16 | API trace |
| 5 | 32 | Bench |
| 6 | 64 | Profile |
| 7 | 128 | Extended profile |

```bash
# Error + Trace + Bench output
HIPBLASLT_LOG_MASK=35 ./hipblaslt-bench -m 128 -n 128 -k 128 --precision f16_r

# Extract reproducible bench commands from any application
HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=bench_log.txt ./your_application
```

### Force a specific solution

When debugging, isolate whether a failure is solution-specific:

```bash
# List all solutions and their indices
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
  --algo_method all --print_kernel_info

# Run with a specific solution index
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r \
  --algo_method index --solution_index <index> -v
```

### CPU validation

Always enable CPU validation (`-v`) when investigating correctness issues:

```bash
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r -v
```

The output includes `norm_error`, `atol`, and `rtol` columns. A non-zero
`norm_error` indicates a mismatch between GPU and CPU results.

### Offline tuning overrides for debugging

You can force the runtime to use a specific solution for all matching problems:

```bash
# Save tuning state to file
HIPBLASLT_TUNING_FILE=debug_tuning.txt ./hipblaslt-bench -m 128 -n 128 -k 128 \
  --precision f16_r --algo_method index --solution_index 5

# Override kernel selection in your application
HIPBLASLT_TUNING_OVERRIDE_FILE=debug_tuning.txt ./your_application
```

### Stream-K configuration overrides

For debugging Stream-K related issues:

| Value | Effect |
|-------|--------|
| 0 | Data-parallel only — no Stream-K (default) |
| 1 | Enable Stream-K in the master library lookup |
| 2 | Enable Stream-K in exact-match logic libraries too |

```bash
# Disable Stream-K (use standard data-parallel kernels)
TENSILE_SOLUTION_SELECTION_METHOD=0 ./hipblaslt-bench -m 4096 -n 4096 -k 4096 \
  --precision f16_r -v

# Force a specific grid size
TENSILE_STREAMK_FIXED_GRID=64 ./hipblaslt-bench -m 4096 -n 4096 -k 4096 \
  --precision f16_r -v
```

### ROCProfiler integration

```bash
# Enable marker trace for profiling
HIPBLASLT_ENABLE_MARKER=1 rocprof --hip-trace ./hipblaslt-bench \
  -m 4096 -n 4096 -k 4096 --precision f16_r
```

> **Cross-reference:** Full environment variable reference is in
> `docs/reference/env-variables.rst`. The logging and heuristics how-to guide
> is at `docs/how-to/use-logging-heuristics.rst`.

---

## 7. "I need to modify the host API"

When adding or changing a public API function, follow this abbreviated checklist.

### Checklist

1. **Header declaration.** Add or modify the function signature in the
   appropriate header under `library/include/hipblaslt/`:
   - C API: `hipblaslt.h`
   - C++ extensions: `hipblaslt-ext.hpp`
   - ExtOp API: `hipblaslt-ext-op.h`

2. **Implementation.** Add the implementation in `library/src/amd_detail/`:
   - Top-level dispatch: `hipblaslt.cpp`, `hipblaslt-ext.cpp`, or
     `hipblaslt-ext-op.cpp` (under `library/src/amd_detail/`)
   - AMD backend: `library/src/amd_detail/rocblaslt/` -- the rocblaslt layer
     handles matmul dispatch and algorithm selection

3. **Tensile host interface.** If the change affects kernel dispatch or
   argument passing, update `library/src/amd_detail/rocblaslt/src/tensile_host.cpp`.

4. **Tests.** Add gtest coverage in `clients/tests/`. For new GEMM
   configurations, add YAML entries to `clients/tests/data/matmul_gtest.yaml`.

5. **Benchmarks.** If the new API path needs benchmarking, update
   `clients/bench/` to expose the new options via command-line flags.

6. **Samples.** Consider adding a usage example in `clients/samples/`.

7. **Documentation.** Update the relevant `.rst` files under `docs/`.

8. **Build and test.**

   ```bash
   inv build --architecture gfx950 --clients
   cd build/release/clients
   ./hipblaslt-test --gtest_filter=*your_new_test*
   ```

> **Cross-reference:** See [Chapter 3 -- Architecture](03-architecture.md)
> for the full layer diagram and [Chapter 5 -- API Guide](05-api-guide.md)
> for API design patterns.

---

## 8. "I need to run CI checks locally"

### Pre-commit hook

hipBLASLt includes a pre-commit hook at `.githooks/pre-commit`. It performs
three checks:

1. **Copyright year update** -- Updates the copyright year in the header of
   modified files (first 10 lines).

2. **Whitespace cleanup** -- Removes trailing whitespace, adds missing newline
   at end of file, and converts non-ASCII UTF-8 to ASCII for files matching
   `*.c`, `*.h`, `*.hpp`, `*.cpp`, `*.cl`, `*.in`, `*.txt`, `*.yaml`, `*.sh`,
   `*.py`, `*.pl`, `*.cmake`, `*.md`, `*.rst`, `*.groovy`.

3. **clang-format** -- If `clang-format` is installed, formats C/C++ source
   files (`*.c`, `*.h`, `*.hpp`, `*.cpp`, `*.cl`, `*.h.in`, `*.hpp.in`,
   `*.cpp.in`) using the `.clang-format` style file.

### Installing the hook

```bash
git config core.hooksPath .githooks
```

### Running the hook manually

You can run the hook on all tracked files (not just staged changes):

```bash
.githooks/pre-commit --reformat
```

Or run it on staged changes only by committing (the hook runs automatically).

### What CI validates

While the specific CI pipeline configuration is managed externally, CI
typically validates:

- **Build** -- Library and clients build successfully for target architectures.
- **Tests** -- `hipblaslt-test` passes (smoke, quick, and/or pre_checkin
  categories depending on the pipeline).
- **Formatting** -- Code conforms to clang-format and whitespace rules.
- **Copyright headers** -- Modified files have current copyright year.

### Running tests locally by category

```bash
cd build/release/clients

# Smoke tests (fastest)
./hipblaslt-test --gtest_filter=*smoke*

# Quick tests
./hipblaslt-test --gtest_filter=*quick*

# Pre-checkin tests
./hipblaslt-test --gtest_filter=*pre_checkin*
```

### Using rtest.py

The `rtest.py` script at the project root provides named test suites:

```bash
python3 rtest.py -e smoke
python3 rtest.py -e regression
```

### TensileLite tests

If your changes touch TensileLite Python code:

```bash
cd tensilelite

# Unit tests
tox -e unit -- Tensile/Tests/unit

# Common test suite
tox -e py3 -- Tensile/Tests -m common

# Coverage report
tox -e coverage
```

> **Cross-reference:** See [Chapter 2 -- Environment Setup](02-environment-setup.md)
> for venv and dependency setup required to run the test suites.
