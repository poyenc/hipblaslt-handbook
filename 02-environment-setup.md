# Chapter 2: Environment Setup

This chapter walks you through getting a working hipBLASLt build from a fresh checkout. It covers prerequisites, cloning, three build methods (from simplest to most flexible), build verification, and advanced build options.

---

## Prerequisites [Essentials]

### Hardware

An AMD GPU supported by hipBLASLt. The supported architectures are defined in `cmake/tensilelite_supported_architectures.cmake` and include:

| Family | Architectures |
|--------|---------------|
| CDNA | gfx908, gfx90a, gfx942, gfx950 |
| RDNA 3 | gfx1100, gfx1101, gfx1102, gfx1103 |
| RDNA 3.5 | gfx1150, gfx1151, gfx1152, gfx1153 |
| RDNA 4 | gfx1200, gfx1201 |
| RDNA 4+ | gfx1250 |

### Software

| Requirement | Minimum version | Notes |
|-------------|-----------------|-------|
| ROCm | 6.0+ (tested) | Conventionally installed at `/opt/rocm` |
| CMake | 3.25.2 | Matches `cmake_minimum_required` in `CMakeLists.txt` |
| Python | 3.8+ | Required for device library generation and test data |
| C++ compiler | `amdclang++` | Ships with ROCm at `/opt/rocm/bin/amdclang++` |
| C compiler | `amdclang` | Ships with ROCm at `/opt/rocm/bin/amdclang` |
| Fortran compiler | `gfortran` | Client builds only |
| LAPACK + BLAS | `liblapack-dev`, `libblas-dev` | Client builds only |
| msgpack-cxx | `libmsgpack-dev` | Serialization library for TensileLite |
| Google Test + Mock | `libgtest-dev`, `libgmock-dev` | Client builds only (test framework) |

Install all non-ROCm dependencies at once (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y gfortran liblapack-dev libblas-dev libmsgpack-dev libgtest-dev libgmock-dev
```

The `cmake` from `apt` is typically too old (3.22 on Ubuntu 22.04; hipBLASLt requires 3.25.2+). Install a recent version from cmake.org:

```bash
wget https://github.com/Kitware/CMake/releases/download/v4.3.2/cmake-4.3.2-linux-x86_64.sh
chmod +x cmake-4.3.2-linux-x86_64.sh
sudo ./cmake-4.3.2-linux-x86_64.sh --skip-license --prefix=/usr/local
```

> **CMake 4.x compatibility note:** CMake 4.x removed compatibility with `cmake_minimum_required` < 3.5. RocRoller fetches yaml-cpp which uses version 3.4. If you see `Compatibility with CMake < 3.5 has been removed`, add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to your cmake command. This is not needed when RocRoller is disabled (`-DHIPBLASLT_ENABLE_ROCROLLER=OFF`).

### Monorepo source dependencies (containers / sparse checkout)

hipBLASLt lives at `projects/hipblaslt/` in the monorepo, but its build references sibling directories under `shared/`. If you're working in a container or sparse checkout, you need to check out (or mount) these additional folders:

| Monorepo path | Used by | Disable option |
|---|---|---|
| `shared/origami` | Host library (Stream-K) + rocisa | None — always required |
| `shared/stinkytofu` | rocisa (ISA assembler for TensileLite) | None — always required |
| `shared/mxdatagenerator` | Client tests/benchmarks (MX format data) | `-DHIPBLASLT_ENABLE_MXDATAGENERATOR=OFF` |
| `shared/rocroller` | Host library (JIT kernels) | `-DHIPBLASLT_ENABLE_ROCROLLER=OFF` |

See [Cloning the Repo](#cloning-the-repo-essentials) for sparse checkout commands.

**Container mount example** (assuming monorepo at `/workspace`):

```bash
docker run -v /path/to/rocm-libraries/projects/hipblaslt:/workspace/projects/hipblaslt \
           -v /path/to/rocm-libraries/shared/origami:/workspace/shared/origami \
           -v /path/to/rocm-libraries/shared/stinkytofu:/workspace/shared/stinkytofu \
           -v /path/to/rocm-libraries/shared/mxdatagenerator:/workspace/shared/mxdatagenerator \
           -v /path/to/rocm-libraries/shared/rocroller:/workspace/shared/rocroller \
           ...
```

`shared/origami` and `shared/stinkytofu` are always required — there is no CMake option to disable them. The optional ones (`rocroller`, `mxdatagenerator`) can be disabled at configure time — see [Method 2](#method-2-cmake-presets) and [Method 3](#method-3-cmake-directly-single-architecture) below for the exact commands.

### Operating system

Supported Linux distros (from `tasks.py` `_supported_distros()`): Ubuntu, CentOS, AlmaLinux, RHEL, Fedora, SLES, openSUSE Leap, Mariner, Azure Linux. Windows is also supported through a separate code path in `tasks.py` (see `README.md` for Windows-specific instructions).

---

## Cloning the Repo [Essentials]

hipBLASLt lives inside the `rocm-libraries` monorepo at `projects/hipblaslt/`. Its build also references sibling directories under `shared/` (see the [monorepo source dependencies](#monorepo-source-dependencies-containers--sparse-checkout) table above).

### Full clone

```bash
git clone https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries/projects/hipblaslt
```

### Sparse checkout (recommended)

Only downloads hipBLASLt and its required shared dependencies:

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipblaslt shared/origami shared/stinkytofu shared/mxdatagenerator shared/rocroller
git checkout develop  # or the branch you are starting from
```

If you don't need RocRoller JIT or MX data generation, you can omit those folders and disable them at configure time:

```bash
git sparse-checkout set projects/hipblaslt shared/origami shared/stinkytofu
```

After either method, all build commands run from `projects/hipblaslt`.

---

## Building hipBLASLt [Essentials]

All build methods run TensileLite Python during the build. Set up a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `requirements.txt` installs:

```
invoke
cmake>=3.25.2
PyYAML
joblib>=1.4.0
packaging
msgpack
simplejson
ujson
orjson
yappi
```

**2. Build:**

```bash
# Basic release build (library only, no test/bench binaries)
inv build --architecture gfx950

# Build with client tests and benchmarks
inv build --architecture gfx950 --clients

# Debug build
inv build --debug --architecture gfx950

# Incremental rebuild (reuses CMake and FetchContent cache)
inv build --architecture gfx950

# Full clean rebuild
inv build --architecture gfx950 --clean

# Install system dependencies, build with clients, and install the package
inv build --install-deps --clients --install-pkg --architecture gfx950

# See all options
inv --help build
```

The build output goes to `build/release/` (or `build/debug/` for debug builds, `build/release-debug/` for `RelWithDebInfo`). The `--architecture` flag accepts a single target (e.g. `gfx950`), multiple targets quoted with semicolons (e.g. `"gfx90a;gfx942"`), or `all` (default).

> **Note:** To build hipBLASLt for ROCm <= 6.2, pass `--legacy-hipblas-direct` to `inv build`.

### Method 2: CMake presets

CMake presets provide named configurations defined in `CMakePresets.json`. They set compiler paths, install prefix, and component toggles automatically. Make sure the venv is activated (see above) or pass `-DPython_EXECUTABLE=$(pwd)/.venv/bin/python -DPython3_EXECUTABLE=$(pwd)/.venv/bin/python` to CMake.

**Available presets:**

| Preset | What it builds |
|--------|----------------|
| `default:release` | Full build for all architectures, installs to `/opt/rocm` |
| `hipblaslt` | Host library only (no device libs, no clients) |
| `gemm-libs` | Device GEMM libraries only |
| `hipblaslt-clients` | Host + device + hipBLASLt client tests/benchmarks (no samples, no TensileLite client) |
| `tensilelite` | TensileLite host + client only |
| `rocisa` | rocisa module only |

```bash
# List available presets
cmake --list-presets

# Full release build (host + device + clients)
cmake --preset default:release \
  -DGPU_TARGETS=gfx950 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DHIPBLASLT_ENABLE_BLIS=OFF
cmake --build build --parallel

# Host library only (no device libs, no clients)
cmake --preset hipblaslt \
  -DGPU_TARGETS=gfx950 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --parallel

# Device GEMM libraries only
# Use TENSILELITE_LOGIC_FILTER to limit which logic files are compiled and avoid
# OOM at link time. "gfx950/Equality/*" builds ~30 files instead of all 617.
# Remove this flag to build the full library (may need 32 GB+ RAM or swap).
cmake --preset gemm-libs \
  -DGPU_TARGETS=gfx950 \
  -DTENSILELITE_LOGIC_FILTER="gfx950/Equality/*"
cmake --build build --parallel

# Host + device + clients (tests and benchmarks)
cmake --preset hipblaslt-clients \
  -DGPU_TARGETS=gfx950 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DHIPBLASLT_ENABLE_BLIS=OFF \
  -DHIPBLASLT_ENABLE_ROCROLLER=OFF \
  -DHIPBLASLT_ENABLE_MXDATAGENERATOR=OFF
cmake --build build --parallel

# TensileLite host + client only
cmake --preset tensilelite \
  -DGPU_TARGETS=gfx950 \
  -DHIPBLASLT_ENABLE_YAML=OFF
cmake --build build --parallel

# rocisa module only
cmake --preset rocisa \
  -DHIPBLASLT_ENABLE_YAML=OFF
cmake --build build --parallel
```

> **Note:** Presets assume ROCm is installed at `/opt/rocm`. Additional version-pinned presets (e.g. `rocm-7.0.0`) exist for specific ROCm releases; run `cmake --list-presets` for the full list. See `CMakePresets.json` for the variables each preset configures.

Append `-D<OPTION>=<VALUE>` to customize the build. The most commonly needed options:

- **`-DGPU_TARGETS=gfx950`** — Build for your GPU only. Default builds all 14 base architectures, which is the main reason builds are slow.
- **`-DCMAKE_POLICY_VERSION_MINIMUM=3.5`** — Required with CMake 4.x when RocRoller is ON (i.e. any preset that builds the host library). Harmless to include unconditionally.
- **`-DHIPBLASLT_ENABLE_BLIS=OFF`** — BLIS is not available via `apt`. Disable unless built from source.
- **`-DHIPBLASLT_ENABLE_ROCROLLER=OFF`** — Disable if `shared/rocroller` is not available. Removes JIT kernel support but precompiled TensileLite kernels still work.
- **`-DHIPBLASLT_ENABLE_MXDATAGENERATOR=OFF`** — Disable if `shared/mxdatagenerator` is not available.
- **`-DHIPBLASLT_ENABLE_YAML=OFF`** — The `rocisa` and `tensilelite` presets enable YAML by default, which requires LLVM development packages (`llvm-dev`). Disable unless `llvm-dev` is installed.
- **`-DTENSILELITE_LOGIC_FILTER="gfx950/Equality/*"`** — Limits which logic files are compiled during device library builds to avoid OOM at link time. Note: `"*"` is the default and has no effect — use a specific subdirectory pattern.

See the [Build Options Cheat Sheet](#build-options-cheat-sheet-deep-dive) for the full list of all CMake options.

### Method 3: CMake directly (single architecture)

For maximum control, configure CMake manually. See options listed above or the [Build Options Cheat Sheet](#build-options-cheat-sheet-deep-dive) for all available flags.

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx950 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DHIPBLASLT_ENABLE_BLIS=OFF

cmake --build build --parallel
```

---

## Verifying the Build [Essentials]

After building with `--clients` (invoke) or with `HIPBLASLT_ENABLE_CLIENT=ON` (CMake), verify the build by running a test and a benchmark.

### Run a quick test

```bash
# If you used invoke:
cd build/release/clients

# If you used CMake presets or CMake directly:
cd build/clients

./hipblaslt-test --gtest_filter=*quick*
```

Expected: test cases pass with output like `[  PASSED  ] N tests.`

> **Note:** `hipblaslt-test` looks for `hipblaslt_gtest.data` in the same
> directory as the binary (resolved via `/proc/self/exe`).  The build system
> places both together, so no special working-directory setup is needed —
> you can also run the binary using its full path from the project root.

### Run a basic benchmark

```bash
./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r
```

Expected: a table of timing results (hipblaslt-gflops, us columns).

To also validate against a CPU reference:

```bash
./hipblaslt-bench --precision f32_r -v
```

---

## Build Options Cheat Sheet [Deep Dive]

Key CMake options and their defaults (from `CMakeLists.txt` and `device-library/CMakeLists.txt`):

### Project-wide options

| Option | Default | When to use |
|--------|---------|-------------|
| `GPU_TARGETS` | all supported | Set to your GPU arch for faster builds (e.g. `gfx950`) |
| `CMAKE_BUILD_TYPE` | Release | `Debug` for debugging, `RelWithDebInfo` for profiling |
| `HIPBLASLT_ENABLE_HOST` | `ON` | `OFF` to skip the main shared library |
| `HIPBLASLT_ENABLE_DEVICE` | `ON` | `OFF` to skip device library generation (faster builds, but matmul tests won't run) |
| `HIPBLASLT_ENABLE_CLIENT` | `ON` | `OFF` to skip test/bench/sample binaries |
| `HIPBLASLT_ENABLE_ROCROLLER` | `ON` | `OFF` to disable JIT kernel generation via RocRoller |
| `HIPBLASLT_ENABLE_LAZY_LOAD` | `ON` | Lazy-loads code objects to reduce init memory; disable with `OFF` for debugging |
| `HIPBLASLT_ENABLE_YAML` | `OFF` | `ON` to use YAML instead of msgpack for config parsing |
| `HIPBLASLT_ENABLE_OPENMP` | `ON` | `OFF` to disable OpenMP (forced off on Windows) |
| `HIPBLASLT_ENABLE_BLIS` | `ON` | CPU reference library for test validation; `OFF` to disable (not available via `apt`) |
| `HIPBLASLT_ENABLE_MXDATAGENERATOR` | `ON` | MX format data generation for tests; `OFF` if `shared/mxdatagenerator` is unavailable |
| `HIPBLASLT_ENABLE_EXTOPS` | `ON` | `OFF` to skip building ExtOp device libraries (softmax, layernorm, amax) |
| `HIPBLASLT_ENABLE_MATRIX_TRANSFORM` | `ON` | `OFF` to skip building matrix transform device libraries |
| `HIPBLASLT_ENABLE_ASAN` | `OFF` | `ON` for Address Sanitizer builds (requires `xnack+` architecture) |
| `HIPBLASLT_ENABLE_MARKER` | `ON` | `OFF` to disable ROCm profiling markers |
| `HIPBLASLT_ENABLE_FETCH` | `OFF` | `ON` to fetch dependencies via FetchContent (otherwise they must be pre-installed) |
| `HIPBLASLT_ENABLE_HIPBLAS_DIRECT` | `OFF` | `ON` to use hipblas header directly (for ROCm <= 6.2; see `--legacy-hipblas-direct` in invoke) |
| `HIPBLASLT_ENABLE_TSAN` | `OFF` | `ON` for Thread Sanitizer builds |
| `HIPBLASLT_ENABLE_COVERAGE` | `OFF` | `ON` to build with coverage support |

### Client options (only when `HIPBLASLT_ENABLE_CLIENT=ON`)

| Option | Default | When to use |
|--------|---------|-------------|
| `HIPBLASLT_BUILD_TESTING` | `ON` | `OFF` to skip gtest binary |
| `HIPBLASLT_ENABLE_SAMPLES` | `ON` | `OFF` to skip sample programs |

### TensileLite options

| Option | Default | When to use |
|--------|---------|-------------|
| `TENSILELITE_ENABLE_HOST` | `ON` | Build the TensileLite C++ host runtime |
| `TENSILELITE_ENABLE_CLIENT` | `OFF` | Build the standalone TensileLite client |
| `TENSILELITE_ENABLE_AUTOBUILD` | `OFF` | Generate wrapper scripts for TensileLite Python (most users should use `Tensile/bin/Tensile` directly instead) |
| `TENSILELITE_BUILD_TESTING` | `OFF` | Build TensileLite host library tests |
| `TENSILELITE_BUILD_PARALLEL_LEVEL` | `""` (uses nproc) | Number of CPU cores for device library builds |
| `TENSILELITE_LOGIC_FILTER` | `""` (uses `*`) | Filter which logic YAML files are processed (e.g. `gfx942/Equality/*`) |
| `TENSILELITE_NO_COMPRESS` | `""` (off) | Set to non-empty to skip code object compression |
| `TENSILELITE_EXPERIMENTAL` | `""` (off) | Set to non-empty to process experimental logic files |

> **Note:** The TensileLite options above are CMake string cache variables, not booleans. An empty string `""` means "use the default behavior" shown in parentheses.

### Device library paths

| Option | Default | When to use |
|--------|---------|-------------|
| `HIPBLASLT_LIBLOGIC_PATH` | `""` (empty, resolves to source `library/` directory) | Custom path to library logic files |
| `HIPBLASLT_TENSILE_LIBPATH` | `<build_dir>/Tensile` | Path to output device GEMM libraries; set to an existing ROCm install's library dir to skip building device libs |

---

## TensileLite Dev Setup [Deep Dive]

If you are working on TensileLite (kernel code generation, logic files, or the C++ host runtime), you need additional setup beyond the main hipBLASLt build.

### Build rocisa (required once)

rocisa is the Python/C++ ISA assembler module used by TensileLite's code generator. Build it once after cloning (or after `pyproject.toml` changes):

```bash
cd tensilelite
pip3 install invoke   # if not already installed
invoke rocisa
```

### Build the TensileLite client

The TensileLite client (`tensilelite-client`) is a standalone C++ executable for running individual kernel tests:

```bash
cd tensilelite

# Build to the default location (tensilelite/build_tmp)
invoke build-client

# Override toolchain and architecture
invoke build-client \
  --gpu-targets gfx950 \
  --rocm-path /opt/rocm-7.3.0 \
  --export-compile-commands
```

### Run individual TensileLite tests

```bash
cd tensilelite
Tensile/bin/Tensile Tensile/Tests/common/exception/<test>.yaml tensile-out
```

### Run the full test suite with tox

```bash
cd tensilelite

# Full common test suite
tox -e py3 -- Tensile/Tests -m common

# Unit tests only
tox -e unit -- Tensile/Tests/unit

# Coverage report
tox -e coverage
```

### Rebuild cheat sheet

| What you changed | Command to rebuild |
|---|---|
| rocisa or stinkytofu C++ sources | `invoke rocisa` |
| tensilelite-client C++ sources | `invoke build-client` |
| TensileLite Python code | No rebuild needed |
| rocisa `pyproject.toml` or `CMakeLists.txt` | `invoke rocisa` |

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TENSILE_NUM_PYTEST_WORKERS` | `4` | Number of parallel pytest workers used by tox |
| `TENSILELITE_CLIENT_ARGS` | (empty) | Additional arguments passed to `invoke build-client` during tox runs |

---

Next: [Architecture](03-architecture.md) -- learn how the layers of hipBLASLt fit together.
