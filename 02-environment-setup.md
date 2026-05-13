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

Install all non-ROCm dependencies at once (Ubuntu/Debian):

```bash
sudo apt install -y gfortran liblapack-dev libblas-dev libmsgpack-dev
```

The `cmake` from `apt` is typically too old (3.22 on Ubuntu 22.04; hipBLASLt requires 3.25.2+). Install a recent version from cmake.org:

```bash
wget https://github.com/Kitware/CMake/releases/download/v4.3.2/cmake-4.3.2-linux-x86_64.sh
chmod +x cmake-4.3.2-linux-x86_64.sh
sudo ./cmake-4.3.2-linux-x86_64.sh --skip-license --prefix=/usr/local
```

> **Container / sparse-checkout note:** If you only have `projects/hipblaslt` mounted (not the full monorepo), RocRoller at `../../shared/rocroller` won't be found. Disable it with `-DHIPBLASLT_ENABLE_ROCROLLER=OFF`. Similarly, disable BLIS with `-DHIPBLASLT_ENABLE_BLIS=OFF` if not installed. A typical container configure command:
>
> ```bash
> cmake --preset hipblaslt-clients -DHIPBLASLT_ENABLE_BLIS=OFF -DHIPBLASLT_ENABLE_ROCROLLER=OFF
> ```

### Operating system

Supported Linux distros (from `tasks.py` `_supported_distros()`): Ubuntu, CentOS, AlmaLinux, RHEL, Fedora, SLES, openSUSE Leap, Mariner, Azure Linux. Windows is also supported through a separate code path in `tasks.py` (see `README.md` for Windows-specific instructions).

---

## Cloning the Repo [Essentials]

hipBLASLt lives inside the `rocm-libraries` monorepo. You can clone the full repo or use sparse checkout to download only hipBLASLt.

### Full clone

```bash
git clone https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries/projects/hipblaslt
```

### Sparse checkout (faster)

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipblaslt
git checkout develop # or the branch you are starting from
```

After either method, all build commands run from `projects/hipblaslt`.

---

## Building hipBLASLt [Essentials]

Three methods are available, listed in recommended order.

### Method 1: invoke (preferred)

The `invoke` task runner wraps CMake with sensible defaults for compiler paths, build type, and dependency management. It is the recommended method for most development.

**1. Create a virtual environment and install Python dependencies:**

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

CMake presets provide named configurations defined in `CMakePresets.json`. They set compiler paths, install prefix, and component toggles automatically.

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

# Full release build
cmake --preset default:release
cmake --build build --parallel

# Host library only
cmake --preset hipblaslt
cmake --build build --parallel

# Device GEMM libraries only
cmake --preset gemm-libs
cmake --build build --parallel

# Host + device + clients (tests and benchmarks)
cmake --preset hipblaslt-clients
cmake --build build --parallel
```

> **Note:** Presets assume ROCm is installed at `/opt/rocm`. Additional version-pinned presets (e.g. `rocm-7.0.0`) exist for specific ROCm releases; run `cmake --list-presets` for the full list. See `CMakePresets.json` for the variables each preset configures.

### Method 3: CMake directly (single architecture)

For maximum control, configure CMake manually. This is useful when you need non-default paths or options.

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx950

cmake --build build --parallel
```

The above command builds everything including client tests and benchmarks (`HIPBLASLT_ENABLE_CLIENT` defaults to `ON` in CMake). Note that `inv build` defaults to clients OFF -- you must pass `--clients` to include them. To build the library only without clients using CMake directly:

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DGPU_TARGETS=gfx950 \
  -DHIPBLASLT_ENABLE_CLIENT=OFF

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
| `HIPBLASLT_ENABLE_BLIS` | `ON` | CPU reference library for test validation; `OFF` to disable |
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
