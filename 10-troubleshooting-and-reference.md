# Chapter 10: Troubleshooting and Reference

This chapter collects common errors with solutions, a complete environment variable reference, debugging tools, and a glossary of key terms.

---

## 1. Common Errors [Essentials]

### Build failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `amdclang++: command not found` | ROCm is not installed or not on `PATH`. | Install ROCm and set `-DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++`. |
| CMake fails or produces wrong layout when configured from repo root | Configuring from the monorepo root triggers the superbuild, which pulls in other projects. | Configure from `projects/hipblaslt` instead of the repo root. |
| `No GPU_TARGETS specified` or architecture mismatch | `GPU_TARGETS` was not set, or the target does not match installed hardware. | Pass `-DGPU_TARGETS=gfx942` (or your GPU) explicitly. Supported targets are listed in `cmake/tensilelite_supported_architectures.cmake`. |
| `TensileCreateLibrary` fails with `ModuleNotFoundError` (e.g. `yaml`, `msgpack`) | The Python environment is missing required packages for TensileLite. | Create a venv, run `pip install -r tensilelite/requirements.txt`, and configure with `-DPython_EXECUTABLE=$(pwd)/.venv/bin/python -DPython3_EXECUTABLE=$(pwd)/.venv/bin/python`. |
| Link errors referencing `tensilelite::tensilelite-host` | Device libraries or host library not built. | Build with `HIPBLASLT_ENABLE_DEVICE=ON` and `TENSILELITE_ENABLE_HOST=ON` (both default ON). |

### Runtime failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No solution found` / `HIPBLASLT_STATUS_NOT_FOUND` | No precompiled kernel matches the problem parameters (precision, transpose, matrix dimensions). | Verify the data type combination is supported. Check that device libraries for your GPU architecture are loaded. Use `HIPBLASLT_LOG_LEVEL=1` to see which problem was rejected. |
| Missing `TensileLibrary_lazy_*.dat` or `.hsaco` at runtime | Device libraries were not built or `HIPBLASLT_TENSILE_LIBPATH` points to the wrong directory. | Build target `tensilelite-device-libraries`, or set `HIPBLASLT_TENSILE_LIBPATH` to the directory containing `TensileLibrary_lazy_<arch>.dat` and code object files (e.g. `build/Tensile/library` or `/opt/rocm/lib/hipblaslt/library`). |
| `Workspace too small` / workspace allocation failure | The workspace buffer passed to `hipblasltMatmul` is smaller than what the selected solution requires. | Call `hipblaslt_ext::matmulIsAlgoSupported` to query the required workspace size, then allocate at least that many bytes via `hipMalloc`. |
| Wrong results or `hipErrorInvalidValue` | Mismatched data types between matmul descriptor and actual buffer pointers, or incorrect leading dimensions / strides. | Double-check that `hipblasltDatatype_t` settings for A, B, C, D match the allocated buffer types. Verify `lda >= m`, `ldb >= k` (for non-transposed), and stride values for batched GEMM. |

### Test failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `hipblaslt_gtest.data` not found | Test data file was not generated. | Build target `hipblaslt-test` (or `hipblaslt-test-data`). The binary is at `build/clients/hipblaslt-test` (or `build/release/clients/hipblaslt-test` for invoke builds) and finds the data file in its own directory via `/proc/self/exe`, so it can be run from anywhere. |
| Tests skip on your GPU | Known bug entries or unsupported architectures. | Check `clients/tests/data/known_bugs.yaml` for entries matching your test and `known_bug_platforms`. |
| Test data generation fails (PyYAML missing) | The Python environment used by CMake lacks PyYAML. | Use the venv: `pip install -r tensilelite/requirements.txt` and reconfigure with the venv Python. |
| Device library not found during tests | `HIPBLASLT_TENSILE_LIBPATH` not set and the default search path does not contain libraries for your arch. | Set `HIPBLASLT_TENSILE_LIBPATH=<path>/build/Tensile/library` or build device libraries first. |

### TensileLite errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `invoke rocisa` fails to build | Missing C++ build tools or wrong Python version. | Ensure a C++17-capable compiler and Python 3.8+ are available. Check `tensilelite/rocisa/` build output for specific errors. |
| `tox` fails with environment errors | Tox cannot find the expected Python version or dependencies. | Run `pip install tox` in your venv. Use `tox -e py3` (not a version-specific env like `py310`) unless you know your setup matches. |
| YAML test errors in TensileLite | Invalid or unsupported fields in a test YAML file. | Compare your YAML against a working example in `tensilelite/Tensile/Tests/`. Check that kernel parameters (e.g. tile sizes, data types) are valid for your target architecture. |

---

## 2. Environment Variables Reference [Essentials]

All environment variables are sourced from `docs/reference/env-variables.rst` and verified against the source code.

### Logging and debugging

| Variable | Description | Default | Valid values |
|----------|-------------|---------|--------------|
| `HIPBLASLT_LOG_LEVEL` | Controls verbosity level of logging output. Levels are cumulative (e.g. level 3 enables error + trace + hints). | `0` (off) | `0`: Off (disabled) | `1`: Error | `2`: Trace (API calls, kernel launch params) | `3`: Hints (performance suggestions) | `4`: Info (general execution info) | `5`: API trace (detailed API call params) |
| `HIPBLASLT_LOG_MASK` | Controls logging via bit mask flags (can be combined with bitwise OR). | `0` (off) | `0`: Off | `1`: Error | `2`: Trace | `4`: Hints | `8`: Info | `16`: API trace | `32`: Bench | `64`: Profile | `128`: Extended profile |
| `HIPBLASLT_LOG_FILE` | Path to log output file. Supports `%i` for process ID substitution. | Not set (logs to stderr) | File path, e.g. `logfile_%i.log` |
| `HIPBLASLT_ENABLE_MARKER` | Enables marker trace for ROCProfiler profiling. Requires building with `-DHIPBLASLT_ENABLE_MARKER=ON`. | `0` (disabled) | `0`: Disabled | `1`: Enable marker trace | `2`: Log output as markers |

### Offline tuning

| Variable | Description | Default | Valid values |
|----------|-------------|---------|--------------|
| `HIPBLASLT_TUNING_FILE` | File path to store tuning results with best solution indices for GEMM problems. Used by `hipblaslt-bench`. | Not set | File path, e.g. `tuning.txt` |
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | File path to load tuning results and override default kernel selection at runtime. | Not set | File path, e.g. `tuning.txt` |
| `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` | Maximum workspace size constraint during tuning. | `128 * 1024 * 1024` (128 MB) | Integer value in bytes |

### Origami with Stream-K configuration

These variables apply globally to all GEMMs in an application.

| Variable | Description | Default | Valid values |
|----------|-------------|---------|--------------|
| `TENSILE_SOLUTION_SELECTION_METHOD` | Controls kernel selection strategy for GEMM operations. Has no effect on MI350 series (Stream-K is always used). | `0` | `0`: Default (standard tuned libraries, no Stream-K) | `2`: Origami with Stream-K |
| `TENSILE_STREAMK_DYNAMIC_GRID` | Controls Stream-K dynamic grid size selection. | `6` | `0`: Disable dynamic grid (use all CUs) | `1`: Reduce CUs only for small problems | `2`: Also reduce CUs for large sizes | `3`: Analytically predict best grid size | `4`: Behave like data parallel | `5`: Use Origami `select_best_grid_size` | `6`: Default (auto-pick optimal count) |
| `TENSILE_STREAMK_FIXED_GRID` | Overrides grid size with a fixed number of workgroups. | Not set | Integer (e.g. `64`) |
| `TENSILE_STREAMK_MAX_CUS` | Maximum number of compute units for Stream-K kernels. | All available CUs | Integer (e.g. `32`) |
| `TENSILE_STREAMK_GRID_MULTIPLIER` | Number of workgroups created per CU. Priority order: `FIXED_GRID` > `DYNAMIC_GRID` > `MAX_CUS` > `GRID_MULTIPLIER`. | Not set (default 1) | Float (e.g. `2.0`) |

### Type overrides

| Variable | Description | Default | Valid values |
|----------|-------------|---------|--------------|
| `HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32` | Overrides the compute type for GEMMs that specify `XF32`. | `-1` (off) | `-1`: Off | `0`: F32 | `1`: XF32 (e.g. TF32) | `2`: F32_BF16 |

### Internal / undocumented (found in source)

These variables are found in the source code but are not part of the official documentation. They may change without notice.

| Variable | Description | Source |
|----------|-------------|--------|
| `HIPBLASLT_TENSILE_LIBPATH` | Override the search path for TensileLite device library files (`.dat`, `.hsaco`/`.co`). | `tensile_host.cpp` |
| `HIPBLASLT_PRELOAD_KERNELS` | When set to `1`, preloads all kernel code objects at library initialization instead of lazy-loading on first use. | `Debug.cpp` |
| `HIPBLASLT_BENCH_PRINT_COMMAND` | When set to `1`, prints the equivalent `hipblaslt-bench` command for each GEMM invocation. | `Debug.cpp` |

---

## 3. Debugging Tools [Deep Dive]

### Logging with HIPBLASLT_LOG_LEVEL

The logging system uses cumulative levels. Setting a higher level automatically enables all lower levels. The implementation in `logging.h` processes `HIPBLASLT_LOG_LEVEL` via a fall-through switch statement:

```
Level 0: Off         (no logging)
Level 1: Error       (only errors)
Level 2: Trace       (errors + API calls with kernel launch parameters)
Level 3: Hints       (errors + trace + performance improvement suggestions)
Level 4: Info        (errors + trace + hints + general execution information)
Level 5: API trace   (all of the above + detailed API call parameters)
```

The corresponding bit mask values (for `HIPBLASLT_LOG_MASK`) from `rocblaslt-types.h`:

```
0:   None
1:   Error
2:   Trace
4:   Hints
8:   Info
16:  API trace
32:  Bench
64:  Profile
128: Extended profile
```

`HIPBLASLT_LOG_MASK` allows selective enabling of specific categories. For example, `HIPBLASLT_LOG_MASK=33` enables Error (1) + Bench (32).

If both `HIPBLASLT_LOG_LEVEL` and `HIPBLASLT_LOG_MASK` are set, `HIPBLASLT_LOG_LEVEL` takes precedence (the code checks it first).

**Example: Capture trace logs to a file:**

```bash
HIPBLASLT_LOG_LEVEL=2 HIPBLASLT_LOG_FILE=trace_%i.log ./my_app
```

### Solution override with HIPBLASLT_TUNING_OVERRIDE_FILE

The tuning override system lets you force specific kernel solutions for specific problem shapes. This is useful for debugging performance regressions or testing specific solutions.

**Workflow:**

1. **Record tuning results** with `hipblaslt-bench`:

   ```bash
   export HIPBLASLT_TUNING_FILE=tuning.txt
   ./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r
   ```

2. **Apply overrides** at runtime:

   ```bash
   export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt
   ./my_app
   ```

**File format** (CSV with header, from `UserDrivenTuningParser.cpp`):

```
transA,transB,batch_count,m,n,k,a_type,b_type,c_type,compute_type,solution_index
N,N,1,4096,4096,4096,f16_r,f16_r,f16_r,f32_r,42
```

Each pair of lines (header + values) defines one override entry. The parser reads the file once and caches the mappings in a thread-safe multimap keyed by the problem description (transpose, dimensions, data types, batch count).

### Profiling with rocprof

Use `rocprof` to profile kernel execution:

```bash
# Basic kernel timing
rocprof --stats ./hipblaslt-bench -m 4096 -n 4096 -k 4096 --precision f16_r

# Detailed trace with timestamps
rocprof -d trace_out --hip-trace --hsa-trace ./my_app

# With hipBLASLt markers (requires build with -DHIPBLASLT_ENABLE_MARKER=ON)
HIPBLASLT_ENABLE_MARKER=1 rocprof --hip-trace ./my_app
```

The marker integration (controlled by `HIPBLASLT_ENABLE_MARKER`) annotates ROCProfiler traces with hipBLASLt API boundaries, making it easier to correlate kernel launches with API calls in the trace output. Setting the variable to `2` redirects log output as markers instead of printing to the log stream.

---

## 4. Glossary [Essentials]

| Term | Definition |
|------|------------|
| **GEMM** | General Matrix-Matrix Multiplication. The core operation: `D = alpha * op(A) * op(B) + beta * C`, potentially with fused operations (activation, bias). |
| **Solution** | A specific kernel implementation for a GEMM problem, characterized by tile sizes, loop unrolling, vector widths, and other parameters. Multiple solutions may exist for the same problem; the library selects the best one. |
| **Logic file** | A YAML file that maps GEMM problem descriptions (data types, sizes, transpose modes) to solution indices. Located in `library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/` and organized by GPU architecture. |
| **Code object** | A compiled GPU binary (`.hsaco` or `.co` file) containing one or more kernel functions. Loaded at runtime by the TensileLite host library. |
| **Matmul descriptor** | A handle (`hipblasLtMatmulDesc_t`) that describes a GEMM operation's configuration: compute type, transpose modes, epilogue (activation, bias), and pointers to auxiliary data. |
| **Workspace** | A temporary GPU memory buffer required by some solutions for intermediate results. The required size varies per solution and must be queried and allocated by the caller before invoking `hipblasltMatmul`. |
| **Fused operation** | An operation combined with GEMM in a single kernel launch, avoiding extra memory round-trips. Supported fusions include activation functions (GELU, ReLU, Swish), bias addition, and scaling. Configured via the epilogue in the matmul descriptor. |
| **Grouped GEMM** | A batch of independent GEMM problems with potentially different sizes, data types, or parameters, dispatched together in a single API call for reduced launch overhead. See `hipblaslt_ext::groupedGemm`. |
| **ExtOp** | Extension operations beyond GEMM: layernorm, softmax, amax. Implemented as precompiled device kernels in `device-library/extops/`. |
| **Split-K** | A parallelization strategy that splits the K (reduction) dimension across multiple workgroups. Each workgroup computes a partial result; a final reduction step combines them. Trades extra workspace memory for better GPU utilization on problems with large K and small M/N. |
| **Workgroup mapping** | The scheme that assigns output tiles to GPU workgroups. Affects cache locality and occupancy. Common strategies include row-major, column-major, and space-filling curve mappings. |
| **Stream-K** | A work-distribution strategy where workgroups process a continuous stream of output tiles rather than a fixed assignment. Improves load balancing for irregular problem shapes. Controlled via `TENSILE_SOLUTION_SELECTION_METHOD` and related env vars. On MI350, Stream-K is always enabled. |

---

## 5. Further Reading [Essentials]

- **hipBLASLt documentation** (Sphinx): [https://rocm.docs.amd.com/projects/hipBLASLt/](https://rocm.docs.amd.com/projects/hipBLASLt/)
- **API reference**: [https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/api-reference.html](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/api-reference.html)
- **Environment variables reference** (Sphinx): [https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/env-variables.html](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/env-variables.html)
- **Offline tuning guide**: [https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-hipblaslt-offline-tuning.html](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-hipblaslt-offline-tuning.html)
- **Stream-K guide**: [https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-streamk.html](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-streamk.html)
- **Contributing guide**: [`../CONTRIBUTING.md`](../CONTRIBUTING.md)
- **ROCm documentation**: [https://rocm.docs.amd.com/](https://rocm.docs.amd.com/)
