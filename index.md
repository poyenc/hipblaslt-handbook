# hipBLASLt Developer Handbook

This handbook is a comprehensive onboarding guide for developers working on hipBLASLt, the AMD ROCm library for general matrix-matrix operations (GEMM) on AMD GPUs. It targets two audiences: junior developers with limited HIP/CUDA experience who need to learn the library from scratch, and senior kernel developers experienced with HIP who are new to hipBLASLt and need to contribute daily.

## How to Use This Handbook

Each chapter uses depth markers to help you focus on what matters for your experience level:

- **[Essentials]** — Core content everyone should read. Provides the foundational understanding needed to work with hipBLASLt.
- **[Deep Dive]** — Detailed content for experienced developers or second-pass reading. Covers internals, advanced workflows, and implementation details.

Pick your reading path below, then follow the chapters in order.

## Junior Developer Path (1-2 days)

If you're new to HIP/CUDA or BLAS libraries, follow this path. Read `[Essentials]` sections in each chapter; skip `[Deep Dive]` on your first pass.

1. [What is hipBLASLt](01-what-is-hipblaslt.md) — understand GEMM and the library's purpose
2. [Environment Setup](02-environment-setup.md) — get a working build
3. [Architecture](03-architecture.md) — read `[Essentials]` sections only for a high-level map
4. [Your First GEMM](04-your-first-gemm.md) — hands-on tutorial with a real sample
5. [API Guide](05-api-guide.md) — learn the API patterns and data types
6. [TensileLite Guide](06-tensilelite-guide.md) — read the overview sections only
7. [Testing and Benchmarking](09-testing-and-benchmarking.md) — run tests and benchmarks
8. [Troubleshooting and Reference](10-troubleshooting-and-reference.md) — error fixes, env vars, glossary

## Senior Kernel Developer Path (half a day)

If you're experienced with HIP kernel development but new to hipBLASLt, follow this path. Read all sections including `[Deep Dive]`.

1. [What is hipBLASLt](01-what-is-hipblaslt.md) — skim for feature overview
2. [Environment Setup](02-environment-setup.md) — get a working build with TensileLite dev setup
3. [Architecture](03-architecture.md) — read all sections for the full stack picture
4. [TensileLite Guide](06-tensilelite-guide.md) — kernel pipeline, logic files, code generation
5. [Host Library Guide](07-host-library-guide.md) — request lifecycle and internals
6. [Daily Workflows](08-daily-workflows.md) — task-oriented recipes for common work
7. [Testing and Benchmarking](09-testing-and-benchmarking.md) — test infrastructure and benchmarking
8. [Troubleshooting and Reference](10-troubleshooting-and-reference.md) — debugging tools, env vars, glossary

## External Resources

- [hipBLASLt Documentation](https://rocm.docs.amd.com/projects/hipBLASLt/)
- [API Reference](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/api-reference.html)
- [Contributing Guide](../CONTRIBUTING.md)
