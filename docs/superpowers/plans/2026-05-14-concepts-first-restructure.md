# Concepts-First Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure Chapters 3, 6, and 7 so core concepts are defined before use, diagrams and tables lead explanations, and source-level detail lives in a reference appendix.

**Architecture:** Each chapter gets a concepts-first rewrite following the spec. Work proceeds sequentially (Ch 3 → Ch 6 → Ch 7 → Ch 11 appendix) because later chapters reference earlier ones. After each chapter is written, a quality gate runs two Opus subagents to verify accuracy and readability before proceeding.

**Tech Stack:** Markdown documentation. Source verification against `/home/poyechen/workspace/repo/rocm-libraries/projects/hipblaslt/`.

**Diagram conventions used throughout:**
- **Sequence diagrams** — actors across top, message arrows between them (component interactions)
- **Pipeline diagrams** — boxes = data/artifacts (nouns), arrow labels = actions (verbs) (data transformations)

---

### Task 1: Back up current chapters

**Files:**
- Copy: `03-architecture.md` → `03-architecture.md.bak`
- Copy: `06-tensilelite-guide.md` → `06-tensilelite-guide.md.bak`
- Copy: `07-host-library-guide.md` → `07-host-library-guide.md.bak`

- [ ] **Step 1: Create backups of all three chapters**

```bash
cp 03-architecture.md 03-architecture.md.bak
cp 06-tensilelite-guide.md 06-tensilelite-guide.md.bak
cp 07-host-library-guide.md 07-host-library-guide.md.bak
```

- [ ] **Step 2: Commit backups**

```bash
git add 03-architecture.md.bak 06-tensilelite-guide.md.bak 07-host-library-guide.md.bak
git commit -m "docs: back up chapters 3, 6, 7 before restructure"
```

---

### Task 2: Write Chapter 3 — sections 1-2

Rewrite the first half of Chapter 3: the concepts glossary and the request lifecycle.

**Files:**
- Modify: `03-architecture.md`

**Source files to verify against:**
- `rocm-libraries/projects/hipblaslt/library/src/amd_detail/hipblaslt.cpp` — public API entry
- `rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/rocblaslt_mat.cpp` — rocblaslt dispatch
- `rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/tensile_host.cpp` — TensileLite dispatch
- `rocm-libraries/projects/hipblaslt/tensilelite/include/Tensile/SolutionLibrary.hpp` — library base class
- `rocm-libraries/projects/hipblaslt/tensilelite/include/Tensile/ExactLogicLibrary.hpp` — selection logic

- [ ] **Step 1: Write S1 — Core Concepts [Essentials]**

Replace the entire file starting from the title. Write:

1. Chapter title and intro paragraph (2-3 sentences: what this chapter covers, pointer to Ch 1 for GEMM basics).
2. Glossary table with 6 entries: Problem, Contraction Problem, Solution, Logic File, Code Object, Library. Each entry gets a one-line definition.
3. Relationship diagram (pipeline style) showing:
   - Problem → has many → Solution
   - Problem + dimensions/ptrs → Contraction Problem
   - Contraction Problem → runtime selection → Kernel launch
4. One paragraph explaining the relationship in prose (backup for the diagram, not a replacement).

Key content to include in glossary definitions:

| Concept             | Definition                                                                |
|---------------------|---------------------------------------------------------------------------|
| Problem             | Abstract GEMM configuration: data types, transpose modes, features        |
|                     | (bias, activation, scaling). Does NOT include dimensions or pointers.     |
| Contraction Problem | A Problem plus concrete runtime parameters: M, N, K dimensions,          |
|                     | leading dimensions, strides, batch count, and data pointers.              |
| Solution            | A kernel implementation that can execute a Problem. Defined by tuning     |
|                     | parameters: tile size, unroll depth, prefetch strategy, matrix            |
|                     | instruction. One Problem type can have many Solutions.                     |
| Logic File          | A YAML file mapping a Problem type to its Solutions. Contains the         |
|                     | problem descriptor, solution definitions, and a size-to-solution          |
|                     | mapping table. Organized by GPU architecture.                             |
| Code Object         | A compiled `.co` (ELF) file containing one or more GPU kernels for a     |
|                     | specific ISA target (e.g., gfx950).                                       |
| Library             | A tree of selection nodes built from logic files at build time.           |
|                     | At runtime, the tree narrows from hardware → problem type → strategy     |
|                     | → individual solution.                                                    |

- [ ] **Step 2: Write S2 — How a GEMM Call Becomes a Kernel [Essentials]**

Write:

1. One-sentence intro: "Every GEMM call passes through four layers before a GPU kernel executes."
2. Sequence diagram:

```
App          hipblaslt.cpp    rocblaslt_mat.cpp   tensile_host.cpp    GPU
 |                |                 |                   |               |
 | hipblasLtMatmul()               |                   |               |
 |───────────────>|                |                   |               |
 |                | rocblaslt_matmul()                 |               |
 |                |────────────────>|                   |               |
 |                |                 | runContractionProblem()           |
 |                |                 |──────────────────>|               |
 |                |                 |                   | launchKernels |
 |                |                 |                   |──────────────>|
```

3. Four short paragraphs (2-3 sentences each), one per layer:
   - **Public API** — casts hipblasLt handles to rocblaslt types, delegates
   - **rocblaslt backend** — validates arguments, builds a `RocblasltContractionProblem`
   - **TensileLite dispatch** — translates to `ContractionProblemGemm`, queries the library tree for the best solution
   - **Kernel launch** — loads the code object (lazy or eager) and submits via HIP

Verify function names against source files listed above.

- [ ] **Step 3: Save progress (do not commit yet — chapter is incomplete)**

---

### Task 3: Write Chapter 3 — sections 3-6

Complete Chapter 3 with the dispatch paths, selection logic, API surfaces, and directory map.

**Files:**
- Modify: `03-architecture.md`

**Source files to verify against:**
- `rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/tensile_host.cpp` — `useRocRoller()` function
- `rocm-libraries/projects/hipblaslt/library/include/hipblaslt/hipblaslt.h` — C API
- `rocm-libraries/projects/hipblaslt/library/include/hipblaslt/hipblaslt-ext.hpp` — C++ ext API
- `rocm-libraries/projects/hipblaslt/library/include/hipblaslt/hipblaslt-ext-op.h` — ExtOps API

- [ ] **Step 1: Write S3 — The Two Dispatch Paths [Essentials]**

Write:

1. One-sentence intro: "hipBLASLt has two backends for kernel dispatch."
2. Comparison table:

| Aspect        | TensileLite              | RocRoller                |
|---------------|--------------------------|--------------------------|
| Kernels       | Precompiled `.co` files  | JIT-compiled at runtime  |
| Default usage | All standard GEMM        | Block-scaled GEMM        |
| Selection     | Logic file lookup        | Origami performance model|
| First-call    | Loads code object on use | JIT compiles kernel      |
| Caching       | Loaded once, reused      | Cached after first JIT   |

3. One paragraph explaining when each path is used: TensileLite is the default; RocRoller activates automatically for block-scaling problems or when forced via handle flag. Verify the `useRocRoller()` conditions against `tensile_host.cpp`.

- [ ] **Step 2: Write S4 — How Solutions Are Selected [Essentials]**

Write:

1. Pipeline diagram: logic files → .dat bundles → library tree (approved format from spec).
2. Tree diagram showing selection priority:

```
Library tree
└── Hardware layer (which GPU?)
    ├── gfx950_id75a3 (exact chip ID)  ← preferred
    └── gfx950 (generic architecture)  ← fallback if no exact match
        └── Problem type (data types, transpose, features)
            ├── Equality    ← checked first (exact dimension match)
            ├── GridBased   ← checked next (heuristic)
            └── FreeSize    ← checked last
```

3. API comparison table:

| User-facing API       | Internal method        | Returns               |
|-----------------------|------------------------|-----------------------|
| `algoGetHeuristic()`  | `findTopSolutions()`   | Ranked top N solutions|
| `getAllAlgos()`        | `findAllSolutions()`   | Every compatible solution|
| (no algo at dispatch) | `findBestSolution()`   | Single best solution  |

4. Two paragraphs explaining the priority cascade and why users usually don't need to tune manually.

- [ ] **Step 3: Write S5 — The Three API Surfaces [Essentials]**

Write a comparison table (trimmed from current S2):

| Surface               | Header              | Style         | Recommended for        |
|-----------------------|---------------------|---------------|------------------------|
| C API                 | `hipblaslt.h`       | Opaque handles| cuBLASLt portability   |
| C++ Extension API     | `hipblaslt-ext.hpp` | Classes        | New ROCm-native code   |
| Extension Operations  | `hipblaslt-ext-op.h`| C functions    | Non-GEMM ops (softmax, |
|                       |                     |               | layernorm, amax)       |

One sentence per surface explaining when to use it. Link to Ch 5 for full API details.

- [ ] **Step 4: Write S6 — Key Directory Map [Essentials]**

Copy the directory table from current Chapter 3 S10 as-is. No changes needed.

- [ ] **Step 5: Commit Chapter 3**

```bash
git add 03-architecture.md
git commit -m "docs: restructure Chapter 3 with concepts-first approach"
```

---

### Task 4: Quality gate — verify Chapter 3

Spawn two Opus subagents in parallel to verify Chapter 3.

**Files:**
- Read: `03-architecture.md` (the just-written chapter)
- Read: source files under `rocm-libraries/projects/hipblaslt/`

- [ ] **Step 1: Spawn Verifier agent**

Spawn an Opus agent (model: opus) with this prompt:

> You are a technical verifier for documentation. Read the file `/home/poyechen/workspace/repo/hipblaslt-handbook/03-architecture.md` and cross-check every technical claim against the hipBLASLt source code at `/home/poyechen/workspace/repo/rocm-libraries/projects/hipblaslt/`.
>
> Check:
> - Function names mentioned exist and do what the chapter says
> - The sequence diagram call chain matches actual code flow
> - The dispatch path conditions (when RocRoller vs TensileLite) are accurate
> - The selection priority (Equality before GridBased, exact chip before fallback) matches `ExactLogicLibrary.hpp`
> - The glossary definitions are technically correct
> - The API surface descriptions match the actual headers
>
> Key source files:
> - `library/src/amd_detail/hipblaslt.cpp`
> - `library/src/amd_detail/rocblaslt/src/rocblaslt_mat.cpp`
> - `library/src/amd_detail/rocblaslt/src/tensile_host.cpp`
> - `tensilelite/include/Tensile/ExactLogicLibrary.hpp`
> - `tensilelite/include/Tensile/MasterSolutionLibrary.hpp`
> - `library/include/hipblaslt/hipblaslt.h`
> - `library/include/hipblaslt/hipblaslt-ext.hpp`
>
> Produce a numbered findings list. For each finding: quote the chapter text, state what the source code actually says, and classify as ERROR (factually wrong), OUTDATED (was correct but source changed), or MISLEADING (technically true but likely to confuse).

- [ ] **Step 2: Spawn Reviewer agent (in parallel with Step 1)**

Spawn an Opus agent (model: opus) with this prompt:

> You are a readability reviewer for developer documentation. Read the file `/home/poyechen/workspace/repo/hipblaslt-handbook/03-architecture.md`.
>
> You are reviewing from the perspective of two audiences:
> 1. A new team member joining the hipBLASLt team with no prior hipBLASLt experience
> 2. An adjacent-team engineer (e.g., from PyTorch or ROCm) who needs a working mental model
>
> Check:
> - Is every term defined before it is used? Flag any term that appears before its definition.
> - Can each diagram be understood without reading surrounding prose? If not, what's missing?
> - Are the tables self-explanatory? Do column headers make sense?
> - Is there any place where a reader would need to stop and ask "what does this mean?"
> - Does the [Essentials] content give a complete mental model without needing [Deep Dive]?
> - Are cross-references to other chapters clear (chapter number + section name)?
>
> Produce a numbered findings list. For each finding: quote the chapter text, state what confused you, and suggest a fix.

- [ ] **Step 3: Exchange findings between agents**

Send the Verifier's findings to the Reviewer and vice versa. Ask each to respond:
- Verifier reviews Reviewer's readability flags and confirms or rebuts with source context.
- Reviewer reviews Verifier's accuracy flags and confirms or rebuts with audience context (e.g., "this simplification is intentional for Essentials readers").

- [ ] **Step 4: Collect joint report and apply fixes**

Gather the agreed-upon fixes from both agents. Apply them to `03-architecture.md`.

- [ ] **Step 5: Commit fixes**

```bash
git add 03-architecture.md
git commit -m "docs: apply quality gate fixes to Chapter 3"
```

---

### Task 5: Write Chapter 6 — sections 1-2

Rewrite the concept-setup sections of Chapter 6.

**Files:**
- Modify: `06-tensilelite-guide.md`

**Source files to verify against:**
- `rocm-libraries/projects/hipblaslt/tensilelite/rocisa/` — rocisa module
- `rocm-libraries/projects/hipblaslt/tensilelite/Tensile/CustomKernels/` — custom kernels
- `rocm-libraries/projects/hipblaslt/tensilelite/Tensile/SolutionLibrary.py` — selection strategies

- [ ] **Step 1: Rewrite S1 — What TensileLite Does [Essentials]**

Keep the existing pipeline diagram from current S1. Trim surrounding prose to essentials. Add one line at the top:

> For terminology used in this chapter (Problem, Solution, Logic File, etc.), see [Chapter 3: Core Concepts](03-architecture.md#1-core-concepts-essentials).

- [ ] **Step 2: Rewrite S2 — TensileLite-Specific Concepts [Essentials]**

Replace current S2 "Key Concepts" entirely. Write:

1. Intro: "This section defines concepts specific to TensileLite that build on the [core glossary in Chapter 3](03-architecture.md#1-core-concepts-essentials)."

2. Concepts table (only terms NOT in Ch 3 glossary):

| Concept              | What it is                                                  | Where it lives      |
|----------------------|-------------------------------------------------------------|---------------------|
| Custom Kernel        | A hand-written assembly kernel (`.s` file) referenced by    | `CustomKernels/`    |
|                      | name in a logic file's solution entry                        |                     |
| rocisa               | Python/C++ ISA code generation module built with nanobind.  | `rocisa/`           |
|                      | Provides register/instruction primitives for kernel writers  |                     |
| Selection Strategy   | How a logic file's size-to-solution mapping works.           | Logic file          |
|                      | Options: Equality (exact match), GridBased (heuristic),      | element 11          |
|                      | Range (range-based), Origami (analytical model)              |                     |
| Code Generation      | The offline Python pipeline that turns problem descriptions  | `Tensile/`          |
| Pipeline             | into assembly source → compiled code objects                 |                     |

3. Pipeline diagram showing the full lifecycle:

```
┌────────────────────────┐
│ Problem description    │
│ (YAML)                 │
└───────────┬────────────┘
            │ Tensile Python toolchain generates
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

4. Absorb rocisa content from current Chapter 3 S6: what rocisa is, how to build it (`invoke rocisa`), the stale-import error hint. Keep it concise (one paragraph + the build command).

- [ ] **Step 3: Save progress (do not commit yet)**

---

### Task 6: Write Chapter 6 — sections 3-7

Complete Chapter 6 with directory layout, logic file anatomy, code generation, build/test, and adding kernels.

**Files:**
- Modify: `06-tensilelite-guide.md`

- [ ] **Step 1: Keep S3 — Directory Layout [Essentials]**

Copy current S3 as-is. No changes.

- [ ] **Step 2: Rewrite S4 — Logic File Anatomy [Deep Dive]**

Add the element map table BEFORE the existing YAML walkthrough:

| Element | Contents                  | Key fields                                |
|---------|---------------------------|-------------------------------------------|
| 0       | Version header            | `MinimumRequiredVersion`                  |
| 1       | Scheduling model          | Architecture name (e.g., `gfx950`)        |
| 2       | Architecture              | Architecture name                         |
| 3       | Device ID filter          | `[Device 75a0]`                           |
| 4       | Problem type description  | Data types, transpose, features           |
| 5       | Solution list             | Kernel tuning parameters                  |
| 6       | Index mapping             | Tensor index roles                        |
| 7       | Size-to-solution mapping  | Maps dimensions to solution indices       |
| 8-9     | Reserved                  | `null`                                    |
| 10      | Performance metric        | `DeviceEfficiency`                        |
| 11      | Selection strategy        | `GridBased`, `Equality`, `Range`, etc.    |

Then keep the existing detailed walkthrough that follows.

- [ ] **Step 3: Rewrite S5 — Code Generation Pipeline [Deep Dive]**

Keep existing content. Absorb Python toolchain content from current Chapter 3 S5:
- Key subsystems table (KernelWriter, Components/, Common/, etc.)
- Toolchain outputs list (.s files, .co files, logic files)

Place this absorbed content at the beginning of S5 as context before the existing stage-by-stage walkthrough.

- [ ] **Step 4: Keep S6 — Building and Testing [Deep Dive]**

Copy current S6 as-is. No changes.

- [ ] **Step 5: Keep S7 — How to Add a New Kernel Solution [Deep Dive]**

Copy current S7 as-is. No changes.

- [ ] **Step 6: Commit Chapter 6**

```bash
git add 06-tensilelite-guide.md
git commit -m "docs: restructure Chapter 6 with concepts-first approach"
```

---

### Task 7: Quality gate — verify Chapter 6

Same process as Task 4 but for Chapter 6.

**Files:**
- Read: `06-tensilelite-guide.md`
- Read: `03-architecture.md` (verify cross-references are correct)
- Read: source files under `rocm-libraries/projects/hipblaslt/tensilelite/`

- [ ] **Step 1: Spawn Verifier agent**

Spawn an Opus agent (model: opus) with this prompt:

> You are a technical verifier. Read `/home/poyechen/workspace/repo/hipblaslt-handbook/06-tensilelite-guide.md` and cross-check against the source at `/home/poyechen/workspace/repo/rocm-libraries/projects/hipblaslt/tensilelite/`.
>
> Check:
> - The TensileLite-specific concepts table is accurate (custom kernels, rocisa, selection strategies, code generation pipeline)
> - The lifecycle pipeline diagram matches the actual build flow
> - The logic file element map matches actual YAML structure (check a real logic file under `library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/`)
> - The directory layout tree matches the actual directory structure
> - rocisa build instructions (`invoke rocisa`) are current
> - The code generation pipeline stages match actual source modules
> - Cross-references to Chapter 3 glossary terms are correct (read `03-architecture.md` S1)
>
> Produce a numbered findings list: quote, actual source truth, classify as ERROR/OUTDATED/MISLEADING.

- [ ] **Step 2: Spawn Reviewer agent (in parallel)**

Spawn an Opus agent (model: opus) with this prompt:

> You are a readability reviewer. Read `/home/poyechen/workspace/repo/hipblaslt-handbook/06-tensilelite-guide.md`.
>
> Also read Chapter 3 (`03-architecture.md`) to verify that cross-references work — if Chapter 6 says "see Chapter 3 for X," confirm Chapter 3 actually defines X.
>
> Check:
> - Does S2 clearly distinguish TensileLite-specific terms from Ch 3 glossary terms?
> - Can the lifecycle pipeline diagram be understood without reading prose?
> - Is the logic file element map useful as a quick reference before the detailed walkthrough?
> - Does absorbed content (rocisa, Python toolchain) flow naturally or feel bolted-on?
> - Any undefined terms, unclear diagrams, or missing context?
>
> Produce a numbered findings list: quote, what confused you, suggested fix.

- [ ] **Step 3: Exchange findings, collect joint report**

Same as Task 4 Step 3.

- [ ] **Step 4: Apply fixes and commit**

```bash
git add 06-tensilelite-guide.md
git commit -m "docs: apply quality gate fixes to Chapter 6"
```

---

### Task 8: Write Chapter 7

Rewrite Chapter 7 with concepts-first approach.

**Files:**
- Modify: `07-host-library-guide.md`

**Source files to verify against:**
- `rocm-libraries/projects/hipblaslt/library/src/amd_detail/hipblaslt.cpp`
- `rocm-libraries/projects/hipblaslt/library/src/amd_detail/hipblaslt-ext.cpp`
- `rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/rocblaslt_mat.cpp`
- `rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/tensile_host.cpp`

- [ ] **Step 1: Write S1 — What the Host Library Does [Essentials]**

Write 2-3 sentences: the host library is the C++ code that sits between the public API and the GPU. Its job is to translate API calls into contraction problems, select the best kernel solution from the library tree, and launch the kernel on a HIP stream. Pointer to Ch 3 glossary.

- [ ] **Step 2: Write S2 — Request Lifecycle [Essentials]**

Write two sequence diagrams.

C API path:

```
App          hipblaslt.cpp    rocblaslt_mat.cpp   tensile_host.cpp    GPU
 |                |                 |                   |               |
 | hipblasLtMatmul()               |                   |               |
 |───────────────>|                |                   |               |
 |                | rocblaslt_matmul()                 |               |
 |                |────────────────>|                   |               |
 |                |                 | runContractionProblem()           |
 |                |                 |──────────────────>|               |
 |                |                 |                   | launchKernels |
 |                |                 |                   |──────────────>|
```

C++ ext API path:

```
App          Gemm              rocblaslt_mat.cpp   tensile_host.cpp    GPU
 |            |                      |                   |               |
 | initialize()                     |                   |               |
 |───────────>|                      |                   |               |
 |            | rocblaslt_makeArgument_cpp()             |               |
 |            |─────────────────────>|                   |               |
 | run()      |                      |                   |               |
 |───────────>|                      |                   |               |
 |            | rocblaslt_run_cpp()   |                   |               |
 |            |─────────────────────>|                   |               |
 |            |                      | runKernelFromInvocation()         |
 |            |                      |──────────────────>|               |
 |            |                      |                   | launchKernels |
 |            |                      |                   |──────────────>|
```

Below each diagram: one paragraph per layer explaining what it does and what it passes to the next. No line numbers.

Verify all function names against source files.

- [ ] **Step 3: Write S3 — Algorithm Selection [Essentials]**

Write:

1. Reference to Ch 3 S4 API table: "The three user-facing entry points for algorithm selection are described in [Chapter 3: How Solutions Are Selected](03-architecture.md#4-how-solutions-are-selected-essentials)."

2. Explain the priority cascade in 2 paragraphs:
   - Hardware level: exact chip-ID match is preferred; generic architecture is fallback. The `isFallbackMatch()` mechanism in the library tree ensures chip-ID-specific tuning wins when available.
   - Strategy level: Equality (exact dimension lookup) is checked before GridBased (heuristic). If Equality has an entry for your exact M, N, K, that solution wins. Otherwise GridBased provides a heuristic-ranked solution.

3. One paragraph on `HIPBLASLT_TUNING_OVERRIDE_FILE`: user-driven tuning override that injects a specific solution index ahead of heuristic results.

- [ ] **Step 4: Write S4 — Lazy Loading [Essentials]**

Write:

1. One-sentence intro: "Lazy loading defers code object loading from startup to first use, reducing memory and startup time."

2. Comparison table:

| Aspect       | Lazy ON (default)         | Lazy OFF                  |
|--------------|---------------------------|---------------------------|
| Startup      | Loads metadata only       | Loads all `.co` files     |
| First call   | Loads `.co` on first use  | No extra latency          |
| Memory       | Grows as kernels are used | Peak memory at startup    |
| Error timing | Missing `.co` at dispatch | Missing `.co` at init     |
| Library file | `TensileLibrary_lazy_<arch>.dat` | `TensileLibrary_<arch>.dat` |

3. One paragraph on debugging implications: first-call latency is normal with lazy loading, not a performance bug. Use `HIPBLASLT_TENSILE_LIBPATH` to verify library path.

- [ ] **Step 5: Keep S5 — Adding a New API Feature [Deep Dive]**

Copy current S3 "Adding a New API Feature" checklist. Remove all line numbers. Keep file names as navigation pointers. The checklist structure (Step 1 through Step 8) stays intact.

- [ ] **Step 6: Commit Chapter 7**

```bash
git add 07-host-library-guide.md
git commit -m "docs: restructure Chapter 7 with concepts-first approach"
```

---

### Task 9: Quality gate — verify Chapter 7

Same process as Tasks 4 and 7.

**Files:**
- Read: `07-host-library-guide.md`
- Read: `03-architecture.md` (verify cross-references)
- Read: source files under `rocm-libraries/projects/hipblaslt/`

- [ ] **Step 1: Spawn Verifier agent**

Spawn an Opus agent (model: opus) with this prompt:

> You are a technical verifier. Read `/home/poyechen/workspace/repo/hipblaslt-handbook/07-host-library-guide.md` and cross-check against the source at `/home/poyechen/workspace/repo/rocm-libraries/projects/hipblaslt/`.
>
> Check:
> - Both sequence diagrams (C API and ext API) have correct function names and call order
> - The ext API path correctly shows `rocblaslt_makeArgument_cpp()` and `rocblaslt_run_cpp()` and `runKernelFromInvocation()`
> - The algorithm selection priority description matches `ExactLogicLibrary::findBestSolution()` logic
> - `HIPBLASLT_TUNING_OVERRIDE_FILE` description is accurate
> - Lazy loading comparison table matches the `#if ROCBLASLT_TENSILE_LAZY_LOAD` branches in `tensile_host.cpp`
> - The "Adding a New API Feature" checklist file names are current
> - Cross-references to Chapter 3 are valid
>
> Key source files:
> - `library/src/amd_detail/hipblaslt.cpp`
> - `library/src/amd_detail/hipblaslt-ext.cpp`
> - `library/src/amd_detail/rocblaslt/src/rocblaslt_mat.cpp`
> - `library/src/amd_detail/rocblaslt/src/tensile_host.cpp`
> - `tensilelite/include/Tensile/ExactLogicLibrary.hpp`
>
> Produce a numbered findings list: quote, actual source truth, classify as ERROR/OUTDATED/MISLEADING.

- [ ] **Step 2: Spawn Reviewer agent (in parallel)**

Spawn an Opus agent (model: opus) with this prompt:

> You are a readability reviewer. Read `/home/poyechen/workspace/repo/hipblaslt-handbook/07-host-library-guide.md`.
>
> Also read Chapter 3 (`03-architecture.md`) and Chapter 6 (`06-tensilelite-guide.md`) to verify cross-references.
>
> Check:
> - Does S1 orient the reader before the sequence diagrams?
> - Can both sequence diagrams be understood independently?
> - Is the difference between C API and ext API paths clear?
> - Does the algorithm selection section work without reading the appendix?
> - Is the lazy loading table self-explanatory?
> - Any undefined terms, unclear diagrams, or missing context?
>
> Produce a numbered findings list: quote, what confused you, suggested fix.

- [ ] **Step 3: Exchange findings, collect joint report**

Same as Task 4 Step 3.

- [ ] **Step 4: Apply fixes and commit**

```bash
git add 07-host-library-guide.md
git commit -m "docs: apply quality gate fixes to Chapter 7"
```

---

### Task 10: Write Chapter 11 — Reference Appendix

Create the new reference appendix collecting all source-level detail removed from Chapters 3, 6, and 7.

**Files:**
- Create: `11-reference-appendix.md`
- Modify: `index.md` (add Chapter 11 to both reading paths)

- [ ] **Step 1: Create 11-reference-appendix.md**

Write:

1. Title and intro: "This appendix collects source-level implementation detail for developers actively working in the hipBLASLt codebase. For conceptual understanding, see Chapters 3, 6, and 7."

2. Organize content from current chapters into sections:

| Section | Source | Content |
|---------|--------|---------|
| Host Library File Reference | Current Ch 3 S3 | File-by-file role tables for `hipblaslt.cpp`, `rocblaslt_mat.cpp`, etc. |
| Request Lifecycle — Detailed Call Chains | Current Ch 7 S1 | Full call chains with line numbers |
| Algorithm Selection Internals | Current Ch 7 S2 | `getBestSolutions()` walkthrough, `_convertToHeuristicResultArray()`, xf32 fallback |
| Library Class Hierarchy | New content | `SolutionLibrary` → `ExactLogicLibrary` → `HardwareSelectionLibrary` / `ProblemSelectionLibrary` tree, with the `findBestSolution()` priority logic from `ExactLogicLibrary.hpp` |
| RocRoller Dispatch Details | Current Ch 3 S9 + Ch 7 S4 | `runRocRollerContractionProblem()` flow, Origami integration, tile size arrays, `SolutionCache`, key source files table |
| Device Libraries | Current Ch 3 S8 | ExtOps, matrix transform, CMake options |
| Lazy Loading Internals | Current Ch 7 S5 | `TensileHost::initialize()` differences, CMake configuration, `ROCBLASLT_TENSILE_LAZY_LOAD` branches |

Copy each section's content from the backup files (`*.md.bak`), preserving line numbers, function signatures, and code blocks as-is.

- [ ] **Step 2: Update index.md**

Add Chapter 11 to both reading paths in `index.md`:

In the Junior Developer Path, add after item 8:
```
9. [Reference Appendix](11-reference-appendix.md) — source-level detail (optional, for deep dives)
```

In the Senior Kernel Developer Path, add after item 8:
```
9. [Reference Appendix](11-reference-appendix.md) — detailed call chains, class hierarchies, internals
```

- [ ] **Step 3: Commit**

```bash
git add 11-reference-appendix.md index.md
git commit -m "docs: add reference appendix (Chapter 11) with source-level detail"
```

---

### Task 11: Quality gate — verify Chapter 11

Same process but focused on accuracy of preserved source-level detail.

**Files:**
- Read: `11-reference-appendix.md`
- Read: source files under `rocm-libraries/projects/hipblaslt/`

- [ ] **Step 1: Spawn Verifier agent**

Spawn an Opus agent (model: opus) with this prompt:

> You are a technical verifier. Read `/home/poyechen/workspace/repo/hipblaslt-handbook/11-reference-appendix.md` and cross-check against the source at `/home/poyechen/workspace/repo/rocm-libraries/projects/hipblaslt/`.
>
> This file contains source-level detail (line numbers, function signatures, class hierarchies). Check:
> - Do the line numbers still match? (Line numbers shift as code evolves — flag any that are off by more than 20 lines)
> - Are function signatures current?
> - Does the library class hierarchy match the actual C++ headers in `tensilelite/include/Tensile/`?
> - Is the RocRoller dispatch flow current? Check `library/src/amd_detail/rocblaslt/src/rocroller/rocroller_host.cpp`
> - Are CMake option names and defaults current? Check the top-level `CMakeLists.txt`
>
> Produce a numbered findings list: quote, actual source truth, classify as ERROR/OUTDATED/MISLEADING.

- [ ] **Step 2: Spawn Reviewer agent (in parallel)**

Spawn an Opus agent (model: opus) with this prompt:

> You are a readability reviewer. Read `/home/poyechen/workspace/repo/hipblaslt-handbook/11-reference-appendix.md`.
>
> This is a reference appendix for developers working in the source code. Check:
> - Is the organization logical? Can a developer find what they need quickly?
> - Are section headers descriptive enough to use as navigation?
> - Do the sections reference back to the conceptual chapters (3, 6, 7) where relevant?
> - Is there any content that should be in the main chapters instead of the appendix?
> - Is there any content that's duplicated between appendix sections?
>
> Produce a numbered findings list: quote, what confused you, suggested fix.

- [ ] **Step 3: Exchange findings, collect joint report**

Same as Task 4 Step 3.

- [ ] **Step 4: Apply fixes and commit**

```bash
git add 11-reference-appendix.md
git commit -m "docs: apply quality gate fixes to Chapter 11"
```

---

### Task 12: Final cleanup

- [ ] **Step 1: Remove backup files**

```bash
git rm 03-architecture.md.bak 06-tensilelite-guide.md.bak 07-host-library-guide.md.bak
```

- [ ] **Step 2: Verify all cross-references**

Search all chapter files for broken cross-references:

```bash
grep -n '03-architecture.md\|06-tensilelite-guide.md\|07-host-library-guide.md\|11-reference-appendix.md' *.md
```

Verify each link target exists (section headers match the `#anchor` format).

- [ ] **Step 3: Commit cleanup**

```bash
git add -A
git commit -m "docs: remove backups and verify cross-references after restructure"
```
