# Concepts-First Restructure of Chapters 3, 6, and 7

## Problem

Chapters 3 (Architecture), 6 (TensileLite Guide), and 7 (Host Library Guide)
overwhelm readers with terminology before establishing what the terms mean.
Chapter 3 opens with a full stack diagram and file paths. Chapters 6 and 7
use terms like "solution," "logic file," and "MasterSolutionLibrary" assuming
the reader already knows them.

A reader unfamiliar with hipBLASLt has to stop and ask "what is this?" at
every paragraph.

## Goal

Restructure all three chapters so that:

1. Core concepts are defined before they are used.
2. Diagrams and tables are the primary way to introduce relationships.
3. Source-level detail (line numbers, function signatures) moves to a
   reference appendix, keeping chapters focused on understanding.

## Audience

Both new team members and adjacent-team engineers (PyTorch, ROCm). The
existing `[Essentials]` / `[Deep Dive]` layering serves both: Essentials
for working mental model, Deep Dive for contributors.

## Diagram Conventions

Two diagram types, each internally consistent, used across all three chapters:

### Sequence diagrams -- component interactions

Actors across the top, message arrows between them. Used for request
lifecycle, API call flows, and multi-component interactions.

```
Actor_A     Actor_B     Actor_C
  |             |           |
  | message()   |           |
  |------------>|           |
  |             | action()  |
  |             |---------->|
```

### Pipeline diagrams -- data transformations

Boxes = data/artifacts (nouns). Arrow labels = actions (verbs). Vertical
flow = sequence. Used for build pipelines, data transformation chains,
and concept relationships.

```
+-----------------+
| Input artifact  |
+--------+--------+
         | action that transforms
         v
+-----------------+
| Output artifact |
+-----------------+
```

## Chapter 3: Architecture

### Current problems

- Opens with a large ASCII stack diagram mixing file paths, function names,
  and components before defining any terms.
- 10 sections that jump between abstraction levels.
- Source-level detail (file tables, rocisa internals, device library build
  options) mixed with conceptual architecture.

### New structure

#### S1. Core Concepts [Essentials]

Glossary table defining the foundational terms all chapters reference:

| Concept              | Definition                                                                 |
|----------------------|----------------------------------------------------------------------------|
| Problem              | Abstract GEMM config: data types, transpose modes, features (bias, etc.)   |
| Contraction Problem  | A Problem + concrete runtime parameters: M, N, K, strides, data pointers   |
| Solution             | A kernel implementation with specific tuning (tile size, unroll, prefetch)  |
| Logic File           | YAML file mapping problem types to solutions with size-based selection     |
| Code Object          | Compiled .co (ELF) file containing GPU kernel(s) for a target ISA         |
| Library              | Tree of selection nodes that narrows hardware -> problem type -> solution   |

Relationship diagram (pipeline style) showing how concepts connect:

```
+----------+            +--------------+
| Problem  |--has many-->| Solution     |
| (abstract)|            | (kernel impl)|
+----------+            +--------------+
     |                        |
     | + dimensions/ptrs      | selected at runtime
     v                        v
+--------------+        +--------------+
| Contraction  |------->| Kernel launch|
| Problem      |        +--------------+
+--------------+
```

#### S2. How a GEMM Call Becomes a Kernel [Essentials]

Sequence diagram showing the 4-layer flow:

```
App       hipblaslt.cpp   rocblaslt     tensile_host    GPU
 |             |              |              |           |
 | hipblasLtMatmul()         |              |           |
 |------------>|              |              |           |
 |             | rocblaslt_matmul()          |           |
 |             |------------->|              |           |
 |             |              | runContractionProblem()  |
 |             |              |------------->|           |
 |             |              |              | launch    |
 |             |              |              |---------->|
```

Each layer gets 2-3 sentences explaining its role. No file paths or line
numbers.

#### S3. The Two Dispatch Paths [Essentials]

Comparison table:

| Aspect         | TensileLite        | RocRoller          |
|----------------|--------------------|--------------------|
| Kernels        | Precompiled .co    | JIT at runtime     |
| When used      | Default path       | Block scaling      |
| Selection      | Logic files        | Origami model      |
| First-call     | Loads .co on use   | JIT compiles       |

Replaces current S9 (RocRoller) at the conceptual level.

#### S4. How Solutions Are Selected [Essentials]

Pipeline diagram showing logic files becoming the runtime library:

```
+---------------------+
| Logic files (YAML)  |
+---------+-----------+
          | TensileCreateLibrary serializes
          v
+---------------------+
| .dat bundles         |
+---------+-----------+
          | MasterSolutionLibrary loads at runtime
          v
+---------------------+
| Library tree         |
+---------------------+
```

Tree structure showing selection priority:

```
Library tree
+-- Hardware layer (which GPU?)
|   +-- gfx950_id75a3 (exact chip) <-- preferred
|   +-- gfx950 generic             <-- fallback
|       +-- Problem type (types, features)
|           +-- Equality   <-- checked first
|           +-- GridBased  <-- checked next
|           +-- FreeSize   <-- checked last
```

API comparison table:

| User-facing API      | Internal call         | Returns    |
|----------------------|-----------------------|------------|
| algoGetHeuristic()   | findTopSolutions()    | ranked N   |
| getAllAlgos()         | findAllSolutions()    | all        |
| (no algo at dispatch)| findBestSolution()    | best 1     |

#### S5. The Three API Surfaces [Essentials]

Comparison table (C API vs C++ ext vs ExtOps). Trimmed from current S2.

#### S6. Key Directory Map [Essentials]

Kept as-is from current S10.

### Content removed from Chapter 3

| Current section                    | Destination               |
|------------------------------------|---------------------------|
| S3 Host library file tables        | Reference appendix        |
| S4 TensileLite runtime internals   | Chapter 7                 |
| S5 Python toolchain                | Chapter 6 S5              |
| S6 rocisa                          | Chapter 6 S2              |
| S8 Device libraries                | Reference appendix        |
| S9 RocRoller details               | Reference appendix        |

## Chapter 6: TensileLite Guide

### Current problems

- S2 "Key Concepts" defines terms already needed in Chapter 3 (like
  "Solution" and "Problem") alongside TensileLite-specific terms, mixing
  abstraction levels.
- No explanation of how logic files become the runtime library.
- Python toolchain and rocisa content lives in Chapter 3 instead of here.

### New structure

#### S1. What TensileLite Does [Essentials]

Kept and trimmed. Pipeline diagram stays. One-line pointer to Ch 3
glossary for terminology.

#### S2. TensileLite-Specific Concepts [Essentials]

Table of terms NOT already in Ch 3 glossary:

| Concept             | What it is                                         |
|---------------------|----------------------------------------------------|
| Custom Kernel       | Hand-written .s file in CustomKernels/             |
| rocisa              | Python/C++ ISA codegen module (nanobind)           |
| Selection Strategy  | How a logic file picks solutions for given          |
|                     | dimensions: Equality, GridBased, Range, Origami    |
| Code Generation     | Python pipeline turning problem descriptions into   |
| Pipeline            | .co files                                          |

Pipeline diagram showing the logic file lifecycle (approved box+arrow
format).

Absorb rocisa content from current Ch 3 S6.

#### S3. Directory Layout [Essentials]

Kept as-is.

#### S4. Logic File Anatomy [Deep Dive]

Add element map table before the YAML walkthrough:

| Element | Contents                    |
|---------|-----------------------------|
| 0       | Version header              |
| 1       | Scheduling model            |
| 2       | Architecture                |
| 3       | Device ID filter            |
| 4       | Problem type description    |
| 5       | Solution list               |
| 6       | Index mapping               |
| 7       | Size-to-solution mapping    |
| 8-9     | Reserved (null)             |
| 10      | Performance metric          |
| 11      | Selection strategy          |

Then walk through each element as current content does.

#### S5. Code Generation Pipeline [Deep Dive]

Kept. Absorb Python toolchain content from current Ch 3 S5.

#### S6. Building and Testing [Deep Dive]

Kept as-is.

#### S7. How to Add a New Kernel Solution [Deep Dive]

Kept as-is.

## Chapter 7: Host Library Guide

### Current problems

- Opens with detailed call chains including line numbers before explaining
  what the host library does.
- Algorithm selection internals (getBestSolutions, ExactLogicLibrary) mixed
  with the user-facing API explanation.
- No ext API sequence diagram despite the ext API being the recommended
  path.

### New structure

#### S1. What the Host Library Does [Essentials]

2-3 sentences defining its job: translate API calls into contraction
problems, select solutions from the library tree, launch kernels.
Pointer to Ch 3 for core concepts.

#### S2. Request Lifecycle [Essentials]

Two sequence diagrams.

C API path:

```
App       hipblaslt.cpp   rocblaslt     tensile_host    GPU
 |             |              |              |           |
 | hipblasLtMatmul()         |              |           |
 |------------>|              |              |           |
 |             | rocblaslt_matmul()          |           |
 |             |------------->|              |           |
 |             |              | runContractionProblem()  |
 |             |              |------------->|           |
 |             |              |              | launch    |
 |             |              |              |---------->|
```

C++ ext API path:

```
App       Gemm         rocblaslt     tensile_host    GPU
 |         |               |              |           |
 | initialize()           |              |           |
 |-------->|               |              |           |
 |         | makeArgument()|              |           |
 |         |-------------->|              |           |
 | run()   |               |              |           |
 |-------->|               |              |           |
 |         | runKernelFromInvocation()    |           |
 |         |---------------------------->|           |
 |         |               |              | launch    |
 |         |               |              |---------->|
```

Each layer: what it does, what it passes to the next. No line numbers.

#### S3. Algorithm Selection [Essentials]

References the API comparison table from Ch 3 S4 (not duplicated).
Conceptual explanation of the priority cascade:
exact device > fallback device, Equality > GridBased.

#### S4. Lazy Loading [Essentials]

Comparison table:

| Aspect       | Lazy ON (default)    | Lazy OFF             |
|--------------|----------------------|----------------------|
| Startup      | Metadata only        | All .co loaded       |
| First call   | Loads .co on use     | No extra latency     |
| Memory       | Grows on demand      | Peak at startup      |
| Debugging    | Errors at dispatch   | Errors at init       |

#### S5. Adding a New API Feature [Deep Dive]

Current S3 checklist, kept. Remove line numbers, keep file names as
pointers.

### Content removed from Chapter 7

| Current section                           | Destination        |
|-------------------------------------------|--------------------|
| S1 detailed call chains (line numbers)    | Reference appendix |
| S2 getBestSolutions() internals           | Reference appendix |
| S4 RocRoller dispatch flow                | Reference appendix |
| S5 CMake/code lazy loading details        | Reference appendix |

## Reference Appendix

All source-level detail removed from Chapters 3, 6, and 7 is collected
in a new standalone file `11-reference-appendix.md`.

Contents:

- Detailed call chains with line numbers and function signatures
- getBestSolutions() internals and ExactLogicLibrary class hierarchy
- RocRoller dispatch flow, source files, and tile size arrays
- Device library build details
- Lazy loading CMake options and code-level behavior
- Library class inheritance tree (SolutionLibrary, ExactLogicLibrary,
  HardwareSelectionLibrary, ProblemSelectionLibrary, etc.)

This appendix serves developers actively working in the source code.
It is referenced from the main chapters but not required for understanding.

## Success Criteria

1. A reader unfamiliar with hipBLASLt can read Ch 3 S1-S4 and understand
   what Problem, Solution, Contraction Problem, Logic File, and Library mean
   without looking anything up.
2. No term is used in any chapter before being defined (in the same chapter
   or via explicit cross-reference to Ch 3).
3. Every section that introduces a relationship between concepts uses a
   diagram or table, not prose alone.
4. Source-level details (line numbers, function signatures) appear only in
   the reference appendix.
5. The `[Essentials]` / `[Deep Dive]` layering is preserved.

## Quality Gate: Source Verification

After each chapter is written, spawn two Opus subagents (1M context) to
verify the content against the actual hipBLASLt source code at
`/home/poyechen/workspace/repo/rocm-libraries/projects/hipblaslt/`.

### Agent roles

| Agent     | Role                                                        |
|-----------|-------------------------------------------------------------|
| Verifier  | Read the chapter and cross-check every technical claim      |
|           | against the source code: function names, call flows,        |
|           | data types, class hierarchies, selection logic, etc.        |
|           | Flag anything inaccurate, outdated, or misleading.          |
| Reviewer  | Read the chapter from the perspective of the target          |
|           | audience (new team member / adjacent-team engineer).        |
|           | Flag undefined terms, unclear diagrams, missing context,    |
|           | and any place where the reader would need to stop and ask   |
|           | "what does this mean?"                                      |

### Process

1. Both agents run in parallel against the same chapter.
2. Each agent produces a findings list (inaccuracies, gaps, unclear spots).
3. The two agents exchange findings and discuss:
   - Verifier challenges Reviewer's readability flags with source context
     ("this term IS defined in Ch 3 S1, cross-ref is correct").
   - Reviewer challenges Verifier's accuracy flags with audience context
     ("this simplification is intentional for Essentials readers").
4. They produce a joint report: agreed fixes, open questions for the author.
5. Fixes are applied before moving to the next chapter.

### Schedule

Run this gate after each chapter is completed, before starting the next:

```
Write Ch 3  -->  Verify Ch 3 (2 agents)  -->  Fix Ch 3
Write Ch 6  -->  Verify Ch 6 (2 agents)  -->  Fix Ch 6
Write Ch 7  -->  Verify Ch 7 (2 agents)  -->  Fix Ch 7
Write Ch 11 -->  Verify Ch 11 (2 agents) -->  Fix Ch 11
```

This ensures errors do not propagate across chapters (e.g., a wrong
definition in Ch 3 would be caught before Ch 6 references it).
