# NAGA internal design documents

These documents hold the design of the NAGA kernel design language and graph API for our AdaptiveCpp fork. They state the design as it stands. They are not a history, and where an older document disagrees with them, these are right.

Status: user-facing design agreed 2026-11-18. Implementation design (how a kernel definition and a graph become code) is not in these documents yet; it is the subject of the research phase that follows them.

## The documents

| File | What it holds |
|---|---|
| `01-kernel-language.md` | The kernel design language: what a kernel definition is, what each construct means, what the toolchain rejects, what the user can rely on. |
| `02-graph-api.md` | The graph as the user sees it: submissions, memory handles, host work, ordering, plus the host IR design carried over from 2026-11-02. |
| `03-vocabulary.md` | Every name the user can write, by level, with its meaning. |
| `04-implementation-research.md` | Findings and proposals from the implementation research. Proposals, not decisions. |
| `history/` | The 2026-11-02 graph API design and its discussion record, verbatim. Superseded record only: never a source of current decisions. |

## What we are building

- A compiler toolchain, not a library and not an embedded DSL. A kernel definition is source our compiler reads. Its constructs are symbols we interpret, and the code we emit need not resemble them.
- One kernel definition is one kernel launch. This is a hard requirement.
- Our job is to transpile a kernel definition into a native SYCL kernel, which AdaptiveCpp then compiles as it does today. AdaptiveCpp already has adaptive kernel IR and JIT specialisation. What it does not have is the host IR our graph needs.
- Kernel code is shaped like the problem, not like the hardware. The user breaks work down into small local operations over input data that each produce a result, and the language makes orchestrating those easy. The user can still reason about how it lowers.
- Everything is a pure function. Nothing is mutated in place as far as the user can see.

## How the work proceeds

- Every feature in these documents is implemented and tested before work resumes on any other project. About one month is the ceiling; anything past that is getting things green, not revisiting design.
- Everything is built the most naive correct way first. Compiler optimisations (reuse of toolchain-allocated memory, atomic strategies, local memory reuse and the like) wait until everything is green. Naive never means more than one launch per kernel definition.
- Remaining work falls into three kinds:
  1. User-facing design, driven by Jack in discussion.
  2. Mechanical technical work with no opinionated answers, done and reported by the research lead: recognition of kernel definitions in the compiler, the functional checks, IR representation and launch placeholders, host-side validation and diagnostics, derivation of the native range and group shape, lowering of item-level pieces, build sequencing, the Level Zero fixes.
  3. Technical design that needs an informed proposal first: the host IR, placing logical groups onto native groups, ordering across native groups inside one launch, the global allocation planner, kernel-wide agreement for `evaluate` and cycles, the host sync for index arrays.

## Set aside and deferred

- **Fusion** (joining kernels into one launch) is set aside. The in-kernel language covers what a reasonable kernel needs. We circle back later.
- **IR primitives for libraries and applications** are a named requirement with no design. Frameworks such as CoSpan declare inputs, outputs and shapes at run time and need to control how their code becomes IR (for example a call like `inputs["x"]` inside a kernel); this is also the door to Python bindings. AdaptiveCpp's dynamic functions extension shows the flavour; ours is to be more sophisticated and composable. It is not designed until the toolchain's real shape is known.
- **Properties for configuring toolchain behaviour** (eager or lazy allocation, immediate or deferred release, reproducible reduce order and the like) are deferred until implementation is under way. Until then defaults are fixed and nothing depends on a property existing.
- **Arbitrary host access to graph memory** outside host work (what `lock` was in the 2026-11-02 design) is explored later.
- **Multi-device contributors** to an in-place combine are later; see `02-graph-api.md`.

## Standing rules for our own code

- Nushell for scripts; nothing taken from the system outside the pixi environment.
- No vendor libraries in our own code. Users remain free to interop with them.
