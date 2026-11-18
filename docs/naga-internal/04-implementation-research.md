# Implementation research: findings and proposals

Status: research of 2026-11-17 by the research lead. Everything here is a finding or a proposal. Nothing here is a decision until it is agreed with Jack. Each finding says where it came from. Section 14 lists what was not read.

## 1. What the fork already gives us

Kernel path (generic SSCP flow), from reading `src/compiler/sscp` and surveys of `src/runtime`:

- Kernels are found by function annotations. A stage-1 pass clones the module, outlines the kernels, writes a per-kernel record (parameters, flags, annotations) and embeds the bitcode in the host binary.
- The user lambda is the single kernel argument. `AggregateArgumentExpansionPass` flattens it into one parameter per non-aggregate member and records, per parameter, its byte offset in the original struct and its original index. Size and type are added from the new signature.
- **Parameter annotations come purely from type names.** Any enclosing struct whose name contains `__acpp_sscp_emit_param_type_annotation_<name>` puts `<name>` on every member inside it. A new wrapper type gives a new annotation with no compiler change; only the runtime has to act on it.
- **Launch values as constants already exist.** `specialized<T>` arguments of 8 bytes or less are recorded in the kernel configuration; the translator replaces the argument with a constant and folds. Known group sizes are replaced the same way.
- **Launch-resolved repetition already exists.** Dynamic functions, with the fork's indexed call sequences, build a wrapper at JIT time that calls each replacement in order with a constant index.
- **A JIT-time value already becomes an array size.** On SPIR-V the dynamic local memory global is replaced by an array sized from a JIT option (`LLVMToSpirv.cpp`, `setDynamicLocalMemoryCapacity`).
- Every one of these feeds the kernel cache key, so a change of structure is a new cached binary. The persistent cache is one file per binary id, written atomically and recorded in the per-application database.
- Atomics (including float add, min and max) and a memory fence with device scope exist as generic builtins.
- Group algorithms exist as typed builtins for nine standard operations on fundamental types. The helper underneath, `wg_reduce`, is generic over the operation.
- Local memory: one dynamic block per kernel. Its size is a launch parameter on CUDA and a JIT-time array on OpenCL and Level Zero. The host uses per-thread buffers.
- Every backend already has a way to make enqueued work wait on something the host decides (section 9).
- **We own the compiler build.** Clang, lld and the runtimes are built from a pinned ROCm llvm-project with our patches, and AdaptiveCpp is built in-tree with its passes linked into clang.

## 2. What the fork lacks

- **No host JIT.** There is no ORC, LLJIT or ExecutionEngine anywhere. CPU kernels are compiled by shelling out to bundled `opt`, `llc` and `ld.lld` and loading the result. The core runtime links no LLVM; LLVM reaches the runtime only through the backend plugins.
- Group algorithms reject a user-written operation at compile time, and float reductions support only plus, multiply, min and max. Struct elements cannot use the helper, because its shuffle bit-casts the value to an integer.
- On the CPU the sub-group barrier is effectively empty and the work-group barrier's memory ordering is unfinished.
- Level Zero: `submit_memcpy` is 1D only, and `get_enqueued_event_handles` casts every element to the Level Zero event type. These are the known breakages to fix first.

## 3. Recognition of kernel definitions

Options:

1. **AST for checks, IR for generation.** Checks and diagnostics run at the Clang AST level, where the user's constructs and source locations exist. Generation runs in the existing stage-1 pipeline, with our constructs surviving to IR as annotated marker functions and types, the way kernel annotations, dynamic functions and the wrapper types already do. Precedent: Enzyme, and AdaptiveCpp itself.
2. **AST to AST.** A `TreeTransform` inside Clang produces a second, plain SYCL AST (the Clacc approach). Closest to "transpile to SYCL" and readable as output, but it needs a carried Clang patch and duplicates what the stage-1 pipeline does.
3. **IR only.** Structure of template-heavy code is lost after inlining unless markers preserve it, and diagnostics in the user's terms are hard.

Because AdaptiveCpp is linked into our own Clang, option 1 and option 2 are both open to us; the limits of a loadable plugin do not apply. Lean: option 1, with option 2 as the fallback if markers do not survive well.

## 4. Launch placeholders

Lean: use the mechanisms of section 1 and add none. Launch values are specialised arguments. Derived group shape and local memory size are build options. Arrays sized at launch follow the SPIR-V local memory precedent. Chain layers and many-to-one mappings follow the indexed call sequence precedent. Handles and array slots are kernel arguments marked by wrapper-type annotations. The host program resolves and validates all of it before the launch.

## 5. One launch: ordering between logical groups

Findings (NVIDIA programming guide; Sorensen and others 2016 and 2021; Merrill and Garland 2016; Smith, Levien and Owens 2025; an AMD paper of 2020):

- Work groups of one launch are not guaranteed to be resident together. A group that spins on a group not yet scheduled deadlocks. No cross-vendor specification guarantees progress between work groups.
- The one model that held on every device tested is that a group which has started keeps running.
- CUDA's grid synchronisation needs a cooperative launch sized to what the device holds, fails if it does not fit, and has no OpenCL or SYCL equivalent.
- Ticket ordering needs only the model that held: a group takes its ticket after it has started and waits only on lower tickets. Merrill and Garland give exactly this as their fix for deadlock.
- Decoupled fallback is the other portable answer: a group that finds a result missing computes it itself, and a compare-and-swap lets one copy publish. It needs the work to be recomputable from inputs, which our pure bodies are. It costs redundant work.
- Residual hazard outside our control: a pre-empted kernel may resume with fewer resources.

Proposal (a hypothesis, to be proven by the experiments in section 11): a **persistent ticket kernel**.

- The native launch has some number of native groups, decoupled from the logical work. A ticket stands for one unit of logical work: a logical group of a definition, in a given cycle iteration. A native group claims the next ticket from an atomic counter, carries it out, and claims again.
- `item` and `local` mappings carry out the source's work inside the consumer's ticket.
- `global` mappings, durable array reads, `evaluate` and cycles are ordered by tickets, with a flag in global memory and a device-scope fence.
- A cycle whose length depends on data needs no more native groups; the same groups keep claiming tickets until the cycle finalizes.
- Naive form: a grid of one native group, which walks all tickets in order and never waits. Larger grids are the same generated code.
- Must not block: larger grids per backend, decoupled fallback, vendor grid synchronisation as a per-backend optimisation.

Open inside it: how tickets map to work in a cycle, since iteration k+1 only has meaning after iteration k's `evaluate`; whether ticket order is always a valid order of the dependencies; how a group barrier behaves inside the claim loop.

## 6. Placing logical groups and local memory

Precedents: one launch whose groups pick their body by id range (PyTorch Inductor's combo kernels); a wrapper sized to the largest requirement, with handlers using a subset of threads (Meld, Pagoda); software-managed shared memory (Pagoda, Meld).

- Naive: the native group shape is the largest any definition needs, and items beyond a definition's logical group size idle for that ticket. Local memory is the sum of all local-backed arguments, carved from the one dynamic block at offsets known at launch.
- Must not block: reuse of local memory where the structure allows, and per-definition native shapes.

## 7. Group bodies and operations

- Standard operations on fundamental types map to the existing group builtins.
- A user-written operation on scalar types can instantiate the generic `wg_reduce` directly.
- Struct elements need our own tree in local memory with barriers.
- Must not block: atomic strategies for many-to-one combines.

## 8. Host IR

Precedents: ORC LLJIT with an object cache; PostgreSQL and ClangJIT, which import precompiled bitcode marked `available_externally` so it inlines while anything not inlined binds to the compiled copy; Proteus, which embeds the IR of annotated host functions and lambdas in the binary and JIT-compiles them at run time.

Proposal:

- An in-process LLJIT compiles the composed host program at `run`.
- Vendor launch primitives ship twice: compiled in the backend plugin, and as bitcode imported into the composed module. The core runtime stays ignorant of vendor internals.
- Host work bodies and submit scopes are embedded as bitcode by a stage-1 pass, in the same container as kernels.
- Runtime objects are referred to by symbol, so a compiled object stays reusable when addresses change.
- Compiled host programs are stored as object files beside the kernel cache.
- Pitfalls reported by others: lambdas need call-site analysis before optimisation hides them; an imported function that touches a non-exported symbol fails at emission; linking LLVM adds tens of megabytes.

Open: where LLVM and the JIT live, in the core runtime or in a new plugin.

## 9. Events, ordering and conditional nodes per backend

| Backend | Orders work on | Wait on a host decision |
|---|---|---|
| CUDA | a stream | `cudaLaunchHostFunc` holds later work in the stream until it returns; it may not call CUDA, and host functions on different streams may be serialised. The fork also defers submission entirely when a dependency is host work. |
| Level Zero | an immediate command list | `zeEventHostSignal`; an append that waits on an unsignalled event may block the calling thread |
| OpenCL | a command queue | a user event, set exactly once |
| OpenMP | a worker thread | the worker waits |

Host tasks today run on a pool of worker threads; a worker blocks on each requirement, runs the body, then signals.

- Naive: gate conditional nodes with each backend's mechanism above, as the fork already does for dependencies on host work.
- Must not block: vendor-native conditional nodes.

## 10. Memory planning, structural identity and caching

- Memory, naive: one allocation per value, released at last use. This is the trivial planner every surveyed compiler starts from.
- Memory, later: lifetime intervals from the graph's ordering and greedy packing of values whose lifetimes do not overlap (IREE, XLA, TVM).
- Identity: hash our own graph primitives as nodes are added, and confirm with an equality check. LLVM's structural hash is not an identity; its maintainers say so.
- Sizes, handles and launch values are bound values and are left out of the hash.

## 11. Experiments before commitment

Each is small with a pass or fail answer, on OpenMP, CUDA, OpenCL and Level Zero where it applies.

- **E1.** Hand a counter across work groups with release and acquire ordering, then with far more groups than the device holds, claiming tickets dynamically. A backend that hangs stays at a grid of one.
- **E2.** `evaluate` inside the ticket loop, with a cycle whose length depends on data.
- **E3.** A group barrier inside the ticket loop.
- **E4.** Host JIT: LLJIT in the runtime, a module composed from an imported vendor launch primitive plus generated glue, compiled, cached and used to launch one existing kernel.
- **E5.** Gating enqueued work on a host decision, per backend.
- **E6.** Front end: one AST-level check with a diagnostic in the user's terms, and one annotated marker surviving to stage-1 IR.

## 12. How the work divides

Housekeeping (Level Zero fixes); the experiments; a vertical slice of one item-body `parallel_for` with a durable output, launched through a minimal graph with a JIT-compiled host program; kernel language breadth on a grid of one; launch-time resolution and rejections; larger grids; graph breadth. The answer key is hand-written native SYCL kernels for the hard cases, one negative test per rejection, and every kernel test asserting exactly one launch.

## 13. Judgement calls for Jack

- Recognition: option 1 or option 2 of section 3, and whether we carry a Clang patch.
- Where the host JIT and LLVM live.
- The persistent ticket kernel as the model for one launch, and whether a grid of one native group is acceptable as the first setting.
- Suggestions not built unless agreed: a plain C++ reference interpreter of the spec as a test oracle; a validator for generated ticket schedules.

## 14. Not read or not verified

- `StaticLocalMemoryPass`; the SPIR-V and PTX scan, predicate, broadcast and atomic implementations; the bodies of `submit_ze_kernel` and `submit_ocl_kernel`; most of `naga/build/lib.nu` and `build.nu`.
- That a user-written operation fails to compile with the group algorithms was read from the headers, not confirmed by compiling.
- The cause of the Level Zero crash is unproven.
- Decoupled fallback where the missing work has its own missing dependencies.
- No experiment in section 11 has been run.
