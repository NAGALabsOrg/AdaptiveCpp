# Graph API extension (AdaptiveCpp fork): design

This is the current design of the graph API extension to our AdaptiveCpp fork. It describes the design as it stands; it is not a history. The discussion that produced it is in `00-discussion-notes.md`, which is a record only and must not be used as a source of current decisions. Where the two differ, this document is right.

Status: design agreed at a high level, 2026-11-02. No roadmap yet.

## Purpose and scope
- The extension takes data movement, ordering, allocation, lifetime, fusion and dynamic I/O out of SlabKit and CoSpan and puts them in the toolchain.
- It lives in its own `sycl::graph` namespace and is a clean break from upstream SYCL. Upstream pieces may be reused internally; we adapt when they change.
- There is no `sycl::queue` anywhere in it. The graph compiles to host code that does exactly what is needed.
- The public API is the same API used to build graphs internally, so users get full control through the same primitives.
- Afterwards, CoSpan is redesigned on top of it: it stays C++, becomes fully runtime-generated (I/O declared at runtime and accessed by name, no compile-time I/O structs), and so becomes wrappable from Python later. SlabKit shrinks to a few containers.

## Responsibility split
- **The graph extension does execution.** It orders work by access on graph pointers, runs graphs back to back without blocking, allows the graph to change between runs, supports loops, and owns allocation, copies and lifetime. It has no notion of convergence.
- **CoSpan does the model.** It declares I/O at runtime with OpenMDAO-style named access (`inputs["x"]`), turns a connection into two ports sharing one graph pointer, records each forward through the graph API, and owns convergence checks and loop exit conditions for solvers. It does no orchestration, ordering, allocation or copies.

## Graph pointers and memory
- **Type.** `sycl::graph::pointer<T>` behaves exactly like a raw pointer in user code and supports every operator a raw pointer does. The type exists to help introspection and the JIT. Its internal value refers to the root allocation's global metadata plus an offset, not to a device address, because each device copy has its own address.
- **Allocation.** Standard malloc-like calls return graph pointers. The graph pointer is the only allocation type in the graph API for now; there is no user-facing USM.
  - malloc reserves nothing. Its only check is that the allocation fits on at least one device, or in host RAM if that is larger than every device.
  - The sum of all allocations may far exceed any device's memory.
  - A real device or host allocation is made only when needed and is never attempted eagerly. A failure is reported when it actually happens, as an async error.
- **Allocation follows use, per device.** Nothing is allocated for a pointer no node uses, and only devices that use it get a copy. When a later run no longer uses it on a device, that copy is freed.
- **Allocation properties.** Properties on a pointer decide where its allocation and free nodes go in the graph (for example eager, once the shape is known, after an upstream free, or at submission), or force allocation on given devices, or keep a host copy. They never make allocation synchronous. Properties are added as concrete scenarios need them; the set is not fixed.
- **Global tracking.** Pointers are not created against a graph. Their state is tracked globally, so several graphs may run over the same pointers.
- **Ownership.** The runtime's global tracker owns the underlying device and host allocations. The user owns the pointer's lifetime through a free call. No graph owns allocations.
- **Slices.** `auto s = sycl::graph::malloc_slice(ptr, start, end)` returns a `sycl::graph::pointer<T>`. Every graph pointer comes from `malloc` or `malloc_slice`, and every one is released with `free`, root or slice alike.
  - malloc creates the root's metadata and immediately creates a first slice spanning the whole allocation. The "root pointer" is that whole-range slice.
  - Each root has an overlap structure in the global tracker (region trees and interval trees are conceptual precedents). A new slice checks the existing slices for overlap and records which ones it depends on. The whole-range slice overlaps everything.
  - Work on disjoint slices does not order against each other.
  - Using any slice on a device allocates the full range of the original allocation on that device.
  - `free` releases one slice. Freeing the whole-range slice before sub-slices is fine. When the last slice is freed, the root leaves global tracking and its allocations are released.
- **Raw pointers.** Graph pointers are used directly, like raw pointers, in kernels and host nodes alike; their accesses are introspected. `sycl::graph::interop::get_pointer<mode>(ptr)` is the escape hatch for when access cannot be inferred, such as handing a raw address to an external API; it gives a raw pointer with a declared access mode, valid only for the duration of its node. Keeping it from escaping is the application's responsibility. Outside the graph, `lock` is the only way to touch data.
- **Pointer rules for analysis.** No pointers to pointers. The graph pointer is the only type allowed in pointer arithmetic. A derived pointer must trace to a captured root; arithmetic that lands in an uncaptured allocation is undefined behavior. No indirect access in the first iteration (ragged data is a data array plus an offsets array; the compiler treats them as two unrelated arrays).

## Graph object and recording
- **Recording mirrors the queue API.**
  ```cpp
  auto graph = sycl::graph::create();
  auto in  = sycl::graph::malloc<int>(1000);
  auto out = sycl::graph::malloc<int>(1000);
  auto h = graph.submit(dev, [&](auto& cgh) {
    cgh.parallel_for(sycl::range<1>(1000), [=](auto id) { out[id] = in[id]; });
  });
  graph.run();
  ```
  Submitting records a node and returns a handle; nothing executes until `run()`. The handler keeps `sycl::local_accessor` and the rest of the familiar shape.
- **Every operation is a node, and every node is non-blocking.** Kernels, host work, interop, memcpy and fill, migration, allocation and free are all nodes.
- **Pipeline.** Nodes and ordering, then one complete IR representation, then compiled host and device code, then execution.
- **Run and rebuild.** `run()` clears a flag so that subsequent submissions build a new graph from scratch. This is what makes graphs dynamic from run to run. If the new graph is identical to the previous one (same kernel bodies, devices and dependencies), the previous compiled program is reused.
- **Identity and caching.** Graph equality must be cheap: a structural hash built as nodes are added, with comparison operators on the primitives. Node identity is structural (the same compiled result is the same node), not call site or loop index. The structure is the operations and their edges. Sizes, launch configurations and which pointers a node touches are bound values: they are not part of the hash and changing them does not recompile.
- **Completion.** `graph.wait()` and `graph.wait_and_throw()` wait for a graph to finish. An event type returned by `run()` is future work.
- **Destruction.** Destroying a graph waits only for that graph's work still in flight.

## Nodes
- **A node is a composition of primitives, each of which generates IR:** the body (device IR, generated when the application is compiled and JIT-lowered as AdaptiveCpp does today), the launch operation (host code that launches the body natively, supplied by the vendor runtime as host IR), a vendor-erased event, the arguments, and the edges.
- **Kernel bodies are plain SYCL:** `sycl::id`, `nd_item`, group functions, local memory. No new item or group types.
- **Host nodes** are JIT-compiled as part of the graph program, like every node except interop. They use graph pointers directly. They run with `std::async` for now, with a `std::future` behind a vendor-erased event; better mechanisms can come later.
- **Graph calls inside host nodes.** A host node body may call graph APIs (for example `malloc`, `realloc`, memcpy, `free`, submissions), but not nested inside other function calls, so the call sites can be inspected. The compiler splits the host node:
  - Each graph call site becomes its own conditional node, ordered after the host node and JIT-compiled the same way as its normal counterpart, gated by a flag.
  - In the host body, each call becomes a launch-configuration setter. Reaching it at run time sets that node's launch configuration and flips its flag; if it is not reached, the conditional node skips (for example, a resize that does not grow skips its `realloc`).
  - Conditional nodes exist every run, so taking a different branch never recompiles. Their accesses come from the call site and count even when they skip. Flags are epoch-tagged per run.
  - A call site reached more than once in one run is an error. Repeated submission belongs in a loop composite.
  - Downstream nodes depend on a conditional node through an event that is valid whether or not it launched: the launched node's normal event if it ran, signalled directly if it skipped, with native stream ordering where the platform allows and the vendor-agnostic fallback otherwise. The exact mechanism is designed per platform later; each major backend has a host-signalable event that device work can be enqueued to wait on before it is signalled (CUDA and HIP stream wait-value operations, Level Zero host-signalled events, OpenCL user events).
  - This is how dynamic sizes are handled: a downstream library (SlabKit, CoSpan) implements resize as a host node that depends on whatever defines the size and calls the allocation APIs itself. The toolchain has no device-sourced-size feature.
- **Interop nodes** call vendor libraries. Their body is an ordinary precompiled function called by the compiled graph; it is not JIT-compiled. It gets raw pointers through `get_pointer<mode>` and the native stream (or vendor equivalent) assigned to the node, so vendor calls land in order. `get_pointer` also works in host nodes, for external host APIs.
- **Migration nodes** move data between devices and the host. The runtime generates and injects them; they carry their own access on the pointers they move and order like any other node.
- **Composite nodes** contain sub-nodes. Loops and fused kernels are composite types, and more can be added; each type is implemented in its own way.
  - Many events in, one event out: inner nodes order granularly on their own dependencies, internal and external, and wait only on what they touch. Everything downstream sees only the composite's single event.
  - Outside, a composite's access set is the union of its inner nodes' access sets.
  - Composites nest, for example a Newton loop inside a time loop.
- **Loops.** A loop is a composite node (`for` or `while`, with early exit) inside the one compiled graph, so solver iterations keep full throughput and automatic ordering. Its condition check is a host node that sets a loop atomic. The loop node behaves like a host node and never blocks downstream work that does not depend on it. A property on the loop declaration selects the mode:
  - **Hard stop (default).** Each next submission waits for the condition node, then continues or exits.
  - **Speculative (opt-in, loop marked speculative-safe).** Iterations keep being submitted while the condition is checked, up to a configurable throttle; submission stops once the flag flips. Iterations already in flight past convergence still complete, so downstream sees a few extra iterations, not the exact converged one.

## Access and ordering
- **Introspection is trivial.** If anything in a node body can read a graph pointer, the node reads it; if anything can write it, the node writes it. A conditional write counts as a write. Access is read, write or both; for ordering, both behaves as write.
- Introspection happens when the application is compiled, including of `get_pointer<mode>` calls in interop and host bodies.
- **Ordering rules.**
  - Declaration order comes first: nodes whose accesses conflict on the same graph pointer run in the order they were declared.
  - Shared reads do not conflict. Disjoint slices do not conflict.
  - A node that writes a pointer is assumed to need its current contents present.
- **Node-marking APIs** override data dependencies among a set of nodes. They mark nodes, not edges, and never override explicit dependencies between nodes. Correct use is the application's responsibility.
  - `depends_on`: orders nodes that share no data.
  - Force-parallel: a set of nodes is treated as mutually unordered even though their accesses conflict (for example tied outputs). Only members of the set are parallel with respect to one another; against all other nodes, normal ordering applies.
- **Within a graph,** ordering is compiled into procedural host code (a CUDA-only graph compiles to effectively procedural CUDA with streams and events).
- **Across graphs,** whoever reaches the data first goes first and waits on any work still in flight on it. Full cross-graph ordering is not in the first iteration; the per-allocation event tracking used by `lock` is its foundation.

## Host access: lock
- Every node that touches a graph pointer adds its type-erased event to that allocation's global metadata.
- `lock(ptr)` waits on all of those events, from every graph. While it is held, new work on that pointer cannot add events or start; recording graphs is still allowed. It copies to the host, the user reads or writes, and at end of scope the data is propagated back where needed and blocked work continues.
- A lock on a slice waits on and holds off only work on overlapping slices.
- The preferred way to touch data on the host is a host node inside the graph; `lock` is the slow, blocking way from outside. Nothing is kept in sync ahead of time.

## Errors
- Errors that occur while `run()` is issuing work, such as a failed kernel launch, are thrown immediately.
- Errors that occur while work executes are async errors, delivered through an async handler as with queues today. This includes kernel runtime errors and allocation failures, since allocations are async nodes.

## Compilation and lowering
- The runtime generates IR as well as compiling it. All IR is composed into one program and compiled in one go, so the compiler optimizes the whole host program. Host JIT uses standard LLVM JIT APIs.
- **Vendor boundary.** The core runtime knows nothing about vendor internals. Vendor runtime libraries implement a strict interface returning vendor-erased primitives (for example `create_parallel_for_node`, returning a fully populated node with vendor-specific IR inside), plus lower-level interfaces for composing vendor-erased JIT primitives into nodes. Vendor runtimes ship pre-generated host IR for things that never change, such as launching a kernel; the kernel and launch configuration are arguments to that code.
- **Stitching.** Mechanical, in node order. Each node exposes `get_event`, and dependents consume it. Injected nodes chain as needed; for example, with on-demand allocation: producer event, then the allocation node on the target device, then the memcpy node, then the work node, which depends on the memcpy's event. Each node turns its event dependencies into IR: native stream ordering when both ends are on the same platform, a vendor-agnostic fallback otherwise.
- **Graph pointers in kernels.** The kernel IR gets a graph-pointer argument slot. The launch code fills it each run with the current device address of the bound pointer on that device. The kernel sees a raw pointer with no lookup, and moving memory never recompiles. If an address stays put, AdaptiveCpp's own adaptivity may still specialize on it; that is its decision.
- **Loops** lower to a for or while loop in the compiled host code. Vendor-agnostic first; vendor-native lowering such as CUDA graph conditional nodes is a later optimization.
- **Cache.** The persistent graph cache lives under the user's acpp directory (for example `~/.acpp/graph`), beside AdaptiveCpp's persistent kernel cache, keyed by the structural hash, holding the compiled host code.
- AdaptiveCpp's device lowering and adaptivity stay upstream's; nothing here overrides them.

## Host-evaluated calls (named runtime I/O)
- A decorator marks a method or function as host-evaluated. Used at a call site inside a kernel body, it means exactly what it would if hoisted out and run first:
  ```cpp
  auto x = inputs["x"];  // host call, same thread as the submission
  graph.submit(dev, [&](auto& cgh) {
    cgh.parallel_for(..., [=](auto id) { auto y = some_function(x); });
  });
  ```
- The call runs as a normal host call, in the submitting thread, and its return value replaces the call site before JIT compilation.
- For now it may only return a graph pointer. The compiler generates a graph-pointer slot in place of the call; the kernel receives the pointer the call returned at run time. Rewiring which pointer it returns does not recompile.
- The decorator must be on the method at the call site, not nested further down the call stack.
- This is what lets CoSpan's string-keyed `inputs["x"]` work inside kernels with no compile-time I/O structs.

## Fusion
- **Manual fusion only.** Automatic fusion is a possible future research direction.
- **Model.** CoSpan's fused system, ported over, runtime-shaped and SYCL-shaped: the same stages and segments, local memory configurations (partitioned and shared), rank rules, group shape rules and data rules, and ticket ordering.
- **API shape (sketch; the real API is CoSpan-like, runtime-shaped).**
  ```cpp
  auto h1 = graph.submit(...);
  auto h2 = graph.submit(...);
  sycl::graph::fuse(h1, h2, ...);
  ```
  The fuse call checks the launch configurations, strips everything but the kernel bodies, generates the fused kernel body and its launch host IR, and collapses the nodes into one composite node.
- **Producer-consumer ordering inside a fuse** is resolved per producer-consumer pair, in this order:
  1. **Introspection.** If introspection deterministically resolves the dependency to specific producer groups, it wins: generated ticket waits are inserted, no tracker memory is used, and marker or hand-written ticket code for that pair is stripped. A result that can only prove "wait on the whole segment" does not count; it is the floor.
  2. **User's choice at the fuse call,** one or the other per pair, never both; the fuse strips the option not chosen. The person doing the fuse chooses, not necessarily the kernel author.
     - **Finished markers.** The producer marks, through a graph API, each element it is the final writer of; at most one work-item may mark an element (more is undefined behavior). The tracker is a binary flag per produced element, release on mark, acquire on read, epoch-tagged so it needs no reset. Consumers spin on the flag and leave the spin early if the producer's segment completes without a mark. This cannot deadlock: tickets are taken when a group starts, so earlier-ticket producer groups are already running. Only pointers written by one stage and read by a later stage get trackers.
     - **Explicit ticket ordering.** The consumer uses the ticket ordering APIs, as in CoSpan, with no tracker. This is the path for external kernels written without fusion in mind.
     - Tradeoffs: if all producers mark, tickets are unnecessary; an external consumer forces tickets unless the producer is patched; many trackers may make tickets cheaper.
  3. **Floor.** Wait on the producer's whole segment. Always correct; cannot deadlock.
- A later stage writing data that an earlier stage reads is not allowed: an error where detectable, undefined behavior otherwise.

## Graph metadata
The graph API generates metadata to help users understand their graph and make decisions for their runtime environment. One example: for each producer-consumer pair in a fuse, which path it took (introspected, marker, ticket or segment floor).

## Extensibility
The design is compositional at run time, including the host code that executes the graph, so batched or chunked systems and Python wrappers fit with at most minor changes. Design for extension within reason.

## Fork logistics
- The extension lives under its own directory, completely disjoint from the rest of the AdaptiveCpp codebase. Vendor runtime libraries keep their extension-specific APIs in their own contained directories as well.
- It is a true extension: it may use upstream core functionality, but it does not modify it. The only changes that go into non-extension code are improvements to core upstream functionality, never anything specific to the graph API.
- Upstream sync, packaging and CI follow the fork's existing practice.
- Further detail takes shape during implementation.

## Future work
- Finer access introspection, such as proving partial or conditional writes.
- A full-overwrite or discard marking, so migration can skip bringing current contents.
- Better host-node execution than `std::async`.
- An event type returned by `run()`.
- Cross-graph ordering.
- Vendor-native loop lowering.
- Automatic fusion.
