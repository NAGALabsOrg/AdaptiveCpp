# Graph API extension (acpp fork): discussion notes

Living notes from the design discussion between Jack and the research lead. Each closed phase of the discussion is appended here. Status: brainstorming; no roadmap yet.

## Direction
Move data movement, ordering, lifetime, fusion and dynamic I/O out of SlabKit/CoSpan and into an AdaptiveCpp-fork extension: a graph API in its own `graph` namespace, a clean break from upstream SYCL (upstream pieces may be reused internally; we adapt when they change).

## Discussion order (agreed approach)
1. What SlabKit and CoSpan give users today, and why each feature exists (user-facing, not implementation).
2. From that, the requirements the graph API must support.
3. Ergonomics of the graph API.
4. Implementation.

## Decided
- Graph pointer is a distinct C++ type (carries its root allocation and offset), not a raw pointer from a special allocator. Raw-pointer escape to vendor libraries happens only inside a host lock or a native-interop node.
- Dependencies are structure: the graph defines a program that is JIT-compiled; ordering changes the program. The cache key covers operations and their edges; sizes, launch configuration and which data a node touches are bound parameters.
- Node identity is structural (same compiled result = same node), not call site or loop index.
- Access analysis rules: no pointers to pointers; the graph pointer is the only type allowed in pointer arithmetic; a derived pointer must trace to a captured root allocation (arithmetic landing in an uncaptured allocation is UB); explicit access-pattern API as a fallback.
- Access granularity: start with whole allocation; revisit.
- Host JIT scope: only the functors passed to graph APIs; their host side is built from IR primitives shipped with the toolchain, per backend; vendor runtime libraries expose the hooks the JIT needs.

## Why run policies exist (Jack, 2026-11-01)
Run policies are a means to an end. The cases they serve:
- One-off setup work inside a model (e.g. computing interpolation weights).
- Reviving that setup later (e.g. dynamic problems where weights must be recomputed after some number of forward calls), after which it goes out of scope again.
- Memory-constrained models: subsystems go out of scope right after their work each iteration so that other subsystems can come into scope, with non-ephemeral work queued in between so execution continues while deallocation/allocation happens.

Allocation/deallocation timing they imply:
- Deallocation on going out of scope is immediate (the point is to free memory for downstream).
- Allocation modes: eager (as soon as possible); deferred until the full shape is known (dynamic shapes from runtime outputs); after an upstream deallocation (handoff between swapped subsystems); lazy at submission (safest, slowest).
- Desired: allocation and deallocation happen asynchronously inside the graph. The graph owns all allocations; containers need no blocking destructors; one graph object (like today's single queue) does the only blocking cleanup at its destruction.

Observation: with a graph re-recorded every run, "in scope" = recorded this run and "revive" = record again; most of the remaining job is the graph's memory/lifetime rules.

## Constraints to keep open
- Batched/chunked systems are future work: the design must not make them harder than necessary. General rule: design for extension within reason.

## Requirements phase (closed 2026-11-01)
Scope: these are the changes to the AdaptiveCpp fork. CoSpan stays C++ and becomes fully runtime-generated: inputs and outputs are declared at runtime and accessed by name, with no compile-time I/O structs. That makes Python wrappers possible later. Python itself is out of scope.

### Responsibility split
- **The graph extension does execution.**
  - It orders everything by access on graph pointers, including back edges: whatever reads or writes a pointer waits, as today.
  - It runs back to back without blocking between runs, allows dynamic changes to the graph between runs, and runs sub-portions of the graph.
  - It owns allocation, copies and lifetime.
  - It has no notion of convergence.
- **CoSpan does the model.**
  - It declares I/O at runtime and gives OpenMDAO-style named access (`inputs["x"]`), with shapes, through its own wrapper, including when the inputs structure is captured in a kernel. How this compiles to typed pointers is open.
  - A connection means two ports become the same graph pointer, so ordering follows from access.
  - It records each forward through the graph API.
  - For solvers, CoSpan owns the convergence check and halting and drives re-runs of a subgraph. The mechanism is open.
  - CoSpan no longer does orchestration, ordering, allocation or copies.

### Graph extension requirements
- **Graph object.**
  - It owns every allocation and is re-recorded every run.
  - Its cache is structural.
  - Blocking happens only at its destruction and in `lock`.
- **Graph pointer with sub-slices as allocations.**
  - A sub-slice of one allocation (root + offset + extent) can be treated as its own allocation for ordering. Work on different sub-slices does not order against each other.
  - Why:
    - SlabKit allocates one large slab as a single allocation. Its pool hands out pages, and its bump arena hands out regions, each a sub-slice of the slab, and each must behave as its own allocation.
    - In future, CoSpan may connect sub-arrays, and components working on different sub-arrays must not order against each other.
  - Ties are NOT sub-slices.
- **Access patterns the graph must express (from CoSpan ties and depends_on).**
  - Shared reads of a whole allocation by many nodes, in parallel. For example, components tied to one design variable, which the optimizer updates once.
  - Writes by several nodes to disjoint sub-slices.
  - Writes by several nodes to the same allocation in a required order, with no data dependency between them (today's `depends_on`).
  - Parallel atomic accumulation by several nodes into any portion of the same allocation.
  - The mechanism is open.
- **Allocation follows use, per device.**
  - By default nothing is allocated for a pointer no node uses, and only devices that use it get a copy.
  - When a re-record stops using it on a device, that copy is freed.
  - Allocation properties override the default: always allocate on a given device or on all devices, or keep the host copy.
- **Allocation timing and frees.**
  - Allocation modes: eager, once the shape is known, after an upstream free, and at submission.
  - Frees are immediate and asynchronous inside the graph.
- **Node kinds.**
  - Kernels (range and group-range).
  - Host nodes: JIT-compiled, they capture graph pointers and need no lock.
  - Native interop and memcpy/fill.
  - Exceptions from host nodes reach the graph. Synchronizing backend calls are not allowed inside host nodes.
- **Host access.**
  - The blessed way is a host node inside the graph.
  - `lock(ptr)` is the only way from outside the graph, and it is slow and blocking:
    1. Wait for in-flight users on every device and host node, and block new users.
    2. Copy to the host the best available way.
    3. The user reads or writes.
    4. At end of scope, do a blocking copy to every remaining device that needs it.
    5. Free the host memory unless a host node uses it or a property keeps it.
  - Nothing is kept in sync ahead of time. Non-blocking snapshots are rejected.
- **Fusion API.**
  - It fuses submissions, not function bodies.
  - A `parallel_for` with a fully described range and functor only records IR primitives and returns a handle.
  - `fuse(handles, params)` composes a new set of primitives from those handles.
  - Each component owns its own submissions. Manual fusion only; automatic fusion is a possible future research direction.
- **JIT API.**
  - It sits below fusion and is for building custom nodes by composing JIT/IR primitives at a fine grain: separate functions, different arguments, separate call definitions. That goes beyond today's call sequences and indexed call sequences.
  - The graph compiler compiles the result. AdaptiveCpp's lowering and adaptivity stay upstream's and the API cannot break them. No hints, no policy.
  - It may be unnecessary if named runtime bindings are enough.
- **Sizes from device-produced values.** Wanted, so that a dependent size does not force a host round trip. Today there is one host wait per shape hook per forward. The mechanism is open.

### Parked (how-questions, not settled now)
- How named bindings lower to typed pointers.
- How a subgraph is re-run under CoSpan's convergence control.
- How device-sourced sizes work.
- How the tie and depends_on access patterns are expressed.

### Discussion split for the AdaptiveCpp phase (agreed starting point, 2026-11-01)
1. Graph pointer and memory model.
2. Graph object and recording.
3. Access analysis and ordering.
4. Node kinds and the host side, including `lock`.
5. Compilation and lowering.
6. Fusion API.
7. JIT API and named runtime bindings.
8. Device-sourced sizes.
9. Fork logistics.

CoSpan and SlabKit come in only lightly, as test cases, and only where needed.

### Evidence
One-pass inventories of SlabKit and CoSpan main (b937876) are in `raw/01-slabkit-and-extensions.md`, `raw/02-cospan-components-launch-fusion.md` and `raw/03-cospan-structure-data-lifetime.md`.

## Topic 1: graph pointer and memory model (closed 2026-11-01)
- **Type.** `sycl::graph::pointer<T>` is a decorated raw pointer. The decoration is the type itself and exists to help the JIT. User code treats it exactly like a raw pointer and supports every operator a raw pointer supports. Its internal value is ours to choose: probably a reference to the root's global metadata plus an offset, not a device data address, because each device copy has its own address.
- **Global tracking, not per graph.** Pointers are not created against a graph. All their state is tracked globally, so several graphs can run in parallel over the same pointers.
- **Ordering.**
  - Within a graph, ordering is compiled into procedural host code. A CUDA-only graph, for example, compiles to effectively procedural CUDA with streams and events.
  - Across graphs, whoever gets to the data first goes first, and waits on any work still in flight on that data anywhere.
- **Allocation API.** Standard malloc-like calls.
  - The graph pointer is the only allocation type in the graph API for now. There is no user-facing USM.
  - malloc reserves nothing. Its only check is that one allocation fits on at least one device, or in host RAM if that is larger than every device.
  - The sum of all allocations may far exceed any device's memory, because allocations follow use.
  - A real USM or system allocation is made only when needed and is never attempted eagerly. A failure is reported when it actually happens.
- **Allocation properties.** The default is allocation follows use. Properties such as eager allocation are added as concrete scenarios need them; we are not enumerating them now.
- **Slices.** `auto s = sycl::graph::slice_pointer(ptr, start, end)` returns a `sycl::graph::pointer<T>`.
  - Each root has a tree or overlap structure in the global tracker. A new slice checks the root and every existing slice for overlap and records the dependencies. The root overlaps everything. Region trees (Legion) and interval trees are conceptual precedents.
  - malloc creates the root's metadata (size of the original allocation and so on) and immediately creates a first slice spanning the whole allocation. The "root pointer" is just that whole-range slice.
  - Using any slice on a device allocates the full range of the original allocation on that device.
  - free frees a slice. Freeing the root's whole-range slice before the sub-slices is fine. When the last slice is freed, the entry is removed from global tracking and the underlying allocations are released.
- **Lifetime ownership.** The runtime (the global tracker) owns the underlying USM and system allocations. The user owns the pointer's lifetime through the free API.
- **Supersedes** two earlier notes that said the graph owns all allocations: the "Desired" bullet under "Why run policies exist", and the "Graph object" bullet under "Graph extension requirements". Allocations are owned by the global tracker, not by a graph.

## Topic 2: graph object and recording (closed 2026-11-01)
- **No sycl::queue anywhere.** The graph compiles to host code that does exactly what is needed, with no queue involved. The public API is the same API we use to build the graph, so users get full control through abstract primitives.
- **Recording API (sketch).** Free functions take the graph and a device and add a node; nothing executes when a node is added.
  ```cpp
  auto graph = sycl::graph::create();
  auto in  = sycl::graph::malloc<int>(1000);
  auto out = sycl::graph::malloc<int>(1000);
  sycl::graph::fill(graph, dev1, in, 5);
  sycl::graph::parallel_for(graph, dev2, sycl::range<1>(1000), [=](auto id){ out[id] = in[id]; });
  sycl::graph::host_exec(graph, [=](){ for (int i = 0; i < 1000; ++i) std::print("{}", out[i]); });
  graph.run();
  ```
- **Pipeline.** Nodes and ordering, then one complete IR representation, then compiled host and device code, then the compiled code runs. Ordering comes from access on graph pointers (introspected or user-supplied access modes). Cross-device data movement (for example, `in` filled on dev1 and read on dev2) is inserted by the graph.
- **A node is a composition of base primitives, each of which generates IR.** The primitives are:
  - the body, as device IR, generated when the application is compiled and JIT-lowered as AdaptiveCpp does today;
  - the launch operation, which is host code that natively launches the body type (host, parallel_for, memcpy, and so on). It is a virtual object returned by the vendor runtime, with a method that returns vendor-specific host IR;
  - a vendor-erased event type;
  - the arguments;
  - possibly the edges.
- **Monolithic compile.** All IR is composed into one representation and compiled in one go, so the compiler can optimize the whole host program. The runtime now generates IR, not just compiles it. Host JIT uses standard LLVM JIT APIs. Vendor runtimes ship pre-generated host IR for things that never change, such as launching a kernel; the kernel and the launch configuration are arguments to that compiled host code.
- **Vendor boundary.** The core runtime knows nothing about vendor internals.
  - Vendor runtime libraries implement a strict interface that returns vendor-erased primitives, for example `create_parallel_for_node`, which returns a fully populated node with vendor-specific IR primitives inside it. Low-level interfaces let vendor-erased JIT primitives be composed into nodes.
  - The vendor-erased event lowers to native ordering between adjacent nodes of the same vendor (for example, CUDA streams and events) and to a cross-vendor fallback otherwise.
- **Host nodes.** For now they run with `std::async`. Their event is a `std::future` behind a vendor-erased implementation: the best route between adjacent host nodes, and a vendor-agnostic fallback between a host node and a device node. Better mechanisms can come later.
- **Run and rebuild.** Calling `run()` clears a flag, so subsequent graph API calls build a new graph from scratch. This is what makes the graph dynamic from run to run.
  - If the next graph is identical (same kernel bodies, same devices, same dependencies, and so on), the previous internal representation, kept in its abstract primitive form, is reused.
  - Equality of two graphs must be cheap to compute, through custom comparison operators on the primitives. A structural hash built as nodes are added is one standard way to do this.
- **Loops (sub-portions for solvers).** A sub-portion of the graph can be defined as a loop, either `for` or `while`, with a way to exit early. The loop is part of the one compiled graph, so solver iterations keep full throughput and automatic ordering. Running a separate small graph per iteration was rejected. CoSpan owns the convergence check that drives the early exit; the graph has no notion of convergence.
- **Type IR override (named I/O).** When the kernel body IR is generated, a type may override its conversion. To start, there is exactly one kind of conversion: from the original type to a graph pointer value, injected as a constant using the IR primitive AdaptiveCpp already uses for constants (similar to what `sycl::specialized` does underneath). For example, `auto x = inputs["x"]` lowers to the graph pointer stored in `inputs` on the host. Fancier conversions can come later.
- **Destruction.** Destroying a graph waits only for that graph's work still in flight.
- **Parked for topic 5 (lowering):**
  - Which value a specialized graph-pointer constant bakes in: the handle (which needs a lookup on the device) or the device address on that device (which means a new specialization when memory moves).
  - How the loop exit condition is lowered. CUDA graph conditional/while nodes are a native precedent when the whole loop body is on one vendor.

### Topic 2 addendum: composite nodes (2026-11-01)
- **Composite nodes are a core concept.** A composite node contains sub-nodes, and composite types can be added later. How each composite type is implemented underneath depends on its type.
  - A loop (`for` or `while`, with early exit) is a composite node.
  - A fused node from the fusion API is likely another composite type; this is to be settled in the fusion topic.
- **Boundary rule: many events in, one event out.**
  - The base nodes inside a composite order their work granularly, by their own dependencies, both internally and on external upstream nodes. Each inner node waits only on what it touches.
  - Everything downstream of the composite sees only the composite's single event, never the events of its inner nodes.
  - Outside the composite, its access set is the union of its inner nodes' access sets.
  - Composites nest; for example, a Newton loop inside an outer time loop.

## Topic 3: access analysis and ordering (closed 2026-11-01)
- **Introspection is trivial.** If anything in a node body can read from a graph pointer, the node reads it. If anything can write to it, the node writes it. Either or both. A branch that writes only sometimes counts as a write, to be safe. No proof of partial versus full writes is attempted.
- **Access modes.** Read, write, or both. For ordering, "both" behaves like write. Read-write is not a separate concept: it is just both flags set.
- **Migration is made of nodes.** Data movement between devices and the host is done by nodes the graph runtime generates and injects automatically. Each carries its own read and write access on the graph pointers it moves, and orders by the same rules as any other node. A node that writes a pointer is assumed to need its current contents present.
- **Ordering.**
  - Declaration order comes first: nodes whose accesses conflict on the same graph pointer run in the order they were declared.
  - Shared reads do not conflict.
  - Disjoint slices do not conflict (topic 1).
- **Node-marking APIs.** Two APIs override data dependencies among a set of nodes. They mark nodes, which do the work, not edges, which order it. They do not override explicit dependencies between nodes.
  - `depends_on`: explicitly order nodes that share no data.
  - Force-parallel: a set of nodes is treated as mutually unordered even though their accesses conflict. This covers tied outputs, which by default would run in declaration order. Only the members of the set are parallel with respect to one another; against every node outside the set, normal ordering rules still apply. For example, with three components tying one output where the first two have a real data dependency, only the last two are marked force-parallel.
  - Correct use is the application's responsibility. We do not hedge against misuse.
- **CoSpan look-ahead (light).** In CoSpan, parallel is the default for tied writers, so CoSpan applies force-parallel automatically, and `depends_on` is the user-provided fallback. Users never call force-parallel directly. By construction, CoSpan does not let users misuse it.
- **Future work, not now:**
  - finer introspection, such as proving partial writes or conditional writes;
  - a full-overwrite or discard marking, so that migration can skip bringing current contents. CoSpan outputs are computed fresh every forward, so they would benefit.

## Topic 4: node kinds and the host side (closed 2026-11-01)
- **Every operation is a node, done asynchronously.** That covers kernels, host work, interop, memcpy/fill, migration, allocation and free. Allocations are always async graph nodes, eager ones included. malloc properties on a graph pointer decide where in the graph its allocation node goes, not whether it is async.
- **Submission mirrors the queue API.**
  - `graph.submit(device, [&](auto& h){ ... })` with a handler, so it looks almost identical to queue submission. This replaces the topic 2 free-function sketch (`sycl::graph::parallel_for(graph, dev, ...)`).
  - Submitting only records; nothing executes until `run()`.
  - Having a handler also keeps `sycl::local_accessor` working for group-range kernels.
- **Kernel bodies are plain SYCL.** `sycl::id`, `nd_item`, group functions, local memory. No new item or group types.
- **Interop and raw pointers.**
  - `auto* p = sycl::graph::interop::get_pointer<sycl::graph::access_mode::write>(graph_ptr);` gives a raw pointer with a declared access mode.
  - It works in interop nodes (vendor libraries) and in host nodes (external host APIs that expect pointers).
  - The interop body is not JIT compiled; it is an ordinary precompiled function called by the compiled graph. Its `get_pointer` calls are introspected when the application is compiled, which tells the graph each node's accesses before it runs.
  - The interop body also needs the native stream or vendor equivalent assigned to its node, so vendor calls land in the right order.
  - A raw pointer is valid only for the duration of its node. Keeping it from escaping is the application's responsibility.
- **Completion and errors.**
  - `graph.wait()` and `graph.wait_and_throw()` wait for a graph to finish. `lock` only waits for in-flight nodes that touch its pointer; the unit of work is a node, not a graph.
  - There is an async handler for graphs, as with queues today.
  - Errors that occur while `run()` is issuing work, such as a failed kernel launch, are thrown immediately.
  - Errors that occur while work executes are async errors: kernel runtime errors, and allocation failures, since allocations are async nodes.
  - An event type returned by `run()` is future work.

## Topic 5: compilation and lowering (closed 2026-11-01)
- **Graph-pointer constants.**
  - The type IR override bakes the graph pointer's handle into the kernel IR as a constant, using AdaptiveCpp's constant primitive. The device address is not baked in.
  - At compile time the graph knows which handles each kernel uses. The generated launch code passes each handle's current device address on that device as an ordinary kernel argument.
  - The kernel sees a raw pointer, with no lookup on the device.
  - When memory moves (for example, a subsystem going out of scope and coming back under allocation-follows-use), the kernel IR does not change and nothing recompiles.
  - When an address happens to stay put, AdaptiveCpp's own adaptivity may still specialize on it. That is its decision, not ours.
- **Stitching.**
  - The IR is stitched together mechanically, in node order. Every operation is a node and every node is non-blocking.
  - Each node exposes an event (`get_event`), and dependent nodes consume it.
  - The runtime inserts nodes according to the rules from earlier topics. Example: with on-demand allocation, the chain is upstream producer event → allocation node on the target device → memcpy node → work node. The work node depends on the memcpy's event, not directly on the producer's.
  - Each node converts its event dependencies into IR:
    - When both ends are on the same platform, it uses native stream ordering in host IR.
    - When the platforms differ, it uses a vendor-agnostic ordering fallback.
- **Loops.**
  - A loop node lowers to a for or while loop in the compiled host code. The loop keeps submitting iterations until its condition is met.
  - We start vendor-agnostic. Vendor-specific lowering, such as CUDA graph conditional nodes, is a later optimization.
  - The condition check is a host node that sets a loop atomic.
  - The loop node itself behaves like a host node, so it never blocks downstream work that does not depend on it.
  - Two modes are set by a property on the loop declaration:
    - **Hard stop (default).** Each next submission waits for the condition host node to complete, then checks the flag and either continues or exits.
    - **Speculative (opt-in only).** The loop must be marked speculative-safe. The next iterations are submitted while the condition is checked. Submission stops once the flag flips, and a configurable throttle bounds the number of iterations in flight. Consequence: iterations already in flight past convergence still complete, so downstream sees the results of a few extra iterations, not the exact converged one. This is why the mode is opt-in and not the default.
- **Cache location.** The persistent graph cache lives under the user's acpp directory (for example `~/.acpp/graph`), beside AdaptiveCpp's existing persistent kernel cache. It is keyed by the structural hash and holds the compiled host code.
- **Future work.**
  - Vendor-native loop lowering, such as CUDA graph conditional and while nodes.
  - Better host-node execution than `std::async`.
- **Supersedes** the "Parked for topic 5 (lowering)" bullets under Topic 2.

## Topic 6: fusion (closed 2026-11-01)
- **Port CoSpan's fused system, runtime-shaped and SYCL-shaped.** The model is the same as CoSpan's: stages and segments, local memory configurations (partitioned and shared), rank rules, group shape rules and data rules. Ticket ordering is supported in the same way.
- **API shape (sketch).** Kernels are submitted as ordinary nodes, then fused:
  ```cpp
  auto h1 = graph.submit(...);
  auto h2 = graph.submit(...);
  sycl::graph::fuse(h1, h2, ...); // real API is CoSpan-like, runtime-shaped
  ```
  The fuse call:
  - gets and checks the launch configurations for runtime validity;
  - strips everything but the kernel bodies;
  - generates the IR for a new kernel body that implements the fuse, plus the host IR that launches it;
  - collapses the fused nodes into a single node, which is a composite node.
- **Producer-consumer ordering inside a fuse.** It is resolved per producer-consumer pair, never globally. The order of precedence is:
  1. **Introspection (best case).** If introspection deterministically resolves the dependency to specific producer groups, it wins outright. Generated ticket waits are inserted, no tracker memory is allocated, and any marker or hand-written ticket code for that pair is stripped. A result that can only prove "wait on the whole segment" is the floor, not the best case.
  2. **Fallback, chosen by the user at the fuse call.** The person doing the fuse chooses, not the kernel author, though they are often the same person. They choose one of two options per pair; it is never both. The fuse strips whatever was not chosen.
     - **Finished markers.** The producer marks, through a graph API, each data element it is the final writer of. At most one work-item may mark an element; more than one is undefined behavior. The tracker is a binary flag per produced element, with release ordering on the mark and acquire ordering on the consumer's read; flags are epoch-tagged so they need no reset. Consumers spin on the flag and exit the spin early if the producer's segment completes without the element being marked. The spin cannot deadlock: tickets are taken when a group starts, so an earlier-ticket producer group is already running. Only producer-consumer pointers get trackers; read-only data, such as ragged offsets, never does.
     - **Explicit ticket ordering.** The consumer kernel uses the ticket ordering APIs, as in CoSpan today, with no tracker. This is the path for external kernels that were not written with fusion in mind and do not mark.
     - **Tradeoffs behind the choice.** If every producer marks, ticket ordering is unnecessary. An external consumer forces tickets unless the producer is patched. Many trackers may make tickets cheaper.
  3. **Floor.** Waiting on the producer's whole segment. It is always correct and cannot deadlock, because waits only go backwards in ticket order.
- **Forbidden pattern.** A later stage writing data that an earlier stage reads is not allowed, as in CoSpan. It is an error where detectable and undefined behavior otherwise.
- **Graph metadata.** The fuse reports which path each producer-consumer pair took: introspected, marker, ticket, or segment floor. This is one of a number of pieces of metadata the graph API generates to help users understand their graph and make decisions for their runtime environment.
- **Supersedes** the "likely another composite type; settled in the fusion topic" note in the Topic 2 composite-node addendum. A fused node is a composite node.

## Topic 7: host-evaluated calls (JIT API) and the host lock (closed 2026-11-02)
- **Host-evaluated call decorator.** This generalizes the type IR override for named I/O (topic 2).
  - A decorator on a method or function marks it as host-evaluated. Nothing else is special about the code. Its meaning is exactly what it would be if it were hoisted out of the kernel body and run first:
    ```cpp
    auto x = inputs["x"];            // host call, same thread as the submission
    graph.submit([&](auto& cgh) {
      cgh.parallel_for(..., [=](auto id) { auto y = some_function(x); });
    });
    ```
  - The call runs as a normal host call, in the same thread as the submission. The call site is replaced by its return value before JIT compilation.
  - For now it may only return a graph pointer. The decorator tells the compiler to generate IR for a graph pointer in place of the call. The value the kernel body receives is the pointer the call returned at run time.
  - The decorator must be on the method at the call site in the kernel body, not nested further down the call stack.
  - Consequence for caching: which allocation the call returns is a bound value, not structure. Rewiring `inputs["x"]` to a different graph pointer does not change the kernel IR and does not recompile.
- **Correction to topic 5 wording.** The kernel IR gets a graph-pointer argument slot, and the launch code fills it each run with the current device address of whatever pointer was bound. Pointer identity is a bound value: it is not part of the structural hash and is not a constant in the kernel IR.
- **Host lock.**
  - Internal, type-erased graph events still exist. Every node that touches a graph pointer adds its event to that allocation's global metadata.
  - `lock(ptr)` waits on all of those events. While it is held, new work on that pointer cannot add events or start. Recording a graph is still allowed; only execution of nodes touching the pointer is held off. When the lock goes out of scope, submissions continue.
  - A lock on a slice uses topic 1's overlap tracking. It waits on and holds off only work on overlapping slices.
  - Because the lock waits on that pointer's events from every graph, it lays the foundation for cross-graph ordering. Cross-graph ordering itself is not in the first iteration.
- **Extensibility.** No further design is needed now. The design is compositional at run time, including the host code that executes the graph, so batched or chunked systems and Python wrappers fit, with at most minor changes.
