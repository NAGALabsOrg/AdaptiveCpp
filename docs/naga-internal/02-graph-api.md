# The graph API: specification

Status: user-facing design agreed 2026-11-18, revising the 2026-11-02 design around kernel definitions. The host IR section is carried over from 2026-11-02 and is the starting point for implementation research, not a finished design. Code is illustrative; names are in the namespace written `sg` (`sycl::graph` in the 2026-11-02 design).

## 1. Purpose and scope

- The graph takes data movement, ordering, allocation and lifetime out of applications and puts them in the toolchain.
- It is a clean break from upstream SYCL. There is no `sycl::queue` anywhere in it. The graph is a recorded program that compiles to host code doing exactly what is needed.
- The graph has no notion of convergence or of a model. Libraries built on it (CoSpan, SlabKit) own those.

## 2. Submission

```cpp
graph.submit(device, [&](auto gh) {
  auto x = sg::array<float, 1>(x_handle, {n});
  gh.define_kernel([=]() { ... });
});
graph.run();
```

- `submit` takes a device and a body that receives a graph handler.
- A submission is exactly one piece of work: a kernel (`define_kernel`), host work (`define_host_work`), or interop work.
- `submit` creates host IR that is JIT compiled; the submit body is itself part of the compiled host program. "Submit runs at the call site" means only that the lambda is consumed at the call site, so capturing by reference is safe. It is the definition inside that is the off-loop work.
- Nothing executes until `run`.
- The graph is built at run time by ordinary host code, as a builder. A host `for` loop around `submit` is one kernel definition submitted many times.
- The graph handler is also where per-submission properties would go; none are designed.

## 3. Memory handles

"Graph pointer" in the 2026-11-02 design is now a **memory handle**.

- A handle is created on the host, for example `sg::create_mem_handle<float>(size)`. The size is optional: a handle may start with none and be sized later by host work.
- Handles are captured by value. In a kernel submission they are used to construct arrays: inputs in the submit scope, durable outputs in the kernel body. A handle is never used as a raw pointer in a kernel body.
- The allocation's size and the size of an array constructed on it must match.
- **Creating a handle reserves nothing.** The only check is that the allocation fits on at least one device, or in host memory. The sum of all allocations may far exceed any device's memory. A real allocation is made only when needed, and a failure is reported when it happens, as an async error.
- **Allocation follows use, per device.** Nothing is allocated for a handle no work uses, and only devices that use it get a copy. When a later run no longer uses it on a device, that copy is freed.
- **Global tracking.** Handles are not created against a graph. Their state is tracked globally, so several graphs may run over the same handles. The runtime's global tracker owns the underlying allocations; the user owns the handle's lifetime through `free`.
- **Slices.** A slice is a handle onto part of an allocation. Each root allocation has an overlap structure in the global tracker. Work on disjoint slices does not order against each other. Using any slice on a device allocates the whole root allocation there. `free` releases one slice; when the last is freed the root leaves tracking and its allocations are released.

## 4. Kernels and ordering

- A kernel submission constructs its input arrays in the submit scope and defines one kernel; see `01-kernel-language.md`.
- **Ordering between pieces of work comes from the roles of arrays on handles and slices**: input, durable output, host read, host write. This replaces access introspection of pointer use.
- Declaration order comes first: work whose roles conflict on the same handle runs in the order it was recorded. Shared reads do not conflict. Disjoint slices do not conflict.
- **Several producers into one handle, without the in-place decorator,** run in declaration order and the last write wins. The toolchain warns when a write is overwritten with no consumer in between: either the decorator was forgotten or a consumer was, and the earlier work is useless.
- Data movement between devices and the host is done by migration nodes the graph injects. They order like any other node.

## 5. In-place combine across kernels

This replaces `depends_on` and force-parallel from the 2026-11-02 design. It is for two or more separate submissions producing into the same handle.

```cpp
auto y = sg::array<float, 1>(sg::inplace(y_handle, std::plus<>{}), contribution_def);
```

- A durable array may be constructed with `inplace(handle, op)` in place of the handle. The decorator is in the kernel so that the kernel compiles to a combine and not a store, and so that the graph knows the submission runs alongside the other contributors.
- Kernels that do so on the same handle are contributors. The graph does not order them against each other.
- The result is the element-wise reduction, with `op`, of the initial value and every contributor's output.
- The initial value is whatever an ordinary, undecorated producer left in the handle beforehand; the graph orders that producer first.
- A reader is ordered after all contributors recorded before it. Contributors recorded after a reader accumulate on top.
- A contributing kernel cannot read the decorated array.
- `op` is associative and commutative; the order is unspecified.
- Contributors to one handle must name the same operation and, for now, be on the same device. Otherwise it is an error.
- Later, for contributors on different devices, the graph will inject a kernel after them that brings the results to one device and reduces element-wise. The meaning does not change.
- Kernels writing to separate slices need no decorator; the user slices the allocation.

## 6. Host work

Ordinary asynchronous host work, integrated into the graph. It is called host work, not host task, so it is not taken for SYCL's host tasks. It works like a kernel submission, except that host work may read and write data as it wishes.

```cpp
graph.submit([&](auto gh) {
  auto x = sg::host_array<float, 1>(x_handle, {size});        // read and write
  auto y = sg::host_array<const float, 1>(y_handle, {size});  // read only
  gh.define_host_work([=]() { ... });
});
```

- **Precise access.** A `host_array` constructed in the submit scope states the access: a const element type reads, a non-const one reads and writes. The graph orders by it.
- **Conservative access.** `sg::get_host_handle(handle)` taken in the submit scope declares that code somewhere in the host work may reallocate or mutate. It allows `realloc(size)`, with a size determined while the work runs; constructing a `host_array` on it from any scope in the host work; and `get_pointer()`. It forces the most conservative access for ordering. A `host_array` may be constructed inside host work only through a host handle.
- Violations are async errors, since they can be deep in a call tree.
- For now, host access to graph memory is allowed only inside host work. Arbitrary host access is explored later.

```cpp
auto data = sg::create_mem_handle<float>();
graph.submit([&](auto gh) {
  auto hdata = sg::get_host_handle(data);
  gh.define_host_work([=]() {
    hdata.realloc(size);
    auto arr = sg::host_array<float, 1>(hdata, {size});
    arr[0] = 1.f;
  });
});
```

## 7. Graph promise and future

A host-only primitive of the graph, for a value host work computes that later submissions need, such as a size that comes from data.

- The pair is created on the host, like a memory handle. It carries one plain value.
- Host work takes the promise in its submit scope, which declares it the producer, and sets it.
- A later submission reads the future in its submit scope, wherever a launch value is allowed: an array's shape, a range, a group shape, a chain's layer count.
- Reading it is the dependency. The graph orders that submission after the producer. There is no explicit wait and no run boundary.
- To a kernel it is an ordinary launch value: a placeholder resolved by the host program just before that launch. For hashing and caching it is a bound value, not structure.
- Async errors: a future read with no producer recorded before it; a promise set twice or never.

## 8. Interop

Users may interop with vendor libraries. Interop work is shaped like a host work submission, with interop handles that give access to native objects. It is not designed further yet.

## 9. Graph cycle

A cycle exists at two levels, and it is the same abstract primitive at both. The in-kernel cycle (`01-kernel-language.md`, section 10) repeats inside one launch. The graph cycle repeats a set of submissions. In CoSpan's terms the first is like an implicit component and the second like a solver over a system of components, but neither is tied to solving systems.

- The graph is built by ordinary host code, so repetition with a count the host already knows needs no primitive: it is a host loop that calls `submit`. The graph cycle is for repetition whose stop condition is only known while the graph runs.
- A graph cycle is a composite node inside the one compiled graph. Its condition check is host work. The cycle never blocks downstream work that does not depend on it.
- A composite has many events in and one event out; its roles on handles are the union of its inner nodes'. Composites nest.
- **Conditional nodes.** What a graph cycle issues depends on a condition only known while the graph runs, so its nodes are conditional nodes, as designed on 2026-11-02. They exist every run, each gated by a flag, so taking a different path never recompiles. A conditional node that is skipped signals its event directly, so its dependents see a valid event either way. Flags are tagged per run. The gating mechanism is per backend.
- Handles carried from one iteration to the next are runtime arguments, tracked at run time.
- **Hard stop (default):** each next iteration waits for the condition, then continues or exits.
- **Speculative (opt-in):** iterations keep being submitted while the condition is checked, up to a throttle. Iterations already in flight past the exit still complete.

## 10. Running, identity and caching

- `run` executes the recorded graph and clears the recording, so later submissions build a new graph. This is what makes graphs dynamic from run to run.
- If the new graph is identical to a previous one, the compiled program is reused.
- Identity is a structural hash built as nodes are added: the operations and their edges. Sizes, launch values, graph futures and which handles are bound are bound values. They are not in the hash and changing them does not recompile the host program.
- `graph.wait()` waits for a graph to finish. Destroying a graph waits only for that graph's work in flight.
- The persistent graph cache lives beside AdaptiveCpp's persistent kernel cache, keyed by the structural hash, holding compiled host code.

## 11. Errors

- Errors while `run` is issuing work, including every launch-time validation in `01-kernel-language.md`, are thrown immediately.
- Errors while work executes are async errors, delivered through an async handler: runtime errors in kernels, allocation failures, host work violations, promise and future misuse.

## 12. Reporting

`graph.report()` gives, after a run, what the toolchain derived and chose, in terms of the user's constructs. The per-kernel content is listed in `01-kernel-language.md`, section 13.

## 13. Host IR (carried over from 2026-11-02)

This is the design as it stood, restated in the current vocabulary. It is the starting point for the implementation research.

- **Pipeline.** Nodes and their ordering, then one complete IR, then compiled host and device code, then execution. All IR is composed into one program and compiled in one go, so the compiler optimises the whole host program. The runtime generates IR as well as compiling it. Host JIT uses standard LLVM JIT interfaces.
- **Every operation is a node and every node is non-blocking:** kernels, host work, interop, copies, migration, allocation and free.
- **A node is a composition of primitives that each generate IR:** the body, the launch operation, a vendor-erased event, the arguments and the edges. For a kernel the body is device IR: our compiler transpiles the kernel definition to a native SYCL kernel, which AdaptiveCpp compiles and JIT-lowers as today. The launch operation is host code that launches the body natively, supplied by the vendor runtime as host IR.
- **Vendor boundary.** The core runtime knows nothing about vendor internals. Vendor runtime libraries implement a strict interface returning vendor-erased primitives, plus lower-level interfaces for composing them into nodes. Vendor runtimes ship pre-generated host IR for what never changes, such as launching a kernel; the kernel and its launch configuration are arguments to it.
- **Stitching** is mechanical, in node order. Each node exposes an event and its dependents consume it. Injected nodes chain as needed: producer, then allocation on the target device, then copy, then the work. An event dependency becomes native stream ordering when both ends are on the same platform, and a vendor-agnostic fallback otherwise.
- **Arrays in kernels.** The kernel gets an argument slot per array. The launch code fills it each run with the current device address of the bound handle. Moving memory never recompiles.
- **Launch values** are placeholders in the kernel IR, resolved and validated by the host program before the launch.
- **Host work** is JIT-compiled as part of the program and run with `std::async` for now, with a future behind a vendor-erased event. Interop work is the exception: a precompiled function the program calls.
- **A graph cycle** lowers to a loop in the compiled host code, with its conditional nodes gated per backend.

Flagged in 2026-11-02 and never worked out:

- The runtime needs LLVM available to JIT host code.
- Host work bodies and vendor launch code must exist as IR at run time, so the application has to carry host IR.

## 14. What changed from 2026-11-02

| 2026-11-02 | Now |
|---|---|
| `sycl::graph::pointer<T>` behaving as a raw pointer in kernels; pointer rules for analysis | Memory handles; arrays constructed on them; never a raw pointer in a kernel |
| Ordering by introspected reads and writes of pointers | Ordering by array roles |
| Kernel bodies are plain SYCL with a launch range | Kernel definitions; no range or group shape on the kernel |
| `depends_on` and force-parallel | In-place combine; declaration order with a warning otherwise |
| `lock` as host access from outside the graph | Host access only inside host work for now |
| Device-sourced sizes through graph calls in host nodes | Host handles with `realloc`, and graph promise and future |
| Host-evaluated calls | Set aside, as the requirement "IR primitives for libraries" |
| Fusion | Set aside |
| Loop composite | Graph cycle; the conditional node machinery stays |

## 15. Open for research or later discussion

- The exact form of a graph cycle's body and stop condition.
- Whether a kernel's output ever needs its old contents brought to the device. An undecorated output is fully defined by its kernel; an `inplace` output is not.
- What a host work submission takes in place of a device.
- The design of interop work.
- Cross-graph ordering, an event returned by `run`, better host execution than `std::async`, vendor-native graph cycle lowering: future work as in 2026-11-02.
