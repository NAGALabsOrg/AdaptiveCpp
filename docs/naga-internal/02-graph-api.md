# The graph API: specification

Status: the graph around a kernel as agreed in the language pass (summary v6; decisions D15, D16, D17). 01-kernel-language.md states the kernel builder this API submits.

## 1. Purpose and scope

An `sg::graph` owns memory handles and the work submitted against them: device kernels, host work, and a reserved interop kind. Submissions are ordered by the roles they take on handles. A graph is built once and run any number of times; between runs only the contents of handles and the launch values change.

## 2. The graph and its handles

```cpp
sg::graph g;
auto x_handle = g.create_mem_handle<float>();
auto n_handle = g.create_mem_handle<int>();
auto y_handle = g.create_mem_handle<float>();

g.submit<sg::host>([&](sg::host_handle& hh) {
  // host work that writes the size into n_handle
});

g.submit(dev, [&](sg::kernel_handle& kh) {                // the builder itself; default kind: device kernel
  auto n     = kh.launch_value(n_handle);                 // sg::launch_value<int>, read by the host at launch
  auto x     = kh.source(x_handle, sg::shape<1>({n}));    // input role
  auto items = kh.work_items({n});
  auto y = kh.map_items(items, [=](auto item) {
    auto v = x[item.id()];
    return item.compute(v, [](float v) { return 2 * v + 1; });
  });
  kh.evaluate(y_handle, y);                               // result role
});

g.run();                                                  // compiles the graph once and runs it
g.run();                                                  // runs the same graph; only data changes
```

- `g.create_mem_handle<T>()` creates a handle of elements of the trivially copyable type T. Its storage comes from a slab pool of the graph, per device, that grows with need: it starts empty and takes a slab from the device when no free block may be used (`pool_reservation_bytes` only pre-sizes it); every allocation is 256-byte aligned. A free block is reused only when that adds no ordering the graph did not already have, otherwise the pool grows. A handle's size is bound per run and may grow from run to run; a request the device refuses is an error naming the handle and the size.
- The submit lambda is the builder; it runs at submit, exactly once. `kh.source(handle, sg::shape<N>)` declares an input and returns the array as elements; `kh.evaluate(handle, elements)` or `kh.evaluate(sg::inplace(handle, op, init), elements)` declares a result.
- `g.submit` takes the work kind as a template argument: a device kernel by default (`g.submit(dev, builder(sg::kernel_handle&))`), `sg::host` (`g.submit<sg::host>(fn(sg::host_handle&))`), and `sg::interop`.
- `g.run()` does not block; `g.wait()` waits for everything issued.

## 3. Roles and ordering

A handle is a source or a result of one kernel, not both; a source and a result on one handle in one kernel is a build error naming the handle. Roles feed the dependency tracker: for the handle roles read, write and inplace the tracker orders nodes with read-after-write, write-after-read and write-after-write between kernel and host nodes, and returns the events each node waits on. The host work, kernel and host work nodes of one graph run in role order on every run.

- Inplace contributors on one handle form a group: they do not order against each other and combine each element atomically (B2 of 01-kernel-language.md).
- Runs may be issued back to back on one out-of-order retargetable queue and are ordered by handle roles; one `g.wait()` at the end covers all of them.
- Copies between host and device and between devices are planned from placement; a host buffer is kept alive until the completion event of the last copy that uses it.

## 4. Host work

`g.submit<sg::host>(fn)` runs `fn(sg::host_handle& hh)` at submit. The function declares the arrays the host work touches: `hh.write(handle, sg::shape<N>({...}))` and `hh.read(handle, sg::shape<N>({...}))` return arrays whose address is bound when the work runs; `hh.run(callable)` is the work itself and executes asynchronously on every run, ordered by its roles. Host work may write any handle, including the handles that hold launch values.

## 5. Launch values

`kh.launch_value(handle)` returns `sg::launch_value<T>`: a number the host reads from the handle just before the node launches, on every run, after the work that produces it has completed. Any work may write the handle. A launch value is a bound value: `+ - * / %` between launch values and integers give bound expressions, accepted where a size, count, grid extent, work-item range, repeat count or slice extent is taken. It never converts to T and never shapes structure: using a launch value as a tile shape extent, a run length, a loop bound or an array extent of the builder is a compile error naming `launch_value<T>` and structure (rejection 8). The selections of a kernel are recomputed only when a launch value they depend on changes.

## 6. Running, rerunning and compiled code

The first run binds sizes, evaluates the selections of every unit, orders the native groups, builds the tables, composes the kernel and launches. Later runs rebind addresses and launch values, recompute only the selections and tables that depend on changed launch values, clear the counters and flags in queue order before each launch and launch the same compiled kernel. The composition of a kernel is its ordered list of task functions; nothing else about a graph (addresses, sizes, selections, launch values) is part of the identity of the compiled kernel. `sg::jit_compile_count()` returns the number of stage-2 JIT compilations the process has performed so far; a rerun with other data and other launch values leaves it unchanged.

## 7. Options

`sg::graph_options` has `pool_reservation_bytes` (what the slab pool of each device takes at its first use; default 0, an empty pool that grows) and `group_size_cap`: the largest native group a kernel of the graph may use; the default (0) is the device's maximum work-group size, and the cap is part of what rejection 1 reports.

## 8. A graph is built once

A submit after the first run is a build error naming the graph as already built. Changing a graph's structure is a separate operation that is not designed yet; until it is, a new graph object is made. There is no key to compute and nothing to hash: the structure is the recorded kernel graph itself.

## 9. Interop

`g.submit<sg::interop>(...)` is declared and reserved. It compiles and raises a build error naming `sg::interop` as not designed (rejection 15 of 01-kernel-language.md).

## 10. Errors

Every rejection of the language raises `sg::build_error`: a rule number, a construct name, a source location and named numbers; its text is `<construct> at <file>:<line>: <explanation> (name = value, ...)`. The catalogue of fifteen rejections is in section 10 of 01-kernel-language.md. Errors of the graph that are not rules of the language (a handle read before anything produced it, mismatched element types, an allocation the device refuses, a submit after the first run) are `sg::error` with a code and a message naming the handle.

## 11. Decisions

| id | decision |
|---|---|
| D15 | The submit lambda is the kernel builder directly; kh.source declares inputs, kh.evaluate takes the result handle or sg::inplace(...); a graph is built once and the first run compiles it. |
| D16 | Any work in the graph (host work, kernels) may produce a value read at launch; such values are bound values only and never structure. |
| D17 | The deferred host read is kh.launch_value(handle) returning sg::launch_value<T>; graph.submit takes the work kind as a template argument, default a device kernel. |
