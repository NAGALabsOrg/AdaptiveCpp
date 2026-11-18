# Kernel design language: specification

Status: agreed 2026-11-18. This document states the design as it stands. Code in it is illustrative: it shows shape, and exact syntax is not locked beyond the names in `03-vocabulary.md`. Names are written in the namespace `sg`.

## 1. Frame

- **A kernel is a definition.** Everything inside a kernel definition is kernel code, read by our compiler. Nothing in it runs on the host, runs once to build something, is recorded, or is attached at run time. The user reads their code with its plain meaning, and that meaning is the contract.
- **One kernel definition is one launch.**
- **Everything is a definition of work.** A `parallel_for` defines parallel work in groups that produces an array. "Lazy" is how to think about it, not a statement about how many times anything is computed.
- **Entirely functional.** There is no in-place mutation as far as the user can see. Everything is producer to consumer.
- **No loop statements anywhere** in a kernel definition: no `for`, `while` or `do` at the top level, in an item body, in a group body, or in any function called from a body.
- **Launch-resolvable.** A value is resolvable at launch when it is known at the moment the kernel is launched: host values captured into the kernel, the shapes of input arrays, graph futures. A value computed by work inside the kernel, including what `evaluate` returns, is never a launch value. Launch-dependent parts of a definition are placeholders in the kernel IR; the host-side code a submission compiles to resolves and validates them before device code is lowered.
- **Logical groups are not native groups.** The user thinks in groups shaped like the problem. How a definition's logical groups are placed onto native work groups follows from the mappings and is otherwise the toolchain's.
- **A kernel has no declared range or group shape.** The native range and group shape are derived from the work inside.

## 2. `parallel_for` with an item body

Think in groups of threads tiled over an array. Each thread produces one value, and the result is a plain multi-dimensional array of values.

- It takes an item range, a group shape, optionally a mapping (section 5), and a body.
- The range is the problem's index space, one to three dimensions. The group shape divides it into logical groups, and every group does the same work.
- The range must be an even multiple of the group shape. There are no partial groups.
- Range and group shape are launch-resolvable.
- Each `parallel_for` has its own range and group shape. Nothing ties two definitions together except mappings and reads of durable arrays.
- **The item handle** carries problem coordinates only: the item's index in the range, its group's index, its index within the group, the range and the group shape. It carries nothing about the machine: no native group, no thread identity, no barrier.
- **The body returns exactly one value of one kernel-valid type**: a float, an integer, a statically sized array, a struct. That type is the element type of the array the definition describes, and the array's shape is the range. Returning a 3D point gives an array of 3D vector values; a struct gives an array of structs. A return is never split into several arrays.

```cpp
auto y_def = sg::parallel_for(sycl::range{n}, sycl::range{64},
                              [=](auto item) { return 2.f * x[item.id()]; });
```

## 3. `parallel_for` with a group body

Think in groups, not threads: a grid of groups, of rank at most three, where each group independently produces its own array, or a scalar after a full reduction. Every group's output has the same shape and rank, but the outputs are independent. Groups are akin to arrays of arrays. (Do not describe this as nesting; that is too narrow.)

Requirements:

1. It is defined over a range of groups, the grid, not over items. No group shape is given. The native group shape for the work is the toolchain's to choose.
2. It must have at least one mapping.
3. Its arguments are group arrays, or group iterators of them (section 6). It never receives item-mapped values.
4. The body uses group-wide operations and the array vocabulary of section 11. How they are carried out is the toolchain's business.
5. The shape of what each group produces is the shape returned by the final operation the body returns. Reductions may be over one dimension, over several, or full: a 64x64 input may become 64x1, or a single value.
6. An item mapping from a group-body definition is not allowed, because a group body defines nothing at the item level.

Example, deliberately complex: one mapping delivers many 64x64x3 groups as an iterator of group arrays; a second delivers a 64x3 group array of ints. The body reduces across the iterator (64x64x3), reduces over the first dimension (64x3), selects by the int array with `map` (64x3), scans the first dimension (64x3), and returns the scan.

## 4. When a definition can become an array

A definition becomes an array when a durable array is constructed from it (section 8), or wherever the toolchain must store it as an array.

- **Item body:** always. The array's shape is the range.
- **Group body, equal ranks:** a group range of l x m x n whose body returns i x j x k describes the same array as an item-body `parallel_for` with group shape i x j x k over the item range i*l x j*m x k*n. The independent chunks stitch side by side. Two groups each scanning 64 give 128; two groups each reducing to one value give 2. Converting an RGB texture to greyscale is a 2D grid of groups over the texture, each reducing away the channel, stitched back into the image.
- **Group body, scalar return:** capturable at any grid rank. The array has the shape of the group range.
- **Group body, differing ranks:** cannot be captured directly. It feeds another `parallel_for`, usually one with an item body, and that definition states the layout.
- The rank check applies only where a definition has to become an array. A group-body definition of any returned rank is a valid intermediate when it is mapped onward: the consumer receives group arrays of whatever shape was returned.

## 5. Mappings

- **What a mapping is.** A mapping feeds subsets of the array one `parallel_for` defines into a later `parallel_for`, expressed group to group. Where there is no mapping and no read of a durable array, there is no dependency, and the two definitions can be scheduled independently.
- **How it is written.** A mapping is a functor of the target group's id. For that target group it returns one or more source maps. Each source map names an earlier definition, which of its groups feed this target group, and the mapped kind: `local`, `item` or `global`.
- **Naming source groups** is done only through the index-space primitives: a single id, a range (including multi-dimensional), a strided range, a stencil, or an index array. Everything else (all groups, boxes, rows and columns, prefixes, pairs, coarsening) composes from these. Boundary behaviour is part of how the stencil is designed, not a further form. Because only these forms exist, a mapping is launch-resolvable by construction. The index-space primitives are not specific to groups; the same ones slice arrays.
- **The index array form.** The array, or a subslice of it that is resolvable at launch, must be an input array of the kernel. This form costs a host synchronisation: the host parses the index array before the launch, as host work the kernel launch depends on in the graph, so unrelated graph work is not blocked. It is a real cost, to be avoided or paid a small number of times.
- **Any relation is allowed.** Many source groups into one target, one source group into many targets, source groups skipped, and counts that differ from one target group to the next are all valid. How each is carried out is lowering.
- **What the body receives.** Each source map is one further argument of the body, in order, after the item or group handle. Each argument comes from exactly one source definition. The mapped kind is on the source map, so bodies can take `auto`. The number of source groups is never a template parameter.
- There is no separate join and no many-map helper. The list of source maps is the join, and a source map naming several groups is the many-to-one case.

```cpp
auto step1 = ...; // 64 groups
auto step2 = ...; // 128 groups, group shape 64

auto step3_mapping = sg::map_groups([=](auto group_id) {
  return sg::sources{
      sg::source_map<sg::mapping::local>(step1, sg::range{2 * group_id, 2 * group_id + 2}),
      sg::source_map<sg::mapping::item>(step2, group_id)};
});

// 32 groups; step1_inputs is an iterator of two group arrays, step2_input a value
auto step3 = sg::parallel_for(sycl::range{2048}, sycl::range{64}, step3_mapping,
                              [=](auto item, auto step1_inputs, auto step2_input) { return ...; });
```

## 6. Mapped kinds and argument types

Two independent things decide an argument's type.

**The kind decides value or array.**

- `item`: each item of the consuming group receives, by value, what the matching item of the source group returned. The argument is the element type itself. The source must be an item-body definition with the consumer's group shape. It uses no memory beyond the item's own scope.
- `local`: the argument is a `group_array` in the consuming group's local memory. Its shape is what the source group produced: 64x64 for item bodies over a 64x64 group, a single value for a group body that fully reduced. There is no shape requirement between source and consumer.
- `global`: exactly as `local`, with the values in global memory the toolchain allocates. It is what the user chooses when there is not enough local memory, or when the source should keep its own parallelism.

**The arity decides plain or iterator.** One source group per target group gives the plain thing. Several source groups for a target group give a `group_iterator` over them, of group arrays or of values. An iterator may hold different counts for different target groups.

| Mapping | one source group per target | many source groups per target |
|---|---|---|
| `mapping::item` | `T` | `group_iterator<T>` |
| `mapping::local` | `group_array<T, N, mapping::local>` | `group_iterator<group_array<T, N, mapping::local>>` |
| `mapping::global` | `group_array<T, N, mapping::global>` | `group_iterator<group_array<T, N, mapping::global>>` |

Common rules:

- The kind never changes what is computed. It changes where values live and what ordering it costs. Choosing it is the tuning; mappings and bodies are the formulation.
- What arrives is read-only.
- `local` and `global` arguments share exactly the same semantics and rules.
- Under `item` and `local`, the source group's work is carried out inside the consumer's native group. For example, a source of 64 logical groups feeding a target of 32 in pairs under `local` is carried out as 32 native groups, each doing its two logical source groups in order with no sync point, at the cost of parallelism in the source. Under `global` that is not required: the source may run on its own native groups, with the dependency ordered inside the launch.

## 7. `wait`

`wait(arg)` is usable in item bodies and group bodies, where `arg` is an argument from a `local` or `global` mapping. Underneath it is a group barrier, paid for only when necessary.

- **Same group shape on source and consumer:** there is no automatic barrier. Item i's source value is element i, and the item may read it back directly. An explicit `wait(arg)` is required before reading elements other items produced. This is what lets a value that is private to an item but read many times be parked in a local array, to reduce register pressure, without paying for a barrier.
- **Different group shapes:** the barrier is enforced automatically.
- `wait` must be uniform across the group, as with a SYCL barrier.
- In a group-body consumer no `wait` is needed; handing an argument to a group operation implies it.
- A `wait` on an argument that already has its barrier is a no-op, not an error.
- The toolchain does not try to catch a read of another item's element made without a `wait`. Bodies are the smallest unit of code and the user reasons about them.
- **Local memory.** The total of local-memory-backed arguments across all body definitions is the maximum local memory the kernel needs. The toolchain may reuse local memory where the structure allows, so the real requirement can be less.

## 8. Graph memory in a kernel

```cpp
graph.submit(device, [&](auto gh) {
  auto x = sg::array<float, 1>(x_handle, {n});      // input: constructed before the kernel definition
  gh.define_kernel([=]() {
    auto y_def = sg::parallel_for(...);             // computes y from x
    auto y = sg::array<float, 1>(y_handle, y_def);  // durable output, readable downstream in this kernel
  });
});
```

- **Inputs.** An input array is constructed in the submit scope, before the kernel definition, from a memory handle and a shape. Rank is in the type and the shape is launch-resolvable. It is read-only and captured by value. It creates no dependency inside the kernel; the graph orders the kernel after whatever produced the data.
- **Durable arrays.** A durable array is constructed inside the kernel from a memory handle and a definition: a `parallel_for`, or what a cycle or chain yields. Its shape and element type come from the definition. It is defined exactly once, by construction. It is an output of the kernel.
- **Durable array with no handle.** Constructed from the definition alone. The toolchain allocates it, scoped to the kernel. It is for when a durable array is needed inside the kernel but is not a real output.
- **Same type, role by place.** `sg::array` built in the submit scope from a shape is an input; built in the kernel from a definition it is an output.
- **One role per handle per kernel.** Within one kernel a handle is either an input or the backing of one durable array, never both, so an in-place update cannot be written. A slice is a handle like any other, so disjoint slices of one allocation may be an input and an output in the same kernel. A later kernel may back a new result with the same handle; the graph orders it.
- **Sizes must match.** The allocation's size and the constructed array's size must match, and the element types must agree. The host-side code checks this at launch.
- **In-place combine.** `sg::inplace(handle, op)` may take the handle's place when constructing a durable array. The array the definition produces is then reduced element-wise, with `op`, into what the handle already holds. It exists for separate kernels contributing to one handle; see `02-graph-api.md`. The kernel that contributes cannot read that array.
- **Global arrays are for durable results.** They are not how work ordinarily passes values to work; mappings are.

### Reading a durable array: the second way to create a dependency

A durable array can be read or sliced inside a later `parallel_for` body in the same kernel. The read creates a dependency on the groups that produce what is read.

- If the slice is resolvable at launch, the consuming group waits only on the groups that produce that slice. Resolvable means the same as for a mapping: built from the item's group coordinates and launch values through a single id, a range, a strided range or a stencil.
- Otherwise, for example an index computed from data, the consumer waits on the whole producing definition before any of its groups run.

This is the escape hatch for work a group mapping does not fit. The cost ladder: a mapping is precise and can avoid global memory; a durable array with a resolvable slice costs global memory and keeps precise ordering; an unresolvable read costs a full wait.

### Toolchain memory

The user never names memory for global mappings, cycle internals, initial values of a cycle given as definitions, or durable arrays with no handle. The toolchain allocates it, per device, using the whole structure of the graph and of each kernel.

## 9. `evaluate`

- `evaluate` takes a definition whose array is exactly one value and gives a kernel-level scalar. Such a definition is a `parallel_for` of a single group of size one, or a group reduction that yields one value.
- It is a general primitive, usable anywhere at kernel level: to branch with an ordinary `if`, or to end a cycle.
- Branching on it is not special. It compiles to branching code, and the only rules on the two sides of the branch are normal C++ scoping rules.
- It is not a launch value. It cannot shape a range, a group shape or a mapping.
- Cost: it is a kernel-wide scalar, so everything it depends on has finished before anything that branches on it proceeds.

## 10. Cycle and chain

These are two unrelated constructs.

### Cycle

A cycle is a block of normal kernel definitions that repeats until a stop condition the user writes. It is a while loop whose outputs feed the next iteration's inputs.

- `make_cycle` takes one or more initial values and a body. Initial values are definitions or arrays.
- The body's arguments are the current iteration, passed by the toolchain, and then one argument per initial value. The number and types of the initial values fix the body's arguments and what it may return.
- The arguments after the iteration are what the previous iteration returned; on the first iteration they are the initial values.
- A plain return continues and feeds the next iteration. A return wrapped in `cycle::final` ends the cycle. The outputs have the same number and types either way.
- There is no iteration limit and no bounded or unbounded option. A limit is the user branching on the iteration argument; any other exit is a branch on `evaluate`. A cycle that never returns `final` runs indefinitely.
- Anything valid at the top level of a kernel is valid in the body, including nested cycles and chains.
- The cycle yields its final outputs, usable like any other definition.

```cpp
auto [u_def, p_def] = sg::make_cycle({u0, p0}, [=](auto iteration, auto u_prev, auto p_prev) {
  auto u_next = ...;        // definitions depending on u_prev, p_prev
  auto p_next = ...;
  auto residual = ...;      // a definition yielding one value
  if (iteration >= max_iterations || sg::evaluate(residual) < tolerance)
    return sg::cycle::final{u_next, p_next};
  return {u_next, p_next};
});
auto u = sg::array<float, 3>(u_handle, u_def);
auto p = sg::array<float, 3>(p_handle, p_def);
```

### Chain

A chain defines, functionally, a series of dependent steps without writing each step and each step-to-step mapping by hand.

- `make_chain` takes the number of layers and a body.
- The body constructs one layer at a time. It is given the layers constructed so far, with random access and empty for the first layer, and the index of the layer being constructed. It returns that layer of work.
- A layer may depend on any earlier layers, or on none.
- The chain yields its layers, indexable, so any layer can feed later work or become durable.
- The number of layers and everything about a chain's structure is resolvable at launch. The layer index is a launch value inside the body.

## 11. Inside a body

### Functional rules

A body is a pure function of its handle, its mapped arguments, read-only input and durable arrays, and launch values captured by value. It writes nothing outside itself, captures nothing mutable or by reference, touches no globals and calls nothing with side effects. Values are initialised once. There are no loop statements, including in functions it calls.

### Arrays in a body: `group_array`

`group_array` is the only array type that can be constructed inside an item body or a group body. It comes from:

- a slice of an `array`, input or durable;
- a construction primitive: `from_range`, `ones`, `zeros`, `filled`, `mask` and the like;
- an array operation: `reduce`, the scans, `map`, element-wise arithmetic and comparisons, `select`, `transform`;
- a mapped argument, which is handed in.

The same expressions are valid in both kinds of body. They differ in how they are carried out: a group body uses group-wide operations, and an item body is limited to sequential implementations.

### Operations

`reduce`, `inclusive_scan`, `exclusive_scan`, `any_of`, `all_of`, `none_of`, `map`.

- **The data sizes the operation.** None of them takes a size or a length. A reduce runs over the dimension being reduced, a scan over the length of what is scanned.
- **Over a `group_array`** they are array operations, over the whole array or along chosen dimensions.
- **Over a `group_iterator`** they are element-wise across the iterator's elements. Reducing an iterator of two 64x64 arrays with plus gives one 64x64 array; reducing an iterator of floats gives a float.
- `map(indices, arr)` selects elements of an array by an index array. A reorder is the case where the indices are a permutation.
- Operators and comparisons are element-wise, always. Comparisons give boolean arrays. `select(bool_array, a, b)` picks element-wise.
- A reduce's operation must be associative and commutative, and its order is unspecified, as with SYCL's group algorithms.

### `transform`

`transform(arr, [subset,] functor)` is the general element-wise operation. The functor receives an element's index and current value and returns the new value, which is unconditionally the array's element type. The optional subset is given by the index-space primitives or a mask. The result is a new array of the same shape and type; elements not operated on are unchanged. A change of element type is not a transform: it is a new array built from an expression, such as a comparison on the array or on a selection of it.

There is no primitive that replaces one element of an array. That is expressed with the above, for example `old + ones(shape) * mask(id)`.

### `recurse`

`recurse` is the generic form and the escape hatch.

- It takes an initial value and a body. The body receives the step index and the previous step's return, and returns the next value.
- A plain return continues. A return wrapped in `recurse::final` ends it, and that value is what `recurse` yields.
- There is no step count. The user decides when it ends, and must, or it runs indefinitely.
- Any of the other operations can be written with it. It also covers traversal: the carried value can be a position, and the final return the thing found, as in a tree walk or a binary search. No iterator abstraction and no search primitive is needed.

```cpp
auto found = sg::recurse(start_position, [=](auto step_idx, const auto& position) { ...; return ...; });
```

The named operations are preferred where they fit because they carry intent: a reduce says the values combine and the grouping is the toolchain's; a `recurse` says each step needs the last.

### Atomics

Atomics are not exposed to the user. A fully functional language removes the need for them in user code:

- a kernel-wide minimum, maximum, sum or "found" is `reduce`, `any_of`, `all_of` or `evaluate`;
- coordination between threads is ordering, which the toolchain owns;
- many contributions to one destination chosen by data (histograms, binning, deposition, assembly, claiming) are written with `recurse` carrying an array, combined by group reductions; or with producers pulled into place by an index array in later work; or with compositions of these.

The toolchain may use atomics underneath, as one of a primitive's documented lowering strategies (section 13).

## 12. What the toolchain rejects

A message names the user's construct and the resolved numbers, never the lowered code.

**At compile time**

- Any loop statement anywhere in a kernel definition, including in a function called from a body.
- A body that is not functional.
- A body that returns nothing, or more than one type.
- A group-body `parallel_for` with no mapping; a group body given an item-mapped argument; a group body using anything outside the group and array vocabulary.
- An item mapping from a group-body definition.
- A body whose mapped arguments do not match its source maps in number.
- A value from `evaluate`, a cycle's iteration argument, or a `recurse` step index used where a launch value is required.
- A cycle or `recurse` body whose returns do not match its initial values in number or type.
- A `transform` functor that does not return the array's element type.
- Two durable arrays on one handle in one kernel.
- A group-body definition captured as an array when its returned rank differs from its group range's rank and it is not a scalar. The message names both ranks and points to refining through another definition.
- A read, in a contributing kernel, of an array constructed with `inplace`.
- A `wait` that is not uniform across the group.

**At launch**, by the host-side code of the submission, thrown immediately

- An item range that is not an even multiple of its group shape.
- A mapping that names a source group that does not exist, including through an index array.
- An item mapping whose source and consumer group shapes differ.
- An `evaluate` over a definition that is not exactly one value.
- A handle whose allocation size does not match the array constructed on it, or whose element type differs.
- A handle used in two roles in one kernel, or overlapping slices used that way.
- Local memory a native group cannot hold. The message names the mapping that pushed it over; a `global` mapping is the way out.

Where the values involved are constants in the source, a launch check may be reported at compile time.

**Not errors**

- A read of a durable array through a slice that is not resolvable at launch. It is valid and costs a full wait; it is reported.
- A cycle or `recurse` that never finalizes.
- A redundant `wait`.
- A read of another item's element without a `wait`. It is the user's to reason about.

## 13. What the user can rely on, and what they are told

**Guaranteed**

- One kernel definition is one launch.
- The native range and group shape are derived; the user never declares them.
- An `item` mapping passes values item to item, with no memory beyond the item.
- A `local` mapping carries out the source's work inside the consumer's native group, with values in local memory.
- A `global` mapping behaves exactly like `local`, with values in global memory the toolchain allocates.
- A barrier happens where the `wait` rules say.
- A durable array lives in the memory the user named.
- The mapped kind the user wrote is the kind used.

**Free for the toolchain**

- How many times a definition is computed.
- Which native group a logical group lands on.
- How ordering across native groups is achieved inside the launch.
- The order of a reduce.
- Reuse of local and global memory.

**Two layers.** Guaranteed semantics say what a primitive means. Documented lowering lists, per primitive, the strategies it may compile to, so a user who knows device code can reason about it. Naive strategies come first; optimised ones are added once everything is green, without changing any guarantee. Group-wide operations map onto SYCL's group operations; item-level ones are sequential code.

**Reported**, in terms of the user's constructs, obtained from the graph after a run:

- the derived native range and group shape;
- local memory: the bound from the definition, and what was actually needed;
- global memory the toolchain allocated, and for what;
- where barriers and ordering across native groups occurred;
- host syncs from index arrays, and full waits from unresolvable reads;
- which strategy each primitive used.

**Tuning** is written in the kernel: group shapes and mapped kinds. Properties that configure toolchain behaviour are deferred.

## 14. What does not fit in one kernel

A relationship between pieces of work that comes from computed data, and a size that comes from computed data, are kernel boundaries. The graph carries them across: see host work and graph futures in `02-graph-api.md`.

## 15. Details not yet settled

- The exact set of construction primitives beyond `from_range`, `ones`, `zeros`, `filled` and `mask`.
- The boundary behaviours a stencil offers.
- The mapping parameter of a `group_array` constructed inside a body. It has no mapping; where it lives is the toolchain's.
- The form the report takes.
