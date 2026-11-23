# Vocabulary reference

Status: agreed 2026-11-18. What is locked is which names exist and what each means. Signatures are illustrative. All names are in the namespace written `sg`.

Naming conventions: operation names follow SYCL and the standard library where the meaning matches. `define_` is for what a submission is, `make_` for cycle and chain, `create_` for host-side objects. `graph_promise` and `graph_future` are named so they are not taken for the standard ones.

## 1. Host side, outside any submission

| Name | Meaning |
|---|---|
| `graph` | the recorded program |
| `create_mem_handle<T>(size?)` | memory handle; size optional |
| `slice(handle, offset, size)` | a handle onto part of an allocation |
| `free(handle)` | release |
| `create_promise<T>()` giving `graph_promise<T>`; `.future()` giving `graph_future<T>` | host-only value passed between submissions |
| `graph.submit(device, [&](auto gh) {...})` | exactly one piece of work |
| `graph.cycle(...)` | graph cycle: repetition spanning submissions whose stop condition is only known while the graph runs |
| `graph.run()`, `graph.wait()` | issue the graph without blocking; block until done |
| `graph.report()` | what was derived and chosen |

## 2. Submit scope

| Name | Meaning |
|---|---|
| `array<T, N>(handle, shape)` | kernel input, read-only |
| `host_array<T, N>(handle, shape)`, `host_array<const T, N>(handle, shape)` | precise host access: read and write, or read only |
| `get_host_handle(handle)` | conservative host access |
| `future.get()` | launch value from host work |
| `gh.define_kernel([=] {...})` | the submission is a kernel |
| `gh.define_host_work([=] {...})` | the submission is host work |
| `gh.define_interop_work([=](auto ih) {...})` | the submission is interop work |

## 3. Index-space primitives

Not specific to groups. Used to name a mapping's source groups, to slice input and durable arrays, and to slice group arrays.

| Name | Meaning |
|---|---|
| `id` | a single position, one to three dimensions |
| `range{begin, end}` | per dimension |
| `strided_range{begin, end, stride}` | |
| `stencil{offsets..., boundary}` | boundary behaviour stated on the stencil |
| `index_array(array_or_subslice)` | positions from an array; as a mapping's source it costs a host sync |

## 4. Kernel definition body

| Name | Meaning |
|---|---|
| `parallel_for(item_range, group_shape, [mapping,] item_body)` | item-body definition; array shape is the item range |
| `parallel_for(group_range, mapping, group_body)` | group-body definition; a grid of groups; mapping required; no group shape |
| `map_groups([=](auto group_id) { return sources{...}; })` | a mapping: functor of the target group id |
| `source_map<mapping::local \| mapping::item \| mapping::global>(source_definition, positions)` | one mapped input, which is one body argument |
| `array<T, N>(handle, definition)` | durable output |
| `array<T, N>(definition)` | durable, in toolchain memory |
| `array<T, N>(inplace(handle, op), definition)` | element-wise combine across kernels |
| `evaluate(definition)` | kernel-level scalar |
| `make_cycle({initials...}, body)`, `cycle::final{...}` | cycle and its ending return |
| `make_chain(num_layers, body)` | yields the layers, indexable |
| `if` / `else` | on launch values or on `evaluate` |

## 5. Mapped argument types

| Mapping | one source group per target | many source groups per target |
|---|---|---|
| `mapping::item` | `T` | `group_iterator<T>` |
| `mapping::local` | `group_array<T, N, mapping::local>` | `group_iterator<group_array<T, N, mapping::local>>` |
| `mapping::global` | `group_array<T, N, mapping::global>` | `group_iterator<group_array<T, N, mapping::global>>` |

## 6. Arrays inside bodies

`group_array` is the only array type constructable in an item body or a group body.

| Source | Names |
|---|---|
| a slice of an `array` (input or durable) | `arr[i]`, slicing through section 3 |
| construction primitives | `from_range`, `ones`, `zeros`, `filled`, `mask`, and the like |
| array operations | section 7 |
| mapped arguments | handed in; section 5 |

## 7. Operations

The same names at item level and group level. A group body carries them out group-wide; an item body sequentially.

| Name | Meaning |
|---|---|
| `reduce` | combine with an associative, commutative operation |
| `inclusive_scan`, `exclusive_scan` | |
| `any_of`, `all_of`, `none_of` | |
| `map(indices, arr)` | select elements of an array by an index array |
| element-wise operators and comparisons | comparisons give boolean arrays |
| `select(bool_array, a, b)` | element-wise choice |
| `transform(arr, [subset,] [=](auto elidx, auto element) {...})` | general element-wise operation; the functor returns the element type; untouched elements are unchanged |

| Applied to | Meaning |
|---|---|
| `group_array` | array operation, whole array or along dimensions |
| `group_iterator<...>` | element-wise across the iterator's elements |

Names that do not exist: `shuffle` (it is the permutation case of `map`); `with` or any single-element update; a separate group return type.

## 8. Item body

| Name | Meaning |
|---|---|
| `item<N>` | `.id()`, `.group_id()`, `.local_id()`, `.range()`, `.group_shape()` |
| sections 6 and 7 | |
| `recurse(initial, body)`, `recurse::final{...}` | generic sequential form; escape hatch |
| `wait(mapped_arg)` | group barrier where the rules require one |
| returns | one value of one kernel-valid type |

## 9. Group body

| Name | Meaning |
|---|---|
| `group<N>` | `.id()`, `.group_range()` |
| sections 6 and 7, over group arrays and group iterators | |
| returns | the result of its final group operation, an array or a scalar; this sets the shape |

## 10. Cycle, chain and recurse bodies

| Body | Arguments | Returns |
|---|---|---|
| cycle | the iteration, passed by the toolchain; then one per initial value | `{...}` to continue, `cycle::final{...}` to end |
| chain | the layers so far (random access, empty for the first); the layer index | one layer of work |
| recurse | the step index; the previous value | a value to continue, `recurse::final{value}` to end |

## 11. Host work body

| Name | Meaning |
|---|---|
| `host_array` indexing | read; write if non-const |
| `host_handle.realloc(size)`, `host_handle.get_pointer()` | |
| `host_array<T, N>(host_handle, shape)` | may be built anywhere in the call tree |
| `promise.set(value)` | |
| ordinary C++ | loops and mutation are fine here |
