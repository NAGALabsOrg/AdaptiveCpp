# The kernel language: specification

Status: the agreed language, summary v6 of the language pass (2026-11-23) with the decisions D2-D17 and the rule ledger W1-W15, B1-B2. This document states the kernel side; 02-graph-api.md states the graph around a kernel and 03-vocabulary.md defines every term.

## 1. Frame

A kernel is built by host code that calls the kernel handle `kh` (B1). The builder runs when the kernel is submitted and records a kernel graph: which arrays are read and written, which views cut them into tiles, which work sets exist, which maps run over them. Which units and links exist is structure and is fixed when the builder returns. Counts and sizes are bound values (D2): they are read by the host just before each launch and never change structure. One kernel definition is one launch (B1).

A kernel tiles its data, declares work tiles (each one or more native work-groups, chosen at construction), and maps work over them. A work tile holds work units; each unit runs a 1-D run of items (native work-items). A unit body is host code that picks tiles; the function given to `unit.run` is the only device code. A kernel may instead be a work-item kernel: plain range launches with no groups, each item reading inputs and only its own earlier results.

## 2. The primitives

| primitive | meaning |
|---|---|
| kh.source(array) | an input array as elements (B1) |
| kh.tile(x, shape) / .count() | x's elements in row-major tiles of shape; a partial last tile goes to an oob policy; the tile count |
| kh.retile(t, shape) | t's elements, in tile order, regrouped |
| kh.join(t) | tiles stitched over their grid (equal ranks) |
| kh.extend_tiles(t, shape, sg::repeat) | each tile broadcast to shape |
| kh.define_tiles(n, shape, fn(t, s)) | index tiles from a host function |
| kh.stencil(offsets, roots) | index tiles: each root plus each offset |
| kh.index_map(map, oob, elements) | each index replaced by its element; out of range goes to oob |
| kh.work_tiles(grid, tile_shape, run) | one native group per grid point, shape tile_shape ++ run (at most 3-D) |
| sg::segment_items(segments, items) | a run of segments x items, each segment its own native group |
| kh.slice_work(work, offset, extent, stride, tile_shape, run) | reuse a subset of a work set's groups, possibly with a new shape |
| kh.map_work(work, body(unit)) | body per unit on the host; one result tile per work tile |
| kh.work_items(range) / kh.map_items(items, body(item)) | a work-item set (plain range launch, no groups) / body per item on the host; a result array |
| item.id() / item.compute(inputs..., fn) | the item's index / fn on its inputs (device; no collectives) |
| kh.mark_local(result) / kh.mark_local<sg::segmented>(result) / kh.mark_local(array_tiles, work) | a result in local memory for other units of its work tile (segmented form required after run_distributed); array tiles in local memory for every work tile of work that pulls them |
| unit.run_distributed(inputs..., fn) | on a segmented work set: each segment runs its share; collectives join across segments (global) |
| unit.run_replicated(inputs..., fn) | on a segmented work set: every segment runs the whole run; collectives stay in the group |
| kh.repeat(n, start, fn) | one structure applied n times; n a bound value |
| kh.evaluate(elements) | assign to a result (inplace results combine) |
| unit.tile() / unit.id() | the unit's work tile / its position in the tile |
| unit.run(inputs..., fn(item, ...)) | fn on every item with its element of each 1-D input |
| sg::flatten(tile) / sg::slice(tile, offset, extent) | a tile as 1-D / part of a tile |
| sg::reduce, sg::inclusive_scan(item, v, op) | combined over the run / up to this item |
| sg::tile_bin(item, v, bins, key) | counts per bin over the run; a run of bins values |

In the builder these are written `kh.source(handle, sg::shape<N>({...}))` and `kh.evaluate(handle, elements)` (or `kh.evaluate(sg::inplace(handle, op, init), elements)`); 02-graph-api.md gives the forms. The oob policies are `sg::fill{v}` and `sg::wrap{}`.

## 3. Data: elements, tiles and views

Elements and tiles are explicit types (D3). `kh.tile` turns elements into tiles, a map over tiles returns tiles, `kh.retile` regroups tiles, `kh.join` is the only way back to elements, and `kh.evaluate` takes elements. A tile size is logical and follows the problem, never the device (M4).

- All sizes are bound values and ranks are in types; shape errors are reported when the builder runs, naming the node (D2).
- A tile size need not divide its array (W15): `kh.tile(x, shape, oob)` makes a partial last tile whose missing positions go to the oob policy, and a join of tiles derived from it trims back to the source extent.
- An index map is tiles of indices applied with `kh.index_map(map, oob, elements)` (D4). `kh.define_tiles` computes its indices on the host; `kh.stencil` builds the index tiles of each root plus each offset; indices outside the source go to the oob policy and there is no separate sentinel.
- A view is a descriptor: strides, offsets, extents, an index-table reference and the oob policy. The toolchain interprets views from the descriptor on the device; a view never holds values.
- `kh.join` requires tiles whose grid rank equals their tile rank; others are retiled first (rejection 12).
- `sg::flatten(tile)` is the tile in row-major order as a 1-D run; `sg::slice(tile, offset, extent)` is part of a tile (W6).
- `sg::tile_bin(item, v, bins, key)` bins within a run and returns a run of bins counts (D5).

## 4. Work: tiles, units, runs, items

A work tile is one or more native work-groups, chosen at construction and fixed (W1); on a segmented work set there is one native group per segment. Each `kh.work_tiles` call is an explicit set of native groups; another set, of any shape, is a parallel set. Using the same work set again uses the same groups; slicing a work set with `kh.slice_work` reuses a subset of its groups, possibly for a sub-problem of a different shape (W1, D6).

- The run is an extra dimension: the native group shape is tile shape ++ run, at most 3-D; a 3-D tile has a run of one item and takes no run argument (W9).
- One launch has one native group size, the largest in the kernel; smaller work tiles leave native work-items unused (W8).
- Work units in a tile are independent, which makes work sliceable (M2). A unit's parallelism is one-dimensional: a run of items; its result is one value (a reduce) or a run (element-wise, scan, binning) (M3).
- A map may use a run shorter than its work set was built with; the result tile's run dimension follows the run used (W15). A run longer than the work set's run is rejected (rejection 14).
- Over the device's group cap the user builds different work tiles, or segments the run with `sg::segment_items`; the data tiling never changes (M5, D10). Fitting the cap happens only at construction.
- A segmented work set is its own type (D11, D12, W12): one native group per segment, each of tile shape x items per segment. Its units use `unit.run_distributed` (collectives are joined across the segments with a global dependency) or `unit.run_replicated` (every segment runs the whole run, item k taking positions k, k + items, ...; collectives stay in the group). `unit.run` does not exist on it.
- A work-item set (`kh.work_items(range)`, one to three dimensions) is a plain range launch with no groups (W13, D13, D14). `kh.map_items` runs its body once per kernel on the host to select inputs by `item.id()`; `item.compute(inputs..., fn)` applies fn on the device with no collectives. An item takes elements of arrays and array tiles and may take another work-item set's result only at its own index; its result is a result array. A kernel uses work tiles or work items, never both; result arrays never feed work tiles and result tiles never feed work items.

## 5. Maps and selection

`kh.map_work(work, body)` runs `body(unit)` on the host once per unit to select tiles (W4). The body returns `unit.run(inputs..., fn)`, `unit.run_distributed(...)` or `unit.run_replicated(...)`; the selection (chosen tile or window of each input, the run length, the function) is recorded and bound to the launch, and recomputed only when a launch value it depends on changes. The inputs of a run are 1-D views of one length. The function receives the item and one element of each input; what it returns decides the result:

- a plain value per item: a result tile of tile shape ++ run length;
- `sg::reduce(item, v, op)`: one value per unit, a result tile of tile shape;
- `sg::inclusive_scan(item, v, op)`: a value per item;
- `sg::tile_bin(item, v, bins, key)`: a run of bins counts.

`sg::reduce` and `sg::inclusive_scan` work for any trivially copyable element and any associative operation, in item order, so operations that are not commutative are correct; no identity is required. A result tile that is pulled becomes a dependency (W11, section 7).

## 6. Backing and marks

The user decides memory backing through this structure (M6). Everything is lazy unless marked or not representable lazily (W3). Backing follows who reads a value (W2, D6):

| value | lives in |
|---|---|
| a result pulled only by the unit that made it | the unit's own items (private scratch) |
| an element-wise result read by other units of the same work tile | recomputed inline, unless marked local |
| a collective result read by other units of the same work tile; a marked result | local memory |
| a result read by another work tile or another work set | global memory |
| an evaluated result | result memory |

`kh.mark_local(result)` holds a result in local memory for other units of its work tile. On a segmented work set the plain form applies to results of `run_replicated` (the whole result in every segment) and `kh.mark_local<sg::segmented>(result)` is required for results of `run_distributed` (each segment its share); each form is an error on the other kind of result (W12, D12). `kh.mark_local(array_tiles, work)` marks tiles of input arrays, including index-map views over inputs, and only those: every work tile of `work` (every segment on a segmented set) that pulls them gets its own copy in local memory, populated by its own native work-items (W14, D13).

In-place results combine: contributors on one handle run in parallel and each element is combined atomically with the named operation after the initializer (B2). Elements of 8 and 16 bits are combined through the containing 32-bit word.

`kh.repeat(n, start, fn)` records fn once and applies it n times, n a bound value (W7); counts derived from `prev` inside fn are bound per iteration, so a new depth never changes the structure.

## 7. Ordering inside a launch

Pulls form a dependency graph between tiles (W11, D9): an edge from each pulled tile to the tile that pulls it; a pull the host cannot see (a tile chosen from kernel data) is an edge from every tile of the source work set, but at run time the pulling item waits only on the tile its index chose (W5, D8). Segments of a distributed run add edges from earlier segments. Each native group runs its tasks (its tile of a set, then any slice tasks) in map order. The toolchain orders the native groups so that every edge comes from an earlier group or an earlier task of the same group; if no order exists, the builder rejects the kernel, naming the tiles on the cycle (rejection 3), and the user breaks it with another work set. So data-chosen pulls only ever go from one work set to another.

## 8. The rules

- A result pulled by the work tile that made it stays a value, in local memory (if marked), or is recomputed; pulled by any other work tile it goes through global memory.
- One launch has one native group size, the largest; smaller work tiles leave work-items unused.
- Pulls form a dependency graph between tiles; a tile chosen from kernel data depends on every tile of its source set. The toolchain orders the native groups so each tile waits only on groups ahead of it; if no order exists, the builder rejects the kernel naming the cycle, and you break it with another work set. So data-chosen pulls only go from one work set to another.
- Over the device's group cap, at construction: build different work tiles, or segment the run. A segmented work set is its own type: its units run_distributed (each segment its share; collectives joined across segments with a global dependency) or run_replicated (every segment the whole run; collectives stay in the group), and results of run_distributed are marked local only as kh.mark_local<sg::segmented> (the plain form is an error there, and the segmented form is an error anywhere else). run_replicated is rejected if its input is a distributed result of its own set. The data tiling never changes.
- A kernel uses work tiles or work items, never both. A work item takes inputs (elements or array tiles) and only its own earlier results (same index); result arrays never feed work tiles.

## 9. Device functions

The function given to `unit.run`, `unit.run_distributed`, `unit.run_replicated` and `item.compute` is the only device code. It and everything it calls must be pure and loop-free:

- no loop of any kind: `for`, `while`, `do`, range-for;
- no goto;
- no recursion, direct or mutual;
- no write to anything but its own locals (no global, no write through a pointer, no increment or compound assignment of a captured variable);
- no call through a function pointer, `std::function` or a virtual function, no call to a function without a visible definition, no mutable or by-reference capture, no non-const static variable;
- captures are trivially copyable, so a function is captured as bytes.

The compiler walks the syntax tree from each device entry through every function it calls and reports each offence as an error naming the construct kind (`loop (for)`, `loop (while)`, `loop (do)`, `loop (range-for)`, `goto`, `recursion`, `write`), the function it is in, its source location and the call chain from the entry. Library code of `sg` and the host builder code are not reported. Loops inside the library collectives are the toolchain's own.

## 10. What the toolchain rejects

Every rejected construct raises a build error whose text is `<construct> at <file>:<line>: <explanation> (name = value, ...)`: the construct, its source location and its numbers in one fixed form, raised by the unit that owns the rule. The catalogue is exactly:

| number | construct | rejection |
|---|---|---|
| 1 | kh.work_tiles | work tile above the device cap |
| 2 | kh.work_tiles | native shape above 3-D or a 3-D tile with a run |
| 3 | kh.index_map | tile dependency cycle |
| 4 | kh.mark_local | plain mark_local on a distributed result |
| 5 | kh.mark_local<sg::segmented> | segmented mark_local on a non-distributed result |
| 6 | unit.run_replicated | run_replicated on a distributed result of its own set |
| 7 | kh.map_items | work tiles and work items in one kernel |
| 8 | launch_value | launch value used as structure |
| 9 | kh.mark_local | mark_local of array tiles given non-array tiles |
| 10 | kh.map_items | work item reading a result array at another index |
| 11 | kh.map_work | result arrays fed to work tiles or result tiles fed to work items |
| 12 | kh.join | join with grid rank differing from tile rank |
| 13 | kh.slice_work | slice_work outside its work set |
| 14 | unit.run | run input longer than the run |
| 15 | sg::interop | submit<sg::interop> |

Rejection 8 is a compile error naming `launch_value<T>` and structure; the others are raised when the builder runs or when the first run prepares the kernel.

## 11. The rule ledger

| id | rule |
|---|---|
| B1 | A kernel is built by host code calling the kernel handle; one kernel definition is one launch; which units and links exist is structure, counts and sizes are bound values. |
| B2 | In-place results: contributors on one handle run in parallel and each element is combined atomically with the named operation and initializer. |
| M2 | Work units in a tile are independent; this makes work sliceable. |
| M3 | A work unit's parallelism is a run of items; its result is one value or a run. |
| M4 | Data is tiled by the problem, never by the device. |
| M5 | A work tile maps to native work-groups chosen at construction; over the cap, build different work tiles or segment the run. |
| M6 | The user decides memory backing through this structure. |
| W1 | A work tile is a native work-group, fixed; each kh.work_tiles call is a set of groups; slicing reuses a subset. |
| W2 | Backing: a value held by the unit, in local memory, or recomputed when pulled by the work tile that made it; global memory otherwise. |
| W3 | Everything is lazy unless marked or not representable lazily. |
| W4 | The map_work body is host code that selects whole tiles; the function given to unit.run is the only device code. |
| W5 | Tiles chosen by data produced in the kernel are known at run time; the wait is on the chosen tile. |
| W6 | Unit bodies can select or slice part of a tile. |
| W7 | kh.repeat repeats one structure n times, n a bound value. |
| W8 | One launch has one native group size, the largest in the kernel. |
| W9 | The run is an extra dimension: native shape is tile shape ++ run, at most 3-D. |
| W10 | What users share is how they think about the problem; the language does not enforce it. |
| W11 | Pulls form a tile dependency graph; the toolchain orders the native groups or rejects the kernel naming the cycle. |
| W12 | A segmented work set is its own type with run_distributed and run_replicated; marks follow the run mode. |
| W13 | A kernel uses work tiles or work items, never both; work items read their own earlier results only. |
| W14 | Array tiles, and only array tiles, can be marked local for a work set. |
| W15 | A tile size need not divide its array; a map may use a shorter run than its work set. |

## 12. The decisions

| id | decision |
|---|---|
| D2 | All sizes are runtime, bound values; ranks are in types; shape errors name the node when the builder runs. |
| D3 | Elements and tiles are explicit types: tile, retile, join, extend_tiles; evaluate takes elements. |
| D4 | Index maps: kh.index_map(map, oob, elements) is the only application; define_tiles and stencil build maps; out-of-range goes to oob. |
| D5 | sg::tile_bin bins within a tile and returns counts per bin. |
| D6 | Work tiles are native work-groups; everything is lazy by default; the map_work body is host code. |
| D7 | Work units hold runs of items; a work tile is never moved between groups; the run is an extra dimension. |
| D8 | Cross-tile results have done flags; a data-chosen pull waits on the chosen tile. |
| D9 | Ordering is a graph problem at tile granularity; a kernel with no valid order is rejected naming the cycle. |
| D10 | Fitting the cap happens at construction: other work tiles or sg::segment_items. |
| D11 | A segmented work set is its own type with run_distributed and run_replicated. |
| D12 | One marking API: kh.mark_local<Mode>; sg::segmented is required after run_distributed. |
| D13 | Work-item sets, oob tiles, array tiles marked local for a work set, shorter runs. |
| D14 | A work item's result maps only to the same work item that produced it. |
| D15 | The submit lambda is the kernel builder; a graph is built once. |
| D16 | Any work may produce a launch value; launch values are bound values and never structure. |
| D17 | The deferred host read is kh.launch_value(handle); submit takes the work kind as a template argument. |
