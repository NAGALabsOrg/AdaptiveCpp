# Vocabulary

The terms of the kernel language and of the graph around it, with the names of summary v6. 01-kernel-language.md specifies the language; 02-graph-api.md specifies the graph.

## 1. Terms of the language

### work tile
One native work-group (W1). A work tile is fixed: it never moves between groups. Its native shape is the tile shape followed by the run, at most 3-D.

### work set
The work tiles made by one `kh.work_tiles` call, or a slice of one made by `kh.slice_work`. A kernel has any number of work sets; another set, of any shape, is a parallel set of groups.

### unit
A work unit: an independent producer of results inside a work tile (M2). Its body is host code that runs once per unit when the kernel is prepared, selects tiles and returns `unit.run`, `unit.run_distributed` or `unit.run_replicated`. `unit.tile()` is its work tile and `unit.id()` its position in the tile.

### run
A unit's one-dimensional parallel computation over its items. The run of a work set is its last dimension; a map may use a shorter run than the work set was built with.

### item
One native work-item of a unit's run, or one native work-item of a work-item set. `item.id()` is its index; `item.compute(inputs..., fn)` applies a device function on the inputs of the item.

### segment
A part of a run that is its own native group: `sg::segment_items(segments, items)` makes a run of segments x items.

### segmented work set
A work set whose run is segmented. It is its own type: its units run `unit.run_distributed` (each segment its share, collectives joined across segments with a global dependency) or `unit.run_replicated` (every segment the whole run, collectives stay in the group), and have no `unit.run`.

### work-item set
A plain range launch made by `kh.work_items(range)`: no groups, no tickets. Its result is a result array; an item reads inputs and only its own earlier results.

### tile
A block of elements of a fixed shape cut from an array or a view: `kh.tile`, `kh.retile`, `kh.extend_tiles`, and the result tiles of `kh.map_work`. Tile sizes follow the problem, never the device (M4).

### view
A descriptor that names values without holding them: tiles, retiles, joins, extensions, index tiles, index maps, `sg::flatten`, `sg::slice`. It is a set of strides, offsets, extents, an index-table reference and an oob policy, interpreted on the device from the descriptor.

### oob policy
What a position outside the source yields: `sg::fill{v}` gives v and `sg::wrap{}` wraps around. Partial last tiles, stencil edges and out-of-range indices all go to it.

### array tile
A tile of an input array, including an index-map view over an input. Array tiles, and only array tiles, can be marked local for a work set with `kh.mark_local(array_tiles, work)` (W14).

### selection
What a unit body picked: the chosen tile or window of each input, the run length and the function of the run. Selections are plain data bound to the launch and recomputed only when a launch value they depend on changes.

### launch value
A number read from a handle by the host just before a node launches, on every run: `kh.launch_value(handle)` returns `sg::launch_value<T>` (D17). It is a bound value.

### bound value
A number that sizes a kernel without shaping it: handle sizes, counts, grid extents, work-item ranges, repeat counts and slice extents. Bound values are expressions of launch values and constants, evaluated by the host before each launch.

### structure
What is fixed when the builder returns: which views, work sets, maps and links exist, tile shapes and run lengths. A launch value is never structure.

### ticket
The number a native group claims with a 32-bit atomic when it starts. The group executes the work tile that holds that ticket in the order of the kernel, so that every wait goes to a lower ticket or to an earlier task of the same group.

### tile dependency graph
The graph over (work set, tile, task) with an edge from each pulled tile to the tile that pulls it; a tile chosen from kernel data has an edge from every tile of its source set; a distributed run adds edges from earlier segments (W11). The host orders the native groups by it or rejects the kernel naming the cycle.

### backing
Where a value lives: private scratch of the unit that made it, recomputed inline, local memory, global memory or result memory (W2, W3, W14). The user decides it through the structure of the kernel and the marks (M6).

## 2. Terms of the graph

- **graph**: `sg::graph`, owner of handles and submissions; built once, run many times.
- **handle**: `sg::mem_handle<T>` made by `g.create_mem_handle<T>()`; its storage comes from the pool of the graph, which grows with need.
- **kernel handle**: `sg::kernel_handle`, the argument of a builder; `kh`.
- **host handle**: `sg::host_handle`, the argument of host work; it declares the arrays the work touches.
- **role**: the way a submission uses a handle: input (`kh.source`), result (`kh.evaluate`), in-place result (`sg::inplace`), host read or host write.
- **work kind**: the template argument of `submit`: a device kernel by default, `sg::host`, `sg::interop`.
- **shape**: `sg::shape<N>`, the extents of an array declared by a role; extents may be bound values.
- **in-place result**: a result that combines the contributions of several kernels element by element with a named operation (B2).

## 3. Names

| name | where |
|---|---|
| kh.source, kh.evaluate, kh.launch_value | roles and launch values on the kernel handle |
| kh.tile, kh.retile, kh.join, kh.extend_tiles | tiles |
| kh.define_tiles, kh.stencil, kh.index_map | index tiles and index maps (D4) |
| kh.work_tiles, kh.slice_work, kh.map_work, sg::segment_items | work sets and maps |
| kh.work_items, kh.map_items, item.id, item.compute | work-item sets |
| kh.mark_local, kh.repeat | marks and repetition |
| unit.tile, unit.id, unit.run, unit.run_distributed, unit.run_replicated | units |
| sg::flatten, sg::slice, sg::reduce, sg::inclusive_scan, sg::tile_bin | views and collectives |
| sg::fill, sg::wrap, sg::repeat | policies |
| sg::graph, sg::host, sg::interop, sg::inplace, sg::shape | graph |
| sg::build_error | a rejection of the language |
