# The revised kernel language: summary for agreement (v3)

This page is all you need to read; delta.md lists what changed since v1. The full record (every kernel with its trace, the mechanisms, the misfit log, the review) is in record.md and kernels.md beside it.

## The language

A kernel tiles its data, declares work tiles (each one or more native work-groups, chosen at construction), and maps work over them. A work tile holds work units; each unit runs a 1-D run of items (native work-items). A unit body is host code that picks tiles; the function given to unit.run is the only device code.

| primitive | meaning |
|---|---|
| kh.source(array) | an input array as elements |
| kh.tile(x, shape) / .count() | x's elements in row-major tiles of shape (extents must divide); the tile count |
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
| kh.mark_local(result) | hold a result in local memory for other units of its work tile |
| unit.run_distributed(inputs..., fn) | on a segmented work set: each segment runs its share; collectives join across segments (global) |
| unit.run_replicated(inputs..., fn) | on a segmented work set: every segment runs the whole run; collectives stay in the group |
| kh.mark_segmented_local(result) | a segmented result in local memory: the whole result in every segment if replicated, each segment's share if distributed |
| kh.repeat(n, start, fn) | one structure applied n times; n a bound value |
| kh.evaluate(elements) | assign to a result (inplace results combine) |
| unit.tile() / unit.id() | the unit's work tile / its position in the tile |
| unit.run(inputs..., fn(item, ...)) | fn on every item with its element of each 1-D input |
| sg::flatten(tile) / sg::slice(tile, offset, extent) | a tile as 1-D / part of a tile |
| sg::reduce, sg::inclusive_scan(item, v, op) | combined over the run / up to this item |
| sg::tile_bin(item, v, bins, key) | counts per bin over the run; a run of bins values |

Rules:
- A result pulled by the work tile that made it stays a value, in local memory (if marked), or is recomputed; pulled by any other work tile it goes through global memory.
- One launch has one native group size, the largest; smaller work tiles leave work-items unused.
- Pulls form a dependency graph between tiles; a tile chosen from kernel data depends on every tile of its source set. The toolchain orders the native groups so each tile waits only on groups ahead of it; if no order exists, the builder rejects the kernel naming the cycle, and you break it with another work set. So data-chosen pulls only go from one work set to another.
- Over the device's group cap, at construction: build different work tiles, or segment the run. A segmented work set is its own type: its units run_distributed (each segment its share; collectives joined across segments with a global dependency) or run_replicated (every segment the whole run; collectives stay in the group), and its results are marked local only with kh.mark_segmented_local. run_replicated is rejected if its input is a distributed result of its own set. The data tiling never changes.

## Four kernels

Common: `sum_items(item, v)` is `sg::reduce(item, v, plus)`; `pass_items` returns v; `dot_items(item, a, b)` is `sg::reduce(item, a * b, plus)`; `sum_tiles(kh, work, tiles)` maps work over tiles[unit.tile()], flattened, with sum_items.

**Global reduction, 64 in fours: levels reuse level 1's groups.**
```cpp
auto x           = kh.source(x_in);                               // 64 elements
auto x_tiles     = kh.tile(x, {4});                               // 16 tiles of 4
auto work1       = kh.work_tiles({16}, {1}, 4);                   // 16 groups of 1 unit x 4 items
auto l1          = sum_tiles(kh, work1, x_tiles);                 // 16 result tiles of {1}
auto l1_tiles    = kh.retile(l1, {4});                            // 4 tiles of 4
auto work2       = kh.slice_work(work1, {3}, {4}, {4}, {1}, 4);   // work1 tiles 3, 7, 11, 15
auto l2          = sum_tiles(kh, work2, l1_tiles);                // 4 result tiles of {1}
auto l2_tiles    = kh.retile(l2, {4});                            // 1 tile of 4
auto work3       = kh.slice_work(work1, {15}, {1}, {1}, {1}, 4);  // work1 tile 15
auto l3          = sum_tiles(kh, work3, l2_tiles);                // 1 result tile of {1}
auto total_elems = kh.join(l3);
total = kh.evaluate(total_elems);
```
16 groups of 4 work-items. Each level-2 tile keeps the level-1 sum it made itself as a value; the other three come through global memory. Traced: 2016.

**Gather from values computed in the kernel: a data-chosen pull.**
```cpp
auto double_items = [=](auto item, auto v) { return 2 * v; };
auto s         = kh.source(s_in);                                 // 256 elements
auto s_tiles   = kh.tile(s, {64});                                // 4 tiles of 64
auto work_src  = kh.work_tiles({4}, {1}, 64);                     // 4 groups
auto src = kh.map_work(work_src, [=](auto unit) {
  auto s_tile = s_tiles[unit.tile()];
  auto s_run  = sg::flatten(s_tile);
  return unit.run(s_run, double_items);
});                                                               // 4 result tiles of {1, 64}
auto src_rows  = kh.retile(src, {64});
auto src_elems = kh.join(src_rows);                               // 256 elements
auto idx       = kh.source(idx_in);
auto idx_tiles = kh.tile(idx, {64});                              // 4 index tiles of 64
auto gathered  = kh.index_map(idx_tiles, sg::fill{0}, src_elems); // chosen by data
auto work_y    = kh.work_tiles({4}, {1}, 64);                     // a later set
auto y = kh.map_work(work_y, [=](auto unit) {
  auto g_tile = gathered[unit.tile()];
  auto g_run  = sg::flatten(g_tile);
  return unit.run(g_run, pass_items);
});
auto y_rows  = kh.retile(y, {64});
auto y_elems = kh.join(y_rows);
y_out = kh.evaluate(y_elems);
```
8 groups. The gather is its own later set, so it may pull any src tile; each item waits on the tile its index chose. Traced: y[1] = 14, y[37] = 6.

**Matrix product, 128 x 128: 16 x 16 outputs, segmented run, segmented local stages.**
```cpp
auto a_bands = kh.tile(A, {16, 128});                             // 8 x 1 tiles
auto b_bands = kh.tile(B, {128, 16});                             // 1 x 8 tiles
auto run     = sg::segment_items(32, 4);                          // 128 items in 32 segments of 4
auto work    = kh.work_tiles({8, 8}, {16, 16}, run);              // 2048 groups of 16 x 16 x 4
auto a_work  = kh.slice_work(work, {0, 0}, {8, 8}, {1, 1}, {16}, run);
auto a_stage = kh.map_work(a_work, [=](auto unit) {
  auto t      = unit.tile();
  auto u      = unit.id();
  auto a_band = a_bands[{t[0], 0}];
  auto a_row  = sg::slice(a_band, {u[0], 0}, {1, 128});
  auto a_run  = sg::flatten(a_row);
  return unit.run_distributed(a_run, pass_items);
});                                                               // this tile's 16 rows of A
kh.mark_segmented_local(a_stage);
auto b_work  = kh.slice_work(work, {0, 0}, {8, 8}, {1, 1}, {16}, run);
auto b_stage = kh.map_work(b_work, [=](auto unit) {
  auto t      = unit.tile();
  auto u      = unit.id();
  auto b_band = b_bands[{0, t[1]}];
  auto b_col  = sg::slice(b_band, {0, u[0]}, {128, 1});
  auto b_run  = sg::flatten(b_col);
  return unit.run_distributed(b_run, pass_items);
});                                                               // this tile's 16 columns of B
kh.mark_segmented_local(b_stage);
auto C_tiles = kh.map_work(work, [=](auto unit) {
  auto t     = unit.tile();
  auto u     = unit.id();
  auto my_a  = a_stage[t];
  auto my_b  = b_stage[t];
  auto a_row = sg::slice(my_a, {u[0], 0}, {1, 128});
  auto b_row = sg::slice(my_b, {u[1], 0}, {1, 128});
  auto a_run = sg::flatten(a_row);
  auto b_run = sg::flatten(b_row);
  return unit.run_distributed(a_run, b_run, dot_items);
});                                                               // 8 x 8 result tiles of {16, 16}
auto C_elems = kh.join(C_tiles);
C_out = kh.evaluate(C_elems);
```

| value | lives in | because |
|---|---|---|
| A, B | global | inputs |
| a_stage, b_stage | segmented local | other units of the same segment read them; marked |
| products in a run | registers | each item's own |
| partial sums per segment | global | the reduce crosses segments |
| C | result memory | evaluated |

Item p of a product run and item p of a stage run fall in the same segment, so the stages never leave local memory. Traced: C[3][5] = 1920, C[127][127] = 2064512.

**Row sums at a cap of 256: a segmented run, no join code.**
```cpp
auto rows = kh.tile(x, {1, 1024});                                // 64 x 1 tiles
auto run  = sg::segment_items(4, 256);                            // 1024 items in 4 segments of 256
auto work = kh.work_tiles({64}, {1}, run);                        // 256 groups of 1 x 256
auto sums = kh.map_work(work, [=](auto unit) {
  auto t   = unit.tile();
  auto row = rows[{t[0], 0}];
  auto r   = sg::flatten(row);
  return unit.run_distributed(r, sum_items);
});                                                               // 64 result tiles of {1}
auto sums_elems = kh.join(sums);
y_out = kh.evaluate(sums_elems);
```
The same code as at cap 1024 except the run argument; the reduce is joined across each row's 4 segments. Traced: y[0] = 523776, y[63] = 588288.

## Decisions for you (defaults marked)

1. Within one work set, may a map's runs be shorter than the set's run, the other items idle (the scan takes each tile's total this way)? (a, default) yes; (b) no, a shorter run needs its own work set.
2. A tile size that does not divide its array: (a, default) a builder error naming the node; (b) a partial last tile filled by an oob policy.
3. Element-wise work with nothing shared is written as runs on a work set. (a, default) only that; (b) also keep kh.map as a plain range launch for kernels with no work sets.
4. Sharing an input tile in local memory is a copy pass on a slice of the same work set, marked local (the matrix product). (a, default) as written; (b) a primitive that marks an input tile local for a work set.

## What changed since the early check

- Work units hold runs of items (names swapped); a work tile is a fixed set of native groups chosen at construction.
- Groups are reused by slicing a work set (kh.slice_work), with any shape that fits.
- Runs are an extra dimension of a result tile; a run may be segmented across native groups (sg::segment_items), giving a segmented work set whose units run_distributed or run_replicated.
- The scan is 16 groups of 64; only the gathered totals cross tiles.
- Pulls form a tile dependency graph that the toolchain orders; a kernel with no valid order is rejected when the builder runs; data-chosen pulls only cross work sets.
- New: kh.stencil, sg::tile_bin as a run collective, sg::segment_items, kh.mark_segmented_local; unit.id is used by the stencils, the matrix product and interpolation.
- Element-wise chains are runs on a work set; no plain kh.map.
- All 14 kernels are written and traced (kernels.md).
