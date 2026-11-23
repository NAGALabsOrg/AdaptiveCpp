# The revised kernel language: summary for agreement

This page is all you need to read. The full record (every kernel with its trace, the mechanisms, the misfit log, the review) is in record.md and kernels.md beside it.

## The language

A kernel tiles its data, declares work tiles (each one a native work-group), and maps work over them. A work tile holds work units; each unit runs a 1-D run of items (native work-items). A unit body is host code that picks tiles; the function given to unit.run is the only device code.

| primitive | meaning |
|---|---|
| kh.source(array) | an input array as elements |
| kh.tile(x, shape) / .count() | x's elements in row-major tiles of shape (extents must divide); the tile count |
| kh.retile(t, shape) | t's elements, in tile order, regrouped |
| kh.join(t) | tiles stitched over their grid (equal ranks) |
| kh.extend_tiles(t, shape, sg::repeat) | each tile broadcast to shape |
| kh.define_tiles(n, shape, fn(t, s)) | index tiles from a host function |
| kh.stencil(offsets, roots) | index tiles: each root plus each offset |
| kh.index_map(map, oob, elements) | each index replaced by its element; out of range goes to oob (sg::fill, sg::wrap) |
| kh.work_tiles(grid, tile_shape, run) | one native group per grid point, shape tile_shape ++ run (at most 3-D) |
| kh.slice_work(work, offset, extent, stride, tile_shape, run) | reuse a subset of a work set's groups, possibly with a new shape |
| kh.map_work(work, body(unit)) | body per unit on the host; one result tile per work tile |
| kh.split_runs(result, pieces) | fit the cap: cut every run into pieces, one partial per piece |
| kh.mark_local(result) | hold a result in local memory for other units of its work tile |
| kh.repeat(n, start, fn) | one structure applied n times; n a bound value |
| kh.evaluate(elements) | assign to a result (inplace results combine) |
| unit.tile() / unit.id() | the unit's work tile / its position in the tile |
| unit.run(inputs..., fn(item, ...)) | fn on every item with its element of each 1-D input |
| sg::flatten(tile) / sg::slice(tile, offset, extent) | a tile as 1-D / part of a tile |
| sg::reduce, sg::inclusive_scan(item, v, op) | combined over the run / up to this item |
| sg::tile_bin(item, v, bins, key) | counts per bin over the run; a run of bins values |

Rules:
- A result tile pulled by the work tile that made it stays a value, in local memory (if marked), or is recomputed; pulled by any other work tile it goes through global memory.
- One launch has one native group size, the largest; smaller work tiles leave work-items unused.
- Pulls form a dependency graph between tiles; a tile chosen from kernel data depends on every tile of its source set. The toolchain orders the native groups so each tile waits only on groups ahead of it; if no order exists, the builder rejects the kernel naming the cycle, and you break it with another work set. So data-chosen pulls only go from one work set to another.
- Over the device's group cap: retile the work, or split the runs and join the pieces. The data tiling never changes.

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
16 groups of 4 work-items. Each level-2 tile pulls only lower tiles and its own; the level-1 sums it made itself stay values, the other three go through global memory. Traced: 2016.

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
8 groups. The gather is a later set, so it may pull any src tile; each item waits on the tile its index chose. Traced: y[1] = 14, y[37] = 6.

**Matrix product, 128 x 128: staged in local memory, retiled to fit the cap.**
```cpp
auto a_bands = kh.tile(A, {16, 128});                             // 8 x 1 tiles
auto b_bands = kh.tile(B, {128, 16});                             // 1 x 8 tiles
auto work    = kh.work_tiles({64, 32}, {2, 4}, 128);              // 2 x 4 outputs x 128 = 1024 per group
auto a_work  = kh.slice_work(work, {0, 0}, {64, 32}, {1, 1}, {2}, 128);
auto a_stage = kh.map_work(a_work, [=](auto unit) {
  auto t      = unit.tile();
  auto u      = unit.id();
  auto row    = 2 * t[0] + u[0];
  auto a_band = a_bands[{row / 16, 0}];
  auto a_row  = sg::slice(a_band, {row % 16, 0}, {1, 128});
  auto a_run  = sg::flatten(a_row);
  return unit.run(a_run, pass_items);
});                                                               // this tile's 2 rows of A
kh.mark_local(a_stage);
auto b_work  = kh.slice_work(work, {0, 0}, {64, 32}, {1, 1}, {4}, 128);
auto b_stage = kh.map_work(b_work, [=](auto unit) {
  auto t      = unit.tile();
  auto u      = unit.id();
  auto col    = 4 * t[1] + u[0];
  auto b_band = b_bands[{0, col / 16}];
  auto b_col  = sg::slice(b_band, {0, col % 16}, {128, 1});
  auto b_run  = sg::flatten(b_col);
  return unit.run(b_run, pass_items);
});                                                               // this tile's 4 columns of B
kh.mark_local(b_stage);
auto C_tiles = kh.map_work(work, [=](auto unit) {
  auto t     = unit.tile();
  auto u     = unit.id();
  auto my_a  = a_stage[t];
  auto my_b  = b_stage[t];
  auto a_row = sg::slice(my_a, {u[0], 0}, {1, 128});
  auto b_row = sg::slice(my_b, {u[1], 0}, {1, 128});
  auto a_run = sg::flatten(a_row);
  auto b_run = sg::flatten(b_row);
  return unit.run(a_run, b_run, dot_items);
});                                                               // 64 x 32 result tiles of {2, 4}
auto C_elems = kh.join(C_tiles);
C_out = kh.evaluate(C_elems);
```
2048 groups of 1024 work-items. 16 x 16 outputs per tile would need 32768, so the work is retiled; the bands stay as they were.

| value | lives in | because |
|---|---|---|
| A, B | global | inputs |
| a_stage, b_stage | local | other units of the same work tile pull them; marked |
| products in a run | registers | each unit's own |
| C | result memory | evaluated |

Traced: C[3][5] = 1920, C[127][127] = 2064512.

**Row sums at a cap of 256: split the runs, join the pieces.**
```cpp
auto rows = kh.tile(x, {1, 1024});                                // 64 x 1 tiles
auto work = kh.work_tiles({64}, {1}, 1024);                       // 1024 per group: too big at 256
auto sums = kh.map_work(work, [=](auto unit) {
  auto t   = unit.tile();
  auto row = rows[{t[0], 0}];
  auto run = sg::flatten(row);
  return unit.run(run, sum_items);
});
auto pieces      = kh.split_runs(sums, 4);                        // 64 x 4 partials, runs of 256
auto pieces_rows = kh.retile(pieces, {4});                        // 64 tiles of 4
auto work_join   = kh.work_tiles({64}, {1}, 4);                   // a later set
auto row_sums    = sum_tiles(kh, work_join, pieces_rows);
auto row_elems   = kh.join(row_sums);
y_out = kh.evaluate(row_elems);
```
256 groups of 256 plus 64 of 4. The rows tiling is unchanged; the body of sums is unchanged. Traced: y[0] = 523776, y[63] = 588288, the same as unsplit.

## Decisions for you (defaults marked)

1. Within one work set, may a map's runs be shorter than the set's run, the other items idle (the scan takes each tile's total on its own group this way)? (a, default) yes; (b) no, a shorter run needs its own work set.
2. A tile size that does not divide its array: (a, default) a builder error naming the node; (b) a partial last tile filled by an oob policy.
3. You used "slice" for two things: reusing a subset of a work set's groups, and cutting runs to fit the cap. (a, default) kh.slice_work for the subset, kh.split_runs for the cap; (b) the other way round; (c) other names.
4. Element-wise work with nothing shared is written as runs on a work set. (a, default) only that; (b) also keep kh.map as a plain range launch for kernels with no work sets.
5. Sharing an input tile in local memory is a copy pass on a slice of the same work set, marked local (the matrix product). (a, default) as written; (b) a primitive that marks an input tile local for a work set.

## What changed since the early check

- Work units hold runs of items (names swapped); a work tile is a fixed native group.
- kh.reuse_work is gone; groups are reused by slicing a work set (kh.slice_work), with any shape that fits.
- Runs are an extra dimension of a result tile.
- The scan is 16 groups of 64; only the gathered totals cross tiles.
- Pulls form a tile dependency graph that the toolchain orders; a kernel with no valid order is rejected when the builder runs; data-chosen pulls only cross work sets.
- New: kh.split_runs (cap fitting), kh.stencil, sg::tile_bin as a run collective; unit.id is used by the stencils, the matrix product and interpolation.
- Element-wise chains are runs on a work set; no plain kh.map.
- All 14 kernels are written and traced (kernels.md).
