# Language pass: full record

Plan sg-language-pass, carried out by the Research Lead. Jack is not required to read this file. Files: summary.md (the page for Jack), early-packet.md (v1, as Jack saw it), kernels.md (all kernel records), this file (ledger, glossary, definitions, mechanisms, misfits, review, register, contact log).

## 1. Rule ledger

Sources: docs/naga-internal/brainstorm/kernel-model.md ("Agreed" 2-6), note sg-redesign-decisions (D2-D14, THE WORK MODEL, the backing and repeat entry), note toolchain-impl-design (topic 1, closed 2026-11-19; check-in 2026-11-21 for inplace). Where a later decision restates an earlier rule, the later governs.

| id | rule | source |
|---|---|---|
| B1 | A kernel is built by host code calling the kernel handle; one kernel definition is one launch; which units and links exist is structure, counts and sizes are bound values. | toolchain-impl-design topic 1 |
| B2 | inplace results: contributors on one handle run in parallel and each element is combined atomically with the named operation and initializer. | toolchain-impl-design check-in 2026-11-21 |
| M2 | Work units in a tile are independent; this makes work sliceable. | kernel-model Agreed 2 |
| M3 | A work unit's parallelism is one-dimensional: a run of items; its result is one value (a reduce) or a run (element-wise, scan, binning). | Agreed 3, D7 X1 |
| M4 | Data is tiled by the problem, never by the device. | Agreed 4 |
| M5 | A work tile maps to native work-groups chosen at construction; over the device cap the user builds different work tiles or segments the run (sg::segment_items); the data tiling never changes. | Agreed 5, D10 |
| M6 | The user decides memory backing through this structure. | Agreed 6 |
| D2 | All sizes are runtime, bound values; ranks are in types; shape errors are reported when the builder runs, naming the node. | D2 |
| D3 | Elements and tiles are explicit; tile, retile, join, extend_tiles; evaluate takes elements. | D3 |
| D4 | Index maps are tiles of indices applied with kh.index_map(map, oob, elements); define_tiles is evaluated on the host; stencil builds index maps; out-of-range indices go to the oob policy. | D4 |
| D5 | tile_bin bins within a tile. | D5 |
| W1 | A work tile is a native work-group, fixed. Each kh.work_tiles call is an explicit set of native groups; another set, of any shape, is a parallel set. Using the same work set again uses the same groups; slicing a work set reuses a subset of its groups, possibly for a sub-problem of a different shape. | D6, D7 X2, D8 |
| W2 | Backing: a result tile pulled in by the work tile that produced it is held as a value (the same unit), in local memory, or lazily (another unit recomputes it unless marked local); pulled in by any other work tile it goes through global memory. | D6, backing entry, D7 X5 |
| W3 | Everything is lazy unless marked (kh.mark_local) or not representable lazily. | D6 |
| W4 | The map_work body is host code that selects whole tiles; unit.run's function is the only device code; a run's input is flattened to 1-D of the run length. | D6, D7 X1 |
| W5 | Tiles chosen by data produced in the kernel are known only at run time; the wait is on the chosen tile. | D6, D8 |
| W6 | Unit bodies have a way to select or slice part of a tile. | D6 |
| W7 | kh.repeat repeats one structure n times, n a bound value. | backing and repeat entry |
| W8 | One launch has one native group size, the largest in the kernel; smaller work tiles leave native work-items unused. | D7 X3 |
| W9 | The run is an extra dimension: native group shape = tile shape ++ run; at most 3-D; a 3-D tile has a run of one item and no run argument. | D7 X4 |
| W10 | What users share is how they think about the problem; the language does not enforce it. | D7 X5 |
| W11 | Pulls form a dependency graph between tiles: an edge from each pulled tile to the tile that pulls it; a pull the host cannot see (a tile chosen from kernel data) is an edge from every tile of the source work set. Each native group runs its tasks (its tile of a set, then any slice tasks) in map order. The toolchain orders the native groups so that every edge comes from an earlier group or an earlier task of the same group; if no order exists the builder rejects the kernel, naming the tiles on the cycle, and the user breaks it with another work set. So data-chosen pulls only ever go from one work set to another. | D9 |
| W12 | A segmented work tile is one native group per segment, each of tile shape x items per segment. A segmented work set is its own type: units run_distributed (collectives joined across segments with a global dependency, edges from earlier segments in the tile graph) or run_replicated (every segment runs the whole run, collectives stay in the group); results are marked local with kh.mark_local: the default form for replicated results (the whole result in every segment), the <sg::segmented> form, required, for distributed results (each segment its share); each form is an error on the other kind of result; run_replicated is rejected if its input is a distributed result of its own set. | D10, D11, D12 |
| W13 | A kernel uses either work tiles or work items, never both. kh.work_items(range) (1-3 dims) is a work-item set: a plain range launch with no groups; work-item sets can be created anew and sliced. A work item takes array tiles and elements of arrays; its result is a result array; a work item may take another work-item set's result only at its own index (the item that produced it), never as tiles; result arrays never feed work tiles and result tiles never feed work items. | D13, D14 |
| W14 | Array tiles (tiles of input arrays, including index-map views over inputs), and only array tiles, can be marked local for a work set: kh.mark_local(array_tiles, work); every work tile (every segment, on a segmented set) that pulls such a tile gets its own copy in local memory, populated by its own native work-items. | D13 |
| W15 | A tile size need not divide its array: kh.tile(x, shape, oob) makes a partial last tile whose missing positions go to the oob policy; a join of tiles derived from it trims back to the source extent. A map may use a shorter run than its work set was built with; the result tile's run dimension follows the run used. | D13, D14 |

## 2. Glossary

- work tile: one native work-group (W1). work set: the work tiles of one kh.work_tiles call, or a slice of one.
- work unit (unit): an independent producer of results in a work tile (M2). item: one native work-item of a unit's run (M3). run: a unit's 1-D parallel computation over its items.
- result tiles: what kh.map_work returns, one per work tile.
- view: tiles or elements that describe values without holding them (tile, retile, join, extend_tiles, index_map, flatten, slice).
- pull: a unit body selecting a tile of a result.

## 3. Primitive definitions (v3, 2026-11-23)

| id | name | takes | returns | role | rules | meaning |
|---|---|---|---|---|---|---|
| P01 | kh.source(array) | an input array | elements | host | B1 | the input array as elements of the kernel |
| P02 | kh.tile(x, shape) | elements, shape, oob policy (optional) | tiles, with .count() | host view | D3, D2 | x's elements in row-major tiles of shape; a partial last tile's missing positions go to oob (W15); .count() the number of tiles (bound) |
| P03 | kh.retile(t, shape) | tiles, shape | tiles | host view | D3 | t's elements, in tile order, regrouped into tiles of shape |
| P04 | kh.join(t) | tiles of the grid's rank | elements | host view | D3 | tiles stitched over the grid; grid and tile of different ranks are retiled first (F9); padding from a partial last tile is trimmed (W15) |
| P05 | kh.work_tiles(grid, tile_shape, run) | shapes, a run count or sg::segment_items | work set | host | W1, W8, W9 | one native group per grid point, of shape tile_shape ++ run (at most 3-D; a 3-D tile_shape takes no run) |
| P06 | kh.map_work(work, body) | work set, host function of unit | result tiles | host | W1, W4, W9, W11 | body runs on the host once per unit to select tiles; one result tile per work tile: tile_shape if units return one value, tile_shape ++ run length if they return runs |
| P07 | unit.run(inputs..., fn) | 1-D views of one length, device function | the unit's result | host call, fn device | W4, M3 | fn on every item with that item's element of each input; the run length is the inputs' length, at most the work set's run (Q5); one value if fn returns a collective reduce, else the run of fn's returns |
| P08 | unit.tile() | | work tile id | host | W4 | the unit's work tile in the work set passed to map_work |
| P09 | unit.id() | | unit index | host | W4 | the unit's position in its work tile |
| P10 | sg::flatten(tile) | tile | 1-D view | host view | W4 | the tile's elements in row-major order |
| P11 | sg::slice(tile, offset, extent) | tile, shapes | tile view | host view | W6 | the part of the tile at offset with extent |
| P12 | sg::reduce(item, v, op) | item, value, op | one value | device collective | M3 | v combined over all items of the run with op |
| P13 | sg::inclusive_scan(item, v, op) | item, value, op | per-item value | device collective | M3 | v combined over items 0 to this one |
| P14 | kh.mark_local<Mode = default>(result) | result tiles | | host | W3, W2, W12 | holds the result in local memory for other units of the same work tile; on a segmented work set the default mode applies to replicated results (whole result per segment) and Mode = sg::segmented is required for distributed results (each segment its share); the wrong mode is a builder error; on array tiles: kh.mark_local(array_tiles, work) gives every work tile (or segment) of work that pulls them its own local copy (W14) |
| P15 | kh.slice_work(work, offset, extent, stride, tile_shape, run) | work set, shapes | work set | host | W1, W11 | the tiles offset + k * stride (k < extent) of work, reused on their groups, with a new tile shape and run that fit the group size; launches no groups, run a count or sg::segment_items |
| P16 | kh.define_tiles(tiles, shape, fn) | counts, host function | index tiles | host | D4 | tile t, slot s holds fn(t, s) |
| P17 | kh.index_map(map, oob, elements) | index tiles, policy, elements | tiles view | host | D4, W5 | each index replaced by that element; out-of-range indices go to oob |
| P18 | kh.extend_tiles(t, shape, how) | tiles, shape, policy | tiles view | host | D3 | each tile reshaped to shape (sg::repeat broadcasts) |
| P19 | kh.evaluate(elements) | elements | a result | host | B1, B2, D3 | assigns the elements to a result (an inplace result combines) |
| P20 | kh.repeat(n, start, fn) | bound count, value, host function | like start | host | W7 | fn recorded once, applied n times; counts derived from prev inside fn are bound per iteration |
| P21 | sg::segment_items(segments, items) | counts | a run argument | host | W12 | a run of segments x items items in segments of items, each segment its own native group |
| P22 | sg::tile_bin(item, v, bins, key) | item, value, count, device function | per-item count | device collective | D5, M3 | item b < bins gets the number of v in the run with key(v) == b; the result is a run of bins values (bins at most the run length) |
| P23 | kh.stencil(offsets, roots) | offsets, a range | index tiles | host | D4 | one tile per root, holding root + each offset |
| P25 | unit.run_distributed(inputs..., fn) | as P07 | the unit's result | host call, fn device | W12 | on a segmented work set: each segment runs its share of the run; collectives join across segments |
| P26 | unit.run_replicated(inputs..., fn) | as P07 | the unit's result | host call, fn device | W12 | on a segmented work set: every segment runs the whole run, item k taking positions k, k + items, ...; collectives stay in the group |
| P27 | kh.work_items(range) | range (1-3 dims) | work-item set | host | W13 | a plain range launch of that many native work-items, no groups |
| P28 | kh.map_items(items, body(item)) | work-item set, host function of item | result array | host | W13, W4 | body runs on the host per item to select its inputs by item.id(); the result array has the set's range |
| P29 | item.compute(inputs..., fn) | values or array tiles, device function | the item's value | host call, fn device | W13 | fn on the item's inputs; no collectives |
| P30 | item.id() | | item index | host | W13 | the item's position in its work-item set |

Changelog:
- v1 2026-11-23: P01-P20.
- v2 2026-11-23 (D7): X1 renames; P15 kh.reuse_work removed (X2); P05 native shape and rank-3 form (X3, X4); P06 result shape (X4); P07 shorter runs (Q5).
- v3 2026-11-23 (D8 and the pass): P15 kh.slice_work added (W1, D8); P21 kh.split_runs added (F11); P22, P23 defined at first use; P04 requires equal ranks (F9); P20 clarified (bound counts per iteration, meaning unchanged); P02 divisibility (Q6). No meaning of a primitive used by an earlier record was changed except by Jack's corrections X1-X4.
- v4 2026-11-23 (D10): kh.split_runs withdrawn (P21 is now sg::segment_items); P24 kh.mark_segmented_local added; P05, P15 take segmented runs; P14 kh.mark_local is rejected on segmented work tiles.
- v5 2026-11-23 (D11): segmented work sets are their own type; P25 unit.run_distributed and P26 unit.run_replicated added; unit.run (P07) and kh.mark_local (P14) do not exist on them; P24 renamed kh.mark_segmented_local and holds results as their run produced them.
- v6 2026-11-23 (D12): P24 kh.mark_segmented_local withdrawn; P14 kh.mark_local takes a mode, sg::segmented required on distributed results.
- v7 2026-11-23 (D13, D14): P02 oob, P04 trims, P14 marks array tiles for a work set, P27-P30 work items.

## 4. Mechanisms

### 4.1 Reusing work-groups for sub-problems of different shapes
- Same work set, shorter run (Q5): K3 step 2 takes each tile's total on its own group, 1 of 64 items active.
- Slice with a stride, same shape: K1 levels 2 and 3 on work1 tiles 3, 7, 11, 15 and 15: 16 groups instead of 21; each level-2 tile pulls only lower tiles and its own (W11).
- Full slice, different shape: K6 stages A rows ({2} x 128) and B columns ({4} x 128) on the same groups as the {2, 4} x 128 product; the stages are marked local and pulled by the same work tiles (W2).
- One-tile slice: K7 sums all 16 partial histograms on work tile 15, the only tile allowed to pull all of its set (W11), as 16 units x 16 items.
Group counts: K1 16 (shape {1, 4}); K6 2048 (shape {2, 4, 128}, stages use 256 and 512 work-items); K7 16 (shape {1, 256}, the slice uses 256).

### 4.2 Fitting the work-group cap (D10)
- Different work tiles, chosen at construction: K6 built as 2 x 4 outputs x 128 = 1024 per group; no cross-segment cost.
- Segmented run: K6 with 16 x 16 outputs and sg::segment_items(32, 4): 2048 groups of 1024; the stages are segmented local and each item reads its own segment; the dot-product reduce is joined across 32 segments (global). K14 at cap 256 with sg::segment_items(4, 256): the reduce is joined across 4 segments; no join code.
- Builder error for an unfitted work set: "work set of kh.map_work at <location>: 16 x 16 x 128 = 32768 work-items exceeds the device's 1024; build smaller work tiles or segment the run".
- Shorter runs (Q5, answered yes): K3 step 2 runs one item on 64-item groups.

### 4.3 The tile dependency graph (W11)
Per kernel, the graph and its order: K1, K2: each slice task depends on level tasks of lower groups; order = tile order. K3: tile t depends on tiles 0..t-1; order = tile order. K5: every work_y group depends on every work_src group (data-chosen pulls); work_src first. K7: group 15's slice task depends on all 16 groups; order = tile order. K8: F on the same set as E would make groups t and 3 - t depend on each other, a cycle, so F is its own set, ordered after. K14: work_join after the pieces. No kernel as written is rejected.

## 5. Misfit log

| id | kernel | where | verdict | resolution |
|---|---|---|---|---|
| F1 | K1 | levels on level 1's groups | withdrawn (D7 X2) | slices (F11) |
| F2 | K1 | reused tile waiting on unstarted groups | withdrawn (D8 ordering) | W11 |
| F3 | K3 v1 | units selecting their own data tile | missing primitive | P09 unit.id(), used by K4, K6, K12, K13 |
| F4 | K3 | the last element of a pulled-in tile | missing primitive | P11 sg::slice |
| F5 | K3 | an index map inside the body | wrong approach | kernel-level define_tiles + index_map |
| F6 | K3 | a pass-through map after the index map | wrong approach | the view is the selection |
| F7 | K3 v1 | work sets of different sizes | resolved (D7 X3) | W8 |
| F8 | K1-K3 | a run of one reading item 63's value | approach right | moved within the group by the toolchain |
| F9 | K3 | join of a 1-D grid of 2-D tiles | wrong approach | retile to 1-D tiles, then join |
| F10 | K3 | one item on a 64-item work set | approach right | Q5 |
| F11 | K1, K14 | Jack used "slice" both for reusing a subset of a work set (D6, D8) and for cutting runs to fit the cap (earlier snippet) | two meanings, so two primitives | P15 kh.slice_work (subset, latest usage), P21 kh.split_runs (cap); names put as Q7 |
| F12 | K2 | slice offset and stride per level inside repeat | approach right | derived from bound counts inside fn (P20) |
| F13 | K4, K12, K13 | units in a tile need their own window | approach right | P09 unit.id() |
| F14 | K5 | gather from data-chosen tiles of its own set would break W11 | wrong approach | gather on a later work set (within its own set a data-chosen pull is a cycle in the tile graph, W11) |
| F15 | K6 | 16 x 16 x 128 over the cap; inputs shared in local memory | approach right for the cap (retile the work); for sharing, no primitive marks an input local: wrong approach | copy pass on a full slice, marked local (D7's GEMM description); Q9 |
| F16 | K7 | D5's tile_bin returns bins values, not one per item | approach right | P22 returns a run of bins values |
| F17 | K7 | combining all partials pulls every tile of the set | approach right | a one-tile slice on the highest tile |
| F18 | K8 | F reads E's tiles in reverse | wrong approach on one set | F on its own later set |
| F19 | K9-K11 | element-wise work without sharing: the earlier kh.map range launch is not in the seed | wrong approach to add it: runs on a work set keep chained values in registers | no new primitive; Q8 |
| F20 | K10 | n known only at run time may not divide the tile | approach right | builder error (default); Q6 |
| F21 | K5, K7 | D8's run-time check of data-chosen pulls within a set | superseded by D9 | the tile graph rejects such kernels when the builder runs; nothing is left to check on the device |
| F22 | K6, K14 | splitting runs after construction, and retiling work, do not fit fixed work tiles (Jack) | wrong approach | segmented runs at construction (P21 sg::segment_items, P14 with sg::segmented) |
| F23 | K6 | a segmented stage read at other segments' positions needs the whole stage in each segment | approach right | unit.run_replicated (P26); K6 does not need it (distributed suffices) |
| F24 | K9-K11 | element-wise chains written as runs on work tiles | wrong approach once work items exist (D13) | K9-K11 as work-item kernels |
| F25 | K6 | copy-pass stages to get input bands into local memory | superseded (D13) | kh.mark_local(a_bands, work) on array tiles |

## 6. Review of decisions

| item | verdict | evidence |
|---|---|---|
| D2 runtime sizes | holds | K2 depth, K10 n, every tile and run size a bound value |
| D3 elements and tiles | holds, with join requiring equal ranks | F9; K3, K5, K7 |
| D4 index maps | holds; waiting on data-chosen tiles is per chosen tile (D8) | K3, K4, K5, K7, K12, K13 |
| D5 tile_bin | changed in form: a device collective over a run returning bins values | K7, F16 |
| tile_bin | as D5 | K7 |
| define_tiles vs work_tiles | different things: index tiles over data vs native groups | K3, K7 use both |
| nesting depth | tile, unit, run sufficed; no kernel needed a 3-D tile | all kernels |

## 7. Decision register

| id | question | options | default | affects | status |
|---|---|---|---|---|---|
| Q1 | reused tile placement | | | | withdrawn (D7 X2) |
| Q2 | work sets of different sizes | a | a | W8 | answered a (D7 X3) |
| Q3 | runs in result tiles | b | b | W9 | answered b (D7 X4) |
| Q4 | packing units that share nothing | | | | withdrawn (D7 X5) |
| Q5 | Within one work set, may a map's runs be shorter than the set's run, the other items idle? | a: yes; b: no, a shorter run needs its own work set | a | P07, K3 | answered a: yes (D13) |
| Q6 | A tile size that does not divide its array (n known at run time)? | a: builder error naming the node; b: a partial last tile filled by an oob policy | a | P02, K10 | answered b: oob policy, not an error (D13) |
| Q7 | Names: kh.slice_work reuses a subset of a work set, kh.split_runs cuts runs to fit the cap. | a: as defined; b: swap; c: other names | a | P15, P21 | withdrawn: kh.split_runs is gone, one meaning of slice remains (D10) |
| Q8 | Element-wise work with nothing shared: written as runs on a work set (K9-K11). | a: only that; b: also keep kh.map as a plain range launch for kernels with no work sets | a | K9-K11 | answered: neither; work-item arrays (D13, D14) |
| Q9 | Sharing an input tile in local memory: a copy pass on a slice of the same work set, marked local (K6). | a: as written; b: a primitive marking an input tile local for a work set | a | K6 | answered b: mark array tiles local for a work set (D13) |

Each option was traced on a dependent kernel: Q5b K3 with a 16-group totals set (y[k] = k + 1); Q6b K10 with n = 300 and a zero-filled last tile (y[299] = a * 299 + b); Q7 names only; Q8b K9 as a range launch (16i + 15); Q9b K6 marking the bands local (C[3][5] = 1920).

## 8. Contact log

| id | date | kind | size | questions | response |
|---|---|---|---|---|---|
| C1 | 2026-11-23 | early packet v1 | 127 lines (bound 120, over by 7: the scan kept uncompressed) | 4 (Q1-Q4) | corrections X1-X6 (D7); Q1, Q4 withdrawn; Q2 a; Q3 b |
| C2 | 2026-11-23 | follow-up in conversation | | 1 (Q5) | none; work under the default |
| C3 | 2026-11-23 | follow-up in conversation | | 1 (the ordering rule) | accepted with the data-chosen contract (D8) |
| C4 | 2026-11-23 | final page | see summary.md | 5 (Q5-Q9) | correction: segmented runs at construction (D10); delta page |
| C5 | 2026-11-23 | in conversation | | 0 | the tile-graph rule (D9), replacing D8's run-time contract |
| C6 | 2026-11-23 | delta page | see delta.md | 0 new (4 remaining) | correction: segmented work sets as their own type, run_distributed / run_replicated (D11) |
| C7 | 2026-11-23 | delta page, round 2 | see delta.md | 0 new (4 remaining) | correction: one kh.mark_local with a segmented mode (D12) |
| C8 | 2026-11-23 | delta page, round 3 | see delta.md | 0 new (4 remaining) | answers to Q5, Q6, Q8, Q9 (D13); chaining confirmed (D14) |
| C9 | 2026-11-23 | delta page, round 4 | see delta.md | 0 | pending |
Total questions put: 11 (Q7 withdrawn).
