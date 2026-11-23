# Language pass: full record

Plan sg-language-pass, carried out by the Research Lead. Jack is not required to read this file; the early packet (early-packet.md, v1, as Jack saw it) and the final summary page project it.

## 1. Rule ledger

Sources: docs/naga-internal/brainstorm/kernel-model.md ("Agreed" 1-6) and note sg-redesign-decisions (D2-D7, THE WORK MODEL, the backing and repeat entry). Baseline frame: note toolchain-impl-design, topic 1 (closed 2026-11-19). Where D7 restates an earlier rule, D7 governs.

| id | rule | source |
|---|---|---|
| B1 | A kernel is built by host code calling the kernel handle; one kernel definition is one launch; which units and links exist is structure, counts and sizes are bound values. | toolchain-impl-design topic 1 |
| M2 | Work units in a tile are independent; this makes work sliceable. | kernel-model Agreed 2 |
| M3 | A work unit's parallelism is one-dimensional: a run of items; its result is one value (a reduce) or a run (element-wise, scan, binning). | Agreed 3, D7 X1 |
| M4 | Data is tiled by the problem, never by the device. | Agreed 4 |
| M5 | A work tile maps to one native work-group; over the device cap the user retiles the work or slices the runs and joins the pieces; sliced work's data is derived from the unsliced work; data tiling never changes. | Agreed 5 |
| M6 | The user decides memory backing through this structure. | Agreed 6 |
| D2 | All sizes are runtime, bound values; ranks are in types; shape errors are reported when the builder runs, naming the node. | D2 |
| D3 | Elements and tiles are explicit; tile, retile, join, extend_tiles; evaluate takes elements. | D3 |
| D4 | Index maps are tiles of indices applied with kh.index_map(map, oob, elements); define_tiles is evaluated on the host for exact dependencies; stencil builds index maps; index arrays from data depend on all of the source; out-of-range indices go to the oob policy. | D4 |
| D5 | tile_bin bins within a tile. | D5 |
| W1 | A work tile is a native work-group, fixed. Each kh.work_tiles call is an explicit set of native groups; another set, of any shape, is a parallel set of groups. Using the same work tiles again uses the same groups; slicing uses a subset. | D6, D7 X2 |
| W2 | Backing: a result tile pulled in by the same work tile that produced it is held as a value (the same unit), in local memory, or lazily (another unit recomputes it unless marked local); a result tile pulled in by any other work tile goes through global memory. | D6, backing entry, D7 X5 |
| W3 | Everything is lazy unless marked (kh.mark_local) or not representable lazily. | D6 |
| W4 | The map_work body is host code that selects whole tiles (exact dependencies, bound data); unit.run's function is the only device code; a run's input is flattened to 1-D of the run length. | D6, D7 X1 |
| W5 | Tile selection knowable only in the kernel uses index_map forms and depends on all source tiles. | D6 |
| W6 | Unit bodies have a way to select or slice part of a tile. | D6 |
| W7 | kh.repeat repeats one structure n times, n a bound value. | backing and repeat entry |
| W8 | One launch has one native group size, the largest in the kernel; smaller work tiles leave native work-items unused. | D7 X3 |
| W9 | The run is an extra dimension: native group shape = tile shape ++ run; at most 3-D; a 3-D tile has a run of one item and no run argument. | D7 X4 |
| W10 | How users decide what to share is how they think about the problem; the language does not enforce it. | D7 X5 |

Removed by D7 X5: the former M1 ("work tiles are defined by what is shared") as a rule.

## 2. Glossary

- work tile: one native work-group (W1).
- work set: the work tiles of one kh.work_tiles call.
- work unit (unit): an independent producer of results in a work tile (M2).
- item: one native work-item of a unit's run (M3).
- run: a unit's 1-D parallel computation over its items (M3).
- result tiles: what kh.map_work returns, one per work tile.
- view: tiles or elements that describe values without holding them (tile, retile, join, extend_tiles, index_map, flatten, slice).

## 3. Primitive definitions (v2, 2026-11-23)

v1 seeded from Jack's snippets; v2 applies D7 (corrections X1-X4). Changes are listed in the changelog.

| id | name | takes | returns | role | rules | meaning |
|---|---|---|---|---|---|---|
| P01 | kh.source(array) | an input array | elements | host | B1 | the input array as elements of the kernel |
| P02 | kh.tile(x, shape) | elements, shape | tiles, with .count() | host view | D3, D2 | x's elements in row-major tiles of shape; .count() is the number of tiles (bound) |
| P03 | kh.retile(t, shape) | tiles, shape | tiles | host view | D3 | t's elements, in tile order, regrouped into tiles of shape |
| P04 | kh.join(t) | tiles | elements | host view | D3 | t's tiles stitched in tile order |
| P05 | kh.work_tiles(grid, tile_shape, run) | shapes, count | work set | host | W1, W8, W9 | one native group per grid point, of shape tile_shape ++ run (at most 3-D; a 3-D tile_shape takes no run argument and has runs of one item) |
| P06 | kh.map_work(work, body) | work set, host function of unit | result tiles | host | W1, W4, W9 | runs body on the host once per unit to select tiles; returns one result tile per work tile: tile_shape if units return one value, tile_shape ++ run length if they return runs |
| P07 | unit.run(inputs..., fn) | one or more 1-D views of one length, device function | the unit's result | host call, fn device | W4, M3 | runs fn on every item with that item's element of each input; the run length is the inputs' length, at most the work set's run (Q5); the result is one value if fn returns a collective reduce, otherwise the run of fn's returns |
| P08 | unit.tile() | | work tile id | host | W4 | the unit's work tile in the work set passed to map_work |
| P09 | unit.id() | | unit index | host | W4 | the unit's position in its work tile (unused so far) |
| P10 | sg::flatten(tile) | tile | 1-D view | host view | W4 | the tile's elements in row-major order |
| P11 | sg::slice(tile, offset, extent) | tile, shapes | tile view | host view | W6 | the part of the tile at offset with extent |
| P12 | sg::reduce(item, v, op) | item, value, op | one value | device collective | M3 | v combined over all items of the run with op (associative, commutative) |
| P13 | sg::inclusive_scan(item, v, op) | item, value, op | per-item value | device collective | M3 | v combined over items 0 to this one |
| P14 | kh.mark_local(result) | result tiles | | host | W3, W2 | holds the result in local memory for other units of the same work tile instead of recomputing it |
| P16 | kh.define_tiles(tiles, shape, fn) | counts, host function | index tiles | host | D4 | tile t, slot s holds fn(t, s) |
| P17 | kh.index_map(map, oob, elements) | index tiles, policy, elements | tiles view | host | D4 | each index replaced by that element; out-of-range indices go to oob |
| P18 | kh.extend_tiles(t, shape, how) | tiles, shape, policy | tiles view | host | D3 | each tile reshaped to shape (sg::repeat broadcasts) |
| P19 | kh.evaluate(elements) | elements | a result | host | B1, D3 | assigns the elements to a result |
| P20 | kh.repeat(n, start, fn) | bound count, value, host function | like start | host | W7 | fn recorded once; applied n times, each on the previous return |
| seed, not yet used | kh.slice_work, kh.stencil, sg::tile_bin | | | | M5, D4, D5 | defined when first used |

Changelog:
- v1 2026-11-23: P01-P20.
- v2 2026-11-23, from D7: X1 renames item/unit (P06-P09, P12-P14 wording; P07 is unit.run; P12, P13 take the item); X2 removes P15 kh.reuse_work; X3 and X4 fix P05's native shape and add the rank-3 form; P06's result shape follows X4; P07's run may be shorter than the work set's run (pending Q5, default).

## 4. Kernel records

Corrections from the early check that apply to every record: X1 (names), X2, X3, X4 (work sets and native shape). K3 also X6.

### K1 Global reduction, 64 in fours

Problem: sum 64 numbers. Reference: `float s = 0; for (int i = 0; i < 64; ++i) s += x[i];` For x[i] = i: 2016. For x[i] = i + 1: 2080.

Derivation: the 16 first-level sums each take 4 consecutive numbers; each sum is one unit with a run of 4 items, one per work tile. The 4 second-level sums take 4 first-level sums each; the last sum takes those 4. Each level reads the previous level's sums from other work tiles.

```cpp
auto plus      = sycl::plus<>{};
auto sum_items = [=](auto item, auto v) { return sg::reduce(item, v, plus); };
auto sum_tiles = [=](auto& kh, auto work, auto tiles) {
  return kh.map_work(work, [=](auto unit) {
    auto tile = tiles[unit.tile()];
    auto run  = sg::flatten(tile);
    return unit.run(run, sum_items);
  });
};

auto x           = kh.source(x_in);                     // 64 elements
auto x_tiles     = kh.tile(x, {4});                     // 16 tiles of 4
auto work1       = kh.work_tiles({16}, {1}, 4);         // 16 groups of 1 unit x 4 items
auto l1          = sum_tiles(kh, work1, x_tiles);       // 16 result tiles of {1}
auto l1_tiles    = kh.retile(l1, {4});                  // 4 tiles of 4
auto work2       = kh.work_tiles({4}, {1}, 4);          // 4 more groups
auto l2          = sum_tiles(kh, work2, l1_tiles);      // 4 result tiles of {1}
auto l2_tiles    = kh.retile(l2, {4});                  // 1 tile of 4
auto work3       = kh.work_tiles({1}, {1}, 4);          // 1 more group
auto l3          = sum_tiles(kh, work3, l2_tiles);      // 1 result tile of {1}
auto total_elems = kh.join(l3);                         // 1 element
total = kh.evaluate(total_elems);
```

Work-groups: 16 + 4 + 1 = 21 groups of shape {1, 4} (P05, W1). A work2 group waits only on work1 groups, an earlier set (lower ids); a work3 group only on work2 groups.

| value | placement | rule |
|---|---|---|
| x | global (input) | B1 |
| x_tiles, l1_tiles, l2_tiles, total_elems | views | D3 |
| each item's v | value | W2 |
| l1 | global: pulled in by work2, another work set | W2 |
| l2 | global: pulled in by work3 | W2 |
| l3 | result memory | B1 |

Trace, x[i] = i: l1[k] = 16k + 6 = 6, 22, ..., 246; l1_tiles t sums to 256t + 120 = 120, 376, 632, 888; total 2016. x[i] = i + 1: 10, 26, ..., 250; 136, 392, 648, 904; 2080. Both equal the reference.
Basis: P01-P08, P10, P12, P19 v2; Q2 = a. Depends on: Q2 (not exercised: one group shape).

### K2 n-level reduction, depth d a launch value

Problem: sum n = 4^d numbers. Reference: as K1. For x[i] = 1: n (d = 3: 64; d = 5: 1024).

Derivation: K1's level recorded once and repeated; each level's tile count is a bound value.

```cpp
// plus, sum_items, sum_tiles as in K1
auto x       = kh.source(x_in);                         // n = 4^d elements
auto x_tiles = kh.tile(x, {4});                         // n/4 tiles of 4
auto count1  = x_tiles.count();                         // n/4, a bound value
auto work1   = kh.work_tiles({count1}, {1}, 4);         // n/4 groups
auto first   = sum_tiles(kh, work1, x_tiles);           // n/4 result tiles of {1}
auto last    = kh.repeat(d - 1, first, [=](auto& kh, auto prev) {
  auto prev_tiles = kh.retile(prev, {4});
  auto count      = prev_tiles.count();
  auto work       = kh.work_tiles({count}, {1}, 4);
  return sum_tiles(kh, work, prev_tiles);
});                                                     // 1 result tile of {1}
auto total_elems = kh.join(last);
total = kh.evaluate(total_elems);
```

Work-groups: one work set per level, n/4 + n/16 + ... + 1 groups of {1, 4} (d = 3: 21; d = 5: 341), a bound value; a new d never recompiles.
Placement: as K1.
Trace, x[i] = 1: d = 3: 16 x 4, 4 x 16, 64. d = 5: 256 x 4, 64 x 16, 16 x 64, 4 x 256, 1024. Equal to the reference.
Basis: as K1 plus P20.

### K3 Whole-array inclusive scan, 1024 values, tiles of 64

Problem: y[i] = x[0] + ... + x[i]. Reference: `int s = 0; for (int i = 0; i < 1024; ++i) { s += x[i]; y[i] = s; }` For x[i] = 1: y[i] = i + 1.

Derivation (X6): each tile of 64 is scanned in its own work tile, one unit with 64 items; the tile's total is taken from its own scan on the same work tile; each tile then needs the totals of the tiles before it, which crosses work tiles; each work tile sums those into its increment and adds it to its own scan.

```cpp
auto plus       = sycl::plus<>{};
auto scan_items = [=](auto item, auto v) { return sg::inclusive_scan(item, v, plus); };
auto sum_items  = [=](auto item, auto v) { return sg::reduce(item, v, plus); };
auto pass_items = [=](auto item, auto v) { return v; };
auto add_items  = [=](auto item, auto s, auto inc) { return s + inc; };

auto x       = kh.source(x_in);                                  // 1024 elements
auto x_tiles = kh.tile(x, {64});                                 // 16 tiles of 64
auto work    = kh.work_tiles({16}, {1}, 64);                     // 16 groups of 1 unit x 64 items

// 1. each work tile scans its tile
auto scanned = kh.map_work(work, [=](auto unit) {
  auto x_tile = x_tiles[unit.tile()];
  auto x_run  = sg::flatten(x_tile);
  return unit.run(x_run, scan_items);
});                                                              // 16 result tiles of {1, 64}

// 2. each work tile's total: the last element of its own scan, a run of one item
auto totals = kh.map_work(work, [=](auto unit) {
  auto my_scan  = scanned[unit.tile()];
  auto my_last  = sg::slice(my_scan, {0, 63}, {1, 1});
  auto last_run = sg::flatten(my_last);
  return unit.run(last_run, pass_items);
});                                                              // 16 result tiles of {1}

// 3. for tile t, slot s: the total of tile s if s < t, else 0
auto totals_elems = kh.join(totals);                             // 16 elements
auto before_map   = kh.define_tiles(16, {64}, [](auto t, auto s) { return s < t ? s : -1; });
auto before       = kh.index_map(before_map, sg::fill{0}, totals_elems);   // 16 tiles of 64

// 4. each work tile's increment
auto increments = kh.map_work(work, [=](auto unit) {
  auto my_before  = before[unit.tile()];
  auto before_run = sg::flatten(my_before);
  return unit.run(before_run, sum_items);
});                                                              // 16 result tiles of {1}

// 5. add the increment to the scan
auto inc_wide = kh.extend_tiles(increments, {1, 64}, sg::repeat);  // 16 tiles of {1, 64}
auto y = kh.map_work(work, [=](auto unit) {
  auto my_scan  = scanned[unit.tile()];
  auto my_inc   = inc_wide[unit.tile()];
  auto scan_run = sg::flatten(my_scan);
  auto inc_run  = sg::flatten(my_inc);
  return unit.run(scan_run, inc_run, add_items);
});                                                              // 16 result tiles of {1, 64}
auto y_rows  = kh.retile(y, {64});                               // 16 tiles of 64
auto y_elems = kh.join(y_rows);                                  // 1024 elements
y_out = kh.evaluate(y_elems);
```

Work-groups: one work set, 16 groups of shape {1, 64} (W9); step 2 uses 1 of each group's 64 items (Q5). Step 3 is the only crossing: work tile t waits on the totals of tiles 0..t-1, lower groups of the same set, which have started.

| value | placement | rule |
|---|---|---|
| x | global (input) | B1 |
| scanned | value: pulled in only by the unit that made it (steps 2 and 5) | W2 |
| totals | value in its group; global for step 3, which other work tiles read | W2 |
| before, inc_wide, x_tiles, y_rows, totals_elems | views | D3, D4 |
| increments | value: pulled in only by the unit that made it | W2 |
| y | result memory | B1 |

Trace, x[i] = 1: scanned tile t = 1..64; totals[t] = 64; before tile t: 64 in slots 0..t-1, else 0; increments[t] = 64t; y tile t, item p = p + 1 + 64t, so y[k] = k + 1; y[0] = 1, y[1023] = 1024. Equal to the reference.
Basis: P01-P08, P10-P13, P16-P19 v2; Q2 = a, Q5 = default. Depends on: Q5.

## 5. Misfit log

| id | kernel | where | verdict | resolution |
|---|---|---|---|---|
| F1 | K1 | levels 2-3 should run on level 1's groups | withdrawn by D7 X2: each work set is its own set of groups; reuse means using the same work set | levels use new work sets (K1 v2) |
| F2 | K1 | a reused tile pinned to a group could wait on groups not yet started | withdrawn with F1: a later work set waits only on earlier sets | none needed |
| F3 | K3 v1 | units selecting their own data tile within one work tile | missing primitive | P09 unit.id(); unused since X6 |
| F4 | K3 | the total is the last element of a pulled-in tile (W6) | missing primitive | P11 sg::slice |
| F5 | K3 | first approach: an index map inside the body over the selected totals | wrong approach: D4's kh.define_tiles and kh.index_map at kernel level express it with host-evaluated exact dependencies | before = index_map(define_tiles(...), fill 0, join(totals)) |
| F6 | K3 | a separate pass-through map returning the gathered values | wrong approach once index maps are views: the view is the selection; step 4 reduces it directly; both steps stay visible | steps 3 and 4 as written |
| F7 | K3 v1 | work sets of different sizes in one launch | resolved by D7 X3 | W8 |
| F8 | K1-K3 | values moving to item 0 of a run: items hold their own values; a run of one reads the value held by item 63 | approach right | the toolchain moves values within a unit's group (a broadcast); covered by W2, held as a value |
| F9 | K3 | joining 16 result tiles of {1, 64} into a 1-D result: join of a 1-D grid of 2-D tiles gives {16, 64} | wrong approach: retile to {64} first, then join | y_rows = retile(y, {64}) |
| F10 | K3 | step 2 runs one item on a work set whose run is 64 | approach right (Jack's step: the total taken on the same work tiles); P07 lets a run be shorter than the work set's run, the other items idle | Q5 |

## 6. Decision register

| id | question | options | default | affects | status |
|---|---|---|---|---|---|
| Q1 | Which group runs a reused work tile? | | | | withdrawn: malformed, a work tile is a native group (D7 X2) |
| Q2 | Work sets of different sizes in one launch | a: one native size, the largest; smaller tiles leave work-items unused | a | W8 | answered a (D7 X3) |
| Q3 | Layout of runs in a result tile | b: an extra dimension | b | W9, P06 | answered b, with rank-3 tiles taking runs of one (D7 X4) |
| Q4 | May units that share nothing be packed into one work tile? | | | | withdrawn: sharing is not enforced (D7 X5) |
| Q5 | Within one work set, may a map_work's runs be shorter than the work set's run, the other items idle? | a: yes (K3 step 2 runs one item on its 64-item groups); b: no, a shorter run needs its own work set (K3 step 2 then gets 16 new groups and scanned crosses through global memory) | a | P07, K3 | put 2026-11-23 |

Q5 traced under both options on K3: both give y[k] = k + 1.

## 7. Contact log

| id | date | kind | size | questions | response |
|---|---|---|---|---|---|
| C1 | 2026-11-23 | early packet v1 | 127 lines (bound 120: over by 7, the scan kept uncompressed) | 4 (Q1-Q4) | 2026-11-23: corrections X1-X6 (note D7); Q1, Q4 withdrawn; Q2 a; Q3 b |
| C2 | 2026-11-23 | escalation-free follow-up in conversation | 1 question | Q5 | pending; work proceeds under the default |
