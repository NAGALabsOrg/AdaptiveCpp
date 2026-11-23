# Language pass: full record

Plan sg-language-pass, carried out by the Research Lead. Jack is not required to read this file; the early packet (early-packet.md) and the final summary page project it.

## 1. Rule ledger

Sources: docs/naga-internal/brainstorm/kernel-model.md ("Agreed", items 1-6) and note sg-redesign-decisions (D2-D6, THE WORK MODEL, the 2026-11-23 backing and repeat entry). Baseline frame: note toolchain-impl-design, topic 1 (closed 2026-11-19).

| id | rule | source |
|---|---|---|
| B1 | A kernel is built by host code calling the kernel handle; one kernel definition is one launch; which units and links exist is structure, counts and sizes are bound values. | toolchain-impl-design topic 1 |
| M1 | Work tiles are defined by what is shared. | kernel-model Agreed 1 |
| M2 | Work items in a tile are independent; this makes work sliceable. | Agreed 2 |
| M3 | An item's parallelism is one-dimensional: a single run; its result is one value (a reduce) or a run (element-wise, scan, binning). | Agreed 3 |
| M4 | Data is tiled so each work tile can select what it shares; data tiling follows the problem, never the device. | Agreed 4 |
| M5 | A work tile maps to one native work-group of items x run length; over the device cap the user retiles the work or slices the runs and joins the pieces; sliced work's data is derived from the unsliced work; data tiling never changes. | Agreed 5 |
| M6 | The user decides memory backing through this structure. | Agreed 6 |
| D2 | All sizes are runtime, bound values; ranks are in types; shape errors are reported when the builder runs, naming the node. | D2 |
| D3 | Elements and tiles are explicit; tile, retile, join, extend_tiles; evaluate takes elements. | D3 |
| D4 | Index maps are tiles of indices applied with kh.index_map(map, oob, elements); define_tiles is evaluated on the host for exact dependencies; stencil builds index maps; index arrays from data depend on all of the source; out-of-range indices go to the oob policy. | D4 |
| D5 | tile_bin bins within a tile. | D5 |
| W1 | A set of work tiles maps to native work-groups; another set is another set of groups; the same set reuses the same groups; slicing uses a subset. | D6 |
| W2 | Backing: the same item reading its own result keeps it as a value; another item of the same work tiles recomputes it unless marked local; different work tiles read it through global memory. | D6, backing entry |
| W3 | Everything is lazy unless marked (kh.mark_local) or not representable lazily. | D6 |
| W4 | The map_work body is host code that selects whole tiles (exact dependencies, bound data); item.run's function is the only device code; a run's input is flattened to 1-D of the unit count. | D6 |
| W5 | Tile selection knowable only in the kernel uses index_map forms and depends on all source tiles. | D6 |
| W6 | Item bodies have a way to select or slice part of a tile. | D6 |
| W7 | kh.repeat repeats one structure n times, n a bound value. | backing and repeat entry |

## 2. Glossary

- work tile: one native work-group's worth of work, defined by what its items share (M1, W1).
- work, work set: the set of work tiles returned by kh.work_tiles or kh.reuse_work.
- item: an independent unit of results in a work tile (M2).
- unit: one lane of an item's run (M3).
- run: the item's 1-D parallel computation over its units (M3).
- result tiles: what kh.map_work returns: one result tile per work tile, holding its items' results.
- view: tiles or elements that describe values without holding them (tile, retile, join, extend_tiles, index_map, flatten, slice).
- reused tile: a work tile of a reused work set; it runs on a group of the work set it reuses.

## 3. Primitive definitions (v1, 2026-11-23)

Seeded from Jack's snippets; P09, P11, P13, P15 added through the misfit log.

| id | name | takes | returns | role | rules | meaning |
|---|---|---|---|---|---|---|
| P01 | kh.source(array) | an input array | elements | host | B1 | the input array as elements of the kernel |
| P02 | kh.tile(x, shape) | elements, shape | tiles, with .count() | host view | D3, D2 | x's elements in row-major tiles of shape; .count() is the number of tiles (bound) |
| P03 | kh.retile(t, shape) | tiles, shape | tiles | host view | D3 | t's elements, in tile order, regrouped into tiles of shape |
| P04 | kh.join(t) | tiles | elements | host view | D3 | t's tiles stitched in tile order |
| P05 | kh.work_tiles(grid, tile_shape, units) | shapes, count | work | host | W1, M1, M3, M5 | one native group per grid point; each runs tile_shape items, each with a run of units |
| P06 | kh.map_work(work, body) | work, host function of item | result tiles | host | W4 | runs body on the host once per item to select tiles; returns one result tile per work tile with its items' results in item order; an item's run is laid contiguously along the last dimension (Q3) |
| P07 | item.run(inputs..., fn) | one or more 1-D views of the unit count, device function | the item's result | host call, fn device | W4, M3 | runs fn on every unit with that unit's element of each input; the result is one value if fn returns a collective reduce, otherwise the run of fn's returns |
| P08 | item.tile() | | work tile id | host | W4 | the item's work tile in the work passed to map_work |
| P09 | item.id() | | item index | host | W4 | the item's position in its work tile |
| P10 | sg::flatten(tile) | tile | 1-D view | host view | W4 | the tile's elements in row-major order |
| P11 | sg::slice(tile, offset, extent) | tile, shapes | tile view | host view | W6 | the part of the tile at offset with extent |
| P12 | sg::reduce(unit, v, op) | unit, value, op | one value | device collective | M3 | v combined over all units of the item with op (associative, commutative) |
| P13 | sg::inclusive_scan(unit, v, op) | unit, value, op | per-unit value | device collective | M3 | v combined over units 0 to this one |
| P14 | kh.mark_local(result) | result tiles | | host | W3, W2 | holds the result in local memory for other items of the same work tiles instead of recomputing it |
| P15 | kh.reuse_work(work, grid, tile_shape, units) | work, shapes, count | work | host | W1, M5 | a work set that launches no groups: its tile k runs on the group of work that delivers the last of k's inputs from work's results, or on group k if it has none (Q1); its tile count is at most work's and items x units at most the group size |
| P16 | kh.define_tiles(tiles, shape, fn) | counts, host function | index tiles | host | D4 | tile t, slot s holds fn(t, s) |
| P17 | kh.index_map(map, oob, elements) | index tiles, policy, elements | tiles view | host | D4 | each index replaced by that element; out-of-range indices go to oob |
| P18 | kh.extend_tiles(t, shape, how) | tiles, shape, policy | tiles view | host | D3 | each tile reshaped to shape (sg::repeat broadcasts) |
| P19 | kh.evaluate(elements) | elements | a result | host | B1, D3 | assigns the elements to a result |
| P20 | kh.repeat(n, start, fn) | bound count, value, host function | like start | host | W7 | fn recorded once; applied n times, each on the previous return |
| seed, not yet used | kh.slice_work, kh.stencil, sg::tile_bin | | | | M5, D4, D5 | defined when first used |

Changelog: v1 2026-11-23, P01-P20 as above. No meaning changed since.

## 4. Kernel records

### K1 Global reduction, 64 in fours

Problem: sum 64 numbers. Reference (plain sequential): `float s = 0; for (int i = 0; i < 64; ++i) s += x[i];` For x[i] = i: 2016. For x[i] = i + 1: 2080.

Derivation: the 16 first-level sums each take 4 consecutive numbers; no sum shares anything with another, so each sum is its own work tile with one item, whose 4 units combine in parallel. The 4 second-level sums each take 4 first-level sums and the last sum takes those 4; they share nothing either and should not need new groups.

```cpp
auto plus      = sycl::plus<>{};
auto sum_units = [=](auto unit, auto v) { return sg::reduce(unit, v, plus); };
auto sum_tiles = [=](auto& kh, auto work, auto tiles) {
  return kh.map_work(work, [=](auto item) {
    auto tile = tiles[item.tile()];
    auto run  = sg::flatten(tile);
    return item.run(run, sum_units);
  });
};

auto x           = kh.source(x_in);                     // 64 elements
auto x_tiles     = kh.tile(x, {4});                     // 16 tiles of 4
auto work1       = kh.work_tiles({16}, {1}, 4);         // grid 16, 1 item per tile, 4 units
auto l1          = sum_tiles(kh, work1, x_tiles);       // 16 result tiles of {1}
auto l1_tiles    = kh.retile(l1, {4});                  // 4 tiles of 4
auto work2       = kh.reuse_work(work1, {4}, {1}, 4);   // 4 work tiles on work1's groups
auto l2          = sum_tiles(kh, work2, l1_tiles);      // 4 result tiles of {1}
auto l2_tiles    = kh.retile(l2, {4});                  // 1 tile of 4
auto work3       = kh.reuse_work(work1, {1}, {1}, 4);   // 1 work tile on work1's groups
auto l3          = sum_tiles(kh, work3, l2_tiles);      // 1 result tile of {1}
auto total_elems = kh.join(l3);                         // 1 element
total = kh.evaluate(total_elems);
```

Work-groups: work1: 16 groups of 1 x 4 = 4 work-items (P05). work2, work3 launch none (P15); work2's tile t runs on whichever of groups 4t..4t+3 finishes last, work3's tile on whichever group finishes level 2 last. Total 16 groups of 4; nothing waits.

| value | placement | rule |
|---|---|---|
| x | global (input) | B1 |
| x_tiles, l1_tiles, l2_tiles, total_elems | views | D3 |
| units' v | value | W2 (own) |
| l1 | global: work2 tile t reads tiles produced by 4 different work tiles | W2 |
| l2 | global, same reason | W2 |
| l3 | value until evaluate; result memory | W2, B1 |

Trace, x[i] = i: x_tiles k = {4k..4k+3}; l1[k] = 16k + 6 = 6, 22, ..., 246. l1_tiles t = l1[4t..4t+3], sum 256t + 120 = 120, 376, 632, 888. l2_tiles 0 = {120, 376, 632, 888}, sum 2016. Equals reference. With x[i] = i + 1: l1 = 10, 26, ..., 250; l2 = 136, 392, 648, 904; total 2080. Equals reference.
Basis: P01-P07, P10, P12, P15, P19 v1; Q1 = a, Q2 = a, Q3 = a, Q4 = a. Depends on: Q1, Q4.

### K2 n-level reduction, depth d a launch value

Problem: sum n = 4^d numbers. Reference: as K1. For x[i] = 1: n (d = 3: 64; d = 5: 1024).

Derivation: K1's levels with the number of levels and each level's count as bound values: one level is recorded once and repeated.

```cpp
// plus, sum_units, sum_tiles as in K1
auto x       = kh.source(x_in);                         // n = 4^d elements
auto x_tiles = kh.tile(x, {4});                         // n/4 tiles of 4
auto count1  = x_tiles.count();                         // n/4, a bound value
auto work1   = kh.work_tiles({count1}, {1}, 4);         // n/4 groups of 4
auto first   = sum_tiles(kh, work1, x_tiles);           // n/4 result tiles of {1}
auto last    = kh.repeat(d - 1, first, [=](auto& kh, auto prev) {
  auto prev_tiles = kh.retile(prev, {4});
  auto count      = prev_tiles.count();
  auto work       = kh.reuse_work(work1, {count}, {1}, 4);
  return sum_tiles(kh, work, prev_tiles);
});                                                     // 1 result tile of {1}
auto total_elems = kh.join(last);
total = kh.evaluate(total_elems);
```

Work-groups: n/4 groups of 4 (d = 3: 16; d = 5: 256); repeated levels launch none. Changing d changes bound values only.
Placement: as K1 (first and each level's result global, W2).
Trace, x[i] = 1: d = 3: first = 16 x 4; levels give 4 x 16, then 64. d = 5: 256 x 4, 64 x 16, 16 x 64, 4 x 256, 1024. Equal to reference.
Basis: as K1 plus P20. Depends on: Q1, Q4.

### K3 Whole-array inclusive scan, 1024 values, tiles of 64

Problem: y[i] = x[0] + ... + x[i]. Reference: `int s = 0; for (int i = 0; i < 1024; ++i) { s += x[i]; y[i] = s; }` For x[i] = 1: y[i] = i + 1.

Derivation: each tile of 64 is scanned on its own; each tile then needs the totals of all tiles before it, so the 16 tile-items share their totals and belong in one work tile of 16 items with 64 units each (1024 work-items). Steps: scan each tile; each item's total is the last element of its scan, taken by one unit; each item gathers the totals of the items before it (else 0); it sums them into its increment; it adds the increment to its scan.

```cpp
auto plus       = sycl::plus<>{};
auto scan_units = [=](auto unit, auto v) { return sg::inclusive_scan(unit, v, plus); };
auto sum_units  = [=](auto unit, auto v) { return sg::reduce(unit, v, plus); };
auto pass_units = [=](auto unit, auto v) { return v; };
auto add_units  = [=](auto unit, auto s, auto inc) { return s + inc; };

auto x       = kh.source(x_in);                                  // 1024 elements
auto x_tiles = kh.tile(x, {64});                                 // 16 tiles of 64
auto work    = kh.work_tiles({1}, {16}, 64);                     // 1 work tile, 16 items, 64 units

// 1. each item scans its own tile
auto scanned = kh.map_work(work, [=](auto item) {
  auto x_tile = x_tiles[item.id()];
  auto x_run  = sg::flatten(x_tile);
  return item.run(x_run, scan_units);
});                                                              // 1 result tile of {1024}
auto scanned_tiles = kh.retile(scanned, {64});                   // 16 tiles: tile i is item i's run

// 2. each item's total: the last element of its scan, on one unit
auto one_unit = kh.reuse_work(work, {1}, {16}, 1);               // same group, 1 unit per item
auto totals = kh.map_work(one_unit, [=](auto item) {
  auto my_scan  = scanned_tiles[item.id()];
  auto my_last  = sg::slice(my_scan, {63}, {1});
  auto last_run = sg::flatten(my_last);
  return item.run(last_run, pass_units);
});                                                              // 1 result tile of {16}
kh.mark_local(totals);                                           // read by the other items

// 3. for item t, slot s: the total of item s if s < t, else 0
auto totals_elems = kh.join(totals);                             // 16 elements
auto before_map   = kh.define_tiles(16, {64}, [](auto t, auto s) { return s < t ? s : -1; });
auto before       = kh.index_map(before_map, sg::fill{0}, totals_elems);   // 16 tiles of 64

// 4. each item's increment: the sum of the totals before it
auto increments = kh.map_work(work, [=](auto item) {
  auto my_before  = before[item.id()];
  auto before_run = sg::flatten(my_before);
  return item.run(before_run, sum_units);
});                                                              // 1 result tile of {16}

// 5. add each item's increment to its scan
auto inc_tiles = kh.retile(increments, {1});                     // 16 tiles of 1
auto inc_wide  = kh.extend_tiles(inc_tiles, {64}, sg::repeat);   // 16 tiles of 64
auto y = kh.map_work(work, [=](auto item) {
  auto my_scan  = scanned_tiles[item.id()];
  auto my_inc   = inc_wide[item.id()];
  auto scan_run = sg::flatten(my_scan);
  auto inc_run  = sg::flatten(my_inc);
  return item.run(scan_run, inc_run, add_units);
});                                                              // 1 result tile of {1024}
auto y_elems = kh.join(y);                                       // 1024 elements
y_out = kh.evaluate(y_elems);
```

Work-groups: work: 1 group of 16 x 64 = 1024 work-items (at the cap of 1024 on CUDA and Intel). one_unit launches none and uses 16 of its 1024 lanes (P15, Q2). Every step runs in that one group.

| value | placement | rule |
|---|---|---|
| x | global (input) | B1 |
| scanned | value: item i reads only its own run (steps 2 and 5) | W2 (own) |
| totals | local: read by other items, marked | W2, W3, P14 |
| before, scanned_tiles, inc_tiles, inc_wide, x_tiles | views | D3, D4 |
| increments | value: item i reads its own | W2 (own) |
| y | value until evaluate; result memory | W2, B1 |

Trace, x[i] = 1: scanned item i = 1..64; totals[i] = 64; before tile t: 64 in slots 0..t-1, else 0; increments[t] = 64t; y item t, unit p = p + 1 + 64t, so y[k] = k + 1; y[0] = 1, y[1023] = 1024. Equals reference.
Basis: P01-P14, P16-P19 v1; Q2 = a, Q3 = a. Depends on: Q2, Q3; P15 (Q1 is not exercised: one_unit's inputs come from its own group).

## 5. Misfit log

| id | kernel | where | verdict | resolution |
|---|---|---|---|---|
| F1 | K1 | levels 2-3 must run on level 1's groups (W1) | approach right: new work_tiles would launch new groups; no seed primitive runs a work set on existing groups (kh.slice_work keeps Jack's meaning: re-cutting a map's runs to fit the cap) | new P15 kh.reuse_work |
| F2 | K1 | a reused tile fixed to group t would wait on groups 4t..4t+3, which may not have started (prototype: never wait on unclaimed work) | approach right | P15 places a reused tile on the group delivering its last input; alternative put to Jack as Q1 |
| F3 | K3 | items must select their own data tile; item.tile() is the work tile | missing primitive | new P09 item.id() |
| F4 | K3 | the total is the last element of a selected tile (W6) | missing primitive | new P11 sg::slice |
| F5 | K3 | first approach: an index map inside the body over the selected totals | wrong approach: D4's kh.define_tiles and kh.index_map at kernel level express it with host-evaluated exact dependencies | before = index_map(define_tiles(...), fill 0, join(totals)) |
| F6 | K3 | a separate pass-through map returning the gathered values (Jack's step 2) | wrong approach once index maps are views: the view is the selection; step 4 reduces it directly; both steps stay visible | steps 3 and 4 as written |
| F7 | K3 | work sets of different sizes in one launch (1024 lanes, then 16) | approach right; one launch has one work-group size | group size is the largest items x units; smaller work tiles leave lanes idle; put to Jack as Q2 |
| F8 | K1-K3 | inscan to unit 0 of an item: units hold their own values; one_unit's single unit reads lane 63 | approach right | the toolchain moves values within an item (a broadcast); recorded under W2 "own result is a value" |

## 6. Decision register

| id | question | options | default | affects | status |
|---|---|---|---|---|---|
| Q1 | Which group runs a reused work tile? | a: the group that delivers its last input (no waiting, no new groups); b: the user names the groups and the builder rejects any that would wait on a later group | a | P15, K1, K2 | put 2026-11-23 |
| Q2 | One launch has one work-group size: how are work sets of different sizes handled? | a: group size is the largest items x units, smaller work tiles leave lanes idle; b: the builder rejects kernels whose work sets differ (K3 then takes totals as a 64-unit reduce of each x tile) | a | P05, P15, K3 | put 2026-11-23 |
| Q3 | How does a result tile lay out items whose result is a run? | a: each run contiguous along the last dimension (K3: 16 x 64 gives {1024}); b: an extra trailing dimension ({16, 64}) | a | P06, K3 | put 2026-11-23 |
| Q4 | May items that share nothing be packed into one work tile? | a: no, a work tile holds only items that share (K1: 16 groups of 4 work-items); b: yes, as tuning (K1 could be 1 group of 16 items x 4 units) | a | M1, K1, K2 | put 2026-11-23 |

Each option was traced on its dependent kernel: Q1b on K1 with work2 on groups 3, 7, 11, 15 gives 2016; Q2b on K3 with totals as 64-unit reduces gives y[k] = k + 1; Q3b on K3 with retile to {1, 64} gives the same y; Q4b on K1 with one group gives 2016.

## 7. Contact log

| id | date | kind | size | questions | response |
|---|---|---|---|---|---|
| C1 | 2026-11-23 | early packet | see early-packet.md | 4 (Q1-Q4) | pending |
