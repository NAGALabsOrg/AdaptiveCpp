# Early check: the reductions and the scan

Three kernels under the work-tile model, each traced to its reference. Full record: record.md beside this file.
Four questions at the end; "defaults" is a full answer.

## Global reduction, 64 in fours
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
- Sums share nothing, so each work tile is one item with 4 units: 16 native groups of 4 work-items.
- Levels 2 and 3 reuse work1's groups: a reused tile runs on the group that delivers its last input, so nothing waits and no groups are added (Q1).
- l1 and l2 cross work tiles, so they live in global memory; each unit's value stays in a register.
- Traced: x[i] = i gives 6, 22, ..., 246, then 120, 376, 632, 888, then 2016.

## n-level reduction, depth d a launch value
```cpp
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
- One level recorded once; d and every level's count are bound values, so a new depth never recompiles. Traced: d = 3 gives 64, d = 5 gives 1024.

## Whole-array scan, 1024 values, tiles of 64
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
- The 16 items share their totals, so they are one work tile: 1 native group of 16 x 64 = 1024 work-items, at the cap.
- Every step runs in that one group: scanned and increments stay in registers (each item reads only its own); totals is marked local because other items read it.
- Traced: x[i] = 1 gives totals 64, increments 64t, y[k] = k + 1.

## Primitives used (new in this pass: item.id, sg::slice, sg::inclusive_scan, kh.reuse_work)
| name | meaning | side |
|---|---|---|
| kh.source(array) | the input array as elements | host |
| kh.tile(x, shape) / .count() | x's elements in row-major tiles of shape; the tile count | host view |
| kh.retile(t, shape) | t's elements in tile order, regrouped | host view |
| kh.join(t) | tiles stitched back into elements | host view |
| kh.work_tiles(grid, tile_shape, units) | one native group per grid point; tile_shape items, each a run of units | host |
| kh.map_work(work, body) | body runs on the host per item; one result tile per work tile, runs contiguous (Q3) | host |
| item.run(inputs..., fn) | fn on every unit with its element of each 1-D input; one value after a reduce, else a run | fn device |
| item.tile() / item.id() | the work tile / the item's position in it | host |
| sg::flatten(tile) / sg::slice(tile, offset, extent) | the tile as 1-D / part of a tile | host view |
| sg::reduce / sg::inclusive_scan(unit, v, op) | combined over all units / over units up to this one | device |
| kh.mark_local(result) | held in local memory for other items of the same work tiles | host |
| kh.reuse_work(work, grid, tile_shape, units) | launches no groups; a tile runs on the group delivering its last input (Q1) | host |
| kh.define_tiles, kh.index_map, kh.extend_tiles, kh.evaluate, kh.repeat | as agreed (D3, D4, repeat) | host |

## Questions (defaults marked)
1. Which group runs a reused work tile? (a, default) the one that delivers its last input: no waiting, no new groups; (b) you name the groups, and the builder rejects any that would wait on a later group.
2. One launch has one work-group size. (a, default) it is the largest items x units, and smaller work tiles leave lanes idle; (b) work sets of different sizes are rejected (the scan then takes totals as a 64-unit reduce of each tile).
3. Result layout for items whose result is a run: (a, default) runs contiguous along the last dimension (16 x 64 gives 1024); (b) an extra dimension ({16, 64}).
4. May items that share nothing be packed into one work tile? (a, default) no: the reduction is 16 groups of 4 work-items; (b) yes, as tuning (one group of 16 items x 4 units).
