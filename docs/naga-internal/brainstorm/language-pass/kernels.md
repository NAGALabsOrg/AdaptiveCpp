# Language pass: kernel records

Each record: problem and reference (plain sequential code), derivation in problem terms, code (one primitive application per line, named intermediates), native work-groups, placement with rule ids (record.md section 1), trace. Corrections from the early check X1-X4 apply to every record; basis: definitions v3, Q2 = a, Q3 = b, Q5-Q9 at their defaults.

## Common definitions

```cpp
auto plus       = sycl::plus<>{};
auto sum_items  = [=](auto item, auto v) { return sg::reduce(item, v, plus); };
auto scan_items = [=](auto item, auto v) { return sg::inclusive_scan(item, v, plus); };
auto pass_items = [=](auto item, auto v) { return v; };
auto add_items  = [=](auto item, auto a, auto b) { return a + b; };
auto dot_items  = [=](auto item, auto a, auto b) { return sg::reduce(item, a * b, plus); };
auto sum_tiles  = [=](auto& kh, auto work, auto tiles) {
  return kh.map_work(work, [=](auto unit) {
    auto tile = tiles[unit.tile()];
    auto run  = sg::flatten(tile);
    return unit.run(run, sum_items);
  });
};
```

## K1 Global reduction, 64 in fours

Reference: `float s = 0; for (int i = 0; i < 64; ++i) s += x[i];` x[i] = i: 2016; x[i] = i + 1: 2080.
Derivation: 16 sums of 4 consecutive numbers, one unit of 4 items each; 4 sums of those; 1 sum of those. Each later level runs on the groups of level 1 that produced the last of its inputs.

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
auto total_elems = kh.join(l3);                                   // 1 element
total = kh.evaluate(total_elems);
```

Work-groups: 16 groups of {1, 4}; the slices launch none. Pulls: work2 tile t (work1 tile 4t + 3) pulls tiles 4t..4t+3, lower or its own; work3 (tile 15) pulls tiles 3, 7, 11, 15 (W11).
Placement: x global input (B1); items' values value (W2); l1: tile 4t + 3's own sum value, the other three global (W2); l2: tile 15's own value, the rest global (W2); l3 result memory (B1); x_tiles, l1_tiles, l2_tiles, total_elems views (D3).
Trace x[i] = i: l1 = 6, 22, ..., 246; l2 = 120, 376, 632, 888; 2016. x[i] = i + 1: 10, ..., 250; 136, 392, 648, 904; 2080.

## K2 n-level reduction, depth d a launch value

Reference: as K1; x[i] = 1: n = 4^d.
Derivation: K1's level recorded once; level j runs on work1 tiles with stride 4^j, offset 4^j - 1.

```cpp
auto x       = kh.source(x_in);                                   // n = 4^d elements
auto x_tiles = kh.tile(x, {4});                                   // n/4 tiles of 4
auto count1  = x_tiles.count();                                   // n/4, bound
auto work1   = kh.work_tiles({count1}, {1}, 4);                   // n/4 groups
auto first   = sum_tiles(kh, work1, x_tiles);                     // n/4 result tiles of {1}
auto last    = kh.repeat(d - 1, first, [=](auto& kh, auto prev) {
  auto prev_tiles = kh.retile(prev, {4});
  auto count      = prev_tiles.count();
  auto stride     = count1 / count;
  auto work       = kh.slice_work(work1, {stride - 1}, {count}, {stride}, {1}, 4);
  return sum_tiles(kh, work, prev_tiles);
});                                                               // 1 result tile of {1}
auto total_elems = kh.join(last);
total = kh.evaluate(total_elems);
```

Work-groups: n/4 groups of {1, 4} (d = 3: 16; d = 5: 256); level tile t is work1 tile stride * t + stride - 1, which pulls tiles at or below itself. A new d never recompiles.
Placement: as K1.
Trace x[i] = 1: d = 3: 4, 16, 64. d = 5: 4, 16, 64, 256, 1024.

## K3 Whole-array inclusive scan, 1024 values, tiles of 64

Reference: `int s = 0; for (int i = 0; i < 1024; ++i) { s += x[i]; y[i] = s; }` x[i] = 1: y[i] = i + 1.
Derivation: each tile of 64 is scanned in its own work tile; its total is taken from its own scan; each tile needs the totals of the tiles before it (the only crossing); it sums them and adds the sum to its scan.

```cpp
auto x       = kh.source(x_in);                                   // 1024 elements
auto x_tiles = kh.tile(x, {64});                                  // 16 tiles of 64
auto work    = kh.work_tiles({16}, {1}, 64);                      // 16 groups of 1 unit x 64 items
// 1. each work tile scans its tile
auto scanned = kh.map_work(work, [=](auto unit) {
  auto x_tile = x_tiles[unit.tile()];
  auto x_run  = sg::flatten(x_tile);
  return unit.run(x_run, scan_items);
});                                                               // 16 result tiles of {1, 64}
// 2. each work tile's total, a run of one item on the same work tiles
auto totals = kh.map_work(work, [=](auto unit) {
  auto my_scan  = scanned[unit.tile()];
  auto my_last  = sg::slice(my_scan, {0, 63}, {1, 1});
  auto last_run = sg::flatten(my_last);
  return unit.run(last_run, pass_items);
});                                                               // 16 result tiles of {1}
// 3. for tile t, slot s: the total of tile s if s < t, else 0
auto totals_elems = kh.join(totals);                              // 16 elements
auto before_map   = kh.define_tiles(16, {64}, [](auto t, auto s) { return s < t ? s : -1; });
auto before       = kh.index_map(before_map, sg::fill{0}, totals_elems);   // 16 tiles of 64
// 4. each work tile's increment
auto increments = kh.map_work(work, [=](auto unit) {
  auto my_before  = before[unit.tile()];
  auto before_run = sg::flatten(my_before);
  return unit.run(before_run, sum_items);
});                                                               // 16 result tiles of {1}
// 5. add the increment to the scan
auto inc_wide = kh.extend_tiles(increments, {1, 64}, sg::repeat);   // 16 tiles of {1, 64}
auto y = kh.map_work(work, [=](auto unit) {
  auto my_scan  = scanned[unit.tile()];
  auto my_inc   = inc_wide[unit.tile()];
  auto scan_run = sg::flatten(my_scan);
  auto inc_run  = sg::flatten(my_inc);
  return unit.run(scan_run, inc_run, add_items);
});                                                               // 16 result tiles of {1, 64}
auto y_rows  = kh.retile(y, {64});                                // 16 tiles of 64
auto y_elems = kh.join(y_rows);                                   // 1024 elements
y_out = kh.evaluate(y_elems);
```

Work-groups: 16 groups of {1, 64}; step 2 uses 1 item per group (Q5). Step 3: tile t pulls tiles 0..t-1 of its set (W11).
Placement: x global input; scanned value (pulled only by its own unit, W2); totals value in its group, global for step 3 (W2); increments value (W2); y result memory; views as named (D3, D4).
Trace x[i] = 1: totals 64; increments 64t; y[k] = k + 1; y[0] = 1, y[1023] = 1024.

## K4 1-D 3-point stencil, 1024 values, zero boundary

Reference: `y[i] = (i > 0 ? x[i-1] : 0) + x[i] + (i < 1023 ? x[i+1] : 0);` x[i] = i: y[0] = 1, y[i] = 3i, y[1023] = 2045.
Derivation: each output sums its value and its two neighbours; 64 outputs per work tile, each a unit with a run of 3.

```cpp
auto x       = kh.source(x_in);                                   // 1024 elements
auto star    = kh.stencil({-1, 0, 1}, sg::range(0, 1024));        // 1024 index tiles of 3
auto windows = kh.index_map(star, sg::fill{0}, x);                // 1024 tiles of 3
auto work    = kh.work_tiles({16}, {64}, 3);                      // 16 groups of 64 units x 3 items
auto y = kh.map_work(work, [=](auto unit) {
  auto t      = unit.tile();
  auto u      = unit.id();
  auto i      = 64 * t[0] + u[0];
  auto window = windows[i];
  auto run    = sg::flatten(window);
  return unit.run(run, sum_items);
});                                                               // 16 result tiles of {64}
auto y_elems = kh.join(y);                                        // 1024 elements
y_out = kh.evaluate(y_elems);
```

Work-groups: 16 groups of {64, 3} = 192 work-items. No pulls of results.
Placement: x global input; windows view over x (D4); values value; y result memory.
Trace: y[0] = 0 + 0 + 1 = 1; y[1] = 0 + 1 + 2 = 3; y[1023] = 1022 + 1023 + 0 = 2045.

## K5 Gather from values computed in the kernel

Reference: `src[j] = 2 * s[j]; y[i] = src[idx[i]];` s[j] = j, idx[i] = (7i) mod 256: y[1] = 14, y[37] = 6.
Derivation: src is computed by one work set; the gather chooses src elements from data, so it runs on a later work set and waits on the tiles its indices choose.

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
auto src_rows  = kh.retile(src, {64});                            // 4 tiles of 64
auto src_elems = kh.join(src_rows);                               // 256 elements
auto idx       = kh.source(idx_in);                               // 256 indices
auto idx_tiles = kh.tile(idx, {64});                              // 4 index tiles of 64
auto gathered  = kh.index_map(idx_tiles, sg::fill{0}, src_elems); // 4 tiles of 64, chosen by data
auto work_y    = kh.work_tiles({4}, {1}, 64);                     // 4 more groups, a later set
auto y = kh.map_work(work_y, [=](auto unit) {
  auto g_tile = gathered[unit.tile()];
  auto g_run  = sg::flatten(g_tile);
  return unit.run(g_run, pass_items);
});                                                               // 4 result tiles of {1, 64}
auto y_rows  = kh.retile(y, {64});
auto y_elems = kh.join(y_rows);
y_out = kh.evaluate(y_elems);
```

Work-groups: 8 groups of {1, 64}. work_y pulls src tiles chosen by idx at run time: an earlier set, always allowed; each item waits on its chosen tile's flag (W5, W11). On work_src itself the same gather would make every tile depend on every tile of its own set, a cycle (W11, F14).
Placement: s, idx global inputs; src global (pulled by another set, W2); gathered view; y result memory.
Trace: y[1] = src[7] = 14; y[37] = src[259 mod 256 = 3] = 6.

## K6 Matrix product, 128 x 128

Reference: `C[i][j] = sum over k of A[i][k] * B[k][j];` A[i][k] = i, B[k][j] = j: C[i][j] = 128ij; C[3][5] = 1920, C[127][127] = 2064512.
Derivation (Jack's): output element (i, j) needs row i of A and column j of B; a block of outputs shares rows and columns, so A is tiled in row bands and B in column bands; each output is a unit whose run multiplies its row by its column and reduces. 16 x 16 outputs x 128 items = 32768 work-items exceeds the cap of 1024, so the work tiles are built as 2 x 4 outputs (data tiling unchanged); the segmented form below keeps 16 x 16. The work tile's rows of A and columns of B are staged once on the same groups and marked local.

```cpp
auto A       = kh.source(A_in);                                   // 128 x 128
auto B       = kh.source(B_in);                                   // 128 x 128
auto a_bands = kh.tile(A, {16, 128});                             // 8 x 1 tiles
auto b_bands = kh.tile(B, {128, 16});                             // 1 x 8 tiles
auto work    = kh.work_tiles({64, 32}, {2, 4}, 128);              // 2048 groups of 2 x 4 units x 128 items
// stage this work tile's 2 rows of A
auto a_work  = kh.slice_work(work, {0, 0}, {64, 32}, {1, 1}, {2}, 128);
auto a_stage = kh.map_work(a_work, [=](auto unit) {
  auto t      = unit.tile();
  auto u      = unit.id();
  auto row    = 2 * t[0] + u[0];
  auto a_band = a_bands[{row / 16, 0}];
  auto a_row  = sg::slice(a_band, {row % 16, 0}, {1, 128});
  auto a_run  = sg::flatten(a_row);
  return unit.run(a_run, pass_items);
});                                                               // 64 x 32 result tiles of {2, 128}
kh.mark_local(a_stage);
// stage this work tile's 4 columns of B
auto b_work  = kh.slice_work(work, {0, 0}, {64, 32}, {1, 1}, {4}, 128);
auto b_stage = kh.map_work(b_work, [=](auto unit) {
  auto t      = unit.tile();
  auto u      = unit.id();
  auto col    = 4 * t[1] + u[0];
  auto b_band = b_bands[{0, col / 16}];
  auto b_col  = sg::slice(b_band, {0, col % 16}, {128, 1});
  auto b_run  = sg::flatten(b_col);
  return unit.run(b_run, pass_items);
});                                                               // 64 x 32 result tiles of {4, 128}
kh.mark_local(b_stage);
// each output: its row of A times its column of B
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
auto C_elems = kh.join(C_tiles);                                  // 128 x 128
C_out = kh.evaluate(C_elems);
```

Work-groups: 2048 groups of {2, 4, 128} = 1024 work-items; the stages use 256 and 512 of them. The stages are slices of the same set, pulled only by their own work tile.
Placement: A, B global inputs; a_stage, b_stage local (pulled by other units of their own work tile, marked, W2, W3); products value; C result memory; bands, slices, C_elems views.
Trace: C[3][5]: row 3 is work tile row 1, unit row 1; column 5 is work tile column 1, unit column 1; a_stage row 1 is A row 3 (all 3s); b_stage row 1 is B column 5 (all 5s); 128 x 15 = 1920. C[127][127] = 128 x 127 x 127 = 2064512.

Segmented form, keeping Jack's 16 x 16 outputs per work tile:

```cpp
auto run     = sg::segment_items(32, 4);                          // 128 items in 32 segments of 4
auto work    = kh.work_tiles({8, 8}, {16, 16}, run);              // 64 tiles x 32 segments = 2048 groups of 16 x 16 x 4
auto a_work  = kh.slice_work(work, {0, 0}, {8, 8}, {1, 1}, {16}, run);
auto a_stage = kh.map_work(a_work, [=](auto unit) {
  auto t      = unit.tile();
  auto u      = unit.id();
  auto a_band = a_bands[{t[0], 0}];
  auto a_row  = sg::slice(a_band, {u[0], 0}, {1, 128});
  auto a_run  = sg::flatten(a_row);
  return unit.run_distributed(a_run, pass_items);
});                                                               // 8 x 8 result tiles of {16, 128}
kh.mark_local<sg::segmented>(a_stage);
auto b_work  = kh.slice_work(work, {0, 0}, {8, 8}, {1, 1}, {16}, run);
auto b_stage = kh.map_work(b_work, [=](auto unit) {
  auto t      = unit.tile();
  auto u      = unit.id();
  auto b_band = b_bands[{0, t[1]}];
  auto b_col  = sg::slice(b_band, {0, u[0]}, {128, 1});
  auto b_run  = sg::flatten(b_col);
  return unit.run_distributed(b_run, pass_items);
});                                                               // 8 x 8 result tiles of {16, 128}
kh.mark_local<sg::segmented>(b_stage);
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

Work-groups: 2048 groups of {16, 16, 4} = 1024; the stages use 64 work-items of each segment. Item p of a product run and item p of a stage run fall in the same segment (p / 4), so the segmented stages are pulled within their segment (W12). The reduce in dot_items is joined across 32 segments: the last segment waits on the others' partials (global, W12).
Placement: A, B global inputs; a_stage, b_stage segmented local (W12, P14); per-segment partials global; C result memory.
Run mode: the stages run distributed (item p of the product reads stage position p, in its own segment); run_replicated would give every segment all 128 positions at 32 x the stage work.
Trace: C[3][5]: work tile (0, 0), unit (3, 5); a_stage row 3 is A row 3 (all 3s), b_stage row 5 is B column 5 (all 5s); 32 segment partials of 4 x 15 = 60 each, joined 1920. C[127][127] = 2064512.

## K7 Histogram, 4096 values, 16 bins

Reference: `h[v[i] % 16] += 1;` v[i] = i mod 16: 256 per bin.
Derivation: each chunk of 256 is binned by one unit; bin b is then the sum of bin b over the 16 chunks, which needs every chunk, so it runs on the last tile of the set.

```cpp
auto key       = [](int v) { return v % 16; };
auto bin_items = [=](auto item, auto v) { return sg::tile_bin(item, v, 16, key); };
auto v         = kh.source(v_in);                                 // 4096 elements
auto v_tiles   = kh.tile(v, {256});                               // 16 tiles of 256
auto work      = kh.work_tiles({16}, {1}, 256);                   // 16 groups of 1 unit x 256 items
auto partial = kh.map_work(work, [=](auto unit) {
  auto v_tile = v_tiles[unit.tile()];
  auto v_run  = sg::flatten(v_tile);
  return unit.run(v_run, bin_items);
});                                                               // 16 result tiles of {1, 16}
auto partial_rows  = kh.retile(partial, {16});                    // 16 tiles of 16
auto partial_elems = kh.join(partial_rows);                       // 256: chunk p, bin b at 16p + b
auto by_bin_map    = kh.define_tiles(16, {16}, [](auto b, auto p) { return 16 * p + b; });
auto by_bin        = kh.index_map(by_bin_map, sg::fill{0}, partial_elems);   // 16 tiles of 16
auto last_tile     = kh.slice_work(work, {15}, {1}, {1}, {16}, 16);          // tile 15: 16 units x 16 items
auto hist = kh.map_work(last_tile, [=](auto unit) {
  auto u      = unit.id();
  auto b_tile = by_bin[u[0]];
  auto b_run  = sg::flatten(b_tile);
  return unit.run(b_run, sum_items);
});                                                               // 1 result tile of {16}
auto hist_elems = kh.join(hist);                                  // 16 elements
h_out = kh.evaluate(hist_elems);
```

Work-groups: 16 groups of {1, 256}; the slice uses 256 work-items of group 15. Tile 15 pulls all tiles 0..15 of its set, allowed (W11).
Placement: v global input; partial: tile 15's own value, the rest global (W2); hist result memory.
Trace: every chunk holds each key 16 times, so each partial bin is 16; hist[b] = 16 x 16 = 256 for b = 0 and b = 15.

## K8 A-to-F, 16 values in tiles of 4

Reference (kernel-language-evaluation.md P8): a = x = i; b = a + 1; c = 2a; d = b + c; e = c + 1001; f(g, l) = e(3 - g, l). So b = i + 1, c = 2i, d = 3i + 1, f(g, l) = 2(4(3 - g) + l) + 1001.
Derivation: A to E are per-element chains on the same tile; F reads E's tiles in reverse, which needs its own later set.

```cpp
auto b_items = [=](auto item, auto v) { return v + 1; };
auto c_items = [=](auto item, auto v) { return 2 * v; };
auto e_items = [=](auto item, auto v) { return v + 1001; };
auto per_tile = [=](auto& kh, auto work, auto tiles, auto fn) {
  return kh.map_work(work, [=](auto unit) {
    auto tile = tiles[unit.tile()];
    auto run  = sg::flatten(tile);
    return unit.run(run, fn);
  });
};
auto x       = kh.source(x_in);                                   // 16 elements
auto x_tiles = kh.tile(x, {4});                                   // 4 tiles of 4
auto w       = kh.work_tiles({4}, {1}, 4);                        // 4 groups
auto a = per_tile(kh, w, x_tiles, pass_items);                    // 4 result tiles of {1, 4}
auto b = per_tile(kh, w, a, b_items);
auto c = per_tile(kh, w, a, c_items);
auto d = kh.map_work(w, [=](auto unit) {
  auto my_b  = b[unit.tile()];
  auto my_c  = c[unit.tile()];
  auto b_run = sg::flatten(my_b);
  auto c_run = sg::flatten(my_c);
  return unit.run(b_run, c_run, add_items);
});
auto e   = per_tile(kh, w, c, e_items);
auto w_f = kh.work_tiles({4}, {1}, 4);                            // F: 4 more groups, a later set
auto f = kh.map_work(w_f, [=](auto unit) {
  auto t     = unit.tile();
  auto e_rev = e[{3 - t[0]}];
  auto run   = sg::flatten(e_rev);
  return unit.run(run, pass_items);
});
// each of b, c, d, f: retile to {4}, join, evaluate
auto b_rows  = kh.retile(b, {4});
auto b_elems = kh.join(b_rows);
b_out = kh.evaluate(b_elems);
// likewise c_out, d_out, f_out
```

Work-groups: 8 groups of {1, 4}.
Placement: a, b, c, d values (pulled only by their own unit, W2); e global (pulled by w_f, W2); b, c, d, f also result memory.
Trace: i = 0: b = 1, c = 0, d = 1, f(0, 0) = e(3, 0) = 2 x 12 + 1001 = 1025; i = 5 (g 1, l 1): b = 6, c = 10, d = 16, f = e(2, 1) = 2 x 9 + 1001 = 1019. Equal to the reference.

## K9 Host loop of element maps

Reference: `y = x; repeat 4 times: y = 2y + 1;` x[i] = i, n = 64: 16i + 15; i = 0: 15, i = 4: 79.

```cpp
auto step_items = [=](auto item, auto v) { return 2 * v + 1; };
auto x       = kh.source(x_in);                                   // 64 elements
auto x_tiles = kh.tile(x, {64});                                  // 1 tile of 64
auto work    = kh.work_tiles({1}, {1}, 64);                       // 1 group
auto y       = per_tile(kh, work, x_tiles, step_items);           // 1 result tile of {1, 64}
for (int k = 1; k < 4; ++k) {
  auto prev = kh.retile(y, {64});
  y = per_tile(kh, work, prev, step_items);
}
auto y_rows  = kh.retile(y, {64});
auto y_elems = kh.join(y_rows);
y_out = kh.evaluate(y_elems);
```

Work-groups: 1 group of {1, 64}; each step on the same work tile. Placement: every intermediate value (own unit, W2). Trace: 2i + 1, 4i + 3, 8i + 7, 16i + 15.

## K10 Run-time-declared component

Reference: `y[i] = a * x[i] + b;` for inputs and n named at run time.

```cpp
auto axb_items = [=](auto item, auto v) { return a * v + b; };   // a, b run-time values
auto x_in    = ins.get<float, 1>("x");                            // declared at run time
auto x       = kh.source(x_in);                                   // n elements
auto x_tiles = kh.tile(x, {256});                                 // n/256 tiles (n divisible, Q6)
auto count   = x_tiles.count();
auto work    = kh.work_tiles({count}, {1}, 256);
auto y       = per_tile(kh, work, x_tiles, axb_items);            // n/256 result tiles of {1, 256}
auto y_rows  = kh.retile(y, {256});
auto y_elems = kh.join(y_rows);
outs.set("y", kh.evaluate(y_elems));
```

Work-groups: n/256 groups of {1, 256}, bound. Placement: values; y result memory.
Trace: (a, b, n) = (2, 1, 256): y[3] = 7; (-1, 0.5, 512): y[10] = -9.5.

## K11 Inplace contributions

Reference: two kernels add 1 and 2 per element into y (initializer 0, plus): 3 everywhere.

```cpp
// in each of two submissions, out = sg::array<sg::result<float>, 1>(sg::inplace(y_handle, plus, 0), sg::shape<1>({64}))
auto scale_items = [=](auto item, auto v) { return k * v; };     // k = 1, then 2
auto ones        = kh.source(ones_in);                            // 64 ones
auto ones_tiles  = kh.tile(ones, {64});
auto work        = kh.work_tiles({1}, {1}, 64);
auto c           = per_tile(kh, work, ones_tiles, scale_items);
auto c_rows      = kh.retile(c, {64});
auto c_elems     = kh.join(c_rows);
out = kh.evaluate(c_elems);
```

Work-groups: 1 group of {1, 64} per submission. Placement: values; the two contributions combine element by element (B2).
Trace: 0 + 1 + 2 = 3 at elements 0 and 63.

## K12 Interpolation

Reference: `y[i] = sum over s of w[4i + s] * src[nbr[4i + s]];` src[j] = j (N = 64), nbr[4i + s] = (i mod 61) + s, w = 0.25, M = 256: y[0] = 1.5, y[100] = 40.5.

```cpp
auto src       = kh.source(src_in);                               // 64 values
auto nbr       = kh.source(nbr_in);                               // 1024 indices
auto wts       = kh.source(w_in);                                 // 1024 weights
auto nbr_tiles = kh.tile(nbr, {4});                               // 256 index tiles of 4
auto vals      = kh.index_map(nbr_tiles, sg::fill{0.f}, src);     // 256 tiles of 4
auto w_tiles   = kh.tile(wts, {4});                               // 256 tiles of 4
auto work      = kh.work_tiles({4}, {64}, 4);                     // 4 groups of 64 units x 4 items
auto y = kh.map_work(work, [=](auto unit) {
  auto t     = unit.tile();
  auto u     = unit.id();
  auto i     = 64 * t[0] + u[0];
  auto v_t   = vals[i];
  auto w_t   = w_tiles[i];
  auto v_run = sg::flatten(v_t);
  auto w_run = sg::flatten(w_t);
  return unit.run(v_run, w_run, dot_items);
});                                                               // 4 result tiles of {64}
auto y_elems = kh.join(y);                                        // 256 elements
y_out = kh.evaluate(y_elems);
```

Work-groups: 4 groups of {64, 4}. src is an input, so the data-chosen indices need no wait. Placement: inputs global; vals view; values; y result memory.
Trace: y[0] = 0.25(0 + 1 + 2 + 3) = 1.5; y[100]: 100 mod 61 = 39: 0.25(39 + 40 + 41 + 42) = 40.5.

## K13 2-D 5-point stencil, 64 x 64, zero boundary

Reference: `y[r][c] = x[r][c] + x[r-1][c] + x[r+1][c] + x[r][c-1] + x[r][c+1]` (0 outside); x[r][c] = 64r + c: y[0][0] = 65, y[1][1] = 325.

```cpp
auto x       = kh.source(x_in);                                   // 64 x 64
auto star    = kh.stencil({{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}}, sg::range({0, 0}, {64, 64}));   // 64 x 64 index tiles of 5
auto windows = kh.index_map(star, sg::fill{0}, x);                // 64 x 64 tiles of 5
auto work    = kh.work_tiles({8, 4}, {8, 16}, 5);                 // 32 groups of 8 x 16 units x 5 items
auto y = kh.map_work(work, [=](auto unit) {
  auto t      = unit.tile();
  auto u      = unit.id();
  auto r      = 8 * t[0] + u[0];
  auto c      = 16 * t[1] + u[1];
  auto window = windows[{r, c}];
  auto run    = sg::flatten(window);
  return unit.run(run, sum_items);
});                                                               // 8 x 4 result tiles of {8, 16}
auto y_elems = kh.join(y);                                        // 64 x 64
y_out = kh.evaluate(y_elems);
```

Work-groups: 32 groups of {8, 16, 5} = 640 work-items (16 x 16 x 5 = 1280 would exceed 1024). Placement: x global input; windows view; values; y result memory.
Trace: y[0][0] = 0 + 0 + 64 + 0 + 1 = 65; y[1][1] = 65 + 1 + 129 + 64 + 66 = 325.

## K14 Row sums, 64 rows x 1024 columns

Reference: `y[r] = sum over c of x[r][c];` x[r][c] = r + c: y[r] = 1024r + 523776; y[0] = 523776, y[63] = 588288.

At cap 1024:

```cpp
auto x    = kh.source(x_in);                                      // 64 x 1024
auto rows = kh.tile(x, {1, 1024});                                // 64 x 1 tiles
auto work = kh.work_tiles({64}, {1}, 1024);                       // 64 groups of 1 unit x 1024 items
auto sums = kh.map_work(work, [=](auto unit) {
  auto t   = unit.tile();
  auto row = rows[{t[0], 0}];
  auto run = sg::flatten(row);
  return unit.run(run, sum_items);
});                                                               // 64 result tiles of {1}
auto sums_elems = kh.join(sums);                                  // 64 elements
y_out = kh.evaluate(sums_elems);
```

At cap 256, built segmented; the body is unchanged:

```cpp
auto x    = kh.source(x_in);                                      // 64 x 1024
auto rows = kh.tile(x, {1, 1024});                                // 64 x 1 tiles
auto run  = sg::segment_items(4, 256);                            // 1024 items in 4 segments of 256
auto work = kh.work_tiles({64}, {1}, run);                        // 64 tiles x 4 segments = 256 groups of 1 x 256
auto sums = kh.map_work(work, [=](auto unit) {
  auto t   = unit.tile();
  auto row = rows[{t[0], 0}];
  auto r   = sg::flatten(row);
  return unit.run_distributed(r, sum_items);
});                                                               // 64 result tiles of {1}
auto sums_elems = kh.join(sums);
y_out = kh.evaluate(sums_elems);
```

Work-groups: cap 1024: 64 groups of {1, 1024}. Cap 256: 64 work tiles x 4 segments = 256 groups of {1, 256}; the reduce is joined across each tile's 4 segments (the last segment waits on the others, global). The rows tiling and the body are unchanged.
Placement: x global input; each segment's partial global (crosses segments, D10); sums result memory.
Trace: y[0] = 523776; y[63] = 63 x 1024 + 523776 = 588288. Cap 256, row 0: segment partials 32640, 98176, 163712, 229248, joined 523776.
