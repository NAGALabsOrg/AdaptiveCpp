# Kernel language proposal: nodes and shaped edges

## For stage 2 of sg-redesign-brainstorm
This is the input to stage 2 (ergonomics) of plan sg-redesign-brainstorm. Every change below names the entry of that plan's language baseline (01-language-baseline) it differs from, so the change register can be filled from it directly. It was produced by plan sg-language-search; the candidates, the stress programs, the criteria and every verdict are in kernel-language-evaluation.md beside this file.

## The language (U36)
A kernel is a graph of nodes built with the kernel builder, the same way the application graph is built with the graph builder.

- **Node.** `kh.node(units, edges..., body)` defines work as a grid of units (1-3 dimensions, launch-resolvable). The body is a pure function of what the edges deliver to one unit, and it returns that unit's product: a value, or an array whose shape is launch-resolvable. The product is the node's unit shape.
- **Source.** `kh.source(array)` makes an input array a node whose units are its elements. Input arrays may also be captured and indexed directly in bodies (gathers).
- **Edge.** `sg::from(upstream, form[, mode])` is one body argument. The form is the topology, a mapping from this unit to upstream units with a fixed count: `one`, `tiles<K...>`, `rows<K>`, `cols<K>`, `window<K, H>(boundary)`, `all`, `ids(index array)` (positions from data), `units_fn(f)` (a pure positions function), and `before(op, select)` (the combine of a selected part of every earlier upstream unit's product). What arrives is a dense array of (fan-in x upstream unit shape), so four single values arrive as an array of four.
- **Mode** (how the upstream work is carried out): `parallel` (upstream units run as their own parallel work; their products pass through global memory; the default for partition forms), `fused` (this unit computes its upstream units itself; nothing stored; recomputed per consumer), `shared` (computed once and stored; the default for overlapping and data forms).
- **Body vocabulary**, on arrays: element-wise arithmetic and comparisons, `map`, `map_index`, `reduce` (whole or along a dimension), `scan`, `contract`, `bin_count`, boundary reads `at_or`. The toolchain decides how many work-items a unit uses and lowers these to group algorithms or sequential code.
- **Results.** `out = kh.evaluate(node)`: the node's products stitched over its unit grid.
- **Sugar.** `kh.map(src, f)` (a one/fused element-wise node); `kh.repeat(n, start, fn)` (a node applied n times with the level as a bound value); `kh.reduce_tree<K>(src, op)` (repeat of a tiles<K> reduce node).
- Unchanged: the builder frame, roles, evaluate, keyed builders, inplace, one kernel definition is one launch.

## The stress programs in it (U37)
- P1, global reduction 64 in fours:
```cpp
auto sum = [](auto t) { return sg::reduce(t, sycl::plus<>{}); };
auto x  = kh.source(in);
auto l1 = kh.node(16, sg::from(x,  sg::tiles<4>), sum);
auto l2 = kh.node(4,  sg::from(l1, sg::tiles<4>), sum);
auto l3 = kh.node(1,  sg::from(l2, sg::tiles<4>), sum);
out = kh.evaluate(l3);
```
- P2, depth a launch value: `out = kh.evaluate(kh.repeat(d, kh.source(in), [=](auto& kh, auto prev) { return kh.node(prev.units() / 4, sg::from(prev, sg::tiles<4>), sum); }));`
- P3, whole-array scan: `auto t = kh.node(16, sg::from(x, sg::tiles<64>), [](auto v) { return sg::scan(v, plus); }); auto y = kh.node(16, sg::from(t, sg::one), sg::from(t, sg::before(plus, sg::last)), [](auto s, int off) { return s + off; });`
- P4, stencil: `auto y = kh.node(16, sg::from(x, sg::window<64, 1>(sg::zero)), [](auto w) { return sg::map_index(64, [=](int i) { return w[i] + w[i + 1] + w[i + 2]; }); });`
- P5, gather: `auto y = kh.node(4, sg::from(kh.source(idx), sg::tiles<64>), [=](auto ix) { return sg::map(ix, [=](int j) { return src[j]; }); });`
- P6, tiled GEMM: `auto c = kh.node({8, 8}, sg::from(kh.source(A), sg::rows<16>), sg::from(kh.source(B), sg::cols<16>), [](auto a, auto b) { return sg::contract(a, b); });`
- P7, histogram: `auto part = kh.node(16, sg::from(kh.source(v), sg::tiles<256>), [](auto t) { return sg::bin_count<16>(t); }); auto hist = kh.node(1, sg::from(part, sg::tiles<16>), [](auto p) { return sg::reduce(p, 0, plus); });`
- P8, A-to-F: `a = kh.node(4, from(x, tiles<4>), fa); b = kh.node(4, from(a, one, fused), fb); c = kh.node(4, from(a, one, fused), fc); d = kh.node(4, from(b, one, fused), from(c, one, fused), fd); e = kh.node(4, from(c, one), fe); f = kh.node(4, from(e, units_fn([](int g) { return 3 - g; })), ff);` then evaluate b, c, d, f.
- P9, host loop: `auto y = kh.source(in); for (int k = 0; k < 4; ++k) y = kh.map(y, [](float v) { return 2 * v + 1; }); out = kh.evaluate(y);`
- P10, run-time component: `auto x = kh.source(ins.get<float, 1>("x")); outs.set("y", kh.evaluate(kh.map(x, [=](float v) { return a * v + b; })));`
- P11, inplace: two submissions, each `out = kh.evaluate(kh.map(kh.source(ones), [=](float v) { return k * v; }));` with out = inplace(y, plus, 0), k = 1 and 2.
No program uses a native-kernel term. P1 shows three nodes joined by edges that state the fan-in.

## What changes from the baseline (U35)
| Baseline entry (01-language-baseline) | Change |
|---|---|
| Builder frame | kept |
| Roles | kept |
| kh.parallel_for | replaced by kh.node (units, edges, body) |
| kh.map_groups and mapping forms | replaced by sg::from(upstream, form, mode); forms gain before(op), window, rows/cols, all |
| evaluate | kept |
| item mapping | replaced by form one with mode fused |
| local mapping | replaced by mode fused |
| global mapping | replaced by modes parallel and shared |
| item body | removed: merged into the unit body (map, map_index) |
| group body | changed: becomes the unit body; no item/group split |
| group expressions | changed: no rank template; adds map, map_index, contract, bin_count, comparisons, at_or |
| keyed builders | kept |
| added | kh.source, kh.map, kh.repeat, kh.reduce_tree |
Core items: builder frame kept; structure vs bound values kept (repeat levels, unit counts, sizes bound); one definition per launch kept; pure bodies kept; declaration order kept. Explicit departure of core 5 (memory only where the user says): parallel edges store unit products by default; consequence: a fan-in edge costs one stored product per upstream unit unless marked fused.

## Why it is sound to implement (U38)
- One launch, as a dispatch loop: each native group takes a ticket from an atomic counter, runs one unit through a uniform switch over node id, stores its product where a parallel or shared edge or a result needs it, then, for each parallel partition edge, increments the consumer unit's counter (inverse of the form, e.g. u / K) and continues with the consumer if it delivered the last input. Nobody waits on a partition edge. Shared and data edges wait on an upstream node-done counter, and tickets are issued in topological node order, so every wait is on work a running group has claimed (decoupled look-back's rule).
- R1 nesting cost: no node's code contains another node's code except through an explicit fused edge; one switch case per node: code size linear in node count.
- R2 CBS: on the host the whole switch inlines into the kernel once; each case is taken on a broadcast value, so barriers stay in uniform control flow; values crossing barriers are ordinary (after the 0e2bac97 fix).
- R3 forward progress: partition edges never wait (last arriver); other waits only on claimed work; correct with serialised workers (OpenMP, one thread) and under occupancy-bound execution.
- R4 Intel compiler: kernels are one regular switch of small unit functions instead of one 150k-line function; the cause of the IGC crash is unknown, so this is an expectation to test, not a guarantee.
- R5 composition: unit functions are body sources compiled at build time and dispatched at JIT time; the dispatch can be built as an indexed call sequence (the fork's dynamic-function extension) or by the composition step; identity is the structure (nodes, forms, modes), counts are bound.
- Purity: bodies are closures checked by BodyChecker from a new entry annotation (also closing today's gap for positions functions). Topology at launch: forms have fixed counts; data forms are bound. Code growth: linear in nodes plus one copy per fused edge.

## The candidates that lost (U39)
- B, named patterns (Futhark, Lift, SkePU line). Core idea: dependencies implied by split, map, reduce, scan, slide, gather, histogram. Lost on C2 (unit hidden in scan, slide, gather, histogram: P3, P4, P5, P7), C3 (arbitrary positions not expressible: P8) and C7 (placement only readable from per-pattern rules: all programs). Kept: patterns as sugar over nodes (kh.map, kh.reduce_tree).
- C, index relations (Halide, blockwise, polyhedral line). Core idea: each array element defined by an expression over slices; dependencies derived by probing. Lost on C2 and C7 (units and materialisation are the toolchain's: all programs) and C3/C5 (bodies must be generic over a probe type; relations re-derived every launch). Kept: at_or boundary reads.

## Open questions for Jack (U40)
1. Default mode for partition edges. Options: parallel (default here: the tree runs in parallel, products stored); fused (nothing stored, the consumer computes its inputs, a reduction becomes serial unless marked). Changes: what the reduction costs by default.
2. Whether placement is visible at all. Options: modes as above (default); no modes, toolchain decides by rules. Changes: W1 purity versus C7 (performance readable from source).
3. Merging item and group bodies into one unit body. Options: merge (default); keep both. Changes: whether "item" survives as a concept; per-element code is written with map / map_index.
4. The prefix edge before(op, select). Options: an edge form (default); a named node pattern (kh.scan). Changes: whether a combine may live on an edge.
5. Depth as structure or bound value. Options: kh.repeat (default, recompiles never); host loop only (each depth a new structure and a recompile). Changes: cache behaviour for launch-valued depth.
6. Fan-in as a template argument or a launch value. Options: template (default for tiles<K>, compile-time shape checks); launch value (fewer recompiles, checks when the builder runs). Changes: when shape errors are reported.
7. Names: node, source, from, units, parallel / fused / shared. Options: these (default) or others. Changes: vocabulary only.

## Decisions in discussion (stage 2)

### D1 Native shape: derivable, never required (Jack, 2026-11-23)
- A node needs a group shape only when it has an edge with fan-in (tiles, window, rows, cols, all, ids, before) or a collective in its body (reduce, scan, contract, bin_count). A kernel with no such node lowers to a plain range parallel_for; the work-group size is the runtime's.
- When a kernel needs groups: one unit runs on one native work-group (from a ticket or as a continuation); a unit's width is the size of its largest delivered input or of its product, whichever is larger; the native group size is the largest width in the kernel, and units of smaller width leave items idle; units with no fan-in and no collective are packed into those groups by a fixed rule; the grid is the number of units that start from a ticket.
- A width (tiling) override is optional configuration on a node, never required.
- The user never has to state a work-group shape, but can always derive the native shape from the source by these rules.
- Replaces: "the toolchain decides how many work-items a unit uses" in The language, and the tiled rewrite of kh.map discussed for kernel 1 (kh.map stays without a tile).

### D2 All sizes are runtime values (Jack, 2026-11-23)
- Fan-ins, tile and window sizes, bin counts and tree fan-ins are runtime arguments: sg::tiles(4), sg::window(64, 1), sg::bin_count(16), kh.reduce_tree(x, 4, op). No sizes as template parameters. Ranks stay in the types; extents are dynamic.
- They are bound values: changing them never recompiles. Shape errors are reported when the builder runs, naming the node. Constant specialisation, where it pays, is the JIT's (AdaptiveCpp adaptivity), never the user's.
- Settles open question 6 (launch values). All tiles<K>, window<K, H>, bin_count<16>, reduce_tree<K> in this document read as their runtime forms.

### D3 Elements and tiles are explicit types (Jack, 2026-11-23)
- `sg::tile(x, k)` turns elements into tiles. A map over tiles always returns tiles, even tiles of one value. `sg::retile(t, k)` regroups tiles. `sg::join(t)` is the only way back to elements. `evaluate` takes elements.
- Tile size is logical. It coincides with the native group shape only in simple kernels; the native shape follows D1.
- Generic topology: `sg::gather_tiles(src, shape, positions)`, where `positions(k, s)` returns, for destination tile k and slot s, a source position or `sg::fill(v)`. It is not tied to any algorithm. The builder evaluates it on the host to find which source tiles each destination tile needs (bound data).
- `sg::subset(t, which)` picks an arbitrary subset of a tile (`sg::last`, index-space forms, masks). `sg::extend_tiles(t, k, how)` reshapes a map argument's tiles to line up with the others (`sg::repeat`).
- A map with several arguments takes tiled arguments with equal tile counts; their tile shapes are independent. Mismatched counts are a builder error naming the map.
- Style: algorithms are written one visible step per layer, never compressed into minimal lines.
- Replaces: kh.node, sg::from and its forms, the modes, and `before`. Names subset, gather_tiles, extend_tiles are placeholders.

Reduction:
```cpp
auto x  = kh.source(x_in);                            // 64 elements
auto l1 = kh.map(sg::tile(x, 4), sum);                // 16 tiles of 1
auto l2 = kh.map(sg::retile(l1, 4), sum);             // 4 tiles of 1
auto l3 = sg::join(kh.map(sg::retile(l2, 4), sum));   // 1 element
total = kh.evaluate(l3);
```

Scan:
```cpp
auto scanned           = kh.map(sg::tile(x, 4), [=](auto t) { return sg::inclusive_scan(t, plus); });  // 4 tiles of 4
auto overlaps          = kh.map(scanned, [](auto t) { return sg::subset(t, sg::last); });             // 4 tiles of 1
auto gathered_overlaps = sg::gather_tiles(overlaps, sg::shape(4),
    [](auto k, auto s) { return s < k ? sg::from_tile(s) : sg::fill(0); });                           // 4 tiles of 4
auto scanned_overlaps  = kh.map(gathered_overlaps, [=](auto t) { return sg::inclusive_scan(t, plus); });
auto increments        = kh.map(scanned_overlaps, [](auto t) { return sg::subset(t, sg::last); });    // 4 tiles of 1
auto y = sg::join(kh.map(scanned, sg::extend_tiles(increments, 4, sg::repeat),
                         [](auto t, auto inc) { return t + inc; }));                                  // 16 elements
out = kh.evaluate(y);
```

### D4 Index maps (Jack, 2026-11-23)
- Every builder call is on `kh`, views included: `kh.tile`, `kh.retile`, `kh.join`, `kh.extend_tiles`, `kh.index_map`, `kh.stencil`, `kh.define_tiles`. Body vocabulary (`sg::reduce`, `sg::inclusive_scan`, `sg::subset`, `sg::bin_count`) stays in `sg`. D3's `sg::` builder spellings read as `kh.`.
- An index map is tiles of indices into an element array. `kh.index_map(map, oob_policy, elements)` is the only thing that applies one; it gives a view of tiles of values. The tiles come from the map, not from the source, so counts need not match the source (interpolation: M targets drawing on N sources).
- Where index maps come from: `kh.define_tiles(tiles, tile_shape, [](auto tile, auto local) { return source_index; })`, computed on the fly; `kh.stencil(offsets, roots)`, a helper that builds an index map (roots a range, index elements, or every element); an index array from data, tiled with `kh.tile`.
- Out-of-bounds policy is given where the map is applied: `sg::fill{v}`, `sg::wrap{}`. Any index outside the source goes to the policy; there is no separate padding sentinel.
- Dependencies: closed-form for stencil and range helpers. `define_tiles` functions are evaluated by the builder on the host once per builder run to build an exact dependency table (default), skipped when a keyed builder reuses its tree. Index arrays from data are opaque: a consumer tile depends on all of the source; this costs only when the source is computed in the kernel, and waits only on claimed work, still one launch.
- Views (index maps included) never hold values; what is stored is the lowering's and `evaluate`'s.
- Replaces: `gather_tiles` (D3).
- Open: the D1 width rule when tiles are wider than the products (stencil tiles of 3).

Stencil:
```cpp
auto x       = kh.source(x_in);                                   // 16 elements
auto stencil = kh.stencil({-1, 0, 1}, sg::range(0, 16));         // 16 tiles of 3 indices
auto windows = kh.index_map(stencil, sg::fill{0}, x);             // 16 tiles of 3 values
auto y = kh.join(kh.map(windows, [](auto w) { return sg::reduce(w, sycl::plus<>{}); }));  // 16 elements
out = kh.evaluate(y);
```

Scan, step 3:
```cpp
auto overlaps   = kh.join(kh.map(scanned, [](auto t) { return sg::subset(t, sg::last); }));  // 4 elements
auto before_map = kh.define_tiles(4, 4, [](auto t, auto l) { return l < t ? l : -1; });
auto gathered   = kh.index_map(before_map, sg::fill{0}, overlaps);                            // 4 tiles of 4
auto increments = kh.map(gathered, [=](auto t) { return sg::reduce(t, plus); });             // 4 tiles of 1
```

Interpolation (new stress program P12):
```cpp
auto src_vals = kh.source(src_in);                                // N source node values
auto nbrs     = kh.source(neighbours_in);                         // M*4 indices
auto weights  = kh.source(weights_in);                            // M*4 weights
auto gathered = kh.index_map(kh.tile(4, nbrs), sg::fill{0}, src_vals);   // M tiles of 4
auto y = kh.join(kh.map(gathered, kh.tile(4, weights),
                        [](auto v, auto w) { return sg::reduce(v * w, sycl::plus<>{}); }));   // M elements
out = kh.evaluate(y);
```

### D5 Data-chosen destinations inside a tile (Jack, 2026-11-23)
- `sg::tile_bin(bin_shape, key)` returns a tile of counts per bin. `sg::tile_bin(bin_shape, key, value_fn, initial, op)` returns a tile of combined values per bin (weighted histograms, deposition, segmented sums).
- `bin_shape` may be multi-dimensional; `key` returns an index of that rank.
- `sg::categorize(cond..., v)` returns the index of the first true condition. Several true conditions is defined (the first wins) but is likely a logic error the toolchain cannot detect. A value with no true condition, or a key outside the bins, is dropped.
- `initial` is applied once per bin per tile; where partial results are combined across tiles it should be the operation's identity (the user's contract).
- Replaces `bin_count`.

Histogram:
```cpp
auto v       = kh.source(v_in);                                                   // 4096 elements
auto partial = kh.map(kh.tile(256, v), [](auto t) {
                 return sg::tile_bin(16, [](int x) { return x % 16; }); });        // 16 tiles of 16 counts
auto by_bin_map = kh.define_tiles(16, 16, [](auto b, auto p) { return p * 16 + b; });
auto by_bin  = kh.index_map(by_bin_map, sg::fill{0}, kh.join(partial));            // 16 tiles of 16
auto hist    = kh.join(kh.map(by_bin, [](auto t) { return sg::reduce(t, sycl::plus<>{}); }));  // 16 elements
out = kh.evaluate(hist);
```
