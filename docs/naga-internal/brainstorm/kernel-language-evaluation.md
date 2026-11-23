# Kernel language search: evaluation record

Working record of plan sg-language-search (Research Lead, carried out alone). Instrument fixed 2026-11-23, before any candidate was defined.

## Frame

### U1 Baseline language
The baseline is the draft "01 Language baseline" of plan sg-redesign-brainstorm (builder frame of 2026-11-19, topics closed in note toolchain-impl-design, interfaces of the sg-core plan section 4, prototype at e7b98d97). Constructs: builder frame (gh.define_kernel(fn(kh)), real host code at run); roles (array input, array<result<T>> result, inplace, host_array); kh.parallel_for (item body: range, group shape, maps, body; group body: group range, >=1 map, body); kh.map_groups<kind>(source, positions) with forms single, block, strided, stencil, ids, index_map; out = kh.evaluate(def); item mapping (value, recomputed per access, same group shape); local mapping (group_array in local memory, computed once per destination group); global mapping (own native groups, global memory, shared, claim/shortcut/wait); item body (item handle, group_array indexing, sequential reduce); group body (group handle, lazy group_reduce / group_inclusive_scan / group_exclusive_scan / element-wise, dim host-resolvable, known-identity ops only); keyed builders (define_kernel(tuple, fn)).
Stale (predate the builder frame, not relied on): docs/naga-internal/01-kernel-language.md sections 1 (nothing runs on the host), 5 (sources{source_map...}, counts may differ per target group), 8 (handle-less durable arrays, reading durable arrays), 9 (scalar evaluate), 10 (make_cycle form, make_chain), 11 (recurse, transform/map/select vocabulary as first-version scope); 02-graph-api.md sections 3 (sized handles, free, realloc, global tracker, events per handle), 4 (overwrite warning), 6 (get_host_handle, realloc), 12 (report); 03-vocabulary.md sections 1, 4, 7, 8, 10 where they name the above.

### U2 Core items (departure form: item / change / consequence)
1. Builder frame: the kernel is built by host code calling a kernel handle. Departing means a declarative or compiler-parsed kernel description; consequence: the frontend must parse kernel structure, host control flow no longer builds structure.
2. Structure vs bound values: which units and links exist is structure (compiled, cached); counts, sizes, addresses, launch values are bound. Departing means some counts become structure; consequence: recompiles when they change.
3. One kernel definition is one launch. Departing means a definition may lower to several launches; consequence: host-side sequencing, launch latency, but forward-progress problems disappear.
4. Pure body sources, checked statically. Departing means effects in bodies; consequence: ordering and recomputation become observable.
5. Compute is cheap, memory is not; durability only where the user says. Departing means the toolchain materialises on its own; consequence: hidden memory cost.
6. Declaration order matters. Departing means order-independent semantics; consequence: canonicalisation for hashing.

### U3 Prototype lessons as lowering rules
R1 (nesting cost, cuda chain_30 321 s / 228k IR lines, quadratic): a lowering must not copy an upstream stage's code into every stage that can reach it; generated code size must be linear in node count.
R2 (CBS): on the CPU every barrier-containing callee is inlined into the kernel and values crossing a barrier must be safe to store or recompute; barriers only in group-uniform control flow; the lowering must tolerate full inlining on the host target.
R3 (forward progress): a group may wait inside one launch only on work a running group has claimed or on groups that precede it in an order the backend guarantees (OpenMP: static schedule); no co-residency assumption; mass spinning is costly on Intel OpenCL; Level Zero hung from 20 layers with real calls (cause unknown).
R4 (Intel IGC crash on a small protocol kernel, layers_8_16_32): generated kernels should be regular and small; cause unknown, so no rule beyond "keep a fallback lowering".
R5 (composition from separately compiled pure pieces): body sources compiled at build time and composed at JIT time; identity by structure; every generated placeholder use must be type-correct for the node's kind; 8/16-bit atomics are not portable and are emulated on 32-bit words.

### U4 Jack's wants (yes/no)
W1 No native-kernel term appears in user code. W2 A kernel is a graph of nodes built with a builder, like the application graph. W3 A node defines work as a function of its input edges. W4 An edge is a topological mapping to an upstream node together with the shape of what that node's unit of work returns. W5 The 64-in-4s reduction reads as layer1 -map12-> layer2 -map23-> layer3.

### U5 Kernel graph vs application graph
- Node: distinct. App node = one piece of work (kernel, host work); kernel node = one unit-of-work definition. Interface: the kernel graph is the body of one app node.
- Edge: distinct. App edges are implied by roles on handles (never written); kernel edges are written mappings. Interface: app-level arrays are the kernel graph's leaves (inputs) and sinks (results).
- Builder: shared pattern (gh in submit, kh in define_kernel); distinct handles.
- Cache key: shared notion (structural hash; bound values excluded); distinct keys (host program vs composed kernel).
- Memory handle: app level only; the kernel sees arrays bound to handles.

### U6 Computation catalogue (problem level)
1. Elementwise map: each output element from the same-index inputs. Dependence: none between units. Unit product: one element.
2. Reduction: one value from many. Dependence: a combine of partial results; associative. Unit product: a partial value.
3. Scan: each output the combine of all earlier inputs. Dependence: each unit needs the total of all earlier units. Unit product: a run of outputs plus its total.
4. Stencil: each output from a fixed neighbourhood. Dependence: overlapping neighbourhoods of input. Unit product: a tile of outputs.
5. Gather: each output from a position computed from data. Dependence: on input positions known only from data. Unit product: elements.
6. Scatter / histogram: many inputs combined into data-chosen destinations. Dependence: many-to-few, data-chosen. Unit product: partial bins.
7. Tiled product (GEMM): output tile from a row band of A and a column band of B. Dependence: a K-loop over tile pairs. Unit product: an output tile.
8. Multi-stage pipeline: stages consuming earlier stages' products (A-to-F). Dependence: a DAG of stages. Unit product: per stage.
9. Run-time-declared component: a stage whose inputs and outputs are named at run time (CoSpan). Dependence: as wired at run time. Unit product: per output.

### U7 Native-term list (C1 yardstick)
native group / work-group (obscures: the problem's own partition); work item / thread identity (obscures: which element is computed); barrier, wait (obscures: which values a computation needs); cross-group synchronisation, ticket, flag, claim (obscures: the dependency itself); atomics (obscures: that a combine is a reduction); group_iterator over single-valued group arrays (obscures: that the upstream units each produced one value, i.e. that the edge is a reduction fan-in); memory-placement kind (local / global) as a required choice (obscures: dependency vs where it lives; tension with C7, recorded as such); group shape doubling as a work-group size (obscures: the problem tile).

### U8 The reduction in the baseline
```cpp
auto leaf = kh.parallel_for(range<1>{64}, range<1>{4}, [=](item<1> it) { return in[it.id()]; });
auto m1 = kh.map_groups<mapping::local>(leaf, [](id<1> g) { return single<1>{g}; });
auto l1 = kh.parallel_for(range<1>{16}, m1, [](group<1>, group_iterator<float, 1, mapping::local> a) {
  return group_reduce<1>(a[0], 0, plus<>{}); });                       // 16 values
auto m2 = kh.map_groups<mapping::global>(l1, [](id<1> g) { return block<1>{id<1>{g * 4}, range<1>{4}}; });
auto l2 = kh.parallel_for(range<1>{4}, range<1>{1}, m2, [](item<1>, group_iterator<float, 0, mapping::global> a) {
  return a[0][0] + a[1][0] + a[2][0] + a[3][0]; });                    // item body: no loop allowed, no group_reduce over an iterator
// layer 3 repeats layer 2 with block of 4 over l2
```
Features that read poorly: (a) the leaf is an identity parallel_for only because a group body needs a mapped source (term: native group / group shape); (b) the mapping kind must be chosen per edge (placement kind); (c) layer 2 receives a group_iterator over four single-valued group arrays (iterator over single-valued arrays) and cannot reduce it with the group vocabulary, so it falls back to an item body with a hand-unrolled sum (work item); (d) the group size 4 is written three times as range, single/block arithmetic and unroll count (group shape as work-group size); (e) nothing in the text says "reduce 64 to 1 in fan-in 4".

## Instrument

### U9 Stress programs (problem, sizes, inputs, outputs, reference)
P1 Global reduction. n = 64, fan-in 4 per level. Input x[i] = i (float). Output one float. Reference 2016.
P2 n-level reduction, depth d a launch value. n = 4^d. Input x[i] = 1. Output one float. Reference n (d = 3: 64; d = 5: 1024).
P3 Whole-array inclusive scan, tile 64. n = 1024. Input x[i] = 1 (int). Output y[i]. Reference y[i] = i + 1.
P4 1-D 3-point stencil, zero boundary. n = 1024. Input x[i] = i. Output y[i] = x[i-1] + x[i] + x[i+1]. Reference y[0] = 1, y[i] = 3i for 0 < i < 1023, y[1023] = 2045.
P5 Gather through a data index array. n = 256. Inputs src[j] = 2j, idx[i] = (7i) mod 256. Output y[i] = src[idx[i]]. Reference y[i] = 2((7i) mod 256).
P6 Tiled GEMM, tile 16x16. 128x128. Inputs A[i][k] = i, B[k][j] = j. Output C. Reference C[i][j] = 128 i j.
P7 Histogram, 16 bins. n = 4096. Input v[i] = i mod 16. Output 16 counts. Reference 256 each.
P8 A-to-F (sg-core plan): A item leaf; B, C local from A; D local from B and C; E global from C; F item from E; sinks B, C, D, F. Inputs and reference as expected_a_to_f in naga/sg-tests/bench/bench_stats.hpp (b = i+1, c = 2i, d = 3i+1, f = 2(4(3-g)+l)+1001, n = 16, groups of 4).
P9 Repeated builder pushes: a host loop of 4 iterations, each layer y = 2*prev + 1 elementwise, n = 64, x[i] = i. Reference 16i + 15.
P10 Run-time-declared component: a component function receives named inputs {"x"} and outputs {"y"} and parameters a, b known only at run time; y = a x + b over n given at run time. Reference a x[i] + b.
P11 Inplace contributions: two kernels contribute 1 and 2 per element into y (n = 64) with plus, initializer 0. Reference 3 each.

### U10 Criteria (verdicts: pass, pass-with-cost (cost stated), fail, cannot-express, unresolved (reason stated); evidence line per cell)
C1 [hard] Problem-level reading: no U7 term in the program; P1 and P2 show one named construct per layer and one per inter-layer mapping, or the candidate states an explicit departure from W5.
C2 [ranking] Unit-of-work shape visible: for every node the source states what one unit of work produces where consumers connect.
C3 [hard] C++ expressibility: a builder expression in the fork's C++ standard, no macros or code generation; a wrong shape or mapping is diagnosed at compile time for static shapes, at host build time with a diagnostic naming the node for launch-valued shapes.
C4 [hard] Static purity: every body checkable as pure by the existing checker or one named extension.
C5 [hard] Launch-time topology: the program states what is structure and what is bound; changing only bound values keeps the identity.
C6 [hard] One native kernel: flattening gives one launch, generated-code size linear in node count with no per-level cloning, no forward-progress need beyond the per-backend findings.
C7 [ranking] Performance model: the source alone determines which values cross global memory, which stay local, which are recomputed.
C8 [ranking] Distance from baseline: pass if only unchanged baseline constructs; pass-with-cost if constructs are added and none changed or removed; fail otherwise.

### U11 Axis
Questions: Q1 what is the primitive description element; Q2 how are dependencies between units of work stated; Q3 where does the shape of a unit's product live. Slots by Q2: A explicit shaped edges per node; B implied by named structural combinators; C derived from index relations between tiled arrays. Rule: no two candidates give the same Q2 answer.

## Research

Sources: three research runs of 2026-11-23 (prior art in ten systems; dependence description; lowering into one launch) and a toolchain survey of HEAD 671ea872.

### U12 Multi-level reductions
- Halide: Func definitions with RDom updates; levels by explicit partial Funcs p(b) += in(4b+r) or rfactor; native blocks/threads appear only in the schedule (gpu_blocks, gpu_threads). Source: Halide tutorials 12, 18.
- Futhark: `map (reduce (+) 0) (unflatten xs : [16][4])` repeated; logical levels by reshaping, no native concepts; launches compiler-chosen. Source: Futhark prelude SOACs.
- JAX/XLA: `x.reshape(16,4).sum(axis=1)` repeated; XLA chooses kernels; Pallas exposes grid and barriers. Source: JAX lax, Pallas reference.
- Triton: program instance over a block; `tl.sum` per program; a tree is three launches of the same kernel; no grid barrier. Source: Triton language API.
- TVM: te.compute with reduce_axis, split + rfactor; thread binding in schedule. Source: TVM reduction tutorial.
- Taichi: scalar-field `+=` recognised as a reduction (thread-local + atomic); explicit levels by loops. Source: Taichi performance docs.
- Kokkos: parallel_reduce; explicit tree = three dispatches; team reductions are team-local. Source: Kokkos parallel_reduce.
- Legion/Regent: tasks over regions, reduction privileges or a task tree; runtime dependence analysis. Source: Legion tutorials, Regent privileges.
- StreamIt: filter `work pop 4 push 1` composed three times in a pipeline; rates on the filter, FIFO edges; no native concepts. Source: StreamIt spec 2.1.
- CUDA cooperative groups: tile collectives, partials per block, then more launches or grid.sync under cooperative launch. Source: CUDA programming guide.

### U13 Patterns (scan, stencil, tiling, gather, histogram)
| system | scan | stencil | tiling/GEMM | gather | histogram |
|---|---|---|---|---|---|
| Halide | recurrence, no parallel whole-array scan | neighbourhood expr | schedule tile + rfactor | nested index | `hist(in(r)) += 1` with rfactor |
| Futhark | `scan`, `exscan` | index neighbours | nested map + reduce, compiler tiles | `map (\i -> t[i])` | `hist`, `reduce_by_index` |
| JAX/XLA | `associative_scan` | `reduce_window` | `dot_general` / Pallas tiles | `lax.gather` | `.at[].add` |
| Triton | per-program `associative_scan`; multi-program by hand | masked halo loads | per-program tile + K loop | pointer loads | `tl.histogram` / atomics |
| TVM | `te.scan`, topi cumsum | te neighbourhood | split/reorder/bind | topi gather | not found |
| Taichi | PrefixSumExecutor (multi-kernel) | inner neighbour loops | by hand, no canonical example | field indexing | atomic `+=` |
| Kokkos | `parallel_scan` | MDRange | teams + scratch / KokkosKernels | View indexing | atomics, ScatterView |
| Legion | by tasks, no primitive | halo partitions | tile subregions | region pointers | reduction privilege |
| StreamIt | stateful filter | `peek` window | filters on tiles | local arrays only | local bins + merge |
| CUDA CG | block scan + totals + offsets | shared halo + sync | shared tiles + sync | plain loads | block bins + atomics |

### U14 Dependence description
- Legion: tasks + region privileges; dependencies derived at run time; projection functors map launch points to subregions.
- StreamIt: filters with pop/peek/push rates, pipelines, split-joins; dependencies structural; static schedule from rates.
- Dask/Ray: tasks keyed by result; `blockwise` index labels relate input blocks to output blocks; resolved at construction, readiness at run time.
- CUDA Graphs: explicit operation nodes and edges; no output shape; one graph launch, K kernels.
- SYCL graphs: command-group nodes, explicit or accessor-implied edges; finalize then replay; K kernels.
- Chapel: forall over domains; distributions give placement; no dependence inference.
- Polyhedral (Polly, PPCG, Tiramisu, Pluto): iteration domains + affine access relations + schedule; dependences derived; tiles are schedule units.
- Skeletons (SkePU, Muesli, Thrust/CUB, Lift/RISE): the pattern name carries the internal dependence; split/join change nesting in the type (Lift).

### U15 Design dimensions
D1 primitive element (unit-of-work node / pattern application / array element definition) — Q1. D2 how dependence is stated (explicit edge / implied by pattern / derived from access relation) — Q2. D3 where unit-product shape lives (on the node or edge / in the pattern's type / on the array domain, units chosen by toolchain) — Q3. D4 who chooses fusion vs materialisation (user per edge / compiler by rules / separate schedule). D5 when dependencies resolve (build / launch / run).

### U16 C++ enforceability
- Static shapes (fan-in, tile sizes as template arguments) are checkable at compile time by deduction; launch-valued shapes are checked when the builder runs, naming the node (the prototype already does this; sg error messages name the construct). Evidence: sg-core prototype, include/hipSYCL/sg/kernel.hpp.
- A builder can type its handles by product type and rank; edge forms can compute the delivered argument type from (form, upstream product type). Evidence: Lift's dependent array types; prototype group_iterator typing.
- Index expressions over placeholders can be extracted by instantiating a generic lambda with a host interval type (Halide-style embedding, prototype probe mode). Restriction: arithmetic only; data-dependent indexing gives "unknown".
- Purity today: BodyChecker.hpp walks from `sg_body_entry` closures, rejects loops, goto, recursion, mutable/ref captures, any write through a pointer or reference, non-const statics, no-body / virtual / function-pointer calls; namespace sg and system headers exempt; positions entries not annotated (coverage gap). An extension needs only new entry annotations.
- The sg headers compile as C++20 and use almost none of its features: concepts and class-type NTTPs are available.

### U17 Lowering into one launch
- Last-arriver continuation (CUDA threadFenceReduction generalised): each producer unit publishes, increments a counter on its consumer unit; the unit completing the last input continues with the consumer; nobody waits. Needs the consumer of a producer unit (the inverse mapping).
- Ticket on activation (Merrill-Garland decoupled look-back): unit ids taken from an atomic counter at start, so any waited-on unit was claimed by a running group; safe under occupancy-bound execution; prefix combine by look-back.
- Forward progress: OpenCL/SYCL give no inter-group guarantee; CUDA ordinary launches none; HSA oldest-workgroup rule; Level Zero none; CPU implementations serialise work-items (CBS/PoCL). Rule: never wait on unclaimed work.
- Code growth comes from inlining nested stages, specialisation and unrolling; the scalable shape is stage functions dispatched by a uniform switch with the graph as data. NVPTX and Intel IGC both support direct non-inlined calls (IGC subroutines; stack calls only for some functions).
Sources: Merrill & Garland 2016; NVIDIA cuda-samples threadFenceReduction; Sorensen et al. CONCUR 2018; Intel IGC CodeGenContext; Hazy Research megakernel 2025; Mirage persistent kernel 2025.

### U18 Toolchain
- SSCP: body sources compiled once, composed at JIT; the prototype clones per layer and per node (CompositionStep.hpp) and the generic pipeline marks every non-kernel function alwaysinline (GlobalInliningAttributorPass.cpp:30-39).
- CBS on the host inlines every barrier-containing callee into the kernel (LoopSplitterInlining.cpp:24-143, KernelFlattening.cpp:22-60); a barrier in a non-inlined call is not handled.
- Kernel identity: kernel_configuration::generate_id hashes base config, options, flags, specialisations, fcall configs, composition configs (unique_hash = fnv1a64 of structure_string), param flags, alignments; persistent cache keyed by that id.
- Dynamic functions (include/hipSYCL/sycl/jit.hpp): define, define_as_call_sequence, define_as_indexed_call_sequence (definition k called with a parameter fixed to k); targets kept noinline.
- Body identity: registry.hpp takes the entry's address literally (FunctionNameExtractionPass).

### U19 Memory cost
Boundary kinds: recomputed per consumer (registers only, cost = repeated compute); held local (one group computes its inputs, local memory, no parallelism across inputs); passed through global (inputs computed in parallel elsewhere, cost = global write + read of the unit products). Rules: (1) a fan-in edge whose inputs are produced in parallel must pass products through global memory, so its cost is one unit product per upstream unit; (2) overlapping or data-chosen edges either recompute or store once; (3) the choice must be readable from the source (C7).

## Candidates

Common to all: the builder frame, roles, evaluate, keyed builders and inplace stay as in U1. `in`, `src`, `idx`, `v`, `A`, `B` are input arrays; `out` a result.

### U20 Candidate A: explicit shaped edges
Axis: Q1 a node (a grid of units, each a pure function of its inputs); Q2 explicit edges `sg::from(upstream, form[, mode])`; Q3 the node's unit product, deduced from the body's return, delivered to consumers as a dense array of (fan-in x product shape).
Concepts: `kh.source(array)` (an input as a node of element units); `kh.node(units, edges..., body)`; edge forms `one`, `tiles<K...>`, `rows<K>`, `cols<K>`, `window<K, H>(boundary)`, `all`, `ids(index array)`, `units_fn(f)` (custom positions, fixed count), `before(op, select)` (combine of the products of all earlier upstream units); edge modes `parallel` (default for partition forms: upstream units run in their own parallel work, products passed through global memory), `fused` (this unit computes its upstream units itself), `shared` (default for overlapping or data forms: computed once, stored); body vocabulary on arrays: element-wise ops, `map`, `map_index`, `reduce`, `scan`, `contract`, `bin_count`; sugar `kh.map(src, f)` (one, fused); `kh.repeat(n, start, fn)`.
Departures: core 5 (compute cheap / memory only where the user says): parallel edges store unit products by default; the user says it by choosing the mode, so it is stated here as an explicit departure of the default. Wants: none.
Diff vs U1: parallel_for replaced by node; map_groups replaced by from(); item/local/global mappings replaced by forms + modes; item and group bodies merged into one unit body; group expressions changed (no rank template, plus map, map_index, contract, bin_count); builder frame, roles, evaluate, keyed builders kept.
Prior art cited: StreamIt rates (U12, U14), Lift dependent array types (U16), last-arriver and look-back (U17).
Programs:
- P1 `auto x = kh.source(in); auto l1 = kh.node(16, sg::from(x, sg::tiles<4>), sum); auto l2 = kh.node(4, sg::from(l1, sg::tiles<4>), sum); auto l3 = kh.node(1, sg::from(l2, sg::tiles<4>), sum); out = kh.evaluate(l3);` with `sum = [](auto t) { return sg::reduce(t, sycl::plus<>{}); }`.
- P2 `out = kh.evaluate(kh.repeat(d, kh.source(in), [](auto& kh, auto prev) { return kh.node(prev.units() / 4, sg::from(prev, sg::tiles<4>), sum); }));`
- P3 `auto t = kh.node(16, sg::from(x, sg::tiles<64>), [](auto v) { return sg::scan(v, plus); }); auto y = kh.node(16, sg::from(t, sg::one), sg::from(t, sg::before(plus, sg::last)), [](auto s, int off) { return s + off; });`
- P4 `auto y = kh.node(16, sg::from(x, sg::window<64, 1>(sg::zero)), [](auto w) { return sg::map_index(64, [=](int i) { return w[i] + w[i + 1] + w[i + 2]; }); });`
- P5 `auto y = kh.node(4, sg::from(kh.source(idx), sg::tiles<64>), [=](auto ix) { return sg::map(ix, [=](int j) { return src[j]; }); });`
- P6 `auto c = kh.node({8, 8}, sg::from(kh.source(A), sg::rows<16>), sg::from(kh.source(B), sg::cols<16>), [](auto a, auto b) { return sg::contract(a, b); });`
- P7 `auto part = kh.node(16, sg::from(kh.source(v), sg::tiles<256>), [](auto t) { return sg::bin_count<16>(t); }); auto hist = kh.node(1, sg::from(part, sg::tiles<16>), [](auto p) { return sg::reduce(p, 0, plus); });`
- P8 `a = kh.node(4, from(x, tiles<4>), fa); b = kh.node(4, from(a, one, fused), fb); c = kh.node(4, from(a, one, fused), fc); d = kh.node(4, from(b, one, fused), from(c, one, fused), fd); e = kh.node(4, from(c, one), fe); f = kh.node(4, from(e, units_fn([](int g) { return 3 - g; })), ff);` evaluate b, c, d, f.
- P9 `auto y = kh.source(in); for (int k = 0; k < 4; ++k) y = kh.map(y, [](float v) { return 2 * v + 1; }); out = kh.evaluate(y);`
- P10 `auto x = kh.source(ins.get<float, 1>("x")); outs.set("y", kh.evaluate(kh.map(x, [=](float v) { return a * v + b; })));`
- P11 two submissions, each `out = kh.evaluate(kh.map(kh.source(ones), [](float v) { return k * v; }));` with out = inplace(y, plus, 0), k = 1 and 2.

### U21 Candidate B: named structural patterns
Axis: Q1 a pattern application on arrays; Q2 implied by the pattern (split, map, zip, reduce, scan, slide, gather, histogram, reverse, reduce_tree); Q3 in the pattern's result type (split<K> makes [n/K][K]).
Concepts: `kh.split<K>(a)`, `kh.map(a, f)` (f over one element or one row), `kh.zip(a, b)`, `kh.reduce`, `kh.reduce_tree<K>`, `kh.scan`, `kh.slide<W>(a, boundary)`, `kh.gather(src, idx)`, `kh.histogram<N>(a, key)`, `kh.reverse`, `kh.outer_map(rows, cols, f)`, `kh.split_rows<K>`, `kh.split_cols<K>`. Fusion fixed by rules: split, map, zip, slide fuse into their consumer; reduce, scan, gather, histogram materialise at their boundary.
Departures: core: none. Wants: W2-W4 (no explicit node graph; dependence is implied by patterns) — explicit departure: edges are not objects, consequence: arbitrary positions functions are not expressible.
Diff vs U1: parallel_for, map_groups, mappings, item and group bodies replaced by patterns; builder frame, roles, evaluate, keyed builders kept.
Prior art: Futhark SOACs, Lift split/join, SkePU, StreamIt combinators (U12-U14).
Programs: P1 `l1 = kh.map(kh.split<4>(x), sg::sum); l2 = kh.map(kh.split<4>(l1), sg::sum); l3 = kh.map(kh.split<4>(l2), sg::sum);`. P2 `kh.reduce_tree<4>(x, plus)`. P3 `kh.scan(x, plus)`. P4 `kh.map(kh.slide<3>(x, sg::zero), [](auto w) { return w[0] + w[1] + w[2]; })`. P5 `kh.gather(src, idx)`. P6 `kh.outer_map(kh.split_rows<16>(A), kh.split_cols<16>(B), sg::contract)`. P7 `kh.histogram<16>(v, [](int x) { return x; })`. P8 `a = kh.map(kh.split<4>(x), fa); b = kh.map(a, fb); c = kh.map(a, fc); d = kh.map(kh.zip(b, c), fd); e = kh.map(c, fe); f = kh.map(kh.reverse(kh.split<4>(e)), ff);`. P9 host loop of `kh.map`. P10 as A with `kh.map`. P11 as A.

### U22 Candidate C: index relations between arrays
Axis: Q1 an array definition: a domain and a pure function of an index; Q2 derived from the access relation of the function (slices, element reads); Q3 on the array domain; units (tiles) chosen by the toolchain.
Concepts: `kh.define<N>(domain, [=](auto i...) { ... })` with generic index parameters; reads `a[i]`, `a.slice(off, n)`, `a.at_or(i, v)`; `sg::reduce(slice, op)`, `sg::count`; the toolchain instantiates each body with a host interval type to derive which tiles of each source a tile needs; data-dependent reads give "whole source".
Departures: core: none. Wants: W2 and W4 (the graph is implicit and the unit is the toolchain's) — explicit departure; consequence: unit shape and fusion are not in the source.
Diff vs U1: parallel_for, map_groups, mappings, bodies replaced by define; builder frame, roles, evaluate, keyed builders kept.
Prior art: Halide Funcs, Dask blockwise, polyhedral access relations (U12, U14).
Programs: P1 `p = kh.define<1>({16}, [=](auto b) { return sg::reduce(x.slice(4 * b, 4), plus); }); q = kh.define<1>({4}, [=](auto b) { return sg::reduce(p.slice(4 * b, 4), plus); }); s = kh.define<0>({}, [=] { return sg::reduce(q.slice(0, 4), plus); });`. P2 host loop of define with slices of the previous level. P3 `y = kh.define<1>({1024}, [=](auto i) { return sg::reduce(x.slice(0, i + 1), plus); })`. P4 `y(i) = x.at_or(i - 1, 0) + x[i] + x.at_or(i + 1, 0)`. P5 `y(i) = src[idx[i]]`. P6 `c(i, j) = sg::reduce(sg::iota(128), [=](auto k) { return A(i, k) * B(k, j); }, plus)`. P7 `h(b) = sg::count(v.slice(0, 4096), b)`. P8 definitions b(i), c(i), d(i) reading a's elements, f(i) reading `e[(3 - i / 4) * 4 + i % 4]`. P9 host loop of define. P10, P11 as A with define.

### U24 Control rows (baseline language)
P1 as U8. P2 host loop repeating U8's layer 2 (hand-unrolled sum per fan-in). P3 cannot-express (the offset of each tile needs the totals of all earlier tiles; mapping counts must be uniform and the first-version vocabulary has no masked select). P4 item body reading `in` at i-1, i, i+1 with boundary branches. P5 item body `return src[idx[it.id()]];`. P6 group body over local mappings of A and B tiles with a group-expression contraction written as element-wise products and group_reduce. P7 cannot-express (no comparison, mask or bin operation in the first-version vocabulary, no loops). P8 the sg-core test graph with local/global/item kinds. P9 host loop of item-body parallel_for. P10 as baseline builder. P11 inplace results.

### U23 Difference record
| | Q1 element | Q2 dependence | Q3 unit-product shape |
|---|---|---|---|
| A | node of units | explicit edges | on the node, delivered by the edge |
| B | pattern application | implied by pattern | in the pattern's result type |
| C | array definition | derived from access relations | on the array domain; units toolchain's |
Q2 answers pairwise distinct: yes.

## Evaluation

### U25 Flattening of A
One launch. Each node compiles to one unit function (its body over the delivered inputs); the kernel is a dispatch loop: the native group takes a ticket from an atomic counter (tickets issued for units of nodes whose inputs are sources, fused, shared or data edges, in topological node order); it runs that unit (a uniform switch over node id calling the unit function once; fused edges call the upstream unit functions inside); it stores the product where a parallel/shared edge or a result needs it; for each parallel partition edge it increments the counter of the consumer unit given by the inverse form (tiles: u / K) and, if it completed the last input, continues with that consumer unit in the same group (last arriver); shared and data edges wait on a node-done counter of the upstream node, which only ever refers to units ticketed earlier, so a wait is always on running or finished work. P1: 16 tickets, continuations 16 -> 4 -> 1, no waits. P2: repeat compiles to one node function with the level as a bound value; units per level bound. P6: 64 tickets, A and B tiles are sources; contract runs the K loop inside the unit. Code size: one switch case per node, each body once (fused upstream bodies once per fused edge): linear in node count. CPU: everything inlines into one switch; cases are taken on a broadcast value (uniform). Forward progress: no wait in P1, P2, P4, P5, P6, P7, P9-P11; P3 (before edge) and P8 (shared edge from c to e) wait only on ticketed-earlier units.

### U26 Flattening of B
Each pattern lowers to unit functions as in A, with the unit fixed by the pattern (split<K>: K elements; scan, slide, gather, histogram: a toolchain tile). reduce_tree is one node function with the level bound. scan lowers to tile scan + look-back. Same dispatch loop, same continuation and ticket rules. Code size linear in pattern count. Forward progress as A.

### U27 Flattening of C
The toolchain picks a tile per definition, instantiates each body on the host with an interval index to get, per consumer tile, the range of source tiles it reads, and builds the inverse table (bound data, per launch). Same dispatch loop; fan-in counters from the derived ranges; data-dependent reads become node-done waits. P1: tiles may cover a whole level (16 outputs in one tile), collapsing the tree. P3: each output reads a growing prefix: the derived table is quadratic unless the prefix is recognised. P6: tile chosen by the toolchain; K loop inside. Code size linear. Forward progress as A.

### U28-U31 Matrices
Notation: per criterion a default verdict with its evidence line applies to every program not listed; listed exceptions carry their own evidence. pass = 2, pass-with-cost = 1, others 0.

Control (U28)
- C1: fail for P1, P2 (iterator over single-valued arrays, kinds, hand-unrolled fan-in: U8), P6, P8 (mapping kinds and group shapes as native terms); cannot-express P3, P7 (U24); pass P4, P5, P9, P10, P11 (item bodies over problem coordinates). Score 10.
- C2: pass P4, P5, P9, P10, P11 (range and group shape on the definition); pass-with-cost P6, P8 (group-body product shape implicit in the returned expression); fail P1, P2 (unit product hidden behind group_iterator of rank-0 arrays); cannot-express P3, P7. Score 12.
- C3: pass for every expressible program (prototype compiles them, sg-core tests); cannot-express P3, P7. Score 18.
- C4: pass for every expressible program (BodyChecker); cannot-express P3, P7. Score 18.
- C5: pass for every expressible program (structure_string excludes counts); cannot-express P3, P7. Score 18.
- C6: fail P2 (prototype chain_30: quadratic, 321 s cuda, ze hang); pass-with-cost P1, P8 (shortcut nesting, small depth); pass P4, P5, P6, P9, P10, P11; cannot-express P3, P7. Score 14.
- C7: pass for every expressible program (kinds explicit); cannot-express P3, P7. Score 18.
- C8: pass for every expressible program (baseline itself); cannot-express P3, P7. Score 18.
Total 126. Hard failures: C1, C6 -> disqualified.

A (U29)
- C1: pass for all (no U7 term; P1 shows three nodes and three tiles<4> edges; P2 one repeat of one node + edge). Score 22.
- C2: pass for all (the edge delivers a dense array of fan-in x product; the body's return states the product). Score 22.
- C3: pass default (typed builder; delivered type from form and product type; static fan-in deduced, U16); pass-with-cost P2 (repeat level unit counts checked when the builder runs), P10 (type-erased named inputs need a checked cast). Score 20.
- C4: pass for all (bodies are closures checked from new entry annotations, U16). Score 22.
- C5: pass default (structure = nodes, forms, modes; unit counts, sizes bound); pass for P2 through repeat (level is bound). Score 22.
- C6: pass default (U25: linear, continuation, no waits); pass-with-cost P3 (look-back waits on ticketed-earlier units, costly under mass spinning on Intel, U17), P8 (shared edge c -> e waits on ticketed-earlier units). Score 20.
- C7: pass for all (mode per edge readable: parallel stores products, fused recomputes, shared stores once; U19). Score 22.
- C8: fail for all (parallel_for, map_groups, bodies replaced). Score 0.
Total 150. No hard failure.

B (U30)
- C1: pass for all (P1 three map(split) layers; split is the mapping). Score 22.
- C2: pass P1, P2, P6, P8, P9, P10, P11 (split<K> and map state the unit); pass-with-cost P3, P4, P5, P7 (unit tile is the toolchain's inside scan, slide, gather, histogram). Score 18.
- C3: pass default (typed pattern builders); pass-with-cost P8 (reversed group mapping needs the reverse pattern; arbitrary positions functions not expressible). Score 21.
- C4: pass for all. Score 22.
- C5: pass for all (pattern parameters bound; reduce_tree level bound). Score 22.
- C6: pass default (U26); pass-with-cost P3, P8 (as A). Score 20.
- C7: pass-with-cost for all (fusion follows fixed pattern rules, readable only by knowing the rules per pattern). Score 11.
- C8: fail for all. Score 0.
Total 136. No hard failure.

C (U31)
- C1: pass default (definitions read like formulas); pass-with-cost P1, P2 (the inter-level mapping is a slice inside the body, not a named construct). Score 20.
- C2: fail for all (units are the toolchain's; U22). Score 0.
- C3: pass-with-cost for all (bodies must be generic over an interval index type; arithmetic index expressions only; U16). Score 11.
- C4: pass for all (device instantiation checked). Score 22.
- C5: pass-with-cost for all (relations re-derived by host probing every launch; identity unchanged). Score 11.
- C6: pass default (U27); pass-with-cost P3 (quadratic derived table), P7 (each bin reads all of v). Score 20.
- C7: fail for all (tiling and materialisation are the toolchain's). Score 0.
- C8: fail for all. Score 0.
Total 84. No hard failure.

### U32 Discrimination
A vs B differ on C2 (P3, P4, P5, P7), C3 (P2, P8, P10), C7 (all). A vs C differ on C1 (P1, P2), C2, C3, C5, C7. B vs C differ on C2, C3, C5, C7. Criteria with a differing cell: C1, C2, C3, C5, C7 (five >= three). Rule holds; no programs appended.

### U33 Selection
Disqualification: control (C1, C6 fail). A, B, C: no hard fail or unresolved; cannot-express 0 each. Ranking: A 150, B 136, C 84. Winner A. Reasons: over B, A states the unit of work where B hides it in scan/slide/gather/histogram (C2 P3, P4, P5, P7), expresses arbitrary positions (C3 P8), and makes placement readable per edge where B relies on per-pattern rules (C7, all programs); over C, A states units and placement (C2, C7 all fail in C) and needs no symbolic probing (C3, C5).

### U34 Borrowings
From B: patterns as sugar over A nodes (`kh.map`, `kh.reduce_tree<K>` = repeat of a tiles<K> node), so common cases stay one line (source cell B C1 P2). From B: reduce_tree's bound level folded into `kh.repeat` (source cell B C5 P2); A C5 P2 re-evaluated: pass (already recorded with repeat). From C: `at_or(i, v)` boundary reads on inputs inside bodies (source C C1 P4). Re-evaluated A cells: none change.
