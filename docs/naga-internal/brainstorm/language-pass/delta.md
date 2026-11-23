# Delta: summary v1 (89b931f3) to v2

Your correction: fitting the cap happens only at construction.

- Removed: kh.split_runs, and retiling work after the fact.
- Added: `sg::segment_items(segments, items)` as a work set's run: the code reads as the full run; each segment is its own native group.
- The contract you take on by segmenting: collectives over the run (reduce, scan, tile_bin) are joined across segments with a global dependency; local memory is `kh.mark_segmented_local`, held per segment; plain `kh.mark_local` is rejected on segmented work tiles.
- Matrix product now keeps your 16 x 16 outputs: `kh.work_tiles({8, 8}, {16, 16}, sg::segment_items(32, 4))`, 2048 groups of 1024. The A and B stages use the same segmentation, so item p of the product reads item p of a stage in its own segment; only the final reduce crosses segments. Traced 1920 at C[3][5].
- Row sums at cap 256: `kh.work_tiles({64}, {1}, sg::segment_items(4, 256))`; the body is the same as at cap 1024; no join code. Traced 523776 and 588288.
- The 2 x 4 matrix product stays in the record as the other lever (smaller work tiles, no cross-segment cost).
- Decision 3 of v1 (two meanings of "slice") is withdrawn: only kh.slice_work remains. Four decisions remain, unchanged.

## Round 2 (D11): summary v3
- A segmented work set is its own type. Its units have `unit.run_distributed(...)` (each segment runs its share; collectives join across segments with a global dependency) and `unit.run_replicated(...)` (every segment runs the whole run; collectives stay in the group); no plain `unit.run`.
- Its results are marked local only with `kh.mark_segmented_local(r)`, held as the run produced them: the whole result in every segment if replicated, each segment's share if distributed.
- The builder rejects `run_replicated` whose input is a distributed result of its own set.
- Matrix product and row sums now say `run_distributed`; nothing else in them changed.

## Round 3 (D12): summary v4
- One marking API: `kh.mark_local(r)`, with a mode. The default is for unsegmented work and for results of `run_replicated` (the whole result in every segment).
- Results of `run_distributed` require `kh.mark_local<sg::segmented>(r)`: you state that the local memory is per segment. The plain form there is an error when the builder runs, telling you to write the segmented form; the segmented form on anything else is also an error.
- `kh.mark_segmented_local` from round 2 is gone. The matrix product's stages now read `kh.mark_local<sg::segmented>(a_stage)`.

## Round 4 (D13, D14): summary v5
- Shorter runs on a work set: allowed; the result tile's run dimension follows the run used.
- `kh.tile(x, shape, oob)`: a partial last tile is padded by the oob policy; join trims back to the array's extent.
- Work-item kernels: `kh.work_items(range)`, `kh.map_items(items, body(item))`, `item.compute(inputs..., fn)`. A plain range launch; each item reads inputs and only its own earlier results; a kernel uses work tiles or work items, never both. The element-wise chain, the run-time component and inplace contributions are now work-item kernels.
- `kh.mark_local(array_tiles, work)`: array tiles held locally by every work tile (every segment) that pulls them. The matrix product drops its staging maps.
- No decisions remain open.
