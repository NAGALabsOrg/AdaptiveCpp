# Delta: summary v1 (89b931f3) to v2

Your correction: fitting the cap happens only at construction.

- Removed: kh.split_runs, and retiling work after the fact.
- Added: `sg::segment_items(segments, items)` as a work set's run: the code reads as the full run; each segment is its own native group.
- The contract you take on by segmenting: collectives over the run (reduce, scan, tile_bin) are joined across segments with a global dependency; local memory is `kh.mark_local_segmented`, held per segment; plain `kh.mark_local` is rejected on segmented work tiles.
- Matrix product now keeps your 16 x 16 outputs: `kh.work_tiles({8, 8}, {16, 16}, sg::segment_items(32, 4))`, 2048 groups of 1024. The A and B stages use the same segmentation, so item p of the product reads item p of a stage in its own segment; only the final reduce crosses segments. Traced 1920 at C[3][5].
- Row sums at cap 256: `kh.work_tiles({64}, {1}, sg::segment_items(4, 256))`; the body is the same as at cap 1024; no join code. Traced 523776 and 588288.
- The 2 x 4 matrix product stays in the record as the other lever (smaller work tiles, no cross-segment cost).
- Decision 3 of v1 (two meanings of "slice") is withdrawn: only kh.slice_work remains. Four decisions remain, unchanged.
