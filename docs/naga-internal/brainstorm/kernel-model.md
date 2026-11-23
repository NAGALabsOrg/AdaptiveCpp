# Kernel model: work tiles

Status: working statement from the stage 2 discussion with Jack, 2026-11-23. It supersedes the work side of kernel-language-proposal.md (the D1 width rule, and work as a map over data tiles). The data-side decisions D2-D5 there stand unless the kernel pass below changes them. Not yet tested against the stress kernels.

## Agreed (Jack)

1. **Work tiles are defined by what is shared.** A work tile is a grid of work items that share data.
2. **Work items are independent.** Each item in a tile produces its results without depending on any other item. This independence is what makes work sliceable.
3. **A work item's parallelism is one-dimensional.** It is a single run over its tile's shared data. Multi-dimensional structure belongs to the tile's grid of items: results that are independent of each other are separate items. An item's result is one value (a reduce) or a run (element-wise, scan, binning).
4. **Data is tiled so that each work tile can select what it shares.** This applies to inputs and to computed arrays. Data tiling follows from the problem, never from what fits a device.
5. **Fitting a device.** A work tile maps to one native work-group of (items x run length) work-items. If that exceeds the device's cap, the user either:
   - retiles the work: fewer items per tile; no join; less sharing; or
   - slices the items' runs: the sharing is kept; the pieces of a reduce are joined by the user with the primitives provided.
   Sliced work needs no new logic for its data: the toolchain derives it from the unsliced work, and the data tiling is unchanged.
6. **The user decides memory backing through this structure**, not the toolchain behind their back.

## Proposed, to test in the kernel pass (Research Lead)

- **Backing by structure:** data a work tile selects is shared by its items (local memory); an item's own run is private; values passing between top-level work maps are global.
- **Addressing:** accesses inside a work tile are written in the work's coordinates (tile id, item id, position in the run), so slicing can be derived. Tile selection is a function of the work tile id, evaluated on the host for exact dependencies.
- **Item vocabulary:** element-wise operations, reduce, scan and tile_bin over the run, with no dimension arguments.
- **Cap check:** items x run length per work tile; exceeding it is an error from the builder naming the work map and its total.
- **Kept from the settled language:** the kernel-level graph between top-level work maps (one launch, last-arriver continuation, waits only on claimed work); tile, retile and join for data; index maps and stencils for building data tiles; evaluate; roles; repeat; all sizes as runtime values (D2).
- **To check:** whether define_tiles and work_tiles are the same thing; whether nesting ever needs to go deeper than tile, item, run.

## Kernel pass

The stress kernels are rewritten under this model one at a time, reduction and scan first, then the rest. Nothing here is recorded as decided until it holds up there.
