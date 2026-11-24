// SPDX-License-Identifier: BSD-2-Clause
// The descriptor of a view: strides, offsets, extents, index-table reference and
// out-of-range policy, shared by host and device.
#ifndef HIPSYCL_SG_DETAIL_VIEW_DESC_HPP
#define HIPSYCL_SG_DETAIL_VIEW_DESC_HPP

#include <cstdint>

namespace sg::detail {

enum class vk : std::uint32_t {
  source = 0,    // elements of an input array
  result = 1,    // tiles stored by a map, tile after tile
  tile = 2,      // elements cut into tiles
  retile = 3,    // tiles regrouped in tile order
  join = 4,      // tiles stitched into elements
  extend = 5,    // tiles broadcast to a shape
  table = 6,     // index tiles from a host table
  index_map = 7, // index tiles replaced by their elements
  array = 8      // elements stored in memory by an earlier map (result array)
};

enum class oob_kind : std::uint32_t { fill = 0, wrap = 1 };

constexpr std::int32_t no_view = -1;

// Element views keep their extent in `shape`; tile views keep their grid in
// `grid` and their tile in `shape`.
struct view_node {
  std::uint32_t kind = 0;
  std::int32_t parent = no_view;  // tile, retile, join, extend: the input; index_map: the elements
  std::int32_t parent2 = no_view; // index_map: the index tiles
  std::uint32_t elem_bytes = 0;
  std::uint32_t grid_rank = 0, shape_rank = 0;
  std::uint32_t oob = 0;
  std::uint32_t base_local = 0;  // result and array: 1 when the base is an offset into local memory
  std::uint32_t wait_flags = 0;  // result and array: reads through data-chosen indices wait on flags
  std::int64_t grid[3] = {1, 1, 1};
  std::int64_t shape[3] = {1, 1, 1};
  std::int64_t trim[3] = {0, 0, 0}; // join: trimmed extent; 0 keeps grid x shape
  std::uint64_t fill[2] = {0, 0};
  std::uint64_t base = 0;       // source, result, array: address or local offset; table: unused
  std::uint64_t table = 0;      // table: byte offset of the int64 entries in the block (or host index)
  std::uint64_t tile_elems = 0; // result: elements stored per tile
  std::uint32_t flag_base = 0, segs = 1; // result: flag slot of tile 0, flags per tile
  std::uint32_t table_rank = 0, pad = 0;
};
static_assert(sizeof(view_node) % 8 == 0);

inline std::int64_t prod3(const std::int64_t v[3]) { return v[0] * v[1] * v[2]; }

// Row-major with dimension 0 slowest; dims beyond the rank are 1.
inline void unflatten3(std::int64_t k, const std::int64_t dims[3], std::int64_t out[3]) {
  for (int d = 2; d >= 0; --d) {
    const std::int64_t e = dims[d] > 0 ? dims[d] : 1;
    out[d] = k % e;
    k /= e;
  }
}
inline std::int64_t flatten3(const std::int64_t id[3], const std::int64_t dims[3]) {
  return (id[0] * dims[1] + id[1]) * dims[2] + id[2];
}

} // namespace sg::detail

#endif
