// SPDX-License-Identifier: BSD-2-Clause
// Work sets: work tiles (each one native work-group, or one group per segment of
// a segmented run), slices of a work set, and sets of work items.
#ifndef HIPSYCL_SG_WORK_SET_HPP
#define HIPSYCL_SG_WORK_SET_HPP

#include "hipSYCL/sg/detail/scope.hpp"
#include "hipSYCL/sg/error.hpp"
#include "hipSYCL/sg/launch_value.hpp"

#include <source_location>
#include <string>

namespace sg {

// A run of `segments` x `items`, each segment its own native group.
struct segment_items {
  std::int64_t segments, items;
  segment_items(detail::sint s, detail::sint i) : segments(s.v), items(i.v) {}
};

class work_set {
public:
  work_set() = default;
  work_set(detail::kgraph *g, int id) : g_(g), id_(id) {}
  int id() const noexcept { return id_; }
  detail::kgraph *graph() const noexcept { return g_; }

private:
  detail::kgraph *g_ = nullptr;
  int id_ = -1;
};

// A work set whose runs are segmented: its units run_distributed or run_replicated.
class segmented_work_set : public work_set {
public:
  using work_set::work_set;
};

class item_set {
public:
  item_set() = default;
  item_set(detail::kgraph *g, int id) : g_(g), id_(id) {}
  int id() const noexcept { return id_; }
  detail::kgraph *graph() const noexcept { return g_; }

private:
  detail::kgraph *g_ = nullptr;
  int id_ = -1;
};

namespace detail {

inline std::int64_t count_tile_sets(const kgraph &g) {
  std::int64_t n = 0;
  for (const work_rec &w : g.works)
    if (!w.items && !w.slice)
      ++n;
  return n;
}
inline std::int64_t count_item_sets(const kgraph &g) {
  std::int64_t n = 0;
  for (const work_rec &w : g.works)
    if (w.items)
      ++n;
  return n;
}

// The counts include the set being added.
[[noreturn]] inline void mixed_kinds(const kgraph &g, const char *construct, source_loc where, bool adding_tiles) {
  throw build_error(7, construct, where,
                    "a kernel uses work tiles (kh.work_tiles) or work items (kh.work_items), never both",
                    {{"work_tile_sets", count_tile_sets(g) + (adding_tiles ? 1 : 0)},
                     {"work_item_sets", count_item_sets(g) + (adding_tiles ? 0 : 1)}});
}

// The native shape of a tile and its run: tile ++ run, at most 3-D, and within the cap.
inline void check_native_shape(const kgraph &g, const char *construct, const dims &tile, std::int64_t items,
                               bool has_run, source_loc where) {
  if (tile.rank < 1)
    throw error{errc::builder_misuse, std::string(construct) + ": a tile shape needs at least one extent"};
  if (tile.rank == 3 && has_run && items > 1)
    throw build_error(2, construct, where, "a 3-D tile leaves no dimension for a run",
                      {{"tile_rank", tile.rank}, {"run", items}});
  if (tile.rank > 3)
    throw build_error(2, construct, where, "a native shape above 3-D", {{"tile_rank", tile.rank}, {"run", items}});
  const std::int64_t product = tile.product() * items;
  if (product > g.group_size_cap)
    throw build_error(1, construct, where, "the work tile does not fit a native group",
                      {{"product", product}, {"cap", g.group_size_cap}});
}

inline int add_tile_work(kgraph &g, const bdims &grid, const dims &tile, std::int64_t run, bool segmented,
                         std::int64_t segs, std::int64_t ips, source_loc where) {
  if (g.has_items)
    mixed_kinds(g, "kh.work_tiles", where, true);
  if (run < 1 || (segmented && (segs < 1 || ips < 1)))
    throw error{errc::builder_misuse, "kh.work_tiles: a run needs at least one item"};
  check_native_shape(g, "kh.work_tiles", tile, segmented ? ips : run, segmented || run > 1, where);
  if (grid.rank != tile.rank)
    throw error{errc::builder_misuse, "kh.work_tiles: the grid has rank " + std::to_string(grid.rank) +
                                          " and the tile has rank " + std::to_string(tile.rank)};
  work_rec w;
  w.grid = grid;
  w.tile = tile;
  w.run = run;
  w.segmented = segmented;
  w.segs = segs;
  w.ips = ips;
  w.where = where;
  g.has_tiles = true;
  return g.add_work(std::move(w));
}

inline int add_slice_work(kgraph &g, int root, const bdims &offset, const bdims &extent, const bdims &stride,
                          const dims &tile, std::int64_t run, source_loc where) {
  if (root < 0 || root >= static_cast<int>(g.works.size()) || g.works[root].items)
    throw error{errc::builder_misuse, "kh.slice_work: the first argument is not a set of work tiles"};
  const work_rec &r = g.works[root];
  if (r.segmented)
    throw error{errc::builder_misuse, "kh.slice_work: a segmented work set cannot be sliced"};
  if (offset.rank != r.grid.rank || extent.rank != r.grid.rank || stride.rank != r.grid.rank)
    throw error{errc::builder_misuse, "kh.slice_work: offset, extent and stride must have the rank of the work set's grid"};
  if (tile.rank < 1)
    throw error{errc::builder_misuse, "kh.slice_work: a tile shape needs at least one extent"};
  check_native_shape(g, "kh.slice_work", tile, run, run > 1, where);
  work_rec w;
  w.slice = true;
  w.root = r.slice ? r.root : root;
  w.grid = extent;
  w.offset = offset;
  w.extent = extent;
  w.stride = stride;
  w.tile = tile;
  w.run = run;
  w.where = where;
  return g.add_work(std::move(w));
}

inline int add_item_work(kgraph &g, const bdims &range, source_loc where) {
  if (g.has_tiles)
    mixed_kinds(g, "kh.work_items", where, false);
  if (range.rank < 1)
    throw error{errc::builder_misuse, "kh.work_items: a range needs at least one extent"};
  work_rec w;
  w.items = true;
  w.grid = range;
  w.where = where;
  g.has_items = true;
  return g.add_work(std::move(w));
}

} // namespace detail

} // namespace sg

#endif
