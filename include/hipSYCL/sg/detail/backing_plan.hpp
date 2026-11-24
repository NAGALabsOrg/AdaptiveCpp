// SPDX-License-Identifier: BSD-2-Clause
// Marks and the backing plan. A value is backed by where it is read: private
// scratch for a unit's own use, recomputation for other units of the same work
// tile, local memory for collective results read by other units of the work
// tile, marked results and marked array tiles, global memory across work tiles
// and sets, result memory when evaluated.
#ifndef HIPSYCL_SG_DETAIL_BACKING_PLAN_HPP
#define HIPSYCL_SG_DETAIL_BACKING_PLAN_HPP

#include "hipSYCL/sg/detail/kernel_preparer.hpp"
#include "hipSYCL/sg/detail/scope.hpp"
#include "hipSYCL/sg/detail/tile_graph.hpp"
#include "hipSYCL/sg/view.hpp"
#include "hipSYCL/sg/work_set.hpp"

#include <algorithm>
#include <cstdint>
#include <source_location>
#include <vector>

namespace sg {

// The mode of kh.mark_local for results of run_distributed.
struct segmented {};

namespace detail {

enum class backing : std::uint32_t { private_scratch, recomputed, local, global, result };

inline const char *to_string(backing b) noexcept {
  switch (b) {
  case backing::private_scratch: return "private";
  case backing::recomputed: return "recomputed";
  case backing::local: return "local";
  case backing::global: return "global";
  case backing::result: return "result";
  }
  return "?";
}

// The per-tile decision for the tiles of one map's results.
struct value_backing {
  int map = -1;
  std::vector<backing> tile;  // by tile of the map
  bool store_global = false;  // some tile crosses work tiles
  bool store_local = false;   // read only inside its group, in local memory
  bool evaluated = false;
  bool marked = false;
  bool marked_segmented = false;
};

inline const view_rec &root_result_view(const kgraph &g, int view) {
  int v = view;
  while (g.views[v].parent != no_view && g.views[v].kind != vk::result)
    v = g.views[v].parent;
  return g.views[v];
}

// true when the view is built over a result of a map somewhere down its parents
inline bool derives_from_result(const kgraph &g, int view) {
  for (int v = view; v != no_view; v = g.views[v].parent) {
    if (g.views[v].kind == vk::result || g.views[v].kind == vk::array)
      return true;
    if (g.views[v].kind == vk::index_map && g.views[v].parent2 != no_view &&
        derives_from_result(g, g.views[v].parent2))
      return true;
  }
  return false;
}

inline void mark_result(kgraph &g, int view, bool segmented_mode, source_loc where) {
  const view_rec &v = g.views[view];
  if (v.kind != vk::result)
    throw error{errc::builder_misuse, "kh.mark_local: the argument is not the result of kh.map_work"};
  const map_rec &m = g.maps[v.map];
  const work_rec &w = g.works[m.work];
  const bool distributed = m.run == run_kind::distributed;
  if (!segmented_mode && distributed)
    throw build_error(4, "kh.mark_local", where,
                      "a result of run_distributed is held per segment: write kh.mark_local<sg::segmented>",
                      {{"segments", w.segs}});
  if (segmented_mode && !distributed)
    throw build_error(5, "kh.mark_local<sg::segmented>", where,
                      "the segmented form applies to results of run_distributed only",
                      {{"segments", w.segs}});
  mark_rec r;
  r.kind = segmented_mode ? mark_kind::segmented : mark_kind::local;
  r.view = view;
  r.where = where;
  g.marks.push_back(r);
}

// How the results of a map reach the sinks: the layout of the evaluated array.
struct eval_layout {
  bool evaluated = false;
  std::int64_t grid[3] = {1, 1, 1}, shape[3] = {1, 1, 1}, extent[3] = {1, 1, 1};
  int slot = -1;
};

inline std::vector<eval_layout> compute_eval_layouts(const prepared_kernel &p) {
  const kgraph &g = p.graph;
  std::vector<eval_layout> out(g.maps.size());
  for (const sink_rec &s : g.sinks) {
    if (s.items)
      continue;
    const view_rec &j = g.views[s.view];
    if (j.kind != vk::join)
      throw error{errc::builder_misuse, "kh.evaluate: results are evaluated from kh.join of result tiles"};
    int tile_view = j.parent;
    const int under_join = tile_view;
    while (g.views[tile_view].kind == vk::retile)
      tile_view = g.views[tile_view].parent;
    if (g.views[tile_view].kind != vk::result)
      throw error{errc::builder_misuse, "kh.evaluate: only retile and join may stand between results and kh.evaluate"};
    eval_layout &l = out[g.views[tile_view].map];
    l.evaluated = true;
    l.slot = s.slot;
    for (int d = 0; d < 3; ++d) {
      l.grid[d] = p.views[under_join].grid[d];
      l.shape[d] = p.views[under_join].shape[d];
      l.extent[d] = p.views[s.view].shape[d];
    }
  }
  return out;
}

// The decision for every tile of every map's results.
inline std::vector<value_backing> compute_backing(const prepared_kernel &p, const tile_graph &tg) {
  const kgraph &g = p.graph;
  std::vector<value_backing> out(g.maps.size());
  const std::vector<eval_layout> layouts = compute_eval_layouts(p);
  for (std::size_t m = 0; m < g.maps.size(); ++m) {
    value_backing &v = out[m];
    v.map = static_cast<int>(m);
    v.tile.assign(static_cast<std::size_t>(p.maps[m].tiles), backing::private_scratch);
    v.evaluated = layouts[m].evaluated;
    for (const mark_rec &mk : g.marks)
      if (mk.kind != mark_kind::array_tiles && g.views[mk.view].map == static_cast<int>(m)) {
        v.marked = true;
        v.marked_segmented = mk.kind == mark_kind::segmented;
      }
    for (const sink_rec &s : g.sinks)
      if (s.sink_task >= 0 && !s.items && g.views[s.view].kind == vk::join)
        v.evaluated = true;
  }
  std::vector<std::vector<char>> seen_consumer(g.maps.size());
  std::vector<std::vector<char>> any_consumer(g.maps.size());
  for (std::size_t m = 0; m < g.maps.size(); ++m)
    any_consumer[m].assign(static_cast<std::size_t>(p.maps[m].tiles), 0);
  for (const pull &pl : tg.pulls) {
    value_backing &v = out[pl.producer_map];
    const map_conc &pm = p.maps[pl.producer_map];
    const task_inst &ct = tg.tasks[pl.consumer];
    const work_conc &pw = p.works[g.maps[pl.producer_map].work];
    backing &b = v.tile[static_cast<std::size_t>(pl.producer_tile)];
    any_consumer[pl.producer_map][static_cast<std::size_t>(pl.producer_tile)] = 1;
    // the producer's group(s) of that tile
    std::int64_t root_tile = pl.producer_tile;
    if (pw.slice) {
      std::int64_t c[3], rc[3];
      unflatten3(pl.producer_tile, pw.grid, c);
      for (int d = 0; d < 3; ++d)
        rc[d] = pw.offset[d] + c[d] * pw.stride[d];
      root_tile = flatten3(rc, p.works[pw.root].grid);
    }
    const bool same_group =
        !pl.by_data && tg.groups[ct.group].work == pw.root && tg.groups[ct.group].tile == root_tile &&
        (!pw.segmented || v.marked_segmented);
    if (!same_group) {
      b = backing::global;
      continue;
    }
    if (b == backing::global)
      continue;
    const bool own_unit = pl.producer_unit == pl.consumer_unit;
    backing want = own_unit ? backing::private_scratch
                            : (pm.rk != result_kind::per_item || v.marked ? backing::local : backing::recomputed);
    if (static_cast<int>(want) > static_cast<int>(b))
      b = want;
  }
  for (std::size_t m = 0; m < g.maps.size(); ++m) {
    value_backing &v = out[m];
    for (std::size_t t = 0; t < v.tile.size(); ++t) {
      if (!any_consumer[m][t] && v.evaluated)
        v.tile[t] = backing::result;
      if (v.tile[t] == backing::global)
        v.store_global = true;
    }
    if (!v.store_global)
      for (std::size_t t = 0; t < v.tile.size(); ++t)
        if (any_consumer[m][t])
          v.store_local = true;
  }
  return out;
}

inline void mark_array_tiles(kgraph &g, int view, int work, source_loc where) {
  if (derives_from_result(g, view))
    throw build_error(9, "kh.mark_local", where,
                      "array tiles are tiles of input data; this view is made of results", {});
  if (work < 0 || work >= static_cast<int>(g.works.size()) || g.works[work].items)
    throw error{errc::builder_misuse, "kh.mark_local: the work argument is not a set of work tiles"};
  mark_rec r;
  r.kind = mark_kind::array_tiles;
  r.view = view;
  r.work = work;
  r.where = where;
  g.marks.push_back(r);
}

} // namespace detail

} // namespace sg

#endif
