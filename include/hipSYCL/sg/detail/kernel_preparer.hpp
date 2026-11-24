// SPDX-License-Identifier: BSD-2-Clause
// Preparation of a recorded kernel graph for one set of launch values: binds
// sizes, instances repeated sections, makes the view descriptors and index
// tables, and runs every unit body once, recording what each unit picked.
#ifndef HIPSYCL_SG_DETAIL_KERNEL_PREPARER_HPP
#define HIPSYCL_SG_DETAIL_KERNEL_PREPARER_HPP

#include "hipSYCL/sg/detail/dev/view_read.hpp"
#include "hipSYCL/sg/detail/repeat.hpp"
#include "hipSYCL/sg/detail/scope.hpp"
#include "hipSYCL/sg/view.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace sg::detail {

// A work set with its numbers bound.
struct work_conc {
  bool items = false, segmented = false, slice = false;
  int root = -1; // the work set whose groups run this set's tasks
  std::int64_t grid[3] = {1, 1, 1};
  int grid_rank = 0;
  std::int64_t tile[3] = {1, 1, 1};
  int tile_rank = 0;
  std::int64_t tiles = 0;  // tiles of this set
  std::int64_t units = 1;  // units of one tile
  std::int64_t run = 1;    // run length of one unit
  std::int64_t segs = 1, ips = 1;
  std::int64_t offset[3] = {0, 0, 0}, stride[3] = {1, 1, 1}; // slice
  std::int64_t native = 1; // work-items of one native group this set needs
  std::int64_t range[3] = {1, 1, 1};
  std::int64_t first_group = 0; // root sets: first native group of the set
};

// A map with its units evaluated.
struct map_conc {
  run_kind kind = run_kind::plain;
  result_kind rk = result_kind::per_item;
  std::int64_t tiles = 0, units = 1;
  std::size_t n_inputs = 0;
  std::int64_t run_used = 0;     // input length of every unit
  std::int64_t bins = 0;
  std::int64_t tile_elems = 0;   // elements of one result tile
  std::int64_t result_shape[3] = {1, 1, 1};
  int result_rank = 0;
  std::vector<input_sel> sels;   // [(tile * units + unit) * n_inputs + input]
  closure_entry closure;
  sizing_info sizing;
  std::uint32_t flag_base = 0;
  std::uint32_t partial_stride = 0;
  std::int64_t stride = 1;       // lanes per unit
  std::int64_t rounds = 1;
  const input_sel &sel(std::int64_t tile, std::int64_t unit, std::size_t j) const {
    return sels[static_cast<std::size_t>((tile * units + unit) * static_cast<std::int64_t>(n_inputs) + static_cast<std::int64_t>(j))];
  }
  input_sel &sel(std::int64_t tile, std::int64_t unit, std::size_t j) {
    return sels[static_cast<std::size_t>((tile * units + unit) * static_cast<std::int64_t>(n_inputs) + static_cast<std::int64_t>(j))];
  }
};

// The result of the preparation of one kernel.
struct prepared_kernel {
  kgraph graph; // repeated sections expanded
  std::vector<std::int64_t> launch_values;
  std::vector<view_node> views;
  std::vector<std::vector<std::int64_t>> tables; // by view
  std::vector<work_conc> works;
  std::vector<map_conc> maps;
  std::vector<std::int64_t> imap_tiles; // unused placeholder for symmetry
  std::int64_t native_size = 1;
  std::int64_t n_flags = 1;      // flag 0 is the ticket counter
  bool items = false;
  std::int64_t n_items = 0;
  std::int64_t items_ext[3] = {1, 1, 1};
  int items_rank = 1;
  std::int64_t unit_bodies_evaluated = 0;
  std::int64_t local_copy_count = 0;
  // marked array tiles: per work set and tile, the distinct (view, tile) pairs its units use
  std::map<std::pair<int, std::int64_t>, std::vector<std::pair<int, std::int64_t>>> copies; // (root work, group tile)
};

namespace prep_detail {

inline std::int64_t eval(const bound_expr &e, const eval_env &env) { return e.evaluate(env); }

inline void pad_dims(const bdims &d, const eval_env &env, std::int64_t out[3]) {
  for (int i = 0; i < 3; ++i)
    out[i] = i < d.rank ? eval(d.v[i], env) : 1;
}

inline std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return b <= 0 ? 0 : (a + b - 1) / b; }

} // namespace prep_detail

class kernel_preparer {
public:
  // `launch_values` are the values of the kernel's launch slots, in order.
  // With `previous`, a map whose numbers do not depend on a launch value that changed keeps its units' selections.
  static prepared_kernel prepare(const kgraph &recorded, const std::vector<std::int64_t> &launch_values,
                                 const prepared_kernel *previous = nullptr) {
    prepared_kernel p;
    p.launch_values = launch_values;
    eval_env env;
    env.launch = p.launch_values.data();
    env.launch_count = p.launch_values.size();
    p.graph = expand_repeats(recorded, env);
    const kgraph &g = p.graph;
    p.views.assign(g.views.size(), view_node{});
    p.tables.assign(g.views.size(), {});
    p.works.assign(g.works.size(), work_conc{});
    p.maps.assign(g.maps.size(), map_conc{});

    env.count_of = [&p](std::int64_t view, int dim) -> std::int64_t {
      const view_node &n = p.views.at(static_cast<std::size_t>(view));
      const bool tiles_view = n.grid_rank > 0;
      if (tiles_view)
        return dim < 0 ? prod3(n.grid) : n.grid[dim];
      return dim < 0 ? prod3(n.shape) : n.shape[dim];
    };

    std::int64_t flags = 1;
    for (const ev_rec &ev : g.events) {
      switch (ev.kind) {
      case ev_kind::view: finalize_view(p, ev.idx, env); break;
      case ev_kind::work: finalize_work(p, ev.idx, env); break;
      case ev_kind::map: {
        evaluate_map(p, ev.idx, env, recorded.repeats.empty() ? previous : nullptr);
        map_conc &m = p.maps[ev.idx];
        m.flag_base = static_cast<std::uint32_t>(flags);
        const work_conc &w = p.works[g.maps[ev.idx].work];
        flags += m.tiles * (w.segmented ? w.segs : 1);
        view_node &rv = p.views[g.maps[ev.idx].result_view];
        rv.flag_base = m.flag_base;
        rv.wait_flags = 1;
        break;
      }
      case ev_kind::imap: break;
      }
    }
    p.n_flags = flags;
    assign_native_size(p);
    return p;
  }

private:
  static void set_ranks(view_node &n, int grid_rank, int shape_rank) {
    n.grid_rank = static_cast<std::uint32_t>(grid_rank);
    n.shape_rank = static_cast<std::uint32_t>(shape_rank);
  }
  static void pad_shape(std::int64_t out[3], const dims &d) {
    for (int i = 0; i < 3; ++i)
      out[i] = i < d.rank ? d.v[i] : 1;
  }

  static void finalize_view(prepared_kernel &p, int id, const eval_env &env) {
    const view_rec &r = p.graph.views[id];
    if (r.kind == vk::result)
      return; // the numbers of a result view come with its map
    view_node &n = p.views[id];
    n = view_node{};
    n.kind = static_cast<std::uint32_t>(r.kind);
    n.parent = r.parent;
    n.parent2 = r.parent2;
    n.elem_bytes = r.elem_bytes;
    n.oob = static_cast<std::uint32_t>(r.oob);
    n.fill[0] = r.fill[0];
    n.fill[1] = r.fill[1];
    switch (r.kind) {
    case vk::source:
    case vk::array:
      prep_detail::pad_dims(r.ext, env, n.shape);
      set_ranks(n, 0, r.ext.rank);
      break;
    case vk::tile: {
      const view_node &par = p.views[r.parent];
      pad_shape(n.shape, r.shape);
      for (int d = 0; d < 3; ++d)
        n.grid[d] = d < r.shape.rank ? prep_detail::ceil_div(par.shape[d], n.shape[d]) : 1;
      set_ranks(n, r.shape.rank, r.shape.rank);
      break;
    }
    case vk::retile: {
      const view_node &par = p.views[r.parent];
      if (r.alias_prev) {
        for (int d = 0; d < 3; ++d) {
          n.grid[d] = par.grid[d];
          n.shape[d] = par.shape[d];
        }
        set_ranks(n, static_cast<int>(par.grid_rank), static_cast<int>(par.shape_rank));
        break;
      }
      pad_shape(n.shape, r.shape);
      const std::int64_t total = prod3(par.grid) * prod3(par.shape);
      const std::int64_t size = prod3(n.shape);
      if (size <= 0 || total % size != 0)
        throw error{errc::shape_mismatch, "kh.retile: " + std::to_string(total) + " elements do not divide into tiles of " +
                                              std::to_string(size)};
      n.grid[0] = total / size;
      set_ranks(n, 1, r.shape.rank);
      break;
    }
    case vk::join: {
      const view_node &par = p.views[r.parent];
      const view_rec &pr = p.graph.views[r.parent];
      for (int d = 0; d < 3; ++d)
        n.shape[d] = par.grid[d] * par.shape[d];
      if (pr.kind == vk::tile) {
        const view_node &src = p.views[pr.parent];
        for (int d = 0; d < 3; ++d)
          n.shape[d] = std::min(n.shape[d], src.shape[d]);
      } else if (par.shape_rank == 1 && par.grid_rank == 1) {
        // results of a map over one tiled input, one result per input element, are trimmed like the input
        int v = r.parent;
        while (p.graph.views[v].kind == vk::retile)
          v = p.graph.views[v].parent;
        if (p.graph.views[v].kind == vk::result) {
          const map_conc &mc = p.maps[static_cast<std::size_t>(p.graph.views[v].map)];
          if (!mc.sels.empty() && mc.rk == result_kind::per_item) {
            const int tv = mc.sels[0].view;
            if (p.graph.views[tv].kind == vk::tile) {
              const view_node &tn = p.views[tv];
              const view_node &src = p.views[p.graph.views[tv].parent];
              if (src.shape_rank == 1 && prod3(tn.grid) * prod3(tn.shape) == n.shape[0])
                n.shape[0] = std::min(n.shape[0], src.shape[0]);
            }
          }
        }
      }
      set_ranks(n, 0, static_cast<int>(par.shape_rank));
      break;
    }
    case vk::extend: {
      const view_node &par = p.views[r.parent];
      pad_shape(n.shape, r.shape);
      for (int d = 0; d < 3; ++d)
        n.grid[d] = par.grid[d];
      set_ranks(n, static_cast<int>(par.grid_rank), r.shape.rank);
      break;
    }
    case vk::table: {
      pad_shape(n.shape, r.shape);
      std::vector<std::int64_t> &t = p.tables[id];
      if (r.define) {
        const std::int64_t count = prep_detail::eval(r.define_n, env);
        n.grid[0] = count;
        set_ranks(n, 1, r.shape.rank);
        n.table_rank = 1;
        const std::int64_t size = prod3(n.shape);
        t.reserve(static_cast<std::size_t>(count * size));
        for (std::int64_t i = 0; i < count; ++i)
          for (std::int64_t s = 0; s < size; ++s)
            t.push_back(r.define_fn(i, s));
      } else {
        std::int64_t begin[3] = {0, 0, 0}, end[3] = {1, 1, 1};
        for (int d = 0; d < r.roots_begin.rank; ++d) {
          begin[d] = prep_detail::eval(r.roots_begin.v[d], env);
          end[d] = prep_detail::eval(r.roots_end.v[d], env);
          n.grid[d] = std::max<std::int64_t>(0, end[d] - begin[d]);
        }
        set_ranks(n, r.roots_begin.rank, 1);
        n.table_rank = static_cast<std::uint32_t>(r.roots_begin.rank);
        const std::int64_t roots = prod3(n.grid);
        t.reserve(static_cast<std::size_t>(roots * static_cast<std::int64_t>(r.offsets.size()) * r.roots_begin.rank));
        for (std::int64_t k = 0; k < roots; ++k) {
          std::int64_t c[3];
          unflatten3(k, n.grid, c);
          for (const auto &o : r.offsets)
            for (int d = 0; d < r.roots_begin.rank; ++d)
              t.push_back(begin[d] + c[d] + o[d]);
        }
      }
      n.table = reinterpret_cast<std::uint64_t>(t.data());
      break;
    }
    case vk::index_map: {
      const view_node &idxn = p.views[r.parent2];
      for (int d = 0; d < 3; ++d) {
        n.grid[d] = idxn.grid[d];
        n.shape[d] = idxn.shape[d];
      }
      set_ranks(n, static_cast<int>(idxn.grid_rank), static_cast<int>(idxn.shape_rank));
      break;
    }
    case vk::result:
      // numbers come with the map
      break;
    }
  }

  static void finalize_work(prepared_kernel &p, int id, const eval_env &env) {
    const work_rec &r = p.graph.works[id];
    work_conc &w = p.works[id];
    w = work_conc{};
    w.items = r.items;
    w.segmented = r.segmented;
    w.slice = r.slice;
    if (r.items) {
      prep_detail::pad_dims(r.grid, env, w.range);
      w.grid_rank = r.grid.rank;
      w.tiles = 1;
      p.items = true;
      p.items_rank = r.grid.rank;
      p.n_items = prod3(w.range);
      for (int d = 0; d < 3; ++d)
        p.items_ext[d] = w.range[d];
      return;
    }
    for (int d = 0; d < 3; ++d)
      w.tile[d] = d < r.tile.rank ? r.tile.v[d] : 1;
    w.tile_rank = r.tile.rank;
    w.units = prod3(w.tile);
    w.run = r.run;
    w.grid_rank = r.grid.rank;
    if (r.slice) {
      const work_conc &root = p.works[r.root];
      w.root = r.root;
      std::int64_t extent[3];
      prep_detail::pad_dims(r.offset, env, w.offset);
      prep_detail::pad_dims(r.extent, env, extent);
      prep_detail::pad_dims(r.stride, env, w.stride);
      for (int d = r.offset.rank; d < 3; ++d)
        w.offset[d] = 0;
      for (int d = 0; d < 3; ++d) {
        const std::int64_t last = w.offset[d] + (extent[d] > 0 ? (extent[d] - 1) * w.stride[d] : 0);
        if (w.offset[d] < 0 || extent[d] < 0 || w.stride[d] < 1 || last >= root.grid[d])
          throw build_error(13, "kh.slice_work", r.where, "the slice reaches beyond its work set",
                            {{"offset", w.offset[d]}, {"extent", extent[d]}, {"stride", w.stride[d]},
                             {"tiles", root.grid[d]}});
        w.grid[d] = extent[d];
      }
      w.tiles = prod3(w.grid);
      w.native = w.units * w.run;
      return;
    }
    w.root = id;
    prep_detail::pad_dims(r.grid, env, w.grid);
    w.tiles = prod3(w.grid);
    w.segmented = r.segmented;
    w.segs = r.segmented ? r.segs : 1;
    w.ips = r.segmented ? r.ips : r.run;
    w.native = w.units * (r.segmented ? r.ips : r.run);
  }

  static void collect_launch(const bound_expr &e, std::set<std::int64_t> &out, const kgraph &g,
                             std::set<int> &views, std::set<int> &maps, const prepared_kernel &prev) {
    e.for_each_launch([&](std::int64_t s) { out.insert(s); });
    e.for_each_view([&](std::int64_t v) { collect_view(static_cast<int>(v), out, g, views, maps, prev); });
  }

  static void collect_bdims(const bdims &d, std::set<std::int64_t> &out, const kgraph &g, std::set<int> &views,
                            std::set<int> &maps, const prepared_kernel &prev) {
    for (int i = 0; i < d.rank; ++i)
      collect_launch(d.v[i], out, g, views, maps, prev);
  }

  static void collect_view(int v, std::set<std::int64_t> &out, const kgraph &g, std::set<int> &views,
                           std::set<int> &maps, const prepared_kernel &prev) {
    if (v == no_view || !views.insert(v).second)
      return;
    const view_rec &r = g.views[v];
    collect_bdims(r.ext, out, g, views, maps, prev);
    collect_bdims(r.roots_begin, out, g, views, maps, prev);
    collect_bdims(r.roots_end, out, g, views, maps, prev);
    collect_launch(r.define_n, out, g, views, maps, prev);
    collect_view(r.parent, out, g, views, maps, prev);
    collect_view(r.parent2, out, g, views, maps, prev);
    if (r.kind == vk::result)
      collect_map(r.map, out, g, views, maps, prev);
  }

  static void collect_map(int m, std::set<std::int64_t> &out, const kgraph &g, std::set<int> &views,
                          std::set<int> &maps, const prepared_kernel &prev) {
    if (m < 0 || !maps.insert(m).second)
      return;
    const work_rec &w = g.works[g.maps[m].work];
    collect_bdims(w.grid, out, g, views, maps, prev);
    collect_bdims(w.offset, out, g, views, maps, prev);
    collect_bdims(w.extent, out, g, views, maps, prev);
    collect_bdims(w.stride, out, g, views, maps, prev);
    if (w.slice && w.root >= 0)
      collect_bdims(g.works[w.root].grid, out, g, views, maps, prev);
    for (const input_sel &s : prev.maps[m].sels)
      collect_view(s.view, out, g, views, maps, prev);
  }

  static bool can_reuse(const prepared_kernel &p, int id, const prepared_kernel *prev) {
    if (!prev || prev->graph.maps.size() != p.graph.maps.size() || prev->graph.views.size() != p.graph.views.size() ||
        prev->maps.size() != p.maps.size() || prev->maps[id].n_inputs != p.graph.maps[id].n_inputs)
      return false;
    std::set<std::int64_t> slots;
    std::set<int> views, maps;
    collect_map(id, slots, p.graph, views, maps, *prev);
    for (std::int64_t s : slots) {
      const std::size_t i = static_cast<std::size_t>(s);
      if (i >= p.launch_values.size() || i >= prev->launch_values.size() || p.launch_values[i] != prev->launch_values[i])
        return false;
    }
    return true;
  }

  static void evaluate_map(prepared_kernel &p, int id, const eval_env &, const prepared_kernel *prev) {
    const map_rec &mr = p.graph.maps[id];
    const work_conc &w = p.works[mr.work];
    map_conc &m = p.maps[id];
    const bool reuse = can_reuse(p, id, prev);
    if (reuse)
      m = prev->maps[id];
    else
      m = map_conc{};
    m.kind = mr.run;
    m.rk = mr.rk;
    m.tiles = w.tiles;
    m.units = w.units;
    m.n_inputs = mr.n_inputs;
    if (!reuse)
      m.sels.resize(static_cast<std::size_t>(m.tiles * m.units) * m.n_inputs);

    unit_env_holder holder(p, id);
    prep_access access{p.views.data(), p.views.size()};
    prep_access *saved = current_prep();
    const std::function<int(int)> *saved_remap = current_view_remap();
    current_prep() = &access;
    current_view_remap() = mr.remap ? &mr.remap : nullptr;
    struct restore {
      prep_access *a;
      const std::function<int(int)> *r;
      ~restore() {
        current_prep() = a;
        current_view_remap() = r;
      }
    } guard{saved, saved_remap};

    bool first = true;
    for (std::int64_t t = 0; t < m.tiles && m.units > 0 && !reuse; ++t) {
      std::int64_t tc[3];
      unflatten3(t, w.grid, tc);
      for (std::int64_t u = 0; u < m.units; ++u) {
        std::int64_t uc[3];
        unflatten3(u, w.tile, uc);
        unit_ctx ctx;
        ctx.tile = idx{};
        ctx.tile.set_rank(w.grid_rank);
        for (int d = 0; d < w.grid_rank; ++d)
          ctx.tile[d] = tc[d];
        ctx.id.set_rank(w.tile_rank);
        for (int d = 0; d < w.tile_rank; ++d)
          ctx.id[d] = uc[d];
        ctx.run_len = w.segmented ? w.segs * w.ips : w.run;
        ctx.env = &holder.env;
        unit_eval_out o = mr.eval(ctx);
        ++p.unit_bodies_evaluated;
        if (o.sels.size() != m.n_inputs)
          throw error{errc::internal, "a unit body returned another number of inputs than its map"};
        if (first) {
          m.closure = o.closure;
          m.run_used = o.sels.empty() ? 0 : o.sels[0].len;
          first = false;
        } else if (o.closure.bytes != m.closure.bytes) {
          throw error{errc::shape_mismatch, "the units of one map must run the same function with the same captures"};
        }
        for (std::size_t j = 0; j < m.n_inputs; ++j) {
          if (o.sels[j].len != m.run_used)
            throw error{errc::shape_mismatch, "the units of one map must run runs of the same length: " +
                                                  std::to_string(o.sels[j].len) + " and " + std::to_string(m.run_used)};
          m.sel(t, u, j) = o.sels[j];
        }
      }
    }
    if (m.run_used == 0 && m.units * m.tiles > 0)
      m.run_used = 1;
    if (!first)
      m.sizing = mr.sizing(m.closure);
    m.bins = m.sizing.bins;
    m.stride = w.segmented ? w.ips : std::max<std::int64_t>(1, m.run_used);
    m.rounds = mr.run == run_kind::replicated ? std::max<std::int64_t>(1, prep_detail::ceil_div(m.run_used, w.ips)) : 1;
    if (m.rk == result_kind::binned && m.bins > m.stride)
      throw error{errc::builder_misuse, "sg::tile_bin: " + std::to_string(m.bins) + " bins need at least as many items in the run"};
    m.partial_stride = (mr.run == run_kind::plain) ? 0 : std::max<std::uint32_t>(16, m.sizing.partial_bytes);

    // the shape of one result tile: the unit grid, then the run
    for (int d = 0; d < 3; ++d)
      m.result_shape[d] = d < w.tile_rank ? w.tile[d] : 1;
    m.result_rank = w.tile_rank;
    if (w.tile_rank < 3 && m.rk != result_kind::reduced) {
      m.result_shape[w.tile_rank] = m.rk == result_kind::binned ? m.bins : m.run_used;
      m.result_rank = w.tile_rank + 1;
    }
    m.tile_elems = prod3(m.result_shape);

    // the result view
    view_node &rv = p.views[mr.result_view];
    rv = view_node{};
    rv.kind = static_cast<std::uint32_t>(vk::result);
    rv.elem_bytes = mr.result_bytes;
    for (int d = 0; d < 3; ++d) {
      rv.grid[d] = w.grid[d];
      rv.shape[d] = m.result_shape[d];
    }
    rv.grid_rank = static_cast<std::uint32_t>(w.grid_rank);
    rv.shape_rank = static_cast<std::uint32_t>(m.result_rank);
    rv.tile_elems = static_cast<std::uint64_t>(m.tile_elems);
    rv.segs = static_cast<std::uint32_t>(w.segmented ? w.segs : 1);
  }

  // Answers whether a view reads a distributed result of the unit's own work set.
  struct unit_env_holder {
    unit_env env;
    unit_env_holder(prepared_kernel &p, int map_id) {
      env.distributed_segments = [&p, map_id](int view) -> std::int64_t {
        const int work = p.graph.maps[map_id].work;
        for (int v = view; v != no_view; v = p.graph.views[v].parent) {
          const view_rec &r = p.graph.views[v];
          if (r.kind == vk::result) {
            const map_rec &pm = p.graph.maps[r.map];
            if (pm.work == work && pm.run == run_kind::distributed)
              return p.works[work].segs;
            return 0;
          }
        }
        return 0;
      };
    }
  };

  static void assign_native_size(prepared_kernel &p) {
    std::int64_t size = 1;
    for (const work_conc &w : p.works)
      if (!w.items)
        size = std::max(size, w.native);
    p.native_size = size;
  }
};

} // namespace sg::detail

#endif
