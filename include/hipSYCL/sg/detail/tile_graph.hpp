// SPDX-License-Identifier: BSD-2-Clause
// The tile dependency graph and the order of the native groups. Every pull of a
// result is an edge from the pulled tile to the pulling task; a pull whose tile
// is chosen by kernel data orders against every tile of its source set. The
// groups get tickets so that every edge comes from a lower ticket or from an
// earlier task of the same group; if no order exists the kernel is rejected,
// naming the tiles of the cycle.
#ifndef HIPSYCL_SG_DETAIL_TILE_GRAPH_HPP
#define HIPSYCL_SG_DETAIL_TILE_GRAPH_HPP

#include "hipSYCL/sg/detail/dev/view_read.hpp"
#include "hipSYCL/sg/detail/kernel_preparer.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>

namespace sg::detail {

struct group_ref {
  int work = -1;            // the root work set
  std::int64_t tile = 0;
  std::int64_t seg = 0;
};

// One map's task for one tile (and segment): what a native group runs.
struct task_inst {
  int map = -1;
  std::int64_t tile = 0; // tile of the map's work set (the slice's own tile for a slice)
  std::int64_t seg = 0;
  int group = -1;
  std::uint32_t flag = 0;
  std::vector<std::uint32_t> waits;
};

// One unit's pull of one produced tile.
struct pull {
  int consumer = -1; // task instance
  std::int64_t consumer_unit = 0;
  int producer_map = -1;
  std::int64_t producer_tile = 0;
  std::int64_t producer_unit = -1; // -1: unknown (chosen by data)
  bool by_data = false;
};

struct tile_graph {
  std::vector<group_ref> groups;
  std::vector<task_inst> tasks;
  std::vector<std::vector<int>> group_tasks; // task instances of each group, in map order
  std::vector<pull> pulls;
  std::vector<int> ticket_of_group, group_of_ticket;
  std::int64_t group_index(const prepared_kernel &p, int root, std::int64_t tile, std::int64_t seg) const {
    const work_conc &w = p.works[root];
    return w.first_group + tile * (w.segmented ? w.segs : 1) + seg;
  }
};

namespace tg_detail {

struct recorder {
  const prepared_kernel &p;
  std::map<std::uint32_t, int> map_of_flag;
  struct hit {
    int map;
    std::int64_t tile, local;
    bool dyn;
  };
  std::vector<hit> hits;
  std::set<int> all_tiles_maps;
  bool reads_data() const { return false; }
  bool reached(const view_node &n, std::int64_t tile, std::int64_t local, bool dyn) {
    const auto it = map_of_flag.find(n.flag_base);
    if (it != map_of_flag.end())
      hits.push_back({it->second, tile, local, dyn});
    return false;
  }
  void data_chosen(const view_node *nodes, int view) {
    // every result tile reachable through the elements the indices choose from
    std::vector<int> stack{view};
    std::set<int> seen;
    while (!stack.empty()) {
      const int v = stack.back();
      stack.pop_back();
      if (v == no_view || !seen.insert(v).second)
        continue;
      const view_node &n = nodes[v];
      if (static_cast<vk>(n.kind) == vk::result) {
        const auto it = map_of_flag.find(n.flag_base);
        if (it != map_of_flag.end())
          all_tiles_maps.insert(it->second);
        continue;
      }
      stack.push_back(n.parent);
    }
  }
};

inline bool has_result_below(const prepared_kernel &p, int view, std::vector<signed char> &memo) {
  if (view == no_view)
    return false;
  signed char &m = memo[view];
  if (m >= 0)
    return m == 1;
  m = 0;
  const view_node &n = p.views[view];
  bool r = static_cast<vk>(n.kind) == vk::result;
  if (!r)
    r = has_result_below(p, n.parent, memo) || has_result_below(p, n.parent2, memo);
  m = r ? 1 : 0;
  return r;
}

} // namespace tg_detail

inline tile_graph build_tile_graph(prepared_kernel &p, source_loc where = {"<kernel>", 0}) {
  const kgraph &g = p.graph;
  tile_graph tg;

  // native groups, in the order the sets were declared
  for (std::size_t w = 0; w < p.works.size(); ++w) {
    work_conc &wc = p.works[w];
    if (wc.items || wc.slice)
      continue;
    wc.first_group = static_cast<std::int64_t>(tg.groups.size());
    for (std::int64_t t = 0; t < wc.tiles; ++t)
      for (std::int64_t s = 0; s < (wc.segmented ? wc.segs : 1); ++s)
        tg.groups.push_back({static_cast<int>(w), t, s});
  }
  tg.group_tasks.assign(tg.groups.size(), {});

  // the task instances: one per tile of a map, and per segment
  std::vector<std::vector<int>> inst_of_map(g.maps.size());
  for (std::size_t m = 0; m < g.maps.size(); ++m) {
    const work_conc &w = p.works[g.maps[m].work];
    const work_conc &root = p.works[w.root];
    const std::int64_t segs = w.segmented ? w.segs : 1;
    for (std::int64_t j = 0; j < p.maps[m].tiles; ++j) {
      std::int64_t root_tile = j;
      if (w.slice) {
        std::int64_t c[3], rc[3];
        unflatten3(j, w.grid, c);
        for (int d = 0; d < 3; ++d)
          rc[d] = w.offset[d] + c[d] * w.stride[d];
        root_tile = flatten3(rc, root.grid);
      }
      for (std::int64_t s = 0; s < segs; ++s) {
        task_inst t;
        t.map = static_cast<int>(m);
        t.tile = j;
        t.seg = s;
        t.group = static_cast<int>(tg.group_index(p, w.root, root_tile, s));
        t.flag = p.maps[m].flag_base + static_cast<std::uint32_t>(j * segs + s);
        inst_of_map[m].push_back(static_cast<int>(tg.tasks.size()));
        tg.group_tasks[t.group].push_back(static_cast<int>(tg.tasks.size()));
        tg.tasks.push_back(std::move(t));
      }
    }
  }

  auto instance = [&](int map, std::int64_t tile, std::int64_t seg) {
    const work_conc &w = p.works[g.maps[map].work];
    const std::int64_t segs = w.segmented ? w.segs : 1;
    return inst_of_map[map][static_cast<std::size_t>(tile * segs + seg)];
  };

  // pulls
  tg_detail::recorder rec{p};
  for (std::size_t m = 0; m < g.maps.size(); ++m)
    rec.map_of_flag[p.maps[m].flag_base] = static_cast<int>(m);
  std::vector<signed char> memo(p.views.size(), -1);

  struct group_edge {
    int from, to;
    bool by_data;
  };
  std::vector<group_edge> edges;
  std::set<std::pair<int, int>> edge_seen;
  auto add_edge = [&](int from_group, int to_group, bool by_data) {
    if (from_group == to_group)
      return;
    if (edge_seen.insert({from_group * 2 + (by_data ? 1 : 0), to_group}).second)
      edges.push_back({from_group, to_group, by_data});
  };

  for (std::size_t m = 0; m < g.maps.size(); ++m) {
    const map_conc &mc = p.maps[m];
    const work_conc &w = p.works[g.maps[m].work];
    const std::int64_t segs = w.segmented ? w.segs : 1;
    for (std::int64_t j = 0; j < mc.tiles; ++j) {
      for (std::int64_t s = 0; s < segs; ++s) {
        const int ci = instance(static_cast<int>(m), j, s);
        task_inst &consumer = tg.tasks[ci];
        std::set<std::uint32_t> waits;
        // earlier segments of a distributed run
        if (mc.kind == run_kind::distributed)
          for (std::int64_t s2 = 0; s2 < s; ++s2)
            waits.insert(tg.tasks[instance(static_cast<int>(m), j, s2)].flag);
        for (std::int64_t u = 0; u < mc.units; ++u)
          for (std::size_t k = 0; k < mc.n_inputs; ++k) {
            const input_sel &sel = mc.sel(j, u, k);
            if (!tg_detail::has_result_below(p, sel.view, memo))
              continue;
            const view_node &vn = p.views[sel.view];
            rec.hits.clear();
            rec.all_tiles_maps.clear();
            std::set<std::pair<int, std::int64_t>> seen_hits;
            for (std::int64_t e = 0; e < sel.len; ++e) {
              std::int64_t c[3];
              unflatten3(e, sel.ext, c);
              for (int d = 0; d < 3; ++d)
                c[d] += sel.off[d];
              std::byte out[16];
              dev::read_view(p.views.data(), sel.view, sel.tile, flatten3(c, vn.shape), nullptr, rec, out);
            }
            for (const auto &h : rec.hits) {
              if (!seen_hits.insert({h.map, h.tile * 1000003 + h.local / std::max<std::int64_t>(1, p.maps[h.map].tile_elems / std::max<std::int64_t>(1, p.maps[h.map].units))}).second)
                continue;
              const map_conc &pm = p.maps[h.map];
              const std::int64_t per_unit = std::max<std::int64_t>(1, pm.tile_elems / std::max<std::int64_t>(1, pm.units));
              const work_conc &pw = p.works[g.maps[h.map].work];
              const std::int64_t psegs = pw.segmented ? pw.segs : 1;
              for (std::int64_t ps = 0; ps < psegs; ++ps) {
                const int pi = instance(h.map, h.tile, ps);
                waits.insert(tg.tasks[pi].flag);
                add_edge(tg.tasks[pi].group, consumer.group, false);
              }
              pull pl;
              pl.consumer = ci;
              pl.consumer_unit = u;
              pl.producer_map = h.map;
              pl.producer_tile = h.tile;
              pl.producer_unit = h.local / per_unit;
              tg.pulls.push_back(pl);
            }
            for (int pm : rec.all_tiles_maps) {
              const work_conc &pw = p.works[g.maps[pm].work];
              const std::int64_t psegs = pw.segmented ? pw.segs : 1;
              for (std::int64_t pt = 0; pt < p.maps[pm].tiles; ++pt) {
                for (std::int64_t ps = 0; ps < psegs; ++ps)
                  add_edge(tg.tasks[instance(pm, pt, ps)].group, consumer.group, true);
                pull pl;
                pl.consumer = ci;
                pl.consumer_unit = u;
                pl.producer_map = pm;
                pl.producer_tile = pt;
                pl.by_data = true;
                tg.pulls.push_back(pl);
              }
            }
          }
        consumer.waits.assign(waits.begin(), waits.end());
      }
    }
  }

  // the order: every edge from a lower ticket; natural order where it is valid
  const std::size_t n = tg.groups.size();
  std::vector<std::vector<int>> out(n);
  std::vector<int> indegree(n, 0);
  for (const group_edge &e : edges) {
    out[e.from].push_back(e.to);
    ++indegree[e.to];
  }
  std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
  for (std::size_t i = 0; i < n; ++i)
    if (indegree[i] == 0)
      ready.push(static_cast<int>(i));
  tg.group_of_ticket.clear();
  while (!ready.empty()) {
    const int u = ready.top();
    ready.pop();
    tg.group_of_ticket.push_back(u);
    for (int v : out[u])
      if (--indegree[v] == 0)
        ready.push(v);
  }
  if (tg.group_of_ticket.size() != n) {
    // a strongly connected component among the groups left over
    std::vector<int> idxs(n, -1), low(n, 0);
    std::vector<char> on(n, 0);
    std::vector<int> stack;
    int counter = 0;
    std::vector<int> best;
    std::vector<char> left(n, 0);
    for (std::size_t i = 0; i < n; ++i)
      left[i] = indegree[i] > 0 ? 1 : 0;
    std::function<void(int)> strong = [&](int v) {
      idxs[v] = low[v] = counter++;
      stack.push_back(v);
      on[v] = 1;
      for (int w : out[v]) {
        if (!left[w])
          continue;
        if (idxs[w] < 0) {
          strong(w);
          low[v] = std::min(low[v], low[w]);
        } else if (on[w]) {
          low[v] = std::min(low[v], idxs[w]);
        }
      }
      if (low[v] == idxs[v]) {
        std::vector<int> comp;
        int w;
        do {
          w = stack.back();
          stack.pop_back();
          on[w] = 0;
          comp.push_back(w);
        } while (w != v);
        if (comp.size() > 1 && (best.empty() || *std::min_element(comp.begin(), comp.end()) <
                                                    *std::min_element(best.begin(), best.end())))
          best = comp;
      }
    };
    for (std::size_t i = 0; i < n; ++i)
      if (left[i] && idxs[i] < 0)
        strong(static_cast<int>(i));
    std::sort(best.begin(), best.end());
    bool by_data = false;
    for (const group_edge &e : edges)
      if (e.by_data && std::find(best.begin(), best.end(), e.from) != best.end() &&
          std::find(best.begin(), best.end(), e.to) != best.end())
        by_data = true;
    std::string tiles;
    for (int gi : best) {
      if (!tiles.empty())
        tiles += ", ";
      tiles += "work set " + std::to_string(tg.groups[gi].work) + " tile " + std::to_string(tg.groups[gi].tile);
      if (p.works[tg.groups[gi].work].segmented)
        tiles += " segment " + std::to_string(tg.groups[gi].seg);
    }
    throw build_error(3, by_data ? "kh.index_map" : "kh.map_work", where,
                      "the tiles pull from each other and no order of the native groups satisfies them: " + tiles +
                          "; break the cycle with another work set",
                      {{"tiles_in_cycle", static_cast<std::int64_t>(best.size())}});
  }
  tg.ticket_of_group.assign(n, 0);
  for (std::size_t t = 0; t < n; ++t)
    tg.ticket_of_group[tg.group_of_ticket[t]] = static_cast<int>(t);
  return tg;
}

} // namespace sg::detail

#endif
