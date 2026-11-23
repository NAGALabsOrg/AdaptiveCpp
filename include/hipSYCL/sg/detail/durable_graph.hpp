// SPDX-License-Identifier: BSD-2-Clause
// The durable graph: layers keyed by (definition, backing), built by a pull walk
// from each sink. Item and local sources are inlined; global sources are shared
// layers.
#ifndef HIPSYCL_SG_DETAIL_DURABLE_GRAPH_HPP
#define HIPSYCL_SG_DETAIL_DURABLE_GRAPH_HPP

#include "hipSYCL/sg/detail/def_graph.hpp"
#include "hipSYCL/sg/detail/type_id.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace sg::detail {

struct inline_node {
  int def = -1;
  int parent = -1;
  int parent_link = -1;
};

struct durable_layer {
  int def = -1;
  bool result_backed = true;
  std::uint32_t slot = 0;
  result_kind kind = result_kind::plain;
  std::string write_symbol;
  std::vector<inline_node> tree; // pre-order; [0] is the root
  std::vector<int> deps;         // layers of global sources, unique
};

struct durable_graph {
  std::vector<durable_layer> layers;
};

// The shared layer of a globally mapped definition: its first plain sink layer,
// else its toolchain layer; -1 if none.
inline int global_layer_of(const durable_graph &g, int def) {
  for (std::size_t i = 0; i < g.layers.size(); ++i)
    if (g.layers[i].def == def && g.layers[i].result_backed &&
        g.layers[i].kind == result_kind::plain)
      return static_cast<int>(i);
  for (std::size_t i = 0; i < g.layers.size(); ++i)
    if (g.layers[i].def == def && !g.layers[i].result_backed)
      return static_cast<int>(i);
  return -1;
}

// Tree index of the child of tree node t through link k, or -1.
inline int child_node(const durable_layer &l, int t, int k) {
  for (std::size_t i = 0; i < l.tree.size(); ++i)
    if (l.tree[i].parent == t && l.tree[i].parent_link == k)
      return static_cast<int>(i);
  return -1;
}

inline durable_graph build_durable_graph(const def_graph &dg) {
  durable_graph out;
  for (const sink &s : dg.sinks) {
    durable_layer l;
    l.def = s.node;
    l.result_backed = true;
    l.slot = s.slot;
    l.kind = s.kind;
    l.write_symbol =
        s.kind == result_kind::plain ? dg.nodes[s.node].store_symbol : s.op_symbol;
    out.layers.push_back(std::move(l));
  }

  for (std::size_t i = 0; i < out.layers.size(); ++i) {
    std::vector<inline_node> tree;
    std::vector<int> deps;
    std::function<void(int, int, int)> add = [&](int def, int parent, int parent_link) {
      const int me = static_cast<int>(tree.size());
      tree.push_back({def, parent, parent_link});
      const def_node &d = dg.nodes[def];
      for (std::size_t k = 0; k < d.links.size(); ++k) {
        const link &lk = d.links[k];
        if (lk.kind == link_kind::global) {
          int gl = global_layer_of(out, lk.source);
          if (gl < 0) {
            durable_layer t;
            t.def = lk.source;
            t.result_backed = false;
            t.write_symbol = dg.nodes[lk.source].store_symbol;
            out.layers.push_back(std::move(t));
            gl = static_cast<int>(out.layers.size() - 1);
          }
          if (std::find(deps.begin(), deps.end(), gl) == deps.end())
            deps.push_back(gl);
        } else {
          add(lk.source, me, static_cast<int>(k));
        }
      }
    };
    add(out.layers[i].def, -1, -1);
    out.layers[i].tree = std::move(tree);
    out.layers[i].deps = std::move(deps);
  }
  return out;
}

// Topological, sources before consumers; ties by layer creation index.
inline std::vector<int> upstream_order(const durable_graph &g) {
  const int n = static_cast<int>(g.layers.size());
  std::vector<int> pending(n);
  for (int i = 0; i < n; ++i)
    pending[i] = static_cast<int>(g.layers[i].deps.size());
  std::vector<bool> done(n, false);
  std::vector<int> order;
  while (static_cast<int>(order.size()) < n) {
    int pick = -1;
    for (int i = 0; i < n; ++i)
      if (!done[i] && pending[i] == 0) {
        pick = i;
        break;
      }
    if (pick < 0)
      throw error{errc::internal, "upstream_order: the durable graph has a cycle"};
    done[pick] = true;
    order.push_back(pick);
    for (int i = 0; i < n; ++i)
      for (int d : g.layers[i].deps)
        if (d == pick)
          --pending[i];
  }
  return order;
}

inline std::string structure_string(const durable_graph &g, const def_graph &dg,
                                    const std::vector<int> &order) {
  std::vector<int> pos(g.layers.size(), -1);
  for (std::size_t p = 0; p < order.size(); ++p)
    pos[order[p]] = static_cast<int>(p);
  std::string s;
  for (int li : order) {
    const durable_layer &l = g.layers[li];
    s += "layer " + std::to_string(static_cast<int>(l.kind)) + " " + l.write_symbol + " deps";
    for (int d : l.deps)
      s += " " + std::to_string(pos[d]);
    s += "\n";
    for (std::size_t t = 0; t < l.tree.size(); ++t) {
      const def_node &d = dg.nodes[l.tree[t].def];
      s += " node " + std::to_string(static_cast<int>(d.kind)) + " " + d.body_symbol + " " +
           d.store_symbol + " " + std::to_string(d.type) + " " + std::to_string(d.rank) +
           " parent " + std::to_string(l.tree[t].parent) + ":" +
           std::to_string(l.tree[t].parent_link) + "\n";
      for (std::size_t k = 0; k < d.links.size(); ++k) {
        const link &lk = d.links[k];
        const int src = lk.kind == link_kind::global
                            ? pos[global_layer_of(g, lk.source)]
                            : child_node(l, static_cast<int>(t), static_cast<int>(k));
        s += "  link " + std::to_string(static_cast<int>(lk.kind)) + " " +
             std::to_string(static_cast<int>(lk.form)) + " " + lk.positions_symbol + " " +
             std::to_string(src) + "\n";
      }
    }
  }
  return s;
}

inline std::uint64_t structure_hash(const durable_graph &g, const def_graph &dg,
                                    const std::vector<int> &order) {
  return fnv1a64(structure_string(g, dg, order));
}

} // namespace sg::detail

#endif
