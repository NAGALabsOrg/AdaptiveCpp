// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_COMPOSITION_BUILDER_HPP
#define HIPSYCL_SG_DETAIL_COMPOSITION_BUILDER_HPP

#include "hipSYCL/glue/llvm-sscp/composition.hpp"
#include "hipSYCL/sg/detail/durable_graph.hpp"

#include <vector>

namespace sg::detail {

// First node number of each layer (indexed by layer, not by order position).
inline std::vector<int> node_bases(const durable_graph &g, const std::vector<int> &order) {
  std::vector<int> base(g.layers.size(), 0);
  int next = 0;
  for (int li : order) {
    base[li] = next;
    next += static_cast<int>(g.layers[li].tree.size());
  }
  return base;
}

inline hipsycl::glue::sscp::composition_config
make_composition_config(const durable_graph &g, const def_graph &dg,
                        const std::vector<int> &order) {
  hipsycl::glue::sscp::composition_config c;
  c.unique_hash = structure_hash(g, dg, order);
  std::vector<int> pos(g.layers.size(), 0);
  for (std::size_t p = 0; p < order.size(); ++p)
    pos[order[p]] = static_cast<int>(p);
  const std::vector<int> base = node_bases(g, order);
  for (int li : order) {
    const durable_layer &l = g.layers[li];
    c.layers.push_back({static_cast<std::uint32_t>(base[li])});
    for (std::size_t t = 0; t < l.tree.size(); ++t) {
      const def_node &d = dg.nodes[l.tree[t].def];
      hipsycl::glue::sscp::composition_node n;
      n.body_symbol = d.body_symbol;
      n.write_symbol = t == 0 ? l.write_symbol : d.store_symbol;
      n.body_kind = static_cast<std::uint32_t>(d.kind);
      for (std::size_t k = 0; k < d.links.size(); ++k) {
        hipsycl::glue::sscp::composition_link cl;
        cl.kind = static_cast<std::uint32_t>(d.links[k].kind);
        cl.positions_symbol = d.links[k].positions_symbol;
        cl.source = d.links[k].kind == link_kind::global
                        ? static_cast<std::uint32_t>(pos[global_layer_of(g, d.links[k].source)])
                        : static_cast<std::uint32_t>(
                              base[li] + child_node(l, static_cast<int>(t), static_cast<int>(k)));
        n.links.push_back(std::move(cl));
      }
      c.nodes.push_back(std::move(n));
    }
  }
  return c;
}

} // namespace sg::detail

#endif
