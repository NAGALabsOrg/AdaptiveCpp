#pragma once
#include "hipSYCL/sg/detail/composition_builder.hpp"
#include "hipSYCL/sg/detail/def_graph.hpp"
#include "hipSYCL/sg/detail/durable_graph.hpp"
#include "hipSYCL/sg/detail/launch_plan.hpp"

#include <string>
#include <vector>

namespace sgt {
using namespace sg::detail;

inline sg::detail::link mk_link(link_kind k, int source, const std::string &pos = "pos", std::size_t count = 1) {
  sg::detail::link l;
  l.kind = k;
  l.source = source;
  l.positions_symbol = pos;
  l.count = count;
  return l;
}

// an item definition of rank 1: `groups` logical groups of `gs` items
inline def_node item_def(const std::string &name, std::size_t groups, std::size_t gs,
                         std::vector<sg::detail::link> links = {}, std::size_t elem = 4) {
  def_node d;
  d.kind = body_kind::item_body;
  d.body_symbol = "body_" + name;
  d.store_symbol = "store_" + name;
  d.type = 1;
  d.elem_bytes = elem;
  d.rank = 1;
  d.range = sycl::range<3>(groups * gs, 1, 1);
  d.group_shape = sycl::range<3>(gs, 1, 1);
  d.links = std::move(links);
  return d;
}

inline def_node group_def(const std::string &name, std::size_t groups, std::size_t out,
                          std::vector<sg::detail::link> links, std::size_t scratch = 0, std::size_t elem = 4) {
  def_node d;
  d.kind = body_kind::group_body;
  d.body_symbol = "gbody_" + name;
  d.store_symbol = "store_" + name;
  d.type = 1;
  d.elem_bytes = elem;
  d.rank = 1;
  d.range = sycl::range<3>(groups, 1, 1);
  d.group_shape = sycl::range<3>(out, 1, 1);
  d.links = std::move(links);
  d.scratch_bytes = scratch;
  return d;
}

inline sink plain_sink(int node, std::uint32_t slot) {
  sink s;
  s.node = node;
  s.slot = slot;
  return s;
}
inline sink inplace_sink(int node, std::uint32_t slot, const std::string &op = "combine_op") {
  sink s;
  s.node = node;
  s.slot = slot;
  s.kind = result_kind::inplace;
  s.op_symbol = op;
  return s;
}

// A-to-F of the design: B, C local from A; D local from B and C; E global from C;
// F item from E; sinks B, C, D, F.
inline def_graph a_to_f() {
  def_graph g;
  g.nodes.push_back(item_def("A", 4, 4));
  g.nodes.push_back(item_def("B", 4, 4, {mk_link(link_kind::local, 0)}));
  g.nodes.push_back(item_def("C", 4, 4, {mk_link(link_kind::local, 0)}));
  g.nodes.push_back(item_def("D", 4, 4, {mk_link(link_kind::local, 1), mk_link(link_kind::local, 2)}));
  g.nodes.push_back(item_def("E", 4, 4, {mk_link(link_kind::global, 2)}));
  g.nodes.push_back(item_def("F", 4, 4, {mk_link(link_kind::item, 4)}));
  g.sinks = {plain_sink(1, 0), plain_sink(2, 1), plain_sink(3, 2), plain_sink(5, 3)};
  return g;
}

inline int layer_of_def(const durable_graph &g, int def) {
  for (std::size_t i = 0; i < g.layers.size(); ++i)
    if (g.layers[i].def == def)
      return static_cast<int>(i);
  return -1;
}
} // namespace sgt
