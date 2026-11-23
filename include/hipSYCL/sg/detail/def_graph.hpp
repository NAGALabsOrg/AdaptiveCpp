// SPDX-License-Identifier: BSD-2-Clause
// The definition graph: what the kernel builder records.
#ifndef HIPSYCL_SG_DETAIL_DEF_GRAPH_HPP
#define HIPSYCL_SG_DETAIL_DEF_GRAPH_HPP

#include "hipSYCL/sg/detail/form_kind.hpp"
#include "hipSYCL/sg/error.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sg::detail {

enum class link_kind : std::uint32_t { item, local, global };
enum class body_kind : std::uint32_t { item_body, group_body };
enum class result_kind : std::uint32_t { plain, inplace };

struct link {
  link_kind kind = link_kind::item;
  int source = -1;
  std::string positions_symbol;
  form_kind form = form_kind::single;
  std::size_t count = 1;
  int closure = -1;
};

struct def_node {
  body_kind kind = body_kind::item_body;
  std::string body_symbol, store_symbol;
  std::uint64_t type = 0;
  std::size_t elem_bytes = 0;
  int rank = 1;
  sycl::range<3> range{1, 1, 1};
  sycl::range<3> group_shape{1, 1, 1};
  std::size_t scratch_bytes = 0;
  std::vector<link> links;
  int closure = -1;

  sycl::range<3> grid() const {
    if (kind == body_kind::group_body)
      return range;
    sycl::range<3> g(1, 1, 1);
    for (int d = 0; d < 3; ++d)
      g[d] = group_shape[d] ? range[d] / group_shape[d] : 0;
    return g;
  }
  sycl::range<3> total_shape() const {
    if (kind == body_kind::item_body)
      return range;
    return sycl::range<3>(range[0] * group_shape[0], range[1] * group_shape[1],
                          range[2] * group_shape[2]);
  }
  std::size_t group_elems() const { return group_shape.size(); }
};

struct sink {
  int node = -1;
  std::uint32_t slot = 0;
  result_kind kind = result_kind::plain;
  std::string op_symbol;
};

struct def_graph {
  std::vector<def_node> nodes;
  std::vector<sink> sinks;
};

inline void validate(const def_graph &g) {
  const int n = static_cast<int>(g.nodes.size());
  for (int i = 0; i < n; ++i) {
    const def_node &d = g.nodes[i];
    if (d.kind == body_kind::group_body && d.links.empty())
      throw error{errc::builder_misuse,
                  "parallel_for " + std::to_string(i) + ": a group body needs at least one mapping"};
    if (d.kind == body_kind::item_body)
      for (int k = 0; k < 3; ++k)
        if (d.group_shape[k] == 0 || d.range[k] % d.group_shape[k] != 0)
          throw error{errc::shape_mismatch,
                      "parallel_for " + std::to_string(i) +
                          ": the range is not a multiple of the group shape"};
    for (const link &l : d.links) {
      if (l.source < 0 || l.source >= i)
        throw error{errc::builder_misuse,
                    "map_groups in parallel_for " + std::to_string(i) +
                        ": a definition can only map definitions created before it"};
      if (l.kind == link_kind::item) {
        const def_node &s = g.nodes[l.source];
        if (s.kind != body_kind::item_body)
          throw error{errc::mapping_mismatch, "an item mapping in parallel_for " +
                                                  std::to_string(i) + " has a group-body source"};
        if (!(s.group_shape == d.group_shape))
          throw error{errc::mapping_mismatch,
                      "an item mapping in parallel_for " + std::to_string(i) +
                          " needs a source with the consumer's group shape"};
      }
    }
  }
  for (std::size_t a = 0; a < g.sinks.size(); ++a) {
    if (g.sinks[a].node < 0 || g.sinks[a].node >= n)
      throw error{errc::builder_misuse, "evaluate: unknown definition"};
    for (std::size_t b = a + 1; b < g.sinks.size(); ++b)
      if (g.sinks[b].node == g.sinks[a].node && g.sinks[b].slot == g.sinks[a].slot)
        throw error{errc::builder_misuse,
                    "evaluate: a definition is evaluated once per result"};
  }
}

} // namespace sg::detail

#endif
