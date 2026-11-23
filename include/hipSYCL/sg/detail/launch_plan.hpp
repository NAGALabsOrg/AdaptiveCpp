// SPDX-License-Identifier: BSD-2-Clause
// Resolves a durable graph into the numbers of one launch: ranges of native
// groups, the native group size, frame sizes, local memory and the layout of the
// bound block and the toolchain block.
#ifndef HIPSYCL_SG_DETAIL_LAUNCH_PLAN_HPP
#define HIPSYCL_SG_DETAIL_LAUNCH_PLAN_HPP

#include "hipSYCL/sg/detail/composition_builder.hpp"
#include "hipSYCL/sg/detail/dev/abi.hpp"
#include "hipSYCL/sg/detail/durable_graph.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sg::detail {

struct device_limits {
  std::size_t max_work_group_size = 1024;
  std::size_t local_mem_bytes = 48 * 1024;
  std::size_t max_groups = 0xffffffffull;
};

struct link_plan {
  std::size_t slots_offset = 0, slot_bytes = 0, child_frame_offset = 0, child_frame_stride = 0,
              child_frame_bytes = 0;
};

struct node_plan {
  std::size_t frame_bytes = 0, scratch_offset = 0;
  std::vector<link_plan> links;
  std::size_t closure_offset = 0;
  std::vector<std::size_t> link_closure_offsets;
  std::size_t link_first = 0;
};

struct layer_plan {
  std::size_t range_offset = 0, groups = 0;
  std::size_t backing_offset = static_cast<std::size_t>(-1); // toolchain layers only
  std::size_t backing_bytes = 0, flags_word_offset = 0;
  sycl::range<3> extent{1, 1, 1};
  int first_node = 0;
};

struct launch_plan {
  std::vector<layer_plan> layers; // by order position
  std::vector<node_plan> nodes;   // by node number
  std::size_t grid = 0, work_group_size = 1, local_bytes = 256, bound_block_bytes = 0,
              flags_byte_offset = 0, flag_words = 0, toolchain_bytes = 0;
  std::size_t layers_offset = 0, nodes_offset = 0, links_offset = 0;
};

inline std::size_t local_bucket(std::size_t bytes) {
  std::size_t b = 256;
  while (b < bytes)
    b *= 2;
  return b;
}

namespace plan_detail {

inline std::size_t a16(std::size_t v) { return (v + 15) / 16 * 16; }
inline std::size_t a8(std::size_t v) { return (v + 7) / 8 * 8; }
inline std::size_t a64(std::size_t v) { return (v + 63) / 64 * 64; }

struct offender_info {
  std::size_t slots = 0, frame = 0;
  std::string text;
};

struct frames {
  const durable_graph &g;
  const def_graph &dg;
  const durable_layer &layer;
  std::vector<node_plan> &nodes;
  int base;
  offender_info &off;

  std::size_t frame(int t) {
    const def_node &d = dg.nodes[layer.tree[t].def];
    node_plan &np = nodes[base + t];
    np.links.assign(d.links.size(), link_plan{});
    std::size_t f = 0, transient = 0;
    std::vector<std::size_t> child(d.links.size(), 0);
    for (std::size_t k = 0; k < d.links.size(); ++k) {
      const link &lk = d.links[k];
      if (lk.kind == link_kind::global)
        continue;
      const int ct = child_node(layer, t, static_cast<int>(k));
      child[k] = frame(ct);
      if (lk.kind == link_kind::local) {
        const def_node &s = dg.nodes[lk.source];
        const std::size_t slot = a16(s.group_elems() * s.elem_bytes);
        np.links[k].slot_bytes = slot;
        np.links[k].slots_offset = f;
        f += slot * lk.count;
        transient = std::max(transient, child[k]);
        if (slot * lk.count >= off.slots) {
          off.slots = slot * lk.count;
          off.text = "the local mapping from parallel_for " + std::to_string(lk.source) +
                     " into parallel_for " + std::to_string(layer.tree[t].def) + " (" +
                     std::to_string(lk.count) + " slots)";
        }
      }
    }
    const std::size_t transient_offset = f;
    f += a16(transient);
    for (std::size_t k = 0; k < d.links.size(); ++k) {
      const link_kind kd = d.links[k].kind;
      np.links[k].child_frame_bytes = child[k];
      if (kd == link_kind::local) {
        np.links[k].child_frame_offset = transient_offset;
      } else if (kd == link_kind::item && child[k] > 0) {
        np.links[k].child_frame_stride = a16(child[k]);
        np.links[k].child_frame_offset = f;
        f += d.links[k].count * a16(child[k]);
      }
    }
    np.scratch_offset = f;
    if (d.kind == body_kind::group_body)
      f += a16(d.scratch_bytes);
    np.frame_bytes = f;
    return f;
  }
};

} // namespace plan_detail

inline launch_plan plan_launch(const durable_graph &g, const def_graph &dg,
                               const std::vector<int> &order,
                               const std::vector<std::size_t> &closure_sizes,
                               const device_limits &lim) {
  using namespace plan_detail;
  launch_plan p;
  const std::vector<int> base = node_bases(g, order);
  std::size_t total_nodes = 0, total_links = 0;
  for (int li : order)
    total_nodes += g.layers[li].tree.size();
  p.nodes.assign(total_nodes, node_plan{});

  bool any_item = false;
  std::size_t max_item_group = 1, max_frame = 0;
  offender_info off;
  std::size_t frame_of_offender = 0;
  for (int li : order) {
    const durable_layer &l = g.layers[li];
    const def_node &root = dg.nodes[l.def];
    layer_plan lp;
    lp.range_offset = p.grid;
    lp.groups = root.grid().size();
    lp.extent = root.total_shape();
    lp.backing_bytes = lp.extent.size() * root.elem_bytes;
    lp.first_node = base[li];
    p.grid += lp.groups;
    p.flag_words += lp.groups;
    for (const inline_node &n : l.tree) {
      const def_node &d = dg.nodes[n.def];
      total_links += d.links.size();
      if (d.kind == body_kind::item_body) {
        any_item = true;
        max_item_group = std::max(max_item_group, d.group_elems());
      }
    }
    frames fc{g, dg, l, p.nodes, base[li], off};
    const std::size_t fb = fc.frame(0);
    if (fb >= max_frame) {
      max_frame = fb;
      frame_of_offender = fb;
    }
    p.layers.push_back(lp);
  }

  if (p.grid > lim.max_groups)
    throw error{errc::limit_exceeded, "the launch needs " + std::to_string(p.grid) +
                                          " native groups; the device allows " +
                                          std::to_string(lim.max_groups)};
  p.work_group_size = any_item ? max_item_group : std::min<std::size_t>(128, lim.max_work_group_size);
  if (p.work_group_size > lim.max_work_group_size)
    throw error{errc::work_group_too_large,
                "the largest logical group has " + std::to_string(p.work_group_size) +
                    " items; the device's work-group limit is " +
                    std::to_string(lim.max_work_group_size)};
  if (max_frame > lim.local_mem_bytes)
    throw error{errc::local_memory_exceeded,
                (off.text.empty() ? std::string{"a local frame"} : off.text) + " needs " +
                    std::to_string(off.slots) + " bytes of slots; the frame is " +
                    std::to_string(frame_of_offender) + " bytes and the device offers " +
                    std::to_string(lim.local_mem_bytes)};
  p.local_bytes = local_bucket(max_frame);

  // bound block
  p.layers_offset = 64;
  p.nodes_offset = a8(p.layers_offset + order.size() * sizeof(dev::layer_rec));
  p.links_offset = a8(p.nodes_offset + total_nodes * sizeof(dev::node_rec));
  std::size_t at = p.links_offset + total_links * sizeof(dev::link_rec);
  std::size_t link_first = 0;
  for (int li : order) {
    const durable_layer &l = g.layers[li];
    for (std::size_t t = 0; t < l.tree.size(); ++t) {
      node_plan &np = p.nodes[base[li] + t];
      const def_node &d = dg.nodes[l.tree[t].def];
      np.link_first = link_first;
      link_first += d.links.size();
      at = a16(at);
      np.closure_offset = at;
      at += d.closure >= 0 ? closure_sizes[d.closure] : 0;
      for (const link &lk : d.links) {
        at = a16(at);
        np.link_closure_offsets.push_back(at);
        at += lk.closure >= 0 ? closure_sizes[lk.closure] : 0;
      }
    }
  }
  p.bound_block_bytes = a64(at);
  p.flags_byte_offset = p.bound_block_bytes;
  std::size_t end = a64(p.flags_byte_offset + p.flag_words * 4);
  std::size_t words = 0;
  for (std::size_t i = 0; i < order.size(); ++i) {
    p.layers[i].flags_word_offset = words;
    words += p.layers[i].groups;
    if (!g.layers[order[i]].result_backed) {
      p.layers[i].backing_offset = end;
      end = a64(end + p.layers[i].backing_bytes);
    }
  }
  p.toolchain_bytes = end;
  return p;
}

} // namespace sg::detail

#endif
