// SPDX-License-Identifier: BSD-2-Clause
// The bound block: every launch-specific record and patched closure, built on the
// host and uploaded in one copy.
#ifndef HIPSYCL_SG_DETAIL_BOUND_BLOCK_HPP
#define HIPSYCL_SG_DETAIL_BOUND_BLOCK_HPP

#include "hipSYCL/sg/detail/dev/abi.hpp"
#include "hipSYCL/sg/detail/launch_plan.hpp"
#include "hipSYCL/sg/detail/tree_store.hpp"

#include <cstddef>
#include <cstring>
#include <vector>

namespace sg::detail {

inline std::vector<std::byte>
build_bound_block(const kernel_tree &t, const launch_plan &p,
                  const std::vector<slot_binding> &by_slot,
                  const std::vector<std::uint64_t> &result_address_by_layer,
                  std::uint64_t block_address) {
  std::vector<std::byte> out(p.bound_block_bytes, std::byte{0});
  auto put = [&](std::size_t off, const void *src, std::size_t n) {
    std::memcpy(out.data() + off, src, n);
  };
  const std::vector<int> base = node_bases(t.durable, t.order);

  dev::launch_ctx ctx{};
  ctx.layers = block_address + p.layers_offset;
  ctx.nodes = block_address + p.nodes_offset;
  ctx.links = block_address + p.links_offset;
  ctx.num_layers = static_cast<std::uint32_t>(t.order.size());
  put(0, &ctx, sizeof(ctx));

  std::size_t node_no = 0;
  for (std::size_t pos = 0; pos < t.order.size(); ++pos) {
    const durable_layer &l = t.durable.layers[t.order[pos]];
    const layer_plan &lp = p.layers[pos];
    const def_node &root = t.defs.nodes[l.def];

    dev::layer_rec lr{};
    lr.range_offset = lp.range_offset;
    lr.groups = lp.groups;
    lr.backing = l.result_backed ? result_address_by_layer.at(pos)
                                 : block_address + lp.backing_offset;
    lr.flags = block_address + p.flags_byte_offset + 4 * lp.flags_word_offset;
    for (int d = 0; d < 3; ++d)
      lr.extent[d] = lp.extent[d];
    lr.root_node = static_cast<std::uint32_t>(lp.first_node);
    lr.result_kind = static_cast<std::uint32_t>(l.kind);
    put(p.layers_offset + pos * sizeof(dev::layer_rec), &lr, sizeof(lr));
    (void)root;

    for (std::size_t tn = 0; tn < l.tree.size(); ++tn, ++node_no) {
      const def_node &d = t.defs.nodes[l.tree[tn].def];
      const node_plan &np = p.nodes[node_no];
      dev::node_rec nr{};
      const sycl::range<3> grid = d.grid();
      for (int k = 0; k < 3; ++k) {
        nr.range[k] = d.range[k];
        nr.group_shape[k] = d.group_shape[k];
        nr.grid[k] = grid[k];
      }
      nr.closure = block_address + np.closure_offset;
      nr.elem_bytes = d.elem_bytes;
      nr.group_elems = d.group_elems();
      nr.frame_bytes = np.frame_bytes;
      nr.scratch_offset = np.scratch_offset;
      nr.rank = static_cast<std::uint32_t>(d.rank);
      nr.body_kind = static_cast<std::uint32_t>(d.kind);
      nr.link_first = static_cast<std::uint32_t>(np.link_first);
      nr.link_count = static_cast<std::uint32_t>(d.links.size());
      put(p.nodes_offset + node_no * sizeof(dev::node_rec), &nr, sizeof(nr));
      if (d.closure >= 0) {
        const std::vector<std::byte> b = patch_closure(t.closures[d.closure], by_slot);
        put(np.closure_offset, b.data(), b.size());
      }
      for (std::size_t k = 0; k < d.links.size(); ++k) {
        const link &lk = d.links[k];
        const def_node &src = t.defs.nodes[lk.source];
        dev::link_rec rr{};
        rr.kind = static_cast<std::uint32_t>(lk.kind);
        rr.form = static_cast<std::uint32_t>(lk.form);
        rr.source = lk.kind == link_kind::global
                        ? static_cast<std::uint32_t>([&] {
                            const int gl = global_layer_of(t.durable, lk.source);
                            for (std::size_t q = 0; q < t.order.size(); ++q)
                              if (t.order[q] == gl)
                                return q;
                            return std::size_t{0};
                          }())
                        : static_cast<std::uint32_t>(
                              base[t.order[pos]] + child_node(l, static_cast<int>(tn), static_cast<int>(k)));
        rr.src_rank = static_cast<std::uint32_t>(src.rank);
        rr.count = lk.count;
        rr.closure = block_address + np.link_closure_offsets[k];
        const link_plan &lpn = np.links[k];
        rr.slot_bytes = lpn.slot_bytes;
        rr.slots_offset = lpn.slots_offset;
        rr.child_frame_offset = lpn.child_frame_offset;
        rr.child_frame_stride = lpn.child_frame_stride;
        rr.child_frame_bytes = lpn.child_frame_bytes;
        const sycl::range<3> sg = src.grid();
        for (int q = 0; q < 3; ++q) {
          rr.src_grid[q] = sg[q];
          rr.src_group_shape[q] = src.group_shape[q];
        }
        rr.src_elem_bytes = src.elem_bytes;
        put(p.links_offset + (np.link_first + k) * sizeof(dev::link_rec), &rr, sizeof(rr));
        if (lk.closure >= 0) {
          const std::vector<std::byte> b = patch_closure(t.closures[lk.closure], by_slot);
          put(np.link_closure_offsets[k], b.data(), b.size());
        }
      }
    }
  }
  return out;
}

} // namespace sg::detail

#endif
