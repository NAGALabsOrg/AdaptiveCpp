// SPDX-License-Identifier: BSD-2-Clause
// Fixed-layout records shared by host and device. All addresses are absolute
// device addresses written per launch.
#ifndef HIPSYCL_SG_DETAIL_DEV_ABI_HPP
#define HIPSYCL_SG_DETAIL_DEV_ABI_HPP

#include <cstdint>

namespace sg::detail::dev {

constexpr std::uint32_t flag_unclaimed = 0, flag_in_progress = 1, flag_done = 2,
                        no_layer = 0xffffffffu;

struct launch_ctx {
  std::uint64_t layers, nodes, links;
  std::uint32_t num_layers, pad;
};

struct layer_rec {
  std::uint64_t range_offset, groups, backing, flags;
  std::uint64_t extent[3];
  std::uint32_t root_node, result_kind;
};

struct node_rec {
  std::uint64_t range[3], group_shape[3], grid[3];
  std::uint64_t closure, elem_bytes, group_elems, frame_bytes, scratch_offset;
  std::uint32_t rank, body_kind, link_first, link_count;
};

struct link_rec {
  std::uint32_t kind, form, source, src_rank;
  std::uint64_t count, closure, slot_bytes, slots_offset, child_frame_offset,
      child_frame_stride, child_frame_bytes;
  std::uint64_t src_grid[3], src_group_shape[3], src_elem_bytes;
};

struct tile_out {
  std::uint64_t base;
  std::uint64_t stride[3]; // in elements
  std::uint64_t elem_bytes;
  std::uint64_t group_shape[3];
};

struct iterator_state {
  const launch_ctx *ctx;
  std::uint32_t node, link;
  std::uint64_t g, l;
  void *frame;
};

static_assert(sizeof(launch_ctx) == 32);
static_assert(sizeof(layer_rec) == 64);
static_assert(sizeof(node_rec) == 128);
static_assert(sizeof(link_rec) == 128);
static_assert(sizeof(tile_out) == 64);

} // namespace sg::detail::dev

#endif
