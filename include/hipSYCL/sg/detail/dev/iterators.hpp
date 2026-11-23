// SPDX-License-Identifier: BSD-2-Clause
// Device-side helpers behind group_iterator and group_array.
#ifndef HIPSYCL_SG_DETAIL_DEV_ITERATORS_HPP
#define HIPSYCL_SG_DETAIL_DEV_ITERATORS_HPP

#include "hipSYCL/sg/detail/dev/abi.hpp"
#include "hipSYCL/sg/detail/dev/drivers.hpp"
#include "hipSYCL/sg/detail/dev/layout.hpp"
#include "hipSYCL/sg/detail/dev/placeholders.hpp"

#include <cstddef>
#include <cstdint>

namespace sg::detail::dev {

// A strided read-only window onto an array: byte base, strides in elements.
struct view_rep {
  std::uint64_t base;
  std::uint64_t stride[3];
  std::uint64_t extent[3];
};

// In probe mode (host, ctx == nullptr) iterator_state::frame points at this.
struct probe_iter_info {
  std::uint64_t shape[3];
  std::uint64_t count;
};

inline bool is_probe(const iterator_state &s) { return s.ctx == nullptr; }

inline view_rep make_probe_view(const iterator_state &s) {
  const probe_iter_info *p = static_cast<const probe_iter_info *>(s.frame);
  view_rep v;
  v.base = 0;
  for (int d = 0; d < 3; ++d) {
    v.extent[d] = p->shape[d];
    v.stride[d] = 0;
  }
  return v;
}

inline std::uint64_t iter_count(const iterator_state &s) {
  if (is_probe(s))
    return static_cast<const probe_iter_info *>(s.frame)->count;
  return link_at(s.ctx, s.node, s.link).count;
}

inline void *iter_child_frame(const iterator_state &s, const link_rec &lr, std::uint64_t k) {
  if (lr.child_frame_stride == 0)
    return nullptr;
  return static_cast<char *>(s.frame) + lr.child_frame_offset + k * lr.child_frame_stride;
}

inline view_rep local_view(const iterator_state &s, std::uint64_t k) {
  if (is_probe(s))
    return make_probe_view(s);
  const link_rec &lr = link_at(s.ctx, s.node, s.link);
  view_rep v;
  v.base = reinterpret_cast<std::uint64_t>(static_cast<char *>(s.frame) + lr.slots_offset +
                                           k * lr.slot_bytes);
  v.stride[2] = 1;
  v.stride[1] = lr.src_group_shape[2];
  v.stride[0] = lr.src_group_shape[1] * lr.src_group_shape[2];
  for (int d = 0; d < 3; ++d)
    v.extent[d] = lr.src_group_shape[d];
  return v;
}

inline view_rep global_view(const iterator_state &s, std::uint64_t k) {
  if (is_probe(s))
    return make_probe_view(s);
  const link_rec &lr = link_at(s.ctx, s.node, s.link);
  std::uint64_t src = 0;
  ph::positions(s.ctx, s.node, s.link, s.g, static_cast<std::uint32_t>(k), &src);
  const layer_rec &layer = layer_at(s.ctx, lr.source);
  const tile_out t = layer_tile(layer, node_at(s.ctx, layer.root_node), src);
  view_rep v;
  v.base = t.base;
  for (int d = 0; d < 3; ++d) {
    v.stride[d] = t.stride[d];
    v.extent[d] = lr.src_group_shape[d];
  }
  return v;
}

template <class T> T item_value_of(const iterator_state &s, std::uint64_t k) {
  T v{};
  if (is_probe(s))
    return v;
  const link_rec &lr = link_at(s.ctx, s.node, s.link);
  std::uint64_t src = 0;
  ph::positions(s.ctx, s.node, s.link, s.g, static_cast<std::uint32_t>(k), &src);
  ph::item_value(s.ctx, s.node, s.link, src, s.l, iter_child_frame(s, lr, k), &v);
  return v;
}

} // namespace sg::detail::dev

#endif
