// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_DEV_LAYOUT_HPP
#define HIPSYCL_SG_DETAIL_DEV_LAYOUT_HPP

#include "hipSYCL/sg/detail/dev/abi.hpp"

#include <cstdint>

namespace sg::detail::dev {

// row-major, dimension 0 slowest
inline void unflatten3(std::uint64_t k, const std::uint64_t dims[3], std::uint64_t out[3]) {
  for (int d = 2; d >= 0; --d) {
    const std::uint64_t e = dims[d] ? dims[d] : 1;
    out[d] = k % e;
    k /= e;
  }
}
inline std::uint64_t flatten3(const std::uint64_t id[3], const std::uint64_t dims[3]) {
  return (id[0] * dims[1] + id[1]) * dims[2] + id[2];
}

inline tile_out layer_tile(const layer_rec &l, const node_rec &n, std::uint64_t g) {
  tile_out t;
  std::uint64_t gid[3], origin[3];
  unflatten3(g, n.grid, gid);
  for (int d = 0; d < 3; ++d)
    origin[d] = gid[d] * n.group_shape[d];
  t.stride[2] = 1;
  t.stride[1] = l.extent[2];
  t.stride[0] = l.extent[1] * l.extent[2];
  t.base = l.backing + flatten3(origin, l.extent) * n.elem_bytes;
  t.elem_bytes = n.elem_bytes;
  for (int d = 0; d < 3; ++d)
    t.group_shape[d] = n.group_shape[d];
  return t;
}

inline tile_out dense_tile(std::uint64_t base, const std::uint64_t group_shape[3],
                           std::uint64_t elem_bytes) {
  tile_out t;
  t.base = base;
  t.stride[2] = 1;
  t.stride[1] = group_shape[2];
  t.stride[0] = group_shape[1] * group_shape[2];
  t.elem_bytes = elem_bytes;
  for (int d = 0; d < 3; ++d)
    t.group_shape[d] = group_shape[d];
  return t;
}

inline std::uint64_t tile_address(const tile_out &t, std::uint64_t l) {
  std::uint64_t lid[3];
  unflatten3(l, t.group_shape, lid);
  return t.base + (lid[0] * t.stride[0] + lid[1] * t.stride[1] + lid[2] * t.stride[2]) *
                      t.elem_bytes;
}

} // namespace sg::detail::dev

#endif
