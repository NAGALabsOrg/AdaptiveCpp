// SPDX-License-Identifier: BSD-2-Clause
// A selected window as a 1-D run: element `pos` of the box a unit pulled from a
// tile, read through the view descriptors or from the group's local copy of the
// tile.
#ifndef HIPSYCL_SG_DETAIL_DEV_ITERATORS_HPP
#define HIPSYCL_SG_DETAIL_DEV_ITERATORS_HPP

#include "hipSYCL/sg/detail/dev/abi.hpp"
#include "hipSYCL/sg/detail/dev/backing.hpp"
#include "hipSYCL/sg/detail/dev/layout.hpp"
#include "hipSYCL/sg/detail/dev/view_read.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sg::detail::dev {

template <class A>
inline A read_window(block_t block, const sel_rec &s, std::int64_t pos, const std::byte *local,
                     device_visitor &w) {
  A v{};
  const view_node *nodes = views_of(block);
  const view_node &vn = nodes[s.view];
  std::int64_t c[3];
  unflatten3(pos, s.ext, c);
  for (int d = 0; d < 3; ++d)
    c[d] += s.off[d];
  const std::int64_t tile_lin = flatten3(c, vn.shape);
  if (s.local_off >= 0) {
    std::memcpy(&v, local + s.local_off + tile_lin * sizeof(A), sizeof(A));
  } else {
    std::byte buf[16];
    if (read_view(nodes, s.view, s.tile, tile_lin, local, w, buf))
      std::memcpy(&v, buf, sizeof(A));
  }
  return v;
}

} // namespace sg::detail::dev

#endif
