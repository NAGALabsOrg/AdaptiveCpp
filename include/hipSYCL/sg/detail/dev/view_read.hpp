// SPDX-License-Identifier: BSD-2-Clause
// Reads the element of a view descriptor, on the host and on the device alike.
// A visitor W decides what happens where a read reaches stored results: the
// plain visitor reads them, the device visitor first waits on the producing
// tile's flags when the tile was chosen by kernel data, and the host recorder
// notes which tiles a window touches.
#ifndef HIPSYCL_SG_DETAIL_DEV_VIEW_READ_HPP
#define HIPSYCL_SG_DETAIL_DEV_VIEW_READ_HPP

#include "hipSYCL/sg/detail/view_desc.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sg::detail::dev {

struct plain_visitor {
  bool reads_data() const { return true; }
  // true: go on and read the element
  bool reached(const view_node &, std::int64_t, std::int64_t, bool) { return true; }
  void data_chosen(const view_node *, int) {}
};

template <class W, int D = 0>
inline bool read_view(const view_node *nodes, int node, std::int64_t tile, std::int64_t local,
                      const std::byte *local_base, W &w, std::byte *out);

inline std::int64_t load_index(const std::byte *p, std::uint32_t bytes) {
  switch (bytes) {
  case 1: { std::int8_t v; std::memcpy(&v, p, 1); return v; }
  case 2: { std::int16_t v; std::memcpy(&v, p, 2); return v; }
  case 4: { std::int32_t v; std::memcpy(&v, p, 4); return v; }
  default: { std::int64_t v; std::memcpy(&v, p, 8); return v; }
  }
}

// The index vector of one element of an index-tile view: from a host table, or
// from integer elements.
template <class W, int D>
inline bool read_index(const view_node *nodes, int node, std::int64_t tile, std::int64_t local,
                       const std::byte *local_base, W &w, std::int64_t idx[3], bool &from_data) {
  const view_node &n = nodes[node];
  idx[0] = idx[1] = idx[2] = 0;
  if (n.kind == static_cast<std::uint32_t>(vk::table)) {
    const std::int64_t *t = reinterpret_cast<const std::int64_t *>(n.table);
    const std::int64_t at = (tile * prod3(n.shape) + local) * n.table_rank;
    for (std::uint32_t k = 0; k < n.table_rank; ++k)
      idx[k] = t[at + k];
    from_data = false;
    return true;
  }
  if constexpr (D < 2) {
    std::byte buf[16];
    if (!read_view<W, D + 1>(nodes, node, tile, local, local_base, w, buf))
      return false;
    idx[0] = load_index(buf, n.elem_bytes);
    from_data = true;
    return true;
  } else {
    return false;
  }
}

template <class W, int D>
inline bool read_view(const view_node *nodes, int node, std::int64_t tile, std::int64_t local,
                      const std::byte *local_base, W &w, std::byte *out) {
  bool coord = false;
  bool dyn = false;
  std::int64_t a = tile, b = local;
  std::int64_t c[3] = {0, 0, 0};
  for (int guard = 0; guard < 64; ++guard) {
    const view_node &n = nodes[node];
    const vk k = static_cast<vk>(n.kind);
    if (coord) {
      if (k == vk::source || k == vk::array) {
        const std::int64_t lin = flatten3(c, n.shape);
        const std::byte *p = n.base_local ? local_base + n.base : reinterpret_cast<const std::byte *>(n.base);
        std::memcpy(out, p + lin * n.elem_bytes, n.elem_bytes);
        return true;
      }
      if (k == vk::join) {
        const view_node &p = nodes[n.parent];
        std::int64_t g[3], l[3];
        for (int d = 0; d < 3; ++d) {
          g[d] = c[d] / p.shape[d];
          l[d] = c[d] % p.shape[d];
        }
        a = flatten3(g, p.grid);
        b = flatten3(l, p.shape);
        node = n.parent;
        coord = false;
        continue;
      }
      return false;
    }
    switch (k) {
    case vk::tile: {
      const view_node &p = nodes[n.parent];
      std::int64_t g[3], l[3];
      unflatten3(a, n.grid, g);
      unflatten3(b, n.shape, l);
      bool outside = false;
      for (int d = 0; d < 3; ++d) {
        c[d] = g[d] * n.shape[d] + l[d];
        if (c[d] >= p.shape[d])
          outside = true;
      }
      if (outside) {
        if (n.oob == static_cast<std::uint32_t>(oob_kind::fill)) {
          std::memcpy(out, n.fill, n.elem_bytes);
          return true;
        }
        for (int d = 0; d < 3; ++d)
          c[d] %= p.shape[d];
      }
      node = n.parent;
      coord = true;
      break;
    }
    case vk::retile: {
      const view_node &p = nodes[n.parent];
      const std::int64_t lin = a * prod3(n.shape) + b;
      const std::int64_t ps = prod3(p.shape);
      a = lin / ps;
      b = lin % ps;
      node = n.parent;
      break;
    }
    case vk::extend: {
      const view_node &p = nodes[n.parent];
      std::int64_t l[3], lp[3] = {0, 0, 0};
      unflatten3(b, n.shape, l);
      for (std::uint32_t d = 0; d < p.shape_rank && d < 3; ++d)
        lp[d] = p.shape[d] == 1 ? 0 : l[d];
      b = flatten3(lp, p.shape);
      node = n.parent;
      break;
    }
    case vk::result: {
      if (!w.reached(n, a, b, dyn))
        return true;
      const std::byte *base = n.base_local ? local_base + n.base : reinterpret_cast<const std::byte *>(n.base);
      const std::uint64_t at = n.base_local ? static_cast<std::uint64_t>(b)
                                            : static_cast<std::uint64_t>(a) * n.tile_elems + b;
      std::memcpy(out, base + at * n.elem_bytes, n.elem_bytes);
      return true;
    }
    case vk::index_map: {
      const view_node &p = nodes[n.parent];
      const bool table = nodes[n.parent2].kind == static_cast<std::uint32_t>(vk::table);
      if (!table && !w.reads_data()) {
        w.data_chosen(nodes, n.parent);
        return true;
      }
      std::int64_t idx[3];
      bool from_data = false;
      if (!read_index<W, D>(nodes, n.parent2, a, b, local_base, w, idx, from_data))
        return false;
      dyn = dyn || from_data;
      bool outside = false;
      for (int d = 0; d < 3; ++d)
        if (idx[d] < 0 || idx[d] >= p.shape[d])
          outside = true;
      if (outside) {
        if (n.oob == static_cast<std::uint32_t>(oob_kind::fill)) {
          std::memcpy(out, n.fill, n.elem_bytes);
          return true;
        }
        for (int d = 0; d < 3; ++d) {
          idx[d] %= p.shape[d];
          if (idx[d] < 0)
            idx[d] += p.shape[d];
        }
      }
      for (int d = 0; d < 3; ++d)
        c[d] = idx[d];
      node = n.parent;
      coord = true;
      break;
    }
    default:
      return false;
    }
  }
  return false;
}

} // namespace sg::detail::dev

#endif
