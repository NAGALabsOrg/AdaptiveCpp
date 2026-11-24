// SPDX-License-Identifier: BSD-2-Clause
// Carries out the backing plan: marked array tiles are copied into local memory
// by the group's own items, and a result element goes to every place the plan
// gave its value: global memory for values that cross work tiles, local memory
// for values only the group reads, result memory when evaluated.
#ifndef HIPSYCL_SG_DETAIL_DEV_BACKING_HPP
#define HIPSYCL_SG_DETAIL_DEV_BACKING_HPP

#include "hipSYCL/sg/detail/dev/layout.hpp"
#include "hipSYCL/sg/detail/dev/view_read.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sg::detail::dev {

// Polls with a relaxed load and issues one acquire fence at device scope once the
// flag is seen. An acquire-ordered load inside the poll makes the Intel GPU
// compiler emit an atomic followed by a tile-scope fence on every poll, which with
// many groups polling stalls the launch until the kernel driver resets it; one
// fence after the flag is seen gives the same ordering.
inline void wait_flag(std::uint32_t *flags, std::uint32_t index) {
  sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      f(flags[index]);
  while (f.load(sycl::memory_order::relaxed) == 0) {
  }
  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
}

// Waits on the flags of a tile chosen by kernel data, then reads.
struct device_visitor {
  std::uint32_t *flags;
  bool reads_data() const { return true; }
  bool reached(const view_node &n, std::int64_t tile, std::int64_t, bool chosen_by_data) {
    if (n.wait_flags && chosen_by_data)
      for (std::uint32_t s = 0; s < n.segs; ++s)
        wait_flag(flags, n.flag_base + static_cast<std::uint32_t>(tile) * n.segs + s);
    return true;
  }
  void data_chosen(const view_node *, int) {}
};

// Every item of the group copies its share of a marked tile into local memory.
inline void copy_tile_to_local(block_t block, const copy_rec &c, std::byte *local, std::uint32_t gid,
                               std::uint32_t group_size, device_visitor &w) {
  const view_node *nodes = views_of(block);
  std::byte buf[16];
  for (std::uint64_t e = gid; e < c.elems; e += group_size)
    if (read_view(nodes, c.view, c.tile, static_cast<std::int64_t>(e), local, w, buf))
      std::memcpy(local + c.local_off + e * c.elem_bytes, buf, c.elem_bytes);
}

// The address in result memory of element `linear` of a map's results taken in
// tile order, or null when the evaluated array is shorter (a trimmed padding).
inline std::byte *eval_address(const task_rec &r, std::int64_t linear) {
  const std::int64_t size = r.eval_shape[0] * r.eval_shape[1] * r.eval_shape[2];
  std::int64_t g[3], l[3], c[3];
  unflatten3(linear / size, r.eval_grid, g);
  unflatten3(linear % size, r.eval_shape, l);
  for (int d = 0; d < 3; ++d) {
    c[d] = g[d] * r.eval_shape[d] + l[d];
    if (c[d] >= r.eval_extent[d])
      return nullptr;
  }
  return reinterpret_cast<std::byte *>(r.eval_addr) + flatten3(c, r.eval_extent) * r.out_bytes;
}

inline void store_result(const task_rec &r, std::byte *local, std::int64_t idx, const void *value) {
  if (r.out_flags & out_global)
    std::memcpy(reinterpret_cast<std::byte *>(r.out_global) +
                    (r.tile_idx * r.out_tile_elems + idx) * r.out_bytes,
                value, r.out_bytes);
  if (r.out_flags & out_local)
    std::memcpy(local + r.local_out_off + idx * r.out_bytes, value, r.out_bytes);
  if (r.out_flags & out_eval) {
    std::byte *p = eval_address(r, r.tile_idx * r.out_tile_elems + idx);
    if (p)
      std::memcpy(p, value, r.out_bytes);
  }
}

} // namespace sg::detail::dev

#endif
