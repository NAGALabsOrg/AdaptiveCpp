// SPDX-License-Identifier: BSD-2-Clause
// Work-item kernels: a plain range launch, no tickets and no flags. Each item
// runs its tasks in order and keeps every intermediate in registers; results go
// only to the item's own index.
#ifndef HIPSYCL_SG_DETAIL_DEV_RANGE_KERNEL_HPP
#define HIPSYCL_SG_DETAIL_DEV_RANGE_KERNEL_HPP

#include "hipSYCL/sg/detail/dev/layout.hpp"
#include "hipSYCL/sg/detail/dev/placeholders.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>

namespace sg::detail {

namespace dev {

template <class T>
inline T read_item_input(block_t block, const item_task_rec &r, int j, std::uint64_t linear,
                         const std::byte *regs) {
  T v{};
  const item_input_rec &in = r.in[j];
  if (in.kind == 2) {
    std::memcpy(&v, regs + in.reg, sizeof(T));
    return v;
  }
  const block_header &h = header_of(block);
  const view_node &n = views_of(block)[in.view];
  std::int64_t c[3];
  std::int64_t ext[3] = {static_cast<std::int64_t>(h.items_ext[0]), static_cast<std::int64_t>(h.items_ext[1]),
                         static_cast<std::int64_t>(h.items_ext[2])};
  unflatten3(static_cast<std::int64_t>(linear), ext, c);
  for (int d = 0; d < 3; ++d) {
    c[d] += in.off[d];
    if (c[d] < 0 || c[d] >= n.shape[d])
      return v;
  }
  std::memcpy(&v, reinterpret_cast<const std::byte *>(n.base) + flatten3(c, n.shape) * sizeof(T), sizeof(T));
  return v;
}

template <class P> inline void run_items(block_t block, std::uint64_t linear) {
  const block_header &h = header_of(block);
  if (linear >= h.n_items)
    return;
  alignas(16) std::byte regs[max_item_regs];
  for (std::uint32_t t = 0; t < h.n_item_tasks; ++t)
    P::dispatch(block, t, linear, nullptr, regs);
}

} // namespace dev

template <class Fn, class... A>
SYCL_EXTERNAL [[clang::annotate("sg_body_entry")]] void item_task(std::uint32_t self, const std::byte *block,
                                                                    std::uint32_t recno, std::uint64_t linear,
                                                                    const void *, void *regs_) {
  using R = std::invoke_result_t<const Fn &, A...>;
  const dev::item_task_rec &r = dev::item_tasks_of(block)[recno];
  if (r.task_id != self)
    return;
  std::byte *regs = static_cast<std::byte *>(regs_);
  const Fn &fn = *reinterpret_cast<const Fn *>(block + r.closure);
  const auto vals = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return std::tuple<A...>{dev::read_item_input<A>(block, r, static_cast<int>(I), linear, regs)...};
  }(std::index_sequence_for<A...>{});
  const R res = std::apply(fn, vals);
  std::memcpy(regs + r.out_reg, &res, sizeof(R));
  if (r.eval)
    std::memcpy(reinterpret_cast<std::byte *>(r.eval_addr) + linear * sizeof(R), &res, sizeof(R));
}

} // namespace sg::detail

#endif
