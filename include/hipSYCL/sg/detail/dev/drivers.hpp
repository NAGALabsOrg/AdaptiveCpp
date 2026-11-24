// SPDX-License-Identifier: BSD-2-Clause
// The generic device driver, written once over a dispatch policy P. Each native
// group claims a ticket with a 32-bit atomic when it starts, copies the tiles
// marked local, and runs the task list of its ticket in order: before each task
// one item waits on the done flags of the task's producers (wait_flag: relaxed
// polls, then one acquire fence at device scope once a flag is seen), then a group
// barrier and an acquire fence for the group's other items; after each task a
// barrier, a release fence and the task's flag.
// Waits only ever target lower tickets or earlier tasks of the same group.
#ifndef HIPSYCL_SG_DETAIL_DEV_DRIVERS_HPP
#define HIPSYCL_SG_DETAIL_DEV_DRIVERS_HPP

#include "hipSYCL/sg/detail/dev/backing.hpp"
#include "hipSYCL/sg/detail/dev/layout.hpp"
#include "hipSYCL/sg/detail/dev/placeholders.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace sg::detail::dev {

using flag_ref = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>;

template <class P> void run_native_group(block_t block, const sycl::nd_item<1> &it, void *local_mem) {
  const block_header &h = header_of(block);
  std::byte *local = static_cast<std::byte *>(local_mem);
  std::uint32_t *flags = flags_of(block);
  const std::uint32_t gid = static_cast<std::uint32_t>(it.get_local_id(0));
  const std::uint32_t group_size = static_cast<std::uint32_t>(it.get_local_range(0));

  std::uint32_t ticket = 0;
  if (gid == 0)
    ticket = flag_ref(flags[0]).fetch_add(1u);
  ticket = sycl::group_broadcast(it.get_group(), ticket);
  if (ticket >= h.n_tickets)
    return;

  const ticket_rec &t = tickets_of(block)[ticket];
  if (t.n_copies > 0) {
    device_visitor vis{flags};
    for (std::uint32_t i = 0; i < t.n_copies; ++i)
      copy_tile_to_local(block, copies_of(block)[t.first_copy + i], local, gid, group_size, vis);
    sycl::group_barrier(it.get_group());
  }

  for (std::uint32_t i = 0; i < t.n_tasks; ++i) {
    const std::uint32_t rec = t.first_task + i;
    const task_rec &r = tasks_of(block)[rec];
    if (r.n_waits > 0) {
      if (gid == 0)
        for (std::uint32_t w = 0; w < r.n_waits; ++w)
          wait_flag(flags, waits_of(block)[r.first_wait + w]);
      sycl::group_barrier(it.get_group());
      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
    }
    P::dispatch(block, rec, 0, &it, local);
    sycl::group_barrier(it.get_group());
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
    if (gid == 0)
      flag_ref(flags[r.flag]).store(1u, sycl::memory_order::release);
  }
}

} // namespace sg::detail::dev

#endif
