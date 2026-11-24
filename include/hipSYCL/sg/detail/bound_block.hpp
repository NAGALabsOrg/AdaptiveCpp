// SPDX-License-Identifier: BSD-2-Clause
// The bound block: one buffer, 256-byte aligned, uploaded per launch and passed
// as the only data argument of the generic kernel.
#ifndef HIPSYCL_SG_DETAIL_BOUND_BLOCK_HPP
#define HIPSYCL_SG_DETAIL_BOUND_BLOCK_HPP

#include "hipSYCL/sg/detail/device_ops.hpp"
#include "hipSYCL/sg/detail/launch_plan.hpp"

#include <cstdint>

namespace sg::detail {

inline kernel_launch_desc make_launch_desc(const launch_plan &p, std::uintptr_t block_address,
                                           const hipsycl::glue::sscp::composition_config *config) {
  if (block_address % 256 != 0)
    throw error{errc::internal, "the bound block is not 256-byte aligned"};
  kernel_launch_desc d;
  d.groups = p.groups;
  d.work_group_size = p.group_size;
  d.local_bytes = p.local_bytes;
  d.data_args = {block_address};
  d.items = p.items;
  d.n_items = p.n_items;
  d.config = config;
  return d;
}

} // namespace sg::detail

#endif
