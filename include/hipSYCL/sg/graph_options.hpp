// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_GRAPH_OPTIONS_HPP
#define HIPSYCL_SG_GRAPH_OPTIONS_HPP

#include <cstddef>

namespace sg {

struct graph_options {
  // Bytes each device's memory pool takes at its first use. 0: the pool starts empty and grows with need.
  std::size_t pool_reservation_bytes = 0;
  // 0: the device's maximum work-group size. The largest native group of a kernel.
  std::size_t group_size_cap = 0;
};

} // namespace sg

#endif
