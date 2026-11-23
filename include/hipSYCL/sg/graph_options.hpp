// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_GRAPH_OPTIONS_HPP
#define HIPSYCL_SG_GRAPH_OPTIONS_HPP

#include <cstddef>

namespace sg {
// 0: the pool reservation follows the limit rule, per device.
struct graph_options {
  std::size_t pool_reservation_bytes = 0;
};
} // namespace sg

#endif
