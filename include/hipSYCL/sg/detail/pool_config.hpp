// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_POOL_CONFIG_HPP
#define HIPSYCL_SG_DETAIL_POOL_CONFIG_HPP

#include <cstddef>
#include <optional>

namespace sg::detail {

// requested if non-zero; else 80% of the free memory if reported; else 80% of total.
inline std::size_t pool_limit_rule(std::optional<std::size_t> free_bytes,
                                   std::size_t total_bytes, std::size_t requested) {
  if (requested != 0)
    return requested;
  const std::size_t base = free_bytes ? *free_bytes : total_bytes;
  return base / 5 * 4 + (base % 5) * 4 / 5;
}

} // namespace sg::detail

#endif
