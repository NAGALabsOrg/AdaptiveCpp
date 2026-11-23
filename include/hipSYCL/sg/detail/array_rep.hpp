// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_ARRAY_REP_HPP
#define HIPSYCL_SG_DETAIL_ARRAY_REP_HPP

#include <cstdint>

namespace sg::detail {

struct array_rep {
  std::uint64_t address;
  std::uint64_t extent[3];
};
static_assert(sizeof(array_rep) == 32);

constexpr std::uint64_t sentinel_tag = 0x53474152ull << 32;

constexpr std::uint64_t sentinel(int rank, std::uint32_t slot) noexcept {
  return sentinel_tag | (std::uint64_t(rank) << 24) | std::uint64_t(slot);
}
constexpr bool is_sentinel(std::uint64_t v) noexcept {
  return (v >> 32) == 0x53474152ull;
}
constexpr int sentinel_rank(std::uint64_t v) noexcept {
  return static_cast<int>((v >> 24) & 0xff);
}
constexpr std::uint32_t sentinel_slot(std::uint64_t v) noexcept {
  return static_cast<std::uint32_t>(v & 0xffffff);
}

} // namespace sg::detail

#endif
