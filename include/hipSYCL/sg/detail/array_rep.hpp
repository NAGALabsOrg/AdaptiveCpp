// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_ARRAY_REP_HPP
#define HIPSYCL_SG_DETAIL_ARRAY_REP_HPP

#include <cstdint>

namespace sg::detail {

// An array as one launch sees it: where it lives and how large it is.
struct array_rep {
  std::uint64_t address = 0;
  std::int64_t extent[3] = {1, 1, 1};
  std::uint32_t rank = 1;
  std::uint32_t elem_bytes = 0;

  std::int64_t count() const noexcept { return extent[0] * extent[1] * extent[2]; }
};

} // namespace sg::detail

#endif
