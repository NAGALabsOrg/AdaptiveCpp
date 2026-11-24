// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_CLOSURE_HPP
#define HIPSYCL_SG_DETAIL_CLOSURE_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace sg::detail {

// The bytes of a trivially copyable device function object.
struct closure_entry {
  std::vector<std::byte> bytes;
  std::size_t align = 8;
};

template <class C> closure_entry capture_closure(const C &c) {
  static_assert(std::is_trivially_copyable_v<C>,
                "sg: a device function and everything it captures must be trivially copyable");
  static_assert(alignof(C) <= 16);
  closure_entry e;
  e.bytes.resize(sizeof(C));
  if (sizeof(C) && !std::is_empty_v<C>)
    std::memcpy(e.bytes.data(), &c, sizeof(C));
  e.align = alignof(C);
  return e;
}

} // namespace sg::detail

#endif
