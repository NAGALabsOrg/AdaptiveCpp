// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_TYPE_ID_HPP
#define HIPSYCL_SG_DETAIL_TYPE_ID_HPP

#include <cstdint>
#include <string_view>

namespace sg::detail {

constexpr std::uint64_t fnv1a64(std::string_view s) noexcept {
  std::uint64_t h = 0xcbf29ce484222325ull;
  for (char c : s) {
    h ^= static_cast<unsigned char>(c);
    h *= 0x100000001b3ull;
  }
  return h;
}

template <class T> std::string_view type_name() noexcept {
  std::string_view p = __PRETTY_FUNCTION__;
  const std::string_view key = "T = ";
  auto b = p.find(key);
  if (b == std::string_view::npos)
    return p;
  b += key.size();
  auto e = p.rfind(']');
  return p.substr(b, e - b);
}

template <class T> std::uint64_t type_id() noexcept {
  return fnv1a64(type_name<T>());
}

} // namespace sg::detail

#endif
