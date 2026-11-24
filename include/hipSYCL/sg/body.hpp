// SPDX-License-Identifier: BSD-2-Clause
// What a device function sees: the item of a run and the collectives over it.
#ifndef HIPSYCL_SG_BODY_HPP
#define HIPSYCL_SG_BODY_HPP

#include "hipSYCL/sg/detail/dev/collectives.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace sg {

// A value combined over the whole run of a unit; one value per unit.
template <class T> struct reduced {
  T value;
  explicit operator T() const { return value; }
  friend reduced operator+(reduced a, const T &b) { return {static_cast<T>(a.value + b)}; }
  friend reduced operator-(reduced a, const T &b) { return {static_cast<T>(a.value - b)}; }
  friend reduced operator*(reduced a, const T &b) { return {static_cast<T>(a.value * b)}; }
  friend reduced operator/(reduced a, const T &b) { return {static_cast<T>(a.value / b)}; }
};

// Counts per bin; lane k of the run holds the count of bin k.
template <class T> struct binned {
  T value;
};

template <class T> struct is_reduced : std::false_type {};
template <class T> struct is_reduced<reduced<T>> : std::true_type {};
template <class T> struct is_binned : std::false_type {};
template <class T> struct is_binned<binned<T>> : std::true_type {};

// The item of a unit's run: a native work-item, on the device; a dummy when the
// toolchain calls the function on the host to learn its shape.
class run_item {
public:
  run_item() = default;
  explicit run_item(detail::dev::run_ctx *ctx) : c(ctx) {}
  // position of the item in the unit's run
  std::int64_t id() const noexcept { return c->pos; }
  std::int64_t run_length() const noexcept { return c->run_len; }

  detail::dev::run_ctx *c = nullptr;
};

template <class T, class Op> reduced<T> reduce(const run_item &item, T v, Op op) {
  detail::dev::run_ctx &c = *item.c;
  if (c.sizing) {
    c.sizing->elem_bytes = c.sizing->elem_bytes > sizeof(T) ? c.sizing->elem_bytes : static_cast<std::uint32_t>(sizeof(T));
    c.sizing->partial_bytes += detail::dev::round16(sizeof(T));
    return {v};
  }
  const T r = detail::dev::local_reduce<T>(c, v, op);
  return {detail::dev::finish_reduce<T>(c, r, op)};
}

template <class T, class Op> T inclusive_scan(const run_item &item, T v, Op op) {
  detail::dev::run_ctx &c = *item.c;
  if (c.sizing) {
    c.sizing->elem_bytes = c.sizing->elem_bytes > sizeof(T) ? c.sizing->elem_bytes : static_cast<std::uint32_t>(sizeof(T));
    c.sizing->partial_bytes += detail::dev::round16(sizeof(T));
    return v;
  }
  T total = v;
  const T local = detail::dev::local_scan<T>(c, v, op, total);
  return detail::dev::finish_scan<T>(c, local, total, op);
}

template <class T, class Key> binned<std::int32_t> tile_bin(const run_item &item, T v, std::int64_t bins, Key key) {
  detail::dev::run_ctx &c = *item.c;
  if (c.sizing) {
    c.sizing->bins = c.sizing->bins > bins ? c.sizing->bins : bins;
    c.sizing->elem_bytes = c.sizing->elem_bytes > 4 ? c.sizing->elem_bytes : 4u;
    c.sizing->partial_bytes += detail::dev::round16(static_cast<std::uint32_t>(4 * bins));
    return {0};
  }
  const std::int32_t r = detail::dev::local_bin<T>(c, v, bins, key);
  return {detail::dev::finish_bin(c, r, bins)};
}

} // namespace sg

#endif
