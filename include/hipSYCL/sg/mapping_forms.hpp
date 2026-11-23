// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_MAPPING_FORMS_HPP
#define HIPSYCL_SG_MAPPING_FORMS_HPP

#include "hipSYCL/sg/detail/form_kind.hpp"
#include "hipSYCL/sg/roles.hpp"

#include <sycl/sycl.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace sg {

namespace detail {
template <int M> sycl::id<M> unflatten_range(std::size_t k, const sycl::range<M> &extent) {
  sycl::id<M> r;
  for (int d = M - 1; d >= 0; --d) {
    const std::size_t e = extent[d] == 0 ? 1 : extent[d];
    r[d] = k % e;
    k /= e;
  }
  return r;
}
template <int M, class I> sycl::id<M> to_id(const I &v) {
  if constexpr (std::is_integral_v<I>) {
    static_assert(M == 1, "an index_map over integers needs rank 1");
    return sycl::id<M>(static_cast<std::size_t>(v));
  } else {
    return sycl::id<M>(v);
  }
}
} // namespace detail

template <int M> struct single {
  static constexpr form_kind kind = form_kind::single;
  sycl::id<M> id;
  std::size_t size() const { return 1; }
  sycl::id<M> operator[](std::size_t) const { return id; }
};

template <int M> struct block {
  static constexpr form_kind kind = form_kind::block;
  sycl::id<M> first;
  sycl::range<M> extent;
  std::size_t size() const { return extent.size(); }
  sycl::id<M> operator[](std::size_t k) const {
    sycl::id<M> c = detail::unflatten_range<M>(k, extent);
    for (int d = 0; d < M; ++d)
      c[d] += first[d];
    return c;
  }
};

template <int M> struct strided {
  static constexpr form_kind kind = form_kind::strided;
  sycl::id<M> first;
  sycl::range<M> extent;
  sycl::range<M> stride;
  std::size_t size() const { return extent.size(); }
  sycl::id<M> operator[](std::size_t k) const {
    sycl::id<M> c = detail::unflatten_range<M>(k, extent);
    for (int d = 0; d < M; ++d)
      c[d] = first[d] + c[d] * stride[d];
    return c;
  }
};

template <int M, std::size_t K> struct stencil {
  static constexpr form_kind kind = form_kind::stencil;
  sycl::id<M> center;
  std::array<sycl::id<M>, K> offsets;
  std::size_t size() const { return K; }
  sycl::id<M> operator[](std::size_t k) const {
    sycl::id<M> r;
    for (int d = 0; d < M; ++d)
      r[d] = center[d] + offsets[k][d];
    return r;
  }
};

template <int M, std::size_t K> struct ids {
  static constexpr form_kind kind = form_kind::ids;
  std::array<sycl::id<M>, K> values;
  std::size_t size() const { return K; }
  sycl::id<M> operator[](std::size_t k) const { return values[k]; }
};

template <int M, class I> struct index_map {
  static constexpr form_kind kind = form_kind::index_map;
  array<I, 1> source;
  std::size_t offset, count;
  std::size_t size() const { return count; }
  sycl::id<M> operator[](std::size_t k) const {
    return detail::to_id<M, I>(source[sycl::id<1>(offset + k)]);
  }
};

} // namespace sg

#endif
