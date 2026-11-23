// SPDX-License-Identifier: BSD-2-Clause
// What a body sees: item and group handles, group_iterator, group_array.
#ifndef HIPSYCL_SG_BODY_HPP
#define HIPSYCL_SG_BODY_HPP

#include "hipSYCL/sg/detail/dev/iterators.hpp"
#include "hipSYCL/sg/kernel_kinds.hpp"

#include <sycl/sycl.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace sg {

template <int N> class item {
public:
  item(sycl::id<N> id, sycl::id<N> group_id, sycl::id<N> local_id, sycl::range<N> range,
       sycl::range<N> group_shape)
      : id_(id), group_id_(group_id), local_id_(local_id), range_(range), group_shape_(group_shape) {}
  sycl::id<N> id() const { return id_; }
  sycl::id<N> group_id() const { return group_id_; }
  sycl::id<N> local_id() const { return local_id_; }
  sycl::range<N> range() const { return range_; }
  sycl::range<N> group_shape() const { return group_shape_; }

private:
  sycl::id<N> id_, group_id_, local_id_;
  sycl::range<N> range_, group_shape_;
};

template <int N> class group {
public:
  group(sycl::id<N> id, sycl::range<N> group_range) : id_(id), group_range_(group_range) {}
  sycl::id<N> id() const { return id_; }
  sycl::range<N> group_range() const { return group_range_; }

private:
  sycl::id<N> id_;
  sycl::range<N> group_range_;
};

namespace detail {
template <int M> sycl::range<M> range_from(const std::uint64_t e[3]) {
  sycl::range<M> r;
  for (int d = 0; d < M; ++d)
    r[d] = e[d];
  return r;
}
} // namespace detail

// A read-only view of one source group's array. It is also a group expression
// (a leaf), see group_expr.hpp.
template <class T, int M, class Kind> class group_array {
public:
  using value_type = T;
  static constexpr int rank = M;
  static constexpr bool is_group_expr = true;

  explicit group_array(const detail::dev::view_rep &v) : v_(v) {}

  sycl::range<M> shape() const { return detail::range_from<M>(v_.extent); }
  std::size_t size() const { return v_.extent[0] * v_.extent[1] * v_.extent[2]; }
  T operator[](const sycl::id<M> &i) const {
    std::uint64_t off = 0;
    for (int d = 0; d < M; ++d)
      off += i[d] * v_.stride[d];
    return *reinterpret_cast<const T *>(v_.base + off * sizeof(T));
  }

  // expression interface
  std::array<std::uint64_t, 3> dims() const { return {v_.extent[0], v_.extent[1], v_.extent[2]}; }
  bool is_scalar() const { return false; }
  std::size_t scratch_bytes() const { return 0; }
  template <class Env> void stage(Env &, std::size_t) const {}
  template <class Env> T read(const Env &, std::size_t, std::size_t k) const {
    std::uint64_t id[3];
    detail::dev::unflatten3(k, v_.extent, id);
    const std::uint64_t off = id[0] * v_.stride[0] + id[1] * v_.stride[1] + id[2] * v_.stride[2];
    return *reinterpret_cast<const T *>(v_.base + off * sizeof(T));
  }

private:
  detail::dev::view_rep v_;
};

template <class T, int M, class Kind> class group_iterator {
public:
  static_assert(M >= 1 && M <= 3);
  using element_type =
      std::conditional_t<std::is_same_v<Kind, mapping::item>, T, group_array<T, M, Kind>>;

  explicit group_iterator(const detail::dev::iterator_state &s) : st_(s) {}
  std::size_t size() const { return detail::dev::iter_count(st_); }
  element_type operator[](std::size_t k) const {
    if constexpr (std::is_same_v<Kind, mapping::item>)
      return detail::dev::item_value_of<T>(st_, k);
    else if constexpr (std::is_same_v<Kind, mapping::local>)
      return group_array<T, M, Kind>(detail::dev::local_view(st_, k));
    else
      return group_array<T, M, Kind>(detail::dev::global_view(st_, k));
  }

private:
  detail::dev::iterator_state st_;
};

template <class T, int M, class Kind>
void wait(const group_iterator<T, M, Kind> &) {
  static_assert(!std::is_same_v<Kind, mapping::item>,
                "sg::wait applies to local and global arguments only");
}

template <class T, int M, class Kind, class Op>
T reduce(const group_array<T, M, Kind> &a, Op op) {
  T acc = sycl::known_identity_v<Op, T>;
  const sycl::range<M> r = a.shape();
  const std::size_t n = r.size();
  for (std::size_t k = 0; k < n; ++k)
    acc = op(acc, a.template read<int>(0, 0, k));
  return acc;
}

} // namespace sg

#endif
