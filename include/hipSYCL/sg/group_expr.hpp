// SPDX-License-Identifier: BSD-2-Clause
// The lazy group-body vocabulary: reductions, scans and element-wise arithmetic
// over group arrays. Evaluation happens in group_eval.hpp.
#ifndef HIPSYCL_SG_GROUP_EXPR_HPP
#define HIPSYCL_SG_GROUP_EXPR_HPP

#include "hipSYCL/sg/body.hpp"
#include "hipSYCL/sg/error.hpp"

#include <sycl/sycl.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace sg {

namespace detail {

using dims3 = std::array<std::uint64_t, 3>;

constexpr std::size_t align16(std::size_t v) { return (v + 15) / 16 * 16; }
inline std::uint64_t prod3(const dims3 &d) { return d[0] * d[1] * d[2]; }

template <class E, class = void> struct is_expr : std::false_type {};
template <class E> struct is_expr<E, std::void_t<decltype(E::is_group_expr)>>
    : std::bool_constant<E::is_group_expr> {};
template <class E> inline constexpr bool is_expr_v = is_expr<std::decay_t<E>>::value;

template <int R> using shape_t = sycl::range<(R > 0 ? R : 1)>;

template <int R> shape_t<R> shape_from(const dims3 &d) {
  shape_t<R> r;
  for (int i = 0; i < (R > 0 ? R : 1); ++i)
    r[i] = d[i];
  return r;
}

[[noreturn]] inline void throw_shape(const char *what) {
  throw error{errc::shape_mismatch, what};
}

// Output dimensions of removing dimension d of a padded shape.
inline dims3 remove_dim(const dims3 &s, int d) {
  dims3 o = {1, 1, 1};
  int c = 0;
  for (int p = 0; p < 3; ++p)
    if (p != d)
      o[c++] = s[p];
  return o;
}
// Input index of output coordinate c with dimension d set to j.
inline void insert_dim(const std::uint64_t c[3], int d, std::uint64_t j, std::uint64_t idx[3]) {
  int ci = 0;
  for (int p = 0; p < 3; ++p)
    idx[p] = (p == d) ? j : c[ci++];
}

} // namespace detail

template <class T> struct scalar_expr {
  using value_type = T;
  static constexpr int rank = 0;
  static constexpr bool is_group_expr = true;
  T v;
  detail::dims3 dims() const { return {1, 1, 1}; }
  bool is_scalar() const { return true; }
  std::size_t scratch_bytes() const { return 0; }
  template <class Env> void stage(Env &, std::size_t) const {}
  template <class Env> T read(const Env &, std::size_t, std::size_t) const { return v; }
};

namespace detail {
template <class E> inline constexpr bool is_scalar_leaf_v =
    std::is_same_v<std::decay_t<E>, scalar_expr<typename std::decay_t<E>::value_type>>;
} // namespace detail

template <int R, class X, class Op> struct reduce_expr {
  static_assert(X::rank == R, "sg::group_reduce<R>: R must equal the rank of its input");
  using value_type = typename X::value_type;
  static constexpr int rank = R > 1 ? R - 1 : 0;
  static constexpr bool is_group_expr = true;
  X x;
  int d;
  Op op;

  detail::dims3 dims() const {
    const detail::dims3 in = x.dims();
    __acpp_if_target_host(if (d < 0 || d >= R) detail::throw_shape(
        "group_reduce: the reduced dimension is outside the rank of its input");)
    return detail::remove_dim(in, d);
  }
  detail::shape_t<rank> shape() const { return detail::shape_from<rank>(dims()); }
  bool is_scalar() const { return false; }
  std::size_t scratch_bytes() const {
    return x.scratch_bytes() + detail::align16(detail::prod3(dims()) * sizeof(value_type));
  }
  template <class Env> void stage(Env &env, std::size_t base) const {
    x.stage(env, base);
    const detail::dims3 in = x.dims();
    const detail::dims3 od = detail::remove_dim(in, d);
    value_type *out = reinterpret_cast<value_type *>(env.scratch + base + x.scratch_bytes());
    const std::uint32_t nout = env.uniform(detail::prod3(od));
    for (std::uint32_t o = 0; o < nout; ++o) {
      std::uint64_t c[3], idx[3];
      detail::dev::unflatten3(o, od.data(), c);
      value_type partial = sycl::known_identity_v<Op, value_type>;
      for (std::uint64_t j = env.lid; j < in[d]; j += env.W) {
        detail::insert_dim(c, d, j, idx);
        partial = op(partial, x.read(env, base, detail::dev::flatten3(idx, in.data())));
      }
      const value_type r = sycl::reduce_over_group(env.group(), partial, op);
      if (env.lid == 0)
        out[o] = r;
    }
    env.sync();
  }
  template <class Env> value_type read(const Env &env, std::size_t base, std::size_t k) const {
    return reinterpret_cast<const value_type *>(env.scratch + base + x.scratch_bytes())[k];
  }
};

template <int R, class X, class Op, bool Exclusive> struct scan_expr {
  static_assert(X::rank == R, "sg::group_*_scan<R>: R must equal the rank of its input");
  using value_type = typename X::value_type;
  static constexpr int rank = R;
  static constexpr bool is_group_expr = true;
  X x;
  int d;
  Op op;
  value_type init;

  detail::dims3 dims() const {
    __acpp_if_target_host(if (d < 0 || d >= R) detail::throw_shape(
        "group scan: the scanned dimension is outside the rank of its input");)
    return x.dims();
  }
  detail::shape_t<rank> shape() const { return detail::shape_from<rank>(dims()); }
  bool is_scalar() const { return false; }
  std::size_t scratch_bytes() const {
    const detail::dims3 s = dims();
    return x.scratch_bytes() + detail::align16(detail::prod3(s) * sizeof(value_type)) +
           2 * detail::align16(s[d] * sizeof(value_type));
  }
  template <class Env> void stage(Env &env, std::size_t base) const {
    x.stage(env, base);
    const detail::dims3 s = x.dims();
    char *region = env.scratch + base + x.scratch_bytes();
    value_type *out = reinterpret_cast<value_type *>(region);
    value_type *bin = reinterpret_cast<value_type *>(
        region + detail::align16(detail::prod3(s) * sizeof(value_type)));
    value_type *bout = reinterpret_cast<value_type *>(
        reinterpret_cast<char *>(bin) + detail::align16(s[d] * sizeof(value_type)));
    const std::uint64_t n = s[d];
    const detail::dims3 od = detail::remove_dim(s, d);
    const std::uint32_t nlines = env.uniform(detail::prod3(od));
    for (std::uint32_t line = 0; line < nlines; ++line) {
      std::uint64_t c[3], idx[3];
      detail::dev::unflatten3(line, od.data(), c);
      for (std::uint64_t j = env.lid; j < n; j += env.W) {
        detail::insert_dim(c, d, j, idx);
        bin[j] = x.read(env, base, detail::dev::flatten3(idx, s.data()));
      }
      env.sync();
      if constexpr (Exclusive)
        sycl::joint_exclusive_scan(env.group(), bin, bin + n, bout, init, op);
      else
        sycl::joint_inclusive_scan(env.group(), bin, bin + n, bout, op);
      env.sync();
      for (std::uint64_t j = env.lid; j < n; j += env.W) {
        detail::insert_dim(c, d, j, idx);
        out[detail::dev::flatten3(idx, s.data())] = bout[j];
      }
      env.sync();
    }
  }
  template <class Env> value_type read(const Env &env, std::size_t base, std::size_t k) const {
    return reinterpret_cast<const value_type *>(env.scratch + base + x.scratch_bytes())[k];
  }
};

template <class Tag, class A, class B> struct binary_expr {
  using value_type =
      std::conditional_t<detail::is_scalar_leaf_v<A>, typename B::value_type,
                         typename A::value_type>;
  static constexpr int rank = detail::is_scalar_leaf_v<A> ? B::rank : A::rank;
  static constexpr bool is_group_expr = true;
  A a;
  B b;

  detail::dims3 dims() const {
    if constexpr (detail::is_scalar_leaf_v<A>) {
      return b.dims();
    } else if constexpr (detail::is_scalar_leaf_v<B>) {
      return a.dims();
    } else {
      const detail::dims3 da = a.dims(), db = b.dims();
      __acpp_if_target_host(if (da != db) detail::throw_shape(
          "element-wise operands have different shapes");)
      return da;
    }
  }
  detail::shape_t<rank> shape() const { return detail::shape_from<rank>(dims()); }
  bool is_scalar() const { return false; }
  std::size_t scratch_bytes() const { return a.scratch_bytes() + b.scratch_bytes(); }
  template <class Env> void stage(Env &env, std::size_t base) const {
    a.stage(env, base);
    b.stage(env, base + a.scratch_bytes());
  }
  template <class Env> value_type read(const Env &env, std::size_t base, std::size_t k) const {
    return Tag::apply(a.read(env, base, k), b.read(env, base + a.scratch_bytes(), k));
  }
};

namespace detail {
struct op_add { template <class T> static T apply(T x, T y) { return x + y; } };
struct op_sub { template <class T> static T apply(T x, T y) { return x - y; } };
struct op_mul { template <class T> static T apply(T x, T y) { return x * y; } };
struct op_div { template <class T> static T apply(T x, T y) { return x / y; } };

template <class Like, class X> auto lift(const X &x) {
  if constexpr (is_expr_v<X>)
    return x;
  else
    return scalar_expr<typename Like::value_type>{static_cast<typename Like::value_type>(x)};
}
template <class Tag, class A, class B> auto make_binary(const A &a, const B &b) {
  if constexpr (is_expr_v<A>) {
    auto lb = lift<A>(b);
    return binary_expr<Tag, A, decltype(lb)>{a, lb};
  } else {
    auto la = lift<B>(a);
    return binary_expr<Tag, decltype(la), B>{la, b};
  }
}
template <class A, class B> inline constexpr bool ew_ok_v =
    (is_expr_v<A> && (is_expr_v<B> || std::is_arithmetic_v<B>)) ||
    (is_expr_v<B> && std::is_arithmetic_v<A>);
} // namespace detail

template <class A, class B, std::enable_if_t<detail::ew_ok_v<A, B>, int> = 0>
auto operator+(const A &a, const B &b) { return detail::make_binary<detail::op_add>(a, b); }
template <class A, class B, std::enable_if_t<detail::ew_ok_v<A, B>, int> = 0>
auto operator-(const A &a, const B &b) { return detail::make_binary<detail::op_sub>(a, b); }
template <class A, class B, std::enable_if_t<detail::ew_ok_v<A, B>, int> = 0>
auto operator*(const A &a, const B &b) { return detail::make_binary<detail::op_mul>(a, b); }
template <class A, class B, std::enable_if_t<detail::ew_ok_v<A, B>, int> = 0>
auto operator/(const A &a, const B &b) { return detail::make_binary<detail::op_div>(a, b); }

template <int R, class X, class Op> auto group_reduce(const X &x, int dim, Op op) {
  static_assert(sycl::has_known_identity_v<Op, typename X::value_type>,
                "sg::group_reduce: the operation needs a known identity");
  return reduce_expr<R, X, Op>{x, dim, op};
}
template <int R, class X, class Op> auto group_inclusive_scan(const X &x, int dim, Op op) {
  static_assert(sycl::has_known_identity_v<Op, typename X::value_type>,
                "sg::group_inclusive_scan: the operation needs a known identity");
  return scan_expr<R, X, Op, false>{x, dim, op, typename X::value_type{}};
}
template <int R, class X, class Op>
auto group_exclusive_scan(const X &x, int dim, typename X::value_type init, Op op) {
  static_assert(sycl::has_known_identity_v<Op, typename X::value_type>,
                "sg::group_exclusive_scan: the operation needs a known identity");
  return scan_expr<R, X, Op, true>{x, dim, op, init};
}
template <class X, class Op> auto group_inclusive_scan(const X &x, Op op) {
  return group_inclusive_scan<1>(x, 0, op);
}
template <class X, class Op>
auto group_exclusive_scan(const X &x, typename X::value_type init, Op op) {
  return group_exclusive_scan<1>(x, 0, init, op);
}

} // namespace sg

#endif
