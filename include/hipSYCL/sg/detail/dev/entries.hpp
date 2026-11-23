// SPDX-License-Identifier: BSD-2-Clause
// The device entries the registry names and composition clones: one template
// instance per user body, positions function, element type and combine op.
#ifndef HIPSYCL_SG_DETAIL_DEV_ENTRIES_HPP
#define HIPSYCL_SG_DETAIL_DEV_ENTRIES_HPP

#include "hipSYCL/sg/body.hpp"
#include "hipSYCL/sg/detail/dev/drivers.hpp"
#include "hipSYCL/sg/detail/dev/group_eval.hpp"
#include "hipSYCL/sg/detail/dev/layout.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>
#include <utility>

namespace sg::detail::dev {

template <int N> sycl::id<N> id_from(const std::uint64_t v[3]) {
  sycl::id<N> r;
  for (int d = 0; d < N; ++d)
    r[d] = v[d];
  return r;
}
template <int N> sycl::range<N> range_of(const std::uint64_t v[3]) {
  sycl::range<N> r;
  for (int d = 0; d < N; ++d)
    r[d] = v[d];
  return r;
}

} // namespace sg::detail::dev

namespace sg::detail {

template <class C, int N, class... Args>
SYCL_EXTERNAL [[clang::annotate("sg_body_entry")]] void
item_body_entry(const dev::launch_ctx *ctx, std::uint32_t node, std::uint64_t g,
                std::uint64_t l, void *frame, void *dst) {
  const dev::node_rec &n = dev::node_at(ctx, node);
  const C &closure = *reinterpret_cast<const C *>(n.closure);
  std::uint64_t gid[3], lid[3], id[3];
  dev::unflatten3(g, n.grid, gid);
  dev::unflatten3(l, n.group_shape, lid);
  for (int d = 0; d < 3; ++d)
    id[d] = gid[d] * n.group_shape[d] + lid[d];
  const sg::item<N> it(dev::id_from<N>(id), dev::id_from<N>(gid), dev::id_from<N>(lid),
                       dev::range_of<N>(n.range), dev::range_of<N>(n.group_shape));
  auto make = [&](std::size_t k) {
    return dev::iterator_state{ctx, node, static_cast<std::uint32_t>(k), g, l, frame};
  };
  auto v = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return closure(it, Args(make(I))...);
  }(std::index_sequence_for<Args...>{});
  dev::ph::write(ctx, node, dst, &v);
}

template <class C, int N, class... Args>
SYCL_EXTERNAL [[clang::annotate("sg_body_entry")]] void
group_body_entry(const dev::launch_ctx *ctx, std::uint32_t node, const void *nd,
                 std::uint64_t g, void *frame, const dev::tile_out *out) {
  const dev::node_rec &n = dev::node_at(ctx, node);
  const C &closure = *reinterpret_cast<const C *>(n.closure);
  std::uint64_t gid[3];
  dev::unflatten3(g, n.grid, gid);
  const sg::group<N> gr(dev::id_from<N>(gid), dev::range_of<N>(n.range));
  auto make = [&](std::size_t k) {
    return dev::iterator_state{ctx, node, static_cast<std::uint32_t>(k), g, 0, frame};
  };
  auto e = [&]<std::size_t... I>(std::index_sequence<I...>) {
    return closure(gr, Args(make(I))...);
  }(std::index_sequence_for<Args...>{});
  dev::evaluate_group_expr<dev::ph>(e, nd, ctx, node, frame, out);
}

template <class Pos, int N, int M>
SYCL_EXTERNAL [[clang::annotate("sg_body_entry")]] void
positions_entry(const dev::launch_ctx *ctx, std::uint32_t node, std::uint32_t link,
                std::uint64_t g, std::uint32_t slot, std::uint64_t *src) {
  const dev::node_rec &n = dev::node_at(ctx, node);
  const dev::link_rec &lr = dev::link_at(ctx, node, link);
  const Pos &closure = *reinterpret_cast<const Pos *>(lr.closure);
  std::uint64_t gid[3];
  dev::unflatten3(g, n.grid, gid);
  const auto form = closure(dev::id_from<N>(gid));
  const sycl::id<M> s = form[slot];
  std::uint64_t sid[3] = {0, 0, 0};
  for (int d = 0; d < M; ++d)
    sid[d] = s[d];
  *src = dev::flatten3(sid, lr.src_grid);
}

template <class T>
SYCL_EXTERNAL [[clang::annotate("sg_body_entry")]] void
store_entry(const dev::launch_ctx *, std::uint32_t, void *dst, const void *src) {
  std::memcpy(dst, src, sizeof(T));
}

namespace dev_atomic {

template <class Op, class T> struct is_plus : std::false_type {};
template <class T> struct is_plus<sycl::plus<>, T> : std::true_type {};
template <class T> struct is_plus<sycl::plus<T>, T> : std::true_type {};
template <class Op, class T> struct is_min : std::false_type {};
template <class T> struct is_min<sycl::minimum<>, T> : std::true_type {};
template <class T> struct is_min<sycl::minimum<T>, T> : std::true_type {};
template <class Op, class T> struct is_max : std::false_type {};
template <class T> struct is_max<sycl::maximum<>, T> : std::true_type {};
template <class T> struct is_max<sycl::maximum<T>, T> : std::true_type {};
template <class Op, class T> struct is_and : std::false_type {};
template <class T> struct is_and<sycl::bit_and<>, T> : std::true_type {};
template <class T> struct is_and<sycl::bit_and<T>, T> : std::true_type {};
template <class Op, class T> struct is_or : std::false_type {};
template <class T> struct is_or<sycl::bit_or<>, T> : std::true_type {};
template <class T> struct is_or<sycl::bit_or<T>, T> : std::true_type {};
template <class Op, class T> struct is_xor : std::false_type {};
template <class T> struct is_xor<sycl::bit_xor<>, T> : std::true_type {};
template <class T> struct is_xor<sycl::bit_xor<T>, T> : std::true_type {};

template <std::size_t S> struct uint_of;
template <> struct uint_of<1> { using type = std::uint8_t; };
template <> struct uint_of<2> { using type = std::uint16_t; };
template <> struct uint_of<4> { using type = std::uint32_t; };
template <> struct uint_of<8> { using type = std::uint64_t; };

template <class T>
using aref = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                              sycl::access::address_space::global_space>;

template <class T, class Op> void combine(T *p, T v) {
  constexpr bool wide = sizeof(T) == 4 || sizeof(T) == 8;
  constexpr bool integral = std::is_integral_v<T>;
  constexpr bool numeric = wide && (integral || std::is_floating_point_v<T>);
  if constexpr (numeric && is_plus<Op, T>::value) {
    aref<T>(*p).fetch_add(v);
  } else if constexpr (numeric && is_min<Op, T>::value) {
    aref<T>(*p).fetch_min(v);
  } else if constexpr (numeric && is_max<Op, T>::value) {
    aref<T>(*p).fetch_max(v);
  } else if constexpr (wide && integral && is_and<Op, T>::value) {
    aref<T>(*p).fetch_and(v);
  } else if constexpr (wide && integral && is_or<Op, T>::value) {
    aref<T>(*p).fetch_or(v);
  } else if constexpr (wide && integral && is_xor<Op, T>::value) {
    aref<T>(*p).fetch_xor(v);
  } else if constexpr (sizeof(T) < 4) {
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p);
    aref<std::uint32_t> ref(*reinterpret_cast<std::uint32_t *>(addr & ~std::uintptr_t{3}));
    const std::uint32_t shift = static_cast<std::uint32_t>(addr & 3u) * 8u;
    const std::uint32_t mask = ((std::uint32_t{1} << (sizeof(T) * 8)) - 1u) << shift;
    std::uint32_t old = ref.load(sycl::memory_order::relaxed);
    while (true) {
      using U = typename uint_of<sizeof(T)>::type;
      const U bits = static_cast<U>((old & mask) >> shift);
      T o;
      std::memcpy(&o, &bits, sizeof(T));
      const T nv = Op{}(o, v);
      U nb;
      std::memcpy(&nb, &nv, sizeof(T));
      const std::uint32_t nu = (old & ~mask) | (static_cast<std::uint32_t>(nb) << shift);
      if (ref.compare_exchange_weak(old, nu, sycl::memory_order::relaxed,
                                    sycl::memory_order::relaxed))
        break;
    }
  } else {
    using U = typename uint_of<sizeof(T)>::type;
    aref<U> ref(*reinterpret_cast<U *>(p));
    U old = ref.load(sycl::memory_order::relaxed);
    while (true) {
      T o;
      std::memcpy(&o, &old, sizeof(T));
      const T nv = Op{}(o, v);
      U nu;
      std::memcpy(&nu, &nv, sizeof(T));
      if (ref.compare_exchange_weak(old, nu, sycl::memory_order::relaxed,
                                    sycl::memory_order::relaxed))
        break;
    }
  }
}

} // namespace dev_atomic

template <class T, class Op>
SYCL_EXTERNAL [[clang::annotate("sg_body_entry")]] void
combine_entry(const dev::launch_ctx *, std::uint32_t, void *dst, const void *src) {
  static_assert(sizeof(T) <= 8);
  T v;
  std::memcpy(&v, src, sizeof(T));
  dev_atomic::combine<T, Op>(static_cast<T *>(dst), v);
}

} // namespace sg::detail

#endif
