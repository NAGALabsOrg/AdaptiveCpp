// SPDX-License-Identifier: BSD-2-Clause
// In-place results: contributors run in parallel and combine each element
// atomically with the named operation. Elements narrower than 32 bits are
// combined by compare-exchange on the naturally aligned 32-bit word that holds
// them, never with sub-word atomics.
#ifndef HIPSYCL_SG_DETAIL_DEV_INPLACE_HPP
#define HIPSYCL_SG_DETAIL_DEV_INPLACE_HPP

#include "hipSYCL/sg/detail/dev/layout.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace sg::detail::dev_atomic {

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
      if (ref.compare_exchange_weak(old, nu, sycl::memory_order::relaxed, sycl::memory_order::relaxed))
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
      if (ref.compare_exchange_weak(old, nu, sycl::memory_order::relaxed, sycl::memory_order::relaxed))
        break;
    }
  }
}

} // namespace sg::detail::dev_atomic

namespace sg::detail {

// The task that combines an item's register into an in-place result element.
template <class T, class Op>
SYCL_EXTERNAL [[clang::annotate("sg_task")]] void sink_task(std::uint32_t self, const std::byte *block,
                                                              std::uint32_t rec, std::uint64_t linear,
                                                              const void *, void *regs) {
  static_assert(sizeof(T) <= 8);
  const dev::item_task_rec &r = dev::item_tasks_of(block)[rec];
  if (r.task_id != self)
    return;
  T v;
  std::memcpy(&v, static_cast<const std::byte *>(regs) + r.in[0].reg, sizeof(T));
  T *dst = reinterpret_cast<T *>(r.eval_addr) + linear;
  dev_atomic::combine<T, Op>(dst, v);
}

} // namespace sg::detail

#endif
