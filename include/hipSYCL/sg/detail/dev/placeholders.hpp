// SPDX-License-Identifier: BSD-2-Clause
// The one placeholder of a composed kernel. At JIT time it becomes an indexed
// call sequence over the task functions of the kernel's composition: each task
// function gets its own index as a constant and returns at once unless that
// index is the task being run. The body here only traps so the host links.
#ifndef HIPSYCL_SG_DETAIL_DEV_PLACEHOLDERS_HPP
#define HIPSYCL_SG_DETAIL_DEV_PLACEHOLDERS_HPP

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

#define SG_DEV_FN                                                                                    \
  extern "C" SYCL_EXTERNAL [[gnu::used]] [[clang::noinline]] [[clang::annotate("acpp_no_s1_device_inline")]] inline

// block, record number, aux (item or zero), the nd_item or null, local or private memory
SG_DEV_FN void __sg_ph_dispatch(const std::byte *, std::uint32_t, std::uint64_t, const void *, void *) {
  __builtin_trap();
}

namespace sg::detail::dev {

// The production policy: the dispatch placeholder.
struct ph {
  static void dispatch(const std::byte *block, std::uint32_t rec, std::uint64_t aux, const void *nd,
                       void *mem) {
    __sg_ph_dispatch(block, rec, aux, nd, mem);
  }
};

} // namespace sg::detail::dev

#endif
