// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_DEV_GENERIC_KERNEL_HPP
#define HIPSYCL_SG_DETAIL_DEV_GENERIC_KERNEL_HPP

#include "hipSYCL/glue/llvm-sscp/composition.hpp"
#include "hipSYCL/sg/detail/dev/drivers.hpp"
#include "hipSYCL/sg/detail/dev/range_kernel.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace sg::detail {

// The kernel every composed launch of work tiles runs. The bound block is its
// one data argument; `composition` carries the task list to the JIT through a
// parameter annotation and the kernel never reads it.
struct generic_kernel {
  std::uintptr_t block;
  hipsycl::glue::sscp::__acpp_sscp_emit_param_type_annotation_composition_config<std::intptr_t> composition;
  sycl::local_accessor<std::byte, 1> local;

  void operator()(sycl::nd_item<1> it) const {
    dev::run_native_group<dev::ph>(reinterpret_cast<const std::byte *>(block), it,
                                   static_cast<void *>(&local[0]));
  }
};

// The kernel of a work-item launch: a plain range, no groups.
struct item_kernel {
  std::uintptr_t block;
  hipsycl::glue::sscp::__acpp_sscp_emit_param_type_annotation_composition_config<std::intptr_t> composition;

  void operator()(sycl::nd_item<1> it) const {
    dev::run_items<dev::ph>(reinterpret_cast<const std::byte *>(block),
                            static_cast<std::uint64_t>(it.get_global_id(0)));
  }
};

} // namespace sg::detail

#endif
