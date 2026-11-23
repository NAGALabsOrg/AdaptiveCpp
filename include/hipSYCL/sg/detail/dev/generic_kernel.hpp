// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_DEV_GENERIC_KERNEL_HPP
#define HIPSYCL_SG_DETAIL_DEV_GENERIC_KERNEL_HPP

#include "hipSYCL/glue/llvm-sscp/composition.hpp"
#include "hipSYCL/sg/detail/dev/drivers.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace sg::detail {

// The one kernel every composed launch runs. `composition` carries the
// description to the JIT through a parameter annotation; the kernel itself
// never reads it.
struct generic_kernel {
  dev::launch_ctx ctx;
  hipsycl::glue::sscp::__acpp_sscp_emit_param_type_annotation_composition_config<std::intptr_t>
      composition;
  sycl::local_accessor<std::byte, 1> frame;

  void operator()(sycl::nd_item<1> it) const {
    dev::run_native_group<dev::ph>(&ctx, it, static_cast<void *>(&frame[0]));
  }
};

} // namespace sg::detail

#endif
