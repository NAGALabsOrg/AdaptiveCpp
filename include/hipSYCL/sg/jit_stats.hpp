// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_JIT_STATS_HPP
#define HIPSYCL_SG_JIT_STATS_HPP

#include "hipSYCL/runtime/kernel_cache.hpp"

#include <cstdint>

namespace sg {

// The number of stage-2 JIT compilations the process has performed so far.
inline std::uint64_t jit_compile_count() { return hipsycl::rt::jit_compilation_count(); }

} // namespace sg

#endif
