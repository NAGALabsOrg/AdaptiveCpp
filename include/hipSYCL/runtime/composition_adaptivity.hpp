/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_COMPOSITION_ADAPTIVITY_HPP
#define HIPSYCL_COMPOSITION_ADAPTIVITY_HPP

#include "hipSYCL/glue/llvm-sscp/composition.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"

#include <cstddef>
#include <string>

namespace hipsycl {
namespace rt {

// The only configuration a composed kernel takes from the adaptivity engine:
// the composition itself, the single-kernel model and the (bucketed) local
// memory size. Nothing launch-specific is baked into the configuration.
inline void configure_composition_kernel(
    kernel_configuration &config, int param_index,
    glue::sscp::composition_config_kernel_property_t property,
    const std::string &kernel_name, std::size_t local_mem_size) {
  config.set_composition_config(param_index, property);
  config.append_base_configuration(kernel_base_config_parameter::single_kernel,
                                   kernel_name);
  config.set_build_option(kernel_build_option::known_local_mem_size,
                          local_mem_size);
}

} // namespace rt
} // namespace hipsycl

#endif
