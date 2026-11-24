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
#ifndef ACPP_GLUE_COMPOSITION_HPP
#define ACPP_GLUE_COMPOSITION_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hipsycl::glue::sscp {

template <class T>
struct __acpp_sscp_emit_param_type_annotation_composition_config {
  T value;
};

// Plain data shared by the host library and the JIT compiler: the ordered task
// functions of one composed kernel. Standard library only.
struct composition_config {
  std::uint64_t unique_hash = 0;
  std::vector<std::string> tasks;

  // Task symbols in index order, each once.
  std::vector<std::string> imported_symbols() const {
    std::vector<std::string> out;
    for (const std::string &s : tasks)
      if (!s.empty() && std::find(out.begin(), out.end(), s) == out.end())
        out.push_back(s);
    return out;
  }
};

using composition_config_kernel_property_t =
    __acpp_sscp_emit_param_type_annotation_composition_config<const composition_config *>;

inline std::intptr_t hide_composition_pointer(const composition_config *p) noexcept {
  return reinterpret_cast<std::intptr_t>(p);
}
inline const composition_config *unhide_composition_pointer(std::intptr_t v) noexcept {
  return reinterpret_cast<const composition_config *>(v);
}

} // namespace hipsycl::glue::sscp

#endif
