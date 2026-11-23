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

// Plain data shared by the host library and the JIT compiler: the structure of
// one composed kernel. Standard library only.
struct composition_link {
  std::uint32_t kind = 0;   // 0 item, 1 local, 2 global
  std::uint32_t source = 0; // node number (item, local) or layer position (global)
  std::string positions_symbol;
};

struct composition_node {
  std::string body_symbol, write_symbol;
  std::uint32_t body_kind = 0; // 0 item body, 1 group body
  std::vector<composition_link> links;
};

struct composition_layer {
  std::uint32_t root_node = 0;
};

struct composition_config {
  std::uint64_t unique_hash = 0;
  std::vector<composition_node> nodes;
  std::vector<composition_layer> layers;

  // Body, write and positions symbols; deduplicated, first use first.
  std::vector<std::string> imported_symbols() const {
    std::vector<std::string> out;
    auto add = [&](const std::string &s) {
      if (!s.empty() && std::find(out.begin(), out.end(), s) == out.end())
        out.push_back(s);
    };
    for (const auto &n : nodes) {
      add(n.body_symbol);
      add(n.write_symbol);
      for (const auto &l : n.links)
        add(l.positions_symbol);
    }
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

inline const std::vector<std::string> &composition_driver_symbols() {
  static const std::vector<std::string> names = {
      "__sg_drv_owner",        "__sg_drv_compute",   "__sg_drv_resolve_layer",
      "__sg_drv_resolve_tree", "__sg_drv_prep",      "__sg_drv_run_group"};
  return names;
}

} // namespace hipsycl::glue::sscp

#endif
