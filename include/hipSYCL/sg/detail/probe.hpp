// SPDX-License-Identifier: BSD-2-Clause
// Probe mode: the host evaluates positions functions and group bodies once, for
// destination group 0, without reading data.
#ifndef HIPSYCL_SG_DETAIL_PROBE_HPP
#define HIPSYCL_SG_DETAIL_PROBE_HPP

#include "hipSYCL/sg/detail/type_id.hpp"
#include "hipSYCL/sg/error.hpp"
#include "hipSYCL/sg/detail/form_kind.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace sg::detail {

inline int &probe_depth() noexcept {
  static thread_local int d = 0;
  return d;
}
inline bool probe_mode() noexcept { return probe_depth() > 0; }
struct probe_scope {
  probe_scope() { ++probe_depth(); }
  ~probe_scope() { --probe_depth(); }
  probe_scope(const probe_scope &) = delete;
  probe_scope &operator=(const probe_scope &) = delete;
};

struct probed_form {
  form_kind kind;
  std::size_t count;
};

template <int N, class P> probed_form probe_positions(const P &fn) {
  probe_scope ps;
  auto form = fn(sycl::id<N>{});
  return {decltype(form)::kind, static_cast<std::size_t>(form.size())};
}

struct prepare_context {
  std::function<std::uint64_t(std::uint64_t)> resolve_dim;
  static prepare_context *&slot() noexcept {
    static thread_local prepare_context *c = nullptr;
    return c;
  }
  static prepare_context *current() noexcept { return slot(); }
};

struct prepare_context_scope {
  explicit prepare_context_scope(prepare_context *c) : prev_(prepare_context::slot()) {
    prepare_context::slot() = c;
  }
  ~prepare_context_scope() { prepare_context::slot() = prev_; }
  prepare_context *prev_;
};

struct probed_group_body {
  std::uint64_t value_type = 0;
  std::size_t elem_bytes = 0;
  sycl::range<3> out_shape{1, 1, 1};
  std::size_t scratch_bytes = 0;
};

} // namespace sg::detail

#endif
