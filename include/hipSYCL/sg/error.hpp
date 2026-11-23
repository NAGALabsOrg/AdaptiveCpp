// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_ERROR_HPP
#define HIPSYCL_SG_ERROR_HPP

#include <stdexcept>
#include <string>

namespace sg {

enum class errc {
  role_conflict,
  size_mismatch,
  type_mismatch,
  never_produced,
  inplace_group_mismatch,
  builder_misuse,
  shape_mismatch,
  mapping_mismatch,
  pool_exhausted,
  local_memory_exceeded,
  work_group_too_large,
  limit_exceeded,
  not_in_submit_scope,
  future_unset,
  internal
};

inline const char *to_string(errc c) noexcept {
  switch (c) {
  case errc::role_conflict: return "role_conflict";
  case errc::size_mismatch: return "size_mismatch";
  case errc::type_mismatch: return "type_mismatch";
  case errc::never_produced: return "never_produced";
  case errc::inplace_group_mismatch: return "inplace_group_mismatch";
  case errc::builder_misuse: return "builder_misuse";
  case errc::shape_mismatch: return "shape_mismatch";
  case errc::mapping_mismatch: return "mapping_mismatch";
  case errc::pool_exhausted: return "pool_exhausted";
  case errc::local_memory_exceeded: return "local_memory_exceeded";
  case errc::work_group_too_large: return "work_group_too_large";
  case errc::limit_exceeded: return "limit_exceeded";
  case errc::not_in_submit_scope: return "not_in_submit_scope";
  case errc::future_unset: return "future_unset";
  case errc::internal: return "internal";
  }
  return "internal";
}

class error : public std::runtime_error {
public:
  error(errc code, const std::string &message)
      : std::runtime_error(message), code_{code} {}
  errc code() const noexcept { return code_; }

private:
  errc code_;
};

} // namespace sg

#endif
