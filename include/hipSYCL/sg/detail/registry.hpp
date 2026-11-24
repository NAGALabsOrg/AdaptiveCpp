// SPDX-License-Identifier: BSD-2-Clause
// Names the device task functions: each function passes the address of a task
// instance literally to enable_function_symbol_reflection, which the compiler
// pass FunctionNameExtractionPass turns into a symbol-name registration.
#ifndef HIPSYCL_SG_DETAIL_REGISTRY_HPP
#define HIPSYCL_SG_DETAIL_REGISTRY_HPP

#include "hipSYCL/glue/reflection.hpp"
#include "hipSYCL/sg/detail/dev/entries.hpp"
#include "hipSYCL/sg/error.hpp"

#include <string>

namespace sg::detail {

struct entry_ref {
  std::string symbol;
};

inline entry_ref finish_ref(const char *name) {
  if (!name || !*name)
    throw error{errc::internal, "a task function has no registered symbol name"};
  return entry_ref{name};
}

template <class Fn, class... A> entry_ref tile_task_ref() {
  hipsycl::glue::reflection::enable_function_symbol_reflection(&tile_task<Fn, A...>);
  return finish_ref(hipsycl::glue::reflection::resolve_function_name(&tile_task<Fn, A...>));
}
template <class Fn, class... A> entry_ref item_task_ref() {
  hipsycl::glue::reflection::enable_function_symbol_reflection(&item_task<Fn, A...>);
  return finish_ref(hipsycl::glue::reflection::resolve_function_name(&item_task<Fn, A...>));
}
template <class T, class Op> entry_ref sink_task_ref() {
  hipsycl::glue::reflection::enable_function_symbol_reflection(&sink_task<T, Op>);
  return finish_ref(hipsycl::glue::reflection::resolve_function_name(&sink_task<T, Op>));
}

} // namespace sg::detail

#endif
