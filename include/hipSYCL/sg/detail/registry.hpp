// SPDX-License-Identifier: BSD-2-Clause
// Names the device entries: each function passes the address of an entry
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
    throw error{errc::internal, "a body entry has no registered symbol name"};
  return entry_ref{name};
}

template <class C, int N, class... Args> entry_ref item_body_entry_ref() {
  hipsycl::glue::reflection::enable_function_symbol_reflection(&item_body_entry<C, N, Args...>);
  return finish_ref(
      hipsycl::glue::reflection::resolve_function_name(&item_body_entry<C, N, Args...>));
}
template <class C, int N, class... Args> entry_ref group_body_entry_ref() {
  hipsycl::glue::reflection::enable_function_symbol_reflection(&group_body_entry<C, N, Args...>);
  return finish_ref(
      hipsycl::glue::reflection::resolve_function_name(&group_body_entry<C, N, Args...>));
}
template <class P, int N, int M> entry_ref positions_entry_ref() {
  hipsycl::glue::reflection::enable_function_symbol_reflection(&positions_entry<P, N, M>);
  return finish_ref(hipsycl::glue::reflection::resolve_function_name(&positions_entry<P, N, M>));
}
template <class T> entry_ref store_entry_ref() {
  hipsycl::glue::reflection::enable_function_symbol_reflection(&store_entry<T>);
  return finish_ref(hipsycl::glue::reflection::resolve_function_name(&store_entry<T>));
}
template <class T, class Op> entry_ref combine_entry_ref() {
  hipsycl::glue::reflection::enable_function_symbol_reflection(&combine_entry<T, Op>);
  return finish_ref(hipsycl::glue::reflection::resolve_function_name(&combine_entry<T, Op>));
}

} // namespace sg::detail

#endif
