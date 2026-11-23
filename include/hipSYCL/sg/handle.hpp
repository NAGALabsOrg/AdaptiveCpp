// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_HANDLE_HPP
#define HIPSYCL_SG_HANDLE_HPP

#include "hipSYCL/sg/detail/scope.hpp"

#include <cstddef>
#include <type_traits>

namespace sg {

class graph;
template <class T> class mem_handle;

namespace detail {
struct handle_access {
  template <class H> static graph_impl *owner(const H &h) { return h.owner_; }
  template <class H> static std::size_t index(const H &h) { return h.index_; }
  template <class H> static H make(graph_impl *o, std::size_t i) {
    H h;
    h.owner_ = o;
    h.index_ = i;
    return h;
  }
};
} // namespace detail

template <class T> class handle_slice {
public:
  handle_slice(const mem_handle<T> &h, std::size_t offset, std::size_t count)
      : handle_(h), offset_(offset), count_(count) {}
  const mem_handle<T> &handle() const noexcept { return handle_; }
  std::size_t offset() const noexcept { return offset_; }
  std::size_t count() const noexcept { return count_; }

private:
  mem_handle<T> handle_;
  std::size_t offset_, count_;
};

template <class T> class mem_handle {
  static_assert(std::is_trivially_copyable_v<T>, "sg::mem_handle: T must be trivially copyable");

public:
  using value_type = T;
  mem_handle() = default;
  bool valid() const noexcept { return owner_ != nullptr; }
  handle_slice<T> slice(std::size_t offset, std::size_t count) const {
    return handle_slice<T>(*this, offset, count);
  }

private:
  detail::graph_impl *owner_ = nullptr;
  std::size_t index_ = 0;
  friend class graph;
  friend struct detail::handle_access;
};

} // namespace sg

#endif
