// SPDX-License-Identifier: BSD-2-Clause
// Roles on handles: a kernel's inputs (kh.source) and results (kh.evaluate), and
// the arrays host work reads and writes.
#ifndef HIPSYCL_SG_ROLES_HPP
#define HIPSYCL_SG_ROLES_HPP

#include "hipSYCL/sg/detail/array_rep.hpp"
#include "hipSYCL/sg/detail/registry.hpp"
#include "hipSYCL/sg/detail/scope.hpp"
#include "hipSYCL/sg/handle.hpp"
#include "hipSYCL/sg/launch_value.hpp"

#include <sycl/sycl.hpp>

#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>

namespace sg {

// One extent of an array: a number, or a launch value read just before launch.
class dim {
public:
  dim(std::size_t v) : e_(v) {}
  dim(const bound_expr &e) : e_(e) {}
  template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
  dim(const launch_value<I> &v) : e_(v) {}
  const bound_expr &expr() const noexcept { return e_; }

private:
  bound_expr e_;
};

template <int N> class shape {
public:
  shape(const sycl::range<N> &r) {
    for (int d = 0; d < N; ++d)
      dims_.v[d] = bound_expr(static_cast<std::int64_t>(r[d]));
    dims_.rank = N;
  }
  shape(std::initializer_list<dim> l) {
    if (static_cast<int>(l.size()) != N)
      throw error{errc::builder_misuse, "a shape of rank " + std::to_string(N) + " was given " +
                                            std::to_string(l.size()) + " extents"};
    int d = 0;
    for (const dim &x : l)
      dims_.v[d++] = x.expr();
    dims_.rank = N;
  }
  const bdims &extents() const noexcept { return dims_; }

private:
  bdims dims_;
};

namespace detail {

template <class T, class Op> struct inplace_spec {
  mem_handle<T> handle;
  std::string op;
  init_kind init;
  T init_value;
};

template <class T> std::vector<std::byte> bytes_of(const T &v) {
  std::vector<std::byte> b(sizeof(T));
  std::memcpy(b.data(), &v, sizeof(T));
  return b;
}

template <class T>
int register_slot(const mem_handle<T> &h, const bdims &ext, role_mode mode, bool is_host,
                  std::string op, init_kind init, std::vector<std::byte> bits) {
  node_record *n = current_scope();
  if (!n)
    throw error{errc::not_in_submit_scope, "an array role needs an active submit scope"};
  if (!h.valid() || handle_access::owner(h) != n->owner)
    throw error{errc::builder_misuse, "a role was built on a null handle or on a handle of another graph"};
  array_slot s;
  s.handle = handle_access::index(h);
  s.type = type_id<std::remove_const_t<T>>();
  s.elem_bytes = sizeof(T);
  s.mode = mode;
  s.is_host = is_host;
  s.rank = ext.rank;
  s.ext = ext;
  // A size bound per run may be smaller than what the handle holds, and a host read may take a prefix.
  bool sized_by_launch = false;
  for (int d = 0; d < ext.rank; ++d)
    ext.v[d].for_each_launch([&](std::int64_t) { sized_by_launch = true; });
  s.whole = !sized_by_launch && !(is_host && mode == role_mode::read);
  s.op = std::move(op);
  s.init = init;
  s.init_bits = std::move(bits);
  n->slots.push_back(std::move(s));
  return static_cast<int>(n->slots.size()) - 1;
}

inline std::size_t rowmajor(const std::uint64_t *idx, const std::uint64_t *ext, int n) {
  std::size_t r = 0;
  for (int d = 0; d < n; ++d)
    r = r * ext[d] + idx[d];
  return r;
}

} // namespace detail

template <class T, class Op> detail::inplace_spec<T, Op> inplace(const mem_handle<T> &h, Op, const T &init) {
  static_assert(sizeof(T) <= 8 && std::is_trivially_copyable_v<T> && std::is_invocable_r_v<T, Op, T, T> &&
                    std::is_default_constructible_v<Op>,
                "sg::inplace: elements are at most 8 bytes and Op must combine two T");
  return {h, std::string(detail::type_name<Op>()), detail::init_kind::value, init};
}
struct reuse_t {};
inline constexpr reuse_t reuse{};
template <class T, class Op> detail::inplace_spec<T, Op> inplace(const mem_handle<T> &h, Op, reuse_t) {
  static_assert(sizeof(T) <= 8 && std::is_trivially_copyable_v<T> && std::is_invocable_r_v<T, Op, T, T> &&
                    std::is_default_constructible_v<Op>,
                "sg::inplace: elements are at most 8 bytes and Op must combine two T");
  return {h, std::string(detail::type_name<Op>()), detail::init_kind::reuse, T{}};
}

// An array that host work reads or writes; its address is bound before the work runs.
template <class T, int N> class host_array {
  using elem_t = std::remove_const_t<T>;

public:
  host_array(const mem_handle<elem_t> &h, const shape<N> &sh) {
    const bool writer = !std::is_const_v<T>;
    slot_ = detail::register_slot<elem_t>(h, sh.extents(),
                                          writer ? detail::role_mode::produce : detail::role_mode::read, true,
                                          {}, detail::init_kind::none, {});
    detail::node_record *n = detail::current_scope();
    if (!n->host_table)
      n->host_table = std::make_shared<detail::host_slot_table>();
    table_ = n->host_table;
    if (table_->address.size() <= static_cast<std::size_t>(slot_)) {
      table_->address.resize(slot_ + 1, nullptr);
      table_->extent.resize(slot_ + 1);
    }
  }
  sycl::range<N> range() const {
    sycl::range<N> r;
    for (int d = 0; d < N; ++d)
      r[d] = table_->extent[slot_][d];
    return r;
  }
  T *data() const { return static_cast<T *>(table_->address[slot_]); }
  T &operator[](const sycl::id<N> &i) const {
    std::uint64_t idx[3] = {};
    std::uint64_t ext[3] = {};
    for (int d = 0; d < N; ++d) {
      idx[d] = i[d];
      ext[d] = table_->extent[slot_][d];
    }
    return data()[detail::rowmajor(idx, ext, N)];
  }

private:
  std::shared_ptr<detail::host_slot_table> table_;
  int slot_ = 0;
};

// What a submit<sg::host> builder sees: the arrays the host work reads and writes,
// and the work itself.
class host_handle {
public:
  explicit host_handle(detail::node_record &n) : n_(&n) {}

  template <class T, int N> host_array<T, N> write(const mem_handle<T> &h, const shape<N> &s) {
    return host_array<T, N>(h, s);
  }
  template <class T, int N> host_array<const T, N> read(const mem_handle<T> &h, const shape<N> &s) {
    return host_array<const T, N>(h, s);
  }
  template <class F> void run(F &&fn) { n_->host_fn = std::function<void()>(std::forward<F>(fn)); }

private:
  detail::node_record *n_;
};

} // namespace sg

#endif
