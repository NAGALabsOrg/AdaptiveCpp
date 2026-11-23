// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_ROLES_HPP
#define HIPSYCL_SG_ROLES_HPP

#include "hipSYCL/sg/detail/array_rep.hpp"
#include "hipSYCL/sg/detail/probe.hpp"
#include "hipSYCL/sg/detail/registry.hpp"
#include "hipSYCL/sg/detail/scope.hpp"
#include "hipSYCL/sg/future.hpp"
#include "hipSYCL/sg/handle.hpp"

#include <sycl/sycl.hpp>

#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>

namespace sg {

template <class T> struct result {};
struct reuse_t {};
inline constexpr reuse_t reuse{};

class dim {
public:
  static constexpr std::uint64_t deferred_flag = 1ull << 63;
  dim(std::size_t v) : enc_(v) {}
  template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
  dim(const launch_value<I> &v) : enc_(deferred_flag | v.read_index()) {}
  bool deferred() const noexcept { return (enc_ & deferred_flag) != 0; }
  std::uint64_t encoded() const noexcept { return enc_; }

private:
  std::uint64_t enc_;
};

template <int N> class shape {
public:
  shape(const sycl::range<N> &r) {
    for (int d = 0; d < N; ++d)
      dims_[d] = r[d];
  }
  shape(std::initializer_list<dim> l) {
    if (static_cast<int>(l.size()) != N)
      throw error{errc::builder_misuse,
                  "a shape of rank " + std::to_string(N) + " was given " +
                      std::to_string(l.size()) + " extents"};
    int d = 0;
    for (const dim &x : l)
      dims_[d++] = x.encoded();
  }
  std::uint64_t encoded(int d) const { return dims_[d]; }

private:
  std::uint64_t dims_[N] = {};
};

namespace detail {

template <class T, class Op> struct inplace_spec {
  mem_handle<T> handle;
  std::string op;
  init_kind init;
  T init_value;
};

inline std::uint64_t resolve_extent(std::uint64_t e) {
  if (!(e & dim::deferred_flag))
    return e;
  prepare_context *c = prepare_context::current();
  if (!c || !c->resolve_dim)
    throw error{errc::builder_misuse,
                "a deferred extent can only be read while a kernel is prepared"};
  return c->resolve_dim(e);
}

template <class T> T sentinel_read() {
  if (probe_mode())
    return T{};
  throw error{errc::builder_misuse,
              "an array built in a submit scope cannot be read outside a body"};
}

template <class T, int N>
std::uint32_t register_slot(const mem_handle<T> &h, const shape<N> &sh, role_mode mode,
                            bool is_host, bool whole, std::size_t offset, std::size_t count,
                            std::string op, init_kind init, std::vector<std::byte> bits) {
  node_record *n = current_scope();
  if (!n)
    throw error{errc::not_in_submit_scope, "an array role needs an active submit scope"};
  if (!h.valid() || handle_access::owner(h) != n->owner)
    throw error{errc::builder_misuse,
                "a role was built on a null handle or on a handle of another graph"};
  array_slot s;
  s.handle = handle_access::index(h);
  s.type = type_id<std::remove_const_t<T>>();
  s.elem_bytes = sizeof(T);
  s.mode = mode;
  s.is_host = is_host;
  s.whole = whole;
  s.offset = offset;
  s.rank = N;
  bool any_deferred = false;
  std::uint64_t prod = 1;
  for (int d = 0; d < 3; ++d) {
    s.extent[d] = d < N ? sh.encoded(d) : 1;
    if (s.extent[d] & dim::deferred_flag)
      any_deferred = true;
    else
      prod *= s.extent[d];
  }
  if (!whole && !any_deferred && count != prod)
    throw error{errc::size_mismatch,
                "a slice of " + std::to_string(count) + " elements was given a shape of " +
                    std::to_string(prod)};
  s.op = std::move(op);
  s.init = init;
  s.init_bits = std::move(bits);
  n->slots.push_back(std::move(s));
  return static_cast<std::uint32_t>(n->slots.size() - 1);
}

template <class T> std::vector<std::byte> bytes_of(const T &v) {
  std::vector<std::byte> b(sizeof(T));
  std::memcpy(b.data(), &v, sizeof(T));
  return b;
}

inline std::size_t rowmajor(const std::uint64_t *idx, const std::uint64_t *ext, int n) {
  std::size_t r = 0;
  for (int d = 0; d < n; ++d)
    r = r * ext[d] + idx[d];
  return r;
}

inline void record_sink(std::uint32_t slot, int definition_index); // defined in tree_store.hpp

} // namespace detail

template <class T, int N> class evaluation;

template <class T, int N> class array {
  static_assert(std::is_trivially_copyable_v<T>, "sg::array: T must be trivially copyable");
  static_assert(N >= 1 && N <= 3);

public:
  using value_type = T;
  array(const mem_handle<T> &h, const shape<N> &sh) {
    const auto slot = detail::register_slot<T, N>(h, sh, detail::role_mode::read, false, true,
                                                  0, 0, {}, detail::init_kind::none, {});
    init(slot, sh);
  }
  array(const handle_slice<T> &s, const shape<N> &sh) {
    const auto slot = detail::register_slot<T, N>(s.handle(), sh, detail::role_mode::read,
                                                  false, false, s.offset(), s.count(), {},
                                                  detail::init_kind::none, {});
    init(slot, sh);
  }
  sycl::range<N> range() const {
    sycl::range<N> r;
    for (int d = 0; d < N; ++d)
      r[d] = detail::resolve_extent(rep_.extent[d]);
    return r;
  }
  std::size_t size() const { return range().size(); }
  const T *data_ptr() const { return reinterpret_cast<const T *>(rep_.address); }
  T operator[](const sycl::id<N> &i) const {
    __acpp_if_target_host(
        if (detail::is_sentinel(rep_.address)) { return detail::sentinel_read<T>(); })
    std::uint64_t idx[3] = {};
    for (int d = 0; d < N; ++d)
      idx[d] = i[d];
    const T *p = reinterpret_cast<const T *>(rep_.address) +
                 detail::rowmajor(idx, rep_.extent, N);
    return *p;
  }

private:
  void init(std::uint32_t slot, const shape<N> &sh) {
    rep_.address = detail::sentinel(N, slot);
    for (int d = 0; d < 3; ++d)
      rep_.extent[d] = d < N ? sh.encoded(d) : 1;
  }
  detail::array_rep rep_;
};

template <class T, int N> class array<result<T>, N> {
  static_assert(std::is_trivially_copyable_v<T>);

public:
  array(const mem_handle<T> &h, const shape<N> &sh) {
    init(detail::register_slot<T, N>(h, sh, detail::role_mode::produce, false, true, 0, 0, {},
                                     detail::init_kind::none, {}),
         sh);
  }
  template <class Op>
  array(const detail::inplace_spec<T, Op> &spec, const shape<N> &sh) {
    const auto slot = detail::register_slot<T, N>(
        spec.handle, sh, detail::role_mode::inplace, false, true, 0, 0, spec.op, spec.init,
        spec.init == detail::init_kind::value ? detail::bytes_of(spec.init_value)
                                              : std::vector<std::byte>{});
    detail::current_scope()->slots[slot].op_symbol =
        detail::combine_entry_ref<T, Op>().symbol;
    init(slot, sh);
  }
  void operator=(const evaluation<T, N> &e) const {
    detail::record_sink(detail::sentinel_slot(rep_.address), e.definition_index());
  }
  sycl::range<N> range() const {
    sycl::range<N> r;
    for (int d = 0; d < N; ++d)
      r[d] = detail::resolve_extent(rep_.extent[d]);
    return r;
  }
  std::size_t size() const { return range().size(); }

private:
  void init(std::uint32_t slot, const shape<N> &sh) {
    rep_.address = detail::sentinel(N, slot);
    for (int d = 0; d < 3; ++d)
      rep_.extent[d] = d < N ? sh.encoded(d) : 1;
  }
  detail::array_rep rep_;
};

template <class T, class Op>
detail::inplace_spec<T, Op> inplace(const mem_handle<T> &h, Op, const T &init) {
  static_assert(sizeof(T) <= 8 && std::is_trivially_copyable_v<T> &&
                    std::is_invocable_r_v<T, Op, T, T> && std::is_default_constructible_v<Op>,
                "sg::inplace: elements are at most 8 bytes and Op must combine two T");
  return {h, std::string(detail::type_name<Op>()), detail::init_kind::value, init};
}
template <class T, class Op>
detail::inplace_spec<T, Op> inplace(const mem_handle<T> &h, Op, reuse_t) {
  static_assert(sizeof(T) <= 8 && std::is_trivially_copyable_v<T> &&
                    std::is_invocable_r_v<T, Op, T, T>,
                "sg::inplace: elements are at most 8 bytes and Op must combine two T");
  return {h, std::string(detail::type_name<Op>()), detail::init_kind::reuse, T{}};
}

template <class T, int N> class host_array {
  using elem_t = std::remove_const_t<T>;

public:
  host_array(const mem_handle<elem_t> &h, const shape<N> &sh) {
    const bool writer = !std::is_const_v<T>;
    slot_ = detail::register_slot<elem_t, N>(
        h, sh, writer ? detail::role_mode::produce : detail::role_mode::read, true, true, 0, 0,
        {}, detail::init_kind::none, {});
    detail::node_record *n = detail::current_scope();
    if (!n->host_table)
      n->host_table = std::make_shared<detail::host_slot_table>();
    table_ = n->host_table;
    if (table_->address.size() <= slot_) {
      table_->address.resize(slot_ + 1, nullptr);
      table_->extent.resize(slot_ + 1);
    }
    for (int d = 0; d < 3; ++d)
      extent_[d] = d < N ? sh.encoded(d) : 1;
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
  std::uint32_t slot_ = 0;
  std::uint64_t extent_[3] = {1, 1, 1};
};

} // namespace sg

#endif
