// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_SCOPE_HPP
#define HIPSYCL_SG_DETAIL_SCOPE_HPP

#include "hipSYCL/sg/detail/array_rep.hpp"
#include "hipSYCL/sg/detail/type_id.hpp"
#include "hipSYCL/sg/error.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace sg {
class kernel_handle;
}

namespace sg::detail {

enum class role_mode : std::uint32_t { read, produce, inplace };
enum class init_kind : std::uint32_t { none, value, reuse };

struct array_slot {
  std::size_t handle = 0;
  std::uint64_t type = 0;
  std::size_t elem_bytes = 0;
  role_mode mode = role_mode::read;
  bool is_host = false;
  bool whole = true;
  std::size_t offset = 0;
  int rank = 1;
  std::uint64_t extent[3] = {1, 1, 1}; // possibly deferred-encoded
  std::string op;
  std::string op_symbol; // the registered combine entry, inplace only
  init_kind init = init_kind::none;
  std::vector<std::byte> init_bits;
};

class graph_impl;

struct host_slot_table {
  std::vector<void *> address;
  std::vector<std::array<std::uint64_t, 3>> extent;
};

class key_holder {
public:
  virtual ~key_holder() = default;
  virtual bool equals(const key_holder &) const = 0;
  virtual std::uint64_t type() const noexcept = 0;
};

template <class... K> class key_holder_impl final : public key_holder {
public:
  explicit key_holder_impl(std::tuple<K...> t) : value_(std::move(t)) {}
  bool equals(const key_holder &o) const override {
    if (o.type() != type())
      return false;
    return value_ == static_cast<const key_holder_impl &>(o).value_;
  }
  std::uint64_t type() const noexcept override {
    return type_id<key_holder_impl<K...>>();
  }

private:
  std::tuple<K...> value_;
};

template <class... K> std::unique_ptr<key_holder> make_key(std::tuple<K...> t) {
  return std::make_unique<key_holder_impl<K...>>(std::move(t));
}

struct node_record {
  enum class kind_t { kernel, host_work } kind = kind_t::kernel;
  int device = 0;
  graph_impl *owner = nullptr;
  std::vector<array_slot> slots;
  std::vector<std::size_t> futures_read; // promise indices, in get() order
  std::function<void(kernel_handle &)> builder;
  std::uint64_t builder_type = 0;
  std::unique_ptr<key_holder> key;
  std::function<void()> host_fn;
  std::shared_ptr<host_slot_table> host_table;
  int defines = 0;
};

inline node_record *&scope_slot() noexcept {
  static thread_local node_record *s = nullptr;
  return s;
}
inline node_record *current_scope() noexcept { return scope_slot(); }
inline void set_current_scope(node_record *n) noexcept { scope_slot() = n; }

class graph_impl;
struct access; // friend marker for handles
struct access_key {};

} // namespace sg::detail

#endif
