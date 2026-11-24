// SPDX-License-Identifier: BSD-2-Clause
// Launch values and bound expressions: numbers the host reads just before each
// launch. They are accepted where a size, count, grid extent, work-item range,
// repeat count or slice extent is taken, and never where structure is fixed.
#ifndef HIPSYCL_SG_LAUNCH_VALUE_HPP
#define HIPSYCL_SG_LAUNCH_VALUE_HPP

#include "hipSYCL/sg/error.hpp"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <vector>

namespace sg {

template <class T> class launch_value;

namespace detail {

enum class eop : std::uint8_t { cst, launch, count_of, add, sub, mul, div, mod };

struct expr_node;
using expr_ptr = std::shared_ptr<const expr_node>;
struct expr_node {
  eop op = eop::cst;
  std::int64_t value = 0; // constant, launch slot, or view id
  int dim = 0;
  expr_ptr a, b;
};

struct eval_env {
  const std::int64_t *launch = nullptr;
  std::size_t launch_count = 0;
  std::function<std::int64_t(std::int64_t view, int dim)> count_of;
};

template <class T> inline constexpr bool dependent_false = false;

} // namespace detail

class bound_expr {
public:
  bound_expr() = default;
  template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
  bound_expr(I v) : cst_(static_cast<std::int64_t>(v)) {}
  template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
  bound_expr(const launch_value<I> &v);

  bool constant() const noexcept { return !node_; }
  std::int64_t constant_value() const noexcept { return cst_; }
  const detail::expr_ptr &node() const noexcept { return node_; }

  std::int64_t evaluate(const detail::eval_env &env) const { return eval(*this, env); }

  static bound_expr launch_slot(std::int64_t slot) {
    return from_node({detail::eop::launch, slot, 0, nullptr, nullptr});
  }
  static bound_expr count_of(std::int64_t view, int dim) {
    return from_node({detail::eop::count_of, view, dim, nullptr, nullptr});
  }
  static bound_expr binary(detail::eop op, const bound_expr &a, const bound_expr &b) {
    if (a.constant() && b.constant())
      return bound_expr(fold(op, a.cst_, b.cst_));
    detail::expr_node n;
    n.op = op;
    n.a = a.as_node();
    n.b = b.as_node();
    return from_node(std::move(n));
  }

  // Rebuilds the expression with every count_of leaf renamed.
  bound_expr remap(const std::function<std::int64_t(std::int64_t)> &view) const {
    if (!node_)
      return *this;
    return bound_expr(remap_node(node_, view));
  }
  // Calls f for every count_of leaf.
  void for_each_view(const std::function<void(std::int64_t)> &f) const { visit(node_, f); }
  bool uses_launch() const { return node_ && uses_launch_node(node_); }
  // Calls f for every launch slot.
  void for_each_launch(const std::function<void(std::int64_t)> &f) const { visit_launch(node_, f); }

  friend bound_expr operator+(const bound_expr &a, const bound_expr &b) { return binary(detail::eop::add, a, b); }
  friend bound_expr operator-(const bound_expr &a, const bound_expr &b) { return binary(detail::eop::sub, a, b); }
  friend bound_expr operator*(const bound_expr &a, const bound_expr &b) { return binary(detail::eop::mul, a, b); }
  friend bound_expr operator/(const bound_expr &a, const bound_expr &b) { return binary(detail::eop::div, a, b); }
  friend bound_expr operator%(const bound_expr &a, const bound_expr &b) { return binary(detail::eop::mod, a, b); }

private:
  explicit bound_expr(detail::expr_ptr n) : node_(std::move(n)) {}
  static bound_expr from_node(detail::expr_node n) {
    return bound_expr(std::make_shared<const detail::expr_node>(std::move(n)));
  }
  detail::expr_ptr as_node() const {
    if (node_)
      return node_;
    auto n = std::make_shared<detail::expr_node>();
    n->op = detail::eop::cst;
    n->value = cst_;
    return n;
  }
  static std::int64_t fold(detail::eop op, std::int64_t a, std::int64_t b) {
    switch (op) {
    case detail::eop::add: return a + b;
    case detail::eop::sub: return a - b;
    case detail::eop::mul: return a * b;
    case detail::eop::div: return b == 0 ? 0 : a / b;
    case detail::eop::mod: return b == 0 ? 0 : a % b;
    default: return 0;
    }
  }
  static std::int64_t eval_node(const detail::expr_node &n, const detail::eval_env &env) {
    switch (n.op) {
    case detail::eop::cst: return n.value;
    case detail::eop::launch:
      if (static_cast<std::size_t>(n.value) >= env.launch_count)
        throw error{errc::internal, "a launch value has no value bound"};
      return env.launch[n.value];
    case detail::eop::count_of:
      if (!env.count_of)
        throw error{errc::internal, "a view count is not available here"};
      return env.count_of(n.value, n.dim);
    default:
      return fold(n.op, eval_node(*n.a, env), eval_node(*n.b, env));
    }
  }
  static std::int64_t eval(const bound_expr &e, const detail::eval_env &env) {
    return e.node_ ? eval_node(*e.node_, env) : e.cst_;
  }
  static detail::expr_ptr remap_node(const detail::expr_ptr &n,
                                     const std::function<std::int64_t(std::int64_t)> &view) {
    auto c = std::make_shared<detail::expr_node>(*n);
    if (n->op == detail::eop::count_of)
      c->value = view(n->value);
    if (n->a)
      c->a = remap_node(n->a, view);
    if (n->b)
      c->b = remap_node(n->b, view);
    return c;
  }
  static void visit(const detail::expr_ptr &n, const std::function<void(std::int64_t)> &f) {
    if (!n)
      return;
    if (n->op == detail::eop::count_of)
      f(n->value);
    visit(n->a, f);
    visit(n->b, f);
  }
  static void visit_launch(const detail::expr_ptr &n, const std::function<void(std::int64_t)> &f) {
    if (!n)
      return;
    if (n->op == detail::eop::launch)
      f(n->value);
    visit_launch(n->a, f);
    visit_launch(n->b, f);
  }
  static bool uses_launch_node(const detail::expr_ptr &n) {
    if (!n)
      return false;
    return n->op == detail::eop::launch || uses_launch_node(n->a) || uses_launch_node(n->b);
  }

  std::int64_t cst_ = 0;
  detail::expr_ptr node_;
};

// A number the host reads from a handle just before each launch. It never
// converts to T and never shapes structure.
template <class T> class launch_value {
  static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "sg::launch_value<T>: T must be arithmetic");

public:
  using value_type = T;
  launch_value() = default;
  launch_value(std::size_t handle, std::int64_t slot) : handle_(handle), slot_(slot) {}
  std::size_t handle() const noexcept { return handle_; }
  std::int64_t slot() const noexcept { return slot_; }

  template <class I = T, std::enable_if_t<std::is_integral_v<I>, int> = 0>
  bound_expr expr() const { return bound_expr::launch_slot(slot_); }

private:
  std::size_t handle_ = 0;
  std::int64_t slot_ = 0;
};

template <class J, std::enable_if_t<std::is_integral_v<J>, int>>
bound_expr::bound_expr(const launch_value<J> &v) : node_(bound_expr::launch_slot(v.slot()).node_) {}

template <class T> struct is_bound : std::false_type {};
template <> struct is_bound<bound_expr> : std::true_type {};
template <class T> struct is_bound<launch_value<T>> : std::true_type {};
template <class T> inline constexpr bool is_bound_v = is_bound<std::remove_cv_t<std::remove_reference_t<T>>>::value;

#define SG_LV_OPS(OP)                                                                                           \
  template <class I, class J>                                                                                  \
  std::enable_if_t<std::is_integral_v<I> && std::is_integral_v<J>, bound_expr> operator OP(                     \
      const launch_value<I> &a, const J &b) {                                                                   \
    return bound_expr(a) OP bound_expr(b);                                                                      \
  }                                                                                                             \
  template <class I, class J>                                                                                  \
  std::enable_if_t<std::is_integral_v<I> && std::is_integral_v<J>, bound_expr> operator OP(                     \
      const J &a, const launch_value<I> &b) {                                                                   \
    return bound_expr(a) OP bound_expr(b);                                                                      \
  }                                                                                                             \
  template <class I>                                                                                           \
  std::enable_if_t<std::is_integral_v<I>, bound_expr> operator OP(const launch_value<I> &a, const bound_expr &b) { \
    return bound_expr(a) OP b;                                                                                  \
  }                                                                                                             \
  template <class I>                                                                                           \
  std::enable_if_t<std::is_integral_v<I>, bound_expr> operator OP(const bound_expr &a, const launch_value<I> &b) { \
    return a OP bound_expr(b);                                                                                  \
  }                                                                                                             \
  template <class I, class J>                                                                                  \
  std::enable_if_t<std::is_integral_v<I> && std::is_integral_v<J>, bound_expr> operator OP(                     \
      const launch_value<I> &a, const launch_value<J> &b) {                                                     \
    return bound_expr(a) OP bound_expr(b);                                                                      \
  }
SG_LV_OPS(+)
SG_LV_OPS(-)
SG_LV_OPS(*)
SG_LV_OPS(/)
SG_LV_OPS(%)
#undef SG_LV_OPS

namespace detail {

// A structural integer: tile extents, run lengths. Launch values and bound
// expressions are refused here.
struct sint {
  std::int64_t v = 0;
  template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
  sint(I x) : v(static_cast<std::int64_t>(x)) {}
  template <class B, std::enable_if_t<is_bound_v<B>, int> = 0>
  sint(const B &) {
    static_assert(dependent_false<B>,
                  "a launch_value<T> is a bound value and cannot be used as structure: tile shape extents, run "
                  "lengths, loop bounds and array extents of the builder are fixed when the builder runs");
  }
};

} // namespace detail

// Fixed extents, at most three.
struct dims {
  std::int64_t v[3] = {1, 1, 1};
  int rank = 0;
  dims() = default;
  dims(std::initializer_list<detail::sint> l) {
    if (l.size() > 3)
      throw error{errc::builder_misuse, "a shape of more than 3 extents"};
    for (const auto &x : l)
      v[rank++] = x.v;
  }
  dims(detail::sint x) : rank(1) { v[0] = x.v; }
  std::int64_t operator[](int d) const { return v[d]; }
  std::int64_t product() const {
    std::int64_t p = 1;
    for (int d = 0; d < rank; ++d)
      p *= v[d];
    return p;
  }
};

// Extents that may be bound expressions.
struct bdims {
  bound_expr v[3] = {1, 1, 1};
  int rank = 0;
  bdims() = default;
  bdims(std::initializer_list<bound_expr> l) {
    if (l.size() > 3)
      throw error{errc::builder_misuse, "a grid of more than 3 extents"};
    for (const auto &x : l)
      v[rank++] = x;
  }
  bdims(const bound_expr &x) : rank(1) { v[0] = x; }
  template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
  bdims(const launch_value<I> &x) : rank(1) { v[0] = bound_expr(x); }
};

} // namespace sg

#endif
