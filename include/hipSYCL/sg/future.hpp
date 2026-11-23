// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_FUTURE_HPP
#define HIPSYCL_SG_FUTURE_HPP

#include "hipSYCL/sg/detail/scope.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace sg {

class graph;

namespace detail {

class promise_base {
public:
  virtual ~promise_base() = default;
  virtual bool is_set() const noexcept = 0;
  virtual std::uint64_t as_u64() const = 0;
};

template <class T> class promise_state final : public promise_base {
public:
  bool is_set() const noexcept override { return set_.load(std::memory_order_acquire); }
  std::uint64_t as_u64() const override {
    if constexpr (std::is_integral_v<T>)
      return static_cast<std::uint64_t>(value_);
    else
      return 0;
  }
  // false if already set
  bool set(const T &v) {
    bool expected = false;
    if (!claimed_.compare_exchange_strong(expected, true))
      return false;
    value_ = v;
    set_.store(true, std::memory_order_release);
    return true;
  }
  const T &value() const { return value_; }

private:
  T value_{};
  std::atomic<bool> claimed_{false};
  std::atomic<bool> set_{false};
};

} // namespace detail

template <class T> class launch_value {
  static_assert(std::is_trivially_copyable_v<T>);

public:
  launch_value(std::shared_ptr<detail::promise_state<T>> s, std::size_t read_index)
      : state_(std::move(s)), read_index_(read_index) {}
  operator T() const {
    if (!state_ || !state_->is_set())
      throw error{errc::future_unset, "a future was read before its promise was set"};
    return state_->value();
  }
  std::size_t read_index() const noexcept { return read_index_; }

private:
  std::shared_ptr<detail::promise_state<T>> state_;
  std::size_t read_index_;
};

template <class T> class graph_future {
public:
  graph_future() = default;
  launch_value<T> get() const {
    detail::node_record *n = detail::current_scope();
    if (!n)
      throw error{errc::not_in_submit_scope, "graph_future::get() needs an active submit scope"};
    n->futures_read.push_back(index_);
    return launch_value<T>(state_, n->futures_read.size() - 1);
  }

private:
  graph_future(std::shared_ptr<detail::promise_state<T>> s, std::size_t i)
      : state_(std::move(s)), index_(i) {}
  std::shared_ptr<detail::promise_state<T>> state_;
  std::size_t index_ = 0;
  template <class> friend class graph_promise;
};

template <class T> class graph_promise {
  static_assert(std::is_trivially_copyable_v<T>);

public:
  graph_promise() = default;
  graph_future<T> future() const { return graph_future<T>(state_, index_); }
  void set(const T &v) const {
    if (!state_->set(v))
      throw error{errc::builder_misuse, "a graph promise was set twice"};
  }
  std::size_t index() const noexcept { return index_; }
  std::shared_ptr<detail::promise_state<T>> state() const { return state_; }

private:
  graph_promise(std::shared_ptr<detail::promise_state<T>> s, std::size_t i)
      : state_(std::move(s)), index_(i) {}
  std::shared_ptr<detail::promise_state<T>> state_;
  std::size_t index_ = 0;
  friend class graph;
};

} // namespace sg

#endif
