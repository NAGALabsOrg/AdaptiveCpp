// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_GRAPH_HPP
#define HIPSYCL_SG_GRAPH_HPP

#include "hipSYCL/sg/detail/graph_impl.hpp"
#include "hipSYCL/sg/detail/pool_config.hpp"
#include "hipSYCL/sg/detail/sycl_device_ops.hpp"
#include "hipSYCL/sg/future.hpp"
#include "hipSYCL/sg/graph_options.hpp"
#include "hipSYCL/sg/handle.hpp"
#include "hipSYCL/sg/kernel.hpp"
#include "hipSYCL/sg/roles.hpp"

#include <sycl/sycl.hpp>

#include <cstdlib>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>

namespace sg {

class submit_handle {
public:
  explicit submit_handle(detail::node_record *n) : node_(n) {}

  template <class B> void define_kernel(B &&builder) {
    set_kernel(std::forward<B>(builder), nullptr);
  }
  template <class... K, class B> void define_kernel(std::tuple<K...> key, B &&builder) {
    static_assert(
        std::is_convertible_v<decltype(std::declval<const std::tuple<K...> &>() ==
                                       std::declval<const std::tuple<K...> &>()),
                              bool>,
        "sg::define_kernel: the key values must be equality comparable");
    set_kernel(std::forward<B>(builder), detail::make_key(std::move(key)));
  }
  template <class F> void define_host_work(F &&fn) {
    node_->kind = detail::node_record::kind_t::host_work;
    node_->host_fn = std::function<void()>(std::forward<F>(fn));
    ++node_->defines;
  }

private:
  template <class B> void set_kernel(B &&builder, std::unique_ptr<detail::key_holder> key) {
    using Bd = std::decay_t<B>;
    node_->kind = detail::node_record::kind_t::kernel;
    node_->builder = [b = Bd(std::forward<B>(builder))](kernel_handle &kh) { b(kh); };
    node_->builder_type = detail::type_id<Bd>();
    node_->key = std::move(key);
    ++node_->defines;
  }
  detail::node_record *node_;
};

namespace detail {

class owning_preparer final : public kernel_prep_service {
public:
  prepared_kernel prepare(node_record &n, const std::vector<slot_binding> &r,
                          const device_limits &l) override {
    return prep_.prepare(n, r, l);
  }

private:
  tree_store store_;
  kernel_preparer prep_{store_};
};

inline graph_services make_real_services(const graph_options &o) {
  auto ops = std::make_unique<sycl_device_ops>();
  device_ops *raw = ops.get();
  struct pools {
    std::map<int, std::unique_ptr<pool>> by_device;
  };
  auto reg = std::make_shared<pools>();

  memory_manager::services ms;
  ms.pool_of = [raw, reg, o](int dev) -> pool & {
    auto it = reg->by_device.find(dev);
    if (it == reg->by_device.end()) {
      const device_desc d = raw->describe(dev);
      const std::size_t reservation =
          pool_limit_rule(d.free_memory, d.total_memory, o.pool_reservation_bytes);
      auto p = std::make_unique<pool>(
          reservation, d.max_single_alloc, [raw, dev](std::size_t n) { return raw->allocate(dev, n); },
          [raw, dev](void *q) { raw->deallocate(dev, q); });
      it = reg->by_device.emplace(dev, std::move(p)).first;
    }
    return *it->second;
  };
  ms.host_alloc = [](std::size_t n) -> void * {
    void *p = std::malloc(n);
    if (!p)
      throw error{errc::pool_exhausted, "the host refused an allocation of " + std::to_string(n) + " bytes"};
    return p;
  };
  ms.host_free = [](void *p) { std::free(p); };
  ms.wait_ops = [raw](const std::vector<op_id> &v) { raw->wait(v); };
  ms.place_of = [raw](int dev) -> place {
    if (dev < 0)
      return place{};
    return place{dev, raw->describe(dev).context};
  };

  graph_services s;
  s.memory = std::make_unique<memory_manager>(std::move(ms));
  s.prep = std::make_unique<owning_preparer>();
  s.launch = std::make_unique<launcher>(*raw);
  s.ops = std::move(ops);
  return s;
}

} // namespace detail

class graph {
public:
  graph() : graph(graph_options{}) {}
  explicit graph(const graph_options &o)
      : impl_(std::make_unique<detail::graph_impl>(o, detail::make_real_services(o))) {}
  graph(const graph &) = delete;
  graph &operator=(const graph &) = delete;
  ~graph() = default;

  template <class T> mem_handle<T> create_mem_handle() {
    return detail::handle_access::make<mem_handle<T>>(
        impl_.get(), impl_->add_handle(detail::type_id<std::remove_const_t<T>>(), sizeof(T)));
  }
  template <class T> graph_promise<T> create_promise() {
    auto st = std::make_shared<detail::promise_state<T>>();
    const std::size_t i = impl_->add_promise(std::static_pointer_cast<detail::promise_base>(st),
                                             [st] { return st->is_set(); });
    return graph_promise<T>(st, i);
  }

  template <class F> void submit(const sycl::device &dev, F &&scope_fn) {
    detail::node_record &n = impl_->begin_scope(dev);
    try {
      submit_handle gh(&n);
      scope_fn(gh);
    } catch (...) {
      impl_->abort_scope();
      throw;
    }
    impl_->end_scope();
  }

  void run() { impl_->run(); }
  void wait() { impl_->wait(); }

private:
  std::unique_ptr<detail::graph_impl> impl_;
};

} // namespace sg

#endif
