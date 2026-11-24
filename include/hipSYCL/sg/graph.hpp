// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_GRAPH_HPP
#define HIPSYCL_SG_GRAPH_HPP

#include "hipSYCL/sg/detail/graph_impl.hpp"
#include "hipSYCL/sg/detail/sycl_device_ops.hpp"
#include "hipSYCL/sg/graph_options.hpp"
#include "hipSYCL/sg/handle.hpp"
#include "hipSYCL/sg/kernel.hpp"
#include "hipSYCL/sg/kernel_kinds.hpp"
#include "hipSYCL/sg/roles.hpp"

#include <sycl/sycl.hpp>

#include <cstdlib>
#include <map>
#include <memory>
#include <type_traits>

namespace sg {

namespace detail {

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
      auto p = std::make_unique<pool>(
          o.pool_reservation_bytes, d.max_single_alloc, [raw, dev](std::size_t n) { return raw->allocate(dev, n); },
          [raw, dev](void *q) { raw->deallocate(dev, q); }, [raw](op_id id) { return raw->finished(id); });
      it = reg->by_device.emplace(dev, std::move(p)).first;
    }
    return *it->second;
  };
  ms.host_alloc = [](std::size_t n) -> void * {
    void *p = std::malloc(n);
    if (!p)
      throw error{errc::allocation_refused, "the host refused an allocation of " + std::to_string(n) + " bytes"};
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
  s.pool_slabs = [reg]() {
    std::size_t n = 0;
    for (const auto &p : reg->by_device)
      n += p.second->slab_count();
    return n;
  };
  s.memory = std::make_unique<memory_manager>(std::move(ms));
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
  explicit graph(std::unique_ptr<detail::graph_impl> impl) : impl_(std::move(impl)) {}
  graph(const graph &) = delete;
  graph &operator=(const graph &) = delete;
  ~graph() = default;

  template <class T> mem_handle<T> create_mem_handle() {
    return detail::handle_access::make<mem_handle<T>>(
        impl_.get(), impl_->add_handle(detail::type_id<std::remove_const_t<T>>(), sizeof(T)));
  }

  // A device kernel: the lambda is the kernel builder and runs now.
  template <class F> void submit(const sycl::device &dev, F &&builder) {
    static_assert(std::is_invocable_v<F &, kernel_handle &>, "the builder of a kernel takes an sg::kernel_handle&");
    detail::node_record &n = impl_->begin_scope(dev);
    try {
      kernel_handle kh(n);
      builder(kh);
    } catch (...) {
      impl_->abort_scope();
      throw;
    }
    impl_->end_scope();
  }

  // Host work: the lambda runs now and describes the arrays and the work.
  template <class Kind, class F, std::enable_if_t<std::is_same_v<Kind, host>, int> = 0> void submit(F &&fn) {
    static_assert(std::is_invocable_v<F &, host_handle &>, "the builder of host work takes an sg::host_handle&");
    detail::node_record &n = impl_->begin_host_scope();
    try {
      host_handle hh(n);
      fn(hh);
    } catch (...) {
      impl_->abort_scope();
      throw;
    }
    impl_->end_scope();
  }

  // Interop is declared and not designed.
  template <class Kind, class... A, std::enable_if_t<std::is_same_v<Kind, interop>, int> = 0>
  void submit(A &&...) {
    throw build_error(15, "sg::interop", {"<submit>", 0},
                              "submit<sg::interop> is not designed: use a device kernel or host work");
  }

  void run() { impl_->run(); }
  void wait() { impl_->wait(); }

  const detail::run_stats &stats() const noexcept { return impl_->stats(); }

  // The slabs the graph's memory pools have taken from the devices so far.
  std::size_t pool_slab_count() const { return impl_->pool_slab_count(); }

private:
  std::unique_ptr<detail::graph_impl> impl_;
};

} // namespace sg

#endif
