/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_ASYNC_HOST_EXECUTOR_HPP
#define HIPSYCL_ASYNC_HOST_EXECUTOR_HPP

#include <chrono>
#include <future>
#include <memory>
#include <vector>

#include "executor.hpp"

namespace hipsycl {
namespace rt {

/// The event of a host operation: the shared future of the std::async launch
/// that runs it. It is complete exactly when that future is ready.
class async_host_event : public dag_node_event {
public:
  explicit async_host_event(std::shared_future<void> completion)
      : _completion{std::move(completion)} {}

  virtual bool is_complete() const override {
    return _completion.wait_for(std::chrono::seconds(0)) ==
           std::future_status::ready;
  }

  virtual void wait() override {
    auto completion = _completion;
    completion.wait();
  }

  std::shared_future<void> get_future() const { return _completion; }

private:
  std::shared_future<void> _completion;
};

/// Runs each host operation as one std::async launch with the asynchronous
/// policy. The executor owns no thread and no queue of operations; the thread
/// of an operation lives as long as the operation's future.
class async_host_executor : public backend_executor {
public:
  virtual bool is_inorder_queue() const override { return false; }
  virtual bool is_outoforder_queue() const override { return true; }
  virtual bool is_taskgraph() const override { return false; }
  virtual bool is_backend_queue() const override { return false; }

  virtual void submit_directly(const dag_node_ptr& node, operation *op,
                               const node_list_t &reqs) override;

  virtual bool can_execute_on_device(const device_id& dev) const override {
    return true;
  }

  virtual bool is_submitted_by_me(const dag_node_ptr& node) const override;
};

}
}

#endif
