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
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <level_zero/ze_api.h>

#ifndef HIPSYCL_ZE_EVENT_HPP
#define HIPSYCL_ZE_EVENT_HPP

#include "../inorder_queue_event.hpp"

namespace hipsycl {
namespace rt {


class ze_node_event : public inorder_queue_event<ze_event_handle_t>
{
public:
  /// Takes ownership of supplied ze_event_handle_t
  ze_node_event(ze_event_handle_t evt, 
    std::shared_ptr<ze_event_pool_handle_t> pool);
  ~ze_node_event();

  virtual bool is_complete() const override;
  virtual void wait() override;

  ze_event_handle_t get_event_handle() const;
  virtual ze_event_handle_t request_backend_event() override;
private:
  ze_event_handle_t _evt;
  std::shared_ptr<ze_event_pool_handle_t> _pool;
};

/// Event for an operation that has not been submitted to the backend yet.
///
/// Operations that must have their requirements completed before they are
/// issued are submitted from another thread, so there is no backend event to
/// describe them at the time the node is created. This event stands in until
/// that thread has issued the operation, at which point it adopts the event
/// that describes it. Waiting on the operation therefore waits for the
/// submission and then for the operation itself, and requesting the backend
/// event blocks until one exists.
///
/// Stamping also signals that the operation has reached the backend queue, so
/// that submissions which rely on enqueue order can wait for it.
class ze_deferred_event : public inorder_queue_event<ze_event_handle_t> {
public:
  virtual bool is_complete() const override;
  virtual void wait() override;

  virtual bool is_submitted() const override { return _is_stamped; }
  virtual void wait_for_submission() const override;

  /// Returns a null handle for an operation that was never issued.
  virtual ze_event_handle_t request_backend_event() override;

  /// Adopt the event describing the operation, which has now been issued.
  /// A null event marks an operation that was never issued, so that anything
  /// waiting for it is released rather than left waiting forever.
  void stamp(std::shared_ptr<inorder_queue_event<ze_event_handle_t>> evt);
private:
  std::atomic<bool> _is_stamped{false};
  std::shared_ptr<inorder_queue_event<ze_event_handle_t>> _evt;
  mutable std::mutex _mutex;
  mutable std::condition_variable _submitted;
};


}
}

#endif
