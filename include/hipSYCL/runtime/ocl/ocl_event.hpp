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
#ifndef HIPSYCL_OCL_EVENT_HPP
#define HIPSYCL_OCL_EVENT_HPP

#include "../inorder_queue_event.hpp"
#include <CL/opencl.hpp>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace hipsycl {
namespace rt {

class ocl_node_event : public inorder_queue_event<cl::Event>
{
public:
  using backend_event_type = cl::Event;
 
  ocl_node_event(device_id dev, cl::Event evt);
  ocl_node_event(device_id dev);

  ~ocl_node_event();

  virtual bool is_complete() const override;
  virtual void wait() override;

  backend_event_type get_event() const;
  device_id get_device() const;

  backend_event_type request_backend_event() override;
private:
  bool _is_empty = false;
  device_id _dev;
  backend_event_type _evt;
};

/// Event for an operation that has not been submitted to the backend yet.
///
/// Operations that must have their dependencies satisfied before they are
/// issued are submitted from another thread, so there is no backend event to
/// describe them at the time the node is created. This event stands in until
/// that thread has issued the operation, at which point it adopts the event
/// that describes it. Waiting on the operation therefore waits for the
/// submission and then for the operation itself, and requesting the backend
/// event blocks until one exists.
///
/// Stamping also signals that the operation has reached the backend queue, so
/// that submissions which rely on enqueue order can wait for it.
class ocl_deferred_event : public inorder_queue_event<cl::Event> {
public:
  using backend_event_type = cl::Event;

  virtual bool is_complete() const override;
  virtual void wait() override;

  virtual bool is_submitted() const override { return _is_stamped; }
  virtual void wait_for_submission() const override;

  backend_event_type request_backend_event() override;

  /// Adopt the event describing the operation, which has now been issued.
  /// A null event marks an operation that was never issued, so that anything
  /// waiting for it is released rather than left waiting forever.
  void stamp(std::shared_ptr<inorder_queue_event<cl::Event>> evt);
private:
  std::atomic<bool> _is_stamped{false};
  std::shared_ptr<inorder_queue_event<cl::Event>> _evt;
  mutable std::mutex _mutex;
  mutable std::condition_variable _submitted;
};

}
}


#endif