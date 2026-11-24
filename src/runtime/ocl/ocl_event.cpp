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
#include "hipSYCL/runtime/ocl/ocl_event.hpp"
#include "hipSYCL/runtime/device_id.hpp"
namespace hipsycl {
namespace rt {

ocl_node_event::ocl_node_event(device_id dev, cl::Event evt)
: _dev{dev}, _evt{evt} {}

ocl_node_event::ocl_node_event(device_id dev)
: _is_empty{true}, _dev{dev}, _evt{} {}

ocl_node_event::~ocl_node_event(){}

bool ocl_node_event::is_complete() const {
  if(_is_empty)
    return true;

  cl_int status;
  cl_int err = _evt.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status);

  if(err != CL_SUCCESS) {
    register_error(__acpp_here(),
                   error_info{"ocl_node_event: Couldn't query event status",
                              error_code{"CL", err}});
    return false;
  }

  if(status < 0) {
    // Command was abnormally terminated
    register_error(__acpp_here(),
                   error_info{"ocl_node_event: Event status indicates that "
                              "device operation was abnormally terminated",
                              error_code{"CL", err}});
    return true;
  }
  
  return status == CL_COMPLETE;
}

void ocl_node_event::wait() {
  if(_is_empty)
    return;

  cl_int err = _evt.wait();
  if(err != CL_SUCCESS) {
    register_error(__acpp_here(),
                   error_info{"ocl_node_event: Waiting for event failed",
                              error_code{"CL", err}});
  }
}

ocl_node_event::backend_event_type ocl_node_event::get_event() const {
  return _evt;
}

device_id ocl_node_event::get_device() const {
  return _dev;
}

ocl_node_event::backend_event_type ocl_node_event::request_backend_event() {
  return get_event();
}

void ocl_deferred_event::stamp(
    std::shared_ptr<inorder_queue_event<cl::Event>> evt) {
  {
    std::lock_guard<std::mutex> lock{_mutex};
    _evt = std::move(evt);
    _is_stamped = true;
  }
  _submitted.notify_all();
}

void ocl_deferred_event::wait_for_submission() const {
  if(_is_stamped)
    return;

  std::unique_lock<std::mutex> lock{_mutex};
  _submitted.wait(lock, [this]{ return _is_stamped.load(); });
}

bool ocl_deferred_event::is_complete() const {
  if(!_is_stamped)
    return false;
  return _evt ? _evt->is_complete() : true;
}

void ocl_deferred_event::wait() {
  wait_for_submission();
  if(_evt)
    _evt->wait();
}

ocl_deferred_event::backend_event_type
ocl_deferred_event::request_backend_event() {
  wait_for_submission();
  // A null cl::Event stands for an operation that was never issued.
  return _evt ? _evt->request_backend_event() : cl::Event{};
}


}
}
