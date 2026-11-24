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
#include <cassert>
#include <exception>
#include <future>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

#include "hipSYCL/runtime/async_host_executor.hpp"
#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/operations.hpp"

namespace hipsycl {
namespace rt {

namespace {

std::shared_future<void> make_ready_future() {
  std::promise<void> ready;
  ready.set_value();
  return ready.get_future().share();
}

}

bool async_host_executor::is_submitted_by_me(const dag_node_ptr& node) const {
  if(!node->is_submitted())
    return false;
  return node->get_assigned_executor() == this;
}

void async_host_executor::submit_directly(const dag_node_ptr& node,
                                          operation *op,
                                          const node_list_t &reqs) {
  assert(!op->is_requirement());

  if(node->is_submitted())
    return;

  auto *host_op = static_cast<async_host_operation *>(op);
  async_host_operation::host_function body = host_op->take_function();
  std::shared_ptr<dag_node_event> preceding = host_op->get_preceding_event();
  node_list_t requirements = reqs;

  std::shared_future<void> completion;
  try {
    completion =
        std::async(std::launch::async,
                   [body = std::move(body),
                    requirements = std::move(requirements),
                    preceding = std::move(preceding)]() mutable {
          for(const auto &req : requirements) {
            if(req)
              req->wait();
          }
          requirements.clear();

          if(preceding)
            preceding->wait();
          preceding.reset();

          try {
            body();
          } catch(const std::exception &e) {
            register_error(
                __acpp_here(),
                error_info{std::string{"async_host: a host operation threw: "} +
                               e.what(),
                           error_type::runtime_error});
          } catch(...) {
            register_error(
                __acpp_here(),
                error_info{"async_host: a host operation threw an exception "
                           "that is not derived from std::exception",
                           error_type::runtime_error});
          }

          // The future becomes ready only after the captured objects are gone.
          body = nullptr;
        }).share();
  } catch(const std::exception &e) {
    register_error(
        __acpp_here(),
        error_info{std::string{"async_host_executor: could not launch a host "
                               "operation: "} +
                       e.what(),
                   error_type::runtime_error});
    completion = make_ready_future();
  }

  node->mark_submitted(std::make_shared<async_host_event>(std::move(completion)));
}

}
}
