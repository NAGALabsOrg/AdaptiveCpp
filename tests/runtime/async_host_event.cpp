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

#include "runtime_test_suite.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>

#include <sycl/sycl.hpp>

#include <hipSYCL/runtime/async_host_executor.hpp>
#include <hipSYCL/runtime/dag_node.hpp>

namespace rt = hipsycl::rt;

namespace {

// Boost cannot print a std::future_status, so a check is given a plain truth value.
bool is_ready_within(std::shared_future<void> &f, std::chrono::seconds limit) {
  return f.wait_for(limit) == std::future_status::ready;
}

bool is_still_pending_after(std::shared_future<void> &f, std::chrono::seconds limit) {
  return f.wait_for(limit) == std::future_status::timeout;
}

}

BOOST_FIXTURE_TEST_SUITE(async_host_event, reset_device_fixture)

// C28.b: the event of a host operation is complete exactly when the future of
// its launch is ready.
BOOST_AUTO_TEST_CASE(event_of_a_host_operation_is_the_future_of_its_launch) {
  using namespace std::chrono_literals;

  sycl::queue q;
  auto gate_is_open = std::make_shared<std::atomic<bool>>(false);

  sycl::event e = q.async_host([gate_is_open]() {
    while(!gate_is_open->load())
      std::this_thread::sleep_for(1ms);
  });

  rt::dag_node_ptr node = sycl::detail::extract_rt_event(e);
  while(!node->is_submitted())
    std::this_thread::sleep_for(1ms);

  std::shared_ptr<rt::dag_node_event> node_event = node->get_event();
  auto *host_event = dynamic_cast<rt::async_host_event *>(node_event.get());
  BOOST_REQUIRE(host_event != nullptr);

  std::shared_future<void> launch = host_event->get_future();

  // gate closed
  BOOST_TEST(!host_event->is_complete());
  BOOST_TEST(is_still_pending_after(launch, 1s));

  // gate open
  *gate_is_open = true;
  BOOST_TEST(is_ready_within(launch, 10s));
  BOOST_TEST(host_event->is_complete());

  q.wait_and_throw();
}

BOOST_AUTO_TEST_SUITE_END()
