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

// Runs 1000 host operations to completion and checks that the process has the
// same number of threads afterwards as before. Linux only.
//
//   async_host_threads_test        exit status 0 when the two counts are equal

#include <sycl/sycl.hpp>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace {

int thread_count() {
  std::ifstream status{"/proc/self/status"};
  std::string line;
  while(std::getline(status, line)) {
    if(line.rfind("Threads:", 0) == 0)
      return std::stoi(line.substr(8));
  }
  return -1;
}

bool count_becomes(int expected, std::chrono::seconds limit) {
  const auto deadline = std::chrono::steady_clock::now() + limit;
  while(std::chrono::steady_clock::now() < deadline) {
    if(thread_count() == expected)
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return thread_count() == expected;
}

}

int main() {
  using namespace std::chrono_literals;

  sycl::queue q;

  // every thread the runtime itself needs exists after the first operation
  q.async_host([]() {}).wait();
  q.wait_and_throw();
  std::this_thread::sleep_for(500ms);

  const int before = thread_count();

  auto finished = std::make_shared<std::atomic<int>>(0);
  for(int i = 0; i < 1000; ++i) {
    q.async_host([finished]() {
      std::this_thread::sleep_for(1ms);
      ++*finished;
    });
  }
  q.wait_and_throw();

  const bool all_ran = finished->load() == 1000;
  const bool same = count_becomes(before, 10s);
  const int after = thread_count();

  std::cout << "threads before: " << before << ", after: " << after
            << ", operations finished: " << finished->load() << std::endl;
  return (all_ran && same) ? 0 : 1;
}
