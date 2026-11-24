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

// A host operation that cannot be launched because the process may not create
// another thread. The process's own thread limit is lowered for a short time,
// so this program runs by itself and only on the OpenMP device. Linux only.
//
//   async_host_unlaunchable_test        exit status 0 when every check passed or the
//                                       program was skipped (message says why)

#include <sycl/sycl.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <system_error>
#include <thread>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

namespace {

int failures = 0;

void check(bool condition, const char *what) {
  std::cout << (condition ? "pass: " : "FAIL: ") << what << std::endl;
  if(!condition)
    ++failures;
}

bool is_complete(sycl::event e) {
  return e.get_info<sycl::info::event::command_execution_status>() ==
         sycl::info::event_command_status::complete;
}

bool becomes_true(const std::atomic<bool> &flag, std::chrono::seconds limit) {
  const auto deadline = std::chrono::steady_clock::now() + limit;
  while(std::chrono::steady_clock::now() < deadline) {
    if(flag.load())
      return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return flag.load();
}

}

int main() {
  using namespace std::chrono_literals;

  std::atomic<int> errors_delivered{0};
  auto handler = [&errors_delivered](sycl::exception_list errors) {
    errors_delivered += static_cast<int>(errors.size());
  };

  sycl::queue q{handler};
  sycl::queue in_order_queue{handler, sycl::property_list{sycl::property::queue::in_order{}}};

  if(q.get_device().get_backend() != sycl::backend::omp) {
    std::cout << "skipped: the default device is not an OpenMP device" << std::endl;
    return 0;
  }

  // a stuck wait ends the program instead of blocking it
  ::alarm(120);

  // every thread the runtime itself needs exists after the first operations
  q.async_host([]() {}).wait();
  in_order_queue.async_host([]() {}).wait();
  q.wait_and_throw();
  in_order_queue.wait_and_throw();

  rlimit original;
  ::getrlimit(RLIMIT_NPROC, &original);
  rlimit lowered = original;
  lowered.rlim_cur = 0;
  ::setrlimit(RLIMIT_NPROC, &lowered);

  bool refused = false;
  try {
    std::thread probe([]() {});
    probe.join();
  } catch(const std::system_error &) {
    refused = true;
  }
  if(!refused) {
    ::setrlimit(RLIMIT_NPROC, &original);
    std::cout << "skipped: the thread limit is not enforced for this process (privileged)"
              << std::endl;
    return 0;
  }

  // C18.a and C19.a: the call returns, the body does not run, one error, the event completes
  std::atomic<bool> body_ran{false};
  bool submit_threw = false;
  sycl::event failed;
  try {
    failed = q.async_host([&body_ran]() { body_ran = true; });
  } catch(...) {
    submit_threw = true;
  }
  failed.wait();
  const bool failed_is_complete = is_complete(failed);
  q.wait_and_throw();
  const int errors_after_first = errors_delivered.load();

  // C19.c: an in-order queue whose previous operation failed
  bool in_order_submit_threw = false;
  sycl::event failed_in_order;
  try {
    failed_in_order = in_order_queue.async_host([]() {});
  } catch(...) {
    in_order_submit_threw = true;
  }
  failed_in_order.wait();
  in_order_queue.wait_and_throw();

  ::setrlimit(RLIMIT_NPROC, &original);

  check(!submit_threw, "C18.a the call does not throw to the submitting thread");
  check(!body_ran.load(), "C18.a the body does not run");
  check(errors_after_first == 1, "C18.a exactly one asynchronous error is delivered");
  check(failed_is_complete, "C19.a the event of the failed operation is complete");
  check(!in_order_submit_threw, "C19.c the in-order call does not throw");

  // C18.b: after the limit is lifted, a new operation runs
  std::atomic<bool> later_ran{false};
  sycl::event later = q.async_host([&later_ran]() { later_ran = true; });
  check(becomes_true(later_ran, 10s), "C18.b a new operation runs after the limit is lifted");
  later.wait();

  // C19.b: dependents of the failed operation
  std::atomic<bool> dependent_ran{false};
  sycl::event dependent =
      q.async_host([&dependent_ran]() { dependent_ran = true; }, failed);
  check(becomes_true(dependent_ran, 10s), "C19.b a dependent host operation runs");
  dependent.wait();

  int *kernel_ran = sycl::malloc_shared<int>(1, q);
  *kernel_ran = 0;
  sycl::event kernel = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(failed);
    cgh.single_task<class async_host_unlaunchable_kernel>([=]() { *kernel_ran = 1; });
  });
  kernel.wait();
  check(*kernel_ran == 1, "C19.b a dependent kernel runs");
  sycl::free(kernel_ran, q);

  // C19.c: the next operation after the failed in-order one is not blocked
  std::atomic<bool> next_ran{false};
  sycl::event next = in_order_queue.async_host([&next_ran]() { next_ran = true; });
  check(becomes_true(next_ran, 10s), "C19.c the next in-order operation runs");
  next.wait();

  q.wait_and_throw();
  in_order_queue.wait_and_throw();

  std::cout << (failures == 0 ? "all checks passed" : "checks failed") << std::endl;
  return failures == 0 ? 0 : 1;
}
