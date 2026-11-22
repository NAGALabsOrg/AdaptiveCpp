// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s

#include <algorithm>
#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

using counter_ref =
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>;

constexpr int n = 64;

struct expectation {
  int first, step, counter_end;
  int second_offset; // < 0: the second output is not checked
};

template <class Body>
bool run_case(sycl::queue &queue, const char *name, int counter_start,
              const expectation &e, Body body) {
  int *counter = sycl::malloc_device<int>(1, queue);
  int *out1 = sycl::malloc_device<int>(n, queue);
  int *out2 = sycl::malloc_device<int>(n, queue);
  queue.memcpy(counter, &counter_start, sizeof(int));

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>{n, 16}, [=](sycl::nd_item<1> item) {
      body(item, counter, out1, out2);
    });
  });

  int end = 0;
  std::vector<int> r1(n), r2(n);
  queue.memcpy(&end, counter, sizeof(int));
  queue.memcpy(r1.data(), out1, n * sizeof(int));
  queue.memcpy(r2.data(), out2, n * sizeof(int)).wait_and_throw();
  sycl::free(counter, queue);
  sycl::free(out1, queue);
  sycl::free(out2, queue);

  bool ok = end == e.counter_end;
  std::vector<int> sorted = r1;
  std::sort(sorted.begin(), sorted.end());
  for (int i = 0; i < n; ++i) {
    ok = ok && sorted[i] == e.first + i * e.step;
    if (e.second_offset >= 0)
      ok = ok && r2[i] == r1[i] + e.second_offset;
  }
  if (ok)
    std::cout << name << ": PASSED\n";
  else
    std::cout << name << ": FAILED counter " << end << " expected "
              << e.counter_end << " first output " << sorted[0] << '\n';
  return ok;
}

int main() {
  sycl::queue queue{sycl::property::queue::in_order{}};
  bool ok = true;

  ok &= run_case(queue, "add1", 0, {0, 1, 64, -1},
                 [](sycl::nd_item<1> item, int *c, int *o1, int *) {
                   int x = counter_ref{*c}.fetch_add(1);
                   sycl::group_barrier(item.get_group());
                   o1[item.get_global_id(0)] = x;
                 });
  ok &= run_case(queue, "add3", 0, {0, 3, 192, -1},
                 [](sycl::nd_item<1> item, int *c, int *o1, int *) {
                   int x = counter_ref{*c}.fetch_add(3);
                   sycl::group_barrier(item.get_group());
                   o1[item.get_global_id(0)] = x;
                 });
  ok &= run_case(queue, "sub_neg2", 0, {0, 2, 128, -1},
                 [](sycl::nd_item<1> item, int *c, int *o1, int *) {
                   int x = counter_ref{*c}.fetch_sub(-2);
                   sycl::group_barrier(item.get_group());
                   o1[item.get_global_id(0)] = x;
                 });
  ok &= run_case(queue, "derived", 0, {1, 2, 64, -1},
                 [](sycl::nd_item<1> item, int *c, int *o1, int *) {
                   int x = counter_ref{*c}.fetch_add(1);
                   int y = 2 * x + 1;
                   sycl::group_barrier(item.get_group());
                   o1[item.get_global_id(0)] = y;
                 });
  ok &= run_case(queue, "sub1", 64, {1, 1, 0, -1},
                 [](sycl::nd_item<1> item, int *c, int *o1, int *) {
                   int x = counter_ref{*c}.fetch_sub(1);
                   sycl::group_barrier(item.get_group());
                   o1[item.get_global_id(0)] = x;
                 });
  ok &= run_case(queue, "two_barriers", 0, {0, 1, 64, 100},
                 [](sycl::nd_item<1> item, int *c, int *o1, int *o2) {
                   int x = counter_ref{*c}.fetch_add(1);
                   sycl::group_barrier(item.get_group());
                   o1[item.get_global_id(0)] = x;
                   sycl::group_barrier(item.get_group());
                   o2[item.get_global_id(0)] = x + 100;
                 });

  return ok ? 0 : 1;
}

// CHECK: add1: PASSED
// CHECK-NEXT: add3: PASSED
// CHECK-NEXT: sub_neg2: PASSED
// CHECK-NEXT: derived: PASSED
// CHECK-NEXT: sub1: PASSED
// CHECK-NEXT: two_barriers: PASSED
