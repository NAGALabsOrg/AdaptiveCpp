// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s

#include <iostream>

#include <sycl/sycl.hpp>

int main() {
  sycl::queue queue{sycl::property::queue::in_order{}};

  int *counter = sycl::malloc_device<int>(1, queue);
  int *output = sycl::malloc_device<int>(8, queue);
  queue.memset(counter, 0, sizeof(int));

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1> item) {
      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          ref{*counter};
      for (int i = 0; i < 8; i++) {
        int x = 0;
        if (item.get_local_id(0) == 0)
          x = ref.fetch_add(1);
        sycl::group_barrier(item.get_group());
        output[i] = x;
      }
    });
  });

  int result[8];
  queue.memcpy(result, output, sizeof(result)).wait_and_throw();
  sycl::free(counter, queue);
  sycl::free(output, queue);

  std::cout << "result:";
  for (int i = 0; i < 8; i++)
    std::cout << ' ' << result[i];
  std::cout << '\n';
}

// CHECK: result: 0 1 2 3 4 5 6 7
