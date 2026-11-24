// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s

// A per-work-item array of 24 bytes with 16-byte alignment is live across a group barrier, so the work-item loops of the sub-CFGs
// share one widened array of these slices. Zeroing and filling the array uses aligned 16-byte accesses, which are only valid if every
// work item's slice keeps the array's alignment, not just the first slice of every second item.

#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

int main() {
  constexpr size_t local_size = 64;
  constexpr size_t groups     = 2;
  constexpr size_t total      = groups * local_size;

  sycl::queue queue{sycl::property::queue::in_order{}};
  long long *out = sycl::malloc_device<long long>(total, queue);

  queue.parallel_for(sycl::nd_range<1>{total, local_size}, [=](sycl::nd_item<1> item) {
    alignas(16) long long v[3] = {0, 0, 0};
    const long long lid = static_cast<long long>(item.get_local_id(0));
    const long long gid = static_cast<long long>(item.get_global_id(0));
    for (int j = 0; j < 3; ++j)
      v[(lid + j) % 3] += lid * (j + 1) + gid;
    sycl::group_barrier(item.get_group());
    out[gid] = v[0] + 1000 * v[1] + 1000000 * v[2];
  });

  std::vector<long long> result(total);
  queue.memcpy(result.data(), out, total * sizeof(long long)).wait_and_throw();
  sycl::free(out, queue);

  bool ok = true;
  for (long long gid = 0; gid < static_cast<long long>(total); ++gid) {
    const long long lid = gid % static_cast<long long>(local_size);
    long long v[3] = {0, 0, 0};
    for (int j = 0; j < 3; ++j)
      v[(lid + j) % 3] += lid * (j + 1) + gid;
    ok = ok && result[gid] == v[0] + 1000 * v[1] + 1000000 * v[2];
  }
  std::cout << "overaligned_local_array: " << (ok ? "PASSED" : "FAILED") << "\n";
  return ok ? 0 : 1;
}

// CHECK: overaligned_local_array: PASSED
