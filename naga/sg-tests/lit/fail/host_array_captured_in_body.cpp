// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh, sg::host_array<int, 1> h) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [h](sg::item<1> it) { return h[it.id()]; });
}
// CHECK: error: static assertion failed{{.*}}trivially copyable
