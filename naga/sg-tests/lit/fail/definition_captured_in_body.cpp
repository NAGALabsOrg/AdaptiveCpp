// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh) {
  auto a = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); });
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [a](sg::item<1>) { return a.index(); });
}
// CHECK: error: static assertion failed{{.*}}trivially copyable
