// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

int fact(int n) { return n ? n * fact(n - 1) : 1; }

void build(sg::kernel_handle &kh) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return fact(int(it.id()[0])); });
}
// CHECK: error: sg: call-cycle:
// CHECK: note: in the body defined here
