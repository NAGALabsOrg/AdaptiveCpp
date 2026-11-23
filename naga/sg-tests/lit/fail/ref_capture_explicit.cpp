// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh) {
  int k = 1;
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [&k](sg::item<1> it) { return k + int(it.id()[0]); });
}
// CHECK: error: sg: ref-capture:
// CHECK: note: in the body defined here
