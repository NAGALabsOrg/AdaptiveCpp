// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) {
    int acc = 0;
    for (int i = 0; i < 3; ++i)
      acc += i;
    return acc + int(it.id()[0]);
  });
}
// CHECK: error: sg: loop:
// CHECK: note: in the body defined here
// CHECK-NOT: __sg_
