// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) {
    int v = int(it.id()[0]);
    goto done;
  done:
    return v;
  });
}
// CHECK: error: sg: goto:
// CHECK: note: in the body defined here
