// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh, int *p) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [p](sg::item<1> it) {
    *p = 1;
    return int(it.id()[0]);
  });
}
// CHECK: error: sg: non-local-write:
// CHECK: note: in the body defined here
