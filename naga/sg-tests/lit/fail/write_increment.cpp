// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

int counter;

void build(sg::kernel_handle &kh) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) {
    ++counter;
    return int(it.id()[0]);
  });
}
// CHECK: error: sg: non-local-write:
// CHECK: note: in the body defined here
