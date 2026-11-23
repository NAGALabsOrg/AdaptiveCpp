// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

int inner(int x) {
  int acc = 0;
  for (int i = 0; i < x; ++i)
    acc += i;
  return acc;
}
int outer(int x) { return inner(x); }

void build(sg::kernel_handle &kh) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return outer(int(it.id()[0])); });
}
// CHECK: error: sg: loop:
// CHECK: note: in the body defined here
// CHECK: note: called from 'outer' here
// CHECK: note: called from 'the body' here
// CHECK-NOT: __sg_
