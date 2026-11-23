// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh, int (*fp)(int)) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [fp](sg::item<1> it) { return fp(int(it.id()[0])); });
}
// CHECK: error: sg: function-pointer-call:
// CHECK: note: in the body defined here
