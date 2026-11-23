// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

int external_function(int);

void build(sg::kernel_handle &kh) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return external_function(int(it.id()[0])); });
}
// CHECK: error: sg: no-body:
// CHECK: note: in the body defined here
