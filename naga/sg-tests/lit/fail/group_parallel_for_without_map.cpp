// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh) {
  kh.parallel_for(sycl::range<1>(2), [](sg::group<1>) { return sg::scalar_expr<int>{1}; });
}
// CHECK: error: no matching member function
