// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

#include <functional>

void build(sg::kernel_handle &kh, std::function<int(int)> f) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [f](sg::item<1> it) { return f(int(it.id()[0])); });
}
// CHECK: error:
