// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh, sg::mem_handle<float> in, sg::mem_handle<int> n_handle) {
  auto n = kh.launch_value(n_handle);
  auto x = kh.source(in, sg::shape<1>({n}));
  auto tiles = kh.tile(x, {n});
}
// CHECK: error: static assertion failed{{.*}}launch_value<int>{{.*}}structure
