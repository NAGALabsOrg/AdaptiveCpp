// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

struct NoEq {
  int v;
};

void define(sg::submit_handle gh) {
  gh.define_kernel(std::make_tuple(NoEq{1}), [](sg::kernel_handle &) {});
}
// CHECK: error:
