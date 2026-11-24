// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

struct NoOp {};

void use(sg::mem_handle<int> h) { auto s = sg::inplace(h, NoOp{}, 0); (void)s; }
// CHECK: error: static assertion failed{{.*}}sg::inplace
