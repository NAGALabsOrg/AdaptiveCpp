// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

struct Big {
  double v[3];
};
struct BigOp {
  Big operator()(Big a, Big) const { return a; }
};

void use(sg::mem_handle<Big> h) { auto s = sg::inplace(h, BigOp{}, Big{}); (void)s; }
// CHECK: error: static assertion failed{{.*}}elements are at most 8 bytes
