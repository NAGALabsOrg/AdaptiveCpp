// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

struct NT {
  NT() {}
  NT(const NT &) {}
  int v;
};
struct NTOp {
  NT operator()(NT a, NT) const { return a; }
};

void use(sg::mem_handle<NT> h) { (void)h; auto s = sg::inplace(h, NTOp{}, NT{}); (void)s; }
// CHECK: error:
