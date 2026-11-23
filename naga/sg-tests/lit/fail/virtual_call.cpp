// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

struct Base {
  virtual int f() { return 1; }
};

void build(sg::kernel_handle &kh) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) {
    Base b;
    Base *p = &b;
    return p->f() + int(it.id()[0]);
  });
}
// CHECK: error: sg: virtual-call:
// CHECK: note: in the body defined here
