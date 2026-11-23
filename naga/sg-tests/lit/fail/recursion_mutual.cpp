// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

int pong(int n);
int ping(int n) { return n ? pong(n - 1) : 0; }
int pong(int n) { return n ? ping(n - 1) : 0; }

void build(sg::kernel_handle &kh) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return ping(int(it.id()[0])); });
}
// CHECK: error: sg: call-cycle:
// CHECK: note: in the body defined here
