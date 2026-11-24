// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

int pong(int n);
int ping(int n) { return n ? pong(n - 1) : 0; }
int pong(int n) { return n ? ping(n - 1) : 0; }

void build(sg::kernel_handle &kh, sg::mem_handle<int> in) {
  auto x = kh.source(in, sg::shape<1>({8}));
  auto items = kh.work_items({8});
  kh.map_items(items, [=](auto item) {
    auto v = x[item.id()];
    return item.compute(v, [](int a) { return ping(a); });
  });
}
// CHECK: error: sg: recursion:
// CHECK: note: in the body defined here
