// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

int counter;

void build(sg::kernel_handle &kh, sg::mem_handle<int> in) {
  auto x = kh.source(in, sg::shape<1>({8}));
  auto items = kh.work_items({8});
  kh.map_items(items, [=](auto item) {
    auto v = x[item.id()];
    return item.compute(v, [](int a) {
      counter += 2;
      return a;
    });
  });
}
// CHECK: error: sg: write:
// CHECK: note: in the body defined here
