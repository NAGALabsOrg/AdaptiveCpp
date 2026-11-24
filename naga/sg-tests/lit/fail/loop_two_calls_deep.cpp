// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

int three(int x) {
  int acc = 0;
  for (int i = 0; i < x; ++i)
    acc += i;
  return acc;
}
int two(int x) { return three(x); }
int one(int x) { return two(x); }

void build(sg::kernel_handle &kh, sg::mem_handle<int> in) {
  auto x = kh.source(in, sg::shape<1>({8}));
  auto items = kh.work_items({8});
  kh.map_items(items, [=](auto item) {
    auto v = x[item.id()];
    return item.compute(v, [](int a) { return one(a); });
  });
}
// CHECK: error: sg: loop (for):{{.*}}in 'three'
// CHECK: note: in the body defined here
// CHECK: note: called from 'two' here
// CHECK: note: called from 'one' here
// CHECK: note: called from 'the body' here
// CHECK-NOT: __sg_
