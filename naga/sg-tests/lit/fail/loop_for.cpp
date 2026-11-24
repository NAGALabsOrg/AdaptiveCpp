// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh, sg::mem_handle<int> in) {
  auto x = kh.source(in, sg::shape<1>({64}));
  auto tiles = kh.tile(x, {4});
  auto work = kh.work_tiles({16}, {1}, 4);
  kh.map_work(work, [=](auto unit) {
    auto tile = tiles[unit.tile()];
    auto run = sg::flatten(tile);
    return unit.run(run, [](auto item, int v) {
      int acc = 0;
      for (int i = 0; i < 3; ++i)
        acc += i;
      return acc + v;
    });
  });
}
// CHECK: error: sg: loop (for):
// CHECK: note: in the body defined here
// CHECK-NOT: __sg_
