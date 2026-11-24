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
      int a[3] = {1, 2, 3};
      int acc = 0;
      for (int e : a)
        acc += e;
      return acc + v;
    });
  });
}
// CHECK: error: sg: loop (range-for):
// CHECK: note: in the body defined here
