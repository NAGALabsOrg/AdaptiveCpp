// RUN: %acpp %s
#include "hipSYCL/sg/sg.hpp"

namespace {
int helper(int x) { return x * 2 + 1; }
} // namespace

void build(sg::kernel_handle &kh, sg::mem_handle<int> in, sg::mem_handle<int> out) {
  auto x = kh.source(in, sg::shape<1>({64}));
  auto tiles = kh.tile(x, {4});
  auto work = kh.work_tiles({16}, {1}, 4);
  auto y = kh.map_work(work, [=](auto unit) {
    auto tile = tiles[unit.tile()];
    auto run = sg::flatten(tile);
    return unit.run(run, [](auto item, int v) {
      int local = 0;
      local += helper(v);
      ++local;
      int tmp[2] = {1, 2};
      tmp[1] = local;
      return tmp[1];
    });
  });
  kh.evaluate(out, kh.join(kh.retile(y, {4})));
}

void build_items(sg::kernel_handle &kh, sg::mem_handle<int> in, sg::mem_handle<int> out) {
  auto x = kh.source(in, sg::shape<1>({8}));
  auto items = kh.work_items({8});
  auto y = kh.map_items(items, [=](auto item) {
    auto v = x[item.id()];
    return item.compute(v, [](int a) { return helper(a) + 1; });
  });
  kh.evaluate(out, y);
}
