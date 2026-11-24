// RUN: %acpp %s
#include "hipSYCL/sg/sg.hpp"

#include <algorithm>
#include <numeric>

void build(sg::kernel_handle &kh, sg::mem_handle<int> in, sg::mem_handle<int> out) {
  auto x = kh.source(in, sg::shape<1>({64}));
  auto tiles = kh.tile(x, {4});
  auto work = kh.work_tiles({16}, {1}, 4);
  auto sums = kh.map_work(work, [=](auto unit) {
    auto tile = tiles[unit.tile()];
    auto run = sg::flatten(tile);
    return unit.run(run, [](auto item, int v) { return sg::reduce(item, v, sycl::plus<>{}); });
  });
  kh.evaluate(out, kh.join(sums));
}

void build_items(sg::kernel_handle &kh, sg::mem_handle<int> in, sg::mem_handle<int> out) {
  auto x = kh.source(in, sg::shape<1>({8}));
  auto items = kh.work_items({8});
  auto y = kh.map_items(items, [=](auto item) {
    auto v = x[item.id()];
    return item.compute(v, [](int a) {
      int w[8];
      std::iota(w, w + 8, a);
      std::reverse(w, w + 8);
      return *std::max_element(w, w + 8);
    });
  });
  kh.evaluate(out, y);
}
