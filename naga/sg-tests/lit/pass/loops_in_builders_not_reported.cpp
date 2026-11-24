// RUN: %acpp %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh, sg::mem_handle<float> in, sg::mem_handle<float> out) {
  auto step = [](float v) { return v + 1.f; };
  auto x = kh.source(in, sg::shape<1>({64}));
  auto items = kh.work_items({64});
  auto y = kh.map_items(items, [=](auto item) {
    auto v = x[item.id()];
    return item.compute(v, step);
  });
  for (int k = 1; k < 4; ++k) {
    auto prev = y.named("prev");
    y = kh.map_items(items, [=](auto item) {
      auto v = prev[item.id()];
      return item.compute(v, step);
    });
  }
  int i = 0;
  while (i < 3)
    ++i;
  kh.evaluate(out, y);
}
