// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh) {
  auto a = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); });
  auto m = kh.map_groups<sg::mapping::global>(a, [](sycl::id<1> g) {
    std::size_t k = 0;
    for (std::size_t i = 0; i < 2; ++i)
      k += g[0];
    return sg::single<1>{sycl::id<1>(k)};
  });
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), m,
                  [](sg::item<1>, sg::group_iterator<int, 1, sg::mapping::global> x) { return x.size(); });
}
// CHECK: error: sg: loop:
// CHECK: note: in the body defined here
