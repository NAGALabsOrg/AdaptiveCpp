// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh) {
  auto a = kh.parallel_for(sycl::range<2>(8, 8), sycl::range<2>(4, 4), [](sg::item<2> it) { return int(it.id()[0]); });
  auto m = kh.map_groups<sg::mapping::local>(a, [](sycl::id<2> g) { return sg::single<2>{g}; });
  kh.parallel_for(sycl::range<1>(2), m, [](sg::group<1>, sg::group_iterator<int, 2, sg::mapping::local> x) {
    return x[0];
  });
}
// CHECK: error: static assertion failed{{.*}}rank <=
