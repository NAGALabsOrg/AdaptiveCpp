// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

struct MyOp {
  int operator()(int a, int b) const { return a + b; }
};

void build(sg::kernel_handle &kh) {
  auto a = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); });
  auto m = kh.map_groups<sg::mapping::local>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
  kh.parallel_for(sycl::range<1>(2), m, [](sg::group<1>, sg::group_iterator<int, 1, sg::mapping::local> x) {
    return sg::group_inclusive_scan(x[0], MyOp{});
  });
}
// CHECK: error: static assertion failed{{.*}}known identity
