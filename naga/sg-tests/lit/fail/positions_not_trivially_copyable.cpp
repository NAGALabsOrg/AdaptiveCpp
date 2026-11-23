// RUN: not %acpp %s 2>&1 | FileCheck %s
#include "hipSYCL/sg/sg.hpp"

struct NT {
  NT() {}
  NT(const NT &) {}
  sg::single<1> operator()(sycl::id<1> g) const { return {g}; }
};

void build(sg::kernel_handle &kh) {
  auto a = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); });
  auto m = kh.map_groups<sg::mapping::local>(a, NT{});
  (void)m;
}
// CHECK: error: static assertion failed{{.*}}trivially copyable
