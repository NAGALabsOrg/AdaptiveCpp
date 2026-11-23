// RUN: %acpp %s
#include "hipSYCL/sg/sg.hpp"

void build(sg::kernel_handle &kh) {
  auto d = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); });
  for (int k = 0; k < 4; ++k) {
    auto m = kh.map_groups<sg::mapping::global>(d, [](sycl::id<1> g) { return sg::single<1>{g}; });
    d = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), m,
                        [](sg::item<1> it, sg::group_iterator<int, 1, sg::mapping::global> a) {
                          return a[0][sycl::id<1>(it.local_id()[0])] + 1;
                        });
  }
  int i = 0;
  while (i < 3)
    ++i;
}
