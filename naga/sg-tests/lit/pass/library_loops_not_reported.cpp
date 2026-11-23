// RUN: %acpp %s
#include "hipSYCL/sg/sg.hpp"

#include <algorithm>
#include <numeric>

void build(sg::kernel_handle &kh) {
  kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) {
    int v[8];
    std::iota(v, v + 8, int(it.id()[0]));
    std::reverse(v, v + 8);
    return *std::max_element(v, v + 8);
  });
}
