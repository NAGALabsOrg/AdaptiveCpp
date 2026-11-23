// RUN: %acpp %s
#include "hipSYCL/sg/sg.hpp"

namespace {
int helper(int x) { return x * 2 + 1; }
} // namespace

void build(sg::kernel_handle &kh, sg::array<int, 1> in) {
  auto a = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [in](sg::item<1> it) {
    int local = 0;
    local += helper(int(it.id()[0]));
    ++local;
    int tmp[2] = {1, 2};
    tmp[1] = local;
    return tmp[1] + in[it.id()];
  });
  auto l = kh.map_groups<sg::mapping::local>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
  auto b = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), l, [](sg::item<1> it, sg::group_iterator<int, 1, sg::mapping::local> x) {
    return x[0][sycl::id<1>(it.local_id()[0])];
  });
  auto g = kh.map_groups<sg::mapping::global>(b, [](sycl::id<1> g) { return sg::single<1>{g}; });
  auto lb = kh.map_groups<sg::mapping::local>(b, [](sycl::id<1> g) { return sg::block<1>{g, sycl::range<1>(1)}; });
  kh.parallel_for(sycl::range<1>(2), lb, [](sg::group<1>, sg::group_iterator<int, 1, sg::mapping::local> x) {
    return sg::group_reduce<1>(x[0], 0, sycl::plus<>{}) + 1;
  });
  (void)g;
}
