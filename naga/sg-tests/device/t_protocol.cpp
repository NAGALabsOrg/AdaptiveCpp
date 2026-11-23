#include "common.hpp"

using namespace sgt;

namespace {
using G1 = sg::group_iterator<int, 1, sg::mapping::global>;

inline auto leaf(sg::kernel_handle &kh, std::size_t groups) {
  return kh.parallel_for(sycl::range<1>(groups * 4), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); });
}
} // namespace

BOOST_AUTO_TEST_SUITE(protocol)

BOOST_AUTO_TEST_CASE(four_to_one_global) {
  const auto got = run_one<int, 1>(sycl::range<1>(16), [](sg::kernel_handle &kh) {
    auto a = leaf(kh, 16);
    auto m = kh.map_groups<sg::mapping::global>(a, [](sycl::id<1> g) {
      return sg::block<1>{sycl::id<1>(g[0] * 4), sycl::range<1>(4)};
    });
    return kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), m, [](sg::item<1> it, G1 a) {
      const sycl::id<1> l(it.local_id()[0]);
      return a[0][l] + a[1][l] + a[2][l] + a[3][l];
    });
  });
  for (int g = 0; g < 4; ++g)
    for (int l = 0; l < 4; ++l) {
      int acc = 0;
      for (int k = 0; k < 4; ++k)
        acc += (4 * g + k) * 4 + l;
      BOOST_REQUIRE_EQUAL(got[g * 4 + l], acc);
    }
}

BOOST_AUTO_TEST_CASE(layers_8_16_32_each_reading_all_previous) {
  const auto got = run_one<int, 1>(sycl::range<1>(128), [](sg::kernel_handle &kh) {
    auto l1 = leaf(kh, 8);
    auto m1 = kh.map_groups<sg::mapping::global>(l1, [](sycl::id<1>) { return sg::block<1>{sycl::id<1>(0), sycl::range<1>(8)}; });
    auto l2 = kh.parallel_for(sycl::range<1>(64), sycl::range<1>(4), m1, [](sg::item<1> it, G1 a) {
      const sycl::id<1> l(it.local_id()[0]);
      return a[0][l] + a[a.size() - 1][l];
    });
    auto m2 = kh.map_groups<sg::mapping::global>(l2, [](sycl::id<1>) { return sg::block<1>{sycl::id<1>(0), sycl::range<1>(16)}; });
    return kh.parallel_for(sycl::range<1>(128), sycl::range<1>(4), m2, [](sg::item<1> it, G1 a) {
      const sycl::id<1> l(it.local_id()[0]);
      return a[0][l] + a[a.size() - 1][l];
    });
  });
  for (int g = 0; g < 32; ++g)
    for (int l = 0; l < 4; ++l) {
      const int v = (l + 28 + l) + (l + 28 + l);
      BOOST_REQUIRE_EQUAL(got[g * 4 + l], v);
    }
}

BOOST_AUTO_TEST_CASE(diamond_repeated) {
  for (int rep = 0; rep < 20; ++rep) {
    const auto got = run_one<int, 1>(sycl::range<1>(16), [](sg::kernel_handle &kh) {
      auto a = leaf(kh, 4);
      auto ma = kh.map_groups<sg::mapping::global>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
      auto b = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), ma, [](sg::item<1> it, G1 a) {
        return a[0][sycl::id<1>(it.local_id()[0])] + 1;
      });
      auto c = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), ma, [](sg::item<1> it, G1 a) {
        return a[0][sycl::id<1>(it.local_id()[0])] * 2;
      });
      auto mb = kh.map_groups<sg::mapping::global>(b, [](sycl::id<1> g) { return sg::single<1>{g}; });
      auto mc = kh.map_groups<sg::mapping::global>(c, [](sycl::id<1> g) { return sg::single<1>{g}; });
      return kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), mb, mc, [](sg::item<1> it, G1 b, G1 c) {
        const sycl::id<1> l(it.local_id()[0]);
        return b[0][l] + c[0][l];
      });
    });
    for (int i = 0; i < 16; ++i)
      BOOST_REQUIRE_EQUAL(got[i], 3 * i + 1);
  }
}

BOOST_AUTO_TEST_CASE(fan_in_of_256) {
  const auto got = run_one<int, 1>(sycl::range<1>(4), [](sg::kernel_handle &kh) {
    auto a = leaf(kh, 256);
    auto m = kh.map_groups<sg::mapping::global>(a, [](sycl::id<1>) { return sg::block<1>{sycl::id<1>(0), sycl::range<1>(256)}; });
    return kh.parallel_for(sycl::range<1>(4), sycl::range<1>(4), m, [](sg::item<1> it, G1 a) {
      const sycl::id<1> l(it.local_id()[0]);
      return a[0][l] + a[255][l];
    });
  });
  for (int l = 0; l < 4; ++l)
    BOOST_REQUIRE_EQUAL(got[l], l + (255 * 4 + l));
}

BOOST_AUTO_TEST_CASE(many_native_groups_few_shared_layers) {
  const std::size_t n = 100000;
  const auto got = run_one<int, 1>(sycl::range<1>(n), [n](sg::kernel_handle &kh) {
    auto s = kh.parallel_for(sycl::range<1>(2), sycl::range<1>(1), [](sg::item<1> it) { return int(it.id()[0]) * 10; });
    auto m = kh.map_groups<sg::mapping::global>(s, [](sycl::id<1> g) { return sg::single<1>{sycl::id<1>(g[0] % 2)}; });
    return kh.parallel_for(sycl::range<1>(n), sycl::range<1>(1), m, [](sg::item<1> it, G1 a) {
      return a[0][sycl::id<1>(0)] + int(it.id()[0]);
    });
  });
  for (std::size_t i = 0; i < n; ++i)
    BOOST_REQUIRE_EQUAL(got[i], int(i % 2) * 10 + int(i));
}

BOOST_AUTO_TEST_SUITE_END()
