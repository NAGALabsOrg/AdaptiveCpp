#include "common.hpp"

#include <algorithm>
#include <climits>

using namespace sgt;

namespace {
template <class T> T val(std::size_t i) { return static_cast<T>(static_cast<int>((i * 7) % 13) - 4); }

template <class T, class Op, class Ref> void reduce1(Ref ref) {
  const auto got = run_one<T, 1>(sycl::range<1>(4), [](sg::kernel_handle &kh) {
    auto a = kh.parallel_for(sycl::range<1>(32), sycl::range<1>(8), [](sg::item<1> it) { return val<T>(it.id()[0]); });
    auto m = kh.map_groups<sg::mapping::local>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
    return kh.parallel_for(sycl::range<1>(4), m, [](sg::group<1>, sg::group_iterator<T, 1, sg::mapping::local> a) {
      return sg::group_reduce<1>(a[0], 0, Op{});
    });
  });
  for (int g = 0; g < 4; ++g) {
    T acc = sycl::known_identity_v<Op, T>;
    for (int l = 0; l < 8; ++l)
      acc = ref(acc, val<T>(g * 8 + l));
    BOOST_REQUIRE_EQUAL(got[g], acc);
  }
}

using L2 = sg::group_iterator<int, 2, sg::mapping::local>;
using L3 = sg::group_iterator<int, 3, sg::mapping::local>;
using L1 = sg::group_iterator<int, 1, sg::mapping::local>;
using G1 = sg::group_iterator<int, 1, sg::mapping::global>;

inline int a2(int r, int c) { return (r * 8 + c) % 11 - 3; }
inline int a3(int x, int y, int z) { return ((x * 16 + y * 4 + z) * 5) % 17 - 6; }

inline auto make2(sg::kernel_handle &kh) {
  return kh.parallel_for(sycl::range<2>(8, 8), sycl::range<2>(4, 4), [](sg::item<2> it) {
    return (int(it.id()[0]) * 8 + int(it.id()[1])) % 11 - 3;
  });
}
inline auto make3(sg::kernel_handle &kh) {
  return kh.parallel_for(sycl::range<3>(4, 4, 4), sycl::range<3>(2, 2, 2), [](sg::item<3> it) {
    return ((int(it.id()[0]) * 16 + int(it.id()[1]) * 4 + int(it.id()[2])) * 5) % 17 - 6;
  });
}
inline auto single2() { return [](sycl::id<2> g) { return sg::single<2>{g}; }; }
inline auto single3() { return [](sycl::id<3> g) { return sg::single<3>{g}; }; }

template <int D> void rank2_reduce() {
  const auto got = run_one<int, 2>(sycl::range<2>(8, 2), [](sg::kernel_handle &kh) {
    auto a = make2(kh);
    auto m = kh.map_groups<sg::mapping::local>(a, single2());
    return kh.parallel_for(sycl::range<2>(2, 2), m, [](sg::group<2>, L2 a) {
      return sg::group_reduce<2>(a[0], D, sycl::plus<>{});
    });
  });
  for (int g0 = 0; g0 < 2; ++g0)
    for (int g1 = 0; g1 < 2; ++g1)
      for (int k = 0; k < 4; ++k) {
        int acc = 0;
        for (int j = 0; j < 4; ++j)
          acc += D == 1 ? a2(g0 * 4 + k, g1 * 4 + j) : a2(g0 * 4 + j, g1 * 4 + k);
        BOOST_REQUIRE_EQUAL(got[(g0 * 4 + k) * 2 + g1], acc);
      }
}

template <int D> void rank3_reduce() {
  const auto got = run_one<int, 3>(sycl::range<3>(4, 4, 2), [](sg::kernel_handle &kh) {
    auto a = make3(kh);
    auto m = kh.map_groups<sg::mapping::local>(a, single3());
    return kh.parallel_for(sycl::range<3>(2, 2, 2), m, [](sg::group<3>, L3 a) {
      return sg::group_reduce<3>(a[0], D, sycl::plus<>{});
    });
  });
  for (int g0 = 0; g0 < 2; ++g0)
    for (int g1 = 0; g1 < 2; ++g1)
      for (int g2 = 0; g2 < 2; ++g2)
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 2; ++j) {
            int acc = 0;
            for (int r = 0; r < 2; ++r) {
              int idx[3];
              int c = 0;
              for (int p = 0; p < 3; ++p)
                idx[p] = p == D ? r : (c++ == 0 ? i : j);
              acc += a3(g0 * 2 + idx[0], g1 * 2 + idx[1], g2 * 2 + idx[2]);
            }
            BOOST_REQUIRE_EQUAL(got[((g0 * 2 + i) * 4 + (g1 * 2 + j)) * 2 + g2], acc);
          }
}
} // namespace

BOOST_AUTO_TEST_SUITE(group_bodies)

BOOST_AUTO_TEST_CASE(reduce_plus_min_max_rank1) {
  reduce1<int, sycl::plus<>>([](int a, int b) { return a + b; });
  reduce1<int, sycl::minimum<>>([](int a, int b) { return std::min(a, b); });
  reduce1<int, sycl::maximum<>>([](int a, int b) { return std::max(a, b); });
  reduce1<float, sycl::plus<>>([](float a, float b) { return a + b; });
  reduce1<float, sycl::maximum<>>([](float a, float b) { return std::max(a, b); });
  reduce1<double, sycl::plus<>>([](double a, double b) { return a + b; });
  reduce1<double, sycl::minimum<>>([](double a, double b) { return std::min(a, b); });
}

BOOST_AUTO_TEST_CASE(reduce_rank2_each_dim) {
  rank2_reduce<0>();
  rank2_reduce<1>();
}

BOOST_AUTO_TEST_CASE(reduce_rank3_each_dim) {
  rank3_reduce<0>();
  rank3_reduce<1>();
  rank3_reduce<2>();
}

BOOST_AUTO_TEST_CASE(reduce_to_rank_zero) {
  const auto got = run_one<int, 3>(sycl::range<3>(2, 2, 2), [](sg::kernel_handle &kh) {
    auto a = make3(kh);
    auto m = kh.map_groups<sg::mapping::local>(a, single3());
    return kh.parallel_for(sycl::range<3>(2, 2, 2), m, [](sg::group<3>, L3 a) {
      return sg::group_reduce<1>(sg::group_reduce<2>(sg::group_reduce<3>(a[0], 1, sycl::plus<>{}), 0, sycl::plus<>{}), 0,
                                 sycl::plus<>{});
    });
  });
  for (int g0 = 0; g0 < 2; ++g0)
    for (int g1 = 0; g1 < 2; ++g1)
      for (int g2 = 0; g2 < 2; ++g2) {
        int acc = 0;
        for (int x = 0; x < 2; ++x)
          for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
              acc += a3(g0 * 2 + x, g1 * 2 + y, g2 * 2 + z);
        BOOST_REQUIRE_EQUAL(got[(g0 * 2 + g1) * 2 + g2], acc);
      }
}

BOOST_AUTO_TEST_CASE(scans_rank1) {
  const auto inc = run_one<int, 1>(sycl::range<1>(32), [](sg::kernel_handle &kh) {
    auto a = kh.parallel_for(sycl::range<1>(32), sycl::range<1>(8), [](sg::item<1> it) { return val<int>(it.id()[0]); });
    auto m = kh.map_groups<sg::mapping::local>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
    return kh.parallel_for(sycl::range<1>(4), m, [](sg::group<1>, L1 a) {
      return sg::group_inclusive_scan(a[0], sycl::plus<>{});
    });
  });
  const auto exc = run_one<int, 1>(sycl::range<1>(32), [](sg::kernel_handle &kh) {
    auto a = kh.parallel_for(sycl::range<1>(32), sycl::range<1>(8), [](sg::item<1> it) { return val<int>(it.id()[0]); });
    auto m = kh.map_groups<sg::mapping::local>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
    return kh.parallel_for(sycl::range<1>(4), m, [](sg::group<1>, L1 a) {
      return sg::group_exclusive_scan(a[0], 5, sycl::plus<>{});
    });
  });
  for (int g = 0; g < 4; ++g) {
    int acc = 0, eacc = 5;
    for (int l = 0; l < 8; ++l) {
      acc += val<int>(g * 8 + l);
      BOOST_REQUIRE_EQUAL(inc[g * 8 + l], acc);
      BOOST_REQUIRE_EQUAL(exc[g * 8 + l], eacc);
      eacc += val<int>(g * 8 + l);
    }
  }
}

BOOST_AUTO_TEST_CASE(scans_rank2_each_dim) {
  for (int dim = 0; dim < 2; ++dim) {
    const auto got = dim == 0
                         ? run_one<int, 2>(sycl::range<2>(8, 8), [](sg::kernel_handle &kh) {
                             auto a = make2(kh);
                             auto m = kh.map_groups<sg::mapping::local>(a, single2());
                             return kh.parallel_for(sycl::range<2>(2, 2), m, [](sg::group<2>, L2 a) {
                               return sg::group_inclusive_scan<2>(a[0], 0, sycl::plus<>{});
                             });
                           })
                         : run_one<int, 2>(sycl::range<2>(8, 8), [](sg::kernel_handle &kh) {
                             auto a = make2(kh);
                             auto m = kh.map_groups<sg::mapping::local>(a, single2());
                             return kh.parallel_for(sycl::range<2>(2, 2), m, [](sg::group<2>, L2 a) {
                               return sg::group_exclusive_scan<2>(a[0], 1, 0, sycl::plus<>{});
                             });
                           });
    for (int r = 0; r < 8; ++r)
      for (int c = 0; c < 8; ++c) {
        const int g0 = r / 4, g1 = c / 4, lr = r % 4, lc = c % 4;
        int acc = 0;
        if (dim == 0) {
          for (int k = 0; k <= lr; ++k)
            acc += a2(g0 * 4 + k, g1 * 4 + lc);
        } else {
          for (int k = 0; k < lc; ++k)
            acc += a2(g0 * 4 + lr, g1 * 4 + k);
        }
        BOOST_REQUIRE_EQUAL(got[r * 8 + c], acc);
      }
  }
}

BOOST_AUTO_TEST_CASE(global_view_reduce) {
  const auto got = run_one<int, 1>(sycl::range<1>(4), [](sg::kernel_handle &kh) {
    auto a = kh.parallel_for(sycl::range<1>(32), sycl::range<1>(8), [](sg::item<1> it) { return val<int>(it.id()[0]); });
    auto m = kh.map_groups<sg::mapping::global>(a, [](sycl::id<1> g) { return sg::single<1>{sycl::id<1>(3 - g[0])}; });
    return kh.parallel_for(sycl::range<1>(4), m, [](sg::group<1>, G1 a) {
      return sg::group_reduce<1>(a[0], 0, sycl::plus<>{});
    });
  });
  for (int g = 0; g < 4; ++g) {
    int acc = 0;
    for (int l = 0; l < 8; ++l)
      acc += val<int>((3 - g) * 8 + l);
    BOOST_REQUIRE_EQUAL(got[g], acc);
  }
}

BOOST_AUTO_TEST_CASE(elementwise_expressions) {
  const auto got = run_one<int, 1>(sycl::range<1>(4), [](sg::kernel_handle &kh) {
    auto a = kh.parallel_for(sycl::range<1>(32), sycl::range<1>(8), [](sg::item<1> it) { return val<int>(it.id()[0]); });
    auto m = kh.map_groups<sg::mapping::local>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
    return kh.parallel_for(sycl::range<1>(4), m, [](sg::group<1>, L1 a) {
      return sg::group_reduce<1>(a[0] * 3 + a[0], 0, sycl::plus<>{});
    });
  });
  for (int g = 0; g < 4; ++g) {
    int acc = 0;
    for (int l = 0; l < 8; ++l)
      acc += val<int>(g * 8 + l) * 4;
    BOOST_REQUIRE_EQUAL(got[g], acc);
  }
}

BOOST_AUTO_TEST_CASE(reduce_reduce_scan_composition) {
  const auto got = run_one<int, 3>(sycl::range<3>(4, 2, 2), [](sg::kernel_handle &kh) {
    auto a = make3(kh);
    auto m = kh.map_groups<sg::mapping::local>(a, single3());
    return kh.parallel_for(sycl::range<3>(2, 2, 2), m, [](sg::group<3>, L3 a) {
      return sg::group_inclusive_scan<1>(sg::group_reduce<2>(sg::group_reduce<3>(a[0], 1, sycl::plus<>{}), 0, sycl::plus<>{}), 0,
                                         sycl::plus<>{});
    });
  });
  for (int g0 = 0; g0 < 2; ++g0)
    for (int g1 = 0; g1 < 2; ++g1)
      for (int g2 = 0; g2 < 2; ++g2) {
        int r2[2] = {0, 0};
        for (int c = 0; c < 2; ++c)
          for (int x = 0; x < 2; ++x)
            for (int y = 0; y < 2; ++y)
              r2[c] += a3(g0 * 2 + x, g1 * 2 + y, g2 * 2 + c);
        BOOST_REQUIRE_EQUAL(got[((g0 * 2 + 0) * 2 + g1) * 2 + g2], r2[0]);
        BOOST_REQUIRE_EQUAL(got[((g0 * 2 + 1) * 2 + g1) * 2 + g2], r2[0] + r2[1]);
      }
}

BOOST_AUTO_TEST_CASE(fewer_outputs_than_work_group_size) {
  sg::graph g(opts());
  auto hx = g.create_mem_handle<int>(), hy = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> ox(hx, sg::shape<1>(sycl::range<1>(64))), oy(hy, sg::shape<1>(sycl::range<1>(4)));
    gh.define_kernel([ox, oy](sg::kernel_handle &kh) {
      auto big = kh.parallel_for(sycl::range<1>(64), sycl::range<1>(64), [](sg::item<1> it) { return int(it.id()[0]); });
      auto a = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), [](sg::item<1> it) { return val<int>(it.id()[0]); });
      auto m = kh.map_groups<sg::mapping::local>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
      auto s = kh.parallel_for(sycl::range<1>(4), m, [](sg::group<1>, L1 a) {
        return sg::group_reduce<1>(a[0], 0, sycl::plus<>{});
      });
      ox = kh.evaluate(big);
      oy = kh.evaluate(s);
    });
  });
  g.run();
  const auto y = read_host(g, dev(), hy, sycl::range<1>(4));
  for (int gi = 0; gi < 4; ++gi) {
    int acc = 0;
    for (int l = 0; l < 4; ++l)
      acc += val<int>(gi * 4 + l);
    BOOST_REQUIRE_EQUAL(y[gi], acc);
  }
}

BOOST_AUTO_TEST_SUITE_END()
