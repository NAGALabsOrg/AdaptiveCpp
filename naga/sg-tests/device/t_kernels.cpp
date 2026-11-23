#include "common.hpp"

#include <algorithm>

using namespace sgt;

namespace {
struct S16 {
  std::int64_t a, b;
  bool operator==(const S16 &o) const { return a == o.a && b == o.b; }
};
struct S512 {
  std::int32_t v[128];
  bool operator==(const S512 &o) const { return std::equal(v, v + 128, o.v); }
};
} // namespace

BOOST_AUTO_TEST_SUITE(item_kernels)

BOOST_AUTO_TEST_CASE(rank1_group_shapes) {
  for (std::size_t gs : {1u, 4u, 16u, 32u}) {
    const auto got = run_one<int, 1>(sycl::range<1>(32), [gs](sg::kernel_handle &kh) {
      return kh.parallel_for(sycl::range<1>(32), sycl::range<1>(gs),
                             [](sg::item<1> it) { return int(it.id()[0]) * 2; });
    });
    for (int i = 0; i < 32; ++i)
      BOOST_REQUIRE_EQUAL(got[i], 2 * i);
  }
}

BOOST_AUTO_TEST_CASE(rank2_row_major_with_non_square_groups) {
  const auto got = run_one<int, 2>(sycl::range<2>(6, 8), [](sg::kernel_handle &kh) {
    return kh.parallel_for(sycl::range<2>(6, 8), sycl::range<2>(2, 4), [](sg::item<2> it) {
      return int(it.id()[0] * 100 + it.id()[1]);
    });
  });
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 8; ++c)
      BOOST_REQUIRE_EQUAL(got[r * 8 + c], r * 100 + c);
}

BOOST_AUTO_TEST_CASE(rank3) {
  const auto got = run_one<int, 3>(sycl::range<3>(4, 6, 8), [](sg::kernel_handle &kh) {
    return kh.parallel_for(sycl::range<3>(4, 6, 8), sycl::range<3>(2, 3, 4), [](sg::item<3> it) {
      return int(it.id()[0] * 10000 + it.id()[1] * 100 + it.id()[2]);
    });
  });
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 6; ++b)
      for (int c = 0; c < 8; ++c)
        BOOST_REQUIRE_EQUAL(got[(a * 6 + b) * 8 + c], a * 10000 + b * 100 + c);
}

BOOST_AUTO_TEST_CASE(item_properties) {
  const auto got = run_one<int, 1>(sycl::range<1>(16), [](sg::kernel_handle &kh) {
    return kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), [](sg::item<1> it) {
      return int(it.group_id()[0] * 1000 + it.local_id()[0] * 100 + it.range()[0] * 10 + it.group_shape()[0]);
    });
  });
  for (int i = 0; i < 16; ++i)
    BOOST_REQUIRE_EQUAL(got[i], (i / 4) * 1000 + (i % 4) * 100 + 16 * 10 + 4);
}

template <class T> void typed() {
  const auto got = run_one<T, 1>(sycl::range<1>(16), [](sg::kernel_handle &kh) {
    return kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4),
                           [](sg::item<1> it) { return static_cast<T>(it.id()[0] + 1); });
  });
  for (int i = 0; i < 16; ++i)
    BOOST_REQUIRE(got[i] == static_cast<T>(i + 1));
}
BOOST_AUTO_TEST_CASE(element_types) {
  typed<std::int8_t>();
  typed<std::uint8_t>();
  typed<std::int16_t>();
  typed<std::int32_t>();
  typed<std::int64_t>();
  typed<unsigned>();
  typed<float>();
  typed<double>();
}

BOOST_AUTO_TEST_CASE(struct_elements) {
  const auto a = run_one<S16, 1>(sycl::range<1>(8), [](sg::kernel_handle &kh) {
    return kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) {
      return S16{std::int64_t(it.id()[0]), std::int64_t(it.id()[0]) * 3};
    });
  });
  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE((a[i] == S16{i, i * 3}));
  const auto b = run_one<S512, 1>(sycl::range<1>(8), [](sg::kernel_handle &kh) {
    return kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) {
      S512 s;
      std::iota(s.v, s.v + 128, int(it.id()[0]) * 1000);
      return s;
    });
  });
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 128; ++k)
      BOOST_REQUIRE_EQUAL(b[i].v[k], i * 1000 + k);
}

BOOST_AUTO_TEST_CASE(two_results_in_one_kernel) {
  sg::graph g(opts());
  auto h1 = g.create_mem_handle<int>();
  auto h2 = g.create_mem_handle<float>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> o1(h1, sg::shape<1>(sycl::range<1>(8)));
    sg::array<sg::result<float>, 1> o2(h2, sg::shape<1>(sycl::range<1>(8)));
    gh.define_kernel([o1, o2](sg::kernel_handle &kh) {
      auto a = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); });
      auto b = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return float(it.id()[0]) * 0.5f; });
      o1 = kh.evaluate(a);
      o2 = kh.evaluate(b);
    });
  });
  g.run();
  const auto r1 = read_host(g, dev(), h1, sycl::range<1>(8));
  const auto r2 = read_host(g, dev(), h2, sycl::range<1>(8));
  for (int i = 0; i < 8; ++i) {
    BOOST_REQUIRE_EQUAL(r1[i], i);
    BOOST_REQUIRE_EQUAL(r2[i], i * 0.5f);
  }
}

BOOST_AUTO_TEST_CASE(slice_input) {
  sg::graph g(opts());
  auto in = g.create_mem_handle<int>(), out = g.create_mem_handle<int>();
  write_host(g, dev(), in, sycl::range<1>(16), iota_vec<int>(16));
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<int, 1> a(in.slice(4, 8), sg::shape<1>(sycl::range<1>(8)));
    sg::array<sg::result<int>, 1> o(out, sg::shape<1>(sycl::range<1>(8)));
    gh.define_kernel([a, o](sg::kernel_handle &kh) {
      o = kh.evaluate(kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4),
                                      [a](sg::item<1> it) { return a[it.id()] * 2; }));
    });
  });
  g.run();
  const auto r = read_host(g, dev(), out, sycl::range<1>(8));
  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_EQUAL(r[i], (i + 4) * 2);
}

BOOST_AUTO_TEST_CASE(zero_range_then_later_nodes) {
  sg::graph g(opts());
  auto e = g.create_mem_handle<int>(), h = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> o(e, sg::shape<1>(sycl::range<1>(0)));
    gh.define_kernel([o](sg::kernel_handle &kh) {
      o = kh.evaluate(kh.parallel_for(sycl::range<1>(0), sycl::range<1>(4), [](sg::item<1>) { return 1; }));
    });
  });
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> o(h, sg::shape<1>(sycl::range<1>(8)));
    gh.define_kernel([o](sg::kernel_handle &kh) {
      o = kh.evaluate(kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4),
                                      [](sg::item<1> it) { return int(it.id()[0]) + 1; }));
    });
  });
  g.run();
  const auto r = read_host(g, dev(), h, sycl::range<1>(8));
  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_EQUAL(r[i], i + 1);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(mappings)

namespace {
using L1 = sg::group_iterator<int, 1, sg::mapping::local>;
using G1 = sg::group_iterator<int, 1, sg::mapping::global>;
using I1 = sg::group_iterator<int, 1, sg::mapping::item>;

template <class Build> std::vector<int> run16(Build b) { return run_one<int, 1>(sycl::range<1>(16), b); }

inline auto make_a(sg::kernel_handle &kh) {
  return kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); });
}
} // namespace

BOOST_AUTO_TEST_CASE(local_single) {
  const auto got = run16([](sg::kernel_handle &kh) {
    auto a = make_a(kh);
    auto m = kh.map_groups<sg::mapping::local>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
    return kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), m, [](sg::item<1> it, L1 a) {
      return a[0][sycl::id<1>(it.local_id()[0])] * 2;
    });
  });
  for (int i = 0; i < 16; ++i)
    BOOST_REQUIRE_EQUAL(got[i], 2 * i);
}

BOOST_AUTO_TEST_CASE(local_ids_slot_order) {
  const auto got = run16([](sg::kernel_handle &kh) {
    auto a = make_a(kh);
    auto m = kh.map_groups<sg::mapping::local>(a, [](sycl::id<1> g) {
      return sg::ids<1, 3>{{sycl::id<1>(g[0]), sycl::id<1>((g[0] + 1) % 4), sycl::id<1>((g[0] + 2) % 4)}};
    });
    return kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), m, [](sg::item<1> it, L1 a) {
      const sycl::id<1> l(it.local_id()[0]);
      return a[0][l] + 10 * a[1][l] + 100 * a[2][l];
    });
  });
  for (int g = 0; g < 4; ++g)
    for (int l = 0; l < 4; ++l)
      BOOST_REQUIRE_EQUAL(got[g * 4 + l], (4 * g + l) + 10 * (4 * ((g + 1) % 4) + l) + 100 * (4 * ((g + 2) % 4) + l));
}

BOOST_AUTO_TEST_CASE(local_block_counts) {
  for (int count : {1, 3, 8}) {
    const auto got = run_one<int, 1>(sycl::range<1>(32), [count](sg::kernel_handle &kh) {
      auto b = kh.parallel_for(sycl::range<1>(32), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); });
      auto m = kh.map_groups<sg::mapping::local>(b, [count](sycl::id<1>) {
        return sg::block<1>{sycl::id<1>(0), sycl::range<1>(std::size_t(count))};
      });
      return kh.parallel_for(sycl::range<1>(32), sycl::range<1>(4), m, [](sg::item<1> it, L1 a) {
        const sycl::id<1> l(it.local_id()[0]);
        return a[0][l] + 100 * a[a.size() - 1][l];
      });
    });
    for (int g = 0; g < 8; ++g)
      for (int l = 0; l < 4; ++l)
        BOOST_REQUIRE_EQUAL(got[g * 4 + l], l + 100 * (4 * (count - 1) + l));
  }
}

BOOST_AUTO_TEST_CASE(index_map_from_data) {
  sg::graph g(opts());
  auto idx = g.create_mem_handle<int>(), out = g.create_mem_handle<int>();
  write_host(g, dev(), idx, sycl::range<1>(8), std::vector<int>{3, 1, 2, 0, 0, 2, 1, 3});
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<int, 1> ix(idx, sg::shape<1>(sycl::range<1>(8)));
    sg::array<sg::result<int>, 1> o(out, sg::shape<1>(sycl::range<1>(8)));
    gh.define_kernel([ix, o](sg::kernel_handle &kh) {
      auto a = make_a(kh);
      auto m = kh.map_groups<sg::mapping::local>(a, [ix](sycl::id<1> g) {
        return sg::index_map<1, int>{ix, g[0] * 2, 2};
      });
      o = kh.evaluate(kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), m, [](sg::item<1> it, L1 a) {
        const sycl::id<1> l(it.local_id()[0]);
        return a[0][l] + 10 * a[1][l];
      }));
    });
  });
  g.run();
  const auto r = read_host(g, dev(), out, sycl::range<1>(8));
  const int ids[8] = {3, 1, 2, 0, 0, 2, 1, 3};
  for (int gi = 0; gi < 2; ++gi)
    for (int l = 0; l < 4; ++l)
      BOOST_REQUIRE_EQUAL(r[gi * 4 + l], (4 * ids[gi * 2] + l) + 10 * (4 * ids[gi * 2 + 1] + l));
}

BOOST_AUTO_TEST_CASE(item_chain_of_three) {
  const auto got = run16([](sg::kernel_handle &kh) {
    auto a = make_a(kh);
    auto ma = kh.map_groups<sg::mapping::item>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
    auto b = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), ma, [](sg::item<1>, I1 a) { return a[0] + 1; });
    auto mb = kh.map_groups<sg::mapping::item>(b, [](sycl::id<1> g) { return sg::single<1>{g}; });
    auto c = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), mb, [](sg::item<1>, I1 b) { return b[0] * 2; });
    auto mc = kh.map_groups<sg::mapping::item>(c, [](sycl::id<1> g) { return sg::single<1>{g}; });
    return kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), mc, [](sg::item<1>, I1 c) { return c[0] - 3; });
  });
  for (int i = 0; i < 16; ++i)
    BOOST_REQUIRE_EQUAL(got[i], (i + 1) * 2 - 3);
}

BOOST_AUTO_TEST_CASE(local_between_different_group_shapes) {
  const auto got = run_one<int, 1>(sycl::range<1>(16), [](sg::kernel_handle &kh) {
    auto a = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(2), [](sg::item<1> it) { return int(it.id()[0]); });
    auto m = kh.map_groups<sg::mapping::local>(a, [](sycl::id<1> g) {
      return sg::block<1>{sycl::id<1>(g[0] * 2), sycl::range<1>(2)};
    });
    return kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), m, [](sg::item<1> it, L1 a) {
      const std::size_t l = it.local_id()[0];
      return a[l / 2][sycl::id<1>(l % 2)];
    });
  });
  for (int i = 0; i < 16; ++i)
    BOOST_REQUIRE_EQUAL(got[i], i);
}

BOOST_AUTO_TEST_CASE(global_far_groups) {
  const auto got = run16([](sg::kernel_handle &kh) {
    auto a = make_a(kh);
    auto m = kh.map_groups<sg::mapping::global>(a, [](sycl::id<1> g) { return sg::single<1>{sycl::id<1>(3 - g[0])}; });
    return kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), m, [](sg::item<1> it, G1 a) {
      return a[0][sycl::id<1>(it.local_id()[0])];
    });
  });
  for (int g = 0; g < 4; ++g)
    for (int l = 0; l < 4; ++l)
      BOOST_REQUIRE_EQUAL(got[g * 4 + l], 4 * (3 - g) + l);
}

BOOST_AUTO_TEST_CASE(global_chain_of_30) {
  const auto got = run16([](sg::kernel_handle &kh) {
    auto d = make_a(kh);
    for (int k = 0; k < 30; ++k) {
      auto m = kh.map_groups<sg::mapping::global>(d, [](sycl::id<1> g) { return sg::single<1>{g}; });
      d = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), m, [](sg::item<1> it, G1 a) {
        return a[0][sycl::id<1>(it.local_id()[0])] + 1;
      });
    }
    return d;
  });
  for (int i = 0; i < 16; ++i)
    BOOST_REQUIRE_EQUAL(got[i], i + 30);
}

BOOST_AUTO_TEST_CASE(same_lambda_at_ten_layers) {
  const auto got = run16([](sg::kernel_handle &kh) {
    auto d = make_a(kh);
    for (int k = 0; k < 10; ++k) {
      auto m = kh.map_groups<sg::mapping::global>(d, [](sycl::id<1> g) { return sg::single<1>{g}; });
      d = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), m, [](sg::item<1> it, G1 a) {
        return a[0][sycl::id<1>(it.local_id()[0])] * 2 % 1000;
      });
    }
    return d;
  });
  for (int i = 0; i < 16; ++i) {
    int v = i;
    for (int k = 0; k < 10; ++k)
      v = v * 2 % 1000;
    BOOST_REQUIRE_EQUAL(got[i], v);
  }
}

BOOST_AUTO_TEST_CASE(global_source_also_evaluated_and_shared) {
  sg::graph g(opts());
  auto ha = g.create_mem_handle<int>(), hb = g.create_mem_handle<int>(), hc = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> oa(ha, sg::shape<1>(sycl::range<1>(16)));
    sg::array<sg::result<int>, 1> ob(hb, sg::shape<1>(sycl::range<1>(16)));
    sg::array<sg::result<int>, 1> oc(hc, sg::shape<1>(sycl::range<1>(16)));
    gh.define_kernel([oa, ob, oc](sg::kernel_handle &kh) {
      auto a = make_a(kh);
      auto m = kh.map_groups<sg::mapping::global>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
      auto b = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), m, [](sg::item<1> it, G1 a) {
        return a[0][sycl::id<1>(it.local_id()[0])] + 100;
      });
      auto c = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), m, [](sg::item<1> it, G1 a) {
        return a[0][sycl::id<1>(it.local_id()[0])] + 200;
      });
      oa = kh.evaluate(a);
      ob = kh.evaluate(b);
      oc = kh.evaluate(c);
    });
  });
  g.run();
  const auto a = read_host(g, dev(), ha, sycl::range<1>(16));
  const auto b = read_host(g, dev(), hb, sycl::range<1>(16));
  const auto c = read_host(g, dev(), hc, sycl::range<1>(16));
  for (int i = 0; i < 16; ++i) {
    BOOST_REQUIRE_EQUAL(a[i], i);
    BOOST_REQUIRE_EQUAL(b[i], i + 100);
    BOOST_REQUIRE_EQUAL(c[i], i + 200);
  }
}

BOOST_AUTO_TEST_CASE(a_to_f) {
  sg::graph g(opts());
  auto hb = g.create_mem_handle<int>(), hc = g.create_mem_handle<int>(), hd = g.create_mem_handle<int>(),
       hf = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> ob(hb, sg::shape<1>(sycl::range<1>(16))), oc(hc, sg::shape<1>(sycl::range<1>(16))),
        od(hd, sg::shape<1>(sycl::range<1>(16))), of(hf, sg::shape<1>(sycl::range<1>(16)));
    gh.define_kernel([ob, oc, od, of](sg::kernel_handle &kh) {
      const sycl::range<1> r(16), gs(4);
      auto A = make_a(kh);
      auto mA = kh.map_groups<sg::mapping::local>(A, [](sycl::id<1> g) { return sg::single<1>{g}; });
      auto B = kh.parallel_for(r, gs, mA, [](sg::item<1> it, L1 a) { return a[0][sycl::id<1>(it.local_id()[0])] + 1; });
      auto C = kh.parallel_for(r, gs, mA, [](sg::item<1> it, L1 a) { return a[0][sycl::id<1>(it.local_id()[0])] * 2; });
      auto mB = kh.map_groups<sg::mapping::local>(B, [](sycl::id<1> g) { return sg::single<1>{g}; });
      auto mC = kh.map_groups<sg::mapping::local>(C, [](sycl::id<1> g) { return sg::single<1>{g}; });
      auto D = kh.parallel_for(r, gs, mB, mC, [](sg::item<1> it, L1 b, L1 c) {
        const sycl::id<1> l(it.local_id()[0]);
        return b[0][l] + c[0][l];
      });
      auto mE = kh.map_groups<sg::mapping::global>(C, [](sycl::id<1> g) { return sg::single<1>{sycl::id<1>(3 - g[0])}; });
      auto E = kh.parallel_for(r, gs, mE, [](sg::item<1> it, G1 c) { return c[0][sycl::id<1>(it.local_id()[0])] + 1000; });
      auto mF = kh.map_groups<sg::mapping::item>(E, [](sycl::id<1> g) { return sg::single<1>{g}; });
      auto F = kh.parallel_for(r, gs, mF, [](sg::item<1>, I1 e) { return e[0] + 1; });
      ob = kh.evaluate(B);
      oc = kh.evaluate(C);
      od = kh.evaluate(D);
      of = kh.evaluate(F);
    });
  });
  g.run();
  const auto b = read_host(g, dev(), hb, sycl::range<1>(16));
  const auto c = read_host(g, dev(), hc, sycl::range<1>(16));
  const auto d = read_host(g, dev(), hd, sycl::range<1>(16));
  const auto f = read_host(g, dev(), hf, sycl::range<1>(16));
  for (int gi = 0; gi < 4; ++gi)
    for (int l = 0; l < 4; ++l) {
      const int i = 4 * gi + l, src = 4 * (3 - gi) + l;
      BOOST_REQUIRE_EQUAL(b[i], i + 1);
      BOOST_REQUIRE_EQUAL(c[i], 2 * i);
      BOOST_REQUIRE_EQUAL(d[i], 3 * i + 1);
      BOOST_REQUIRE_EQUAL(f[i], 2 * src + 1000 + 1);
    }
}

BOOST_AUTO_TEST_SUITE_END()
