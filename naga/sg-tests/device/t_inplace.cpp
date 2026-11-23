#include "common.hpp"

#include <functional>

using namespace sgt;

namespace {
using G1 = sg::group_iterator<int, 1, sg::mapping::global>;

// Two definitions contribute to one inplace result of 16 elements.
template <class T, class Op, class Ref> void contribute(T init, Ref ref) {
  sg::graph g(opts());
  auto h = g.create_mem_handle<T>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<T>, 1> out(sg::inplace(h, Op{}, init), sg::shape<1>(sycl::range<1>(16)));
    gh.define_kernel([out](sg::kernel_handle &kh) {
      auto a = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4),
                               [](sg::item<1> it) { return static_cast<T>(it.id()[0] % 5 + 1); });
      auto b = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4),
                               [](sg::item<1> it) { return static_cast<T>(it.id()[0] % 3 + 2); });
      out = kh.evaluate(a);
      out = kh.evaluate(b);
    });
  });
  g.run();
  const auto r = read_host(g, dev(), h, sycl::range<1>(16));
  for (int i = 0; i < 16; ++i) {
    const T v1 = static_cast<T>(i % 5 + 1), v2 = static_cast<T>(i % 3 + 2);
    BOOST_REQUIRE(r[i] == ref(ref(init, v1), v2));
  }
}
} // namespace

BOOST_AUTO_TEST_SUITE(inplace_suite)

BOOST_AUTO_TEST_CASE(integer_ops) {
  contribute<std::int32_t, sycl::plus<>>(0, [](auto a, auto b) { return a + b; });
  contribute<std::uint32_t, sycl::plus<>>(0u, [](auto a, auto b) { return a + b; });
  contribute<std::int64_t, sycl::plus<>>(0, [](auto a, auto b) { return a + b; });
  contribute<std::uint64_t, sycl::plus<>>(0u, [](auto a, auto b) { return a + b; });
  contribute<std::int32_t, sycl::minimum<>>(100, [](auto a, auto b) { return std::min(a, b); });
  contribute<std::uint32_t, sycl::maximum<>>(0u, [](auto a, auto b) { return std::max(a, b); });
  contribute<std::int64_t, sycl::minimum<>>(100, [](auto a, auto b) { return std::min(a, b); });
  contribute<std::uint64_t, sycl::maximum<>>(0u, [](auto a, auto b) { return std::max(a, b); });
  contribute<std::uint32_t, sycl::bit_or<>>(0u, [](auto a, auto b) { return a | b; });
  contribute<std::uint32_t, sycl::bit_and<>>(0xffu, [](auto a, auto b) { return a & b; });
  contribute<std::int64_t, sycl::bit_xor<>>(0, [](auto a, auto b) { return a ^ b; });
}

BOOST_AUTO_TEST_CASE(floating_point) {
  contribute<float, sycl::plus<>>(0.f, [](auto a, auto b) { return a + b; });
  contribute<double, sycl::plus<>>(0.0, [](auto a, auto b) { return a + b; });
  contribute<float, sycl::minimum<>>(100.f, [](auto a, auto b) { return std::min(a, b); });
  contribute<double, sycl::maximum<>>(0.0, [](auto a, auto b) { return std::max(a, b); });
}

BOOST_AUTO_TEST_CASE(compare_exchange_path) {
  contribute<float, std::multiplies<>>(1.f, [](auto a, auto b) { return a * b; });
  contribute<double, std::multiplies<>>(1.0, [](auto a, auto b) { return a * b; });
  contribute<std::int8_t, std::multiplies<>>(1, [](auto a, auto b) { return static_cast<std::int8_t>(a * b); });
  contribute<std::int16_t, std::multiplies<>>(1, [](auto a, auto b) { return static_cast<std::int16_t>(a * b); });
  contribute<std::uint8_t, sycl::plus<>>(0, [](auto a, auto b) { return static_cast<std::uint8_t>(a + b); });
}

BOOST_AUTO_TEST_CASE(initializer_once_per_group_and_reuse_continues) {
  sg::graph g(opts());
  auto h = g.create_mem_handle<int>();
  auto contribute_value = [&](int v, bool reuse) {
    g.submit(dev(), [&, v, reuse](sg::submit_handle gh) {
      auto make = [&](auto spec) {
        sg::array<sg::result<int>, 1> out(spec, sg::shape<1>(sycl::range<1>(8)));
        gh.define_kernel([out, v](sg::kernel_handle &kh) {
          out = kh.evaluate(kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [v](sg::item<1>) { return v; }));
        });
      };
      if (reuse)
        make(sg::inplace(h, sycl::plus<>{}, sg::reuse));
      else
        make(sg::inplace(h, sycl::plus<>{}, 0));
    });
  };
  contribute_value(1, false);
  contribute_value(2, false); // same group: no second initializer
  g.run();
  auto r = read_host(g, dev(), h, sycl::range<1>(8)); // the reader closes the group
  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_EQUAL(r[i], 3);
  contribute_value(4, true); // continues from 3
  g.run();
  r = read_host(g, dev(), h, sycl::range<1>(8));
  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_EQUAL(r[i], 7);
}

BOOST_AUTO_TEST_CASE(read_by_a_later_kernel) {
  sg::graph g(opts());
  auto h = g.create_mem_handle<int>(), o = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> out(sg::inplace(h, sycl::plus<>{}, 10), sg::shape<1>(sycl::range<1>(8)));
    gh.define_kernel([out](sg::kernel_handle &kh) {
      out = kh.evaluate(kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); }));
    });
  });
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<int, 1> in(h, sg::shape<1>(sycl::range<1>(8)));
    sg::array<sg::result<int>, 1> out(o, sg::shape<1>(sycl::range<1>(8)));
    gh.define_kernel([in, out](sg::kernel_handle &kh) {
      out = kh.evaluate(kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [in](sg::item<1> it) { return in[it.id()] * 2; }));
    });
  });
  g.run();
  const auto r = read_host(g, dev(), o, sycl::range<1>(8));
  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_EQUAL(r[i], (10 + i) * 2);
}

BOOST_AUTO_TEST_CASE(inplace_never_backs_a_global_mapping) {
  sg::graph g(opts());
  auto hd = g.create_mem_handle<int>(), he = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> od(sg::inplace(hd, sycl::plus<>{}, 0), sg::shape<1>(sycl::range<1>(16)));
    sg::array<sg::result<int>, 1> oe(he, sg::shape<1>(sycl::range<1>(16)));
    gh.define_kernel([od, oe](sg::kernel_handle &kh) {
      auto d = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); });
      auto m = kh.map_groups<sg::mapping::global>(d, [](sycl::id<1> g) { return sg::single<1>{g}; });
      auto e = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), m, [](sg::item<1> it, G1 a) {
        return a[0][sycl::id<1>(it.local_id()[0])] + 1000;
      });
      od = kh.evaluate(d);
      oe = kh.evaluate(e);
    });
  });
  g.run();
  const auto d = read_host(g, dev(), hd, sycl::range<1>(16));
  const auto e = read_host(g, dev(), he, sycl::range<1>(16));
  for (int i = 0; i < 16; ++i) {
    BOOST_REQUIRE_EQUAL(d[i], i);
    BOOST_REQUIRE_EQUAL(e[i], i + 1000);
  }
}

BOOST_AUTO_TEST_CASE(inplace_from_a_group_body) {
  sg::graph g(opts());
  auto h = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> out(sg::inplace(h, sycl::plus<>{}, 100), sg::shape<1>(sycl::range<1>(4)));
    gh.define_kernel([out](sg::kernel_handle &kh) {
      auto a = kh.parallel_for(sycl::range<1>(32), sycl::range<1>(8), [](sg::item<1>) { return 1; });
      auto m = kh.map_groups<sg::mapping::local>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
      out = kh.evaluate(kh.parallel_for(sycl::range<1>(4), m, [](sg::group<1>, sg::group_iterator<int, 1, sg::mapping::local> a) {
        return sg::group_reduce<1>(a[0], 0, sycl::plus<>{});
      }));
    });
  });
  g.run();
  const auto r = read_host(g, dev(), h, sycl::range<1>(4));
  for (int i = 0; i < 4; ++i)
    BOOST_REQUIRE_EQUAL(r[i], 108);
}

BOOST_AUTO_TEST_SUITE_END()
