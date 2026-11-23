#include "common.hpp"

#include <atomic>
#include <chrono>
#include <thread>

using namespace sgt;

namespace {
// out = in * factor, as a kernel with a key
template <class K>
void scale(sg::graph &g, const sycl::device &d, sg::mem_handle<int> in, sg::mem_handle<int> out, std::size_t n,
           int factor, K key) {
  g.submit(d, [&](sg::submit_handle gh) {
    sg::array<int, 1> a(in, sg::shape<1>(sycl::range<1>(n)));
    sg::array<sg::result<int>, 1> o(out, sg::shape<1>(sycl::range<1>(n)));
    gh.define_kernel(key, [a, o, n, factor](sg::kernel_handle &kh) {
      o = kh.evaluate(kh.parallel_for(sycl::range<1>(n), sycl::range<1>(4),
                                      [a, factor](sg::item<1> it) { return a[it.id()] * factor; }));
    });
  });
}
void scale_unkeyed(sg::graph &g, const sycl::device &d, sg::mem_handle<int> in, sg::mem_handle<int> out,
                   std::size_t n, const int *factor) {
  g.submit(d, [&](sg::submit_handle gh) {
    sg::array<int, 1> a(in, sg::shape<1>(sycl::range<1>(n)));
    sg::array<sg::result<int>, 1> o(out, sg::shape<1>(sycl::range<1>(n)));
    gh.define_kernel([a, o, n, factor](sg::kernel_handle &kh) {
      const int f = *factor;
      o = kh.evaluate(kh.parallel_for(sycl::range<1>(n), sycl::range<1>(4),
                                      [a, f](sg::item<1> it) { return a[it.id()] * f; }));
    });
  });
}
} // namespace

BOOST_AUTO_TEST_SUITE(graph_runtime)

BOOST_AUTO_TEST_CASE(raw_war_waw) {
  sg::graph g(opts());
  auto in = g.create_mem_handle<int>(), out = g.create_mem_handle<int>();
  write_host(g, dev(), in, sycl::range<1>(16), iota_vec<int>(16));
  scale(g, dev(), in, out, 16, 2, std::make_tuple(0)); // RAW on in
  write_host(g, dev(), in, sycl::range<1>(16), iota_vec<int>(16, 100)); // WAR on in
  write_host(g, dev(), in, sycl::range<1>(16), iota_vec<int>(16, 200)); // WAW on in
  g.run();
  const auto o = read_host(g, dev(), out, sycl::range<1>(16));
  const auto i = read_host(g, dev(), in, sycl::range<1>(16));
  for (int k = 0; k < 16; ++k) {
    BOOST_REQUIRE_EQUAL(o[k], 2 * k);
    BOOST_REQUIRE_EQUAL(i[k], 200 + k);
  }
}

BOOST_AUTO_TEST_CASE(shared_readers_and_contents_survive_runs) {
  sg::graph g(opts());
  auto in = g.create_mem_handle<int>(), o1 = g.create_mem_handle<int>(), o2 = g.create_mem_handle<int>();
  write_host(g, dev(), in, sycl::range<1>(16), iota_vec<int>(16));
  g.run();
  scale(g, dev(), in, o1, 16, 3, std::make_tuple(0));
  scale(g, dev(), in, o2, 16, 5, std::make_tuple(1));
  g.run();
  const auto a = read_host(g, dev(), o1, sycl::range<1>(16));
  const auto b = read_host(g, dev(), o2, sycl::range<1>(16));
  for (int k = 0; k < 16; ++k) {
    BOOST_REQUIRE_EQUAL(a[k], 3 * k);
    BOOST_REQUIRE_EQUAL(b[k], 5 * k);
  }
}

BOOST_AUTO_TEST_CASE(larger_and_smaller_producers_redefine) {
  sg::graph g(opts());
  auto h = g.create_mem_handle<int>();
  for (std::size_t n : {16u, 4096u, 8u, 20000u, 4u}) {
    g.submit(dev(), [&](sg::submit_handle gh) {
      sg::array<sg::result<int>, 1> o(h, sg::shape<1>(sycl::range<1>(n)));
      gh.define_kernel([o, n](sg::kernel_handle &kh) {
        o = kh.evaluate(kh.parallel_for(sycl::range<1>(n), sycl::range<1>(4),
                                        [](sg::item<1> it) { return int(it.id()[0]) + 1; }));
      });
    });
    const auto r = read_host(g, dev(), h, sycl::range<1>(n));
    for (std::size_t k = 0; k < n; ++k)
      BOOST_REQUIRE_EQUAL(r[k], int(k) + 1);
  }
}

BOOST_AUTO_TEST_CASE(host_array_write_kernel_host_array_read) {
  sg::graph g(opts());
  auto in = g.create_mem_handle<int>(), out = g.create_mem_handle<int>();
  write_host(g, dev(), in, sycl::range<1>(8), iota_vec<int>(8, 1));
  scale(g, dev(), in, out, 8, 10, std::make_tuple(0));
  const auto r = read_host(g, dev(), out, sycl::range<1>(8));
  for (int k = 0; k < 8; ++k)
    BOOST_REQUIRE_EQUAL(r[k], (k + 1) * 10);
}

BOOST_AUTO_TEST_CASE(future_sets_a_shape) {
  sg::graph g(opts());
  auto p = g.create_promise<int>();
  auto h = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) { gh.define_host_work([p] { p.set(8); }); });
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> o(h, sg::shape<1>({sg::dim(p.future().get())}));
    gh.define_kernel([o](sg::kernel_handle &kh) {
      const sycl::range<1> r = o.range();
      o = kh.evaluate(kh.parallel_for(r, sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); }));
    });
  });
  g.run();
  const auto r = read_host(g, dev(), h, sycl::range<1>(8));
  for (int k = 0; k < 8; ++k)
    BOOST_REQUIRE_EQUAL(r[k], k);
}

BOOST_AUTO_TEST_CASE(future_sets_a_builder_value) {
  sg::graph g(opts());
  auto p = g.create_promise<int>();
  auto h = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) { gh.define_host_work([p] { p.set(7); }); });
  g.submit(dev(), [&](sg::submit_handle gh) {
    auto v = p.future().get();
    sg::array<sg::result<int>, 1> o(h, sg::shape<1>(sycl::range<1>(8)));
    gh.define_kernel([o, v](sg::kernel_handle &kh) {
      const int k = v;
      o = kh.evaluate(kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [k](sg::item<1> it) { return int(it.id()[0]) + k; }));
    });
  });
  g.run();
  const auto r = read_host(g, dev(), h, sycl::range<1>(8));
  for (int k = 0; k < 8; ++k)
    BOOST_REQUIRE_EQUAL(r[k], k + 7);
}

BOOST_AUTO_TEST_CASE(keyed_builder_reuse_with_new_arrays) {
  sg::graph g(opts());
  for (int round = 0; round < 3; ++round) {
    auto in = g.create_mem_handle<int>(), out = g.create_mem_handle<int>();
    write_host(g, dev(), in, sycl::range<1>(16), iota_vec<int>(16, round * 10));
    scale(g, dev(), in, out, 16, 4, std::make_tuple(7));
    const auto r = read_host(g, dev(), out, sycl::range<1>(16));
    for (int k = 0; k < 16; ++k)
      BOOST_REQUIRE_EQUAL(r[k], 4 * (round * 10 + k));
  }
}

BOOST_AUTO_TEST_CASE(different_key_gives_the_other_structure) {
  sg::graph g(opts());
  auto in = g.create_mem_handle<int>(), o2 = g.create_mem_handle<int>(), o3 = g.create_mem_handle<int>();
  write_host(g, dev(), in, sycl::range<1>(16), iota_vec<int>(16));
  scale(g, dev(), in, o2, 16, 2, std::make_tuple(2));
  scale(g, dev(), in, o3, 16, 3, std::make_tuple(3));
  const auto a = read_host(g, dev(), o2, sycl::range<1>(16));
  const auto b = read_host(g, dev(), o3, sycl::range<1>(16));
  for (int k = 0; k < 16; ++k) {
    BOOST_REQUIRE_EQUAL(a[k], 2 * k);
    BOOST_REQUIRE_EQUAL(b[k], 3 * k);
  }
}

BOOST_AUTO_TEST_CASE(unkeyed_builder_follows_a_host_variable) {
  sg::graph g(opts());
  auto in = g.create_mem_handle<int>(), o1 = g.create_mem_handle<int>(), o2 = g.create_mem_handle<int>();
  write_host(g, dev(), in, sycl::range<1>(16), iota_vec<int>(16));
  g.run();
  int factor = 2;
  scale_unkeyed(g, dev(), in, o1, 16, &factor);
  g.run();
  factor = 3;
  scale_unkeyed(g, dev(), in, o2, 16, &factor);
  g.run();
  const auto a = read_host(g, dev(), o1, sycl::range<1>(16));
  const auto b = read_host(g, dev(), o2, sycl::range<1>(16));
  for (int k = 0; k < 16; ++k) {
    BOOST_REQUIRE_EQUAL(a[k], 2 * k);
    BOOST_REQUIRE_EQUAL(b[k], 3 * k);
  }
}

BOOST_AUTO_TEST_CASE(cross_device_copies) {
  for (const auto &[first, second] : device_pairs()) {
    sg::graph g(opts());
    auto in = g.create_mem_handle<int>(), mid = g.create_mem_handle<int>(), out = g.create_mem_handle<int>();
    write_host(g, first, in, sycl::range<1>(16), iota_vec<int>(16));
    scale(g, second, in, mid, 16, 2, std::make_tuple(0)); // host -> second
    scale(g, first, mid, out, 16, 3, std::make_tuple(1)); // second -> first
    const auto r = read_host(g, first, out, sycl::range<1>(16));
    for (int k = 0; k < 16; ++k)
      BOOST_REQUIRE_EQUAL(r[k], 6 * k);
  }
}

BOOST_AUTO_TEST_CASE(back_to_back_runs_wait_at_the_end) {
  sg::graph g(opts());
  auto in = g.create_mem_handle<int>();
  write_host(g, dev(), in, sycl::range<1>(16), iota_vec<int>(16));
  g.run();
  std::vector<sg::mem_handle<int>> outs;
  for (int k = 0; k < 8; ++k) {
    outs.push_back(g.create_mem_handle<int>());
    scale(g, dev(), in, outs.back(), 16, k + 1, std::make_tuple(100 + k));
    g.run();
  }
  g.wait();
  for (int k = 0; k < 8; ++k) {
    const auto r = read_host(g, dev(), outs[k], sycl::range<1>(16));
    for (int i = 0; i < 16; ++i)
      BOOST_REQUIRE_EQUAL(r[i], (k + 1) * i);
  }
}

BOOST_AUTO_TEST_CASE(destructor_waits) {
  auto flag = std::make_shared<std::atomic<int>>(0);
  {
    sg::graph g(opts());
    g.submit(dev(), [&](sg::submit_handle gh) {
      gh.define_host_work([flag] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        flag->store(1);
      });
    });
    g.run();
  }
  BOOST_CHECK_EQUAL(flag->load(), 1);
}

BOOST_AUTO_TEST_CASE(many_runs_of_varying_sizes_with_a_small_pool) {
  sg::graph g(opts(4));
  for (int round = 0; round < 40; ++round) {
    const std::size_t n = 4 * (1 + (round * 37) % 700);
    auto h = g.create_mem_handle<int>();
    g.submit(dev(), [&](sg::submit_handle gh) {
      sg::array<sg::result<int>, 1> o(h, sg::shape<1>(sycl::range<1>(n)));
      gh.define_kernel([o, n](sg::kernel_handle &kh) {
        o = kh.evaluate(kh.parallel_for(sycl::range<1>(n), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]) * 3; }));
      });
    });
    const auto r = read_host(g, dev(), h, sycl::range<1>(n));
    for (std::size_t k = 0; k < n; ++k)
      BOOST_REQUIRE_EQUAL(r[k], int(k) * 3);
  }
}

BOOST_AUTO_TEST_SUITE_END()
