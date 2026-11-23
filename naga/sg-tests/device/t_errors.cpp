#include "common.hpp"

using namespace sgt;

namespace {
using L1 = sg::group_iterator<int, 1, sg::mapping::local>;

sg::mem_handle<int> good_node(sg::graph &g) {
  auto ok = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> o(ok, sg::shape<1>(sycl::range<1>(8)));
    gh.define_kernel([o](sg::kernel_handle &kh) {
      o = kh.evaluate(kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]) * 2; }));
    });
  });
  return ok;
}
void check_ok(sg::graph &g, sg::mem_handle<int> ok) {
  const auto r = read_host(g, dev(), ok, sycl::range<1>(8));
  for (int i = 0; i < 8; ++i)
    BOOST_REQUIRE_EQUAL(r[i], i * 2);
}
void produce(sg::graph &g, sg::mem_handle<int> h, std::size_t n) {
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> o(h, sg::shape<1>(sycl::range<1>(n)));
    gh.define_kernel([o, n](sg::kernel_handle &kh) {
      o = kh.evaluate(kh.parallel_for(sycl::range<1>(n), sycl::range<1>(4), [](sg::item<1>) { return 1; }));
    });
  });
}
} // namespace

BOOST_AUTO_TEST_SUITE(errors_at_submit)

BOOST_AUTO_TEST_CASE(nested_submit) {
  sg::graph g(opts());
  SG_CHECK_ERRC(g.submit(dev(), [&](sg::submit_handle) { g.submit(dev(), [](sg::submit_handle) {}); }), builder_misuse);
}
BOOST_AUTO_TEST_CASE(no_define_and_two_defines) {
  sg::graph g(opts());
  SG_CHECK_ERRC(g.submit(dev(), [](sg::submit_handle) {}), builder_misuse);
  SG_CHECK_ERRC(g.submit(dev(), [](sg::submit_handle gh) {
                  gh.define_host_work([] {});
                  gh.define_host_work([] {});
                }),
                builder_misuse);
}
BOOST_AUTO_TEST_CASE(outside_a_scope) {
  sg::graph g(opts());
  auto h = g.create_mem_handle<int>();
  SG_CHECK_ERRC((sg::array<int, 1>(h, sg::shape<1>(sycl::range<1>(4)))), not_in_submit_scope);
  auto p = g.create_promise<int>();
  SG_CHECK_ERRC(p.future().get(), not_in_submit_scope);
}
BOOST_AUTO_TEST_CASE(role_misuse) {
  sg::graph g(opts());
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::mem_handle<int> null;
    SG_CHECK_ERRC((sg::array<int, 1>(null, sg::shape<1>(sycl::range<1>(4)))), builder_misuse);
    SG_CHECK_ERRC(sg::shape<2>({1, 2, 3}), builder_misuse);
    gh.define_host_work([] {});
  });
  g.run();
  g.wait();
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(errors_from_run)

BOOST_AUTO_TEST_CASE(never_produced) {
  sg::graph g(opts());
  auto ok = good_node(g);
  auto h = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::host_array<const int, 1> a(h, sg::shape<1>(sycl::range<1>(4)));
    gh.define_host_work([a] {});
  });
  SG_CHECK_ERRC(g.run(), never_produced);
  check_ok(g, ok);
}

BOOST_AUTO_TEST_CASE(type_mismatch) {
  sg::graph g(opts());
  auto ok = good_node(g);
  auto h = g.create_mem_handle<int>();
  produce(g, h, 8);
  g.run();
  auto hf = sg::detail::handle_access::make<sg::mem_handle<float>>(sg::detail::handle_access::owner(h),
                                                                   sg::detail::handle_access::index(h));
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::host_array<const float, 1> a(hf, sg::shape<1>(sycl::range<1>(8)));
    gh.define_host_work([a] {});
  });
  SG_CHECK_ERRC(g.run(), type_mismatch);
  check_ok(g, ok);
}

BOOST_AUTO_TEST_CASE(size_mismatch) {
  sg::graph g(opts());
  auto ok = good_node(g);
  auto h = g.create_mem_handle<int>();
  produce(g, h, 8);
  g.run();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::host_array<const int, 1> a(h, sg::shape<1>(sycl::range<1>(16)));
    gh.define_host_work([a] {});
  });
  SG_CHECK_ERRC(g.run(), size_mismatch);
  check_ok(g, ok);
}

BOOST_AUTO_TEST_CASE(role_conflict) {
  sg::graph g(opts());
  auto ok = good_node(g);
  auto h = g.create_mem_handle<int>();
  produce(g, h, 8);
  g.run();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<int, 1> in(h, sg::shape<1>(sycl::range<1>(8)));
    sg::array<sg::result<int>, 1> out(h, sg::shape<1>(sycl::range<1>(8)));
    gh.define_kernel([in, out](sg::kernel_handle &kh) {
      out = kh.evaluate(kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [in](sg::item<1> it) { return in[it.id()]; }));
    });
  });
  SG_CHECK_ERRC(g.run(), role_conflict);
  check_ok(g, ok);
}

BOOST_AUTO_TEST_CASE(inplace_group_mismatch) {
  sg::graph g(opts());
  auto ok = good_node(g);
  auto h = g.create_mem_handle<int>();
  auto contribute = [&](auto op) {
    g.submit(dev(), [&](sg::submit_handle gh) {
      sg::array<sg::result<int>, 1> out(sg::inplace(h, op, 0), sg::shape<1>(sycl::range<1>(8)));
      gh.define_kernel([out](sg::kernel_handle &kh) {
        out = kh.evaluate(kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1>) { return 1; }));
      });
    });
  };
  contribute(sycl::plus<>{});
  contribute(sycl::minimum<>{});
  SG_CHECK_ERRC(g.run(), inplace_group_mismatch);
  check_ok(g, ok);
}

BOOST_AUTO_TEST_CASE(shape_mismatch_between_definition_and_result) {
  sg::graph g(opts());
  auto ok = good_node(g);
  auto h = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> out(h, sg::shape<1>(sycl::range<1>(8)));
    gh.define_kernel([out](sg::kernel_handle &kh) {
      out = kh.evaluate(kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), [](sg::item<1>) { return 1; }));
    });
  });
  SG_CHECK_ERRC(g.run(), shape_mismatch);
  check_ok(g, ok);
}

BOOST_AUTO_TEST_CASE(item_mapping_from_a_group_body) {
  sg::graph g(opts());
  auto ok = good_node(g);
  auto h = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> out(h, sg::shape<1>(sycl::range<1>(4)));
    gh.define_kernel([out](sg::kernel_handle &kh) {
      auto a = kh.parallel_for(sycl::range<1>(32), sycl::range<1>(8), [](sg::item<1>) { return 1; });
      auto m = kh.map_groups<sg::mapping::local>(a, [](sycl::id<1> g) { return sg::single<1>{g}; });
      auto grp = kh.parallel_for(sycl::range<1>(4), m, [](sg::group<1>, L1 a) {
        return sg::group_reduce<1>(a[0], 0, sycl::plus<>{});
      });
      auto mi = kh.map_groups<sg::mapping::item>(grp, [](sycl::id<1> g) { return sg::single<1>{g}; });
      out = kh.evaluate(kh.parallel_for(sycl::range<1>(4), sycl::range<1>(1), mi,
                                        [](sg::item<1>, sg::group_iterator<int, 1, sg::mapping::item> x) { return x[0]; }));
    });
  });
  SG_CHECK_ERRC(g.run(), mapping_mismatch);
  check_ok(g, ok);
}

BOOST_AUTO_TEST_CASE(evaluate_twice) {
  sg::graph g(opts());
  auto ok = good_node(g);
  auto h = g.create_mem_handle<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> out(h, sg::shape<1>(sycl::range<1>(8)));
    gh.define_kernel([out](sg::kernel_handle &kh) {
      auto d = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1>) { return 1; });
      out = kh.evaluate(d);
      out = kh.evaluate(d);
    });
  });
  SG_CHECK_ERRC(g.run(), builder_misuse);
  check_ok(g, ok);
}

BOOST_AUTO_TEST_CASE(work_group_too_large) {
  sg::graph g(opts());
  auto ok = good_node(g);
  auto h = g.create_mem_handle<int>();
  const std::size_t gs = dev().get_info<sycl::info::device::max_work_group_size>() + 1;
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> out(h, sg::shape<1>(sycl::range<1>(gs)));
    gh.define_kernel([out, gs](sg::kernel_handle &kh) {
      out = kh.evaluate(kh.parallel_for(sycl::range<1>(gs), sycl::range<1>(gs), [](sg::item<1>) { return 1; }));
    });
  });
  SG_CHECK_ERRC(g.run(), work_group_too_large);
  check_ok(g, ok);
}

BOOST_AUTO_TEST_CASE(local_memory_exceeded) {
  sg::graph g(opts());
  auto ok = good_node(g);
  auto h = g.create_mem_handle<int>();
  const std::size_t local_limit = dev().get_info<sycl::info::device::local_mem_size>();
  BOOST_TEST_MESSAGE("device local_mem_size: " << local_limit);
  if (local_limit > (std::size_t(1) << 30)) {
    BOOST_WARN_MESSAGE(false, "SKIPPED local_memory_exceeded: the device reports no finite local memory limit (" +
                                  std::to_string(local_limit) + " bytes)");
    return;
  }
  // each local slot holds one source group of 64 ints (256 B): one more slot than the device's local memory holds
  const std::size_t slots = local_limit / 256 + 1;
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<int>, 1> out(h, sg::shape<1>(sycl::range<1>(4)));
    gh.define_kernel([out, slots](sg::kernel_handle &kh) {
      auto a = kh.parallel_for(sycl::range<1>(64), sycl::range<1>(64), [](sg::item<1>) { return 1; });
      auto m = kh.map_groups<sg::mapping::local>(a, [slots](sycl::id<1>) {
        return sg::block<1>{sycl::id<1>(0), sycl::range<1>(slots)};
      });
      out = kh.evaluate(kh.parallel_for(sycl::range<1>(4), sycl::range<1>(4), m, [](sg::item<1>, L1 a) {
        return int(a.size());
      }));
    });
  });
  SG_CHECK_ERRC(g.run(), local_memory_exceeded);
  check_ok(g, ok);
}

BOOST_AUTO_TEST_CASE(pool_full) {
  sg::graph g(opts(1));
  auto h = g.create_mem_handle<int>();
  produce(g, h, 1u << 22);
  SG_CHECK_ERRC(g.run(), pool_exhausted);
}

BOOST_AUTO_TEST_CASE(future_unset) {
  sg::graph g(opts());
  auto ok = good_node(g);
  auto p = g.create_promise<int>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    auto v = p.future().get();
    (void)v;
    gh.define_host_work([] {});
  });
  SG_CHECK_ERRC(g.run(), future_unset);
  check_ok(g, ok);
}

BOOST_AUTO_TEST_SUITE_END()
