#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include <atomic>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace sgt;

namespace {

std::string level() {
  const char *l = std::getenv("ACPP_ADAPTIVITY_LEVEL");
  return l ? l : "default";
}

} // namespace

BOOST_AUTO_TEST_SUITE(reruns)

// graph A: a host node writes n and x, the kernel computes y = 2x + 1 over n items, a host node keeps y
BOOST_AUTO_TEST_CASE(graph_a_changes_n_and_contents_without_a_new_compilation) {
  BOOST_TEST_MESSAGE("ACPP_ADAPTIVITY_LEVEL = " << level());
  sg::graph g;
  auto n_h = g.create_mem_handle<int>();
  auto x_h = g.create_mem_handle<float>();
  auto y_h = g.create_mem_handle<float>();
  auto writes = std::make_shared<std::atomic<int>>(0);
  auto reads = std::make_shared<std::atomic<int>>(0);
  auto saved = std::make_shared<std::vector<std::vector<float>>>(2);

  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto n = hh.write(n_h, sg::shape<1>({1}));
    auto x = hh.write(x_h, sg::shape<1>({128}));
    hh.run([n, x, writes] {
      const int r = writes->fetch_add(1);
      n.data()[0] = r == 0 ? 64 : 128;
      for (int i = 0; i < 128; ++i)
        x.data()[i] = static_cast<float>(i + (r == 0 ? 0 : 1000));
    });
  });
  g.submit(test_device(), [&](sg::kernel_handle &kh) { rerun_a(kh, x_h, n_h, y_h); });
  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto y = hh.read(y_h, sg::shape<1>({64}));
    hh.run([y, reads, saved] {
      const int r = reads->fetch_add(1);
      (*saved)[r].assign(y.data(), y.data() + 64);
    });
  });

  g.run();
  g.wait();
  const std::uint64_t after_first = sg::jit_compile_count();
  BOOST_CHECK_GE(after_first, 1u);
  g.run();
  g.wait();
  const std::uint64_t after_second = sg::jit_compile_count();
  BOOST_CHECK_EQUAL(after_second, after_first);

  for (int i = 0; i < 64; ++i) {
    BOOST_REQUIRE_EQUAL((*saved)[0][i], 2.f * i + 1.f);
    BOOST_REQUIRE_EQUAL((*saved)[1][i], 2.f * i + 2001.f);
  }
}

// the memory rule: a graph run twenty times back to back takes no more slabs from the device after its first run
BOOST_AUTO_TEST_CASE(twenty_back_to_back_runs_take_no_more_slabs_after_the_first_run) {
  constexpr int runs = 20;
  sg::graph g;
  auto x = g.create_mem_handle<float>();
  auto y = g.create_mem_handle<float>();
  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto a = hh.write(x, sg::shape<1>({64}));
    hh.run([a] {
      for (int i = 0; i < 64; ++i)
        a.data()[i] = static_cast<float>(i);
    });
  });
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k9(kh, x, y, 1, 64); });
  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto a = hh.read(y, sg::shape<1>({64}));
    hh.run([a] { (void)a.data(); });
  });

  BOOST_CHECK_EQUAL(g.pool_slab_count(), 0u);
  g.run();
  const std::size_t after_first = g.pool_slab_count();
  BOOST_CHECK_GE(after_first, 1u);
  for (int r = 1; r < runs; ++r)
    g.run();
  g.wait();
  BOOST_CHECK_EQUAL(g.pool_slab_count(), after_first);
}

// graph B: K2 with the depth as a launch value; run 1 depth 3 over 64 ones, run 2 depth 5 over 1024 ones
BOOST_AUTO_TEST_CASE(graph_b_changes_the_depth_without_a_new_compilation) {
  BOOST_TEST_MESSAGE("ACPP_ADAPTIVITY_LEVEL = " << level());
  sg::graph g;
  auto n_h = g.create_mem_handle<int>();
  auto d_h = g.create_mem_handle<int>();
  auto x_h = g.create_mem_handle<float>();
  auto t_h = g.create_mem_handle<float>();
  auto writes = std::make_shared<std::atomic<int>>(0);
  auto reads = std::make_shared<std::atomic<int>>(0);
  auto saved = std::make_shared<std::vector<float>>(2, 0.f);

  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto n = hh.write(n_h, sg::shape<1>({1}));
    auto d = hh.write(d_h, sg::shape<1>({1}));
    auto x = hh.write(x_h, sg::shape<1>({1024}));
    hh.run([n, d, x, writes] {
      const int r = writes->fetch_add(1);
      n.data()[0] = r == 0 ? 64 : 1024;
      d.data()[0] = r == 0 ? 3 : 5;
      for (int i = 0; i < 1024; ++i)
        x.data()[i] = 1.f;
    });
  });
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k2(kh, x_h, n_h, d_h, t_h); });
  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto t = hh.read(t_h, sg::shape<1>({1}));
    hh.run([t, reads, saved] {
      const int r = reads->fetch_add(1);
      (*saved)[r] = t.data()[0];
    });
  });

  g.run();
  g.wait();
  const std::uint64_t after_first = sg::jit_compile_count();
  BOOST_CHECK_GE(after_first, 1u);
  g.run();
  g.wait();
  BOOST_CHECK_EQUAL(sg::jit_compile_count(), after_first);
  BOOST_CHECK_EQUAL((*saved)[0], 64.f);
  BOOST_CHECK_EQUAL((*saved)[1], 1024.f);
}

// a work-item kernel whose range goes from 64 to 100000 items: the two counts select different automatic
// group sizes on OpenMP, so a kernel launched with the count as its range would be compiled twice
BOOST_AUTO_TEST_CASE(a_work_item_kernel_changes_its_range_without_a_new_compilation) {
  BOOST_TEST_MESSAGE("ACPP_ADAPTIVITY_LEVEL = " << level());
  constexpr int big = 100000;
  sg::graph g;
  auto n_h = g.create_mem_handle<int>();
  auto x_h = g.create_mem_handle<float>();
  auto y_h = g.create_mem_handle<float>();
  auto writes = std::make_shared<std::atomic<int>>(0);
  auto reads = std::make_shared<std::atomic<int>>(0);
  auto saved = std::make_shared<std::vector<std::vector<float>>>(2);

  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto n = hh.write(n_h, sg::shape<1>({1}));
    auto x = hh.write(x_h, sg::shape<1>({big}));
    hh.run([n, x, writes] {
      const int r = writes->fetch_add(1);
      n.data()[0] = r == 0 ? 64 : big;
      for (int i = 0; i < big; ++i)
        x.data()[i] = static_cast<float>(i + (r == 0 ? 0 : 1000));
    });
  });
  g.submit(test_device(), [&](sg::kernel_handle &kh) { rerun_a(kh, x_h, n_h, y_h); });
  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto y = hh.read(y_h, sg::shape<1>({64}));
    hh.run([y, reads, saved] {
      const int r = reads->fetch_add(1);
      (*saved)[r].assign(y.data(), y.data() + 64);
    });
  });

  g.run();
  g.wait();
  const std::uint64_t after_first = sg::jit_compile_count();
  BOOST_CHECK_GE(after_first, 1u);
  g.run();
  g.wait();
  BOOST_CHECK_EQUAL(sg::jit_compile_count(), after_first);
  for (int i = 0; i < 64; ++i) {
    BOOST_REQUIRE_EQUAL((*saved)[0][i], 2.f * i + 1.f);
    BOOST_REQUIRE_EQUAL((*saved)[1][i], 2.f * i + 2001.f);
  }
}

BOOST_AUTO_TEST_SUITE_END()
