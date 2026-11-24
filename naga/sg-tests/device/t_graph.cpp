#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include <atomic>
#include <memory>
#include <vector>

using namespace sgt;

BOOST_AUTO_TEST_SUITE(graph_runs)

BOOST_AUTO_TEST_CASE(twenty_back_to_back_runs_each_see_their_own_data) {
  constexpr int runs = 20;
  sg::graph g;
  auto x = g.create_mem_handle<float>();
  auto y = g.create_mem_handle<float>();
  auto writes = std::make_shared<std::atomic<int>>(0);
  auto snapshots = std::make_shared<std::atomic<int>>(0);
  auto saved = std::make_shared<std::vector<std::vector<float>>>(runs);

  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto a = hh.write(x, sg::shape<1>({64}));
    hh.run([a, writes] {
      const int r = writes->fetch_add(1);
      for (int i = 0; i < 64; ++i)
        a.data()[i] = static_cast<float>(i + r);
    });
  });
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k9(kh, x, y, 1, 64); });
  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto a = hh.read(y, sg::shape<1>({64}));
    hh.run([a, snapshots, saved] {
      const int r = snapshots->fetch_add(1);
      (*saved)[r].assign(a.data(), a.data() + 64);
    });
  });

  for (int r = 0; r < runs; ++r)
    g.run();
  g.wait();

  BOOST_REQUIRE_EQUAL(writes->load(), runs);
  BOOST_REQUIRE_EQUAL(snapshots->load(), runs);
  for (int r = 0; r < runs; ++r) {
    BOOST_REQUIRE_EQUAL((*saved)[r].size(), 64u);
    for (int i = 0; i < 64; ++i)
      BOOST_REQUIRE_EQUAL((*saved)[r][i], 2.f * (i + r) + 1.f);
  }
}

BOOST_AUTO_TEST_SUITE_END()
