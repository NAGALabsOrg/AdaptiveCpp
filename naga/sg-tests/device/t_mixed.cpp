#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include <cstdint>
#include <vector>

using namespace sgt;

namespace {

template <class T> std::vector<T> per_tile_scan(const std::vector<T> &x) {
  return run_kernel<T, T>(x, 256, [](auto &kh, auto xh, auto yh) {
    auto xs = kh.source(xh, sg::shape<1>({256}));
    auto ts = kh.tile(xs, {64});
    auto work = kh.work_tiles({4}, {1}, 64);
    auto y = kh.map_work(work, [=](auto unit) {
      auto tile = ts[unit.tile()];
      auto run = sg::flatten(tile);
      return unit.run(run, scan_items);
    });
    kh.evaluate(yh, kh.join(kh.retile(y, {64})));
  });
}

template <class T> void contribute(sg::kernel_handle &kh, H<T> ones_in, H<T> y, T k) {
  auto scale = [=](T v) { return static_cast<T>(k * v); };
  auto ones = kh.source(ones_in, sg::shape<1>({64}));
  auto items = kh.work_items({64});
  auto c = kh.map_items(items, [=](auto item) {
    auto v = ones[item.id()];
    return item.compute(v, scale);
  });
  kh.evaluate(sg::inplace(y, sycl::plus<T>{}, T{0}), c);
}

template <class T> std::vector<T> two_contributions() {
  sg::graph g;
  auto ones = g.create_mem_handle<T>();
  auto y = g.create_mem_handle<T>();
  host_write(g, ones, std::vector<T>(64, T{1}));
  g.submit(test_device(), [&](sg::kernel_handle &kh) { contribute<T>(kh, ones, y, T{1}); });
  g.submit(test_device(), [&](sg::kernel_handle &kh) { contribute<T>(kh, ones, y, T{2}); });
  std::vector<T> out;
  host_read(g, y, 64, out);
  g.run();
  g.wait();
  return out;
}

} // namespace

BOOST_AUTO_TEST_SUITE(mixed_widths)

BOOST_AUTO_TEST_CASE(m1_uint8_inclusive_scan_per_tile) {
  std::vector<std::uint8_t> x(256);
  for (int i = 0; i < 256; ++i)
    x[i] = static_cast<std::uint8_t>(i);
  const auto y = per_tile_scan(x);
  BOOST_REQUIRE_EQUAL(y.size(), 256u);
  for (int t = 0; t < 4; ++t) {
    std::uint8_t acc = 0;
    for (int k = 0; k < 64; ++k) {
      acc = static_cast<std::uint8_t>(acc + x[64 * t + k]);
      BOOST_REQUIRE_EQUAL(static_cast<int>(y[64 * t + k]), static_cast<int>(acc));
    }
  }
  for (int i : {63, 127, 191, 255})
    BOOST_CHECK_EQUAL(static_cast<int>(y[i]), 224);
  BOOST_CHECK_EQUAL(static_cast<int>(y[64]), 64);
  BOOST_CHECK_EQUAL(static_cast<int>(y[65]), 129);
}

BOOST_AUTO_TEST_CASE(m2_uint16_reduce_per_tile) {
  std::vector<std::uint16_t> x(256);
  for (int i = 0; i < 256; ++i)
    x[i] = static_cast<std::uint16_t>(1000 + i);
  const auto y = run_kernel<std::uint16_t, std::uint16_t>(x, 4, [](auto &kh, auto xh, auto yh) {
    auto xs = kh.source(xh, sg::shape<1>({256}));
    auto ts = kh.tile(xs, {64});
    auto work = kh.work_tiles({4}, {1}, 64);
    auto r = sum_tiles(kh, work, ts);
    kh.evaluate(yh, kh.join(r));
  });
  BOOST_REQUIRE_EQUAL(y.size(), 4u);
  const int want[4] = {480, 4576, 8672, 12768};
  for (int t = 0; t < 4; ++t)
    BOOST_CHECK_EQUAL(static_cast<int>(y[t]), want[t]);
}

BOOST_AUTO_TEST_CASE(m3_uint16_reduce_of_a_uint8_and_a_uint16_array) {
  std::vector<std::uint8_t> a(256);
  std::vector<std::uint16_t> b(256);
  for (int i = 0; i < 256; ++i) {
    a[i] = static_cast<std::uint8_t>(i);
    b[i] = static_cast<std::uint16_t>(1000 + i);
  }
  sg::graph g;
  auto ah = g.create_mem_handle<std::uint8_t>();
  auto bh = g.create_mem_handle<std::uint16_t>();
  auto yh = g.create_mem_handle<std::uint16_t>();
  host_write(g, ah, a);
  host_write(g, bh, b);
  g.submit(test_device(), [&](sg::kernel_handle &kh) {
    auto as = kh.source(ah, sg::shape<1>({256}));
    auto bs = kh.source(bh, sg::shape<1>({256}));
    auto at = kh.tile(as, {64});
    auto bt = kh.tile(bs, {64});
    auto work = kh.work_tiles({4}, {1}, 64);
    auto mixed = [](auto item, auto av, auto bv) {
      const std::uint16_t s = static_cast<std::uint16_t>(static_cast<std::uint16_t>(av) + bv);
      return sg::reduce(item, s, sycl::plus<>{});
    };
    auto r = kh.map_work(work, [=](auto unit) {
      auto a_tile = at[unit.tile()];
      auto b_tile = bt[unit.tile()];
      auto a_run = sg::flatten(a_tile);
      auto b_run = sg::flatten(b_tile);
      return unit.run(a_run, b_run, mixed);
    });
    kh.evaluate(yh, kh.join(r));
  });
  std::vector<std::uint16_t> y;
  host_read(g, yh, 4, y);
  g.run();
  g.wait();
  BOOST_REQUIRE_EQUAL(y.size(), 4u);
  const int want[4] = {2496, 10688, 18880, 27072};
  for (int t = 0; t < 4; ++t)
    BOOST_CHECK_EQUAL(static_cast<int>(y[t]), want[t]);
}

BOOST_AUTO_TEST_CASE(m4_inplace_contributions_to_uint8_and_uint16) {
  for (auto v : two_contributions<std::uint8_t>())
    BOOST_CHECK_EQUAL(static_cast<int>(v), 3);
  for (auto v : two_contributions<std::uint16_t>())
    BOOST_CHECK_EQUAL(static_cast<int>(v), 3);
}

BOOST_AUTO_TEST_CASE(m5_chain_of_three_work_sets_over_three_widths) {
  sg::graph g;
  auto xh = g.create_mem_handle<std::uint8_t>();
  auto yh = g.create_mem_handle<std::int32_t>();
  std::vector<std::uint8_t> x(64);
  for (int i = 0; i < 64; ++i)
    x[i] = static_cast<std::uint8_t>(i);
  host_write(g, xh, x);
  g.submit(test_device(), [&](sg::kernel_handle &kh) {
    auto xs = kh.source(xh, sg::shape<1>({64}));
    auto ts = kh.tile(xs, {4});
    auto plus_one = [](auto, auto v) { return static_cast<std::uint8_t>(v + 1); };
    auto plus_300 = [](auto, auto v) { return static_cast<std::uint16_t>(v + 300); };
    auto twice = [](auto, auto v) { return static_cast<std::int32_t>(v) * 2; };
    auto w1 = kh.work_tiles({16}, {1}, 4);
    auto a = kh.map_work(w1, [=](auto unit) {
      auto tile = ts[unit.tile()];
      auto run = sg::flatten(tile);
      return unit.run(run, plus_one);
    });
    auto w2 = kh.work_tiles({16}, {1}, 4);
    auto b = kh.map_work(w2, [=](auto unit) {
      auto tile = a[unit.tile()];
      auto run = sg::flatten(tile);
      return unit.run(run, plus_300);
    });
    auto w3 = kh.work_tiles({16}, {1}, 4);
    auto c = kh.map_work(w3, [=](auto unit) {
      auto tile = b[unit.tile()];
      auto run = sg::flatten(tile);
      return unit.run(run, twice);
    });
    kh.evaluate(yh, kh.join(kh.retile(c, {4})));
  });
  std::vector<std::int32_t> y;
  host_read(g, yh, 64, y);
  g.run();
  g.wait();
  BOOST_REQUIRE_EQUAL(y.size(), 64u);
  for (int i = 0; i < 64; ++i)
    BOOST_REQUIRE_EQUAL(y[i], 2 * i + 602);
  BOOST_CHECK_EQUAL(y[0], 602);
  BOOST_CHECK_EQUAL(y[63], 728);
}

BOOST_AUTO_TEST_SUITE_END()
