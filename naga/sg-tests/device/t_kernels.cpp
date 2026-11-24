#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include <cstdint>
#include <string>
#include <vector>

using namespace sgt;

namespace {

int max_group() { return static_cast<int>(test_device().get_info<sycl::info::device::max_work_group_size>()); }

// a kernel of 1024 items per group needs a device that allows it
bool group_of_1024_fits() {
  if (max_group() >= 1024)
    return true;
  BOOST_TEST_MESSAGE("SKIPPED: the device's largest work-group is " << max_group() << ", the kernel needs 1024");
  return false;
}

template <class T> H<T> input(sg::graph &g, const std::vector<T> &v) {
  auto h = g.create_mem_handle<T>();
  host_write(g, h, v);
  return h;
}

template <class T> void same(const std::vector<T> &got, const std::vector<T> &want) {
  BOOST_REQUIRE_EQUAL(got.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i)
    BOOST_REQUIRE_MESSAGE(got[i] == want[i], "element " << i << ": got " << got[i] << ", want " << want[i]);
}

std::vector<float> run_k1(const std::vector<float> &x) {
  sg::graph g;
  auto in = input(g, x);
  auto total = g.create_mem_handle<float>();
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k1(kh, in, total); });
  std::vector<float> out;
  host_read(g, total, 1, out);
  g.run();
  g.wait();
  return out;
}

std::vector<float> run_k2(int depth) {
  int n = 1;
  for (int i = 0; i < depth; ++i)
    n *= 4;
  sg::graph g;
  auto x = input(g, std::vector<float>(n, 1.f));
  auto nh = input(g, std::vector<int>{n});
  auto dh = input(g, std::vector<int>{depth});
  auto total = g.create_mem_handle<float>();
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k2(kh, x, nh, dh, total); });
  std::vector<float> out;
  host_read(g, total, 1, out);
  g.run();
  g.wait();
  return out;
}

std::vector<int> run_k6(bool segmented) {
  std::vector<int> a(128 * 128), b(128 * 128);
  for (int i = 0; i < 128; ++i)
    for (int k = 0; k < 128; ++k) {
      a[i * 128 + k] = i;
      b[i * 128 + k] = k;
    }
  sg::graph g;
  auto ah = input(g, a);
  auto bh = input(g, b);
  auto ch = g.create_mem_handle<int>();
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k6(kh, ah, bh, ch, segmented); });
  std::vector<int> out;
  host_read(g, ch, 128 * 128, out);
  g.run();
  g.wait();
  return out;
}

std::vector<int> run_k14(int cap, bool segmented) {
  std::vector<int> x(64 * 1024);
  for (int r = 0; r < 64; ++r)
    for (int c = 0; c < 1024; ++c)
      x[r * 1024 + c] = r + c;
  sg::graph_options o;
  o.group_size_cap = cap;
  sg::graph g(o);
  auto xh = input(g, x);
  auto yh = g.create_mem_handle<int>();
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k14(kh, xh, yh, segmented); });
  std::vector<int> out;
  host_read(g, yh, 64, out);
  g.run();
  g.wait();
  return out;
}

std::vector<float> run_k10(float a, float b, int n, bool tiled) {
  sg::graph g;
  auto x = input(g, iota_f(n));
  auto y = g.create_mem_handle<float>();
  g.submit(test_device(), [&](sg::kernel_handle &kh) {
    if (tiled)
      k10_tiled(kh, x, y, a, b, n);
    else
      k10(kh, x, y, a, b, n);
  });
  std::vector<float> out;
  host_read(g, y, n, out);
  g.run();
  g.wait();
  return out;
}

} // namespace

BOOST_AUTO_TEST_SUITE(kernels)

BOOST_AUTO_TEST_CASE(k01_global_reduction_of_i) {
  const auto got = run_k1(iota_f(64));
  same(got, std::vector<float>{2016.f});
}

BOOST_AUTO_TEST_CASE(k01_global_reduction_of_i_plus_1) {
  const auto got = run_k1(iota_f(64, 1.f));
  same(got, std::vector<float>{2080.f});
}

BOOST_AUTO_TEST_CASE(k02_depth_3_over_64_ones) { same(run_k2(3), std::vector<float>{64.f}); }

BOOST_AUTO_TEST_CASE(k02_depth_5_over_1024_ones) { same(run_k2(5), std::vector<float>{1024.f}); }

BOOST_AUTO_TEST_CASE(k03_inclusive_scan_of_1024_ones) {
  const auto got = run_kernel<int, int>(std::vector<int>(1024, 1), 1024,
                                        [](auto &kh, auto x, auto y) { k3(kh, x, y); });
  std::vector<int> want(1024);
  for (int k = 0; k < 1024; ++k)
    want[k] = k + 1;
  same(got, want);
}

BOOST_AUTO_TEST_CASE(k04_three_point_stencil) {
  const auto x = iota_i(1024);
  const auto got = run_kernel<int, int>(x, 1024, [](auto &kh, auto in, auto y) { k4(kh, in, y); });
  std::vector<int> want(1024);
  for (int i = 0; i < 1024; ++i)
    want[i] = (i > 0 ? x[i - 1] : 0) + x[i] + (i < 1023 ? x[i + 1] : 0);
  same(got, want);
  BOOST_CHECK_EQUAL(got[0], 1);
  BOOST_CHECK_EQUAL(got[1023], 2045);
}

BOOST_AUTO_TEST_CASE(k05_gather_from_values_computed_in_the_kernel) {
  std::vector<int> idx(256);
  for (int i = 0; i < 256; ++i)
    idx[i] = (7 * i) % 256;
  sg::graph g;
  auto s = input(g, iota_i(256));
  auto ix = input(g, idx);
  auto y = g.create_mem_handle<int>();
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k5(kh, s, ix, y); });
  std::vector<int> out;
  host_read(g, y, 256, out);
  g.run();
  g.wait();
  std::vector<int> want(256);
  for (int i = 0; i < 256; ++i)
    want[i] = 2 * ((7 * i) % 256);
  same(out, want);
  BOOST_CHECK_EQUAL(out[1], 14);
  BOOST_CHECK_EQUAL(out[37], 6);
}

BOOST_AUTO_TEST_CASE(k06_matrix_product_at_cap_1024) {
  if (!group_of_1024_fits())
    return;
  const auto got = run_k6(false);
  std::vector<int> want(128 * 128);
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j)
      want[i * 128 + j] = 128 * i * j;
  same(got, want);
  BOOST_CHECK_EQUAL(got[3 * 128 + 5], 1920);
  BOOST_CHECK_EQUAL(got[127 * 128 + 127], 2064512);
}

BOOST_AUTO_TEST_CASE(k06_matrix_product_segmented) {
  if (!group_of_1024_fits())
    return;
  const auto got = run_k6(true);
  std::vector<int> want(128 * 128);
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j)
      want[i * 128 + j] = 128 * i * j;
  same(got, want);
}

BOOST_AUTO_TEST_CASE(k07_histogram_of_4096_values) {
  std::vector<int> v(4096);
  for (int i = 0; i < 4096; ++i)
    v[i] = i % 16;
  const auto got = run_kernel<int, int>(v, 16, [](auto &kh, auto x, auto y) { k7(kh, x, y); });
  same(got, std::vector<int>(16, 256));
}

BOOST_AUTO_TEST_CASE(k08_a_to_f) {
  sg::graph g;
  auto x = input(g, iota_i(16));
  auto b = g.create_mem_handle<int>(), c = g.create_mem_handle<int>(), d = g.create_mem_handle<int>(),
       f = g.create_mem_handle<int>();
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k8(kh, x, b, c, d, f); });
  std::vector<int> ob, oc, od, of;
  host_read(g, b, 16, ob);
  host_read(g, c, 16, oc);
  host_read(g, d, 16, od);
  host_read(g, f, 16, of);
  g.run();
  g.wait();
  std::vector<int> wb(16), wc(16), wd(16), wf(16);
  for (int i = 0; i < 16; ++i) {
    wb[i] = i + 1;
    wc[i] = 2 * i;
    wd[i] = 3 * i + 1;
  }
  for (int gi = 0; gi < 4; ++gi)
    for (int l = 0; l < 4; ++l)
      wf[4 * gi + l] = 2 * (4 * (3 - gi) + l) + 1001;
  same(ob, wb);
  same(oc, wc);
  same(od, wd);
  same(of, wf);
  BOOST_CHECK_EQUAL(of[0], 1025);
  BOOST_CHECK_EQUAL(of[5], 1019);
}

BOOST_AUTO_TEST_CASE(k09_host_loop_of_element_maps) {
  const auto got = run_kernel<float, float>(iota_f(64), 64, [](auto &kh, auto x, auto y) { k9(kh, x, y); });
  std::vector<float> want(64);
  for (int i = 0; i < 64; ++i)
    want[i] = 16.f * i + 15.f;
  same(got, want);
  BOOST_CHECK_EQUAL(got[0], 15.f);
  BOOST_CHECK_EQUAL(got[4], 79.f);
}

BOOST_AUTO_TEST_CASE(k10_run_time_component_n_256) {
  const auto got = run_k10(2.f, 1.f, 256, false);
  std::vector<float> want(256);
  for (int i = 0; i < 256; ++i)
    want[i] = 2.f * i + 1.f;
  same(got, want);
  BOOST_CHECK_EQUAL(got[3], 7.f);
}

BOOST_AUTO_TEST_CASE(k10_run_time_component_n_512) {
  const auto got = run_k10(-1.f, 0.5f, 512, false);
  std::vector<float> want(512);
  for (int i = 0; i < 512; ++i)
    want[i] = -1.f * i + 0.5f;
  same(got, want);
  BOOST_CHECK_EQUAL(got[10], -9.5f);
}

BOOST_AUTO_TEST_CASE(k10_run_time_component_tiled_n_300) {
  const auto got = run_k10(2.f, 1.f, 300, true);
  BOOST_REQUIRE_EQUAL(got.size(), 300u);
  std::vector<float> want(300);
  for (int i = 0; i < 300; ++i)
    want[i] = 2.f * i + 1.f;
  same(got, want);
  BOOST_CHECK_EQUAL(got[299], 599.f);
}

BOOST_AUTO_TEST_CASE(k11_two_inplace_contributions) {
  sg::graph g;
  auto ones = input(g, std::vector<float>(64, 1.f));
  auto y = g.create_mem_handle<float>();
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k11(kh, ones, y, 1.f); });
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k11(kh, ones, y, 2.f); });
  std::vector<float> out;
  host_read(g, y, 64, out);
  g.run();
  g.wait();
  same(out, std::vector<float>(64, 3.f));
}

BOOST_AUTO_TEST_CASE(k12_interpolation) {
  std::vector<int> nbr(1024);
  for (int i = 0; i < 256; ++i)
    for (int s = 0; s < 4; ++s)
      nbr[4 * i + s] = (i % 61) + s;
  sg::graph g;
  auto src = input(g, iota_f(64));
  auto nb = input(g, nbr);
  auto w = input(g, std::vector<float>(1024, 0.25f));
  auto y = g.create_mem_handle<float>();
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k12(kh, src, nb, w, y); });
  std::vector<float> out;
  host_read(g, y, 256, out);
  g.run();
  g.wait();
  std::vector<float> want(256);
  for (int i = 0; i < 256; ++i)
    want[i] = 0.25f * (4 * (i % 61) + 6);
  same(out, want);
  BOOST_CHECK_EQUAL(out[0], 1.5f);
  BOOST_CHECK_EQUAL(out[100], 40.5f);
}

BOOST_AUTO_TEST_CASE(k13_two_dimensional_five_point_stencil) {
  std::vector<float> x(64 * 64);
  for (int r = 0; r < 64; ++r)
    for (int c = 0; c < 64; ++c)
      x[r * 64 + c] = static_cast<float>(64 * r + c);
  const auto got = run_kernel<float, float>(x, 64 * 64, [](auto &kh, auto in, auto y) { k13(kh, in, y); });
  std::vector<float> want(64 * 64);
  auto at = [&](int r, int c) { return (r < 0 || r > 63 || c < 0 || c > 63) ? 0.f : x[r * 64 + c]; };
  for (int r = 0; r < 64; ++r)
    for (int c = 0; c < 64; ++c)
      want[r * 64 + c] = at(r, c) + at(r - 1, c) + at(r + 1, c) + at(r, c - 1) + at(r, c + 1);
  same(got, want);
  BOOST_CHECK_EQUAL(got[0], 65.f);
  BOOST_CHECK_EQUAL(got[64 + 1], 325.f);
}

BOOST_AUTO_TEST_CASE(k14_row_sums_at_cap_1024) {
  if (!group_of_1024_fits())
    return;
  const auto got = run_k14(1024, false);
  std::vector<int> want(64);
  for (int r = 0; r < 64; ++r)
    want[r] = 1024 * r + 523776;
  same(got, want);
  BOOST_CHECK_EQUAL(got[0], 523776);
  BOOST_CHECK_EQUAL(got[63], 588288);
}

BOOST_AUTO_TEST_CASE(k14_row_sums_at_cap_256) {
  const auto got = run_k14(256, true);
  std::vector<int> want(64);
  for (int r = 0; r < 64; ++r)
    want[r] = 1024 * r + 523776;
  same(got, want);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(task_code)

BOOST_AUTO_TEST_CASE(k4_spot_values) {
  const auto got = run_kernel<int, int>(iota_i(1024), 1024, [](auto &kh, auto x, auto y) { k4(kh, x, y); });
  BOOST_CHECK_EQUAL(got[0], 1);
  BOOST_CHECK_EQUAL(got[1], 3);
  BOOST_CHECK_EQUAL(got[1023], 2045);
}

BOOST_AUTO_TEST_CASE(a_run_of_one_item_on_a_group_of_64_gives_each_tile_its_own_total) {
  const auto got = run_kernel<int, int>(std::vector<int>(1024, 1), 16, [](auto &kh, auto x_in, auto totals_out) {
    auto x = kh.source(x_in, sg::shape<1>({1024}));
    auto x_tiles = kh.tile(x, {64});
    auto work = kh.work_tiles({16}, {1}, 64);
    auto scanned = kh.map_work(work, [=](auto unit) {
      auto x_tile = x_tiles[unit.tile()];
      auto x_run = sg::flatten(x_tile);
      return unit.run(x_run, scan_items);
    });
    auto totals = kh.map_work(work, [=](auto unit) {
      auto my_scan = scanned[unit.tile()];
      auto my_last = sg::slice(my_scan, {0, 63}, {1, 1});
      auto last_run = sg::flatten(my_last);
      return unit.run(last_run, pass_items);
    });
    auto totals_rows = kh.retile(totals, {1});
    kh.evaluate(totals_out, kh.join(totals_rows));
  });
  same(got, std::vector<int>(16, 64));
}

BOOST_AUTO_TEST_CASE(maps_of_run_3_and_run_2_over_one_work_set_match_sequential_references) {
  sg::graph g;
  auto xh = input(g, iota_i(192));
  auto y3h = g.create_mem_handle<int>();
  auto y2h = g.create_mem_handle<int>();
  g.submit(test_device(), [&](sg::kernel_handle &kh) {
    auto x = kh.source(xh, sg::shape<1>({192}));
    auto ts = kh.tile(x, {3});
    auto work = kh.work_tiles({1}, {64}, 3);
    auto y3 = kh.map_work(work, [=](auto unit) {
      auto u = unit.id();
      auto tile = ts[u[0]];
      auto run = sg::flatten(tile);
      return unit.run(run, sum_items);
    });
    auto y2 = kh.map_work(work, [=](auto unit) {
      auto u = unit.id();
      auto tile = ts[u[0]];
      auto head = sg::slice(tile, {0}, {2});
      auto run = sg::flatten(head);
      return unit.run(run, sum_items);
    });
    kh.evaluate(y3h, kh.join(y3));
    kh.evaluate(y2h, kh.join(y2));
  });
  std::vector<int> o3, o2;
  host_read(g, y3h, 64, o3);
  host_read(g, y2h, 64, o2);
  g.run();
  g.wait();
  std::vector<int> w3(64), w2(64);
  for (int u = 0; u < 64; ++u) {
    w3[u] = 9 * u + 3;
    w2[u] = 6 * u + 1;
  }
  same(o3, w3);
  same(o2, w2);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(item_kernels)

BOOST_AUTO_TEST_CASE(k9_gives_16i_plus_15) {
  const auto got = run_kernel<float, float>(iota_f(64), 64, [](auto &kh, auto x, auto y) { k9(kh, x, y); });
  BOOST_CHECK_EQUAL(got[0], 15.f);
  BOOST_CHECK_EQUAL(got[4], 79.f);
}

BOOST_AUTO_TEST_CASE(a_range_of_1000003_items_matches_the_reference_and_the_last_element_is_written) {
  const int n = 1000003;
  std::vector<float> x(n);
  for (int i = 0; i < n; ++i)
    x[i] = static_cast<float>(i % 1000);
  const auto got = run_kernel<float, float>(x, n, [n](auto &kh, auto in, auto y) { k9(kh, in, y, 1, n); });
  BOOST_REQUIRE_EQUAL(got.size(), static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    BOOST_REQUIRE_EQUAL(got[i], 2.f * x[i] + 1.f);
}

BOOST_AUTO_TEST_SUITE_END()
