#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include <vector>

using namespace sgt;

namespace {

sg::graph_options cap_256() {
  sg::graph_options o;
  o.group_size_cap = 256;
  return o;
}

template <class Build>
std::vector<int> run_segmented(const std::vector<int> &x, std::size_t out_n, Build build) {
  return run_kernel<int, int>(x, out_n, build, cap_256());
}

} // namespace

BOOST_AUTO_TEST_SUITE(segmented_runs)

BOOST_AUTO_TEST_CASE(row_0_of_k14_joins_four_segment_partials_into_523776) {
  std::vector<int> x(64 * 1024);
  for (int r = 0; r < 64; ++r)
    for (int c = 0; c < 1024; ++c)
      x[r * 1024 + c] = r + c;
  const auto got = run_segmented(x, 64, [](auto &kh, auto xh, auto yh) { k14(kh, xh, yh, true); });
  BOOST_CHECK_EQUAL(got[0], 523776);
  // the four partials of row 0 are the sums of its four runs of 256
  int partials[4] = {};
  for (int s = 0; s < 4; ++s)
    for (int c = 256 * s; c < 256 * (s + 1); ++c)
      partials[s] += c;
  BOOST_CHECK_EQUAL(partials[0], 32640);
  BOOST_CHECK_EQUAL(partials[1], 98176);
  BOOST_CHECK_EQUAL(partials[2], 163712);
  BOOST_CHECK_EQUAL(partials[3], 229248);
  BOOST_CHECK_EQUAL(partials[0] + partials[1] + partials[2] + partials[3], 523776);
}

BOOST_AUTO_TEST_CASE(a_scan_of_1024_ones_in_four_segments_adds_the_earlier_totals) {
  const auto got = run_segmented(std::vector<int>(1024, 1), 1024, [](auto &kh, auto x_in, auto y_out) {
    auto x = kh.source(x_in, sg::shape<1>({1024}));
    auto ts = kh.tile(x, {1024});
    auto run = sg::segment_items(4, 256);
    auto work = kh.work_tiles({1}, {1}, run);
    auto y = kh.map_work(work, [=](auto unit) {
      auto tile = ts[unit.tile()];
      auto flat = sg::flatten(tile);
      return unit.run_distributed(flat, scan_items);
    });
    kh.evaluate(y_out, kh.join(kh.retile(y, {1024})));
  });
  std::vector<int> want(1024);
  for (int i = 0; i < 1024; ++i)
    want[i] = i + 1;
  BOOST_REQUIRE_EQUAL(got.size(), want.size());
  for (int i = 0; i < 1024; ++i)
    BOOST_REQUIRE_EQUAL(got[i], want[i]);
}

BOOST_AUTO_TEST_CASE(tile_bin_of_1024_values_in_four_segments_gives_64_per_bin) {
  std::vector<int> v(1024);
  for (int i = 0; i < 1024; ++i)
    v[i] = i % 16;
  const auto got = run_segmented(v, 16, [](auto &kh, auto x_in, auto y_out) {
    auto x = kh.source(x_in, sg::shape<1>({1024}));
    auto ts = kh.tile(x, {1024});
    auto run = sg::segment_items(4, 256);
    auto work = kh.work_tiles({1}, {1}, run);
    auto bins = kh.map_work(work, [=](auto unit) {
      auto tile = ts[unit.tile()];
      auto flat = sg::flatten(tile);
      return unit.run_distributed(flat, bin_items);
    });
    kh.evaluate(y_out, kh.join(kh.retile(bins, {16})));
  });
  BOOST_REQUIRE_EQUAL(got.size(), 16u);
  for (int b : got)
    BOOST_CHECK_EQUAL(b, 64);
}

BOOST_AUTO_TEST_CASE(a_replicated_reduce_of_1024_ones_gives_1024) {
  const auto got = run_segmented(std::vector<int>(1024, 1), 1, [](auto &kh, auto x_in, auto y_out) {
    auto x = kh.source(x_in, sg::shape<1>({1024}));
    auto ts = kh.tile(x, {1024});
    auto run = sg::segment_items(4, 256);
    auto work = kh.work_tiles({1}, {1}, run);
    auto total = kh.map_work(work, [=](auto unit) {
      auto tile = ts[unit.tile()];
      auto flat = sg::flatten(tile);
      return unit.run_replicated(flat, sum_items);
    });
    kh.evaluate(y_out, kh.join(total));
  });
  BOOST_REQUIRE_EQUAL(got.size(), 1u);
  BOOST_CHECK_EQUAL(got[0], 1024);
}

BOOST_AUTO_TEST_SUITE_END()
