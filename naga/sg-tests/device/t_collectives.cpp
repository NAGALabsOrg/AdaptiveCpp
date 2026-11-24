#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

using namespace sgt;

namespace {

struct S4 {
  std::int16_t a;
  std::uint8_t b, c;
  friend bool operator==(const S4 &x, const S4 &y) { return x.a == y.a && x.b == y.b && x.c == y.c; }
};
struct F3 {
  float x, y, z;
  friend bool operator==(const F3 &p, const F3 &q) { return p.x == q.x && p.y == q.y && p.z == q.z; }
};
struct M2 {
  std::uint32_t m[4];
  friend bool operator==(const M2 &p, const M2 &q) {
    return p.m[0] == q.m[0] && p.m[1] == q.m[1] && p.m[2] == q.m[2] && p.m[3] == q.m[3];
  }
};

struct add_op {
  template <class T> T operator()(T a, T b) const { return static_cast<T>(a + b); }
  S4 operator()(S4 a, S4 b) const {
    return S4{static_cast<std::int16_t>(a.a + b.a), static_cast<std::uint8_t>(a.b + b.b),
              static_cast<std::uint8_t>(a.c + b.c)};
  }
  F3 operator()(F3 a, F3 b) const { return F3{a.x + b.x, a.y + b.y, a.z + b.z}; }
};
struct matmul_op {
  M2 operator()(M2 a, M2 b) const {
    return M2{{a.m[0] * b.m[0] + a.m[1] * b.m[2], a.m[0] * b.m[1] + a.m[1] * b.m[3],
               a.m[2] * b.m[0] + a.m[3] * b.m[2], a.m[2] * b.m[1] + a.m[3] * b.m[3]}};
  }
};

template <class T> T make(int i) {
  if constexpr (std::is_same_v<T, S4>)
    return S4{static_cast<std::int16_t>(i % 13 - 6), static_cast<std::uint8_t>(i % 7), static_cast<std::uint8_t>(i % 5)};
  else if constexpr (std::is_same_v<T, F3>)
    return F3{static_cast<float>(i % 9), static_cast<float>(i % 4) - 1.f, static_cast<float>(i % 6)};
  else if constexpr (std::is_same_v<T, M2>)
    return M2{{static_cast<std::uint32_t>(1 + i % 3), static_cast<std::uint32_t>(i % 2), static_cast<std::uint32_t>(i % 3),
               static_cast<std::uint32_t>(1 + i % 2)}};
  else
    return static_cast<T>((i * 7 + 3) % 11 - 5);
}

template <class T> void show(std::string &s, const T &v) {
  if constexpr (std::is_arithmetic_v<T>)
    s += std::to_string(static_cast<double>(v));
  else
    s += "{struct}";
}

int max_group() { return static_cast<int>(test_device().get_info<sycl::info::device::max_work_group_size>()); }

// one result per tile (reduce) or per item (scan); `units` units of `run` items in each of `tiles` work tiles
template <class T, class Op, bool Scan>
std::vector<T> run_on_device(const std::vector<T> &x, int run, int tiles, int units) {
  sg::graph g;
  auto xh = g.create_mem_handle<T>();
  auto yh = g.create_mem_handle<T>();
  host_write(g, xh, x);
  const int per_tile = run * units;
  const int out_n = Scan ? tiles * per_tile : tiles * units;
  g.submit(test_device(), [&](sg::kernel_handle &kh) {
    auto xs = kh.source(xh, sg::shape<1>({x.size()}));
    auto ts = kh.tile(xs, {run});
    auto work = kh.work_tiles({tiles}, {units}, run);
    auto fn = [](auto item, auto v) {
      if constexpr (Scan)
        return sg::inclusive_scan(item, v, Op{});
      else
        return sg::reduce(item, v, Op{});
    };
    auto r = kh.map_work(work, [=](auto unit) {
      auto t = unit.tile();
      auto u = unit.id();
      auto tile = ts[units * t[0] + u[0]];
      auto flat = sg::flatten(tile);
      return unit.run(flat, fn);
    });
    if constexpr (Scan)
      kh.evaluate(yh, kh.join(kh.retile(r, {run})));
    else
      kh.evaluate(yh, kh.join(r));
  });
  std::vector<T> out;
  host_read(g, yh, out_n, out);
  g.run();
  g.wait();
  return out;
}

template <class T, class Op, bool Scan> void check_case(int run, int tiles, int units) {
  const int n = run * tiles * units;
  std::vector<T> x(n);
  for (int i = 0; i < n; ++i)
    x[i] = make<T>(i);
  const std::vector<T> got = run_on_device<T, Op, Scan>(x, run, tiles, units);
  std::vector<T> want;
  for (int block = 0; block < tiles * units; ++block) {
    T acc = x[block * run];
    if (Scan)
      want.push_back(acc);
    for (int k = 1; k < run; ++k) {
      acc = Op{}(acc, x[block * run + k]);
      if (Scan)
        want.push_back(acc);
    }
    if (!Scan)
      want.push_back(acc);
  }
  BOOST_REQUIRE_EQUAL(got.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i)
    if (!(got[i] == want[i])) {
      std::string msg = std::string(Scan ? "scan" : "reduce") + " of run " + std::to_string(run) + ", element " +
                        std::to_string(i) + ": got ";
      show(msg, got[i]);
      msg += ", want ";
      show(msg, want[i]);
      BOOST_FAIL(msg);
    }
}

template <class T, class Op> void check_type() {
  for (int run : {1, 3, 4, 5, 64, 100, 1024}) {
    if (run > max_group()) {
      BOOST_TEST_MESSAGE("SKIPPED run " << run << ": the device's largest work-group is " << max_group());
      continue;
    }
    check_case<T, Op, false>(run, 3, 1);
    check_case<T, Op, true>(run, 3, 1);
  }
  check_case<T, Op, false>(3, 1, 64);
  check_case<T, Op, true>(3, 1, 64);
}

} // namespace

BOOST_AUTO_TEST_SUITE(collectives)

BOOST_AUTO_TEST_CASE(int8) { check_type<std::int8_t, add_op>(); }
BOOST_AUTO_TEST_CASE(uint8) { check_type<std::uint8_t, add_op>(); }
BOOST_AUTO_TEST_CASE(int16) { check_type<std::int16_t, add_op>(); }
BOOST_AUTO_TEST_CASE(uint16) { check_type<std::uint16_t, add_op>(); }
BOOST_AUTO_TEST_CASE(int32) { check_type<std::int32_t, add_op>(); }
BOOST_AUTO_TEST_CASE(uint64) { check_type<std::uint64_t, add_op>(); }
BOOST_AUTO_TEST_CASE(float32) { check_type<float, add_op>(); }
BOOST_AUTO_TEST_CASE(struct_of_int16_and_two_uint8) { check_type<S4, add_op>(); }
BOOST_AUTO_TEST_CASE(struct_of_three_floats) { check_type<F3, add_op>(); }

BOOST_AUTO_TEST_CASE(float64_where_the_device_has_it) {
  if (!test_device().has(sycl::aspect::fp64)) {
    BOOST_TEST_MESSAGE("SKIPPED: the device reports no fp64");
    return;
  }
  check_type<double, add_op>();
}

BOOST_AUTO_TEST_CASE(the_two_by_two_matrix_product_is_combined_left_to_right) {
  for (int run : {2, 5, 64, 100}) {
    if (run > max_group())
      continue;
    check_case<M2, matmul_op, false>(run, 3, 1);
    check_case<M2, matmul_op, true>(run, 3, 1);
  }
  check_case<M2, matmul_op, false>(3, 1, 64);
}

BOOST_AUTO_TEST_CASE(tile_bin_with_16_bins_over_256_items_gives_16_per_bin) {
  const auto got = run_kernel<int, int>(
      [] {
        std::vector<int> v(256);
        for (int i = 0; i < 256; ++i)
          v[i] = i % 16;
        return v;
      }(),
      16, [](auto &kh, auto x_in, auto y_out) {
        auto xs = kh.source(x_in, sg::shape<1>({256}));
        auto ts = kh.tile(xs, {256});
        auto work = kh.work_tiles({1}, {1}, 256);
        auto bins = kh.map_work(work, [=](auto unit) {
          auto tile = ts[unit.tile()];
          auto run = sg::flatten(tile);
          return unit.run(run, bin_items);
        });
        kh.evaluate(y_out, kh.join(kh.retile(bins, {16})));
      });
  BOOST_CHECK_EQUAL(got.size(), 16u);
  for (int b : got)
    BOOST_CHECK_EQUAL(b, 16);
}

BOOST_AUTO_TEST_SUITE_END()
