#include "../common/check.hpp"
#include "../common/graphs.hpp"
#include "../common/mocks.hpp"

#include "hipSYCL/sg/detail/dev/view_read.hpp"

#include <set>
#include <string>

using namespace sgt;

namespace {

// Builds one kernel on the mocks and prepares it, so views can be read on the host.
struct view_lab {
  rig r;
  std::unique_ptr<prepared_kernel> pk;
  std::vector<int> xs;

  template <class T, class Build> void build(Build b, const std::vector<T> &data, std::vector<std::int64_t> values = {}) {
    auto x = r.g->create_mem_handle<T>();
    r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { b(kh, x); });
    pk = std::make_unique<prepared_kernel>(
        kernel_preparer::prepare(r.impl->nodes().at(0)->graph, values));
    source_data_.assign(reinterpret_cast<const std::byte *>(data.data()),
                        reinterpret_cast<const std::byte *>(data.data()) + data.size() * sizeof(T));
    for (std::size_t v = 0; v < pk->views.size(); ++v)
      if (static_cast<vk>(pk->views[v].kind) == vk::source)
        pk->views[v].base = reinterpret_cast<std::uint64_t>(source_data_.data());
  }

  template <class T> T read(int view, std::int64_t tile, std::int64_t local) {
    dev::plain_visitor w;
    std::byte out[16] = {};
    dev::read_view(pk->views.data(), view, tile, local, nullptr, w, out);
    T v;
    std::memcpy(&v, out, sizeof(T));
    return v;
  }

  std::vector<std::byte> source_data_;
};

} // namespace

BOOST_AUTO_TEST_SUITE(closures_and_launch)

BOOST_AUTO_TEST_CASE(closure_round_trips_byte_identical) {
  const std::int8_t a = -5;
  const std::uint16_t b = 40000;
  const float c = 2.5f;
  auto fn = [=](int v) { return static_cast<float>(v + a + b) * c; };
  const closure_entry e = capture_closure(fn);
  BOOST_REQUIRE_EQUAL(e.bytes.size(), sizeof(fn));
  alignas(decltype(fn)) std::byte copy[sizeof(fn)];
  std::memcpy(copy, e.bytes.data(), sizeof(fn));
  BOOST_CHECK_EQUAL(std::memcmp(copy, &fn, sizeof(fn)), 0);
  BOOST_CHECK_EQUAL((*reinterpret_cast<const decltype(fn) *>(copy))(3), fn(3));
}

BOOST_AUTO_TEST_CASE(symbol_names_follow_the_function_type) {
  auto make = [] { return [](auto, int v) { return v + 1; }; };
  auto one = make();
  using One = decltype(one);
  const std::string first = tile_task_ref<One, int>().symbol;
  const std::string again = tile_task_ref<One, int>().symbol;
  BOOST_CHECK_EQUAL(first, again);
  BOOST_CHECK(!first.empty());
  auto other_scope = [] { return [](auto, int v) { return v + 1; }; };
  auto two = other_scope();
  using Two = decltype(two);
  const std::string second = tile_task_ref<Two, int>().symbol;
  BOOST_CHECK_NE(first, second);
}

BOOST_AUTO_TEST_CASE(launch_record_has_one_aligned_data_argument) {
  rig r;
  auto x = r.g->create_mem_handle<float>();
  auto total = r.g->create_mem_handle<float>();
  host_write(*r.g, x, iota_f(64));
  r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { k1(kh, x, total); });
  r.g->run();
  BOOST_REQUIRE_EQUAL(r.ops->launches.size(), 1u);
  const auto &d = r.ops->launches[0].desc;
  BOOST_REQUIRE_EQUAL(d.data_args.size(), 1u);
  BOOST_CHECK_EQUAL(d.data_args[0] % 256, 0u);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(launch_values)

BOOST_AUTO_TEST_CASE(division_is_a_bound_expression) {
  sg::launch_value<int> n(0, 0);
  const sg::bound_expr q = n / 4;
  sg::detail::eval_env env;
  std::int64_t v64[1] = {64};
  env.launch = v64;
  env.launch_count = 1;
  BOOST_CHECK_EQUAL(q.evaluate(env), 16);
  std::int64_t v128[1] = {128};
  env.launch = v128;
  BOOST_CHECK_EQUAL(q.evaluate(env), 32);
  BOOST_CHECK_EQUAL(((n + 3) * 2 - 1).evaluate(env), 261);
  BOOST_CHECK_EQUAL((n % 100).evaluate(env), 28);
}

BOOST_AUTO_TEST_CASE(constants_fold) {
  const sg::bound_expr e = sg::bound_expr(6) * sg::bound_expr(7);
  BOOST_CHECK(e.constant());
  BOOST_CHECK_EQUAL(e.constant_value(), 42);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(views)

BOOST_AUTO_TEST_CASE(partial_last_tile_is_padded_and_join_trims) {
  view_lab lab;
  int tiled = -1, joined = -1;
  lab.build<int>(
      [&](sg::kernel_handle &kh, sg::mem_handle<int> h) {
        auto x = kh.source(h, sg::shape<1>({10}));
        auto t = kh.tile(x, {4}, sg::fill{-7});
        tiled = t.id();
        joined = kh.join(t).id();
      },
      iota_i(10));
  BOOST_CHECK_EQUAL(lab.pk->views[tiled].grid[0], 3);
  BOOST_CHECK_EQUAL(lab.read<int>(tiled, 2, 1), 9);
  BOOST_CHECK_EQUAL(lab.read<int>(tiled, 2, 2), -7);
  BOOST_CHECK_EQUAL(lab.read<int>(tiled, 2, 3), -7);
  BOOST_CHECK_EQUAL(lab.pk->views[joined].shape[0], 10);
}

BOOST_AUTO_TEST_CASE(stencil_edges_go_to_the_out_of_range_policy) {
  view_lab lab;
  int windows = -1;
  lab.build<int>(
      [&](sg::kernel_handle &kh, sg::mem_handle<int> h) {
        auto x = kh.source(h, sg::shape<1>({1024}));
        auto star = kh.stencil({-1, 0, 1}, sg::range(0, 1024));
        windows = kh.index_map(star, sg::fill{-1}, x).id();
      },
      iota_i(1024, 100));
  BOOST_CHECK_EQUAL(lab.pk->views[windows].grid[0], 1024);
  BOOST_CHECK_EQUAL(lab.read<int>(windows, 0, 0), -1);
  BOOST_CHECK_EQUAL(lab.read<int>(windows, 0, 1), 100);
  BOOST_CHECK_EQUAL(lab.read<int>(windows, 0, 2), 101);
  BOOST_CHECK_EQUAL(lab.read<int>(windows, 1023, 0), 100 + 1022);
  BOOST_CHECK_EQUAL(lab.read<int>(windows, 1023, 1), 100 + 1023);
  BOOST_CHECK_EQUAL(lab.read<int>(windows, 1023, 2), -1);
}

BOOST_AUTO_TEST_CASE(retile_regroups_in_tile_order) {
  view_lab lab;
  int regrouped = -1;
  lab.build<int>(
      [&](sg::kernel_handle &kh, sg::mem_handle<int> h) {
        auto x = kh.source(h, sg::shape<1>({16}));
        auto ones = kh.tile(x, {1});
        regrouped = kh.retile(ones, {4}).id();
      },
      iota_i(16));
  BOOST_CHECK_EQUAL(lab.pk->views[regrouped].grid[0], 4);
  for (int t = 0; t < 4; ++t)
    for (int l = 0; l < 4; ++l)
      BOOST_CHECK_EQUAL(lab.read<int>(regrouped, t, l), 4 * t + l);
}

BOOST_AUTO_TEST_CASE(join_of_tiles_with_another_grid_rank_is_rejected) {
  rig r;
  auto h = r.g->create_mem_handle<int>();
  try {
    r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) {
      auto x = kh.source(h, sg::shape<2>({4, 4}));
      (void)x;
      auto star = kh.stencil({{0, 0}, {1, 0}}, sg::range({0, 0}, {4, 4}));
      kh.join(star);
    });
    BOOST_FAIL("no throw");
  } catch (const sg::build_error &e) {
    const std::string w = e.what();
    BOOST_CHECK(w.find("kh.join") != std::string::npos);
    BOOST_CHECK(w.find("grid_rank = 2") != std::string::npos);
    BOOST_CHECK(w.find("tile_rank = 1") != std::string::npos);
    BOOST_CHECK_EQUAL(e.rule(), 12);
  }
}

BOOST_AUTO_TEST_CASE(extend_repeats_the_element) {
  view_lab lab;
  int wide = -1;
  lab.build<int>(
      [&](sg::kernel_handle &kh, sg::mem_handle<int> h) {
        auto x = kh.source(h, sg::shape<1>({4}));
        auto ones = kh.tile(x, {1});
        wide = kh.extend_tiles(ones, {1, 64}, sg::repeat).id();
      },
      iota_i(4, 10));
  for (int l = 0; l < 64; ++l)
    BOOST_CHECK_EQUAL(lab.read<int>(wide, 2, l), 12);
}

BOOST_AUTO_TEST_CASE(tiles_and_elements_count_through_the_expression) {
  view_lab lab;
  sg::bound_expr count;
  int t = -1;
  lab.build<int>(
      [&](sg::kernel_handle &kh, sg::mem_handle<int> h) {
        auto x = kh.source(h, sg::shape<1>({64}));
        auto tt = kh.tile(x, {4});
        t = tt.id();
        count = tt.count();
      },
      iota_i(64));
  sg::detail::eval_env env;
  env.count_of = [&](std::int64_t v, int dim) { return dim < 0 ? prod3(lab.pk->views[v].grid) : lab.pk->views[v].grid[dim]; };
  BOOST_CHECK_EQUAL(count.evaluate(env), 16);
  BOOST_CHECK_EQUAL(t, 1);
}

BOOST_AUTO_TEST_SUITE_END()
