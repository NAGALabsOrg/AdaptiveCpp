#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include <string>
#include <vector>

using namespace sgt;

namespace {

template <class F> std::string failure(F f) {
  try {
    f();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "no error";
}

bool has(const std::string &s, const std::string &part) { return s.find(part) != std::string::npos; }

void expect(const std::string &message, std::initializer_list<const char *> fragments) {
  for (const char *f : fragments)
    BOOST_CHECK_MESSAGE(has(message, f), "\"" << f << "\" is missing from: " << message);
}

sg::graph_options cap(int c) {
  sg::graph_options o;
  o.group_size_cap = c;
  return o;
}

template <class T> H<T> with(sg::graph &g, std::vector<T> v) {
  auto h = g.create_mem_handle<T>();
  host_write(g, h, std::move(v));
  return h;
}

} // namespace

BOOST_AUTO_TEST_SUITE(rejections)

BOOST_AUTO_TEST_CASE(r01_work_tile_above_the_cap) {
  sg::graph g(cap(1024));
  expect(failure([&] {
           g.submit(test_device(), [&](sg::kernel_handle &kh) { kh.work_tiles({8, 8}, {16, 16}, 128); });
         }),
         {"kh.work_tiles", "32768", "1024"});
}

BOOST_AUTO_TEST_CASE(r02_native_shape_above_3d) {
  sg::graph g;
  expect(failure([&] { g.submit(test_device(), [&](sg::kernel_handle &kh) { kh.work_tiles({4}, {2, 2, 2}, 4); }); }),
         {"kh.work_tiles", "3", "4"});
}

BOOST_AUTO_TEST_CASE(r03_tile_dependency_cycle) {
  sg::graph g;
  auto s = with(g, iota_i(256));
  auto ix = with(g, iota_i(256));
  auto y = g.create_mem_handle<int>();
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k5(kh, s, ix, y, true); });
  expect(failure([&] { g.run(); }),
         {"kh.index_map", "work set 0 tile 0", "work set 0 tile 1", "work set 0 tile 2", "work set 0 tile 3"});
}

BOOST_AUTO_TEST_CASE(r04_plain_mark_local_on_a_distributed_result) {
  sg::graph g(cap(256));
  auto x = g.create_mem_handle<int>(), y = g.create_mem_handle<int>();
  expect(failure([&] { g.submit(test_device(), [&](sg::kernel_handle &kh) { k14(kh, x, y, true, true); }); }),
         {"kh.mark_local", "4"});
}

BOOST_AUTO_TEST_CASE(r05_segmented_mark_local_on_a_result_that_is_not_distributed) {
  sg::graph g;
  auto x = g.create_mem_handle<float>();
  expect(failure([&] {
           g.submit(test_device(), [&](sg::kernel_handle &kh) {
             auto xs = kh.source(x, sg::shape<1>({64}));
             auto ts = kh.tile(xs, {4});
             auto work1 = kh.work_tiles({16}, {1}, 4);
             auto l1 = sum_tiles(kh, work1, ts);
             kh.mark_local<sg::segmented>(l1);
           });
         }),
         {"kh.mark_local<sg::segmented>", "1"});
}

BOOST_AUTO_TEST_CASE(r06_replicated_run_over_a_distributed_result_of_its_own_set) {
  sg::graph g(cap(256));
  auto x = with(g, std::vector<int>(64 * 1024, 1));
  auto y = g.create_mem_handle<int>();
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k14(kh, x, y, true, false, true); });
  expect(failure([&] { g.run(); }), {"unit.run_replicated", "4"});
}

BOOST_AUTO_TEST_CASE(r07_work_tiles_and_work_items_in_one_kernel) {
  sg::graph g;
  auto x = g.create_mem_handle<float>(), y = g.create_mem_handle<float>(), total = g.create_mem_handle<float>();
  expect(failure([&] {
           g.submit(test_device(), [&](sg::kernel_handle &kh) {
             k9(kh, x, y);
             k1(kh, x, total);
           });
         }),
         {"work_tile_sets = 1", "work_item_sets = 1", "kh.work_tiles", "kh.work_items"});
}

BOOST_AUTO_TEST_CASE(r08_launch_value_as_structure_is_a_compile_error) {
  const auto &c = sg::rejection_catalogue();
  BOOST_REQUIRE_GE(c.size(), 8u);
  BOOST_CHECK_EQUAL(c[7].number, 8);
  BOOST_CHECK_EQUAL(std::string(c[7].construct), "launch_value");
}

BOOST_AUTO_TEST_CASE(r09_mark_local_of_array_tiles_given_result_tiles) {
  sg::graph g;
  auto x = g.create_mem_handle<float>();
  expect(failure([&] {
           g.submit(test_device(), [&](sg::kernel_handle &kh) {
             auto xs = kh.source(x, sg::shape<1>({64}));
             auto ts = kh.tile(xs, {4});
             auto work1 = kh.work_tiles({16}, {1}, 4);
             auto l1 = sum_tiles(kh, work1, ts);
             kh.mark_local(l1, work1);
           });
         }),
         {"kh.mark_local", "array tiles"});
}

BOOST_AUTO_TEST_CASE(r10_item_reading_a_result_array_at_another_index) {
  sg::graph g;
  auto x = g.create_mem_handle<float>();
  expect(failure([&] {
           g.submit(test_device(), [&](sg::kernel_handle &kh) {
             auto xs = kh.source(x, sg::shape<1>({64}));
             auto items = kh.work_items({64});
             auto first = kh.map_items(items, [=](auto item) {
               auto v = xs[item.id()];
               return item.compute(v, step_fn);
             });
             auto prev = first.named("prev");
             kh.map_items(items, [=](auto item) {
               auto v = prev[item.id() + 1];
               return item.compute(v, step_fn);
             });
           });
         }),
         {"kh.map_items", "prev", "1"});
}

BOOST_AUTO_TEST_CASE(r11_result_array_fed_to_work_tiles) {
  sg::graph g;
  auto x = g.create_mem_handle<float>();
  expect(failure([&] {
           g.submit(test_device(), [&](sg::kernel_handle &kh) {
             auto xs = kh.source(x, sg::shape<1>({64}));
             auto items = kh.work_items({64});
             auto first = kh.map_items(items, [=](auto item) {
               auto v = xs[item.id()];
               return item.compute(v, step_fn);
             });
             kh.tile(first, {4});
           });
         }),
         {"kh.map_work", "kh.map_items"});
}

BOOST_AUTO_TEST_CASE(r12_join_over_a_grid_of_another_rank) {
  sg::graph g;
  auto h = g.create_mem_handle<int>();
  expect(failure([&] {
           g.submit(test_device(), [&](sg::kernel_handle &kh) {
             auto x = kh.source(h, sg::shape<2>({4, 4}));
             (void)x;
             auto star = kh.stencil({{0, 0}, {1, 0}}, sg::range({0, 0}, {4, 4}));
             kh.join(star);
           });
         }),
         {"kh.join", "2", "1"});
}

BOOST_AUTO_TEST_CASE(r13_slice_beyond_its_work_set) {
  sg::graph g;
  auto x = with(g, iota_f(64));
  g.submit(test_device(), [&](sg::kernel_handle &kh) {
    auto xs = kh.source(x, sg::shape<1>({64}));
    auto work1 = kh.work_tiles({16}, {1}, 4);
    kh.slice_work(work1, {15}, {2}, {1}, {1}, 4);
  });
  expect(failure([&] { g.run(); }), {"kh.slice_work", "15", "2", "1", "16"});
}

BOOST_AUTO_TEST_CASE(r14_run_input_longer_than_the_run) {
  sg::graph g;
  auto x = with(g, iota_i(8));
  g.submit(test_device(), [&](sg::kernel_handle &kh) {
    auto xs = kh.source(x, sg::shape<1>({8}));
    auto ts = kh.tile(xs, {8});
    auto work = kh.work_tiles({1}, {1}, 4);
    kh.map_work(work, [=](auto unit) {
      auto tile = ts[unit.tile()];
      auto run = sg::flatten(tile);
      return unit.run(run, sum_items);
    });
  });
  expect(failure([&] { g.run(); }), {"unit.run", "8", "4"});
}

BOOST_AUTO_TEST_CASE(r15_interop_is_not_designed) {
  sg::graph g;
  expect(failure([&] { g.submit<sg::interop>(test_device(), 1); }), {"sg::interop"});
}

BOOST_AUTO_TEST_SUITE_END()
