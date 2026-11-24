#include "../common/check.hpp"
#include "../common/graphs.hpp"
#include "../common/mocks.hpp"

#include <algorithm>
#include <set>
#include <string>

using namespace sgt;

BOOST_AUTO_TEST_SUITE(roles)

BOOST_AUTO_TEST_CASE(a_handle_is_a_source_or_a_result_not_both) {
  rig r;
  auto h = r.g->create_mem_handle<float>();
  try {
    r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) {
      auto x = kh.source(h, sg::shape<1>({64}));
      auto items = kh.work_items({64});
      auto y = kh.map_items(items, [=](auto item) {
        auto v = x[item.id()];
        return item.compute(v, step_fn);
      });
      kh.evaluate(h, y);
    });
    BOOST_FAIL("no throw");
  } catch (const sg::error &e) {
    BOOST_CHECK(e.code() == sg::errc::build);
    BOOST_CHECK(std::string(e.what()).find("handle 0") != std::string::npos);
  }
}

BOOST_AUTO_TEST_CASE(the_graph_example_orders_host_work_before_the_kernel) {
  rig r;
  auto x = r.g->create_mem_handle<float>();
  auto n = r.g->create_mem_handle<int>();
  auto y = r.g->create_mem_handle<float>();
  host_write(*r.g, n, std::vector<int>{64});
  host_write(*r.g, x, iota_f(64));
  r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { rerun_a(kh, x, n, y); });
  r.g->run();
  BOOST_REQUIRE_EQUAL(r.ops->hosts.size(), 2u);
  BOOST_REQUIRE_EQUAL(r.ops->launches.size(), 1u);
  BOOST_REQUIRE_EQUAL(r.launch->deps.size(), 1u);
  const auto &deps = r.launch->deps[0];
  // the two host nodes are operations 1 and 2; the launch waits on both
  BOOST_CHECK(std::find(deps.begin(), deps.end(), 1u) != deps.end());
  BOOST_CHECK(std::find(deps.begin(), deps.end(), 2u) != deps.end());
  BOOST_CHECK_EQUAL(r.ops->log.back(), "launch");
  // nothing waits on the kernel: its operation is the last one and no host node lists it
  const auto launch_id = r.ops->next;
  for (const auto &h : r.ops->hosts)
    BOOST_CHECK(std::find(h.deps.begin(), h.deps.end(), launch_id) == h.deps.end());
  BOOST_CHECK(std::all_of(deps.begin(), deps.end(), [&](auto d) { return d < launch_id; }));
}

BOOST_AUTO_TEST_CASE(a_read_before_any_producer_is_an_error) {
  rig r;
  auto x = r.g->create_mem_handle<float>();
  auto y = r.g->create_mem_handle<float>();
  r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { k9(kh, x, y); });
  SG_CHECK_ERRC(r.g->run(), never_produced);
}

BOOST_AUTO_TEST_CASE(inplace_with_an_operation_that_cannot_combine_is_a_compile_error) {
  // lit/fail/inplace_op_not_invocable.cpp holds the program; here only the positive form
  static_assert(std::is_invocable_r_v<float, sycl::plus<float>, float, float>);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(errors)

BOOST_AUTO_TEST_CASE(text_has_construct_location_explanation_and_numbers) {
  const sg::build_error e(1, "kh.work_tiles", {"f.cpp", 10}, "the work tile does not fit", {{"cap", 1024}});
  const std::string w = e.what();
  BOOST_CHECK_EQUAL(w, "kh.work_tiles at f.cpp:10: the work tile does not fit (cap = 1024)");
  BOOST_CHECK_EQUAL(e.rule(), 1);
  BOOST_CHECK_EQUAL(e.construct(), "kh.work_tiles");
  BOOST_REQUIRE_EQUAL(e.numbers().size(), 1u);
  BOOST_CHECK_EQUAL(e.numbers()[0].value, 1024);
}

BOOST_AUTO_TEST_CASE(numbers_are_rendered_in_order) {
  const sg::build_error e(2, "kh.work_tiles", {"a.cpp", 3}, "x", {{"tile_rank", 3}, {"run", 4}});
  BOOST_CHECK(std::string(e.what()).find("(tile_rank = 3, run = 4)") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(the_catalogue_lists_fifteen_rejections) {
  const auto &c = sg::rejection_catalogue();
  BOOST_REQUIRE_EQUAL(c.size(), 15u);
  for (std::size_t i = 0; i < c.size(); ++i)
    BOOST_CHECK_EQUAL(c[i].number, static_cast<int>(i) + 1);
  const char *constructs[15] = {"kh.work_tiles",
                                "kh.work_tiles",
                                "kh.index_map",
                                "kh.mark_local",
                                "kh.mark_local<sg::segmented>",
                                "unit.run_replicated",
                                "kh.map_items",
                                "launch_value",
                                "kh.mark_local",
                                "kh.map_items",
                                "kh.map_work",
                                "kh.join",
                                "kh.slice_work",
                                "unit.run",
                                "sg::interop"};
  for (std::size_t i = 0; i < 15; ++i)
    BOOST_CHECK_EQUAL(std::string(c[i].construct), constructs[i]);
}

BOOST_AUTO_TEST_SUITE_END()
