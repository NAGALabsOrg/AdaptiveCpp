#include "../common/check.hpp"
#include "../common/graphs.hpp"
#include "../common/mocks.hpp"

#include <algorithm>
#include <set>
#include <string>

using namespace sgt;

namespace {

template <class T> sg::mem_handle<T> h_of(rig &r) { return r.g->create_mem_handle<T>(); }

std::unique_ptr<prepared_kernel> prep(rig &r, std::size_t node, std::vector<std::int64_t> values = {}) {
  return std::make_unique<prepared_kernel>(kernel_preparer::prepare(r.impl->nodes().at(node)->graph, values));
}

template <class F> void submit_kernel(rig &r, F f) { r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { f(kh); }); }

template <class F> std::string message_of(F f) {
  try {
    f();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "no error";
}

bool has(const std::string &s, const std::string &part) { return s.find(part) != std::string::npos; }

} // namespace

BOOST_AUTO_TEST_SUITE(graph_and_submission)

BOOST_AUTO_TEST_CASE(the_builder_runs_once_at_submit) {
  rig r;
  auto x = h_of<float>(r), total = h_of<float>(r);
  int calls = 0;
  host_write(*r.g, x, iota_f(64));
  r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) {
    ++calls;
    k1(kh, x, total);
  });
  BOOST_CHECK_EQUAL(calls, 1);
  BOOST_CHECK_EQUAL(r.ops->launches.size(), 0u);
  r.g->run();
  r.g->run();
  BOOST_CHECK_EQUAL(calls, 1);
}

BOOST_AUTO_TEST_CASE(a_graph_is_built_once) {
  rig r;
  auto x = h_of<float>(r), total = h_of<float>(r);
  host_write(*r.g, x, iota_f(64));
  r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { k1(kh, x, total); });
  r.g->run();
  const std::string m = message_of([&] { host_write(*r.g, x, iota_f(64)); });
  BOOST_CHECK(has(m, "already built"));
  SG_CHECK_ERRC(r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { k1(kh, x, total); }), build);
}

BOOST_AUTO_TEST_CASE(interop_is_rejected_as_not_designed) {
  rig r;
  const std::string m = message_of([&] { r.g->submit<sg::interop>(sycl::device{}, 1); });
  BOOST_CHECK(has(m, "sg::interop"));
  BOOST_CHECK(has(m, "not designed"));
}

BOOST_AUTO_TEST_CASE(the_group_size_cap_defaults_to_the_device_and_can_be_set) {
  {
    rig r;
    auto x = h_of<float>(r), total = h_of<float>(r);
    r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { k1(kh, x, total); });
    BOOST_CHECK_EQUAL(r.impl->nodes()[0]->graph.group_size_cap, 1024);
  }
  {
    sg::graph_options o;
    o.group_size_cap = 256;
    rig r(o);
    auto x = h_of<float>(r), total = h_of<float>(r);
    r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { k1(kh, x, total); });
    BOOST_CHECK_EQUAL(r.impl->nodes()[0]->graph.group_size_cap, 256);
  }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(work_sets)

BOOST_AUTO_TEST_CASE(k1_sets_and_slices) {
  rig r;
  auto x = h_of<float>(r), total = h_of<float>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) { k1(kh, x, total); });
  auto pk = prep(r, 0);
  BOOST_REQUIRE_EQUAL(pk->works.size(), 3u);
  BOOST_CHECK_EQUAL(pk->works[0].tiles, 16);
  BOOST_CHECK_EQUAL(pk->works[0].tile[0], 1);
  BOOST_CHECK_EQUAL(pk->works[0].run, 4);
  BOOST_CHECK(pk->works[1].slice && pk->works[2].slice);
  const tile_graph tg = build_tile_graph(*pk);
  BOOST_CHECK_EQUAL(tg.groups.size(), 16u);
  BOOST_CHECK_EQUAL(pk->works[1].root, 0);
  BOOST_CHECK_EQUAL(pk->works[2].root, 0);
  for (const auto &group : tg.groups)
    BOOST_CHECK_EQUAL(group.work, 0);
}

BOOST_AUTO_TEST_CASE(k6_segmented_form_is_2048_groups_of_16x16x4) {
  rig r;
  auto a = h_of<int>(r), b = h_of<int>(r), c = h_of<int>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) { k6(kh, a, b, c, true); });
  auto pk = prep(r, 0);
  const tile_graph tg = build_tile_graph(*pk);
  BOOST_CHECK_EQUAL(tg.groups.size(), 2048u);
  BOOST_CHECK_EQUAL(pk->native_size, 1024);
  BOOST_CHECK_EQUAL(pk->works[0].tile[0], 16);
  BOOST_CHECK_EQUAL(pk->works[0].tile[1], 16);
  BOOST_CHECK_EQUAL(pk->works[0].ips, 4);
}

BOOST_AUTO_TEST_CASE(k14_at_cap_256_is_256_groups_of_256) {
  sg::graph_options o;
  o.group_size_cap = 256;
  rig r(o);
  auto x = h_of<int>(r), y = h_of<int>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) { k14(kh, x, y, true); });
  auto pk = prep(r, 0);
  const tile_graph tg = build_tile_graph(*pk);
  BOOST_CHECK_EQUAL(tg.groups.size(), 256u);
  BOOST_CHECK_EQUAL(pk->native_size, 256);
  BOOST_CHECK_EQUAL(pk->works[0].tile_rank, 1);
  BOOST_CHECK_EQUAL(pk->works[0].tile[0], 1);
  BOOST_CHECK_EQUAL(pk->works[0].segs, 4);
  BOOST_CHECK_EQUAL(pk->works[0].ips, 256);
}

BOOST_AUTO_TEST_CASE(a_work_tile_above_the_cap_is_rejected) {
  rig r;
  const std::string m = message_of([&] {
    submit_kernel(r, [&](sg::kernel_handle &kh) { kh.work_tiles({8, 8}, {16, 16}, 128); });
  });
  BOOST_CHECK(has(m, "kh.work_tiles"));
  BOOST_CHECK(has(m, "32768"));
  BOOST_CHECK(has(m, "1024"));
}

BOOST_AUTO_TEST_CASE(a_native_shape_above_3d_is_rejected) {
  rig r;
  const std::string m = message_of([&] {
    submit_kernel(r, [&](sg::kernel_handle &kh) { kh.work_tiles({4}, {2, 2, 2}, 4); });
  });
  BOOST_CHECK(has(m, "kh.work_tiles"));
  BOOST_CHECK(has(m, "tile_rank = 3"));
  BOOST_CHECK(has(m, "run = 4"));
}

BOOST_AUTO_TEST_CASE(a_slice_beyond_its_set_is_rejected) {
  rig r;
  auto x = h_of<float>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) {
    auto xs = kh.source(x, sg::shape<1>({64}));
    auto work1 = kh.work_tiles({16}, {1}, 4);
    kh.slice_work(work1, {15}, {2}, {1}, {1}, 4);
  });
  const std::string m = message_of([&] { prep(r, 0); });
  BOOST_CHECK(has(m, "kh.slice_work"));
  BOOST_CHECK(has(m, "offset = 15"));
  BOOST_CHECK(has(m, "extent = 2"));
  BOOST_CHECK(has(m, "stride = 1"));
  BOOST_CHECK(has(m, "tiles = 16"));
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(work_tile_maps)

BOOST_AUTO_TEST_CASE(k4_unit_selects_its_window) {
  rig r;
  auto x = h_of<int>(r), y = h_of<int>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) { k4(kh, x, y); });
  auto pk = prep(r, 0);
  const map_conc &m = pk->maps[0];
  const input_sel &s = m.sel(3, 5, 0);
  BOOST_CHECK_EQUAL(s.tile, 64 * 3 + 5);
  BOOST_CHECK_EQUAL(s.tile, 197);
  BOOST_CHECK_EQUAL(s.len, 3);
}

BOOST_AUTO_TEST_CASE(k1_level_two_selects_tiles_of_its_slice) {
  rig r;
  auto x = h_of<float>(r), total = h_of<float>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) { k1(kh, x, total); });
  auto pk = prep(r, 0);
  const input_sel &s = pk->maps[1].sel(2, 0, 0);
  BOOST_CHECK_EQUAL(s.tile, 2);
  tile_graph tg = build_tile_graph(*pk);
  std::set<std::int64_t> produced;
  for (const pull &p : tg.pulls) {
    const task_inst &c = tg.tasks[p.consumer];
    if (c.map == 1 && c.tile == 2 && p.producer_map == 0)
      produced.insert(p.producer_tile);
  }
  BOOST_CHECK((produced == std::set<std::int64_t>{8, 9, 10, 11}));
}

BOOST_AUTO_TEST_CASE(k3_increment_step_selects_its_before_tile) {
  rig r;
  auto x = h_of<int>(r), y = h_of<int>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) { k3(kh, x, y); });
  auto pk = prep(r, 0);
  const input_sel &s = pk->maps[2].sel(5, 0, 0);
  BOOST_CHECK_EQUAL(s.tile, 5);
  BOOST_CHECK_EQUAL(s.len, 64);
}

BOOST_AUTO_TEST_CASE(a_run_longer_than_the_work_set_run_is_rejected) {
  rig r;
  auto x = h_of<int>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) {
    auto xs = kh.source(x, sg::shape<1>({8}));
    auto ts = kh.tile(xs, {8});
    auto work = kh.work_tiles({1}, {1}, 4);
    kh.map_work(work, [=](auto unit) {
      auto tile = ts[unit.tile()];
      auto run = sg::flatten(tile);
      return unit.run(run, sum_items);
    });
  });
  const std::string m = message_of([&] { prep(r, 0); });
  BOOST_CHECK(has(m, "unit.run"));
  BOOST_CHECK(has(m, "input_length = 8"));
  BOOST_CHECK(has(m, "run_length = 4"));
}

BOOST_AUTO_TEST_CASE(a_replicated_run_over_a_distributed_result_is_rejected) {
  sg::graph_options o;
  o.group_size_cap = 256;
  rig r(o);
  auto x = h_of<int>(r), y = h_of<int>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) { k14(kh, x, y, true, false, true); });
  const std::string m = message_of([&] { prep(r, 0); });
  BOOST_CHECK(has(m, "unit.run_replicated"));
  BOOST_CHECK(has(m, "segments = 4"));
}

BOOST_AUTO_TEST_CASE(bodies_run_once_per_unit_and_not_again_for_the_same_launch_values) {
  rig r;
  auto x = h_of<float>(r), total = h_of<float>(r);
  auto count = std::make_shared<int>(0);
  host_write(*r.g, x, iota_f(64));
  submit_kernel(r, [&](sg::kernel_handle &kh) {
    auto xs = kh.source(x, sg::shape<1>({64}));
    auto ts = kh.tile(xs, {4});
    auto work = kh.work_tiles({16}, {1}, 4);
    auto l1 = kh.map_work(work, [=](auto unit) {
      ++*count;
      auto tile = ts[unit.tile()];
      auto run = sg::flatten(tile);
      return unit.run(run, sum_items);
    });
    auto elems = kh.join(l1);
    kh.evaluate(total, elems);
  });
  BOOST_CHECK_EQUAL(*count, 0);
  r.g->run();
  BOOST_CHECK_EQUAL(*count, 16);
  r.g->run();
  BOOST_CHECK_EQUAL(*count, 16);
  BOOST_CHECK_EQUAL(r.g->stats().unit_bodies, 16u);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(work_item_maps)

BOOST_AUTO_TEST_CASE(k9_records_four_item_tasks_over_one_range) {
  rig r;
  auto x = h_of<float>(r), y = h_of<float>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) { k9(kh, x, y); });
  const sg::detail::kgraph &g = r.impl->nodes()[0]->graph;
  BOOST_CHECK_EQUAL(g.imaps.size(), 4u);
  std::size_t tasks = 0;
  for (const auto &m : g.imaps)
    tasks += m.tasks.size();
  BOOST_CHECK_EQUAL(tasks, 4u);
  BOOST_CHECK_EQUAL(g.works.size(), 1u);
}

BOOST_AUTO_TEST_CASE(a_result_array_is_read_only_at_the_items_own_index) {
  rig r;
  auto x = h_of<float>(r);
  const std::string m = message_of([&] {
    submit_kernel(r, [&](sg::kernel_handle &kh) {
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
  });
  BOOST_CHECK(has(m, "kh.map_items"));
  BOOST_CHECK(has(m, "prev"));
  BOOST_CHECK(has(m, "offset = 1"));
}

BOOST_AUTO_TEST_CASE(a_result_array_never_feeds_work_tiles) {
  rig r;
  auto x = h_of<float>(r), y = h_of<float>(r);
  const std::string m = message_of([&] {
    submit_kernel(r, [&](sg::kernel_handle &kh) {
      auto xs = kh.source(x, sg::shape<1>({64}));
      auto items = kh.work_items({64});
      auto first = kh.map_items(items, [=](auto item) {
        auto v = xs[item.id()];
        return item.compute(v, step_fn);
      });
      kh.tile(first, {4});
    });
  });
  BOOST_CHECK(has(m, "kh.map_work"));
  BOOST_CHECK(has(m, "kh.map_items"));
}

BOOST_AUTO_TEST_CASE(work_tiles_and_work_items_never_share_a_kernel) {
  rig r;
  auto x = h_of<float>(r), y = h_of<float>(r), total = h_of<float>(r);
  const std::string m = message_of([&] {
    submit_kernel(r, [&](sg::kernel_handle &kh) {
      k9(kh, x, y);
      k1(kh, x, total);
    });
  });
  BOOST_CHECK(has(m, "work_tile_sets = 1"));
  BOOST_CHECK(has(m, "work_item_sets = 1"));
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(marks_and_backing)

namespace {
struct planned {
  std::unique_ptr<prepared_kernel> pk;
  tile_graph tg;
  std::vector<value_backing> backing;
};
planned plan_of(rig &r) {
  planned p;
  p.pk = prep(r, 0);
  p.tg = build_tile_graph(*p.pk);
  p.backing = compute_backing(*p.pk, p.tg);
  return p;
}
} // namespace

BOOST_AUTO_TEST_CASE(k1_keeps_the_own_tile_private_and_shares_the_rest_globally) {
  rig r;
  auto x = h_of<float>(r), total = h_of<float>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) { k1(kh, x, total); });
  const planned p = plan_of(r);
  const auto &l1 = p.backing[0].tile;
  for (int t = 0; t < 16; ++t)
    BOOST_CHECK_MESSAGE(l1[t] == ((t % 4 == 3) ? backing::private_scratch : backing::global), "l1 tile " << t);
}

BOOST_AUTO_TEST_CASE(k6_marks_the_bands_and_shares_the_segmented_partials) {
  rig r;
  auto a = h_of<int>(r), b = h_of<int>(r), c = h_of<int>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) { k6(kh, a, b, c, false); });
  {
    auto pk = prep(r, 0);
    tile_graph tg = build_tile_graph(*pk);
    const launch_plan lp = encode_launch(*pk, tg, std::vector<slot_binding>(3), 0, device_limits{});
    const dev::block_t block = lp.block.data();
    const dev::ticket_rec *tickets = dev::tickets_of(block);
    BOOST_REQUIRE_GT(dev::header_of(block).n_tickets, 0u);
    for (std::uint32_t t = 0; t < dev::header_of(block).n_tickets; ++t)
      BOOST_CHECK_EQUAL(tickets[t].n_copies, 2u);
    int array_marks = 0;
    for (const mark_rec &mk : pk->graph.marks)
      array_marks += mk.kind == mark_kind::array_tiles ? 1 : 0;
    BOOST_CHECK_EQUAL(array_marks, 2);
    const dev::copy_rec *copies = dev::copies_of(block);
    for (std::uint32_t t = 0; t < dev::header_of(block).n_tickets; ++t) {
      const dev::copy_rec &a_band = copies[tickets[t].first_copy];
      const dev::copy_rec &b_band = copies[tickets[t].first_copy + 1];
      BOOST_CHECK_EQUAL(a_band.elems, 16u * 128u);
      BOOST_CHECK_EQUAL(b_band.elems, 128u * 16u);
      BOOST_CHECK_NE(a_band.local_off, b_band.local_off);
    }
  }
  rig r2;
  auto a2 = h_of<int>(r2), b2 = h_of<int>(r2), c2 = h_of<int>(r2);
  submit_kernel(r2, [&](sg::kernel_handle &kh) { k6(kh, a2, b2, c2, true); });
  auto pk = prep(r2, 0);
  tile_graph tg = build_tile_graph(*pk);
  const launch_plan lp = encode_launch(*pk, tg, std::vector<slot_binding>(3), 0, device_limits{});
  const dev::task_rec *tasks = dev::tasks_of(lp.block.data());
  BOOST_CHECK_NE(tasks[0].partial_addr, 0u);
  BOOST_CHECK_EQUAL(tasks[0].run_kind, static_cast<std::uint32_t>(run_kind::distributed));
}

BOOST_AUTO_TEST_CASE(k8_keeps_a_b_c_private_and_shares_e) {
  rig r;
  auto x = h_of<int>(r), b = h_of<int>(r), c = h_of<int>(r), d = h_of<int>(r), f = h_of<int>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) { k8(kh, x, b, c, d, f); });
  const planned p = plan_of(r);
  for (int m : {0, 1, 2})
    for (backing t : p.backing[m].tile)
      BOOST_CHECK(t == backing::private_scratch);
  BOOST_CHECK(p.backing[3].evaluated);
  for (backing t : p.backing[4].tile)
    BOOST_CHECK(t == backing::global);
}

BOOST_AUTO_TEST_CASE(plain_mark_local_on_a_distributed_result_is_rejected) {
  sg::graph_options o;
  o.group_size_cap = 256;
  rig r(o);
  auto x = h_of<int>(r), y = h_of<int>(r);
  const std::string m = message_of([&] { submit_kernel(r, [&](sg::kernel_handle &kh) { k14(kh, x, y, true, true); }); });
  BOOST_CHECK(has(m, "kh.mark_local"));
  BOOST_CHECK(has(m, "segments = 4"));
}

BOOST_AUTO_TEST_CASE(segmented_mark_local_on_another_result_is_rejected) {
  rig r;
  auto x = h_of<float>(r);
  const std::string m = message_of([&] {
    submit_kernel(r, [&](sg::kernel_handle &kh) {
      auto xs = kh.source(x, sg::shape<1>({64}));
      auto ts = kh.tile(xs, {4});
      auto work1 = kh.work_tiles({16}, {1}, 4);
      auto l1 = sum_tiles(kh, work1, ts);
      kh.mark_local<sg::segmented>(l1);
    });
  });
  BOOST_CHECK(has(m, "kh.mark_local<sg::segmented>"));
  BOOST_CHECK(has(m, "segments = 1"));
}

BOOST_AUTO_TEST_CASE(array_tiles_marking_refuses_result_tiles) {
  rig r;
  auto x = h_of<float>(r);
  const std::string m = message_of([&] {
    submit_kernel(r, [&](sg::kernel_handle &kh) {
      auto xs = kh.source(x, sg::shape<1>({64}));
      auto ts = kh.tile(xs, {4});
      auto work1 = kh.work_tiles({16}, {1}, 4);
      auto l1 = sum_tiles(kh, work1, ts);
      kh.mark_local(l1, work1);
    });
  });
  BOOST_CHECK(has(m, "kh.mark_local"));
  BOOST_CHECK(has(m, "array tiles"));
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(repeat)

BOOST_AUTO_TEST_CASE(depth_changes_numbers_and_never_structure) {
  rig r;
  auto x = h_of<float>(r), total = h_of<float>(r);
  auto n = h_of<int>(r), d = h_of<int>(r);
  submit_kernel(r, [&](sg::kernel_handle &kh) { k2(kh, x, n, d, total); });
  auto three = prep(r, 0, {64, 3});
  auto five = prep(r, 0, {1024, 5});
  BOOST_CHECK(three->graph.tasks == five->graph.tasks);
  BOOST_CHECK_EQUAL(task_list_hash(three->graph.tasks), task_list_hash(five->graph.tasks));
  BOOST_CHECK_EQUAL(three->works[0].tiles, 16);
  BOOST_CHECK_EQUAL(five->works[0].tiles, 256);
  BOOST_CHECK_EQUAL(three->maps.size(), 3u);
  BOOST_CHECK_EQUAL(five->maps.size(), 5u);
}

BOOST_AUTO_TEST_SUITE_END()
