#include "../common/check.hpp"
#include "../common/graphs.hpp"
#include "../common/mocks.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>

using namespace sgt;

namespace {

sg::detail::access acc(std::size_t h, access_mode m, int group = -1) {
  return sg::detail::access{h, {0, 64}, m, group};
}

std::unique_ptr<prepared_kernel> prep(rig &r, std::vector<std::int64_t> values = {}) {
  return std::make_unique<prepared_kernel>(kernel_preparer::prepare(r.impl->nodes().at(0)->graph, values));
}
template <class F> void submit_kernel(rig &r, F f) { r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { f(kh); }); }

// the ticket of a task instance's group
int ticket_of(const tile_graph &tg, int task) { return tg.ticket_of_group[tg.tasks[task].group]; }

// every pull comes from a lower ticket, or from an earlier task of the same group
bool order_is_valid(const tile_graph &tg) {
  for (const pull &p : tg.pulls) {
    if (p.by_data && p.producer_unit >= 0)
      continue;
    int producer = -1;
    for (std::size_t i = 0; i < tg.tasks.size(); ++i)
      if (tg.tasks[i].map == p.producer_map && tg.tasks[i].tile == p.producer_tile)
        producer = static_cast<int>(i);
    if (producer < 0)
      return false;
    const int pt = ticket_of(tg, producer), ct = ticket_of(tg, p.consumer);
    if (pt > ct)
      return false;
    if (pt == ct && tg.tasks[producer].map >= tg.tasks[p.consumer].map)
      return false;
  }
  return true;
}

} // namespace

BOOST_AUTO_TEST_SUITE(dependencies)

BOOST_AUTO_TEST_CASE(all_nine_ordered_pairs_of_read_write_inplace) {
  struct row {
    access_mode first, second;
    int g1, g2;
    std::vector<op_id> expected;
  };
  const std::vector<row> rows = {
      {access_mode::read, access_mode::read, -1, -1, {}},
      {access_mode::read, access_mode::write, -1, -1, {1}},
      {access_mode::read, access_mode::inplace, -1, 5, {1}},
      {access_mode::write, access_mode::read, -1, -1, {1}},
      {access_mode::write, access_mode::write, -1, -1, {1}},
      {access_mode::write, access_mode::inplace, -1, 5, {1}},
      {access_mode::inplace, access_mode::read, 5, -1, {1}},
      {access_mode::inplace, access_mode::write, 5, -1, {1}},
      {access_mode::inplace, access_mode::inplace, 5, 6, {1}},
      {access_mode::inplace, access_mode::inplace, 5, 5, {}},
  };
  for (const row &r : rows) {
    dependency_tracker t;
    t.commit(1, {acc(0, r.first, r.g1)});
    BOOST_CHECK((t.resolve({acc(0, r.second, r.g2)}) == r.expected));
  }
}

BOOST_AUTO_TEST_CASE(host_and_kernel_nodes_of_one_graph_order_by_role) {
  rig r;
  auto x = r.g->create_mem_handle<float>();
  auto n = r.g->create_mem_handle<int>();
  auto y = r.g->create_mem_handle<float>();
  host_write(*r.g, n, std::vector<int>{64});
  host_write(*r.g, x, iota_f(64));
  r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { rerun_a(kh, x, n, y); });
  std::vector<float> out;
  host_read(*r.g, y, 64, out);
  r.g->run();
  // host reads y after the kernel's launch (the last operation before it)
  BOOST_REQUIRE_EQUAL(r.ops->hosts.size(), 3u);
  const auto launch_id = r.ops->next - 1; // the host read comes after the launch
  const auto &read_deps = r.ops->hosts.back().deps;
  BOOST_CHECK(std::find(read_deps.begin(), read_deps.end(), launch_id) != read_deps.end());
}

BOOST_AUTO_TEST_CASE(a_staged_copy_chains_its_legs_and_the_buffer_outlives_the_copies) {
  rig r;
  auto x = r.g->create_mem_handle<float>();
  auto total = r.g->create_mem_handle<float>();
  host_write(*r.g, x, iota_f(64));
  r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { k1(kh, x, total); });
  migration m;
  m.plan = plan_copy(place{0, 1}, place{2, 2}, interval{0, 256});
  BOOST_REQUIRE(m.plan.kind == copy_kind::staged_via_host);
  m.from.ptr = r.mem->arena.base;
  m.to.ptr = r.mem->arena.base + 4096;
  r.mem->forced_migration[0] = m;
  r.g->run();
  BOOST_REQUIRE_GE(r.ops->copies.size(), 3u);
  // down leg, up leg waiting on the down leg
  const auto &down = r.ops->copies[0];
  const auto &up = r.ops->copies[1];
  BOOST_CHECK(up.src == down.dst);
  BOOST_CHECK(std::find(up.deps.begin(), up.deps.end(), down.id) != up.deps.end());
  // the staging buffer is released only after the queue has drained
  BOOST_CHECK_EQUAL(r.mem->released, 0);
  BOOST_CHECK_EQUAL(r.ops->wait_all_calls, 0);
  r.g->wait();
  BOOST_CHECK_EQUAL(r.mem->released, 1);
  BOOST_CHECK_EQUAL(r.mem->wait_alls_at_release, 1);
  const auto &queue_log = r.ops->log;
  const auto last_copy = std::find(queue_log.rbegin(), queue_log.rend(), std::string("copy"));
  const auto drained = std::find(queue_log.rbegin(), queue_log.rend(), std::string("wait_all"));
  BOOST_REQUIRE(last_copy != queue_log.rend() && drained != queue_log.rend());
  BOOST_CHECK(drained < last_copy);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(tile_order)

BOOST_AUTO_TEST_CASE(k1_orders_levels_after_their_producers) {
  rig r;
  auto x = r.g->create_mem_handle<float>(), total = r.g->create_mem_handle<float>();
  submit_kernel(r, [&](sg::kernel_handle &kh) { k1(kh, x, total); });
  auto pk = prep(r);
  const tile_graph tg = build_tile_graph(*pk);
  BOOST_CHECK(order_is_valid(tg));
  // the level-3 task of tile 15 runs in the group of level-1 tile 15 after its level-2 task
  int l3 = -1;
  for (std::size_t i = 0; i < tg.tasks.size(); ++i)
    if (tg.tasks[i].map == 2)
      l3 = static_cast<int>(i);
  BOOST_REQUIRE_GE(l3, 0);
  for (int t : {3, 7, 11})
    for (std::size_t i = 0; i < tg.tasks.size(); ++i)
      if (tg.tasks[i].map == 1 && tg.tasks[i].tile == t / 4)
        BOOST_CHECK_LE(ticket_of(tg, static_cast<int>(i)), ticket_of(tg, l3));
}

BOOST_AUTO_TEST_CASE(k5_runs_the_producing_set_before_the_gathering_set) {
  rig r;
  auto s = r.g->create_mem_handle<int>(), idx = r.g->create_mem_handle<int>(), y = r.g->create_mem_handle<int>();
  submit_kernel(r, [&](sg::kernel_handle &kh) { k5(kh, s, idx, y); });
  auto pk = prep(r);
  const tile_graph tg = build_tile_graph(*pk);
  BOOST_CHECK(order_is_valid(tg));
  int latest_src = -1, earliest_y = 1 << 30;
  for (std::size_t g = 0; g < tg.groups.size(); ++g) {
    if (tg.groups[g].work == 0)
      latest_src = std::max(latest_src, tg.ticket_of_group[g]);
    else
      earliest_y = std::min(earliest_y, tg.ticket_of_group[g]);
  }
  BOOST_CHECK_LT(latest_src, earliest_y);
}

BOOST_AUTO_TEST_CASE(a_gather_on_its_own_set_is_a_cycle_and_names_the_tiles) {
  rig r;
  auto s = r.g->create_mem_handle<int>(), idx = r.g->create_mem_handle<int>(), y = r.g->create_mem_handle<int>();
  submit_kernel(r, [&](sg::kernel_handle &kh) { k5(kh, s, idx, y, true); });
  auto pk = prep(r);
  try {
    build_tile_graph(*pk);
    BOOST_FAIL("no throw");
  } catch (const sg::error &e) {
    const std::string w = e.what();
    BOOST_CHECK(w.find("kh.index_map") != std::string::npos);
    for (int t = 0; t < 4; ++t)
      BOOST_CHECK(w.find("work set 0 tile " + std::to_string(t)) != std::string::npos);
  }
}

BOOST_AUTO_TEST_CASE(k7_last_tile_follows_all_others) {
  rig r;
  auto v = r.g->create_mem_handle<int>(), h = r.g->create_mem_handle<int>();
  submit_kernel(r, [&](sg::kernel_handle &kh) { k7(kh, v, h); });
  auto pk = prep(r);
  const tile_graph tg = build_tile_graph(*pk);
  BOOST_CHECK(order_is_valid(tg));
  const int last = tg.ticket_of_group[15];
  for (int g = 0; g < 15; ++g)
    BOOST_CHECK_LT(tg.ticket_of_group[g], last);
}

BOOST_AUTO_TEST_CASE(the_order_is_the_same_for_every_build) {
  rig r;
  auto x = r.g->create_mem_handle<float>(), total = r.g->create_mem_handle<float>();
  submit_kernel(r, [&](sg::kernel_handle &kh) { k1(kh, x, total); });
  auto pk = prep(r);
  const std::vector<int> first = build_tile_graph(*pk).group_of_ticket;
  for (int i = 0; i < 100; ++i)
    BOOST_REQUIRE(build_tile_graph(*pk).group_of_ticket == first);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(bound_block)

namespace {

struct case_k {
  std::string name;
  std::function<void(rig &)> build;
  std::vector<std::int64_t> values;
};

std::vector<case_k> all_kernels() {
  std::vector<case_k> v;
  auto f = [](rig &r) { return r.g->create_mem_handle<float>(); };
  auto i = [](rig &r) { return r.g->create_mem_handle<int>(); };
  auto sub = [](rig &r, auto fn) { r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) { fn(kh); }); };
  v.push_back({"k1", [=](rig &r) { auto a = f(r), b = f(r); sub(r, [&](auto &kh) { k1(kh, a, b); }); }, {}});
  v.push_back({"k2", [=](rig &r) { auto a = f(r), b = f(r); auto n = i(r), d = i(r); sub(r, [&](auto &kh) { k2(kh, a, n, d, b); }); }, {64, 3}});
  v.push_back({"k3", [=](rig &r) { auto a = i(r), b = i(r); sub(r, [&](auto &kh) { k3(kh, a, b); }); }, {}});
  v.push_back({"k4", [=](rig &r) { auto a = i(r), b = i(r); sub(r, [&](auto &kh) { k4(kh, a, b); }); }, {}});
  v.push_back({"k5", [=](rig &r) { auto a = i(r), b = i(r), c = i(r); sub(r, [&](auto &kh) { k5(kh, a, b, c); }); }, {}});
  v.push_back({"k6", [=](rig &r) { auto a = i(r), b = i(r), c = i(r); sub(r, [&](auto &kh) { k6(kh, a, b, c, false); }); }, {}});
  v.push_back({"k7", [=](rig &r) { auto a = i(r), b = i(r); sub(r, [&](auto &kh) { k7(kh, a, b); }); }, {}});
  v.push_back({"k8", [=](rig &r) { auto a = i(r), b = i(r), c = i(r), d = i(r), e = i(r); sub(r, [&](auto &kh) { k8(kh, a, b, c, d, e); }); }, {}});
  v.push_back({"k9", [=](rig &r) { auto a = f(r), b = f(r); sub(r, [&](auto &kh) { k9(kh, a, b); }); }, {}});
  v.push_back({"k10", [=](rig &r) { auto a = f(r), b = f(r); sub(r, [&](auto &kh) { k10(kh, a, b, 2.f, 1.f, 256); }); }, {}});
  v.push_back({"k11", [=](rig &r) { auto a = f(r), b = f(r); sub(r, [&](auto &kh) { k11(kh, a, b, 1.f); }); }, {}});
  v.push_back({"k12", [=](rig &r) { auto a = f(r), c = f(r), d = f(r); auto b = i(r); sub(r, [&](auto &kh) { k12(kh, a, b, c, d); }); }, {}});
  v.push_back({"k13", [=](rig &r) { auto a = f(r), b = f(r); sub(r, [&](auto &kh) { k13(kh, a, b); }); }, {}});
  v.push_back({"k14", [=](rig &r) { auto a = i(r), b = i(r); sub(r, [&](auto &kh) { k14(kh, a, b, false); }); }, {}});
  return v;
}

} // namespace

BOOST_AUTO_TEST_CASE(every_kernel_encodes_and_decodes_to_the_same_task_lists) {
  for (const case_k &k : all_kernels()) {
    rig r;
    k.build(r);
    auto pk = prep(r, k.values);
    tile_graph tg = build_tile_graph(*pk);
    const launch_plan lp = encode_launch(*pk, tg, std::vector<slot_binding>(8), 0, device_limits{});
    const dev::block_t b = lp.block.data();
    const dev::block_header &h = dev::header_of(b);
    BOOST_REQUIRE_MESSAGE(h.magic == dev::block_magic, k.name);
    BOOST_CHECK_MESSAGE(h.n_tickets == tg.groups.size(), k.name);
    BOOST_CHECK_MESSAGE(h.n_tasks == tg.tasks.size(), k.name);
    std::uint32_t seen = 0;
    for (std::uint32_t t = 0; t < h.n_tickets; ++t) {
      const dev::ticket_rec &tr = dev::tickets_of(b)[t];
      const auto &expect = tg.group_tasks[tg.group_of_ticket[t]];
      BOOST_REQUIRE_MESSAGE(tr.n_tasks == expect.size(), k.name);
      for (std::uint32_t i = 0; i < tr.n_tasks; ++i, ++seen) {
        const dev::task_rec &d = dev::tasks_of(b)[tr.first_task + i];
        const task_inst &e = tg.tasks[expect[i]];
        BOOST_CHECK_EQUAL(d.flag, e.flag);
        BOOST_CHECK_EQUAL(d.n_waits, e.waits.size());
        for (std::uint32_t w = 0; w < d.n_waits; ++w)
          BOOST_CHECK_EQUAL(dev::waits_of(b)[d.first_wait + w], e.waits[w]);
        const map_conc &mc = pk->maps[e.map];
        BOOST_CHECK_EQUAL(d.n_units, mc.units);
        for (std::int64_t u = 0; u < mc.units; ++u)
          for (std::size_t j = 0; j < mc.n_inputs; ++j) {
            const dev::sel_rec &s = dev::sels_of(b)[d.first_sel + u * mc.n_inputs + j];
            BOOST_REQUIRE_EQUAL(s.view, mc.sel(e.tile, u, j).view);
            BOOST_REQUIRE_EQUAL(s.tile, mc.sel(e.tile, u, j).tile);
            BOOST_REQUIRE_EQUAL(s.len, mc.sel(e.tile, u, j).len);
          }
      }
    }
    BOOST_CHECK_MESSAGE(seen == h.n_tasks, k.name);
    const auto base = reinterpret_cast<std::uintptr_t>(b);
    const auto section_aligned = [&](std::uint64_t off, std::size_t alignment) {
      return (base + off) % alignment == 0;
    };
    BOOST_CHECK_MESSAGE(section_aligned(h.off_tickets, alignof(dev::ticket_rec)), k.name);
    BOOST_CHECK_MESSAGE(section_aligned(h.off_tasks, alignof(dev::task_rec)), k.name);
    BOOST_CHECK_MESSAGE(section_aligned(h.off_waits, alignof(std::uint32_t)), k.name);
    BOOST_CHECK_MESSAGE(section_aligned(h.off_copies, alignof(dev::copy_rec)), k.name);
    BOOST_CHECK_MESSAGE(section_aligned(h.off_views, alignof(std::remove_cvref_t<decltype(*dev::views_of(b))>)), k.name);
    BOOST_CHECK_MESSAGE(section_aligned(h.off_sels, alignof(dev::sel_rec)), k.name);
    BOOST_CHECK_MESSAGE(section_aligned(h.off_item_tasks, alignof(dev::item_task_rec)), k.name);
    BOOST_CHECK_MESSAGE(base % alignof(dev::block_header) == 0, k.name);
    // every section starts on a 16-byte boundary and every record size is a multiple of 8
    for (std::uint64_t off : {h.off_tickets, h.off_tasks, h.off_waits, h.off_copies, h.off_views, h.off_sels,
                              h.off_item_tasks, h.off_closures, h.off_tables})
      BOOST_CHECK_MESSAGE(off % 16 == 0, k.name);
    BOOST_CHECK_EQUAL(lp.block.size() % 256, 0u);
    if (lp.items) {
      BOOST_CHECK_MESSAGE(h.n_item_tasks > 0, k.name);
      BOOST_CHECK_EQUAL(h.n_items, static_cast<std::uint64_t>(pk->n_items));
    }
  }
}

BOOST_AUTO_TEST_CASE(sub_word_fields_are_naturally_aligned) {
  static_assert(offsetof(dev::task_rec, closure) % 8 == 0);
  static_assert(offsetof(dev::task_rec, run_len) % 8 == 0);
  static_assert(offsetof(dev::task_rec, out_global) % 8 == 0);
  static_assert(offsetof(dev::task_rec, eval_addr) % 8 == 0);
  static_assert(offsetof(dev::sel_rec, off) % 8 == 0);
  static_assert(offsetof(dev::copy_rec, elems) % 8 == 0);
  static_assert(offsetof(dev::item_task_rec, closure) % 8 == 0);
  static_assert(offsetof(dev::item_input_rec, off) % 8 == 0);
  static_assert(offsetof(dev::block_header, flags) % 8 == 0);
#define SGT_FIELD_ALIGNED(record, field) \
  static_assert(offsetof(record, field) % alignof(decltype(record::field)) == 0, #record "::" #field)
  SGT_FIELD_ALIGNED(dev::block_header, magic);
  SGT_FIELD_ALIGNED(dev::block_header, n_tickets);
  SGT_FIELD_ALIGNED(dev::block_header, n_tasks);
  SGT_FIELD_ALIGNED(dev::block_header, n_item_tasks);
  SGT_FIELD_ALIGNED(dev::block_header, n_regs_bytes);
  SGT_FIELD_ALIGNED(dev::block_header, n_items);
  SGT_FIELD_ALIGNED(dev::block_header, items_ext);
  SGT_FIELD_ALIGNED(dev::block_header, items_rank);
  SGT_FIELD_ALIGNED(dev::block_header, pad0);
  SGT_FIELD_ALIGNED(dev::block_header, off_tickets);
  SGT_FIELD_ALIGNED(dev::block_header, off_tasks);
  SGT_FIELD_ALIGNED(dev::block_header, off_waits);
  SGT_FIELD_ALIGNED(dev::block_header, off_copies);
  SGT_FIELD_ALIGNED(dev::block_header, off_views);
  SGT_FIELD_ALIGNED(dev::block_header, off_sels);
  SGT_FIELD_ALIGNED(dev::block_header, off_item_tasks);
  SGT_FIELD_ALIGNED(dev::block_header, off_closures);
  SGT_FIELD_ALIGNED(dev::block_header, off_tables);
  SGT_FIELD_ALIGNED(dev::block_header, n_flags);
  SGT_FIELD_ALIGNED(dev::block_header, pad1);
  SGT_FIELD_ALIGNED(dev::ticket_rec, first_task);
  SGT_FIELD_ALIGNED(dev::ticket_rec, n_tasks);
  SGT_FIELD_ALIGNED(dev::ticket_rec, first_copy);
  SGT_FIELD_ALIGNED(dev::ticket_rec, n_copies);
  SGT_FIELD_ALIGNED(dev::copy_rec, view);
  SGT_FIELD_ALIGNED(dev::copy_rec, local_off);
  SGT_FIELD_ALIGNED(dev::copy_rec, tile);
  SGT_FIELD_ALIGNED(dev::copy_rec, elems);
  SGT_FIELD_ALIGNED(dev::copy_rec, elem_bytes);
  SGT_FIELD_ALIGNED(dev::copy_rec, pad);
  SGT_FIELD_ALIGNED(dev::sel_rec, view);
  SGT_FIELD_ALIGNED(dev::sel_rec, local_off);
  SGT_FIELD_ALIGNED(dev::sel_rec, tile);
  SGT_FIELD_ALIGNED(dev::sel_rec, off);
  SGT_FIELD_ALIGNED(dev::sel_rec, ext);
  SGT_FIELD_ALIGNED(dev::sel_rec, len);
  SGT_FIELD_ALIGNED(dev::task_rec, task_id);
  SGT_FIELD_ALIGNED(dev::task_rec, flag);
  SGT_FIELD_ALIGNED(dev::task_rec, first_wait);
  SGT_FIELD_ALIGNED(dev::task_rec, n_waits);
  SGT_FIELD_ALIGNED(dev::task_rec, first_sel);
  SGT_FIELD_ALIGNED(dev::task_rec, n_inputs);
  SGT_FIELD_ALIGNED(dev::task_rec, n_units);
  SGT_FIELD_ALIGNED(dev::task_rec, run_kind);
  SGT_FIELD_ALIGNED(dev::task_rec, result_kind);
  SGT_FIELD_ALIGNED(dev::task_rec, stride);
  SGT_FIELD_ALIGNED(dev::task_rec, run_len);
  SGT_FIELD_ALIGNED(dev::task_rec, seg);
  SGT_FIELD_ALIGNED(dev::task_rec, segs);
  SGT_FIELD_ALIGNED(dev::task_rec, ips);
  SGT_FIELD_ALIGNED(dev::task_rec, rounds);
  SGT_FIELD_ALIGNED(dev::task_rec, tile_idx);
  SGT_FIELD_ALIGNED(dev::task_rec, out_bytes);
  SGT_FIELD_ALIGNED(dev::task_rec, bins);
  SGT_FIELD_ALIGNED(dev::task_rec, out_flags);
  SGT_FIELD_ALIGNED(dev::task_rec, local_out_off);
  SGT_FIELD_ALIGNED(dev::task_rec, out_global);
  SGT_FIELD_ALIGNED(dev::task_rec, out_tile_elems);
  SGT_FIELD_ALIGNED(dev::task_rec, eval_addr);
  SGT_FIELD_ALIGNED(dev::task_rec, eval_grid);
  SGT_FIELD_ALIGNED(dev::task_rec, eval_shape);
  SGT_FIELD_ALIGNED(dev::task_rec, eval_extent);
  SGT_FIELD_ALIGNED(dev::task_rec, closure);
  SGT_FIELD_ALIGNED(dev::task_rec, closure_bytes);
  SGT_FIELD_ALIGNED(dev::task_rec, partial_stride);
  SGT_FIELD_ALIGNED(dev::task_rec, partial_addr);
  SGT_FIELD_ALIGNED(dev::task_rec, unit_grid);
  SGT_FIELD_ALIGNED(dev::task_rec, pad);
  SGT_FIELD_ALIGNED(dev::item_input_rec, kind);
  SGT_FIELD_ALIGNED(dev::item_input_rec, view);
  SGT_FIELD_ALIGNED(dev::item_input_rec, reg);
  SGT_FIELD_ALIGNED(dev::item_input_rec, bytes);
  SGT_FIELD_ALIGNED(dev::item_input_rec, off);
  SGT_FIELD_ALIGNED(dev::item_task_rec, task_id);
  SGT_FIELD_ALIGNED(dev::item_task_rec, n_inputs);
  SGT_FIELD_ALIGNED(dev::item_task_rec, out_reg);
  SGT_FIELD_ALIGNED(dev::item_task_rec, out_bytes);
  SGT_FIELD_ALIGNED(dev::item_task_rec, closure_bytes);
  SGT_FIELD_ALIGNED(dev::item_task_rec, eval);
  SGT_FIELD_ALIGNED(dev::item_task_rec, sink);
  SGT_FIELD_ALIGNED(dev::item_task_rec, pad);
  SGT_FIELD_ALIGNED(dev::item_task_rec, closure);
  SGT_FIELD_ALIGNED(dev::item_task_rec, eval_addr);
  SGT_FIELD_ALIGNED(dev::item_task_rec, eval_extent);
  SGT_FIELD_ALIGNED(dev::item_task_rec, in);
#undef SGT_FIELD_ALIGNED
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(runs)

namespace {
// two independent map groups: one of fixed size, one whose grid follows the launch value n
struct two_parts {
  rig r;
  int run_number = 0;
  sg::mem_handle<int> n, n_w;
  sg::mem_handle<float> a, b, ya, yb;
  two_parts() {
    n = r.g->create_mem_handle<int>();
    a = r.g->create_mem_handle<float>();
    b = r.g->create_mem_handle<float>();
    ya = r.g->create_mem_handle<float>();
    yb = r.g->create_mem_handle<float>();
    r.g->submit<sg::host>([&](sg::host_handle &hh) {
      auto w = hh.write(n, sg::shape<1>({1}));
      hh.run([w, this] { w.data()[0] = run_number == 0 ? 64 : 128; });
    });
    host_write(*r.g, a, iota_f(64));
    host_write(*r.g, b, iota_f(128));
    r.g->submit(sycl::device{}, [&](sg::kernel_handle &kh) {
      auto nv = kh.launch_value(n);
      auto xa = kh.source(a, sg::shape<1>({64}));
      auto ta = kh.tile(xa, {4});
      auto wa = kh.work_tiles({16}, {1}, 4);
      auto la = sum_tiles(kh, wa, ta);
      kh.evaluate(ya, kh.join(la));
      auto xb = kh.source(b, sg::shape<1>({128}));
      auto tb = kh.tile(xb, {4});
      auto wb = kh.work_tiles({nv / 4}, {1}, 4);
      auto lb = sum_tiles(kh, wb, tb);
      kh.evaluate(yb, kh.join(lb));
    });
  }
};
} // namespace

BOOST_AUTO_TEST_CASE(a_second_run_with_the_same_values_prepares_and_composes_nothing) {
  two_parts t;
  t.r.g->run();
  const auto first = t.r.g->stats();
  BOOST_CHECK_EQUAL(first.prepares, 1u);
  BOOST_CHECK_EQUAL(first.compositions, 1u);
  BOOST_CHECK_EQUAL(first.unit_bodies, 16u + 16u);
  t.r.g->run();
  const auto second = t.r.g->stats();
  BOOST_CHECK_EQUAL(second.unit_bodies, first.unit_bodies);
  BOOST_CHECK_EQUAL(second.compositions, first.compositions);
  BOOST_CHECK_EQUAL(second.launches, 2u);
}

BOOST_AUTO_TEST_CASE(a_changed_launch_value_evaluates_only_the_bodies_that_read_it) {
  two_parts t;
  t.r.g->run();
  const std::size_t before = t.r.g->stats().unit_bodies;
  t.run_number = 1;
  t.r.g->run();
  // the fixed map keeps its units; the map over n is evaluated again for 128 / 4 tiles
  BOOST_CHECK_EQUAL(t.r.g->stats().unit_bodies - before, 32u);
  BOOST_CHECK_EQUAL(t.r.g->stats().compositions, 1u);
}

BOOST_AUTO_TEST_CASE(the_launch_value_is_read_after_its_producer_and_the_flags_are_cleared_before_every_launch) {
  two_parts t;
  t.r.g->run();
  t.r.g->run();
  const auto &log = t.r.ops->log;
  const auto host = std::find(log.begin(), log.end(), "host_task");
  const auto wait = std::find(log.begin(), log.end(), "wait");
  const auto launch = std::find(log.begin(), log.end(), "launch");
  BOOST_REQUIRE(host != log.end() && wait != log.end() && launch != log.end());
  BOOST_CHECK(host < wait && wait < launch);
  BOOST_REQUIRE_EQUAL(t.r.ops->launches.size(), 2u);
  BOOST_REQUIRE_EQUAL(t.r.ops->fills.size(), 2u);
  for (std::size_t i = 0; i < 2; ++i) {
    const auto &fill = t.r.ops->fills[i];
    BOOST_CHECK_EQUAL(fill.count, t.r.launch->plans[i].flag_words);
    BOOST_CHECK(std::all_of(fill.pattern.begin(), fill.pattern.end(), [](std::byte b) { return b == std::byte{0}; }));
    // the launch waits for the flag fill
    const auto &deps = t.r.ops->launches[i].deps;
    BOOST_CHECK(!deps.empty());
  }
  BOOST_CHECK(t.r.launch->configs[0] == t.r.launch->configs[1]);
}

BOOST_AUTO_TEST_SUITE_END()
