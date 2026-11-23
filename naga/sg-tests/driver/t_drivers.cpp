#include "world.hpp"

#include <algorithm>
#include <set>
#include <thread>

using namespace sgd;

BOOST_AUTO_TEST_SUITE(driver_suite)

BOOST_AUTO_TEST_CASE(owner_skips_claimed_groups) {
  world w;
  w.add_layer(2, 0, 0, 8);
  w.add_node(2, 4, 0);
  w.flags = {flag_in_progress, flag_done};
  w.upload();
  w.run<0>(2, 4);
  const auto log = w.log();
  BOOST_CHECK_EQUAL(count_tag(log, t_body), 0u);
  BOOST_CHECK_EQUAL(count_tag(log, t_run_layer), 8u);
  const auto f = w.read_flags();
  BOOST_CHECK_EQUAL(f[0], flag_in_progress);
  BOOST_CHECK_EQUAL(f[1], flag_done);
}

BOOST_AUTO_TEST_CASE(native_group_without_layer_returns) {
  world w;
  w.add_layer(2, 0, 0, 8);
  w.add_node(2, 4, 0);
  w.upload();
  w.run<0>(3, 4);
  const auto log = w.log();
  BOOST_CHECK_EQUAL(count_tag(log, t_run_layer), 8u);
}

BOOST_AUTO_TEST_CASE(compute_order_and_tiles_item_body) {
  world w;
  w.add_layer(1, 0, 0, 4);
  w.add_node(1, 4, 0);
  w.add_node(1, 4, 0);
  w.add_local_link(0, 1, 2, 4, 0, 16, 64);
  w.upload();
  w.run<0>(1, 4);
  const auto log = w.log();
  BOOST_CHECK_EQUAL(count_tag(log, t_resolve_tree), 8u);
  BOOST_CHECK_EQUAL(count_tag(log, t_prep), 8u);
  BOOST_CHECK_EQUAL(count_tag(log, t_run_source), 8u);
  BOOST_CHECK_EQUAL(count_tag(log, t_body), 4u);
  std::size_t last_source = 0, first_body = log.size();
  std::set<std::uint64_t> addrs, ls;
  for (std::size_t i = 0; i < log.size(); ++i) {
    if (log[i].tag == t_run_source)
      last_source = i;
    if (log[i].tag == t_body) {
      first_body = std::min(first_body, i);
      BOOST_CHECK_EQUAL(log[i].b, flag_in_progress);
      addrs.insert(log[i].x);
      ls.insert(log[i].l);
    }
  }
  BOOST_CHECK(last_source < first_body);
  BOOST_CHECK_EQUAL(addrs.size(), 4u);
  BOOST_CHECK_EQUAL(ls.size(), 4u);
  for (std::uint64_t l : ls)
    BOOST_CHECK(addrs.count(w.out_addr(0) + 4 * l) == 1);
  BOOST_CHECK_EQUAL(w.read_flags()[0], flag_done);
}

BOOST_AUTO_TEST_CASE(item_body_covers_group_when_group_is_larger_than_the_native_group) {
  world w;
  w.add_layer(1, 0, 0, 10);
  w.add_node(1, 10, 0);
  w.upload();
  w.run<0>(1, 4);
  const auto log = w.log();
  std::set<std::uint64_t> ls;
  for (const auto &e : log)
    if (e.tag == t_body)
      ls.insert(e.l);
  BOOST_CHECK_EQUAL(ls.size(), 10u);
  BOOST_CHECK_EQUAL(count_tag(log, t_body), 10u);
}

BOOST_AUTO_TEST_CASE(group_body_is_collective) {
  world w;
  w.add_layer(3, 0, 0, 3);
  w.add_node(3, 1, 1);
  w.upload();
  w.run<0>(3, 4);
  const auto log = w.log();
  BOOST_CHECK_EQUAL(count_tag(log, t_group_body), 12u);
  std::set<std::uint64_t> gs;
  for (const auto &e : log)
    if (e.tag == t_group_body)
      gs.insert(e.g);
  BOOST_CHECK_EQUAL(gs.size(), 3u);
  for (std::uint32_t f : w.read_flags())
    BOOST_CHECK_EQUAL(f, flag_done);
}

BOOST_AUTO_TEST_CASE(resolve_tree_calls_global_resolve_for_global_links) {
  world w;
  w.add_layer(1, 0, 0, 4);
  w.add_node(1, 4, 0);
  w.add_node(1, 4, 0);
  w.add_local_link(0, 1, 3, 4, 0, 16, 64);
  w.links.back().kind = 2;
  w.upload();
  w.run<0>(1, 4);
  const auto log = w.log();
  BOOST_CHECK_EQUAL(count_tag(log, t_global_resolve), 12u);
  BOOST_CHECK_EQUAL(count_tag(log, t_resolve_tree), 0u);
  BOOST_CHECK_EQUAL(count_tag(log, t_prep), 0u);
  std::set<std::uint64_t> srcs;
  for (const auto &e : log)
    if (e.tag == t_global_resolve)
      srcs.insert(e.g);
  BOOST_CHECK((srcs == std::set<std::uint64_t>{0, 1, 2}));
}

BOOST_AUTO_TEST_CASE(resolve_layer_done_does_nothing) {
  world w;
  w.add_layer(1, 0, 0, 4);
  w.add_node(1, 4, 0);
  w.flags = {flag_done};
  w.upload();
  w.run<1>(1, 4);
  BOOST_CHECK_EQUAL(count_tag(w.log(), t_body), 0u);
  BOOST_CHECK_EQUAL(w.read_flags()[0], flag_done);
}

BOOST_AUTO_TEST_CASE(resolve_layer_unclaimed_computes_inline) {
  world w;
  w.add_layer(1, 0, 0, 4);
  w.add_node(1, 4, 0);
  w.upload();
  w.run<1>(1, 4);
  BOOST_CHECK_EQUAL(count_tag(w.log(), t_body), 4u);
  BOOST_CHECK_EQUAL(w.read_flags()[0], flag_done);
}

BOOST_AUTO_TEST_CASE(resolve_layer_waits_for_another_group) {
  if (std::thread::hardware_concurrency() < 2) {
    BOOST_WARN_MESSAGE(false, "SKIPPED resolve_layer_waits_for_another_group: needs at least 2 hardware threads, found " +
                                  std::to_string(std::thread::hardware_concurrency()));
    return;
  }
  world w;
  w.add_layer(2, 0, 0, 8);
  w.add_node(2, 4, 0);
  w.flags = {flag_in_progress, flag_unclaimed};
  w.upload();
  w.run<2>(2, 4);
  BOOST_CHECK_EQUAL(count_tag(w.log(), t_body), 0u);
  BOOST_CHECK_EQUAL(w.read_flags()[0], flag_done);
}

BOOST_AUTO_TEST_SUITE_END()
