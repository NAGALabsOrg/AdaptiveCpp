#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include "hipSYCL/sg/detail/bound_block.hpp"
#include "hipSYCL/sg/detail/closure.hpp"
#include "hipSYCL/sg/detail/dev/abi.hpp"
#include "hipSYCL/sg/detail/dev/layout.hpp"
#include "hipSYCL/sg/detail/tree_store.hpp"

#include <algorithm>
#include <cstring>
#include <set>

using namespace sg::detail;
using namespace sgt;

BOOST_AUTO_TEST_SUITE(def_graph_suite)
BOOST_AUTO_TEST_CASE(shapes) {
  const def_node i = item_def("a", 4, 8);
  BOOST_CHECK_EQUAL(i.grid()[0], 4u);
  BOOST_CHECK_EQUAL(i.total_shape()[0], 32u);
  BOOST_CHECK_EQUAL(i.group_elems(), 8u);
  const def_node g = group_def("g", 5, 3, {mk_link(link_kind::local, 0)});
  BOOST_CHECK_EQUAL(g.grid()[0], 5u);
  BOOST_CHECK_EQUAL(g.total_shape()[0], 15u);
  BOOST_CHECK_EQUAL(g.group_elems(), 3u);
}
BOOST_AUTO_TEST_CASE(passing_graph) { validate(a_to_f()); }
BOOST_AUTO_TEST_CASE(rules) {
  {
    def_graph g;
    g.nodes.push_back(group_def("g", 1, 1, {}));
    SG_CHECK_ERRC(validate(g), builder_misuse);
  }
  {
    def_graph g;
    g.nodes.push_back(item_def("a", 1, 1, {mk_link(link_kind::local, 0)}));
    SG_CHECK_ERRC(validate(g), builder_misuse);
  }
  {
    def_graph g;
    g.nodes.push_back(item_def("a", 1, 1));
    g.nodes.push_back(item_def("b", 1, 1, {mk_link(link_kind::local, -1)}));
    SG_CHECK_ERRC(validate(g), builder_misuse);
  }
  {
    def_graph g;
    g.nodes.push_back(item_def("a", 1, 1));
    g.sinks = {plain_sink(3, 0)};
    SG_CHECK_ERRC(validate(g), builder_misuse);
  }
  {
    def_graph g;
    g.nodes.push_back(item_def("a", 1, 1));
    g.sinks = {plain_sink(0, 0), plain_sink(0, 0)};
    SG_CHECK_ERRC(validate(g), builder_misuse);
  }
  {
    def_graph g;
    g.nodes.push_back(group_def("g", 1, 1, {mk_link(link_kind::local, 0)}));
    g.nodes.clear();
    g.nodes.push_back(item_def("a", 1, 1));
    g.nodes.push_back(group_def("g", 1, 1, {mk_link(link_kind::local, 0)}));
    g.nodes.push_back(item_def("c", 1, 1, {mk_link(link_kind::item, 1)}));
    SG_CHECK_ERRC(validate(g), mapping_mismatch);
  }
  {
    def_graph g;
    g.nodes.push_back(item_def("a", 1, 2));
    g.nodes.push_back(item_def("b", 1, 4, {mk_link(link_kind::item, 0)}));
    SG_CHECK_ERRC(validate(g), mapping_mismatch);
  }
  {
    def_graph g;
    g.nodes.push_back(item_def("a", 1, 3));
    g.nodes[0].range = sycl::range<3>(10, 1, 1);
    SG_CHECK_ERRC(validate(g), shape_mismatch);
  }
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(durable_graph_suite)
BOOST_AUTO_TEST_CASE(single_sink) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 2));
  g.sinks = {plain_sink(0, 0)};
  const durable_graph d = build_durable_graph(g);
  BOOST_REQUIRE_EQUAL(d.layers.size(), 1u);
  BOOST_CHECK(d.layers[0].result_backed);
  BOOST_CHECK_EQUAL(d.layers[0].write_symbol, "store_a");
  BOOST_CHECK_EQUAL(d.layers[0].tree.size(), 1u);
}
BOOST_AUTO_TEST_CASE(item_and_local_inlined) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 2));
  g.nodes.push_back(item_def("b", 2, 2, {mk_link(link_kind::item, 0)}));
  g.nodes.push_back(item_def("c", 2, 2, {mk_link(link_kind::local, 1)}));
  g.sinks = {plain_sink(2, 0)};
  const durable_graph d = build_durable_graph(g);
  BOOST_REQUIRE_EQUAL(d.layers.size(), 1u);
  BOOST_CHECK_EQUAL(d.layers[0].tree.size(), 3u);
  BOOST_CHECK(d.layers[0].deps.empty());
}
BOOST_AUTO_TEST_CASE(local_pulled_twice_appears_twice) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 2));
  g.nodes.push_back(item_def("d", 2, 2, {mk_link(link_kind::local, 0), mk_link(link_kind::local, 0)}));
  g.sinks = {plain_sink(1, 0)};
  const durable_graph d = build_durable_graph(g);
  BOOST_CHECK_EQUAL(d.layers[0].tree.size(), 3u);
}
BOOST_AUTO_TEST_CASE(global_shared) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 2));
  g.nodes.push_back(item_def("x", 2, 2, {mk_link(link_kind::global, 0)}));
  g.nodes.push_back(item_def("y", 2, 2, {mk_link(link_kind::global, 0)}));
  g.sinks = {plain_sink(1, 0), plain_sink(2, 1)};
  const durable_graph d = build_durable_graph(g);
  BOOST_REQUIRE_EQUAL(d.layers.size(), 3u);
  BOOST_CHECK(!d.layers[2].result_backed);
  BOOST_CHECK((d.layers[0].deps == std::vector<int>{2}));
  BOOST_CHECK((d.layers[1].deps == std::vector<int>{2}));
}
BOOST_AUTO_TEST_CASE(plain_sink_is_the_shared_layer) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 2));
  g.nodes.push_back(item_def("x", 2, 2, {mk_link(link_kind::global, 0)}));
  g.sinks = {plain_sink(0, 0), plain_sink(1, 1)};
  const durable_graph d = build_durable_graph(g);
  BOOST_REQUIRE_EQUAL(d.layers.size(), 2u);
  BOOST_CHECK((d.layers[1].deps == std::vector<int>{0}));
}
BOOST_AUTO_TEST_CASE(inplace_sink_does_not_back_global) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 2));
  g.nodes.push_back(item_def("x", 2, 2, {mk_link(link_kind::global, 0)}));
  g.sinks = {inplace_sink(0, 0), plain_sink(1, 1)};
  const durable_graph d = build_durable_graph(g);
  BOOST_REQUIRE_EQUAL(d.layers.size(), 3u);
  BOOST_CHECK(!d.layers[2].result_backed);
  BOOST_CHECK((d.layers[1].deps == std::vector<int>{2}));
}
BOOST_AUTO_TEST_CASE(item_of_item_reaches_global) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 2));
  g.nodes.push_back(item_def("b", 2, 2, {mk_link(link_kind::global, 0)}));
  g.nodes.push_back(item_def("c", 2, 2, {mk_link(link_kind::item, 1)}));
  g.sinks = {plain_sink(2, 0)};
  const durable_graph d = build_durable_graph(g);
  BOOST_REQUIRE_EQUAL(d.layers.size(), 2u);
  BOOST_CHECK_EQUAL(d.layers[0].tree.size(), 2u);
  BOOST_CHECK((d.layers[0].deps == std::vector<int>{1}));
}
BOOST_AUTO_TEST_CASE(a_to_f) {
  const def_graph g = sgt::a_to_f();
  const durable_graph d = build_durable_graph(g);
  BOOST_REQUIRE_EQUAL(d.layers.size(), 4u);
  BOOST_CHECK_EQUAL(d.layers[0].def, 1);
  BOOST_CHECK_EQUAL(d.layers[1].def, 2);
  BOOST_CHECK_EQUAL(d.layers[2].def, 3);
  BOOST_CHECK_EQUAL(d.layers[3].def, 5);
  BOOST_CHECK_EQUAL(d.layers[0].tree.size(), 2u);
  BOOST_CHECK_EQUAL(d.layers[1].tree.size(), 2u);
  BOOST_CHECK_EQUAL(d.layers[2].tree.size(), 5u);
  BOOST_CHECK_EQUAL(d.layers[3].tree.size(), 2u);
  BOOST_CHECK((d.layers[3].deps == std::vector<int>{1}));
  BOOST_CHECK(d.layers[0].deps.empty());
  int a_count = 0;
  for (const inline_node &n : d.layers[2].tree)
    a_count += n.def == 0;
  BOOST_CHECK_EQUAL(a_count, 2);
}
BOOST_AUTO_TEST_CASE(order_and_hash) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 2));
  g.nodes.push_back(item_def("b", 2, 2, {mk_link(link_kind::global, 0)}));
  g.nodes.push_back(item_def("c", 2, 2, {mk_link(link_kind::global, 1)}));
  g.sinks = {plain_sink(2, 0), plain_sink(1, 1), plain_sink(0, 2)};
  const durable_graph d = build_durable_graph(g);
  const auto order = upstream_order(d);
  std::vector<int> pos(d.layers.size());
  for (std::size_t p = 0; p < order.size(); ++p)
    pos[order[p]] = static_cast<int>(p);
  for (std::size_t i = 0; i < d.layers.size(); ++i)
    for (int dep : d.layers[i].deps)
      BOOST_CHECK(pos[dep] < pos[i]);
  // independent sinks: creation order decides
  def_graph h;
  h.nodes.push_back(item_def("a", 2, 2));
  h.nodes.push_back(item_def("b", 2, 2));
  h.sinks = {plain_sink(0, 0), plain_sink(1, 1)};
  def_graph h2 = h;
  h2.sinks = {plain_sink(1, 1), plain_sink(0, 0)};
  const durable_graph d1 = build_durable_graph(h), d2 = build_durable_graph(h2);
  BOOST_CHECK((upstream_order(d1) == std::vector<int>{0, 1}));
  BOOST_CHECK_NE(structure_hash(d1, h, upstream_order(d1)), structure_hash(d2, h2, upstream_order(d2)));
}
BOOST_AUTO_TEST_CASE(hash_ignores_values_and_sees_structure) {
  auto hash_of = [](const def_graph &g) {
    const durable_graph d = build_durable_graph(g);
    return structure_hash(d, g, upstream_order(d));
  };
  const def_graph base = sgt::a_to_f();
  const auto h0 = hash_of(base);
  def_graph v = base;
  v.nodes[0].range = sycl::range<3>(64, 1, 1);
  v.nodes[3].links[0].count = 9;
  v.nodes[1].closure = 3;
  v.nodes[2].scratch_bytes = 77;
  v.sinks[0].slot = 9;
  BOOST_CHECK_EQUAL(hash_of(v), h0);
  const std::vector<std::function<void(def_graph &)>> changes = {
      [](def_graph &g) { g.nodes[1].body_symbol = "other"; },
      [](def_graph &g) { g.nodes[1].store_symbol = "other"; },
      [](def_graph &g) { g.nodes[1].type = 77; },
      [](def_graph &g) { g.nodes[1].rank = 2; },
      [](def_graph &g) { g.nodes[3].links[0].positions_symbol = "other"; },
      [](def_graph &g) { g.nodes[3].links[0].form = sg::form_kind::block; },
      [](def_graph &g) { g.nodes[4].links[0].kind = link_kind::local; },
      [](def_graph &g) { g.sinks[1].kind = result_kind::inplace; g.sinks[1].op_symbol = "op"; },
  };
  for (const auto &c : changes) {
    def_graph m = base;
    c(m);
    BOOST_CHECK_NE(hash_of(m), h0);
  }
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(composition_builder_suite)
BOOST_AUTO_TEST_CASE(numbering) {
  const def_graph g = sgt::a_to_f();
  const durable_graph d = build_durable_graph(g);
  const auto order = upstream_order(d);
  const auto c = make_composition_config(d, g, order);
  BOOST_CHECK_EQUAL(c.nodes.size(), 2u + 2u + 5u + 2u);
  BOOST_REQUIRE_EQUAL(c.layers.size(), 4u);
  BOOST_CHECK_EQUAL(c.layers[0].root_node, 0u);
  BOOST_CHECK_EQUAL(c.layers[1].root_node, 2u);
  BOOST_CHECK_EQUAL(c.layers[2].root_node, 4u);
  BOOST_CHECK_EQUAL(c.layers[3].root_node, 9u);
  BOOST_CHECK_EQUAL(c.nodes[0].write_symbol, "store_B");
  BOOST_CHECK_EQUAL(c.nodes[1].write_symbol, "store_A");
  BOOST_CHECK_EQUAL(c.nodes[0].links[0].source, 1u);        // child node
  BOOST_CHECK_EQUAL(c.nodes[9].links[0].kind, 0u);          // F item-links to E
  BOOST_CHECK_EQUAL(c.nodes[9].links[0].source, 10u);       // child node
  BOOST_CHECK_EQUAL(c.nodes[10].links[0].kind, 2u);         // E global-links to C
  BOOST_CHECK_EQUAL(c.nodes[10].links[0].source, 1u);       // order position of layer C
  BOOST_CHECK_EQUAL(c.unique_hash, structure_hash(d, g, order));
}
BOOST_AUTO_TEST_CASE(global_source_is_layer_position) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 2));
  g.nodes.push_back(item_def("b", 2, 2, {mk_link(link_kind::global, 0)}));
  g.sinks = {plain_sink(1, 0), plain_sink(0, 1)};
  const durable_graph d = build_durable_graph(g);
  const auto order = upstream_order(d);
  const auto c = make_composition_config(d, g, order);
  // layer for a comes first in the order
  BOOST_CHECK_EQUAL(c.nodes[1].links[0].source, 0u);
}
BOOST_AUTO_TEST_CASE(imported_symbols) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 2));
  g.nodes.push_back(item_def("b", 2, 2, {mk_link(link_kind::item, 0, "p")}));
  g.nodes.push_back(item_def("c", 2, 2, {mk_link(link_kind::item, 0, "p")}));
  g.nodes[2].body_symbol = "body_b";
  g.sinks = {plain_sink(1, 0), plain_sink(2, 1)};
  const durable_graph d = build_durable_graph(g);
  const auto c = make_composition_config(d, g, upstream_order(d));
  BOOST_CHECK_EQUAL(c.nodes.size(), 4u);
  const auto syms = c.imported_symbols();
  BOOST_CHECK_EQUAL(std::count(syms.begin(), syms.end(), std::string("body_b")), 1);
  BOOST_CHECK(std::find(syms.begin(), syms.end(), std::string("")) == syms.end());
  BOOST_CHECK_EQUAL(syms.front(), "body_b");
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(launch_plan_suite)
namespace {
device_limits limits() { return device_limits{1024, 48 * 1024, 0xffffffffull}; }
struct three_layers {
  def_graph g;
  durable_graph d;
  std::vector<int> order;
  three_layers() {
    g.nodes.push_back(item_def("a", 8, 4));
    g.nodes.push_back(item_def("b", 16, 4, {mk_link(link_kind::global, 0)}));
    g.nodes.push_back(item_def("c", 32, 4, {mk_link(link_kind::global, 1)}));
    g.sinks = {plain_sink(0, 0), plain_sink(1, 1), plain_sink(2, 2)};
    d = build_durable_graph(g);
    order = upstream_order(d);
  }
};
} // namespace
BOOST_AUTO_TEST_CASE(grid_and_offsets) {
  three_layers t;
  const auto p = plan_launch(t.d, t.g, t.order, {}, limits());
  BOOST_CHECK_EQUAL(p.grid, 56u);
  BOOST_CHECK_EQUAL(p.layers[0].range_offset, 0u);
  BOOST_CHECK_EQUAL(p.layers[1].range_offset, 8u);
  BOOST_CHECK_EQUAL(p.layers[2].range_offset, 24u);
  BOOST_CHECK_EQUAL(p.layers[1].groups, 16u);
  BOOST_CHECK_EQUAL(p.layers[1].extent[0], 64u);
  BOOST_CHECK_EQUAL(p.layers[1].backing_bytes, 64u * 4);
  BOOST_CHECK_EQUAL(p.work_group_size, 4u);
  BOOST_CHECK_EQUAL(p.flag_words, 56u);
  BOOST_CHECK_EQUAL(p.local_bytes, 256u);
}
BOOST_AUTO_TEST_CASE(work_group_size) {
  def_graph g;
  g.nodes.push_back(group_def("g", 3, 2, {mk_link(link_kind::local, 0)}));
  g.nodes.insert(g.nodes.begin(), item_def("a", 3, 2));
  g.nodes[1].links[0].source = 0;
  g.sinks = {plain_sink(1, 0)};
  durable_graph d = build_durable_graph(g);
  auto p = plan_launch(d, g, upstream_order(d), {}, limits());
  BOOST_CHECK_EQUAL(p.work_group_size, 2u);
  def_graph h;
  h.nodes.push_back(item_def("a", 3, 2));
  h.nodes.push_back(group_def("g", 3, 2, {mk_link(link_kind::global, 0)}));
  h.sinks = {plain_sink(1, 0)};
  d = build_durable_graph(h);
  auto q = plan_launch(d, h, upstream_order(d), {}, limits());
  BOOST_CHECK_EQUAL(q.work_group_size, 2u);
  def_graph only;
  only.nodes.push_back(item_def("a", 3, 2));
  only.nodes.push_back(group_def("g", 3, 2, {mk_link(link_kind::global, 0)}));
  only.nodes[0].kind = body_kind::group_body; // both group bodies
  only.nodes[0].links = {mk_link(link_kind::local, 0)};
  only.nodes[0].links.clear();
  only.sinks = {};
  durable_graph e;
  e.layers.push_back({});
  e.layers[0].def = 1;
  e.layers[0].tree = {{1, -1, -1}};
  const auto r = plan_launch(e, only, {0}, {}, limits());
  BOOST_CHECK_EQUAL(r.work_group_size, 128u);
}
BOOST_AUTO_TEST_CASE(too_large_and_limits_in_order) {
  three_layers t;
  device_limits l = limits();
  l.max_work_group_size = 2;
  SG_CHECK_ERRC(plan_launch(t.d, t.g, t.order, {}, l), work_group_too_large);
  l.max_groups = 10;
  SG_CHECK_ERRC(plan_launch(t.d, t.g, t.order, {}, l), limit_exceeded);
}
BOOST_AUTO_TEST_CASE(local_frame) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 4));
  g.nodes.push_back(item_def("b", 2, 4, {mk_link(link_kind::local, 0, "pos", 3)}));
  g.sinks = {plain_sink(1, 0)};
  const durable_graph d = build_durable_graph(g);
  const auto order = upstream_order(d);
  const auto p = plan_launch(d, g, order, {0, 0, 0}, limits());
  const node_plan &np = p.nodes[0];
  BOOST_CHECK_EQUAL(np.links[0].slot_bytes, 16u);
  BOOST_CHECK_EQUAL(np.links[0].slots_offset, 0u);
  BOOST_CHECK_EQUAL(np.frame_bytes, 48u);
  BOOST_CHECK_EQUAL(p.local_bytes, 256u);
  device_limits l = limits();
  l.local_mem_bytes = 32;
  try {
    plan_launch(d, g, order, {0, 0, 0}, l);
    BOOST_FAIL("no throw");
  } catch (const sg::error &e) {
    BOOST_CHECK(e.code() == sg::errc::local_memory_exceeded);
    const std::string m = e.what();
    BOOST_CHECK(m.find("48") != std::string::npos);
    BOOST_CHECK(m.find("32") != std::string::npos);
  }
}
BOOST_AUTO_TEST_CASE(group_body_scratch) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 4));
  g.nodes.push_back(group_def("g", 2, 1, {mk_link(link_kind::local, 0)}, 40));
  g.sinks = {plain_sink(1, 0)};
  const durable_graph d = build_durable_graph(g);
  const auto p = plan_launch(d, g, upstream_order(d), {0, 0}, limits());
  BOOST_CHECK_EQUAL(p.nodes[0].scratch_offset, 16u);
  BOOST_CHECK_EQUAL(p.nodes[0].frame_bytes, 16u + 48u);
}
BOOST_AUTO_TEST_CASE(bucket) {
  BOOST_CHECK_EQUAL(local_bucket(0), 256u);
  BOOST_CHECK_EQUAL(local_bucket(1), 256u);
  BOOST_CHECK_EQUAL(local_bucket(256), 256u);
  BOOST_CHECK_EQUAL(local_bucket(257), 512u);
  BOOST_CHECK_EQUAL(local_bucket(4097), 8192u);
}
BOOST_AUTO_TEST_CASE(bound_block_layout) {
  def_graph g;
  g.nodes.push_back(item_def("a", 2, 4));
  g.nodes.push_back(item_def("b", 2, 4, {mk_link(link_kind::global, 0)}));
  g.sinks = {plain_sink(1, 0)};
  const durable_graph d = build_durable_graph(g);
  const auto order = upstream_order(d);
  const auto p = plan_launch(d, g, order, {24, 8}, limits());
  BOOST_CHECK_EQUAL(p.layers_offset, 64u);
  BOOST_CHECK(p.nodes_offset >= p.layers_offset + 2 * sizeof(dev::layer_rec));
  BOOST_CHECK(p.links_offset >= p.nodes_offset + 2 * sizeof(dev::node_rec));
  for (const node_plan &n : p.nodes) {
    BOOST_CHECK_EQUAL(n.closure_offset % 16, 0u);
    for (auto o : n.link_closure_offsets)
      BOOST_CHECK_EQUAL(o % 16, 0u);
  }
  BOOST_CHECK_EQUAL(p.bound_block_bytes % 64, 0u);
  BOOST_CHECK_EQUAL(p.flags_byte_offset, p.bound_block_bytes);
  BOOST_CHECK_EQUAL(p.flag_words, 4u);
  // the toolchain layer (def a) has a 64-aligned backing after the flags
  std::size_t toolchain_layers = 0;
  for (std::size_t i = 0; i < order.size(); ++i)
    if (!d.layers[order[i]].result_backed) {
      ++toolchain_layers;
      BOOST_CHECK_EQUAL(p.layers[i].backing_offset % 64, 0u);
      BOOST_CHECK(p.layers[i].backing_offset >= p.flags_byte_offset + 16);
    }
  BOOST_CHECK_EQUAL(toolchain_layers, 1u);
  BOOST_CHECK(p.toolchain_bytes >= p.flags_byte_offset + p.flag_words * 4 + 8 * 4);
}
BOOST_AUTO_TEST_CASE(grid_zero) {
  def_graph g;
  g.nodes.push_back(item_def("a", 0, 4));
  g.sinks = {plain_sink(0, 0)};
  const durable_graph d = build_durable_graph(g);
  const auto p = plan_launch(d, g, upstream_order(d), {}, limits());
  BOOST_CHECK_EQUAL(p.grid, 0u);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(layout_suite)
BOOST_AUTO_TEST_CASE(record_sizes) {
  BOOST_CHECK_EQUAL(sizeof(dev::launch_ctx), 32u);
  BOOST_CHECK_EQUAL(sizeof(dev::layer_rec), 64u);
  BOOST_CHECK_EQUAL(sizeof(dev::node_rec), 128u);
  BOOST_CHECK_EQUAL(sizeof(dev::link_rec), 128u);
  BOOST_CHECK_EQUAL(sizeof(dev::tile_out), 64u);
  BOOST_CHECK_EQUAL(offsetof(dev::layer_rec, extent), 32u);
  BOOST_CHECK_EQUAL(offsetof(dev::node_rec, closure), 72u);
  BOOST_CHECK_EQUAL(offsetof(dev::node_rec, rank), 112u);
  BOOST_CHECK_EQUAL(offsetof(dev::link_rec, count), 16u);
  BOOST_CHECK_EQUAL(offsetof(dev::link_rec, src_grid), 72u);
}
BOOST_AUTO_TEST_CASE(flatten_unflatten) {
  const std::uint64_t dims[3] = {3, 4, 5};
  for (std::uint64_t k = 0; k < 60; ++k) {
    std::uint64_t id[3];
    dev::unflatten3(k, dims, id);
    BOOST_CHECK_EQUAL(dev::flatten3(id, dims), k);
    BOOST_CHECK_EQUAL(id[2], k % 5);
    BOOST_CHECK_EQUAL(id[0], k / 20);
  }
}
BOOST_AUTO_TEST_CASE(tiles_cover_extent) {
  struct cfg { std::uint64_t grid[3], gs[3]; };
  for (const cfg &c : {cfg{{4, 1, 1}, {2, 1, 1}}, cfg{{2, 3, 1}, {2, 2, 1}}, cfg{{2, 2, 2}, {1, 2, 3}},
                       cfg{{1, 1, 1}, {3, 3, 3}}}) {
    dev::node_rec n{};
    dev::layer_rec l{};
    std::uint64_t total = 1;
    for (int d = 0; d < 3; ++d) {
      n.grid[d] = c.grid[d];
      n.group_shape[d] = c.gs[d];
      l.extent[d] = c.grid[d] * c.gs[d];
      total *= l.extent[d];
    }
    n.elem_bytes = 4;
    l.backing = 1000;
    std::set<std::uint64_t> seen;
    const std::uint64_t groups = c.grid[0] * c.grid[1] * c.grid[2];
    const std::uint64_t per = c.gs[0] * c.gs[1] * c.gs[2];
    for (std::uint64_t g = 0; g < groups; ++g) {
      const auto t = dev::layer_tile(l, n, g);
      for (std::uint64_t k = 0; k < per; ++k) {
        const auto a = dev::tile_address(t, k);
        BOOST_CHECK(a >= 1000 && a < 1000 + total * 4 && (a - 1000) % 4 == 0);
        seen.insert(a);
      }
    }
    BOOST_CHECK_EQUAL(seen.size(), total);
  }
}
BOOST_AUTO_TEST_CASE(layer_tile_origin_row_major_2d) {
  dev::node_rec n{};
  dev::layer_rec l{};
  n.grid[0] = 2; n.grid[1] = 2; n.grid[2] = 1;
  n.group_shape[0] = 2; n.group_shape[1] = 3; n.group_shape[2] = 1;
  n.elem_bytes = 8;
  l.extent[0] = 4; l.extent[1] = 6; l.extent[2] = 1;
  l.backing = 0;
  const auto t = dev::layer_tile(l, n, 3); // group (1,1): origin (2,3)
  BOOST_CHECK_EQUAL(t.base, (2u * 6u + 3u) * 8u);
  BOOST_CHECK_EQUAL(t.stride[0], 6u);
  BOOST_CHECK_EQUAL(dev::tile_address(t, 4), t.base + (1u * 6u + 1u) * 8u);
}
BOOST_AUTO_TEST_CASE(dense) {
  const std::uint64_t gs[3] = {2, 3, 1};
  const auto t = dev::dense_tile(64, gs, 4);
  for (std::uint64_t l = 0; l < 6; ++l)
    BOOST_CHECK_EQUAL(dev::tile_address(t, l), 64 + l * 4);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(bound_block_suite)
namespace {
struct fixture {
  kernel_tree tree;
  launch_plan plan;
  std::vector<slot_binding> binds;
  fixture() {
    tree.defs.nodes.push_back(item_def("a", 2, 4));
    tree.defs.nodes.push_back(item_def("b", 2, 4, {mk_link(link_kind::global, 0)}));
    tree.defs.nodes[1].links[0].form = sg::form_kind::single;
    tree.defs.sinks = {plain_sink(1, 0)};
    array_rep r{};
    r.address = sentinel(1, 0);
    r.extent[0] = 8;
    tree.closures.push_back(capture_closure(r));
    tree.closures.push_back(capture_closure(r));
    tree.closures.push_back(capture_closure(std::uint64_t{0x1234}));
    tree.defs.nodes[0].closure = 0;
    tree.defs.nodes[1].closure = 1;
    tree.defs.nodes[1].links[0].closure = 2;
    tree.durable = build_durable_graph(tree.defs);
    tree.order = upstream_order(tree.durable);
    std::vector<std::size_t> sizes;
    for (auto &c : tree.closures)
      sizes.push_back(c.bytes.size());
    plan = plan_launch(tree.durable, tree.defs, tree.order, sizes, {1024, 48 * 1024, 1u << 30});
    binds.resize(1);
    binds[0].address = 0xAAA000;
    binds[0].extent[0] = 8;
  }
};
} // namespace
BOOST_AUTO_TEST_CASE(contents) {
  fixture f;
  const std::uint64_t base = 0x100000;
  const auto b = build_bound_block(f.tree, f.plan, f.binds, {0, 0xAAA000}, base);
  BOOST_CHECK_EQUAL(b.size(), f.plan.bound_block_bytes);
  dev::launch_ctx ctx;
  std::memcpy(&ctx, b.data(), sizeof(ctx));
  BOOST_CHECK_EQUAL(ctx.layers, base + f.plan.layers_offset);
  BOOST_CHECK_EQUAL(ctx.nodes, base + f.plan.nodes_offset);
  BOOST_CHECK_EQUAL(ctx.links, base + f.plan.links_offset);
  BOOST_CHECK_EQUAL(ctx.num_layers, 2u);
  dev::layer_rec l0, l1;
  std::memcpy(&l0, b.data() + f.plan.layers_offset, sizeof(l0));
  std::memcpy(&l1, b.data() + f.plan.layers_offset + sizeof(l0), sizeof(l1));
  BOOST_CHECK_EQUAL(l0.backing, base + f.plan.layers[0].backing_offset); // toolchain layer
  BOOST_CHECK_EQUAL(l1.backing, 0xAAA000u);                              // result
  BOOST_CHECK_EQUAL(l1.flags, base + f.plan.flags_byte_offset + 4 * f.plan.layers[1].flags_word_offset);
  BOOST_CHECK_EQUAL(l1.groups, 2u);
  dev::node_rec n1;
  std::memcpy(&n1, b.data() + f.plan.nodes_offset + sizeof(n1), sizeof(n1));
  BOOST_CHECK_EQUAL(n1.closure, base + f.plan.nodes[1].closure_offset);
  BOOST_CHECK_EQUAL(n1.group_shape[0], 4u);
  BOOST_CHECK_EQUAL(n1.link_count, 1u);
  dev::link_rec k;
  std::memcpy(&k, b.data() + f.plan.links_offset, sizeof(k));
  BOOST_CHECK_EQUAL(k.kind, 2u);
  BOOST_CHECK_EQUAL(k.source, 0u);
  std::uint64_t patched;
  std::memcpy(&patched, b.data() + f.plan.nodes[1].closure_offset, 8);
  BOOST_CHECK_EQUAL(patched, 0xAAA000u);
  BOOST_CHECK(f.tree.closures[1].bytes == capture_closure([&] { array_rep r{}; r.address = sentinel(1, 0); r.extent[0] = 8; return r; }()).bytes);
}
BOOST_AUTO_TEST_CASE(rebinding_and_determinism) {
  fixture f;
  const auto a = build_bound_block(f.tree, f.plan, f.binds, {0, 0xAAA000}, 0x100000);
  const auto a2 = build_bound_block(f.tree, f.plan, f.binds, {0, 0xAAA000}, 0x100000);
  BOOST_CHECK(a == a2);
  f.binds[0].address = 0xBBB000;
  const auto c = build_bound_block(f.tree, f.plan, f.binds, {0, 0xBBB000}, 0x100000);
  std::size_t diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    diff += a[i] != c[i];
  BOOST_CHECK(diff > 0);
  // padding after the last record of the block is zero
  for (std::size_t i = f.plan.bound_block_bytes - 8; i < f.plan.bound_block_bytes; ++i)
    BOOST_CHECK(c[i] == std::byte{0});
}
BOOST_AUTO_TEST_SUITE_END()
