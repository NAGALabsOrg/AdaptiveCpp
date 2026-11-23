#include "../common/check.hpp"
#include "../common/mocks.hpp"

#include "hipSYCL/runtime/composition_adaptivity.hpp"
#include "hipSYCL/sg/detail/kernel_preparer.hpp"
#include "hipSYCL/sg/detail/launcher.hpp"
#include "hipSYCL/sg/detail/tree_store.hpp"
#include "hipSYCL/sg/kernel.hpp"
#include "hipSYCL/sg/mapping_forms.hpp"
#include "hipSYCL/sg/roles.hpp"

using namespace sg;
using namespace sg::detail;

namespace {
device_limits lim() { return device_limits{1024, 48 * 1024, 1u << 30}; }

struct builder_a {};
struct builder_b {};

// a node with one result slot of `n` ints and a builder running `body`
struct kernel_node {
  sgt::rig r;
  node_record *n;
  array<result<int>, 1> *out = nullptr;
  std::unique_ptr<array<result<int>, 1>> out_holder;
  kernel_node(std::size_t count = 8) {
    n = &r.impl->begin_scope(sycl::device{});
    auto h = handle_access::make<mem_handle<int>>(r.impl.get(), r.impl->add_handle(type_id<int>(), 4));
    out_holder = std::make_unique<array<result<int>, 1>>(h, shape<1>(sycl::range<1>(count)));
    n->kind = node_record::kind_t::kernel;
    n->defines = 1;
  }
  ~kernel_node() { r.impl->abort_scope(); }
  template <class B> void set_builder(B b, std::uint64_t type = type_id<builder_a>()) {
    n->builder = std::move(b);
    n->builder_type = type;
  }
  std::vector<slot_binding> bindings(std::size_t count = 8) {
    std::vector<slot_binding> v(1);
    v[0].address = 0x1000;
    v[0].extent[0] = count;
    return v;
  }
};
} // namespace

BOOST_AUTO_TEST_SUITE(tree_store_suite)
BOOST_AUTO_TEST_CASE(find_insert) {
  tree_store s;
  auto k1 = make_key(std::make_tuple(1, 2));
  BOOST_CHECK(!s.find(1, *k1));
  auto t = std::make_shared<kernel_tree>();
  s.insert(1, make_key(std::make_tuple(1, 2)), t);
  BOOST_CHECK(s.find(1, *k1) == t);
  auto k2 = make_key(std::make_tuple(1, 3));
  BOOST_CHECK(!s.find(1, *k2));
  BOOST_CHECK(!s.find(2, *k1));
  auto t2 = std::make_shared<kernel_tree>();
  s.insert(1, make_key(std::make_tuple(1, 3)), t2);
  BOOST_CHECK(s.find(1, *k1) == t);
  BOOST_CHECK(s.find(1, *k2) == t2);
  auto k3 = make_key(std::make_tuple(std::string("x")));
  BOOST_CHECK(!s.find(1, *k3));
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(builder_suite)
BOOST_AUTO_TEST_CASE(item_definition) {
  kernel_node kn;
  auto *out = kn.out_holder.get();
  int calls = 0;
  kn.set_builder([out, &calls](kernel_handle &kh) {
    ++calls;
    auto d = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1>) { return 7; });
    *out = kh.evaluate(d);
  });
  tree_store store;
  kernel_preparer prep(store);
  const auto pk = prep.prepare(*kn.n, kn.bindings(), lim());
  BOOST_CHECK_EQUAL(calls, 1);
  const auto &defs = pk.tree->defs;
  BOOST_REQUIRE_EQUAL(defs.nodes.size(), 1u);
  BOOST_CHECK(defs.nodes[0].kind == body_kind::item_body);
  BOOST_CHECK_EQUAL(defs.nodes[0].range[0], 8u);
  BOOST_CHECK_EQUAL(defs.nodes[0].group_shape[0], 4u);
  BOOST_CHECK_EQUAL(defs.nodes[0].type, type_id<int>());
  BOOST_CHECK(!defs.nodes[0].body_symbol.empty());
  BOOST_CHECK(!defs.nodes[0].store_symbol.empty());
  BOOST_REQUIRE_EQUAL(defs.sinks.size(), 1u);
  BOOST_CHECK_EQUAL(defs.sinks[0].slot, 0u);
  BOOST_CHECK_EQUAL(pk.plan.grid, 2u);
  BOOST_CHECK_EQUAL(pk.plan.work_group_size, 4u);
}
BOOST_AUTO_TEST_CASE(non_multiple_range) {
  kernel_node kn;
  kn.set_builder([](kernel_handle &kh) {
    kh.parallel_for(sycl::range<1>(10), sycl::range<1>(4), [](sg::item<1>) { return 1; });
  });
  tree_store store;
  kernel_preparer prep(store);
  SG_CHECK_ERRC(prep.prepare(*kn.n, kn.bindings(), lim()), shape_mismatch);
}
BOOST_AUTO_TEST_CASE(maps_and_group_body) {
  kernel_node kn(4);
  auto *out = kn.out_holder.get();
  kn.set_builder([out](kernel_handle &kh) {
    auto a = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); });
    auto m = kh.map_groups<mapping::local>(a, [](sycl::id<1> g) { return sg::block<1>{g, sycl::range<1>(2)}; });
    auto g = kh.parallel_for(sycl::range<1>(4), m,
                             [](sg::group<1>, group_iterator<int, 1, mapping::local> it) {
                               return group_reduce<1>(it[0], 0, sycl::plus<>{});
                             });
    *out = kh.evaluate(g);
  });
  tree_store store;
  kernel_preparer prep(store);
  const auto pk = prep.prepare(*kn.n, kn.bindings(4), lim());
  const auto &defs = pk.tree->defs;
  BOOST_REQUIRE_EQUAL(defs.nodes.size(), 2u);
  BOOST_CHECK(defs.nodes[1].kind == body_kind::group_body);
  BOOST_REQUIRE_EQUAL(defs.nodes[1].links.size(), 1u);
  BOOST_CHECK(defs.nodes[1].links[0].kind == link_kind::local);
  BOOST_CHECK(defs.nodes[1].links[0].form == form_kind::block);
  BOOST_CHECK_EQUAL(defs.nodes[1].links[0].count, 2u);
  BOOST_CHECK_EQUAL(defs.nodes[1].links[0].source, 0);
  BOOST_CHECK_EQUAL(defs.nodes[1].group_shape[0], 1u);
  BOOST_CHECK(defs.nodes[1].scratch_bytes > 0);
  BOOST_CHECK_EQUAL(defs.nodes[1].type, type_id<int>());
}
BOOST_AUTO_TEST_CASE(evaluate_twice_and_non_result) {
  kernel_node kn;
  auto *out = kn.out_holder.get();
  kn.set_builder([out](kernel_handle &kh) {
    auto d = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1>) { return 1; });
    *out = kh.evaluate(d);
    *out = kh.evaluate(d);
  });
  tree_store store;
  kernel_preparer prep(store);
  SG_CHECK_ERRC(prep.prepare(*kn.n, kn.bindings(), lim()), builder_misuse);
}
BOOST_AUTO_TEST_CASE(shape_mismatch_at_evaluate) {
  kernel_node kn(8);
  auto *out = kn.out_holder.get();
  kn.set_builder([out](kernel_handle &kh) {
    auto d = kh.parallel_for(sycl::range<1>(16), sycl::range<1>(4), [](sg::item<1>) { return 1; });
    *out = kh.evaluate(d);
  });
  tree_store store;
  kernel_preparer prep(store);
  SG_CHECK_ERRC(prep.prepare(*kn.n, kn.bindings(), lim()), shape_mismatch);
}
BOOST_AUTO_TEST_CASE(host_loop_makes_definitions) {
  kernel_node kn;
  auto *out = kn.out_holder.get();
  kn.set_builder([out](kernel_handle &kh) {
    auto d = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1>) { return 1; });
    for (int i = 0; i < 3; ++i) {
      auto m = kh.map_groups<mapping::global>(d, [](sycl::id<1> g) { return single<1>{g}; });
      d = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), m,
                          [](sg::item<1>, group_iterator<int, 1, mapping::global> it) { return int(it.size()); });
    }
    *out = kh.evaluate(d);
  });
  tree_store store;
  kernel_preparer prep(store);
  const auto pk = prep.prepare(*kn.n, kn.bindings(), lim());
  BOOST_CHECK_EQUAL(pk.tree->defs.nodes.size(), 4u);
  BOOST_CHECK_EQUAL(pk.tree->durable.layers.size(), 4u);
}
BOOST_AUTO_TEST_CASE(keys) {
  kernel_node kn;
  auto *out = kn.out_holder.get();
  int calls = 0;
  kn.set_builder([out, &calls](kernel_handle &kh) {
    ++calls;
    auto d = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1>) { return 1; });
    *out = kh.evaluate(d);
  });
  tree_store store;
  kernel_preparer prep(store);
  // no key: builder runs every time
  prep.prepare(*kn.n, kn.bindings(), lim());
  prep.prepare(*kn.n, kn.bindings(), lim());
  BOOST_CHECK_EQUAL(calls, 2);
  kn.n->key = make_key(std::make_tuple(1));
  auto first = prep.prepare(*kn.n, kn.bindings(), lim());
  BOOST_CHECK_EQUAL(calls, 3);
  kn.n->key = make_key(std::make_tuple(1));
  auto second = prep.prepare(*kn.n, kn.bindings(), lim());
  BOOST_CHECK_EQUAL(calls, 3);
  BOOST_CHECK(first.tree == second.tree);
  kn.n->key = make_key(std::make_tuple(2));
  prep.prepare(*kn.n, kn.bindings(), lim());
  BOOST_CHECK_EQUAL(calls, 4);
  kn.n->builder_type = type_id<builder_b>();
  kn.n->key = make_key(std::make_tuple(1));
  prep.prepare(*kn.n, kn.bindings(), lim());
  BOOST_CHECK_EQUAL(calls, 5);
  // a reused tree against a result of another size
  kn.n->builder_type = type_id<builder_a>();
  kn.n->key = make_key(std::make_tuple(1));
  SG_CHECK_ERRC(prep.prepare(*kn.n, kn.bindings(9), lim()), shape_mismatch);
}
BOOST_AUTO_TEST_CASE(builder_exception_not_stored) {
  kernel_node kn;
  int calls = 0;
  kn.set_builder([&calls](kernel_handle &) {
    ++calls;
    throw std::runtime_error("boom");
  });
  kn.n->key = make_key(std::make_tuple(1));
  tree_store store;
  kernel_preparer prep(store);
  BOOST_CHECK_THROW(prep.prepare(*kn.n, kn.bindings(), lim()), std::runtime_error);
  kn.n->key = make_key(std::make_tuple(1));
  BOOST_CHECK_THROW(prep.prepare(*kn.n, kn.bindings(), lim()), std::runtime_error);
  BOOST_CHECK_EQUAL(calls, 2);
}
BOOST_AUTO_TEST_CASE(plan_equals_plan_launch) {
  kernel_node kn;
  auto *out = kn.out_holder.get();
  kn.set_builder([out](kernel_handle &kh) {
    auto d = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1>) { return 1; });
    *out = kh.evaluate(d);
  });
  tree_store store;
  kernel_preparer prep(store);
  const auto pk = prep.prepare(*kn.n, kn.bindings(), lim());
  std::vector<std::size_t> sizes;
  for (auto &c : pk.tree->closures)
    sizes.push_back(c.bytes.size());
  const auto p = plan_launch(pk.tree->durable, pk.tree->defs, pk.tree->order, sizes, lim());
  BOOST_CHECK_EQUAL(p.grid, pk.plan.grid);
  BOOST_CHECK_EQUAL(p.bound_block_bytes, pk.plan.bound_block_bytes);
  BOOST_CHECK_EQUAL(p.toolchain_bytes, pk.plan.toolchain_bytes);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(launcher_suite)
namespace {
struct launched {
  kernel_node kn;
  prepared_kernel pk;
  tree_store store;
  std::vector<char> block;
  launched(std::size_t count = 8) : kn(count) {
    auto *out = kn.out_holder.get();
    kn.set_builder([out](kernel_handle &kh) {
      auto d = kh.parallel_for(sycl::range<1>(8), sycl::range<1>(4), [](sg::item<1>) { return 1; });
      *out = kh.evaluate(d);
    });
    kernel_preparer prep(store);
    pk = prep.prepare(*kn.n, kn.bindings(), lim());
    block.resize(pk.plan.toolchain_bytes + 64);
  }
};
} // namespace
BOOST_AUTO_TEST_CASE(one_launch_with_upload_and_fill) {
  launched l;
  sgt::mock_ops ops;
  launcher lc(ops);
  placement tc;
  tc.mem.ptr = l.block.data();
  const op_id k = lc.launch(0, l.pk, l.kn.bindings(), {0x1000}, tc, {42});
  BOOST_REQUIRE_EQUAL(ops.launches.size(), 1u);
  BOOST_REQUIRE_EQUAL(ops.copies.size(), 1u);
  BOOST_REQUIRE_EQUAL(ops.fills.size(), 1u);
  BOOST_CHECK_EQUAL(ops.copies[0].bytes, l.pk.plan.bound_block_bytes);
  BOOST_CHECK((ops.copies[0].deps == std::vector<op_id>{42}));
  BOOST_CHECK((ops.fills[0].deps == std::vector<op_id>{42}));
  BOOST_CHECK_EQUAL(ops.fills[0].count, l.pk.plan.flag_words);
  BOOST_CHECK_EQUAL(ops.launches[0].deps.size(), 2u);
  BOOST_CHECK_EQUAL(ops.launches[0].desc.groups, l.pk.plan.grid);
  BOOST_CHECK_EQUAL(ops.launches[0].desc.work_group_size, l.pk.plan.work_group_size);
  BOOST_CHECK_EQUAL(ops.launches[0].desc.local_bytes, l.pk.plan.local_bytes);
  BOOST_CHECK(ops.launches[0].desc.config == l.pk.tree->config.get());
  BOOST_CHECK_EQUAL(k, ops.next);
  lc.release_finished();
}
BOOST_AUTO_TEST_CASE(grid_zero_is_a_host_task) {
  kernel_node kn(0);
  kn.set_builder([out = kn.out_holder.get()](kernel_handle &kh) {
    auto d = kh.parallel_for(sycl::range<1>(0), sycl::range<1>(4), [](sg::item<1>) { return 1; });
    *out = kh.evaluate(d);
  });
  tree_store store;
  kernel_preparer prep(store);
  auto pk = prep.prepare(*kn.n, kn.bindings(0), lim());
  sgt::mock_ops ops;
  launcher lc(ops);
  placement tc;
  lc.launch(0, pk, kn.bindings(0), {0x1000}, tc, {5});
  BOOST_CHECK_EQUAL(ops.launches.size(), 0u);
  BOOST_CHECK_EQUAL(ops.hosts.size(), 1u);
  BOOST_CHECK((ops.hosts[0].deps == std::vector<op_id>{5}));
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(graph_impl_suite)
namespace {
struct submitter {
  sgt::rig r;
  sycl::device dev;
  void host(std::function<void(node_record &)> arrays, std::function<void()> fn = [] {}) {
    node_record &n = r.impl->begin_scope(dev);
    arrays(n);
    n.kind = node_record::kind_t::host_work;
    n.host_fn = fn;
    ++n.defines;
    r.impl->end_scope();
  }
  mem_handle<int> handle() {
    return handle_access::make<mem_handle<int>>(r.impl.get(), r.impl->add_handle(type_id<int>(), 4));
  }
};
} // namespace
BOOST_AUTO_TEST_CASE(submit_calls_nothing) {
  submitter s;
  auto h = s.handle();
  s.host([&](node_record &) { array<result<int>, 1> a(h, shape<1>(sycl::range<1>(4))); });
  BOOST_CHECK(s.r.mem->log.empty());
  BOOST_CHECK(s.r.ops->log.empty());
  BOOST_CHECK_EQUAL(s.r.prep->calls, 0);
}
BOOST_AUTO_TEST_CASE(scope_misuse) {
  submitter s;
  s.r.impl->begin_scope(s.dev);
  SG_CHECK_ERRC(s.r.impl->begin_scope(s.dev), builder_misuse);
  SG_CHECK_ERRC(s.r.impl->end_scope(), builder_misuse); // no define
  submitter t;
  node_record &n = t.r.impl->begin_scope(t.dev);
  n.defines = 2;
  SG_CHECK_ERRC(t.r.impl->end_scope(), builder_misuse);
}
BOOST_AUTO_TEST_CASE(run_in_order_and_clear) {
  submitter s;
  auto h = s.handle();
  int order = 0, a = 0, b = 0;
  s.host([&](node_record &) { array<result<int>, 1> x(h, shape<1>(sycl::range<1>(4))); }, [&] { a = ++order; });
  s.host([&](node_record &) { array<int, 1> x(h, shape<1>(sycl::range<1>(4))); }, [&] { b = ++order; });
  s.r.impl->run();
  BOOST_CHECK_EQUAL(a, 1);
  BOOST_CHECK_EQUAL(b, 2);
  BOOST_CHECK_EQUAL(s.r.ops->hosts.size(), 2u);
  BOOST_CHECK((s.r.ops->hosts[1].deps == std::vector<op_id>{1}));
  s.r.impl->run();
  BOOST_CHECK_EQUAL(s.r.ops->hosts.size(), 2u);
}
BOOST_AUTO_TEST_CASE(check_failure_after_earlier_nodes) {
  submitter s;
  auto h = s.handle(), g = s.handle();
  s.host([&](node_record &) { array<result<int>, 1> x(h, shape<1>(sycl::range<1>(4))); });
  s.host([&](node_record &) { array<int, 1> x(g, shape<1>(sycl::range<1>(4))); });
  SG_CHECK_ERRC(s.r.impl->run(), never_produced);
  BOOST_CHECK_EQUAL(s.r.ops->hosts.size(), 1u);
  s.r.impl->run();
  BOOST_CHECK_EQUAL(s.r.ops->hosts.size(), 1u);
}
BOOST_AUTO_TEST_CASE(memory_calls_and_commit_order) {
  submitter s;
  auto h = s.handle();
  s.host([&](node_record &) { array<result<int>, 1> x(h, shape<1>(sycl::range<1>(4))); });
  s.host([&](node_record &) { array<int, 1> x(h, shape<1>(sycl::range<1>(4))); });
  s.r.impl->run();
  const auto &log = s.r.mem->log;
  BOOST_REQUIRE(log.size() >= 5);
  BOOST_CHECK_EQUAL(log[0], "for_producer 0");
  const auto use = std::find(log.begin(), log.end(), "note_use 0 1");
  const auto prod = std::find(log.begin(), log.end(), "note_produced 0 1");
  BOOST_CHECK(use != log.end() && prod != log.end() && use < prod);
  BOOST_CHECK(std::find(log.begin(), log.end(), "for_reader 0") != log.end());
}
BOOST_AUTO_TEST_CASE(migration_copies_then_commits) {
  submitter s;
  auto h = s.handle();
  s.host([&](node_record &) { array<result<int>, 1> x(h, shape<1>(sycl::range<1>(4))); });
  s.host([&](node_record &) { array<int, 1> x(h, shape<1>(sycl::range<1>(4))); });
  migration m;
  m.plan.kind = copy_kind::direct;
  m.plan.steps.push_back({place{0, 1}, place{1, 1}, interval{0, 16}, 1});
  m.from.ptr = s.r.mem->arena.data();
  m.to.ptr = s.r.mem->arena.data() + 8192;
  s.r.mem->forced_migration[0] = m;
  s.r.impl->run();
  BOOST_REQUIRE_EQUAL(s.r.ops->copies.size(), 1u);
  BOOST_CHECK_EQUAL(s.r.ops->copies[0].dev, 1);
  const auto &log = s.r.mem->log;
  BOOST_CHECK(std::find(log.begin(), log.end(), "commit_migration 0") != log.end());
}
BOOST_AUTO_TEST_CASE(fill_for_inplace_value) {
  submitter s;
  auto h = s.handle();
  s.host([&](node_record &) {
    array<result<int>, 1> x(inplace(h, sycl::plus<>{}, 3), shape<1>(sycl::range<1>(1)));
  });
  s.r.impl->run();
  BOOST_REQUIRE_EQUAL(s.r.ops->fills.size(), 1u);
  BOOST_CHECK_EQUAL(s.r.ops->fills[0].count, 1u);
  BOOST_CHECK_EQUAL(s.r.ops->fills[0].pattern.size(), 4u);
}
BOOST_AUTO_TEST_CASE(kernel_node_flow) {
  submitter s;
  auto h = s.handle();
  node_record &n = s.r.impl->begin_scope(s.dev);
  array<result<int>, 1> x(h, shape<1>(sycl::range<1>(4)));
  n.kind = node_record::kind_t::kernel;
  n.builder = [](kernel_handle &) {};
  ++n.defines;
  s.r.impl->end_scope();
  s.r.impl->run();
  BOOST_CHECK_EQUAL(s.r.prep->calls, 1);
  BOOST_CHECK_EQUAL(s.r.launch->deps.size(), 1u);
  const auto &log = s.r.mem->log;
  const auto acq = std::find(log.begin(), log.end(), "acquire_toolchain");
  const auto rel = std::find_if(log.begin(), log.end(), [](const std::string &l) { return l.rfind("release_toolchain", 0) == 0; });
  BOOST_CHECK(acq != log.end() && rel != log.end() && acq < rel);
  BOOST_CHECK_EQUAL(s.r.prep->seen[0][0].extent[0], 4u);
}
BOOST_AUTO_TEST_CASE(futures) {
  submitter s;
  auto h = s.handle();
  graph_promise<int> prom;
  auto st = std::make_shared<promise_state<int>>();
  const auto pi = s.r.impl->add_promise(st, [st] { return st->is_set(); });
  s.host([&](node_record &) { array<result<int>, 1> x(h, shape<1>(sycl::range<1>(6))); }, [st] { st->set(6); });
  s.host(
      [&](node_record &n) {
        n.futures_read.push_back(pi);
        array<int, 1> x(h, shape<1>({dim(launch_value<int>(st, 0))}));
      },
      [] {});
  s.r.impl->run();
  BOOST_CHECK(st->is_set());
  BOOST_CHECK_EQUAL(s.r.ops->hosts.size(), 2u);
  // never set
  submitter t;
  auto st2 = std::make_shared<promise_state<int>>();
  const auto pj = t.r.impl->add_promise(st2, [st2] { return st2->is_set(); });
  t.host([&](node_record &n) { n.futures_read.push_back(pj); });
  SG_CHECK_ERRC(t.r.impl->run(), future_unset);
}
BOOST_AUTO_TEST_CASE(host_table_filled_before_host_task) {
  submitter s;
  auto h = s.handle();
  bool seen = false;
  std::shared_ptr<host_array<int, 1>> ha;
  s.host(
      [&](node_record &) { ha = std::make_shared<host_array<int, 1>>(h, shape<1>(sycl::range<1>(4))); },
      [&] { seen = ha->data() != nullptr && ha->range()[0] == 4; });
  s.r.impl->run();
  BOOST_CHECK(seen);
}
BOOST_AUTO_TEST_CASE(wait_calls) {
  submitter s;
  s.r.impl->wait();
  BOOST_CHECK_EQUAL(s.r.ops->wait_all_calls, 1);
  BOOST_CHECK_EQUAL(s.r.launch->released, 1);
  BOOST_CHECK_EQUAL(s.r.mem->released, 1);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(composition_adaptivity_suite)
namespace {
hipsycl::glue::sscp::composition_config cfg(std::uint64_t hash) {
  hipsycl::glue::sscp::composition_config c;
  c.unique_hash = hash;
  return c;
}
hipsycl::rt::kernel_configuration make(const hipsycl::glue::sscp::composition_config *c, int index = 0) {
  hipsycl::rt::kernel_configuration k;
  if (c)
    hipsycl::rt::configure_composition_kernel(k, index, {c}, "kern", 1024);
  return k;
}
} // namespace
BOOST_AUTO_TEST_CASE(ids) {
  const auto a = cfg(1), a2 = cfg(1), b = cfg(2);
  BOOST_CHECK(make(&a).generate_id() == make(&a2).generate_id());
  BOOST_CHECK(make(&a).generate_id() != make(&b).generate_id());
  BOOST_CHECK(make(&a, 0).generate_id() != make(&a, 1).generate_id());
  BOOST_CHECK(make(&a).generate_id() != make(nullptr).generate_id());
  BOOST_CHECK(make(nullptr).generate_id() == hipsycl::rt::kernel_configuration{}.generate_id());
}
BOOST_AUTO_TEST_CASE(sets_only_composition_state) {
  const auto a = cfg(7);
  auto k = make(&a);
  BOOST_REQUIRE_EQUAL(k.get_composition_configs().size(), 1u);
  BOOST_CHECK_EQUAL(k.get_composition_configs()[0].value->unique_hash, 7u);
}
BOOST_AUTO_TEST_SUITE_END()
