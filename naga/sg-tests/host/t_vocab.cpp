#include "../common/check.hpp"
#include "../common/mocks.hpp"

#include "hipSYCL/sg/body.hpp"
#include "hipSYCL/sg/detail/tree_store.hpp"
#include "hipSYCL/sg/group_expr.hpp"
#include "hipSYCL/sg/kernel.hpp"
#include "hipSYCL/sg/mapping_forms.hpp"
#include "hipSYCL/sg/roles.hpp"

#include <thread>

using namespace sg;
using namespace sg::detail;

namespace {
struct scope_fixture {
  sgt::rig r;
  node_record *n;
  scope_fixture() { n = &r.impl->begin_scope(sycl::device{}); }
  ~scope_fixture() { r.impl->abort_scope(); }
  template <class T> mem_handle<T> handle() {
    return handle_access::make<mem_handle<T>>(r.impl.get(), r.impl->add_handle(type_id<T>(), sizeof(T)));
  }
};

using view3 = group_array<float, 3, mapping::local>;
using view2 = group_array<float, 2, mapping::local>;
using view1 = group_array<float, 1, mapping::local>;
template <class V> V view(std::uint64_t a, std::uint64_t b = 1, std::uint64_t c = 1) {
  dev::view_rep v{};
  v.extent[0] = a;
  v.extent[1] = b;
  v.extent[2] = c;
  v.stride[0] = b * c;
  v.stride[1] = c;
  v.stride[2] = 1;
  return V(v);
}
} // namespace

BOOST_AUTO_TEST_SUITE(roles_suite)
BOOST_AUTO_TEST_CASE(slots_in_order) {
  scope_fixture f;
  auto h = f.handle<float>();
  array<float, 2> in(h, shape<2>({4, 6}));
  array<result<float>, 1> out(h, shape<1>(sycl::range<1>(3)));
  BOOST_REQUIRE_EQUAL(f.n->slots.size(), 2u);
  BOOST_CHECK(f.n->slots[0].mode == role_mode::read);
  BOOST_CHECK(f.n->slots[1].mode == role_mode::produce);
  BOOST_CHECK_EQUAL(f.n->slots[0].rank, 2);
  BOOST_CHECK_EQUAL(f.n->slots[0].extent[0], 4u);
  BOOST_CHECK_EQUAL(f.n->slots[0].extent[1], 6u);
  BOOST_CHECK_EQUAL(f.n->slots[0].extent[2], 1u);
  BOOST_CHECK_EQUAL(f.n->slots[0].elem_bytes, 4u);
  std::uint64_t first;
  std::memcpy(&first, &in, 8);
  BOOST_CHECK(is_sentinel(first));
  BOOST_CHECK_EQUAL(sentinel_slot(first), 0u);
  BOOST_CHECK_EQUAL(in.range()[1], 6u);
}
BOOST_AUTO_TEST_CASE(outside_scope) {
  sgt::rig r;
  auto h = handle_access::make<mem_handle<float>>(r.impl.get(), 0);
  SG_CHECK_ERRC((array<float, 1>(h, shape<1>(sycl::range<1>(2)))), not_in_submit_scope);
}
BOOST_AUTO_TEST_CASE(bad_handles) {
  scope_fixture f;
  mem_handle<float> null;
  SG_CHECK_ERRC((array<float, 1>(null, shape<1>(sycl::range<1>(2)))), builder_misuse);
  sgt::rig other;
  auto h = handle_access::make<mem_handle<float>>(other.impl.get(), 0);
  SG_CHECK_ERRC((array<float, 1>(h, shape<1>(sycl::range<1>(2)))), builder_misuse);
  SG_CHECK_ERRC(shape<2>({1, 2, 3}), builder_misuse);
}
BOOST_AUTO_TEST_CASE(slice_count) {
  scope_fixture f;
  auto h = f.handle<float>();
  SG_CHECK_ERRC((array<float, 1>(h.slice(2, 5), shape<1>(sycl::range<1>(4)))), size_mismatch);
  array<float, 1> ok(h.slice(2, 4), shape<1>(sycl::range<1>(4)));
  BOOST_CHECK(!f.n->slots.back().whole);
  BOOST_CHECK_EQUAL(f.n->slots.back().offset, 2u);
}
BOOST_AUTO_TEST_CASE(deferred_dims) {
  scope_fixture f;
  auto h = f.handle<float>();
  auto p = handle_access::make<mem_handle<float>>(f.r.impl.get(), 0);
  (void)p;
  auto st = std::make_shared<promise_state<int>>();
  const auto pi = f.r.impl->add_promise(st, [st] { return st->is_set(); });
  graph_promise<int> prom;
  (void)prom;
  f.n->futures_read.push_back(pi);
  launch_value<int> lv(st, 0);
  array<float, 1> a(h, shape<1>({dim(lv)}));
  BOOST_CHECK(f.n->slots[0].extent[0] & dim::deferred_flag);
  SG_CHECK_ERRC(a.range(), builder_misuse);
  prepare_context ctx;
  ctx.resolve_dim = [](std::uint64_t) { return std::uint64_t{9}; };
  prepare_context_scope pcs(&ctx);
  BOOST_CHECK_EQUAL(a.range()[0], 9u);
}
BOOST_AUTO_TEST_CASE(sentinel_read_and_patched_read) {
  scope_fixture f;
  auto h = f.handle<int>();
  array<int, 2> a(h, shape<2>({2, 3}));
  SG_CHECK_ERRC(a[sycl::id<2>(1, 1)], builder_misuse);
  {
    probe_scope ps;
    BOOST_CHECK_EQUAL(a[sycl::id<2>(1, 1)], 0);
  }
  int data[6] = {0, 1, 2, 3, 4, 5};
  closure_entry e = capture_closure(a);
  slot_binding b;
  b.address = reinterpret_cast<std::uint64_t>(data);
  b.extent[0] = 2;
  b.extent[1] = 3;
  const auto bytes = patch_closure(e, {b});
  alignas(array<int, 2>) unsigned char raw[sizeof(array<int, 2>)];
  std::memcpy(raw, bytes.data(), sizeof(raw));
  const auto &patched = *reinterpret_cast<const array<int, 2> *>(raw);
  BOOST_CHECK_EQUAL(patched[sycl::id<2>(1, 2)], 5);
  BOOST_CHECK_EQUAL(patched[sycl::id<2>(1, 0)], 3);
}
BOOST_AUTO_TEST_CASE(inplace_slot) {
  scope_fixture f;
  auto h = f.handle<int>();
  array<result<int>, 1> a(inplace(h, sycl::plus<>{}, 5), shape<1>(sycl::range<1>(1)));
  const array_slot &s = f.n->slots[0];
  BOOST_CHECK(s.mode == role_mode::inplace);
  BOOST_CHECK(s.init == init_kind::value);
  BOOST_CHECK_EQUAL(s.op, std::string(type_name<sycl::plus<>>()));
  BOOST_CHECK_EQUAL(s.init_bits.size(), 4u);
  BOOST_CHECK(!s.op_symbol.empty());
  array<result<int>, 1> b(inplace(h, sycl::plus<>{}, reuse), shape<1>(sycl::range<1>(1)));
  BOOST_CHECK(f.n->slots[1].init == init_kind::reuse);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(future_suite)
BOOST_AUTO_TEST_CASE(get_outside_scope) {
  graph_future<int> f;
  SG_CHECK_ERRC(f.get(), not_in_submit_scope);
}
BOOST_AUTO_TEST_CASE(record_set_read) {
  sgt::rig r;
  auto st = std::make_shared<promise_state<int>>();
  const auto pi = r.impl->add_promise(st, [st] { return st->is_set(); });
  node_record &n = r.impl->begin_scope(sycl::device{});
  launch_value<int> lv(st, 0);
  n.futures_read.push_back(pi);
  BOOST_CHECK_EQUAL(n.futures_read[0], pi);
  SG_CHECK_ERRC((void)static_cast<int>(lv), future_unset);
  BOOST_CHECK(st->set(41));
  BOOST_CHECK_EQUAL(static_cast<int>(lv), 41);
  BOOST_CHECK(!st->set(42));
  r.impl->abort_scope();
}
BOOST_AUTO_TEST_CASE(cross_thread_visibility) {
  auto st = std::make_shared<promise_state<int>>();
  std::thread t([st] { st->set(7); });
  t.join();
  BOOST_CHECK(st->is_set());
  BOOST_CHECK_EQUAL(st->value(), 7);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(forms_suite)
BOOST_AUTO_TEST_CASE(single_block_strided) {
  single<2> s{sycl::id<2>(1, 2)};
  BOOST_CHECK_EQUAL(s.size(), 1u);
  BOOST_CHECK_EQUAL(s[0][1], 2u);
  sg::block<2> b{sycl::id<2>(1, 1), sycl::range<2>(2, 3)};
  BOOST_CHECK_EQUAL(b.size(), 6u);
  BOOST_CHECK_EQUAL(b[4][0], 2u);
  BOOST_CHECK_EQUAL(b[4][1], 2u);
  strided<1> st{sycl::id<1>(1), sycl::range<1>(3), sycl::range<1>(4)};
  BOOST_CHECK_EQUAL(st.size(), 3u);
  BOOST_CHECK_EQUAL(st[2][0], 9u);
}
BOOST_AUTO_TEST_CASE(stencil_ids) {
  stencil<1, 3> s{sycl::id<1>(5), {sycl::id<1>(-1 + 1), sycl::id<1>(1), sycl::id<1>(2)}};
  BOOST_CHECK_EQUAL(s.size(), 3u);
  BOOST_CHECK_EQUAL(s[2][0], 7u);
  ids<2, 2> i{{sycl::id<2>(3, 4), sycl::id<2>(5, 6)}};
  BOOST_CHECK_EQUAL(i.size(), 2u);
  BOOST_CHECK_EQUAL(i[1][1], 6u);
  BOOST_CHECK(decltype(s)::kind == form_kind::stencil);
  BOOST_CHECK(decltype(i)::kind == form_kind::ids);
}
BOOST_AUTO_TEST_CASE(index_map_sizes_without_reading) {
  scope_fixture f;
  auto h = f.handle<int>();
  array<int, 1> src(h, shape<1>(sycl::range<1>(100)));
  index_map<1, int> m{src, 10, 3};
  probe_scope ps;
  BOOST_CHECK_EQUAL(m.size(), 3u);
  BOOST_CHECK(decltype(m)::kind == form_kind::index_map);
}
BOOST_AUTO_TEST_CASE(index_map_reads_after_patch) {
  scope_fixture f;
  auto h = f.handle<int>();
  array<int, 1> src(h, shape<1>(sycl::range<1>(8)));
  int data[8] = {0, 10, 20, 30, 40, 50, 60, 70};
  index_map<1, int> m{src, 2, 3};
  closure_entry e = capture_closure(m);
  slot_binding b;
  b.address = reinterpret_cast<std::uint64_t>(data);
  b.extent[0] = 8;
  const auto bytes = patch_closure(e, {b});
  index_map<1, int> p{src, 0, 0};
  std::memcpy(&p, bytes.data(), sizeof(p));
  BOOST_CHECK_EQUAL(p.size(), 3u);
  BOOST_CHECK_EQUAL(p[1][0], 30u);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(probe_suite)
BOOST_AUTO_TEST_CASE(mode) {
  BOOST_CHECK(!probe_mode());
  {
    probe_scope a;
    BOOST_CHECK(probe_mode());
    bool other = true;
    std::thread t([&] { other = probe_mode(); });
    t.join();
    BOOST_CHECK(!other);
  }
  BOOST_CHECK(!probe_mode());
}
BOOST_AUTO_TEST_CASE(positions) {
  int calls = 0;
  struct cap { int *c; };
  auto fn = [&](sycl::id<2> id) {
    ++calls;
    BOOST_CHECK_EQUAL(id[0], 0u);
    return sg::block<2>{id, sycl::range<2>(2, 3)};
  };
  const auto pf = probe_positions<2>(fn);
  BOOST_CHECK(pf.kind == form_kind::block);
  BOOST_CHECK_EQUAL(pf.count, 6u);
  BOOST_CHECK_EQUAL(calls, 1);
  const auto p2 = probe_positions<1>([](sycl::id<1> i) { return single<1>{i}; });
  BOOST_CHECK(p2.kind == form_kind::single);
  BOOST_CHECK_EQUAL(p2.count, 1u);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(group_expr_suite)
BOOST_AUTO_TEST_CASE(reduce_shapes) {
  auto v = view<view3>(4, 5, 6);
  auto r0 = group_reduce<3>(v, 0, sycl::plus<>{});
  static_assert(decltype(r0)::rank == 2);
  BOOST_CHECK_EQUAL(r0.shape()[0], 5u);
  BOOST_CHECK_EQUAL(r0.shape()[1], 6u);
  auto r1 = group_reduce<3>(v, 1, sycl::plus<>{});
  BOOST_CHECK_EQUAL(r1.shape()[0], 4u);
  BOOST_CHECK_EQUAL(r1.shape()[1], 6u);
  auto r2 = group_reduce<3>(v, 2, sycl::plus<>{});
  BOOST_CHECK_EQUAL(r2.shape()[1], 5u);
  auto rr = group_reduce<2>(r1, 0, sycl::maximum<>{});
  static_assert(decltype(rr)::rank == 1);
  BOOST_CHECK_EQUAL(rr.shape()[0], 6u);
  auto one = group_reduce<1>(view<view1>(9), 0, sycl::plus<>{});
  static_assert(decltype(one)::rank == 0);
  BOOST_CHECK_EQUAL(one.shape()[0], 1u);
  auto scan = group_inclusive_scan<2>(view<view2>(3, 4), 1, sycl::plus<>{});
  static_assert(decltype(scan)::rank == 2);
  BOOST_CHECK_EQUAL(scan.shape()[1], 4u);
  auto both = group_inclusive_scan<1>(group_reduce<2>(view<view2>(3, 4), 0, sycl::plus<>{}), 0, sycl::plus<>{});
  BOOST_CHECK_EQUAL(both.shape()[0], 4u);
}
BOOST_AUTO_TEST_CASE(composition_of_three) {
  auto v = view<view3>(4, 5, 6);
  auto a = group_reduce<3>(v, 1, sycl::plus<>{});
  auto b = group_reduce<2>(a, 0, sycl::plus<>{});
  auto c = group_inclusive_scan(b, sycl::plus<>{});
  BOOST_CHECK_EQUAL(c.shape()[0], 6u);
  static_assert(decltype(c)::rank == 1);
  static_assert(std::is_same_v<typename decltype(c)::value_type, float>);
  const std::size_t sa = 4 * 6 * 4, sb = 6 * 4;
  BOOST_CHECK_EQUAL(a.scratch_bytes(), (sa + 15) / 16 * 16);
  BOOST_CHECK_EQUAL(b.scratch_bytes(), a.scratch_bytes() + (sb + 15) / 16 * 16);
  const std::size_t scan_own = (6 * 4 + 15) / 16 * 16 + 2 * ((6 * 4 + 15) / 16 * 16);
  BOOST_CHECK_EQUAL(c.scratch_bytes(), b.scratch_bytes() + scan_own);
}
BOOST_AUTO_TEST_CASE(elementwise) {
  auto a = view<view1>(8), b = view<view1>(8);
  auto s = a + b * 2.0f;
  BOOST_CHECK_EQUAL(s.shape()[0], 8u);
  BOOST_CHECK_EQUAL(s.scratch_bytes(), 0u);
  auto r = group_reduce<1>(a, 0, sycl::plus<>{});
  auto over = r + r;
  BOOST_CHECK_EQUAL(over.scratch_bytes(), 2 * r.scratch_bytes());
  auto diff = view<view1>(7);
  SG_CHECK_ERRC(((a + diff).shape()), shape_mismatch);
  SG_CHECK_ERRC(group_reduce<1>(a, 3, sycl::plus<>{}).shape(), shape_mismatch);
  SG_CHECK_ERRC(group_reduce<1>(a, -1, sycl::plus<>{}).shape(), shape_mismatch);
}
BOOST_AUTO_TEST_SUITE_END()
