#include "../common/check.hpp"

#include "hipSYCL/sg/detail/copy_plan.hpp"
#include "hipSYCL/sg/detail/dependency.hpp"
#include "hipSYCL/sg/detail/memory_manager.hpp"
#include "hipSYCL/sg/detail/pool.hpp"
#include "hipSYCL/sg/detail/pool_config.hpp"
#include "hipSYCL/sg/detail/role_check.hpp"
#include "hipSYCL/sg/detail/slab_bitmap.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <random>

using namespace sg::detail;

BOOST_AUTO_TEST_SUITE(slab_bitmap_suite)
BOOST_AUTO_TEST_CASE(all_free) {
  for (std::size_t n : {1u, 63u, 64u, 65u, 128u, 1000u}) {
    slab_bitmap b(n);
    BOOST_CHECK_EQUAL(b.slot_count(), n);
    BOOST_CHECK_EQUAL(b.free_count(), n);
    BOOST_CHECK_EQUAL(b.largest_free_run(), n);
  }
}
BOOST_AUTO_TEST_CASE(ascending_and_reuse) {
  slab_bitmap b(10);
  for (std::size_t i = 0; i < 10; ++i)
    BOOST_CHECK_EQUAL(b.allocate(1), i);
  BOOST_CHECK_EQUAL(b.allocate(1), slab_bitmap::npos);
  b.release(3, 1);
  b.release(7, 1);
  BOOST_CHECK_EQUAL(b.allocate(1), 3u);
  BOOST_CHECK_EQUAL(b.allocate(1), 7u);
}
BOOST_AUTO_TEST_CASE(exhaustion_unchanged) {
  slab_bitmap b(70);
  BOOST_CHECK_EQUAL(b.allocate(71), slab_bitmap::npos);
  BOOST_CHECK_EQUAL(b.free_count(), 70u);
  BOOST_CHECK_EQUAL(b.allocate(0), slab_bitmap::npos);
  BOOST_CHECK_EQUAL(b.allocate(70), 0u);
  BOOST_CHECK_EQUAL(b.allocate(1), slab_bitmap::npos);
}
BOOST_AUTO_TEST_CASE(phantom_bits) {
  slab_bitmap b(65);
  BOOST_CHECK_EQUAL(b.allocate(65), 0u);
  BOOST_CHECK_EQUAL(b.allocate(1), slab_bitmap::npos);
}
BOOST_AUTO_TEST_CASE(runs) {
  slab_bitmap b(300);
  BOOST_CHECK_EQUAL(b.allocate(5), 0u);
  BOOST_CHECK_EQUAL(b.allocate(62), 5u);   // crosses into the second word
  BOOST_CHECK_EQUAL(b.allocate(130), 67u); // over 64
  BOOST_CHECK_EQUAL(b.free_count(), 300u - 197u);
}
BOOST_AUTO_TEST_CASE(fragmentation_and_coalescing) {
  slab_bitmap b(8);
  for (int i = 0; i < 8; ++i)
    b.allocate(1);
  b.release(1, 1);
  b.release(3, 1);
  b.release(5, 1);
  BOOST_CHECK_EQUAL(b.allocate(2), slab_bitmap::npos);
  BOOST_CHECK_EQUAL(b.largest_free_run(), 1u);
  b.release(2, 1);
  BOOST_CHECK_EQUAL(b.largest_free_run(), 3u);
  BOOST_CHECK_EQUAL(b.allocate(3), 1u);
}
BOOST_AUTO_TEST_CASE(bad_release) {
  slab_bitmap b(8);
  SG_CHECK_ERRC(b.release(0, 1), internal);
  b.allocate(2);
  SG_CHECK_ERRC(b.release(7, 4), internal);
  b.release(0, 2);
  SG_CHECK_ERRC(b.release(0, 2), internal);
}
BOOST_AUTO_TEST_CASE(random_against_model) {
  const std::size_t N = 200;
  slab_bitmap b(N);
  std::vector<bool> used(N, false);
  std::vector<std::pair<std::size_t, std::size_t>> live;
  std::mt19937 rng(12345);
  for (int step = 0; step < 4000; ++step) {
    if (live.empty() || rng() % 2) {
      const std::size_t n = 1 + rng() % 70;
      std::size_t want = slab_bitmap::npos;
      for (std::size_t s = 0; s + n <= N && want == slab_bitmap::npos; ++s) {
        bool ok = true;
        for (std::size_t i = 0; i < n && ok; ++i)
          ok = !used[s + i];
        if (ok)
          want = s;
      }
      const std::size_t got = b.allocate(n);
      BOOST_REQUIRE_EQUAL(got, want);
      if (got != slab_bitmap::npos) {
        for (std::size_t i = 0; i < n; ++i)
          used[got + i] = true;
        live.push_back({got, n});
      }
    } else {
      const std::size_t k = rng() % live.size();
      const auto r = live[k];
      live.erase(live.begin() + k);
      b.release(r.first, r.second);
      for (std::size_t i = 0; i < r.second; ++i)
        used[r.first + i] = false;
    }
  }
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(pool_config_suite)
BOOST_AUTO_TEST_CASE(rule) {
  BOOST_CHECK_EQUAL(pool_limit_rule(1000, 5000, 77), 77u);
  BOOST_CHECK_EQUAL(pool_limit_rule(1000, 5000, 0), 800u);
  BOOST_CHECK_EQUAL(pool_limit_rule(std::nullopt, 5000, 0), 4000u);
  BOOST_CHECK_EQUAL(pool_limit_rule(0, 5000, 0), 0u);
  const std::size_t big = std::numeric_limits<std::size_t>::max();
  const std::size_t r = pool_limit_rule(std::nullopt, big, 0);
  BOOST_CHECK(r < big && r > big / 10 * 7);
}
BOOST_AUTO_TEST_SUITE_END()

namespace {
struct arena {
  std::vector<std::unique_ptr<char[]>> blocks;
  std::vector<std::size_t> allocs;
  std::vector<void *> frees;
  std::function<void *(std::size_t)> alloc() {
    return [this](std::size_t n) {
      blocks.emplace_back(new char[n]);
      allocs.push_back(n);
      return static_cast<void *>(blocks.back().get());
    };
  }
  std::function<void(void *)> dealloc() {
    return [this](void *p) { frees.push_back(p); };
  }
};
} // namespace

BOOST_AUTO_TEST_SUITE(pool_suite)
BOOST_AUTO_TEST_CASE(eager_slabs) {
  arena a;
  {
    pool p(10 * 1024 + 500, 4 * 1024 + 7, a.alloc(), a.dealloc());
    BOOST_REQUIRE_EQUAL(a.allocs.size(), 3u);
    BOOST_CHECK_EQUAL(a.allocs[0], 4096u);
    BOOST_CHECK_EQUAL(a.allocs[1], 4096u);
    BOOST_CHECK_EQUAL(a.allocs[2], 2048u);
    BOOST_CHECK_EQUAL(p.reserved(), 10240u);
  }
  BOOST_CHECK_EQUAL(a.frees.size(), 3u);
}
BOOST_AUTO_TEST_CASE(allocate_basics) {
  arena a;
  pool p(8 * 1024, 4 * 1024, a.alloc(), a.dealloc());
  BOOST_CHECK(p.allocate(0).ptr == nullptr);
  pool_block x = p.allocate(1);
  pool_block y = p.allocate(1025);
  BOOST_CHECK_EQUAL(x.nodes, 1u);
  BOOST_CHECK_EQUAL(y.nodes, 2u);
  BOOST_CHECK(static_cast<char *>(y.ptr) == static_cast<char *>(x.ptr) + 1024);
  BOOST_CHECK_EQUAL(p.used(), 3 * 1024u);
  pool_block z = p.allocate(4096); // does not fit slab 0 any more, goes to slab 1
  BOOST_CHECK_EQUAL(z.slab, 1u);
}
BOOST_AUTO_TEST_CASE(exhausted_messages) {
  arena a;
  pool p(4 * 1024, 4 * 1024, a.alloc(), a.dealloc());
  pool_block x = p.allocate(1024);
  p.allocate(1024);
  pool_block z = p.allocate(1024);
  p.allocate(1024);
  p.release(x, {});
  p.release(z, {});
  try {
    p.allocate(2048);
    BOOST_FAIL("no throw");
  } catch (const sg::error &e) {
    BOOST_CHECK(e.code() == sg::errc::pool_exhausted);
    BOOST_CHECK(std::string(e.what()).find("fragmented") != std::string::npos);
    BOOST_CHECK(std::string(e.what()).find("2048") != std::string::npos);
  }
  try {
    p.allocate(8192);
    BOOST_FAIL("no throw");
  } catch (const sg::error &e) {
    BOOST_CHECK(std::string(e.what()).find("full") != std::string::npos);
  }
}
BOOST_AUTO_TEST_CASE(reuse_and_waits) {
  arena a;
  pool p(4 * 1024, 4 * 1024, a.alloc(), a.dealloc());
  pool_block x = p.allocate(2048);
  void *ptr = x.ptr;
  p.release(x, {5, 6});
  pool_block y = p.allocate(2048);
  BOOST_CHECK(y.ptr == ptr);
  BOOST_CHECK((y.wait_for == std::vector<op_id>{5, 6}));
  p.release(y, {9});
  pool_block z = p.allocate(1024);
  BOOST_CHECK((z.wait_for == std::vector<op_id>{9}));
  pool_block w = p.allocate(1024);
  BOOST_CHECK((w.wait_for == std::vector<op_id>{9}));
  pool_block v = p.allocate(1024);
  BOOST_CHECK(v.wait_for.empty());
}
BOOST_AUTO_TEST_CASE(double_release_throws_and_determinism) {
  arena a;
  pool p(4 * 1024, 4 * 1024, a.alloc(), a.dealloc());
  pool_block x = p.allocate(1024);
  p.release(x, {});
  SG_CHECK_ERRC(p.release(x, {}), internal);
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(dependency_suite)
namespace {
sg::detail::access acc(std::size_t h, std::size_t b, std::size_t e, access_mode m, int g = -1) {
  return sg::detail::access{h, {b, e}, m, g};
}
} // namespace
BOOST_AUTO_TEST_CASE(raw_war_waw) {
  dependency_tracker t;
  t.commit(1, {acc(0, 0, 10, access_mode::write)});
  BOOST_CHECK((t.resolve({acc(0, 0, 10, access_mode::read)}) == std::vector<op_id>{1}));
  t.commit(2, {acc(0, 0, 10, access_mode::read)});
  t.commit(3, {acc(0, 0, 10, access_mode::read)});
  BOOST_CHECK((t.resolve({acc(0, 0, 10, access_mode::write)}) == std::vector<op_id>{1, 2, 3}));
  BOOST_CHECK((t.resolve({acc(0, 0, 10, access_mode::read)}) == std::vector<op_id>{1}));
}
BOOST_AUTO_TEST_CASE(intervals) {
  dependency_tracker t;
  t.commit(1, {acc(0, 0, 10, access_mode::write)});
  BOOST_CHECK(t.resolve({acc(0, 10, 20, access_mode::write)}).empty());
  BOOST_CHECK(!t.resolve({acc(0, 9, 20, access_mode::write)}).empty());
  BOOST_CHECK(t.resolve({acc(0, 5, 5, access_mode::write)}).empty());
  BOOST_CHECK(t.resolve({acc(1, 0, 10, access_mode::write)}).empty());
}
BOOST_AUTO_TEST_CASE(inplace) {
  dependency_tracker t;
  t.commit(1, {acc(0, 0, 8, access_mode::write)});
  t.commit(2, {acc(0, 0, 8, access_mode::inplace, 4)});
  BOOST_CHECK((t.resolve({acc(0, 0, 8, access_mode::inplace, 4)}) == std::vector<op_id>{1}));
  BOOST_CHECK((t.resolve({acc(0, 0, 8, access_mode::inplace, 5)}) == std::vector<op_id>{1, 2}));
  BOOST_CHECK((t.resolve({acc(0, 0, 8, access_mode::read)}) == std::vector<op_id>{1, 2}));
  BOOST_CHECK((t.resolve({acc(0, 0, 8, access_mode::write)}) == std::vector<op_id>{1, 2}));
}
BOOST_AUTO_TEST_CASE(cover_replaces_and_resolve_records_nothing) {
  dependency_tracker t;
  t.commit(1, {acc(0, 2, 4, access_mode::read)});
  t.commit(2, {acc(0, 0, 10, access_mode::write)});
  (void)t.resolve({acc(0, 0, 10, access_mode::write)});
  BOOST_CHECK((t.resolve({acc(0, 0, 10, access_mode::read)}) == std::vector<op_id>{2}));
  BOOST_CHECK((t.resolve({acc(0, 0, 10, access_mode::write)}) == std::vector<op_id>{2}));
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(copy_plan_suite)
BOOST_AUTO_TEST_CASE(kinds) {
  const place host{-1, 0}, d0{0, 1}, d1{1, 1}, d2{2, 2};
  BOOST_CHECK(plan_copy(d0, d0, {0, 8}).kind == copy_kind::none);
  BOOST_CHECK(plan_copy(d0, host, {4, 4}).kind == copy_kind::none);
  auto a = plan_copy(host, d0, {0, 8});
  BOOST_CHECK(a.kind == copy_kind::direct);
  BOOST_CHECK_EQUAL(a.steps.size(), 1u);
  BOOST_CHECK_EQUAL(a.steps[0].exec_device, 0);
  auto b = plan_copy(d0, d1, {0, 8});
  BOOST_CHECK(b.kind == copy_kind::direct);
  auto c = plan_copy(d0, d2, {8, 24});
  BOOST_REQUIRE(c.kind == copy_kind::staged_via_host);
  BOOST_REQUIRE_EQUAL(c.steps.size(), 2u);
  BOOST_CHECK_EQUAL(c.steps[0].exec_device, 0);
  BOOST_CHECK_EQUAL(c.steps[1].exec_device, 2);
  BOOST_CHECK_EQUAL(c.steps[0].bytes.begin, 8u);
  BOOST_CHECK_EQUAL(c.steps[1].bytes.end, 24u);
}
BOOST_AUTO_TEST_SUITE_END()

namespace {
array_slot slot(std::size_t h, role_mode m, std::uint64_t type, std::uint64_t n, int rank = 1) {
  array_slot s;
  s.handle = h;
  s.type = type;
  s.elem_bytes = 4;
  s.mode = m;
  s.rank = rank;
  s.extent[0] = n;
  return s;
}
} // namespace

BOOST_AUTO_TEST_SUITE(role_check_suite)
BOOST_AUTO_TEST_CASE(produce_then_read) {
  handle_table t;
  int g = 0;
  check_roles(t, {slot(0, role_mode::produce, 1, 8)}, 0, g);
  BOOST_CHECK(t[0].produced);
  check_roles(t, {slot(0, role_mode::read, 1, 8)}, 0, g);
}
BOOST_AUTO_TEST_CASE(read_rules) {
  handle_table t;
  int g = 0;
  SG_CHECK_ERRC(check_roles(t, {slot(0, role_mode::read, 1, 8)}, 0, g), never_produced);
  check_roles(t, {slot(0, role_mode::produce, 1, 8)}, 0, g);
  SG_CHECK_ERRC(check_roles(t, {slot(0, role_mode::read, 2, 8)}, 0, g), type_mismatch);
  SG_CHECK_ERRC(check_roles(t, {slot(0, role_mode::read, 1, 9)}, 0, g), size_mismatch);
  array_slot sl = slot(0, role_mode::read, 1, 4);
  sl.whole = false;
  sl.offset = 6;
  SG_CHECK_ERRC(check_roles(t, {sl}, 0, g), size_mismatch);
  sl.offset = 4;
  check_roles(t, {sl}, 0, g);
}
BOOST_AUTO_TEST_CASE(conflicts_in_node) {
  handle_table t;
  int g = 0;
  SG_CHECK_ERRC(check_roles(t, {slot(0, role_mode::produce, 1, 8), slot(0, role_mode::produce, 1, 8)},
                            0, g),
                role_conflict);
  BOOST_CHECK(t.empty() || !t[0].produced);
  check_roles(t, {slot(0, role_mode::produce, 1, 8)}, 0, g);
  check_roles(t, {slot(0, role_mode::read, 1, 8), slot(0, role_mode::read, 1, 8)}, 0, g);
  SG_CHECK_ERRC(check_roles(t, {slot(0, role_mode::read, 1, 8), slot(0, role_mode::produce, 1, 8)},
                            0, g),
                role_conflict);
}
BOOST_AUTO_TEST_CASE(inplace_groups) {
  handle_table t;
  int g = 0;
  array_slot a = slot(0, role_mode::inplace, 1, 4);
  a.op = "plus";
  a.init = init_kind::value;
  a.init_bits = {std::byte{0}};
  auto e1 = check_roles(t, {a}, 0, g);
  BOOST_CHECK(e1[0].new_group && e1[0].needs_fill);
  auto e2 = check_roles(t, {a}, 0, g);
  BOOST_CHECK(!e2[0].new_group && !e2[0].needs_fill);
  BOOST_CHECK_EQUAL(e1[0].group, e2[0].group);
  array_slot b = a;
  b.extent[0] = 5;
  SG_CHECK_ERRC(check_roles(t, {b}, 0, g), inplace_group_mismatch);
  b = a;
  b.type = 2;
  SG_CHECK_ERRC(check_roles(t, {b}, 0, g), inplace_group_mismatch);
  b = a;
  b.op = "min";
  SG_CHECK_ERRC(check_roles(t, {b}, 0, g), inplace_group_mismatch);
  b = a;
  b.init_bits = {std::byte{1}};
  SG_CHECK_ERRC(check_roles(t, {b}, 0, g), inplace_group_mismatch);
  SG_CHECK_ERRC(check_roles(t, {a}, 1, g), inplace_group_mismatch);
  try {
    check_roles(t, {b}, 0, g);
  } catch (const sg::error &e) {
    BOOST_CHECK(std::string(e.what()).find("initializer") != std::string::npos);
  }
  check_roles(t, {slot(0, role_mode::read, 1, 4)}, 0, g);
  auto e3 = check_roles(t, {a}, 0, g);
  BOOST_CHECK(e3[0].new_group);
  BOOST_CHECK_NE(e3[0].group, e1[0].group);
}
BOOST_AUTO_TEST_CASE(inplace_reuse) {
  handle_table t;
  int g = 0;
  array_slot r = slot(0, role_mode::inplace, 1, 4);
  r.op = "plus";
  r.init = init_kind::reuse;
  SG_CHECK_ERRC(check_roles(t, {r}, 0, g), never_produced);
  check_roles(t, {slot(0, role_mode::produce, 1, 4)}, 0, g);
  r.extent[0] = 5;
  SG_CHECK_ERRC(check_roles(t, {r}, 0, g), size_mismatch);
  r.extent[0] = 4;
  r.type = 2;
  SG_CHECK_ERRC(check_roles(t, {r}, 0, g), type_mismatch);
  r.type = 1;
  auto e = check_roles(t, {r}, 0, g);
  BOOST_CHECK(e[0].new_group && !e[0].needs_fill);
}
BOOST_AUTO_TEST_CASE(producer_closes_group) {
  handle_table t;
  int g = 0;
  array_slot a = slot(0, role_mode::inplace, 1, 4);
  a.op = "plus";
  a.init = init_kind::value;
  a.init_bits = {std::byte{0}};
  check_roles(t, {a}, 0, g);
  check_roles(t, {slot(0, role_mode::produce, 1, 4)}, 0, g);
  BOOST_CHECK_EQUAL(t[0].inplace_group, -1);
}
BOOST_AUTO_TEST_SUITE_END()

namespace {
struct mem_env {
  arena ar;
  std::map<int, std::unique_ptr<pool>> pools;
  std::vector<std::string> log;
  std::vector<void *> host_live;
  memory_manager::services services() {
    memory_manager::services s;
    s.pool_of = [this](int d) -> pool & {
      auto &p = pools[d];
      if (!p)
        p = std::make_unique<pool>(16 * 1024, 16 * 1024, ar.alloc(), ar.dealloc());
      return *p;
    };
    s.host_alloc = [this](std::size_t n) {
      log.push_back("host_alloc");
      void *p = ::operator new(n);
      host_live.push_back(p);
      return p;
    };
    s.host_free = [this](void *p) {
      log.push_back("host_free");
      host_live.erase(std::find(host_live.begin(), host_live.end(), p));
      ::operator delete(p);
    };
    s.wait_ops = [this](const std::vector<op_id> &v) {
      log.push_back("wait" + std::to_string(v.size()));
    };
    s.place_of = [](int d) { return d < 0 ? place{} : place{d, d < 2 ? 1u : 2u}; };
    return s;
  }
};
} // namespace

BOOST_AUTO_TEST_SUITE(memory_manager_suite)
BOOST_AUTO_TEST_CASE(producer_allocates_and_fits) {
  mem_env env;
  memory_manager m(env.services());
  const auto h = m.add_handle(1, 4);
  auto p = m.for_producer(h, 0, 100);
  BOOST_CHECK(p.mem.ptr != nullptr);
  BOOST_CHECK_EQUAL(p.mem.device, 0);
  BOOST_CHECK(!m.handle(h).produced);
  m.note_produced(h, 100, 7);
  BOOST_CHECK(m.handle(h).produced);
  env.log.clear();
  auto q = m.for_producer(h, 0, 200);
  BOOST_CHECK(q.mem.ptr == p.mem.ptr);
  BOOST_CHECK(q.wait_for.empty());
  BOOST_CHECK(env.log.empty());
}
BOOST_AUTO_TEST_CASE(growth_releases_without_waiting) {
  mem_env env;
  memory_manager m(env.services());
  const auto h = m.add_handle(1, 4);
  auto p = m.for_producer(h, 0, 100);
  m.note_produced(h, 100, 7);
  auto q = m.for_producer(h, 0, 1000);
  BOOST_CHECK(q.mem.ptr != p.mem.ptr);
  BOOST_CHECK(std::none_of(env.log.begin(), env.log.end(),
                           [](const std::string &l) { return l.rfind("wait", 0) == 0; }));
  BOOST_CHECK(m.handle(h).users.empty());
}
BOOST_AUTO_TEST_CASE(host_block_freed_only_by_release_finished) {
  mem_env env;
  memory_manager m(env.services());
  const auto h = m.add_handle(1, 4);
  m.for_producer(h, -1, 10);
  m.note_produced(h, 10, 7);
  auto mig = m.for_reader(h, 0);
  BOOST_REQUIRE(mig.has_value());
  m.commit_migration(h, *mig, 9);
  BOOST_CHECK_EQUAL(env.host_live.size(), 1u);
  BOOST_CHECK(std::find(env.log.begin(), env.log.end(), "host_free") == env.log.end());
  m.release_finished();
  BOOST_CHECK(env.host_live.empty());
  BOOST_CHECK(std::none_of(env.log.begin(), env.log.end(),
                           [](const std::string &l) { return l.rfind("wait", 0) == 0; }));
}
BOOST_AUTO_TEST_CASE(host_growth_defers_the_old_block) {
  mem_env env;
  memory_manager m(env.services());
  const auto h = m.add_handle(1, 4);
  m.for_producer(h, -1, 10);
  m.note_produced(h, 10, 7);
  m.for_producer(h, -1, 100);
  BOOST_CHECK_EQUAL(env.host_live.size(), 2u);
  m.release_finished();
  BOOST_CHECK_EQUAL(env.host_live.size(), 1u);
}
BOOST_AUTO_TEST_CASE(pool_error_leaves_state) {
  mem_env env;
  memory_manager m(env.services());
  const auto h = m.add_handle(1, 4);
  auto p = m.for_producer(h, 0, 100);
  SG_CHECK_ERRC(m.for_producer(h, 0, 1u << 20), pool_exhausted);
  BOOST_CHECK(m.handle(h).mem.ptr == p.mem.ptr);
}
BOOST_AUTO_TEST_CASE(host_handles) {
  mem_env env;
  {
    memory_manager m(env.services());
    const auto h = m.add_handle(1, 4);
    m.for_producer(h, -1, 10);
    BOOST_CHECK_EQUAL(m.handle(h).mem.device, -1);
    BOOST_CHECK_EQUAL(env.host_live.size(), 1u);
  }
  BOOST_CHECK(env.host_live.empty());
}
BOOST_AUTO_TEST_CASE(readers_and_migration) {
  mem_env env;
  memory_manager m(env.services());
  const auto h = m.add_handle(1, 4);
  m.for_producer(h, 0, 100);
  m.note_produced(h, 100, 7);
  BOOST_CHECK(!m.for_reader(h, 0).has_value());
  auto direct = m.for_reader(h, 1);
  BOOST_REQUIRE(direct.has_value());
  BOOST_CHECK(direct->plan.kind == copy_kind::direct);
  BOOST_CHECK(std::find(direct->wait_for.begin(), direct->wait_for.end(), 7u) !=
              direct->wait_for.end());
  BOOST_CHECK_EQUAL(m.handle(h).mem.device, 0); // for_reader changed nothing
  auto staged = m.for_reader(h, 2);
  BOOST_REQUIRE(staged.has_value());
  BOOST_CHECK(staged->plan.kind == copy_kind::staged_via_host);
  m.commit_migration(h, *staged, 9);
  BOOST_CHECK_EQUAL(m.handle(h).mem.device, 2);
  BOOST_CHECK((m.handle(h).users == std::vector<op_id>{9}));
}
BOOST_AUTO_TEST_CASE(toolchain) {
  mem_env env;
  memory_manager m(env.services());
  auto a = m.acquire_toolchain(0, 2048);
  m.release_toolchain(0, a, 5);
  auto b = m.acquire_toolchain(0, 2048);
  BOOST_CHECK(b.mem.ptr == a.mem.ptr);
  BOOST_CHECK((b.wait_for == std::vector<op_id>{5}));
}
BOOST_AUTO_TEST_SUITE_END()
