#include "../common/check.hpp"

#include "hipSYCL/sg/detail/copy_plan.hpp"
#include "hipSYCL/sg/detail/dependency.hpp"
#include "hipSYCL/sg/detail/device_ops.hpp"
#include "hipSYCL/sg/detail/memory_manager.hpp"
#include "hipSYCL/sg/detail/pool.hpp"
#include "hipSYCL/sg/detail/role_check.hpp"
#include "hipSYCL/sg/detail/slab_bitmap.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>

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

namespace {
struct arena {
  struct raw_free {
    void operator()(void *p) const { std::free(p); }
  };
  std::vector<std::unique_ptr<void, raw_free>> blocks;
  std::vector<std::size_t> allocs;
  std::vector<void *> frees;
  std::size_t skew = 0; // makes the allocation start off the 256-byte boundary
  std::function<void *(std::size_t)> alloc() {
    return [this](std::size_t n) {
      void *p = std::aligned_alloc(256, (n + skew + 255) / 256 * 256);
      blocks.emplace_back(p);
      allocs.push_back(n);
      return static_cast<void *>(static_cast<char *>(p) + skew);
    };
  }
  std::function<void(void *)> dealloc() {
    return [this](void *p) { frees.push_back(p); };
  }
};
} // namespace

BOOST_AUTO_TEST_SUITE(pool_suite)
BOOST_AUTO_TEST_CASE(a_zero_reservation_takes_nothing_until_the_first_request_and_then_one_slab) {
  arena a;
  {
    pool p(0, 64u << 20, a.alloc(), a.dealloc());
    BOOST_CHECK(a.allocs.empty());
    BOOST_CHECK_EQUAL(p.slab_count(), 0u);
    pool_block x = p.allocate(100, {});
    BOOST_REQUIRE_EQUAL(a.allocs.size(), 1u);
    BOOST_CHECK_EQUAL(a.allocs[0], pool::min_slab_bytes);
    BOOST_CHECK_EQUAL(p.slab_count(), 1u);
    BOOST_CHECK_EQUAL(x.nodes, 1u);
  }
}
BOOST_AUTO_TEST_CASE(a_slab_is_the_request_rounded_up_to_the_minimum_and_capped_at_the_largest_allocation) {
  arena a;
  pool p(0, 64u << 20, a.alloc(), a.dealloc());
  p.allocate(pool::min_slab_bytes + 1, {}); // above the minimum: the request plus alignment slack, in whole nodes
  BOOST_REQUIRE_EQUAL(a.allocs.size(), 1u);
  BOOST_CHECK_EQUAL(a.allocs[0], pool::min_slab_bytes + 2 * pool::node_bytes);
  arena b;
  pool q(0, 6 * 1024, b.alloc(), b.dealloc()); // the device's largest allocation is 6 KiB
  q.allocate(2048, {});
  BOOST_REQUIRE_EQUAL(b.allocs.size(), 1u);
  BOOST_CHECK_EQUAL(b.allocs[0], 6 * 1024u);
  try {
    q.allocate(7 * 1024, {});
    BOOST_FAIL("no throw");
  } catch (const sg::error &e) {
    BOOST_CHECK(e.code() == sg::errc::allocation_refused);
    BOOST_CHECK(std::string(e.what()).find("7168") != std::string::npos);
  }
}
BOOST_AUTO_TEST_CASE(a_reservation_of_r_bytes_is_taken_at_construction_and_requests_that_fit_take_nothing_more) {
  arena a;
  {
    pool p(10 * 1024 + 500, 4 * 1024 + 7, a.alloc(), a.dealloc());
    BOOST_REQUIRE_EQUAL(a.allocs.size(), 3u);
    BOOST_CHECK_EQUAL(a.allocs[0], 4096u);
    BOOST_CHECK_EQUAL(a.allocs[1], 4096u);
    BOOST_CHECK_EQUAL(a.allocs[2], 2048u);
    BOOST_CHECK_EQUAL(p.reserved(), 10240u);
    p.allocate(3000, {});
    p.allocate(3000, {});
    p.allocate(2000, {});
    BOOST_CHECK_EQUAL(a.allocs.size(), 3u);
  }
  BOOST_CHECK_EQUAL(a.frees.size(), 3u); // every slab returned exactly once
}
BOOST_AUTO_TEST_CASE(allocate_basics) {
  arena a;
  pool p(8 * 1024, 4 * 1024, a.alloc(), a.dealloc());
  BOOST_CHECK(p.allocate(0, {}).ptr == nullptr);
  pool_block x = p.allocate(1, {});
  pool_block y = p.allocate(1025, {});
  BOOST_CHECK_EQUAL(x.nodes, 1u);
  BOOST_CHECK_EQUAL(y.nodes, 2u);
  BOOST_CHECK(static_cast<char *>(y.ptr) == static_cast<char *>(x.ptr) + 1024);
  BOOST_CHECK_EQUAL(p.used(), 3 * 1024u);
  pool_block z = p.allocate(4096, {}); // does not fit slab 0 any more, goes to slab 1
  BOOST_CHECK_EQUAL(z.slab, 1u);
}
BOOST_AUTO_TEST_CASE(a_block_is_reused_only_when_its_last_users_finished_or_are_dependencies) {
  arena a;
  std::set<op_id> done;
  pool p(0, 2048, a.alloc(), a.dealloc(), [&done](op_id o) { return done.count(o) != 0; }); // a slab holds one block
  pool_block x = p.allocate(2048, {});
  p.release(x, {5, 6}); // 5 and 6 are unfinished
  BOOST_CHECK_EQUAL(p.slab_count(), 1u);
  // 6 is not a dependency of the request: not reused, a new slab is taken
  pool_block y = p.allocate(2048, {5});
  BOOST_CHECK_EQUAL(p.slab_count(), 2u);
  BOOST_CHECK(y.slab == 1u);
  // both users are dependencies of the request: reused, nothing new taken
  pool_block z = p.allocate(2048, {5, 6, 9});
  BOOST_CHECK_EQUAL(p.slab_count(), 2u);
  BOOST_CHECK(z.ptr == x.ptr);
  // finished users need no dependency
  p.release(z, {5, 6});
  done = {5, 6};
  pool_block w = p.allocate(2048, {});
  BOOST_CHECK_EQUAL(p.slab_count(), 2u);
  BOOST_CHECK(w.ptr == x.ptr);
}
BOOST_AUTO_TEST_CASE(randomized_steps_never_overlap_and_stay_aligned) {
  for (std::size_t skew : {std::size_t{0}, std::size_t{16}}) {
    arena a;
    a.skew = skew;
    pool p(256 * 1024, 64 * 1024, a.alloc(), a.dealloc());
    std::mt19937 rng(777);
    std::vector<pool_block> live;
    for (int step = 0; step < 10000; ++step) {
      if (live.empty() || rng() % 2) {
        pool_block b = p.allocate(1 + rng() % 9000, {});
        BOOST_REQUIRE(reinterpret_cast<std::uintptr_t>(b.ptr) % 256 == 0);
        for (const pool_block &o : live) {
          const char *lo = static_cast<const char *>(b.ptr), *ol = static_cast<const char *>(o.ptr);
          BOOST_REQUIRE(lo + b.nodes * pool::node_bytes <= ol || ol + o.nodes * pool::node_bytes <= lo);
        }
        live.push_back(b);
      } else {
        const std::size_t k = rng() % live.size();
        p.release(live[k], {});
        live.erase(live.begin() + k);
      }
    }
  }
}
BOOST_AUTO_TEST_CASE(double_release_throws_and_determinism) {
  arena a;
  pool p(4 * 1024, 4 * 1024, a.alloc(), a.dealloc());
  pool_block x = p.allocate(1024, {});
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
  std::set<op_id> finished_ops;
  std::size_t reservation = 0;
  std::size_t largest = 64u << 20; // the device's largest single allocation
  std::size_t refuse_above = std::numeric_limits<std::size_t>::max(); // the device refuses larger requests
  std::vector<std::string> log;
  std::vector<void *> host_live;
  memory_manager::services services() {
    memory_manager::services s;
    s.pool_of = [this](int d) -> pool & {
      auto &p = pools[d];
      if (!p) {
        auto alloc = ar.alloc();
        p = std::make_unique<pool>(
            reservation, largest,
            [this, alloc](std::size_t n) -> void * {
              if (n > refuse_above)
                throw sg::error{sg::errc::allocation_refused, "refused " + std::to_string(n)};
              return alloc(n);
            },
            ar.dealloc(), [this](op_id o) { return finished_ops.count(o) != 0; });
      }
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
  auto p = m.for_producer(h, 0, 100, {});
  BOOST_CHECK(p.mem.ptr != nullptr);
  BOOST_CHECK_EQUAL(p.mem.device, 0);
  BOOST_CHECK(!m.handle(h).produced);
  m.note_produced(h, 100, 7);
  BOOST_CHECK(m.handle(h).produced);
  env.log.clear();
  auto q = m.for_producer(h, 0, 200, {});
  BOOST_CHECK(q.mem.ptr == p.mem.ptr);
  BOOST_CHECK(env.log.empty());
}
BOOST_AUTO_TEST_CASE(growth_releases_without_waiting) {
  mem_env env;
  memory_manager m(env.services());
  const auto h = m.add_handle(1, 4);
  auto p = m.for_producer(h, 0, 100, {});
  m.note_produced(h, 100, 7);
  auto q = m.for_producer(h, 0, 1000, {});
  BOOST_CHECK(q.mem.ptr != p.mem.ptr);
  BOOST_CHECK(std::none_of(env.log.begin(), env.log.end(),
                           [](const std::string &l) { return l.rfind("wait", 0) == 0; }));
  BOOST_CHECK(m.handle(h).users.empty());
}
BOOST_AUTO_TEST_CASE(host_block_freed_only_by_release_finished) {
  mem_env env;
  memory_manager m(env.services());
  const auto h = m.add_handle(1, 4);
  m.for_producer(h, -1, 10, {});
  m.note_produced(h, 10, 7);
  auto mig = m.for_reader(h, 0, {});
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
  m.for_producer(h, -1, 10, {});
  m.note_produced(h, 10, 7);
  m.for_producer(h, -1, 100, {});
  BOOST_CHECK_EQUAL(env.host_live.size(), 2u);
  m.release_finished();
  BOOST_CHECK_EQUAL(env.host_live.size(), 1u);
}
BOOST_AUTO_TEST_CASE(with_a_zero_reservation_nothing_is_taken_before_the_first_request_which_takes_one_slab) {
  mem_env env;
  memory_manager m(env.services());
  const auto h = m.add_handle(1, 4);
  BOOST_CHECK(env.ar.allocs.empty());
  m.for_producer(h, 0, 100, {});
  BOOST_CHECK_EQUAL(env.ar.allocs.size(), 1u);
  BOOST_CHECK_EQUAL(env.pools.at(0)->slab_count(), 1u);
}
BOOST_AUTO_TEST_CASE(a_request_the_device_refuses_names_the_handle_and_the_size) {
  mem_env env;
  env.refuse_above = 1024;
  memory_manager m(env.services());
  m.add_handle(1, 4);
  const auto h = m.add_handle(1, 4);
  try {
    m.for_producer(h, 0, 1000, {});
    BOOST_FAIL("no throw");
  } catch (const sg::error &e) {
    BOOST_CHECK(e.code() == sg::errc::allocation_refused);
    const std::string w = e.what();
    BOOST_CHECK(w.find("handle 1") != std::string::npos);
    BOOST_CHECK(w.find("4000") != std::string::npos);
  }
  BOOST_CHECK(m.handle(h).mem.ptr == nullptr);
}
BOOST_AUTO_TEST_CASE(a_free_block_with_an_unfinished_unrelated_user_is_not_reused) {
  mem_env env;
  env.largest = 4096; // a slab holds one block of 4000 bytes
  memory_manager m(env.services());
  const auto b = m.add_handle(1, 1);
  auto t = m.acquire_toolchain(0, 4000, {});
  m.release_toolchain(0, t, 11); // the block is free; its last user, operation 11, is unfinished
  BOOST_CHECK_EQUAL(env.pools.at(0)->slab_count(), 1u);
  m.for_producer(b, 0, 4000, {}); // 11 is not a dependency of the request: a new slab
  BOOST_CHECK_EQUAL(env.pools.at(0)->slab_count(), 2u);
}
BOOST_AUTO_TEST_CASE(a_free_block_whose_user_finished_or_is_a_dependency_is_reused_with_no_slab_taken) {
  for (int how : {0, 1}) {
    mem_env env;
    env.largest = 4096;
    memory_manager m(env.services());
    const auto b = m.add_handle(1, 1);
    auto t = m.acquire_toolchain(0, 4000, {});
    m.release_toolchain(0, t, 11);
    if (how == 0)
      env.finished_ops.insert(11); // finished
    // how == 1: operation 11 is a dependency of the requesting work
    auto pb = m.for_producer(b, 0, 4000, how == 1 ? std::vector<op_id>{11} : std::vector<op_id>{});
    BOOST_CHECK_EQUAL(env.pools.at(0)->slab_count(), 1u);
    BOOST_CHECK(pb.mem.ptr == t.mem.ptr);
  }
}
BOOST_AUTO_TEST_CASE(a_reservation_is_taken_at_first_use_and_requests_that_fit_take_nothing_more) {
  mem_env env;
  env.reservation = 16 * 1024;
  env.largest = 16 * 1024;
  memory_manager m(env.services());
  const auto h = m.add_handle(1, 4);
  const auto g = m.add_handle(1, 4);
  m.for_producer(h, 0, 100, {});
  BOOST_CHECK_EQUAL(env.ar.allocs.size(), 1u);
  m.for_producer(g, 0, 3000, {});
  BOOST_CHECK_EQUAL(env.ar.allocs.size(), 1u);
}
BOOST_AUTO_TEST_CASE(destroying_the_pools_returns_every_slab_exactly_once) {
  mem_env env;
  env.largest = 4096;
  {
    memory_manager m(env.services());
    const auto h = m.add_handle(1, 1);
    const auto g = m.add_handle(1, 1);
    m.for_producer(h, 0, 4000, {});
    m.for_producer(g, 0, 4000, {});
    BOOST_CHECK_EQUAL(env.ar.allocs.size(), 2u);
  }
  BOOST_CHECK(env.ar.frees.empty()); // slabs stay with the pool until it is destroyed
  env.pools.clear();
  BOOST_REQUIRE_EQUAL(env.ar.frees.size(), 2u);
  BOOST_CHECK(env.ar.frees[0] != env.ar.frees[1]);
}
BOOST_AUTO_TEST_CASE(host_handles) {
  mem_env env;
  {
    memory_manager m(env.services());
    const auto h = m.add_handle(1, 4);
    m.for_producer(h, -1, 10, {});
    BOOST_CHECK_EQUAL(m.handle(h).mem.device, -1);
    BOOST_CHECK_EQUAL(env.host_live.size(), 1u);
  }
  BOOST_CHECK(env.host_live.empty());
}
BOOST_AUTO_TEST_CASE(readers_and_migration) {
  mem_env env;
  memory_manager m(env.services());
  const auto h = m.add_handle(1, 4);
  m.for_producer(h, 0, 100, {});
  m.note_produced(h, 100, 7);
  BOOST_CHECK(!m.for_reader(h, 0, {}).has_value());
  auto direct = m.for_reader(h, 1, {});
  BOOST_REQUIRE(direct.has_value());
  BOOST_CHECK(direct->plan.kind == copy_kind::direct);
  BOOST_CHECK(std::find(direct->wait_for.begin(), direct->wait_for.end(), 7u) !=
              direct->wait_for.end());
  BOOST_CHECK_EQUAL(m.handle(h).mem.device, 0); // for_reader changed nothing
  auto staged = m.for_reader(h, 2, {});
  BOOST_REQUIRE(staged.has_value());
  BOOST_CHECK(staged->plan.kind == copy_kind::staged_via_host);
  m.commit_migration(h, *staged, 9);
  BOOST_CHECK_EQUAL(m.handle(h).mem.device, 2);
  BOOST_CHECK((m.handle(h).users == std::vector<op_id>{9}));
}
BOOST_AUTO_TEST_CASE(fill_writes_exactly_the_requested_elements) {
  for (std::size_t bytes : {1u, 2u, 4u, 12u, 16u}) {
    std::vector<unsigned char> buf(67 * bytes + 2 * 16, 0xee);
    std::vector<unsigned char> pattern(bytes);
    for (std::size_t i = 0; i < bytes; ++i)
      pattern[i] = static_cast<unsigned char>(i + 1);
    fill_range(buf.data() + 16, pattern.data(), bytes, 67);
    for (std::size_t i = 0; i < 16; ++i) {
      BOOST_CHECK_EQUAL(buf[i], 0xee);
      BOOST_CHECK_EQUAL(buf[16 + 67 * bytes + i], 0xee);
    }
    for (std::size_t e = 0; e < 67; ++e)
      for (std::size_t b = 0; b < bytes; ++b)
        BOOST_REQUIRE_EQUAL(buf[16 + e * bytes + b], pattern[b]);
  }
}
BOOST_AUTO_TEST_CASE(toolchain) {
  mem_env env;
  memory_manager m(env.services());
  auto a = m.acquire_toolchain(0, 2048, {});
  m.release_toolchain(0, a, 5);
  auto b = m.acquire_toolchain(0, 2048, {5}); // the launch depends on the previous launch: its block is reused
  BOOST_CHECK(b.mem.ptr == a.mem.ptr);
  BOOST_CHECK_EQUAL(env.ar.allocs.size(), 1u);
}
BOOST_AUTO_TEST_SUITE_END()
