#include "../common/check.hpp"

#include "hipSYCL/sg/detail/array_rep.hpp"
#include "hipSYCL/sg/detail/closure.hpp"
#include "hipSYCL/sg/detail/type_id.hpp"
#include "hipSYCL/sg/error.hpp"

#include <cstring>
#include <sycl/sycl.hpp>

namespace a { struct S {}; }
namespace b { struct S {}; }

using namespace sg::detail;

BOOST_AUTO_TEST_SUITE(type_id_suite)
BOOST_AUTO_TEST_CASE(fnv_vectors) {
  BOOST_CHECK_EQUAL(fnv1a64(""), 0xcbf29ce484222325ull);
  BOOST_CHECK_EQUAL(fnv1a64("a"), 0xaf63dc4c8601ec8cull);
}
BOOST_AUTO_TEST_CASE(ids) {
  BOOST_CHECK_EQUAL(type_id<int>(), type_id<int>());
  BOOST_CHECK_NE(type_id<int>(), type_id<unsigned>());
  BOOST_CHECK_NE(type_id<a::S>(), type_id<b::S>());
  BOOST_CHECK_NE(type_id<sycl::plus<>>(), type_id<sycl::minimum<>>());
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(error_suite)
BOOST_AUTO_TEST_CASE(round_trip) {
  sg::error e{sg::errc::size_mismatch, "four"};
  BOOST_CHECK(e.code() == sg::errc::size_mismatch);
  BOOST_CHECK_EQUAL(std::string(e.what()), "four");
  BOOST_CHECK_EQUAL(std::string(sg::to_string(sg::errc::pool_exhausted)), "pool_exhausted");
  BOOST_CHECK_EQUAL(std::string(sg::to_string(sg::errc::internal)), "internal");
  BOOST_CHECK_EQUAL(std::string(sg::to_string(sg::errc::role_conflict)), "role_conflict");
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(array_rep_suite)
BOOST_AUTO_TEST_CASE(sentinels) {
  for (int rank = 1; rank <= 3; ++rank)
    for (std::uint32_t slot : {0u, 1u, 2u, (1u << 24) - 1}) {
      const auto s = sentinel(rank, slot);
      BOOST_CHECK(is_sentinel(s));
      BOOST_CHECK_EQUAL(sentinel_rank(s), rank);
      BOOST_CHECK_EQUAL(sentinel_slot(s), slot);
    }
  BOOST_CHECK(!is_sentinel(0));
  BOOST_CHECK(!is_sentinel(5));
  BOOST_CHECK(!is_sentinel(0x00007f0012345678ull));
}
BOOST_AUTO_TEST_SUITE_END()

namespace {
std::vector<std::byte> bytes(std::size_t n) { return std::vector<std::byte>(n, std::byte{0}); }
void put64(std::vector<std::byte> &b, std::size_t o, std::uint64_t v) { std::memcpy(b.data() + o, &v, 8); }
} // namespace

BOOST_AUTO_TEST_SUITE(closure_suite)
BOOST_AUTO_TEST_CASE(sites) {
  auto b = bytes(128);
  put64(b, 0, sentinel(2, 3));
  put64(b, 32, sentinel(1, 7));
  put64(b, 64, sentinel(3, 0));
  const auto s = find_patch_sites(b);
  BOOST_REQUIRE_EQUAL(s.size(), 3u);
  BOOST_CHECK_EQUAL(s[0].offset, 0u);
  BOOST_CHECK_EQUAL(s[0].rank, 2);
  BOOST_CHECK_EQUAL(s[0].slot, 3u);
  BOOST_CHECK_EQUAL(s[1].offset, 32u);
  BOOST_CHECK_EQUAL(s[1].slot, 7u);
  BOOST_CHECK_EQUAL(s[2].offset, 64u);
  BOOST_CHECK_EQUAL(s[2].rank, 3);
}
BOOST_AUTO_TEST_CASE(adjacent_arrays) {
  auto b = bytes(64);
  put64(b, 0, sentinel(1, 0));
  put64(b, 32, sentinel(1, 1));
  BOOST_CHECK_EQUAL(find_patch_sites(b).size(), 2u);
}
BOOST_AUTO_TEST_CASE(skipped_candidates) {
  auto b = bytes(64);
  put64(b, 0, sentinel(0, 1));
  put64(b, 32, sentinel(4, 1));
  BOOST_CHECK(find_patch_sites(b).empty());
  auto u = bytes(64);
  put64(u, 4, sentinel(1, 1)); // unaligned
  BOOST_CHECK(find_patch_sites(u).empty());
  auto t = bytes(40);
  put64(t, 16, sentinel(1, 1)); // 16 + 32 > 40
  BOOST_CHECK(find_patch_sites(t).empty());
}
BOOST_AUTO_TEST_CASE(ordinary_pointers) {
  auto b = bytes(64);
  put64(b, 0, 0x00007ffd12345678ull);
  put64(b, 8, 12345);
  BOOST_CHECK(find_patch_sites(b).empty());
}
struct two_arrays {
  array_rep x, y;
  int k;
};
BOOST_AUTO_TEST_CASE(capture) {
  two_arrays c{};
  c.x.address = sentinel(2, 4);
  c.x.extent[0] = 3;
  c.y.address = sentinel(1, 9);
  c.k = 77;
  const closure_entry e = capture_closure(c);
  BOOST_CHECK_EQUAL(e.bytes.size(), sizeof(c));
  BOOST_CHECK_EQUAL(e.align, alignof(two_arrays));
  BOOST_REQUIRE_EQUAL(e.sites.size(), 2u);
  BOOST_CHECK_EQUAL(e.sites[0].slot, 4u);
  BOOST_CHECK_EQUAL(e.sites[1].slot, 9u);
  BOOST_CHECK_EQUAL(std::memcmp(e.bytes.data(), &c, sizeof(c)), 0);
}
BOOST_AUTO_TEST_CASE(patch) {
  two_arrays c{};
  c.x.address = sentinel(2, 0);
  c.y.address = sentinel(1, 1);
  c.k = 5;
  const closure_entry e = capture_closure(c);
  const closure_entry before = e;
  std::vector<slot_binding> by(2);
  by[0].address = 0x1000;
  by[0].extent[0] = 4;
  by[0].extent[1] = 6;
  by[1].address = 0x2000;
  by[1].extent[0] = 9;
  const auto out = patch_closure(e, by);
  two_arrays p;
  std::memcpy(&p, out.data(), sizeof(p));
  BOOST_CHECK_EQUAL(p.x.address, 0x1000u);
  BOOST_CHECK_EQUAL(p.x.extent[0], 4u);
  BOOST_CHECK_EQUAL(p.x.extent[1], 6u);
  BOOST_CHECK_EQUAL(p.y.address, 0x2000u);
  BOOST_CHECK_EQUAL(p.y.extent[0], 9u);
  BOOST_CHECK_EQUAL(p.k, 5);
  BOOST_CHECK(e.bytes == before.bytes);
  by.resize(1);
  SG_CHECK_ERRC(patch_closure(e, by), internal);
}
BOOST_AUTO_TEST_SUITE_END()
