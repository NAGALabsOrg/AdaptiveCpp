#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include "hipSYCL/sg/detail/dev/inplace.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

using namespace sgt;

namespace {

template <class T> void contribute(sg::kernel_handle &kh, H<T> ones_in, H<T> y, T k, int n) {
  auto scale = [=](T v) { return static_cast<T>(k * v); };
  auto ones = kh.source(ones_in, sg::shape<1>({n}));
  auto items = kh.work_items({n});
  auto c = kh.map_items(items, [=](auto item) {
    auto v = ones[item.id()];
    return item.compute(v, scale);
  });
  kh.evaluate(sg::inplace(y, sycl::plus<T>{}, T{0}), c);
}

template <class T> std::vector<T> combine_contributions(const std::vector<int> &ks, int n) {
  sg::graph g;
  auto ones = g.create_mem_handle<T>();
  auto y = g.create_mem_handle<T>();
  host_write(g, ones, std::vector<T>(n, T{1}));
  for (int k : ks)
    g.submit(test_device(), [&](sg::kernel_handle &kh) { contribute<T>(kh, ones, y, static_cast<T>(k), n); });
  std::vector<T> out;
  host_read(g, y, n, out);
  g.run();
  g.wait();
  return out;
}

// the atomic combine of the sink task, on a buffer whose first element is at `offset` bytes
template <class T> void combine_at_offset(std::size_t offset) {
  constexpr int n = 67;
  const std::size_t bytes = offset + n * sizeof(T) + 16;
  sycl::queue q(test_device());
  std::byte *buf = sycl::aligned_alloc_device<std::byte>(256, bytes, q);
  std::vector<std::byte> init(bytes, std::byte{0xAB});
  std::memset(init.data() + offset, 0, n * sizeof(T));
  q.memcpy(buf, init.data(), bytes).wait();
  q.parallel_for(sycl::range<1>(2 * n), [=](sycl::id<1> i) {
    T *elements = reinterpret_cast<T *>(buf + offset);
    const T v = static_cast<T>(1 + i[0] / n);
    sg::detail::dev_atomic::combine<T, sycl::plus<T>>(elements + i[0] % n, v);
  });
  q.wait_and_throw();
  std::vector<std::byte> got(bytes);
  q.memcpy(got.data(), buf, bytes).wait();
  sycl::free(buf, q);
  for (std::size_t i = 0; i < offset; ++i)
    BOOST_CHECK_EQUAL(static_cast<int>(got[i]), 0xAB);
  for (std::size_t i = offset + n * sizeof(T); i < bytes; ++i)
    BOOST_CHECK_EQUAL(static_cast<int>(got[i]), 0xAB);
  for (int e = 0; e < n; ++e) {
    T v;
    std::memcpy(&v, got.data() + offset + e * sizeof(T), sizeof(T));
    BOOST_REQUIRE_EQUAL(static_cast<int>(v), 3);
  }
}

} // namespace

BOOST_AUTO_TEST_SUITE(inplace_results)

BOOST_AUTO_TEST_CASE(k11_two_contributions_give_3_at_every_element) {
  const auto out = combine_contributions<float>({1, 2}, 64);
  BOOST_REQUIRE_EQUAL(out.size(), 64u);
  for (float v : out)
    BOOST_CHECK_EQUAL(v, 3.f);
}

BOOST_AUTO_TEST_CASE(narrow_elements_combine_through_the_containing_word_at_every_byte_offset) {
  for (std::size_t offset : {0u, 1u, 2u, 3u}) {
    combine_at_offset<std::uint8_t>(offset);
    if (offset % 2 == 0)
      combine_at_offset<std::uint16_t>(offset);
  }
}

BOOST_AUTO_TEST_CASE(narrow_graph_results_hold_3_in_uint8_and_uint16) {
  const auto a = combine_contributions<std::uint8_t>({1, 2}, 64);
  const auto b = combine_contributions<std::uint16_t>({1, 2}, 64);
  for (auto v : a)
    BOOST_CHECK_EQUAL(static_cast<int>(v), 3);
  for (auto v : b)
    BOOST_CHECK_EQUAL(static_cast<int>(v), 3);
}

BOOST_AUTO_TEST_CASE(three_contributors_combine_0_1_2_3_into_6) {
  const auto out = combine_contributions<std::int32_t>({1, 2, 3}, 64);
  for (auto v : out)
    BOOST_CHECK_EQUAL(v, 6);
}

BOOST_AUTO_TEST_SUITE_END()
