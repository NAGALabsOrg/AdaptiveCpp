#include "world.hpp"

#include "hipSYCL/sg/body.hpp"
#include "hipSYCL/sg/group_expr.hpp"

#include <array>
#include <climits>
#include <functional>
#include <numeric>

using namespace sgd;

namespace {
using V = std::vector<int>;
using S3 = std::array<std::uint64_t, 3>;

std::uint64_t prod(const S3 &s) { return s[0] * s[1] * s[2]; }
S3 remove_dim(const S3 &s, int d) {
  S3 o = {1, 1, 1};
  int c = 0;
  for (int p = 0; p < 3; ++p)
    if (p != d)
      o[c++] = s[p];
  return o;
}
std::uint64_t flat(const S3 &i, const S3 &s) { return (i[0] * s[1] + i[1]) * s[2] + i[2]; }
S3 unflat(std::uint64_t k, const S3 &s) {
  S3 r{};
  for (int d = 2; d >= 0; --d) {
    r[d] = k % s[d];
    k /= s[d];
  }
  return r;
}
S3 with_dim(const S3 &c, int d, std::uint64_t j) {
  S3 i{};
  int ci = 0;
  for (int p = 0; p < 3; ++p)
    i[p] = p == d ? j : c[ci++];
  return i;
}

template <class Op> V ref_reduce(const V &in, const S3 &s, int d, Op op, int ident) {
  const S3 od = remove_dim(s, d);
  V out(prod(od));
  for (std::uint64_t o = 0; o < out.size(); ++o) {
    const S3 c = unflat(o, od);
    int acc = ident;
    for (std::uint64_t j = 0; j < s[d]; ++j)
      acc = op(acc, in[flat(with_dim(c, d, j), s)]);
    out[o] = acc;
  }
  return out;
}
template <class Op> V ref_scan(const V &in, const S3 &s, int d, Op op, bool exclusive, int init) {
  const S3 od = remove_dim(s, d);
  V out(in.size());
  for (std::uint64_t line = 0; line < prod(od); ++line) {
    const S3 c = unflat(line, od);
    int acc = init;
    for (std::uint64_t j = 0; j < s[d]; ++j) {
      const auto idx = flat(with_dim(c, d, j), s);
      if (exclusive) {
        out[idx] = acc;
        acc = op(acc, in[idx]);
      } else {
        acc = j == 0 ? in[idx] : op(acc, in[idx]);
        out[idx] = acc;
      }
    }
  }
  return out;
}

V pattern(std::size_t n) {
  V v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = static_cast<int>((i * 37 + 11) % 23) - 7;
  return v;
}

template <int R, class Make>
V run_expr(const V &in, const S3 &shape, std::size_t out_count, std::size_t W, Make make,
           std::size_t *writes = nullptr) {
  world w;
  const std::size_t in_bytes = (in.size() * 4 + 15) / 16 * 16;
  w.add_layer(1, 0, 0, 1);
  const auto ni = w.add_node(1, 1, 1);
  w.nodes[ni].scratch_offset = in_bytes;
  w.upload();
  int *inp = sycl::malloc_device<int>(in.size(), w.q);
  w.q.memcpy(inp, in.data(), in.size() * 4).wait();
  const launch_ctx *c = w.ctx();
  const std::uint64_t out_base = w.out_addr(0);
  const std::size_t frame_bytes = in_bytes + 64 * 1024 / 4;
  const std::size_t n = in.size();
  const S3 sh = shape;
  w.q.submit([&](sycl::handler &h) {
    sycl::local_accessor<std::byte, 1> fr{sycl::range<1>(frame_bytes), h};
    h.parallel_for(sycl::nd_range<1>(W, W), [=](sycl::nd_item<1> it) {
      char *frame = reinterpret_cast<char *>(&fr[0]);
      for (std::size_t i = it.get_local_id(0); i < n; i += W)
        reinterpret_cast<int *>(frame)[i] = inp[i];
      sycl::group_barrier(it.get_group());
      view_rep v{};
      v.base = reinterpret_cast<std::uint64_t>(frame);
      v.extent[0] = sh[0];
      v.extent[1] = sh[1];
      v.extent[2] = sh[2];
      v.stride[0] = sh[1] * sh[2];
      v.stride[1] = sh[2];
      v.stride[2] = 1;
      auto e = make(sg::group_array<int, R, sg::mapping::local>(v));
      const auto od = e.dims();
      std::uint64_t od3[3] = {od[0], od[1], od[2]};
      const tile_out t = dense_tile(out_base, od3, 4);
      evaluate_group_expr<mock<0>>(e, &it, c, 0, frame, &t);
    });
  });
  w.q.wait_and_throw();
  V out(out_count);
  w.q.memcpy(out.data(), w.d + off_out, out_count * 4).wait();
  if (writes)
    *writes = count_tag(w.log(), t_write);
  sycl::free(inp, w.q);
  return out;
}
} // namespace

BOOST_AUTO_TEST_SUITE(group_eval_suite)

BOOST_AUTO_TEST_CASE(reduce_rank1_all_ops_and_group_sizes) {
  const V in = pattern(37);
  for (std::size_t W : {1u, 4u, 16u}) {
    std::size_t writes = 0;
    auto p = run_expr<1>(in, {37, 1, 1}, 1, W,
                         [](auto v) { return sg::group_reduce<1>(v, 0, sycl::plus<>{}); }, &writes);
    BOOST_CHECK_EQUAL(p[0], std::accumulate(in.begin(), in.end(), 0));
    BOOST_CHECK_EQUAL(writes, 1u);
    auto mn = run_expr<1>(in, {37, 1, 1}, 1, W, [](auto v) { return sg::group_reduce<1>(v, 0, sycl::minimum<>{}); });
    BOOST_CHECK_EQUAL(mn[0], *std::min_element(in.begin(), in.end()));
    auto mx = run_expr<1>(in, {37, 1, 1}, 1, W, [](auto v) { return sg::group_reduce<1>(v, 0, sycl::maximum<>{}); });
    BOOST_CHECK_EQUAL(mx[0], *std::max_element(in.begin(), in.end()));
  }
}

BOOST_AUTO_TEST_CASE(reduce_rank2_each_dim) {
  const S3 s = {5, 7, 1};
  const V in = pattern(35);
  for (int d = 0; d < 2; ++d) {
    const V want = ref_reduce(in, s, d, std::plus<int>{}, 0);
    for (std::size_t W : {4u, 16u}) {
      std::size_t writes = 0;
      V got;
      if (d == 0)
        got = run_expr<2>(in, s, want.size(), W, [](auto v) { return sg::group_reduce<2>(v, 0, sycl::plus<>{}); }, &writes);
      else
        got = run_expr<2>(in, s, want.size(), W, [](auto v) { return sg::group_reduce<2>(v, 1, sycl::plus<>{}); }, &writes);
      BOOST_CHECK(got == want);
      BOOST_CHECK_EQUAL(writes, want.size());
    }
  }
}

BOOST_AUTO_TEST_CASE(reduce_rank3_each_dim) {
  const S3 s = {3, 4, 5};
  const V in = pattern(60);
  const V w0 = ref_reduce(in, s, 0, std::plus<int>{}, 0);
  const V w1 = ref_reduce(in, s, 1, std::plus<int>{}, 0);
  const V w2 = ref_reduce(in, s, 2, std::plus<int>{}, 0);
  BOOST_CHECK(run_expr<3>(in, s, w0.size(), 8, [](auto v) { return sg::group_reduce<3>(v, 0, sycl::plus<>{}); }) == w0);
  BOOST_CHECK(run_expr<3>(in, s, w1.size(), 8, [](auto v) { return sg::group_reduce<3>(v, 1, sycl::plus<>{}); }) == w1);
  BOOST_CHECK(run_expr<3>(in, s, w2.size(), 8, [](auto v) { return sg::group_reduce<3>(v, 2, sycl::plus<>{}); }) == w2);
}

BOOST_AUTO_TEST_CASE(chained_reduces_to_rank_zero) {
  const S3 s = {3, 4, 5};
  const V in = pattern(60);
  const V a = ref_reduce(in, s, 1, std::plus<int>{}, 0);
  const V b = ref_reduce(a, {3, 5, 1}, 0, std::plus<int>{}, 0);
  const V c = ref_reduce(b, {5, 1, 1}, 0, std::plus<int>{}, 0);
  auto got = run_expr<3>(in, s, 1, 8, [](auto v) {
    return sg::group_reduce<1>(sg::group_reduce<2>(sg::group_reduce<3>(v, 1, sycl::plus<>{}), 0, sycl::plus<>{}), 0,
                               sycl::plus<>{});
  });
  BOOST_CHECK_EQUAL(got[0], c[0]);
  auto two = run_expr<3>(in, s, b.size(), 8, [](auto v) {
    return sg::group_reduce<2>(sg::group_reduce<3>(v, 1, sycl::plus<>{}), 0, sycl::plus<>{});
  });
  BOOST_CHECK(two == b);
}

BOOST_AUTO_TEST_CASE(scans_each_dim) {
  const S3 s = {5, 7, 1};
  const V in = pattern(35);
  for (std::size_t W : {4u, 16u}) {
    BOOST_CHECK(run_expr<2>(in, s, 35, W, [](auto v) { return sg::group_inclusive_scan<2>(v, 0, sycl::plus<>{}); }) ==
                ref_scan(in, s, 0, std::plus<int>{}, false, 0));
    BOOST_CHECK(run_expr<2>(in, s, 35, W, [](auto v) { return sg::group_inclusive_scan<2>(v, 1, sycl::plus<>{}); }) ==
                ref_scan(in, s, 1, std::plus<int>{}, false, 0));
    BOOST_CHECK(run_expr<2>(in, s, 35, W, [](auto v) { return sg::group_exclusive_scan<2>(v, 0, 100, sycl::plus<>{}); }) ==
                ref_scan(in, s, 0, std::plus<int>{}, true, 100));
    BOOST_CHECK(run_expr<2>(in, s, 35, W, [](auto v) { return sg::group_exclusive_scan<2>(v, 1, 0, sycl::plus<>{}); }) ==
                ref_scan(in, s, 1, std::plus<int>{}, true, 0));
  }
  const V in1 = pattern(20);
  BOOST_CHECK(run_expr<1>(in1, {20, 1, 1}, 20, 8, [](auto v) { return sg::group_inclusive_scan(v, sycl::plus<>{}); }) ==
              ref_scan(in1, {20, 1, 1}, 0, std::plus<int>{}, false, 0));
  BOOST_CHECK(run_expr<1>(in1, {20, 1, 1}, 20, 8, [](auto v) { return sg::group_exclusive_scan(v, 3, sycl::plus<>{}); }) ==
              ref_scan(in1, {20, 1, 1}, 0, std::plus<int>{}, true, 3));
}

BOOST_AUTO_TEST_CASE(scan_of_reduce) {
  const S3 s = {4, 6, 1};
  const V in = pattern(24);
  const V r = ref_reduce(in, s, 0, std::plus<int>{}, 0);
  const V want = ref_scan(r, {6, 1, 1}, 0, std::plus<int>{}, false, 0);
  BOOST_CHECK(run_expr<2>(in, s, 6, 8, [](auto v) {
                return sg::group_inclusive_scan<1>(sg::group_reduce<2>(v, 0, sycl::plus<>{}), 0, sycl::plus<>{});
              }) == want);
}

BOOST_AUTO_TEST_CASE(elementwise) {
  const S3 s = {5, 7, 1};
  const V in = pattern(35);
  V want(35);
  for (std::size_t i = 0; i < 35; ++i)
    want[i] = (in[i] + in[i]) * 3;
  BOOST_CHECK(run_expr<2>(in, s, 35, 8, [](auto v) { return (v + v) * 3; }) == want);
  const V r = ref_reduce(want, s, 1, std::plus<int>{}, 0);
  BOOST_CHECK(run_expr<2>(in, s, r.size(), 8, [](auto v) { return sg::group_reduce<2>((v + v) * 3, 1, sycl::plus<>{}); }) == r);
}

BOOST_AUTO_TEST_CASE(outputs_smaller_and_larger_than_the_group) {
  const S3 s = {40, 3, 1};
  const V in = pattern(120);
  const V big = ref_reduce(in, s, 1, std::plus<int>{}, 0); // 40 outputs
  const V small = ref_reduce(in, s, 0, std::plus<int>{}, 0); // 3 outputs
  for (std::size_t W : {4u, 64u}) {
    std::size_t writes = 0;
    BOOST_CHECK(run_expr<2>(in, s, big.size(), W, [](auto v) { return sg::group_reduce<2>(v, 1, sycl::plus<>{}); }, &writes) == big);
    BOOST_CHECK_EQUAL(writes, big.size());
    BOOST_CHECK(run_expr<2>(in, s, small.size(), W, [](auto v) { return sg::group_reduce<2>(v, 0, sycl::plus<>{}); }, &writes) == small);
    BOOST_CHECK_EQUAL(writes, small.size());
  }
}

BOOST_AUTO_TEST_SUITE_END()
