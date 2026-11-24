#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include "hipSYCL/sg/detail/dev/view_read.hpp"

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

using namespace sgt;
using namespace sg::detail;

namespace {

struct query {
  std::uint32_t first_node; // the descriptor's first node in the node array
  std::int32_t top;         // the node read
  std::int64_t tile, local;
};

// A set of random descriptors: the nodes, the data they read, the questions asked, and
// the answer of an independent brute-force reference for each question.
struct population {
  std::vector<view_node> nodes; // bases and tables hold offsets into `blob` until patched
  std::vector<std::byte> blob;
  std::vector<query> queries;
  std::vector<std::int32_t> expected;

  std::uint64_t put(const void *p, std::size_t bytes) {
    while (blob.size() % 8)
      blob.push_back(std::byte{0});
    const std::uint64_t at = blob.size();
    blob.resize(blob.size() + bytes);
    std::memcpy(blob.data() + at, p, bytes);
    return at;
  }
};

view_node make_node(vk k, std::uint32_t grid_rank, std::uint32_t shape_rank) {
  view_node n;
  n.kind = static_cast<std::uint32_t>(k);
  n.elem_bytes = 4;
  n.grid_rank = grid_rank;
  n.shape_rank = shape_rank;
  return n;
}

std::int32_t pick(std::int64_t i, std::int64_t len, std::uint32_t oob, std::int32_t fill, const std::vector<std::int32_t> &d) {
  if (i < 0 || i >= len)
    return oob == static_cast<std::uint32_t>(oob_kind::fill) ? fill : d[((i % len) + len) % len];
  return d[i];
}

void add_tile_population(population &p, std::mt19937 &rng, int rank) {
  std::int64_t S[3] = {1, 1, 1}, T[3] = {1, 1, 1}, G[3] = {1, 1, 1};
  std::int64_t count = 1;
  for (int d = 0; d < rank; ++d) {
    S[d] = 1 + rng() % 9;
    T[d] = 1 + rng() % 4;
    G[d] = (S[d] + T[d] - 1) / T[d];
    count *= S[d];
  }
  std::vector<std::int32_t> data(count);
  for (auto &v : data)
    v = static_cast<std::int32_t>(rng() % 1000);
  const std::int32_t fill = -7;
  const std::uint32_t oob = rng() % 2;
  const std::uint32_t first = static_cast<std::uint32_t>(p.nodes.size());
  view_node src = make_node(vk::source, 0, rank);
  std::copy(S, S + 3, src.shape);
  src.base = p.put(data.data(), data.size() * 4);
  view_node tile = make_node(vk::tile, rank, rank);
  tile.parent = 0;
  tile.oob = oob;
  tile.fill[0] = static_cast<std::uint32_t>(fill);
  std::copy(G, G + 3, tile.grid);
  std::copy(T, T + 3, tile.shape);
  p.nodes.push_back(src);
  p.nodes.push_back(tile);
  for (std::int64_t a = 0; a < prod3(G); ++a)
    for (std::int64_t b = 0; b < prod3(T); ++b) {
      std::int64_t g[3], l[3], c[3];
      unflatten3(a, G, g);
      unflatten3(b, T, l);
      bool outside = false;
      for (int d = 0; d < 3; ++d) {
        c[d] = g[d] * T[d] + l[d];
        outside = outside || c[d] >= S[d];
      }
      std::int32_t want;
      if (outside && oob == static_cast<std::uint32_t>(oob_kind::fill)) {
        want = fill;
      } else {
        for (int d = 0; d < 3; ++d)
          c[d] %= S[d];
        want = data[flatten3(c, S)];
      }
      p.queries.push_back({first, 1, a, b});
      p.expected.push_back(want);
    }
}

void add_retile_population(population &p, std::mt19937 &rng) {
  const std::int64_t L = 1 + rng() % 40;
  const std::int64_t T1 = 1 + rng() % 5;
  const std::int64_t G1 = (L + T1 - 1) / T1;
  const std::int64_t total = G1 * T1;
  std::vector<std::int64_t> divisors;
  for (std::int64_t d = 1; d <= total; ++d)
    if (total % d == 0)
      divisors.push_back(d);
  const std::int64_t T2 = divisors[rng() % divisors.size()];
  std::vector<std::int32_t> data(L);
  for (auto &v : data)
    v = static_cast<std::int32_t>(rng() % 1000);
  const std::int32_t fill = -9;
  const std::uint32_t first = static_cast<std::uint32_t>(p.nodes.size());
  view_node src = make_node(vk::source, 0, 1);
  src.shape[0] = L;
  src.base = p.put(data.data(), data.size() * 4);
  view_node tile = make_node(vk::tile, 1, 1);
  tile.parent = 0;
  tile.oob = static_cast<std::uint32_t>(oob_kind::fill);
  tile.fill[0] = static_cast<std::uint32_t>(fill);
  tile.grid[0] = G1;
  tile.shape[0] = T1;
  view_node re = make_node(vk::retile, 1, 1);
  re.parent = 1;
  re.grid[0] = total / T2;
  re.shape[0] = T2;
  p.nodes.push_back(src);
  p.nodes.push_back(tile);
  p.nodes.push_back(re);
  for (std::int64_t a = 0; a < total / T2; ++a)
    for (std::int64_t b = 0; b < T2; ++b) {
      p.queries.push_back({first, 2, a, b});
      const std::int64_t lin = a * T2 + b;
      p.expected.push_back(lin < L ? data[lin] : fill);
    }
}

void add_extend_population(population &p, std::mt19937 &rng) {
  const std::int64_t L = 1 + rng() % 12;
  const std::int64_t K = 1 + rng() % 70;
  std::vector<std::int32_t> data(L);
  for (auto &v : data)
    v = static_cast<std::int32_t>(rng() % 1000);
  const std::uint32_t first = static_cast<std::uint32_t>(p.nodes.size());
  view_node src = make_node(vk::source, 0, 1);
  src.shape[0] = L;
  src.base = p.put(data.data(), data.size() * 4);
  view_node tile = make_node(vk::tile, 1, 1);
  tile.parent = 0;
  tile.grid[0] = L;
  tile.shape[0] = 1;
  view_node ext = make_node(vk::extend, 1, 2);
  ext.parent = 1;
  ext.grid[0] = L;
  ext.shape[0] = 1;
  ext.shape[1] = K;
  p.nodes.push_back(src);
  p.nodes.push_back(tile);
  p.nodes.push_back(ext);
  for (std::int64_t a = 0; a < L; ++a)
    for (std::int64_t b = 0; b < K; ++b) {
      p.queries.push_back({first, 2, a, b});
      p.expected.push_back(data[a]);
    }
}

void add_table_population(population &p, std::mt19937 &rng) {
  const std::int64_t L = 1 + rng() % 30;
  const std::int64_t Tt = 1 + rng() % 4;
  const std::int64_t Gt = 1 + rng() % 8;
  std::vector<std::int32_t> data(L);
  for (auto &v : data)
    v = static_cast<std::int32_t>(rng() % 1000);
  std::vector<std::int64_t> table(Gt * Tt);
  for (auto &v : table)
    v = static_cast<std::int64_t>(rng() % (L + 6)) - 3;
  const std::int32_t fill = -5;
  const std::uint32_t oob = rng() % 2;
  const std::uint32_t first = static_cast<std::uint32_t>(p.nodes.size());
  view_node src = make_node(vk::source, 0, 1);
  src.shape[0] = L;
  src.base = p.put(data.data(), data.size() * 4);
  view_node tab = make_node(vk::table, 1, 1);
  tab.grid[0] = Gt;
  tab.shape[0] = Tt;
  tab.table_rank = 1;
  tab.table = p.put(table.data(), table.size() * 8);
  view_node im = make_node(vk::index_map, 1, 1);
  im.parent = 0;
  im.parent2 = 1;
  im.oob = oob;
  im.fill[0] = static_cast<std::uint32_t>(fill);
  im.grid[0] = Gt;
  im.shape[0] = Tt;
  p.nodes.push_back(src);
  p.nodes.push_back(tab);
  p.nodes.push_back(im);
  for (std::int64_t a = 0; a < Gt; ++a)
    for (std::int64_t b = 0; b < Tt; ++b) {
      p.queries.push_back({first, 2, a, b});
      p.expected.push_back(pick(table[a * Tt + b], L, oob, fill, data));
    }
}

void add_data_index_population(population &p, std::mt19937 &rng) {
  const std::int64_t L = 1 + rng() % 30;
  const std::int64_t M = 1 + rng() % 20;
  const std::int64_t Tt = 1 + rng() % 4;
  const std::int64_t Gt = (M + Tt - 1) / Tt;
  std::vector<std::int32_t> data(L), idx(M);
  for (auto &v : data)
    v = static_cast<std::int32_t>(rng() % 1000);
  for (auto &v : idx)
    v = static_cast<std::int32_t>(rng() % (L + 6)) - 3;
  const std::int32_t fill = -5;
  const std::uint32_t oob = rng() % 2;
  const std::uint32_t first = static_cast<std::uint32_t>(p.nodes.size());
  view_node src = make_node(vk::source, 0, 1);
  src.shape[0] = L;
  src.base = p.put(data.data(), data.size() * 4);
  view_node isrc = make_node(vk::source, 0, 1);
  isrc.shape[0] = M;
  isrc.base = p.put(idx.data(), idx.size() * 4);
  view_node itile = make_node(vk::tile, 1, 1);
  itile.parent = 1;
  itile.oob = static_cast<std::uint32_t>(oob_kind::fill);
  itile.fill[0] = static_cast<std::uint32_t>(-1);
  itile.grid[0] = Gt;
  itile.shape[0] = Tt;
  view_node im = make_node(vk::index_map, 1, 1);
  im.parent = 0;
  im.parent2 = 2;
  im.oob = oob;
  im.fill[0] = static_cast<std::uint32_t>(fill);
  im.grid[0] = Gt;
  im.shape[0] = Tt;
  p.nodes.push_back(src);
  p.nodes.push_back(isrc);
  p.nodes.push_back(itile);
  p.nodes.push_back(im);
  for (std::int64_t a = 0; a < Gt; ++a)
    for (std::int64_t b = 0; b < Tt; ++b) {
      const std::int64_t ci = a * Tt + b;
      const std::int64_t index = ci < M ? idx[ci] : -1;
      p.queries.push_back({first, 3, a, b});
      p.expected.push_back(pick(index, L, oob, fill, data));
    }
}

std::vector<view_node> patched(const std::vector<view_node> &nodes, const std::byte *blob_base) {
  std::vector<view_node> out = nodes;
  for (view_node &n : out) {
    if (n.kind == static_cast<std::uint32_t>(vk::source))
      n.base = reinterpret_cast<std::uint64_t>(blob_base) + n.base;
    if (n.kind == static_cast<std::uint32_t>(vk::table))
      n.table = reinterpret_cast<std::uint64_t>(blob_base) + n.table;
  }
  return out;
}

} // namespace

BOOST_AUTO_TEST_SUITE(device_views)

BOOST_AUTO_TEST_CASE(two_thousand_random_descriptors_read_on_the_device_as_the_brute_force_reference_says) {
  std::mt19937 rng(20261123);
  population p;
  for (int i = 0; i < 2000; ++i) {
    switch (i % 6) {
    case 0: add_tile_population(p, rng, 1); break;
    case 1: add_tile_population(p, rng, 2 + static_cast<int>(rng() % 2)); break;
    case 2: add_retile_population(p, rng); break;
    case 3: add_extend_population(p, rng); break;
    case 4: add_table_population(p, rng); break;
    default: add_data_index_population(p, rng); break;
    }
  }
  BOOST_REQUIRE_EQUAL(p.queries.size(), p.expected.size());

  // the host evaluator: the same reader on host memory
  const std::vector<view_node> host_nodes = patched(p.nodes, p.blob.data());
  for (std::size_t q = 0; q < p.queries.size(); ++q) {
    const query &k = p.queries[q];
    dev::plain_visitor w;
    std::byte out[16] = {};
    BOOST_REQUIRE(dev::read_view(host_nodes.data() + k.first_node, k.top, k.tile, k.local, nullptr, w, out));
    std::int32_t v;
    std::memcpy(&v, out, 4);
    BOOST_REQUIRE_MESSAGE(v == p.expected[q], "query " << q << ": host evaluator " << v << ", reference " << p.expected[q]);
  }

  // the same questions on the device
  sycl::queue queue(test_device());
  std::byte *blob = sycl::malloc_device<std::byte>(p.blob.size(), queue);
  queue.memcpy(blob, p.blob.data(), p.blob.size()).wait();
  const std::vector<view_node> dev_nodes = patched(p.nodes, blob);
  view_node *nodes = sycl::malloc_device<view_node>(dev_nodes.size(), queue);
  query *queries = sycl::malloc_device<query>(p.queries.size(), queue);
  std::int32_t *answers = sycl::malloc_device<std::int32_t>(p.queries.size(), queue);
  queue.memcpy(nodes, dev_nodes.data(), dev_nodes.size() * sizeof(view_node)).wait();
  queue.memcpy(queries, p.queries.data(), p.queries.size() * sizeof(query)).wait();
  const std::size_t count = p.queries.size();
  queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
    const query &k = queries[i[0]];
    dev::plain_visitor w;
    std::byte out[16] = {};
    std::int32_t v = 0x7fffffff;
    if (dev::read_view(nodes + k.first_node, k.top, k.tile, k.local, nullptr, w, out))
      std::memcpy(&v, out, 4);
    answers[i[0]] = v;
  });
  queue.wait_and_throw();
  std::vector<std::int32_t> got(count);
  queue.memcpy(got.data(), answers, count * 4).wait();
  sycl::free(blob, queue);
  sycl::free(nodes, queue);
  sycl::free(queries, queue);
  sycl::free(answers, queue);
  for (std::size_t q = 0; q < count; ++q)
    BOOST_REQUIRE_MESSAGE(got[q] == p.expected[q], "query " << q << ": device " << got[q] << ", reference " << p.expected[q]);
}

BOOST_AUTO_TEST_CASE(a_gather_reads_a_tile_chosen_by_data_after_its_owner_published) {
  std::vector<int> idx(256);
  for (int i = 0; i < 256; ++i)
    idx[i] = (7 * i) % 256;
  sg::graph g;
  auto s = g.create_mem_handle<int>();
  auto ix = g.create_mem_handle<int>();
  auto y = g.create_mem_handle<int>();
  host_write(g, s, iota_i(256));
  host_write(g, ix, idx);
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k5(kh, s, ix, y); });
  std::vector<int> out;
  host_read(g, y, 256, out);
  g.run();
  g.wait();
  BOOST_CHECK_EQUAL(out[1], 14);
}

BOOST_AUTO_TEST_SUITE_END()
