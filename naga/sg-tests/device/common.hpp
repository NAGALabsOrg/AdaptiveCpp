#pragma once
#include "../common/check.hpp"

#include "hipSYCL/sg/sg.hpp"

#include <boost/test/unit_test.hpp>

#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

namespace sgt {

inline sycl::device dev() { return sycl::device{sycl::default_selector_v}; }

inline sg::graph_options opts(std::size_t mib = 64) {
  sg::graph_options o;
  o.pool_reservation_bytes = mib << 20;
  return o;
}

template <class T, int N>
void write_host(sg::graph &g, const sycl::device &d, sg::mem_handle<T> h, sycl::range<N> r,
                std::vector<T> data) {
  g.submit(d, [&](sg::submit_handle gh) {
    sg::host_array<T, N> a(h, sg::shape<N>(r));
    gh.define_host_work([a, data, r] {
      for (std::size_t i = 0; i < r.size(); ++i)
        a.data()[i] = data[i];
    });
  });
}

template <class T, int N>
std::vector<T> read_host(sg::graph &g, const sycl::device &d, sg::mem_handle<T> h, sycl::range<N> r) {
  auto out = std::make_shared<std::vector<T>>(r.size());
  g.submit(d, [&](sg::submit_handle gh) {
    sg::host_array<const T, N> a(h, sg::shape<N>(r));
    gh.define_host_work([a, out, r] {
      for (std::size_t i = 0; i < r.size(); ++i)
        (*out)[i] = a.data()[i];
    });
  });
  g.run();
  g.wait();
  return *out;
}

// One result of element type T over `total`, produced by whatever `build(kh)` defines.
template <class T, int N, class Build>
std::vector<T> run_one(sycl::range<N> total, Build build) {
  sg::graph g(opts());
  auto h = g.create_mem_handle<T>();
  g.submit(dev(), [&](sg::submit_handle gh) {
    sg::array<sg::result<T>, N> out(h, sg::shape<N>(total));
    gh.define_kernel([out, build](sg::kernel_handle &kh) { out = kh.evaluate(build(kh)); });
  });
  g.run();
  return read_host(g, dev(), h, total);
}

template <class T> std::vector<T> iota_vec(std::size_t n, T start = T{}) {
  std::vector<T> v(n);
  std::iota(v.begin(), v.end(), start);
  return v;
}

inline std::vector<std::pair<sycl::device, sycl::device>> device_pairs() {
  std::vector<std::pair<sycl::device, sycl::device>> pairs;
  const auto all = sycl::device::get_devices();
  for (const auto &a : all)
    for (const auto &b : all)
      if (!(a == b))
        pairs.emplace_back(a, b);
  return pairs;
}

inline std::optional<std::pair<sycl::device, sycl::device>> two_devices() {
  const auto pairs = device_pairs();
  if (pairs.empty())
    return std::nullopt;
  return pairs.front();
}

} // namespace sgt
