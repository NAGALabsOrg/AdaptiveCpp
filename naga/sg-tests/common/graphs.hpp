// The kernels K1..K14 of docs/naga-internal/brainstorm/language-pass/kernels.md as
// builders, with their inputs and sequential references, plus the chain, mixed-width
// and rerun programs. Shared by the host cells (structure) and the device cells (results).
#pragma once
#include "hipSYCL/sg/sg.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace sgt {

template <class T> using H = sg::mem_handle<T>;

// ---- the common definitions ----
inline auto sum_items = [](auto item, auto v) { return sg::reduce(item, v, sycl::plus<>{}); };
inline auto scan_items = [](auto item, auto v) { return sg::inclusive_scan(item, v, sycl::plus<>{}); };
inline auto pass_items = [](auto, auto v) { return v; };
inline auto add_items = [](auto, auto a, auto b) { return a + b; };
inline auto dot_items = [](auto item, auto a, auto b) { return sg::reduce(item, a * b, sycl::plus<>{}); };

template <class Kh, class W, class T> auto sum_tiles(Kh &kh, const W &work, const sg::tiles<T> &ts) {
  return kh.map_work(work, [=](auto unit) {
    auto tile = ts[unit.tile()];
    auto run = sg::flatten(tile);
    return unit.run(run, sum_items);
  });
}

// ---- K1 global reduction, 64 in fours ----
template <class Kh> void k1(Kh &kh, H<float> x_in, H<float> total) {
  auto x = kh.source(x_in, sg::shape<1>({64}));
  auto x_tiles = kh.tile(x, {4});
  auto work1 = kh.work_tiles({16}, {1}, 4);
  auto l1 = sum_tiles(kh, work1, x_tiles);
  auto l1_tiles = kh.retile(l1, {4});
  auto work2 = kh.slice_work(work1, {3}, {4}, {4}, {1}, 4);
  auto l2 = sum_tiles(kh, work2, l1_tiles);
  auto l2_tiles = kh.retile(l2, {4});
  auto work3 = kh.slice_work(work1, {15}, {1}, {1}, {1}, 4);
  auto l3 = sum_tiles(kh, work3, l2_tiles);
  auto total_elems = kh.join(l3);
  kh.evaluate(total, total_elems);
}

// ---- K2 n-level reduction, depth a launch value ----
template <class Kh> void k2(Kh &kh, H<float> x_in, H<int> n_in, H<int> depth_in, H<float> total) {
  auto n = kh.launch_value(n_in);
  auto d = kh.launch_value(depth_in);
  auto x = kh.source(x_in, sg::shape<1>({n}));
  auto x_tiles = kh.tile(x, {4});
  auto count1 = x_tiles.count();
  auto work1 = kh.work_tiles({count1}, {1}, 4);
  auto first = sum_tiles(kh, work1, x_tiles);
  auto last = kh.repeat(d - 1, first, [=](auto &kh2, auto prev) {
    auto prev_tiles = kh2.retile(prev, {4});
    auto count = prev_tiles.count();
    auto stride = count1 / count;
    auto work = kh2.slice_work(work1, {stride - 1}, {count}, {stride}, {1}, 4);
    return sum_tiles(kh2, work, prev_tiles);
  });
  auto total_elems = kh.join(last);
  kh.evaluate(total, total_elems);
}

// ---- K3 whole-array inclusive scan, 1024 values, tiles of 64 ----
template <class Kh> void k3(Kh &kh, H<int> x_in, H<int> y_out) {
  auto x = kh.source(x_in, sg::shape<1>({1024}));
  auto x_tiles = kh.tile(x, {64});
  auto work = kh.work_tiles({16}, {1}, 64);
  auto scanned = kh.map_work(work, [=](auto unit) {
    auto x_tile = x_tiles[unit.tile()];
    auto x_run = sg::flatten(x_tile);
    return unit.run(x_run, scan_items);
  });
  auto totals = kh.map_work(work, [=](auto unit) {
    auto my_scan = scanned[unit.tile()];
    auto my_last = sg::slice(my_scan, {0, 63}, {1, 1});
    auto last_run = sg::flatten(my_last);
    return unit.run(last_run, pass_items);
  });
  auto totals_rows = kh.retile(totals, {1});
  auto totals_elems = kh.join(totals_rows);
  auto before_map = kh.define_tiles(16, {64}, [](auto t, auto s) { return s < t ? s : -1; });
  auto before = kh.index_map(before_map, sg::fill{0}, totals_elems);
  auto increments = kh.map_work(work, [=](auto unit) {
    auto my_before = before[unit.tile()];
    auto before_run = sg::flatten(my_before);
    return unit.run(before_run, sum_items);
  });
  auto inc_wide = kh.extend_tiles(increments, {1, 64}, sg::repeat);
  auto y = kh.map_work(work, [=](auto unit) {
    auto my_scan = scanned[unit.tile()];
    auto my_inc = inc_wide[unit.tile()];
    auto scan_run = sg::flatten(my_scan);
    auto inc_run = sg::flatten(my_inc);
    return unit.run(scan_run, inc_run, add_items);
  });
  auto y_rows = kh.retile(y, {64});
  auto y_elems = kh.join(y_rows);
  kh.evaluate(y_out, y_elems);
}

// ---- K4 1-D 3-point stencil, 1024 values ----
template <class Kh> void k4(Kh &kh, H<int> x_in, H<int> y_out) {
  auto x = kh.source(x_in, sg::shape<1>({1024}));
  auto star = kh.stencil({-1, 0, 1}, sg::range(0, 1024));
  auto windows = kh.index_map(star, sg::fill{0}, x);
  auto work = kh.work_tiles({16}, {64}, 3);
  auto y = kh.map_work(work, [=](auto unit) {
    auto t = unit.tile();
    auto u = unit.id();
    auto i = 64 * t[0] + u[0];
    auto window = windows[i];
    auto run = sg::flatten(window);
    return unit.run(run, sum_items);
  });
  auto y_elems = kh.join(y);
  kh.evaluate(y_out, y_elems);
}

// ---- K5 gather from values computed in the kernel; the gather may be taken on the producing set ----
inline auto double_items = [](auto, auto v) { return 2 * v; };
template <class Kh> void k5(Kh &kh, H<int> s_in, H<int> idx_in, H<int> y_out, bool gather_on_own_set = false) {
  auto s = kh.source(s_in, sg::shape<1>({256}));
  auto s_tiles = kh.tile(s, {64});
  auto work_src = kh.work_tiles({4}, {1}, 64);
  auto src = kh.map_work(work_src, [=](auto unit) {
    auto s_tile = s_tiles[unit.tile()];
    auto s_run = sg::flatten(s_tile);
    return unit.run(s_run, double_items);
  });
  auto src_rows = kh.retile(src, {64});
  auto src_elems = kh.join(src_rows);
  auto idx = kh.source(idx_in, sg::shape<1>({256}));
  auto idx_tiles = kh.tile(idx, {64});
  auto gathered = kh.index_map(idx_tiles, sg::fill{0}, src_elems);
  auto work_y = gather_on_own_set ? work_src : kh.work_tiles({4}, {1}, 64);
  auto y = kh.map_work(work_y, [=](auto unit) {
    auto g_tile = gathered[unit.tile()];
    auto g_run = sg::flatten(g_tile);
    return unit.run(g_run, pass_items);
  });
  auto y_rows = kh.retile(y, {64});
  auto y_elems = kh.join(y_rows);
  kh.evaluate(y_out, y_elems);
}

// ---- K6 matrix product, 128 x 128: the cap-1024 form and the segmented form ----
template <class Kh> void k6(Kh &kh, H<int> a_in, H<int> b_in, H<int> c_out, bool segmented) {
  auto A = kh.source(a_in, sg::shape<2>({128, 128}));
  auto B = kh.source(b_in, sg::shape<2>({128, 128}));
  auto a_bands = kh.tile(A, {16, 128});
  auto b_bands = kh.tile(B, {128, 16});
  if (!segmented) {
    auto work = kh.work_tiles({64, 32}, {2, 4}, 128);
    kh.mark_local(a_bands, work);
    kh.mark_local(b_bands, work);
    auto C_tiles = kh.map_work(work, [=](auto unit) {
      auto t = unit.tile();
      auto u = unit.id();
      auto row = 2 * t[0] + u[0];
      auto col = 4 * t[1] + u[1];
      auto a_band = a_bands[{row / 16, 0}];
      auto b_band = b_bands[{0, col / 16}];
      auto a_row = sg::slice(a_band, {row % 16, 0}, {1, 128});
      auto b_col = sg::slice(b_band, {0, col % 16}, {128, 1});
      auto a_run = sg::flatten(a_row);
      auto b_run = sg::flatten(b_col);
      return unit.run(a_run, b_run, dot_items);
    });
    auto C_elems = kh.join(C_tiles);
    kh.evaluate(c_out, C_elems);
  } else {
    auto run = sg::segment_items(32, 4);
    auto work = kh.work_tiles({8, 8}, {16, 16}, run);
    kh.mark_local(a_bands, work);
    kh.mark_local(b_bands, work);
    auto C_tiles = kh.map_work(work, [=](auto unit) {
      auto t = unit.tile();
      auto u = unit.id();
      auto a_band = a_bands[{t[0], 0}];
      auto b_band = b_bands[{0, t[1]}];
      auto a_row = sg::slice(a_band, {u[0], 0}, {1, 128});
      auto b_col = sg::slice(b_band, {0, u[1]}, {128, 1});
      auto a_run = sg::flatten(a_row);
      auto b_run = sg::flatten(b_col);
      return unit.run_distributed(a_run, b_run, dot_items);
    });
    auto C_elems = kh.join(C_tiles);
    kh.evaluate(c_out, C_elems);
  }
}

// ---- K7 histogram, 4096 values, 16 bins ----
inline constexpr auto bin_key = [](int v) { return v % 16; };
inline constexpr auto bin_items = [](auto item, auto v) { return sg::tile_bin(item, v, 16, bin_key); };
template <class Kh> void k7(Kh &kh, H<int> v_in, H<int> h_out) {
  auto v = kh.source(v_in, sg::shape<1>({4096}));
  auto v_tiles = kh.tile(v, {256});
  auto work = kh.work_tiles({16}, {1}, 256);
  auto partial = kh.map_work(work, [=](auto unit) {
    auto v_tile = v_tiles[unit.tile()];
    auto v_run = sg::flatten(v_tile);
    return unit.run(v_run, bin_items);
  });
  auto partial_rows = kh.retile(partial, {16});
  auto partial_elems = kh.join(partial_rows);
  auto by_bin_map = kh.define_tiles(16, {16}, [](auto b, auto p) { return 16 * p + b; });
  auto by_bin = kh.index_map(by_bin_map, sg::fill{0}, partial_elems);
  auto last_tile = kh.slice_work(work, {15}, {1}, {1}, {16}, 16);
  auto hist = kh.map_work(last_tile, [=](auto unit) {
    auto u = unit.id();
    auto b_tile = by_bin[u[0]];
    auto b_run = sg::flatten(b_tile);
    return unit.run(b_run, sum_items);
  });
  auto hist_elems = kh.join(hist);
  kh.evaluate(h_out, hist_elems);
}

// ---- K8 A-to-F, 16 values in tiles of 4 ----
inline auto b_items = [](auto, auto v) { return v + 1; };
inline auto c_items = [](auto, auto v) { return 2 * v; };
inline auto e_items = [](auto, auto v) { return v + 1001; };
template <class Kh, class W, class T, class F> auto per_tile(Kh &kh, const W &work, const sg::tiles<T> &ts, F fn) {
  return kh.map_work(work, [=](auto unit) {
    auto tile = ts[unit.tile()];
    auto run = sg::flatten(tile);
    return unit.run(run, fn);
  });
}
template <class Kh> void k8(Kh &kh, H<int> x_in, H<int> b_out, H<int> c_out, H<int> d_out, H<int> f_out) {
  auto x = kh.source(x_in, sg::shape<1>({16}));
  auto x_tiles = kh.tile(x, {4});
  auto w = kh.work_tiles({4}, {1}, 4);
  auto a = per_tile(kh, w, x_tiles, pass_items);
  auto b = per_tile(kh, w, a, b_items);
  auto c = per_tile(kh, w, a, c_items);
  auto d = kh.map_work(w, [=](auto unit) {
    auto my_b = b[unit.tile()];
    auto my_c = c[unit.tile()];
    auto b_run = sg::flatten(my_b);
    auto c_run = sg::flatten(my_c);
    return unit.run(b_run, c_run, add_items);
  });
  auto e = per_tile(kh, w, c, e_items);
  auto w_f = kh.work_tiles({4}, {1}, 4);
  auto f = kh.map_work(w_f, [=](auto unit) {
    auto t = unit.tile();
    auto e_rev = e[{3 - t[0]}];
    auto run = sg::flatten(e_rev);
    return unit.run(run, pass_items);
  });
  auto b_elems = kh.join(kh.retile(b, {4}));
  kh.evaluate(b_out, b_elems);
  auto c_elems = kh.join(kh.retile(c, {4}));
  kh.evaluate(c_out, c_elems);
  auto d_elems = kh.join(kh.retile(d, {4}));
  kh.evaluate(d_out, d_elems);
  auto f_elems = kh.join(kh.retile(f, {4}));
  kh.evaluate(f_out, f_elems);
}

// ---- K9 a host loop of element maps ----
inline auto step_fn = [](float v) { return 2 * v + 1; };
template <class Kh> void k9(Kh &kh, H<float> x_in, H<float> y_out, int steps = 4, int n = 64) {
  auto x = kh.source(x_in, sg::shape<1>({n}));
  auto items = kh.work_items({n});
  auto y = kh.map_items(items, [=](auto item) {
    auto v = x[item.id()];
    return item.compute(v, step_fn);
  });
  for (int k = 1; k < steps; ++k) {
    auto prev = y.named("prev");
    y = kh.map_items(items, [=](auto item) {
      auto v = prev[item.id()];
      return item.compute(v, step_fn);
    });
  }
  kh.evaluate(y_out, y);
}

// ---- K10 run-time-declared component: y = a * x + b over n items, or tiled ----
template <class Kh> void k10(Kh &kh, H<float> x_in, H<float> y_out, float a, float b, int n) {
  auto axb_fn = [=](float v) { return a * v + b; };
  auto x = kh.source(x_in, sg::shape<1>({n}));
  auto items = kh.work_items({n});
  auto y = kh.map_items(items, [=](auto item) {
    auto v = x[item.id()];
    return item.compute(v, axb_fn);
  });
  kh.evaluate(y_out, y);
}
template <class Kh> void k10_tiled(Kh &kh, H<float> x_in, H<float> y_out, float a, float b, int n) {
  auto axb = [=](auto, auto v) { return a * v + b; };
  auto x = kh.source(x_in, sg::shape<1>({n}));
  auto x_tiles = kh.tile(x, {256}, sg::fill{0.f});
  auto work = kh.work_tiles({(n + 255) / 256}, {1}, 256);
  auto y = kh.map_work(work, [=](auto unit) {
    auto tile = x_tiles[unit.tile()];
    auto run = sg::flatten(tile);
    return unit.run(run, axb);
  });
  auto y_rows = kh.retile(y, {256});
  auto y_elems = kh.join(y_rows);
  kh.evaluate(y_out, y_elems);
}

// ---- K11 one in-place contribution of k * ones ----
template <class Kh> void k11(Kh &kh, H<float> ones_in, H<float> y_inplace, float k) {
  auto scale_fn = [=](float v) { return k * v; };
  auto ones = kh.source(ones_in, sg::shape<1>({64}));
  auto items = kh.work_items({64});
  auto c = kh.map_items(items, [=](auto item) {
    auto v = ones[item.id()];
    return item.compute(v, scale_fn);
  });
  kh.evaluate(sg::inplace(y_inplace, sycl::plus<float>{}, 0.f), c);
}

// ---- K12 interpolation ----
template <class Kh> void k12(Kh &kh, H<float> src_in, H<int> nbr_in, H<float> w_in, H<float> y_out) {
  auto src = kh.source(src_in, sg::shape<1>({64}));
  auto nbr = kh.source(nbr_in, sg::shape<1>({1024}));
  auto wts = kh.source(w_in, sg::shape<1>({1024}));
  auto nbr_tiles = kh.tile(nbr, {4});
  auto vals = kh.index_map(nbr_tiles, sg::fill{0.f}, src);
  auto w_tiles = kh.tile(wts, {4});
  auto work = kh.work_tiles({4}, {64}, 4);
  auto y = kh.map_work(work, [=](auto unit) {
    auto t = unit.tile();
    auto u = unit.id();
    auto i = 64 * t[0] + u[0];
    auto v_t = vals[i];
    auto w_t = w_tiles[i];
    auto v_run = sg::flatten(v_t);
    auto w_run = sg::flatten(w_t);
    return unit.run(v_run, w_run, dot_items);
  });
  auto y_elems = kh.join(y);
  kh.evaluate(y_out, y_elems);
}

// ---- K13 2-D 5-point stencil, 64 x 64 ----
template <class Kh> void k13(Kh &kh, H<float> x_in, H<float> y_out) {
  auto x = kh.source(x_in, sg::shape<2>({64, 64}));
  auto star = kh.stencil({{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}}, sg::range({0, 0}, {64, 64}));
  auto windows = kh.index_map(star, sg::fill{0.f}, x);
  auto work = kh.work_tiles({8, 4}, {8, 16}, 5);
  auto y = kh.map_work(work, [=](auto unit) {
    auto t = unit.tile();
    auto u = unit.id();
    auto r = 8 * t[0] + u[0];
    auto c = 16 * t[1] + u[1];
    auto window = windows[{r, c}];
    auto run = sg::flatten(window);
    return unit.run(run, sum_items);
  });
  auto y_elems = kh.join(y);
  kh.evaluate(y_out, y_elems);
}

// ---- K14 row sums, 64 rows x 1024 columns: whole groups, or segmented (cap 256) ----
template <class Kh> void k14(Kh &kh, H<int> x_in, H<int> y_out, bool segmented, bool plain_mark = false,
                            bool replicated = false) {
  auto x = kh.source(x_in, sg::shape<2>({64, 1024}));
  auto rows = kh.tile(x, {1, 1024});
  if (!segmented) {
    auto work = kh.work_tiles({64}, {1}, 1024);
    auto sums = kh.map_work(work, [=](auto unit) {
      auto t = unit.tile();
      auto row = rows[{t[0], 0}];
      auto run = sg::flatten(row);
      return unit.run(run, sum_items);
    });
    auto sums_elems = kh.join(sums);
    kh.evaluate(y_out, sums_elems);
  } else {
    auto run = sg::segment_items(4, 256);
    auto work = kh.work_tiles({64}, {1}, run);
    auto sums = kh.map_work(work, [=](auto unit) {
      auto t = unit.tile();
      auto row = rows[{t[0], 0}];
      auto r = sg::flatten(row);
      return unit.run_distributed(r, sum_items);
    });
    if (plain_mark)
      kh.mark_local(sums);
    if (replicated) {
      auto again = kh.map_work(work, [=](auto unit) {
        auto t = unit.tile();
        auto s = sums[t[0]];
        auto r = sg::flatten(s);
        return unit.run_replicated(r, sum_items);
      });
      (void)again;
    }
    auto sums_elems = kh.join(sums);
    kh.evaluate(y_out, sums_elems);
  }
}

// ---- the chain: N work sets over 64 elements in tiles of 4 ----
template <int K> struct chain_add {
  template <class I, class V> V operator()(I, V v) const { return v + static_cast<V>(K); }
};

template <int K, int Max, class Kh, class T>
auto chain_layers(Kh &kh, int n, const sg::tiles<T> &prev, bool first) {
  if constexpr (K > Max) {
    return prev;
  } else {
    if (K > n)
      return prev;
    auto work = kh.work_tiles({16}, {1}, 4);
    auto layer = kh.map_work(work, [=](auto unit) {
      auto t = unit.tile();
      auto src = first ? prev[t[0]] : prev[15 - t[0]];
      auto run = sg::flatten(src);
      return unit.run(run, chain_add<K>{});
    });
    return chain_layers<K + 1, Max>(kh, n, static_cast<const sg::tiles<T> &>(layer), false);
  }
}

template <class Kh> void chain(Kh &kh, int n, H<int> x_in, H<int> y_out) {
  auto x = kh.source(x_in, sg::shape<1>({64}));
  auto x_tiles = kh.tile(x, {4});
  auto last = chain_layers<1, 30>(kh, n, x_tiles, true);
  auto rows = kh.retile(last, {4});
  auto y_elems = kh.join(rows);
  kh.evaluate(y_out, y_elems);
}

inline std::vector<int> chain_reference(int n, const std::vector<int> &x) {
  std::vector<int> y(64);
  const int s = n * (n + 1) / 2;
  for (int i = 0; i < 64; ++i) {
    const int src = (n % 2 == 1) ? i : 4 * (15 - i / 4) + i % 4;
    y[i] = x[src] + s;
  }
  return y;
}

// ---- the graph of K9's kind with a launch value: y = 2x + 1 over n items ----
template <class Kh> void rerun_a(Kh &kh, H<float> x_in, H<int> n_in, H<float> y_out) {
  auto n = kh.launch_value(n_in);
  auto x = kh.source(x_in, sg::shape<1>({n}));
  auto items = kh.work_items({n});
  auto y = kh.map_items(items, [=](auto item) {
    auto v = x[item.id()];
    return item.compute(v, step_fn);
  });
  kh.evaluate(y_out, y);
}

// ---- host nodes that put data into a handle and take it out ----
template <class T> void host_write(sg::graph &g, const H<T> &h, std::vector<T> data) {
  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto a = hh.write(h, sg::shape<1>({data.size()}));
    hh.run([a, d = data] {
      for (std::size_t i = 0; i < d.size(); ++i)
        a.data()[i] = d[i];
    });
  });
}

// `out` must live until the graph has been waited on
template <class T> void host_read(sg::graph &g, const H<T> &h, std::size_t n, std::vector<T> &out) {
  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto a = hh.read(h, sg::shape<1>({n}));
    hh.run([a, n, &out] {
      out.assign(a.data(), a.data() + n);
    });
  });
}

// The device of a cell: the default selector honours ACPP_VISIBILITY_MASK.
inline sycl::device test_device() { return sycl::queue{}.get_device(); }

// One input handle, one output handle, one kernel; returns the output.
template <class Out, class In, class Build>
std::vector<Out> run_kernel(const std::vector<In> &in, std::size_t out_n, Build build, sg::graph_options o = {}) {
  sg::graph g(o);
  auto x = g.create_mem_handle<In>();
  auto y = g.create_mem_handle<Out>();
  host_write(g, x, in);
  g.submit(test_device(), [&](sg::kernel_handle &kh) { build(kh, x, y); });
  std::vector<Out> out;
  host_read(g, y, out_n, out);
  g.run();
  g.wait();
  return out;
}

// the inputs and references of the suite
inline std::vector<float> iota_f(int n, float start = 0.f) {
  std::vector<float> v(n);
  for (int i = 0; i < n; ++i)
    v[i] = start + static_cast<float>(i);
  return v;
}
inline std::vector<int> iota_i(int n, int start = 0) {
  std::vector<int> v(n);
  for (int i = 0; i < n; ++i)
    v[i] = start + i;
  return v;
}

} // namespace sgt
