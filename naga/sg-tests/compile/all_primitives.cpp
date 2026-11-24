// A program that uses every primitive of the agreed summary and every form of the graph around
// a kernel. It is compiled, not run: acpp --acpp-targets=generic -std=c++20 -c.
#include "hipSYCL/sg/sg.hpp"

#include <cstdint>

namespace {

const auto plus = sycl::plus<>{};
auto sum_items = [](auto item, auto v) { return sg::reduce(item, v, plus); };
auto scan_items = [](auto item, auto v) { return sg::inclusive_scan(item, v, plus); };
auto pass_items = [](auto item, auto v) { return v; };
auto dot_items = [](auto item, auto a, auto b) { return sg::reduce(item, a * b, plus); };
const auto key = [](int v) { return v % 16; };
auto bin_items = [](auto item, auto v) { return sg::tile_bin(item, v, 16, key); };
auto step_fn = [](float v) { return 2 * v + 1; };

// work tiles: source, tile, count, retile, join, extend_tiles, flatten, slice, reduce, scan, bin
void work_tiles_program(sg::kernel_handle &kh, sg::mem_handle<int> x_in, sg::mem_handle<int> y_out,
                        sg::mem_handle<int> s_out) {
  auto x = kh.source(x_in, sg::shape<1>({1024}));
  auto x_tiles = kh.tile(x, {64}, sg::fill{0});
  auto count = x_tiles.count();
  auto work = kh.work_tiles({count}, {1}, 64);
  auto scanned = kh.map_work(work, [=](auto unit) {
    auto tile = x_tiles[unit.tile()];
    auto run = sg::flatten(tile);
    return unit.run(run, scan_items);
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
    return unit.run(scan_run, inc_run, [](auto item, int a, int b) { return a + b; });
  });
  auto y_rows = kh.retile(y, {64});
  auto y_elems = kh.join(y_rows);
  kh.evaluate(y_out, y_elems);

  auto v_tiles = kh.tile(x, {256});
  auto bin_work = kh.work_tiles({4}, {1}, 256);
  auto bins = kh.map_work(bin_work, [=](auto unit) {
    auto v_tile = v_tiles[unit.tile()];
    auto v_run = sg::flatten(v_tile);
    return unit.run(v_run, bin_items);
  });
  kh.evaluate(s_out, kh.join(kh.retile(bins, {16})));
}

// stencil, define_tiles, index_map, slice_work, mark_local, repeat
void structure_program(sg::kernel_handle &kh, sg::mem_handle<float> a_in, sg::mem_handle<float> b_out,
                       sg::mem_handle<float> c_out, sg::mem_handle<float> total_out,
                       sg::mem_handle<int> n_handle) {
  auto n = kh.launch_value(n_handle);
  auto a = kh.source(a_in, sg::shape<1>({1024}));
  auto star = kh.stencil({-1, 0, 1}, sg::range(0, 1024));
  auto windows = kh.index_map(star, sg::wrap{}, a);
  auto work = kh.work_tiles({16}, {64}, 3);
  auto b = kh.map_work(work, [=](auto unit) {
    auto t = unit.tile();
    auto u = unit.id();
    auto i = 64 * t[0] + u[0];
    auto window = windows[i];
    auto run = sg::flatten(window);
    return unit.run(run, sum_items);
  });
  kh.evaluate(b_out, kh.join(b));

  auto x_tiles = kh.tile(a, {4});
  auto work1 = kh.work_tiles({n / 4}, {1}, 4);
  auto first = kh.map_work(work1, [=](auto unit) {
    auto tile = x_tiles[unit.tile()];
    auto run = sg::flatten(tile);
    return unit.run(run, sum_items);
  });
  kh.mark_local(first);
  auto work2 = kh.slice_work(work1, {3}, {4}, {4}, {1}, 4);
  auto l1_tiles = kh.retile(first, {4});
  auto second = kh.map_work(work2, [=](auto unit) {
    auto tile = l1_tiles[unit.tile()];
    auto run = sg::flatten(tile);
    return unit.run(run, sum_items);
  });
  kh.evaluate(c_out, kh.join(second));
  auto last = kh.repeat(n - 1, first, [=](auto &kh2, auto prev) {
    auto prev_tiles = kh2.retile(prev, {4});
    auto level = kh2.work_tiles({1}, {1}, 4);
    return kh2.map_work(level, [=](auto unit) {
      auto tile = prev_tiles[unit.tile()];
      auto run = sg::flatten(tile);
      return unit.run(run, sum_items);
    });
  });
  kh.evaluate(total_out, kh.join(last));
}

// segmented work sets: segment_items, run_distributed, run_replicated, mark_local in its other forms
void segmented_program(sg::kernel_handle &kh, sg::mem_handle<int> a_in, sg::mem_handle<int> b_in,
                       sg::mem_handle<int> c_out, sg::mem_handle<int> r_out) {
  auto A = kh.source(a_in, sg::shape<2>({128, 128}));
  auto B = kh.source(b_in, sg::shape<2>({128, 128}));
  auto a_bands = kh.tile(A, {16, 128});
  auto b_bands = kh.tile(B, {128, 16});
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
  kh.mark_local<sg::segmented>(C_tiles);
  kh.evaluate(c_out, kh.join(C_tiles));

  auto rows = kh.tile(A, {1, 128});
  auto row_run = sg::segment_items(4, 32);
  auto row_work = kh.work_tiles({128}, {1}, row_run);
  auto sums = kh.map_work(row_work, [=](auto unit) {
    auto t = unit.tile();
    auto row = rows[{t[0], 0}];
    auto r = sg::flatten(row);
    return unit.run_replicated(r, sum_items);
  });
  kh.evaluate(r_out, kh.join(sums));
}

// work items: work_items, map_items, item.id, item.compute, inplace
void items_program(sg::kernel_handle &kh, sg::mem_handle<float> x_in, sg::mem_handle<float> y_out,
                   sg::mem_handle<float> acc) {
  auto x = kh.source(x_in, sg::shape<1>({64}));
  auto items = kh.work_items({64});
  auto y = kh.map_items(items, [=](auto item) {
    auto v = x[item.id()];
    return item.compute(v, step_fn);
  });
  auto prev = y.named("prev");
  auto z = kh.map_items(items, [=](auto item) {
    auto v = prev[item.id()];
    return item.compute(v, step_fn);
  });
  kh.evaluate(y_out, z);
  kh.evaluate(sg::inplace(acc, sycl::plus<float>{}, 0.f), y);
}

} // namespace

// the graph around a kernel: sg::graph, create_mem_handle, submit (kernel, sg::host, sg::interop), run
void all_forms() {
  sg::graph_options options;
  options.group_size_cap = 1024;
  sg::graph g(options);
  auto x_handle = g.create_mem_handle<float>();
  auto n_handle = g.create_mem_handle<int>();
  auto y_handle = g.create_mem_handle<float>();
  auto ints_in = g.create_mem_handle<int>();
  auto ints_out = g.create_mem_handle<int>();
  auto bins_out = g.create_mem_handle<int>();
  auto b_out = g.create_mem_handle<float>();
  auto c_out = g.create_mem_handle<float>();
  auto total_out = g.create_mem_handle<float>();
  auto acc = g.create_mem_handle<float>();
  auto a2 = g.create_mem_handle<int>();
  auto b2 = g.create_mem_handle<int>();
  auto c2 = g.create_mem_handle<int>();
  auto r2 = g.create_mem_handle<int>();
  sycl::device dev{sycl::default_selector_v};

  g.submit<sg::host>([&](sg::host_handle &hh) {
    auto n = hh.write(n_handle, sg::shape<1>({1}));
    hh.run([n] { n.data()[0] = 64; });
  });

  g.submit(dev, [&](sg::kernel_handle &kh) {
    sg::launch_value<int> n = kh.launch_value(n_handle);
    auto x = kh.source(x_handle, sg::shape<1>({n}));
    auto items = kh.work_items({n});
    auto y = kh.map_items(items, [=](auto item) {
      auto v = x[item.id()];
      return item.compute(v, [](float v) { return 2 * v + 1; });
    });
    kh.evaluate(y_handle, y);
  });
  g.submit(dev, [&](sg::kernel_handle &kh) { work_tiles_program(kh, ints_in, ints_out, bins_out); });
  g.submit(dev, [&](sg::kernel_handle &kh) {
    structure_program(kh, x_handle, b_out, c_out, total_out, n_handle);
  });
  g.submit(dev, [&](sg::kernel_handle &kh) { segmented_program(kh, a2, b2, c2, r2); });
  g.submit(dev, [&](sg::kernel_handle &kh) { items_program(kh, x_handle, y_handle, acc); });

  try {
    g.submit<sg::interop>(dev, 0);
  } catch (const sg::error &) {
  }

  g.run();
  g.wait();
  std::uint64_t compiled = sg::jit_compile_count();
  (void)compiled;
}

int main() {
  (void)&all_forms;
  return 0;
}
