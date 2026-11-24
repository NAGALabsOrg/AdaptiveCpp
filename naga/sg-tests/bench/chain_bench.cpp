// sg_chain_bench --n <layers>: runs chain(n) of the chain suite once, cold, and prints the time of
// that first run including its JIT compilation.
#include "../common/device_report.hpp"
#include "../common/graphs.hpp"
#include "bench_stats.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  const auto n = sgt::bench::parse_layers(args);
  if (!n) {
    std::fprintf(stderr, "usage: sg_chain_bench --n <1..30>\n");
    return 2;
  }
  sgt::print_cell_line(true);
  const auto x = sgt::iota_i(64);
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<int> got;
  try {
    got = sgt::run_kernel<int, int>(x, 64, [n](auto &kh, auto xh, auto yh) { sgt::chain(kh, *n, xh, yh); });
  } catch (const std::exception &e) {
    std::fprintf(stderr, "sg_chain_bench: %s\n", e.what());
    std::printf("%s\n", sgt::bench::format_result_line(*n, false, 0.0).c_str());
    return 1;
  }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  const bool ok = got == sgt::chain_reference(*n, x);
  std::printf("%s\n", sgt::bench::format_result_line(*n, ok, ms).c_str());
  return ok ? 0 : 1;
}
