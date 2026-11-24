#pragma once

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace sgt::bench {

// `--n <layers>`: the length of the chain.
inline std::optional<int> parse_layers(const std::vector<std::string> &argv) {
  if (argv.size() != 2 || argv[0] != "--n")
    return std::nullopt;
  std::size_t used = 0;
  int v = 0;
  try {
    v = std::stoi(argv[1], &used);
  } catch (...) {
    return std::nullopt;
  }
  if (used != argv[1].size() || v < 1 || v > 30)
    return std::nullopt;
  return v;
}

inline std::string format_result_line(int n, bool ok, double cold_ms) {
  char buf[256];
  std::snprintf(buf, sizeof buf, "sg_chain_bench n=%d ok=%d cold_ms=%.1f", n, ok ? 1 : 0, cold_ms);
  return buf;
}

} // namespace sgt::bench
