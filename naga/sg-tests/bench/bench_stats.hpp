#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace sgt::bench {

struct args {
  std::string case_name;
  int warm_runs = 20;
  int warmup_runs = 3;
  std::size_t big_groups = 16384;
  int layers = 30;
};

inline const std::vector<std::string> &case_names() {
  static const std::vector<std::string> names = {"plain_item", "ten_layers", "layers_8_16_32", "chain_30",
                                                 "a_to_f"};
  return names;
}

inline std::optional<long long> parse_number(const std::string &s) {
  if (s.empty())
    return std::nullopt;
  std::size_t used = 0;
  long long v = 0;
  try {
    v = std::stoll(s, &used);
  } catch (...) {
    return std::nullopt;
  }
  if (used != s.size())
    return std::nullopt;
  return v;
}

inline std::optional<args> parse_args(const std::vector<std::string> &argv) {
  args a;
  bool have_case = false;
  for (std::size_t i = 0; i < argv.size(); i += 2) {
    if (i + 1 >= argv.size())
      return std::nullopt;
    const std::string &flag = argv[i], &value = argv[i + 1];
    if (flag == "--case") {
      const auto &names = case_names();
      if (std::find(names.begin(), names.end(), value) == names.end())
        return std::nullopt;
      a.case_name = value;
      have_case = true;
      continue;
    }
    const auto n = parse_number(value);
    if (!n)
      return std::nullopt;
    if (flag == "--warm-runs" && *n >= 1)
      a.warm_runs = static_cast<int>(*n);
    else if (flag == "--warmup-runs" && *n >= 0)
      a.warmup_runs = static_cast<int>(*n);
    else if (flag == "--layers" && *n >= 1)
      a.layers = static_cast<int>(*n);
    else if (flag == "--big-groups" && *n >= 1)
      a.big_groups = static_cast<std::size_t>(*n);
    else
      return std::nullopt;
  }
  if (!have_case)
    return std::nullopt;
  return a;
}

inline double median(std::vector<double> v) {
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

inline bool rejit_detected(const std::vector<double> &warmup_ms, double median_warm_ms) {
  const double limit = std::max(100.0, 10.0 * median_warm_ms);
  return std::any_of(warmup_ms.begin(), warmup_ms.end(), [&](double v) { return v > limit; });
}

struct result {
  std::string case_name;
  bool ok;
  double cold_ms;
  double warm_ms;
  std::optional<double> big_ms;
  bool rejit;
};

inline std::string format_result_line(const result &r) {
  char buf[512];
  char big[64];
  if (r.big_ms)
    std::snprintf(big, sizeof big, "%.3f", *r.big_ms);
  else
    std::snprintf(big, sizeof big, "NA");
  std::snprintf(buf, sizeof buf, "sg_jit_bench case=%s ok=%d cold_ms=%.1f warm_ms=%.3f big_ms=%s rejit=%d",
                r.case_name.c_str(), r.ok ? 1 : 0, r.cold_ms, r.warm_ms, big, r.rejit ? 1 : 0);
  return buf;
}

inline int expected_plain_item(std::size_t i) { return static_cast<int>(3 * i + 1); }

inline int expected_ten_layers(std::size_t i) {
  int v = static_cast<int>(i);
  for (int k = 0; k < 10; ++k)
    v = v * 2 % 1000;
  return v;
}

inline int expected_layers_8_16_32(std::size_t i) { return static_cast<int>(4 * (i % 4) + 56); }

inline int expected_chain_30(std::size_t i) { return static_cast<int>(i + 30); }

struct a_to_f_values {
  int b, c, d, f;
};

inline a_to_f_values expected_a_to_f(std::size_t i) {
  const std::size_t gi = i / 4, l = i % 4;
  return {static_cast<int>(i + 1), static_cast<int>(2 * i), static_cast<int>(3 * i + 1),
          static_cast<int>(2 * (4 * (3 - gi) + l) + 1001)};
}

} // namespace sgt::bench
