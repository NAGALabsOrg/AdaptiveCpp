// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_COPY_PLAN_HPP
#define HIPSYCL_SG_DETAIL_COPY_PLAN_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sg::detail {

struct place {
  int device = -1; // -1 = host
  std::uint64_t context = 0;
  bool operator==(const place &) const = default;
};
struct interval {
  std::size_t begin = 0, end = 0;
  bool empty() const noexcept { return end <= begin; }
};
enum class copy_kind { none, direct, staged_via_host };
struct copy_step {
  place from, to;
  interval bytes;
  int exec_device = -1;
};
struct copy_plan {
  copy_kind kind = copy_kind::none;
  std::vector<copy_step> steps;
};

inline copy_plan plan_copy(place from, place to, interval bytes) {
  copy_plan p;
  if (bytes.empty() || from == to)
    return p;
  if (from.device < 0 || to.device < 0 || from.context == to.context) {
    p.kind = copy_kind::direct;
    p.steps.push_back({from, to, bytes, from.device < 0 ? to.device : from.device});
    return p;
  }
  p.kind = copy_kind::staged_via_host;
  const place host{};
  p.steps.push_back({from, host, bytes, from.device});
  p.steps.push_back({host, to, bytes, to.device});
  return p;
}

} // namespace sg::detail

#endif
