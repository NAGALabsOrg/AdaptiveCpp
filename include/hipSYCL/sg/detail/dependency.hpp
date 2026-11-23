// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_DEPENDENCY_HPP
#define HIPSYCL_SG_DETAIL_DEPENDENCY_HPP

#include "hipSYCL/sg/detail/copy_plan.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sg::detail {

using op_id = std::uint64_t;

enum class access_mode { read, write, inplace };

struct access {
  std::size_t handle = 0;
  interval bytes;
  access_mode mode = access_mode::read;
  int group = -1;
};

class dependency_tracker {
public:
  std::vector<op_id> resolve(const std::vector<access> &accesses) const {
    std::vector<op_id> out;
    for (const access &a : accesses) {
      if (a.bytes.empty())
        continue;
      auto it = records_.find(a.handle);
      if (it == records_.end())
        continue;
      for (const record &r : it->second)
        if (conflicts(a, r))
          out.push_back(r.op);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
  }

  void commit(op_id op, const std::vector<access> &accesses) {
    for (const access &a : accesses) {
      if (a.bytes.empty())
        continue;
      auto &list = records_[a.handle];
      if (a.mode == access_mode::write)
        list.erase(std::remove_if(list.begin(), list.end(),
                                  [&](const record &r) {
                                    return r.bytes.begin >= a.bytes.begin &&
                                           r.bytes.end <= a.bytes.end;
                                  }),
                   list.end());
      list.push_back({op, a.bytes, a.mode, a.group});
    }
  }

private:
  struct record {
    op_id op;
    interval bytes;
    access_mode mode;
    int group;
  };
  static bool conflicts(const access &a, const record &r) {
    if (!(a.bytes.begin < r.bytes.end && r.bytes.begin < a.bytes.end))
      return false;
    if (a.mode == access_mode::read && r.mode == access_mode::read)
      return false;
    if (a.mode == access_mode::inplace && r.mode == access_mode::inplace &&
        a.group == r.group)
      return false;
    return true;
  }
  std::unordered_map<std::size_t, std::vector<record>> records_;
};

} // namespace sg::detail

#endif
