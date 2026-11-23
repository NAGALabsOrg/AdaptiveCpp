// SPDX-License-Identifier: BSD-2-Clause
// The role rules: producers define a handle's size, readers must match it,
// inplace contributors of one group must agree.
#ifndef HIPSYCL_SG_DETAIL_ROLE_CHECK_HPP
#define HIPSYCL_SG_DETAIL_ROLE_CHECK_HPP

#include "hipSYCL/sg/detail/scope.hpp"
#include "hipSYCL/sg/error.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace sg::detail {

struct handle_view {
  bool produced = false;
  std::uint64_t type = 0;
  std::size_t count = 0;
  int inplace_group = -1;
  std::string op;
  init_kind init = init_kind::none;
  std::vector<std::byte> init_bits;
  int device = 0;
};
using handle_table = std::vector<handle_view>;

struct role_effect {
  bool needs_fill = false;
  bool new_group = false;
  int group = -1;
};

inline std::size_t slot_count_of(const array_slot &s) {
  return static_cast<std::size_t>(s.extent[0] * s.extent[1] * s.extent[2]);
}

inline std::vector<role_effect> check_roles(handle_table &table,
                                            const std::vector<array_slot> &slots, int device,
                                            int &next_group) {
  for (std::size_t a = 0; a < slots.size(); ++a)
    for (std::size_t b = a + 1; b < slots.size(); ++b) {
      const array_slot &x = slots[a], &y = slots[b];
      if (x.handle != y.handle || (x.mode == role_mode::read && y.mode == role_mode::read))
        continue;
      const std::size_t xe = x.offset + slot_count_of(x), ye = y.offset + slot_count_of(y);
      if (x.offset < ye && y.offset < xe)
        throw error{errc::role_conflict, "handle " + std::to_string(x.handle) +
                                             " is used by conflicting roles in one submit"};
    }

  handle_table next = table;
  int group_counter = next_group;
  std::vector<role_effect> effects(slots.size());
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const array_slot &s = slots[i];
    if (next.size() <= s.handle)
      next.resize(s.handle + 1);
    handle_view &v = next[s.handle];
    const std::string h = "handle " + std::to_string(s.handle);
    const std::size_t count = slot_count_of(s);
    switch (s.mode) {
    case role_mode::read:
      if (!v.produced)
        throw error{errc::never_produced, h + " is read but nothing has produced it"};
      if (v.type != s.type)
        throw error{errc::type_mismatch,
                    h + " holds a different element type than the array reading it"};
      if (s.offset + count > v.count || (s.whole && count != v.count))
        throw error{errc::size_mismatch,
                    h + " holds " + std::to_string(v.count) + " elements; the array reads " +
                        std::to_string(count) + " from offset " + std::to_string(s.offset)};
      v.inplace_group = -1;
      break;
    case role_mode::produce:
      v.produced = true;
      v.count = count;
      v.type = s.type;
      v.inplace_group = -1;
      break;
    case role_mode::inplace: {
      auto mismatch = [&](const char *what) {
        throw error{errc::inplace_group_mismatch,
                    h + ": inplace contributors disagree on the " + what};
      };
      if (v.inplace_group >= 0) {
        if (v.count != count)
          mismatch("shape");
        if (v.type != s.type)
          mismatch("element type");
        if (v.op != s.op)
          mismatch("operation");
        if (v.init != s.init || (s.init == init_kind::value && v.init_bits != s.init_bits))
          mismatch("initializer");
        if (v.device != device)
          mismatch("device");
        effects[i].group = v.inplace_group;
      } else {
        if (s.init == init_kind::reuse) {
          if (!v.produced)
            throw error{errc::never_produced,
                        h + ": inplace with reuse but nothing has produced it"};
          if (v.type != s.type)
            throw error{errc::type_mismatch, h + ": inplace with reuse has another element type"};
          if (v.count != count)
            throw error{errc::size_mismatch,
                        h + ": inplace with reuse must match the earlier writer's size"};
        } else {
          v.count = count;
          v.type = s.type;
        }
        v.produced = true;
        v.inplace_group = group_counter++;
        v.op = s.op;
        v.init = s.init;
        v.init_bits = s.init_bits;
        v.device = device;
        effects[i].new_group = true;
        effects[i].needs_fill = s.init == init_kind::value;
        effects[i].group = v.inplace_group;
      }
      break;
    }
    }
  }
  table = std::move(next);
  next_group = group_counter;
  return effects;
}

} // namespace sg::detail

#endif
