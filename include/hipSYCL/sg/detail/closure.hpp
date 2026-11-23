// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_CLOSURE_HPP
#define HIPSYCL_SG_DETAIL_CLOSURE_HPP

#include "hipSYCL/sg/detail/array_rep.hpp"
#include "hipSYCL/sg/error.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace sg::detail {

struct patch_site {
  std::size_t offset;
  int rank;
  std::uint32_t slot;
};

struct closure_entry {
  std::vector<std::byte> bytes;
  std::vector<patch_site> sites;
  std::size_t align = 8;
};

struct slot_binding {
  std::uint64_t address = 0;
  std::uint64_t extent[3] = {1, 1, 1};
};

inline std::vector<patch_site> find_patch_sites(const std::vector<std::byte> &b) {
  std::vector<patch_site> out;
  for (std::size_t o = 0; o + sizeof(array_rep) <= b.size();) {
    std::uint64_t v;
    std::memcpy(&v, b.data() + o, 8);
    if (is_sentinel(v)) {
      const int r = sentinel_rank(v);
      if (r >= 1 && r <= 3) {
        out.push_back({o, r, sentinel_slot(v)});
        o += sizeof(array_rep);
        continue;
      }
    }
    o += 8;
  }
  return out;
}

template <class C> closure_entry capture_closure(const C &c) {
  static_assert(std::is_trivially_copyable_v<C>,
                "sg: a body, mapping function or captured object must be trivially copyable");
  static_assert(alignof(C) <= 16);
  closure_entry e;
  e.bytes.resize(sizeof(C));
  if (sizeof(C))
    std::memcpy(e.bytes.data(), &c, sizeof(C));
  e.align = alignof(C);
  e.sites = find_patch_sites(e.bytes);
  return e;
}

inline std::vector<std::byte> patch_closure(const closure_entry &c,
                                            const std::vector<slot_binding> &by_slot) {
  std::vector<std::byte> out = c.bytes;
  for (const patch_site &s : c.sites) {
    if (s.slot >= by_slot.size())
      throw error{errc::internal, "patch_closure: slot out of range"};
    const slot_binding &b = by_slot[s.slot];
    std::memcpy(out.data() + s.offset, &b.address, 8);
    for (int d = 0; d < 3; ++d) {
      if (d < s.rank)
        std::memcpy(out.data() + s.offset + 8 + 8 * d, &b.extent[d], 8);
    }
  }
  return out;
}

} // namespace sg::detail

#endif
