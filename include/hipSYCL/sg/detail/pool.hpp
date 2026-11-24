// SPDX-License-Identifier: BSD-2-Clause
// Per-device memory pool that grows with need: slabs taken from the device, fixed 1 KiB nodes, a free
// bitmap on the host. Allocation and release are host bookkeeping. A free block is reused only when
// every last user of its nodes has finished or is an operation the requesting work already depends on;
// otherwise the pool takes a new slab.
#ifndef HIPSYCL_SG_DETAIL_POOL_HPP
#define HIPSYCL_SG_DETAIL_POOL_HPP

#include "hipSYCL/sg/detail/dependency.hpp"
#include "hipSYCL/sg/detail/slab_bitmap.hpp"
#include "hipSYCL/sg/error.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace sg::detail {

struct pool_block {
  void *ptr = nullptr;
  std::size_t bytes = 0, slab = 0, first = 0, nodes = 0;
};

class pool {
public:
  static constexpr std::size_t node_bytes = 1024;
  static constexpr std::size_t alignment = 256;
  // The smallest slab a request takes from the device.
  static constexpr std::size_t min_slab_bytes = 4u << 20;

  // reservation: bytes taken from the device at construction (0: none; slabs follow requests).
  // finished(op): whether an operation has completed. Null: no operation counts as finished.
  pool(std::size_t reservation, std::size_t max_single_alloc, std::function<void *(std::size_t)> alloc,
       std::function<void(void *)> dealloc, std::function<bool(op_id)> finished = nullptr)
      : max_single_(max_single_alloc), alloc_(std::move(alloc)), dealloc_(std::move(dealloc)),
        finished_(std::move(finished)) {
    std::size_t remaining = reservation;
    try {
      while (remaining >= node_bytes) {
        const std::size_t sz = std::min(max_single_, remaining) / node_bytes * node_bytes;
        if (sz < node_bytes)
          break;
        add_slab(sz);
        remaining -= sz;
      }
    } catch (...) {
      free_slabs();
      throw;
    }
  }
  pool(const pool &) = delete;
  pool &operator=(const pool &) = delete;
  ~pool() { free_slabs(); }

  // deps: the operations the requesting work already depends on.
  pool_block allocate(std::size_t bytes, const std::vector<op_id> &deps) {
    pool_block b;
    if (bytes == 0)
      return b;
    const std::size_t nodes = (bytes + node_bytes - 1) / node_bytes;
    std::map<op_id, bool> known;
    for (std::size_t s = 0; s < slabs_.size(); ++s)
      if (take_run(s, nodes, deps, known, b))
        return b;
    // grow: one slab sized to the request, at least the minimum, at most the largest single allocation
    const std::size_t cap = max_single_ / node_bytes * node_bytes;
    std::size_t sz = (std::max(nodes * node_bytes + alignment, min_slab_bytes) + node_bytes - 1) / node_bytes * node_bytes;
    sz = std::min(sz, cap);
    if (sz < nodes * node_bytes)
      throw error{errc::allocation_refused, "a request of " + std::to_string(bytes) +
                                                " bytes exceeds the device's largest single allocation of " +
                                                std::to_string(max_single_) + " bytes"};
    add_slab(sz);
    if (!take_run(slabs_.size() - 1, nodes, deps, known, b))
      throw error{errc::allocation_refused, "a request of " + std::to_string(bytes) +
                                                " bytes does not fit a slab of " + std::to_string(sz) + " bytes"};
    return b;
  }

  void release(const pool_block &b, const std::vector<op_id> &users) {
    if (b.nodes == 0)
      return;
    slab &sl = slabs_.at(b.slab);
    sl.bm.release(b.first, b.nodes);
    for (std::size_t i = 0; i < b.nodes; ++i)
      sl.users[b.first + i] = users;
  }

  std::size_t slab_count() const { return slabs_.size(); }
  std::size_t used() const {
    std::size_t n = 0;
    for (const slab &s : slabs_)
      n += (s.bm.slot_count() - s.bm.free_count()) * node_bytes;
    return n;
  }
  std::size_t reserved() const {
    std::size_t n = 0;
    for (const slab &s : slabs_)
      n += s.bytes;
    return n;
  }

private:
  struct slab {
    std::byte *base;    // as allocated
    std::byte *aligned; // first node
    std::size_t bytes;  // usable bytes, whole nodes
    slab_bitmap bm;
    std::vector<std::vector<op_id>> users;
  };

  void add_slab(std::size_t sz) {
    std::byte *p = static_cast<std::byte *>(alloc_(sz));
    // every node starts on a 256-byte boundary: skip the unaligned head of a slab
    const std::size_t skew = (alignment - reinterpret_cast<std::uintptr_t>(p) % alignment) % alignment;
    const std::size_t usable = (sz - skew) / node_bytes * node_bytes;
    slabs_.push_back(
        slab{p, p + skew, usable, slab_bitmap(usable / node_bytes), std::vector<std::vector<op_id>>(usable / node_bytes)});
  }

  bool usable_user(op_id u, const std::vector<op_id> &deps, std::map<op_id, bool> &known) const {
    if (std::find(deps.begin(), deps.end(), u) != deps.end())
      return true;
    auto it = known.find(u);
    if (it == known.end())
      it = known.emplace(u, finished_ ? finished_(u) : false).first;
    return it->second;
  }

  // The first run of free nodes whose last users all finished or are dependencies of the request.
  bool take_run(std::size_t s, std::size_t nodes, const std::vector<op_id> &deps, std::map<op_id, bool> &known,
                pool_block &out) {
    slab &sl = slabs_[s];
    if (sl.bm.free_count() < nodes)
      return false;
    std::size_t run = 0, start = 0;
    for (std::size_t i = 0; i < sl.bm.slot_count(); ++i) {
      bool ok = sl.bm.is_free(i);
      for (std::size_t u = 0; ok && u < sl.users[i].size(); ++u)
        ok = usable_user(sl.users[i][u], deps, known);
      if (!ok) {
        run = 0;
        continue;
      }
      if (run == 0)
        start = i;
      if (++run == nodes) {
        sl.bm.claim(start, nodes);
        for (std::size_t k = 0; k < nodes; ++k)
          sl.users[start + k].clear();
        out.ptr = sl.aligned + start * node_bytes;
        out.bytes = nodes * node_bytes;
        out.slab = s;
        out.first = start;
        out.nodes = nodes;
        return true;
      }
    }
    return false;
  }

  void free_slabs() {
    for (slab &s : slabs_)
      dealloc_(s.base);
    slabs_.clear();
  }

  std::size_t max_single_;
  std::function<void *(std::size_t)> alloc_;
  std::function<void(void *)> dealloc_;
  std::function<bool(op_id)> finished_;
  std::vector<slab> slabs_;
};

} // namespace sg::detail

#endif
