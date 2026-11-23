// SPDX-License-Identifier: BSD-2-Clause
// Per-device memory pool: eager slabs, fixed 1 KiB nodes, a free bitmap on the
// host. Allocation and release are host bookkeeping.
#ifndef HIPSYCL_SG_DETAIL_POOL_HPP
#define HIPSYCL_SG_DETAIL_POOL_HPP

#include "hipSYCL/sg/detail/dependency.hpp"
#include "hipSYCL/sg/detail/slab_bitmap.hpp"
#include "hipSYCL/sg/error.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace sg::detail {

struct pool_block {
  void *ptr = nullptr;
  std::size_t bytes = 0, slab = 0, first = 0, nodes = 0;
  std::vector<op_id> wait_for;
};

class pool {
public:
  static constexpr std::size_t node_bytes = 1024;

  pool(std::size_t reservation, std::size_t max_single_alloc,
       std::function<void *(std::size_t)> alloc, std::function<void(void *)> dealloc)
      : alloc_(std::move(alloc)), dealloc_(std::move(dealloc)) {
    std::size_t remaining = reservation;
    try {
      while (remaining >= node_bytes) {
        std::size_t sz = std::min(max_single_alloc, remaining) / node_bytes * node_bytes;
        if (sz < node_bytes)
          break;
        void *p = alloc_(sz);
        slabs_.push_back(slab{static_cast<std::byte *>(p), sz, slab_bitmap(sz / node_bytes),
                              std::vector<std::vector<op_id>>(sz / node_bytes)});
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

  pool_block allocate(std::size_t bytes) {
    pool_block b;
    if (bytes == 0)
      return b;
    const std::size_t nodes = (bytes + node_bytes - 1) / node_bytes;
    for (std::size_t s = 0; s < slabs_.size(); ++s) {
      slab &sl = slabs_[s];
      const std::size_t first = sl.bm.allocate(nodes);
      if (first == slab_bitmap::npos)
        continue;
      for (std::size_t i = 0; i < nodes; ++i) {
        auto &u = sl.users[first + i];
        b.wait_for.insert(b.wait_for.end(), u.begin(), u.end());
        u.clear();
      }
      std::sort(b.wait_for.begin(), b.wait_for.end());
      b.wait_for.erase(std::unique(b.wait_for.begin(), b.wait_for.end()), b.wait_for.end());
      b.ptr = sl.base + first * node_bytes;
      b.bytes = bytes;
      b.slab = s;
      b.first = first;
      b.nodes = nodes;
      return b;
    }
    std::size_t free_nodes = 0;
    for (const slab &sl : slabs_)
      free_nodes += sl.bm.free_count();
    const bool full = free_nodes < nodes;
    throw error{errc::pool_exhausted,
                std::string("the pool is ") + (full ? "full" : "fragmented") + ": requested " +
                    std::to_string(bytes) + " bytes, largest free run " +
                    std::to_string(largest_free_run()) + " bytes, used " +
                    std::to_string(used()) + " bytes of " + std::to_string(reserved()) +
                    " reserved"};
  }

  void release(const pool_block &b, const std::vector<op_id> &users) {
    if (b.nodes == 0)
      return;
    slab &sl = slabs_.at(b.slab);
    sl.bm.release(b.first, b.nodes);
    for (std::size_t i = 0; i < b.nodes; ++i)
      sl.users[b.first + i] = users;
  }

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
  std::size_t largest_free_run() const {
    std::size_t best = 0;
    for (const slab &s : slabs_)
      best = std::max(best, s.bm.largest_free_run());
    return best * node_bytes;
  }

private:
  struct slab {
    std::byte *base;
    std::size_t bytes;
    slab_bitmap bm;
    std::vector<std::vector<op_id>> users;
  };
  void free_slabs() {
    for (slab &s : slabs_)
      dealloc_(s.base);
    slabs_.clear();
  }
  std::function<void *(std::size_t)> alloc_;
  std::function<void(void *)> dealloc_;
  std::vector<slab> slabs_;
};

} // namespace sg::detail

#endif
