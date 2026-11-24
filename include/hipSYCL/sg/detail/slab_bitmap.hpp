// SPDX-License-Identifier: BSD-2-Clause
// Port of the two-level free bitmap of SlabKit's PoolImpl. 1 = free.
//   l0[i] bit j: slot (i*64 + j) is free
//   l1[i] bit j: l0[i*64 + j] has at least one free bit
#ifndef HIPSYCL_SG_DETAIL_SLAB_BITMAP_HPP
#define HIPSYCL_SG_DETAIL_SLAB_BITMAP_HPP

#include "hipSYCL/sg/error.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sg::detail {

class slab_bitmap {
public:
  static constexpr std::size_t npos = static_cast<std::size_t>(-1);

  explicit slab_bitmap(std::size_t slots) : slot_count_{slots}, free_count_{slots} {
    l0_.assign((slot_count_ + 63) / 64, ~std::uint64_t{0});
    l1_.assign((l0_.size() + 63) / 64, ~std::uint64_t{0});
    if (const std::size_t bits = slot_count_ % 64; bits != 0 && !l0_.empty())
      l0_.back() = (std::uint64_t{1} << bits) - 1;
    if (const std::size_t bits = l0_.size() % 64; bits != 0 && !l1_.empty())
      l1_.back() = (std::uint64_t{1} << bits) - 1;
  }

  std::size_t slot_count() const { return slot_count_; }
  std::size_t free_count() const { return free_count_; }

  std::size_t allocate(std::size_t n) {
    if (n == 0 || free_count_ < n)
      return npos;
    const std::size_t start = (n == 1) ? find_one() : find_contiguous(n);
    if (start == npos)
      return npos;
    for (std::size_t i = 0; i < n; ++i)
      mark_used(start + i);
    return start;
  }

  // Takes the n free slots from first on; every one of them must be free.
  void claim(std::size_t first, std::size_t n) {
    if (first >= slot_count_ || n > slot_count_ - first)
      throw error{errc::internal, "slab_bitmap::claim: range out of bounds"};
    for (std::size_t i = 0; i < n; ++i)
      if (!is_free(first + i))
        throw error{errc::internal, "slab_bitmap::claim: slot is not free"};
    for (std::size_t i = 0; i < n; ++i)
      mark_used(first + i);
  }

  void release(std::size_t first, std::size_t n) {
    if (n == 0)
      return;
    if (first >= slot_count_ || n > slot_count_ - first)
      throw error{errc::internal, "slab_bitmap::release: range out of bounds"};
    for (std::size_t i = 0; i < n; ++i)
      if (is_free(first + i))
        throw error{errc::internal, "slab_bitmap::release: slot is already free"};
    for (std::size_t i = 0; i < n; ++i)
      mark_free(first + i);
  }

  bool is_free(std::size_t slot) const { return (l0_[slot / 64] >> (slot % 64)) & 1u; }

  std::size_t largest_free_run() const {
    std::size_t best = 0, cur = 0;
    for (std::size_t i = 0; i < slot_count_; ++i) {
      if (is_free(i)) {
        ++cur;
        if (cur > best)
          best = cur;
      } else {
        cur = 0;
      }
    }
    return best;
  }

private:
  void mark_used(std::size_t slot) {
    const std::size_t word = slot / 64;
    l0_[word] &= ~(std::uint64_t{1} << (slot % 64));
    if (l0_[word] == 0)
      l1_[word / 64] &= ~(std::uint64_t{1} << (word % 64));
    --free_count_;
  }
  void mark_free(std::size_t slot) {
    const std::size_t word = slot / 64;
    l0_[word] |= (std::uint64_t{1} << (slot % 64));
    l1_[word / 64] |= (std::uint64_t{1} << (word % 64));
    ++free_count_;
  }

  std::size_t find_one() const {
    for (std::size_t w = 0; w < l1_.size(); ++w) {
      if (l1_[w] == 0)
        continue;
      const std::size_t group = w * 64 + static_cast<std::size_t>(std::countr_zero(l1_[w]));
      if (group >= l0_.size() || l0_[group] == 0)
        continue;
      const std::size_t slot =
          group * 64 + static_cast<std::size_t>(std::countr_zero(l0_[group]));
      if (slot < slot_count_)
        return slot;
    }
    return npos;
  }

  std::size_t find_contiguous(std::size_t n) const {
    return n <= 64 ? find_short(n) : find_long(n);
  }

  // lowest-addressed run of n <= 64 free slots, within a word or straddling a boundary
  std::size_t find_short(std::size_t n) const {
    const std::size_t nwords = l0_.size();
    for (std::size_t w = 0; w < nwords; ++w) {
      std::uint64_t mask = l0_[w];
      for (std::size_t s = 1; s < n; ++s)
        mask &= (l0_[w] >> s);
      std::size_t inner = npos;
      if (mask != 0) {
        const std::size_t slot = w * 64 + static_cast<std::size_t>(std::countr_zero(mask));
        if (slot + n <= slot_count_)
          inner = slot;
      }
      std::size_t straddle = npos;
      if (w + 1 < nwords) {
        const auto tail = static_cast<std::size_t>(std::countl_one(l0_[w]));
        if (tail != 0) {
          const auto head = static_cast<std::size_t>(std::countr_one(l0_[w + 1]));
          if (tail + head >= n) {
            const std::size_t slot = w * 64 + (64 - tail);
            if (slot + n <= slot_count_)
              straddle = slot;
          }
        }
      }
      const std::size_t best = inner < straddle ? inner : straddle;
      if (best != npos)
        return best;
    }
    return npos;
  }

  // n > 64: a partial head, full words, a partial tail
  std::size_t find_long(std::size_t n) const {
    const std::size_t nwords = l0_.size();
    for (std::size_t w = 0; w < nwords;) {
      if (l0_[w] == 0) {
        ++w;
        continue;
      }
      const auto head = static_cast<std::size_t>(std::countl_one(l0_[w]));
      if (head == 0) {
        ++w;
        continue;
      }
      // a run starting inside word w's high end
      const std::size_t after = head >= n ? 0 : n - head;
      if (head >= n) { // cannot happen for n > 64, kept for safety
        return w * 64 + (64 - head);
      }
      const std::size_t full = after / 64, rem = after % 64;
      std::size_t ww = w + 1;
      bool ok = true;
      for (std::size_t f = 0; f < full; ++f, ++ww)
        if (ww >= nwords || l0_[ww] != ~std::uint64_t{0}) {
          ok = false;
          break;
        }
      if (ok && rem > 0) {
        if (ww >= nwords || static_cast<std::size_t>(std::countr_one(l0_[ww])) < rem)
          ok = false;
      }
      if (ok) {
        const std::size_t slot = w * 64 + (64 - head);
        if (slot + n <= slot_count_)
          return slot;
      }
      ++w;
    }
    return npos;
  }

  std::size_t slot_count_;
  std::size_t free_count_;
  std::vector<std::uint64_t> l0_;
  std::vector<std::uint64_t> l1_;
};

} // namespace sg::detail

#endif
