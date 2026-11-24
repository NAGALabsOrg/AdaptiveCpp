// SPDX-License-Identifier: BSD-2-Clause
// Segmented runs. A segmented work set has one native group per segment.
// Distributed runs: every segment computes its share and publishes a partial to
// global memory; a reduce or bin result is made by the highest segment from all
// partials, a scan adds the totals of the earlier segments. Replicated runs:
// every segment runs the whole run, round after round, carrying the combined
// value of the earlier rounds.
#ifndef HIPSYCL_SG_DETAIL_DEV_SEGMENTS_HPP
#define HIPSYCL_SG_DETAIL_DEV_SEGMENTS_HPP

#include "hipSYCL/sg/detail/dev/abi.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sg::detail::dev {

// What the host learns about a function by calling it with dummy values.
struct sizing_rec {
  std::int64_t bins = 0;
  std::uint32_t elem_bytes = 0;    // largest element a collective works on
  std::uint32_t partial_bytes = 0; // global partial slots per unit and segment
};

struct run_ctx {
  const void *nd = nullptr; // sycl::nd_item<1>
  std::byte *ws = nullptr;  // local workspace
  std::uint32_t gid = 0, lanes = 1, k = 0, unit = 0, n_units = 1, group_size = 1;
  std::int64_t pos = 0;     // run position of this lane in this round
  std::int64_t valid = 0;   // valid lanes of this unit in this round
  bool active = false;
  bool real_unit = false; // the lane belongs to a unit of the map
  std::uint32_t kind = 0, seg = 0, segs = 1, ips = 1, round = 0, rounds = 1;
  std::int64_t run_len = 0;
  std::uint64_t partial_addr = 0;
  std::uint32_t partial_stride = 0;
  std::int64_t tile_idx = 0;
  std::uint32_t call_off = 0;
  sizing_rec *sizing = nullptr;
};

inline std::uint32_t round16(std::uint32_t v) { return (v + 15u) / 16u * 16u; }

inline const sycl::nd_item<1> &nd_of(const run_ctx &c) { return *static_cast<const sycl::nd_item<1> *>(c.nd); }
inline void group_sync(const run_ctx &c) { sycl::group_barrier(nd_of(c).get_group()); }

// The partial slot of one segment for the collective call being made.
inline std::byte *partial_slot(const run_ctx &c, std::uint32_t segment) {
  const std::uint64_t at =
      ((static_cast<std::uint64_t>(c.tile_idx) * c.segs + segment) * c.n_units + c.unit) * c.partial_stride + c.call_off;
  return reinterpret_cast<std::byte *>(c.partial_addr) + at;
}

// Whether a segment has any position of the run.
inline bool segment_has_share(const run_ctx &c, std::uint32_t segment) {
  return static_cast<std::int64_t>(segment) * c.ips < c.run_len;
}

template <class T> inline T load_partial(const std::byte *p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}
template <class T> inline void store_partial(std::byte *p, const T &v) { std::memcpy(p, &v, sizeof(T)); }

// A reduced value of this segment (distributed) or round (replicated), made
// into the value of the whole run.
template <class T, class Op> inline T finish_reduce(run_ctx &c, T r, Op op) {
  T result = r;
  if (c.kind == 2) { // replicated: carry across rounds in this segment's own slot
    std::byte *slot = partial_slot(c, c.seg);
    if (c.round > 0)
      result = op(load_partial<T>(slot), r);
    group_sync(c);
    if (c.k == 0 && c.valid > 0)
      store_partial<T>(slot, result);
    group_sync(c);
  } else if (c.kind == 1) { // distributed: partials, highest segment combines
    if (c.segs > 1) {
      if (c.k == 0 && c.valid > 0)
        store_partial<T>(partial_slot(c, c.seg), r);
      if (c.seg + 1 == c.segs) {
        bool have = false;
        T acc = r;
        for (std::uint32_t s = 0; s + 1 < c.segs; ++s)
          if (segment_has_share(c, s)) {
            const T p = load_partial<T>(partial_slot(c, s));
            acc = have ? op(acc, p) : p;
            have = true;
          }
        const bool own = c.valid > 0;
        result = own ? (have ? op(acc, r) : r) : acc;
      }
    }
  }
  c.call_off += round16(sizeof(T));
  return result;
}

// An inclusive scan of this segment's share (or round), made into the scan of the whole run.
template <class T, class Op> inline T finish_scan(run_ctx &c, T local, T total, Op op) {
  T result = local;
  if (c.kind == 2) {
    std::byte *slot = partial_slot(c, c.seg);
    const bool carried = c.round > 0;
    T before = local;
    if (carried) {
      before = load_partial<T>(slot);
      result = op(before, local);
    }
    group_sync(c);
    if (c.k == 0 && c.valid > 0)
      store_partial<T>(slot, carried ? op(before, total) : total);
    group_sync(c);
  } else if (c.kind == 1 && c.segs > 1) {
    if (c.k == 0 && c.valid > 0)
      store_partial<T>(partial_slot(c, c.seg), total);
    if (c.seg > 0) {
      bool have = false;
      T acc = local;
      for (std::uint32_t s = 0; s < c.seg; ++s)
        if (segment_has_share(c, s)) {
          const T p = load_partial<T>(partial_slot(c, s));
          acc = have ? op(acc, p) : p;
          have = true;
        }
      result = have ? op(acc, local) : local;
    }
  }
  c.call_off += round16(sizeof(T));
  return result;
}

} // namespace sg::detail::dev

#endif
