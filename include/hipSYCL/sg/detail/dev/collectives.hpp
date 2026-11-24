// SPDX-License-Identifier: BSD-2-Clause
// Collectives over the run of one unit, built from local memory and barriers:
// any trivially copyable element, any associative operation, in item order. No
// known identity is needed and no sub-32-bit atomics are used.
#ifndef HIPSYCL_SG_DETAIL_DEV_COLLECTIVES_HPP
#define HIPSYCL_SG_DETAIL_DEV_COLLECTIVES_HPP

#include "hipSYCL/sg/detail/dev/segments.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace sg::detail::dev {

// The combination of the valid lanes of the unit, in lane order.
template <class T, class Op> inline T local_reduce(run_ctx &c, T v, Op op) {
  T *ws = reinterpret_cast<T *>(c.ws);
  ws[c.gid] = v;
  group_sync(c);
  for (std::uint32_t s = 1; s < c.lanes; s <<= 1) {
    if (c.k % (2 * s) == 0 && static_cast<std::int64_t>(c.k) + s < c.valid)
      ws[c.gid] = op(ws[c.gid], ws[c.gid + s]);
    group_sync(c);
  }
  const T r = ws[c.gid - c.k];
  group_sync(c);
  return r;
}

// The inclusive scan of the valid lanes of the unit, in lane order; `total` is the unit's total.
template <class T, class Op> inline T local_scan(run_ctx &c, T v, Op op, T &total) {
  T *ws = reinterpret_cast<T *>(c.ws);
  ws[c.gid] = v;
  group_sync(c);
  for (std::uint32_t s = 1; s < c.lanes; s <<= 1) {
    T t = ws[c.gid];
    if (c.k >= s && static_cast<std::int64_t>(c.k) < c.valid)
      t = op(ws[c.gid - s], ws[c.gid]);
    group_sync(c);
    ws[c.gid] = t;
    group_sync(c);
  }
  const T r = ws[c.gid];
  total = ws[c.gid - c.k + (c.valid > 0 ? c.valid - 1 : 0)];
  group_sync(c);
  return r;
}

// Counts per bin over the unit's valid lanes: lane k holds the count of bin k.
template <class T, class Key>
inline std::int32_t local_bin(run_ctx &c, T v, std::int64_t bins, Key key) {
  std::int32_t *cnt = reinterpret_cast<std::int32_t *>(c.ws);
  const std::int64_t total = static_cast<std::int64_t>(c.n_units) * bins;
  for (std::int64_t i = c.gid; i < total; i += c.group_size)
    cnt[i] = 0;
  group_sync(c);
  if (c.active) {
    const std::int64_t b = static_cast<std::int64_t>(key(v));
    if (b >= 0 && b < bins) {
      sycl::atomic_ref<std::int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                       sycl::access::address_space::local_space>
          ref(cnt[static_cast<std::int64_t>(c.unit) * bins + b]);
      ref.fetch_add(1);
    }
  }
  group_sync(c);
  const std::int32_t r =
      (c.real_unit && c.k < bins) ? cnt[static_cast<std::int64_t>(c.unit) * bins + c.k] : 0;
  group_sync(c);
  return r;
}

// The counts of the whole run: partials joined across segments, or carried across rounds.
inline std::int32_t finish_bin(run_ctx &c, std::int32_t r, std::int64_t bins) {
  std::int32_t result = r;
  const bool lane_ok = c.real_unit && c.k < bins;
  std::int32_t *own = reinterpret_cast<std::int32_t *>(partial_slot(c, c.seg));
  if (c.kind == 2) {
    if (c.round > 0 && lane_ok)
      result = r + own[c.k];
    group_sync(c);
    if (lane_ok)
      own[c.k] = result;
    group_sync(c);
  } else if (c.kind == 1 && c.segs > 1) {
    if (lane_ok)
      own[c.k] = r;
    if (c.seg + 1 == c.segs && lane_ok) {
      std::int32_t acc = r;
      for (std::uint32_t s = 0; s + 1 < c.segs; ++s)
        if (segment_has_share(c, s))
          acc += reinterpret_cast<const std::int32_t *>(partial_slot(c, s))[c.k];
      result = acc;
    }
  }
  c.call_off += round16(static_cast<std::uint32_t>(4 * bins));
  return result;
}

} // namespace sg::detail::dev

#endif
