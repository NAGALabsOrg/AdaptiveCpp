// SPDX-License-Identifier: BSD-2-Clause
// The generic device drivers, written once over a placeholder policy P. The
// production policy is `ph`; the composition step clones the extern "C"
// __sg_drv_* wrappers per layer and per node.
#ifndef HIPSYCL_SG_DETAIL_DEV_DRIVERS_HPP
#define HIPSYCL_SG_DETAIL_DEV_DRIVERS_HPP

#include "hipSYCL/sg/detail/dev/abi.hpp"
#include "hipSYCL/sg/detail/dev/layout.hpp"
#include "hipSYCL/sg/detail/dev/placeholders.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace sg::detail::dev {

inline const node_rec &node_at(ctx_t c, std::uint32_t n) {
  return reinterpret_cast<const node_rec *>(c->nodes)[n];
}
inline const link_rec &link_at(ctx_t c, std::uint32_t node, std::uint32_t k) {
  return reinterpret_cast<const link_rec *>(c->links)[node_at(c, node).link_first + k];
}
inline const layer_rec &layer_at(ctx_t c, std::uint32_t l) {
  return reinterpret_cast<const layer_rec *>(c->layers)[l];
}
inline const sycl::nd_item<1> &nd_of(const void *p) {
  return *static_cast<const sycl::nd_item<1> *>(p);
}
inline void group_sync(const void *nd) { sycl::group_barrier(nd_of(nd).get_group()); }
inline std::uint64_t local_id_of(const void *nd) { return nd_of(nd).get_local_id(0); }
inline std::uint64_t local_size_of(const void *nd) { return nd_of(nd).get_local_range(0); }

using flag_ref = sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>;

inline std::uint32_t *flag_ptr(ctx_t c, std::uint32_t layer, std::uint64_t g) {
  return reinterpret_cast<std::uint32_t *>(layer_at(c, layer).flags) + g;
}
inline std::uint32_t uniform(const void *nd, std::uint32_t v) {
  return sycl::group_broadcast(nd_of(nd).get_group(), v);
}

template <class P> void owner(ctx_t, std::uint32_t, std::uint32_t, std::uint64_t, const void *, void *);
template <class P> void compute(ctx_t, std::uint32_t, std::uint32_t, std::uint64_t, const void *, void *);
template <class P> void resolve_layer(ctx_t, std::uint32_t, std::uint32_t, std::uint64_t, const void *, void *);
template <class P> void resolve_tree(ctx_t, std::uint32_t, std::uint32_t, std::uint64_t, const void *, void *);
template <class P> void prep(ctx_t, std::uint32_t, std::uint32_t, std::uint64_t, const void *, void *);
template <class P>
void run_group(ctx_t, std::uint32_t, std::uint32_t, std::uint64_t, const void *, void *, const tile_out *);

// How a driver calls its siblings: directly for mock policies, through the
// extern "C" wrappers (which composition clones) for the production policy.
template <class P> struct drv {
  static void compute(ctx_t c, std::uint32_t n, std::uint32_t l, std::uint64_t g, const void *nd, void *f) {
    dev::compute<P>(c, n, l, g, nd, f);
  }
  static void resolve_tree(ctx_t c, std::uint32_t n, std::uint32_t l, std::uint64_t g, const void *nd, void *f) {
    dev::resolve_tree<P>(c, n, l, g, nd, f);
  }
  static void prep(ctx_t c, std::uint32_t n, std::uint32_t l, std::uint64_t g, const void *nd, void *f) {
    dev::prep<P>(c, n, l, g, nd, f);
  }
  static void run_group(ctx_t c, std::uint32_t n, std::uint32_t l, std::uint64_t g, const void *nd, void *f, const tile_out *o) {
    dev::run_group<P>(c, n, l, g, nd, f, o);
  }
};

template <class P>
void owner(ctx_t c, std::uint32_t node, std::uint32_t layer, std::uint64_t g, const void *nd, void *frame) {
  std::uint32_t win = 0;
  if (local_id_of(nd) == 0) {
    flag_ref f{*flag_ptr(c, layer, g)};
    std::uint32_t expected = flag_unclaimed;
    win = f.compare_exchange_strong(expected, flag_in_progress, sycl::memory_order::relaxed,
                                    sycl::memory_order::relaxed)
              ? 1
              : 0;
  }
  win = uniform(nd, win);
  if (win)
    drv<P>::compute(c, node, layer, g, nd, frame);
}

template <class P>
void compute(ctx_t c, std::uint32_t node, std::uint32_t layer, std::uint64_t g, const void *nd, void *frame) {
  drv<P>::resolve_tree(c, node, layer, g, nd, frame);
  drv<P>::prep(c, node, layer, g, nd, frame);
  group_sync(nd);
  const tile_out t = layer_tile(layer_at(c, layer), node_at(c, node), g);
  drv<P>::run_group(c, node, layer, g, nd, frame, &t);
  group_sync(nd);
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
  if (local_id_of(nd) == 0) {
    flag_ref f{*flag_ptr(c, layer, g)};
    f.store(flag_done, sycl::memory_order::release);
  }
}

template <class P>
void resolve_layer(ctx_t c, std::uint32_t node, std::uint32_t layer, std::uint64_t g, const void *nd, void *frame) {
  std::uint32_t d = 0;
  if (local_id_of(nd) == 0) {
    flag_ref f{*flag_ptr(c, layer, g)};
    const std::uint32_t seen = f.load(sycl::memory_order::acquire);
    std::uint32_t expected = flag_unclaimed;
    const bool won = f.compare_exchange_strong(expected, flag_in_progress,
                                               sycl::memory_order::relaxed,
                                               sycl::memory_order::relaxed);
    d = won ? 1u : (seen == flag_done ? 0u : 2u);
  }
  d = uniform(nd, d);
  if (d == 1) {
    drv<P>::compute(c, node, layer, g, nd, frame);
  } else if (d == 2) {
    if (local_id_of(nd) == 0) {
      flag_ref f{*flag_ptr(c, layer, g)};
      while (f.load(sycl::memory_order::acquire) != flag_done) {
      }
    }
    group_sync(nd);
  }
  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
}

template <class P>
void resolve_tree(ctx_t c, std::uint32_t node, std::uint32_t, std::uint64_t g, const void *nd, void *frame) {
  const node_rec &n = node_at(c, node);
  const std::uint32_t links = uniform(nd, n.link_count);
  for (std::uint32_t k = 0; k < links; ++k) {
    const link_rec &lr = link_at(c, node, k);
    const std::uint32_t kind = uniform(nd, lr.kind);
    const std::uint32_t count = uniform(nd, static_cast<std::uint32_t>(lr.count));
    for (std::uint32_t s = 0; s < count; ++s) {
      std::uint64_t src = 0;
      P::positions(c, node, k, g, s, &src);
      if (kind == 2)
        P::global_resolve(c, node, k, src, nd, frame);
      else
        P::resolve_tree(c, node, k, src, nd, frame);
    }
  }
}

template <class P>
void prep(ctx_t c, std::uint32_t node, std::uint32_t, std::uint64_t g, const void *nd, void *frame) {
  const node_rec &n = node_at(c, node);
  char *base = static_cast<char *>(frame);
  const std::uint32_t links = uniform(nd, n.link_count);
  for (std::uint32_t k = 0; k < links; ++k) {
    const link_rec &lr = link_at(c, node, k);
    const std::uint32_t kind = uniform(nd, lr.kind);
    const std::uint32_t count = uniform(nd, static_cast<std::uint32_t>(lr.count));
    const std::uint32_t strided = uniform(nd, lr.child_frame_stride > 0 ? 1u : 0u);
    if (kind == 1) {
      for (std::uint32_t s = 0; s < count; ++s) {
        std::uint64_t src = 0;
        P::positions(c, node, k, g, s, &src);
        P::prep(c, node, k, src, nd, base + lr.child_frame_offset);
        group_sync(nd);
        const tile_out slot = dense_tile(
            reinterpret_cast<std::uint64_t>(base + lr.slots_offset + s * lr.slot_bytes),
            lr.src_group_shape, lr.src_elem_bytes);
        P::run_source(c, node, k, src, nd, base + lr.child_frame_offset, &slot);
        group_sync(nd);
      }
    } else if (kind == 0 && strided) {
      for (std::uint32_t s = 0; s < count; ++s) {
        std::uint64_t src = 0;
        P::positions(c, node, k, g, s, &src);
        P::prep(c, node, k, src, nd, base + lr.child_frame_offset + s * lr.child_frame_stride);
        group_sync(nd);
      }
    }
  }
}

template <class P>
void run_group(ctx_t c, std::uint32_t node, std::uint32_t, std::uint64_t g, const void *nd,
               void *frame, const tile_out *out) {
  const node_rec &n = node_at(c, node);
  if (n.body_kind == 0) {
    for (std::uint64_t l = local_id_of(nd); l < n.group_elems; l += local_size_of(nd))
      P::body(c, node, g, l, frame, reinterpret_cast<void *>(tile_address(*out, l)));
  } else {
    P::group_body(c, node, nd, g, frame, out);
  }
}

template <class P> void run_native_group(ctx_t c, const sycl::nd_item<1> &it, void *frame) {
  const std::uint64_t n = it.get_group(0);
  for (std::uint32_t i = 0; i < c->num_layers; ++i) {
    const layer_rec &l = layer_at(c, i);
    if (n >= l.range_offset && n < l.range_offset + l.groups) {
      P::run_layer(c, i, n - l.range_offset, &it, frame);
      return;
    }
  }
}

} // namespace sg::detail::dev

// Production wrappers: what the composition step clones.
#define SG_DRV_DECL(name, ...)                                                           \
  extern "C" SYCL_EXTERNAL [[gnu::used]] [[clang::noinline]]                             \
  [[clang::annotate("acpp_no_s1_device_inline")]] inline void __sg_drv_##name(__VA_ARGS__)

SG_DRV_DECL(owner, const sg::detail::dev::launch_ctx *, std::uint32_t, std::uint32_t, std::uint64_t, const void *, void *);
SG_DRV_DECL(compute, const sg::detail::dev::launch_ctx *, std::uint32_t, std::uint32_t, std::uint64_t, const void *, void *);
SG_DRV_DECL(resolve_layer, const sg::detail::dev::launch_ctx *, std::uint32_t, std::uint32_t, std::uint64_t, const void *, void *);
SG_DRV_DECL(resolve_tree, const sg::detail::dev::launch_ctx *, std::uint32_t, std::uint32_t, std::uint64_t, const void *, void *);
SG_DRV_DECL(prep, const sg::detail::dev::launch_ctx *, std::uint32_t, std::uint32_t, std::uint64_t, const void *, void *);
SG_DRV_DECL(run_group, const sg::detail::dev::launch_ctx *, std::uint32_t, std::uint32_t, std::uint64_t, const void *, void *, const sg::detail::dev::tile_out *);

namespace sg::detail::dev {
template <> struct drv<ph> {
  static void compute(ctx_t c, std::uint32_t n, std::uint32_t l, std::uint64_t g, const void *nd, void *f) {
    __sg_drv_compute(c, n, l, g, nd, f);
  }
  static void resolve_tree(ctx_t c, std::uint32_t n, std::uint32_t l, std::uint64_t g, const void *nd, void *f) {
    __sg_drv_resolve_tree(c, n, l, g, nd, f);
  }
  static void prep(ctx_t c, std::uint32_t n, std::uint32_t l, std::uint64_t g, const void *nd, void *f) {
    __sg_drv_prep(c, n, l, g, nd, f);
  }
  static void run_group(ctx_t c, std::uint32_t n, std::uint32_t l, std::uint64_t g, const void *nd, void *f, const tile_out *o) {
    __sg_drv_run_group(c, n, l, g, nd, f, o);
  }
};
} // namespace sg::detail::dev

#define SG_DRV_DEF(name)                                                                 \
  SG_DRV_DECL(name, const sg::detail::dev::launch_ctx *c, std::uint32_t n, std::uint32_t l, \
              std::uint64_t g, const void *nd, void *f) {                                \
    sg::detail::dev::name<sg::detail::dev::ph>(c, n, l, g, nd, f);                       \
  }

SG_DRV_DEF(owner)
SG_DRV_DEF(compute)
SG_DRV_DEF(resolve_layer)
SG_DRV_DEF(resolve_tree)
SG_DRV_DEF(prep)

SG_DRV_DECL(run_group, const sg::detail::dev::launch_ctx *c, std::uint32_t n, std::uint32_t l,
            std::uint64_t g, const void *nd, void *f, const sg::detail::dev::tile_out *o) {
  sg::detail::dev::run_group<sg::detail::dev::ph>(c, n, l, g, nd, f, o);
}

#endif
