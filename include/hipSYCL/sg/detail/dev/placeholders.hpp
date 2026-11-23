// SPDX-License-Identifier: BSD-2-Clause
// Placeholders the composition step replaces at JIT time. Their bodies only
// trap so the host binary links; no call to one survives composition.
#ifndef HIPSYCL_SG_DETAIL_DEV_PLACEHOLDERS_HPP
#define HIPSYCL_SG_DETAIL_DEV_PLACEHOLDERS_HPP

#include "hipSYCL/sg/detail/dev/abi.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

#define SG_DEV_FN                                                                      \
  extern "C" SYCL_EXTERNAL [[gnu::used]] [[clang::noinline]] [[clang::annotate("acpp_no_s1_device_inline")]] inline

SG_DEV_FN void __sg_ph_positions(const sg::detail::dev::launch_ctx *, std::uint32_t,
                                 std::uint32_t, std::uint64_t, std::uint32_t,
                                 std::uint64_t *) {
  __builtin_trap();
}
SG_DEV_FN void __sg_ph_item_value(const sg::detail::dev::launch_ctx *, std::uint32_t,
                                  std::uint32_t, std::uint64_t, std::uint64_t, void *,
                                  void *) {
  __builtin_trap();
}
SG_DEV_FN void __sg_ph_prep(const sg::detail::dev::launch_ctx *, std::uint32_t,
                            std::uint32_t, std::uint64_t, const void *, void *) {
  __builtin_trap();
}
SG_DEV_FN void __sg_ph_run_source(const sg::detail::dev::launch_ctx *, std::uint32_t,
                                  std::uint32_t, std::uint64_t, const void *, void *,
                                  const sg::detail::dev::tile_out *) {
  __builtin_trap();
}
SG_DEV_FN void __sg_ph_resolve_tree(const sg::detail::dev::launch_ctx *, std::uint32_t,
                                    std::uint32_t, std::uint64_t, const void *, void *) {
  __builtin_trap();
}
SG_DEV_FN void __sg_ph_global_resolve(const sg::detail::dev::launch_ctx *, std::uint32_t,
                                      std::uint32_t, std::uint64_t, const void *, void *) {
  __builtin_trap();
}
SG_DEV_FN void __sg_ph_body(const sg::detail::dev::launch_ctx *, std::uint32_t,
                            std::uint64_t, std::uint64_t, void *, void *) {
  __builtin_trap();
}
SG_DEV_FN void __sg_ph_group_body(const sg::detail::dev::launch_ctx *, std::uint32_t,
                                  const void *, std::uint64_t, void *,
                                  const sg::detail::dev::tile_out *) {
  __builtin_trap();
}
SG_DEV_FN void __sg_ph_write(const sg::detail::dev::launch_ctx *, std::uint32_t, void *,
                             const void *) {
  __builtin_trap();
}
SG_DEV_FN void __sg_ph_run_layer(const sg::detail::dev::launch_ctx *, std::uint32_t,
                                 std::uint64_t, const void *, void *) {
  __builtin_trap();
}

namespace sg::detail::dev {

using ctx_t = const launch_ctx *;

// The production policy: forwards to the placeholders above.
struct ph {
  static void positions(ctx_t c, std::uint32_t node, std::uint32_t link, std::uint64_t g,
                        std::uint32_t slot, std::uint64_t *src) {
    __sg_ph_positions(c, node, link, g, slot, src);
  }
  static void item_value(ctx_t c, std::uint32_t node, std::uint32_t link, std::uint64_t srcg,
                         std::uint64_t l, void *childframe, void *dst) {
    __sg_ph_item_value(c, node, link, srcg, l, childframe, dst);
  }
  static void prep(ctx_t c, std::uint32_t node, std::uint32_t link, std::uint64_t srcg,
                   const void *nd, void *childframe) {
    __sg_ph_prep(c, node, link, srcg, nd, childframe);
  }
  static void run_source(ctx_t c, std::uint32_t node, std::uint32_t link, std::uint64_t srcg,
                         const void *nd, void *childframe, const tile_out *out) {
    __sg_ph_run_source(c, node, link, srcg, nd, childframe, out);
  }
  static void resolve_tree(ctx_t c, std::uint32_t node, std::uint32_t link,
                           std::uint64_t srcg, const void *nd, void *frame) {
    __sg_ph_resolve_tree(c, node, link, srcg, nd, frame);
  }
  static void global_resolve(ctx_t c, std::uint32_t node, std::uint32_t link,
                             std::uint64_t srcg, const void *nd, void *frame) {
    __sg_ph_global_resolve(c, node, link, srcg, nd, frame);
  }
  static void body(ctx_t c, std::uint32_t node, std::uint64_t g, std::uint64_t l, void *frame,
                   void *dst) {
    __sg_ph_body(c, node, g, l, frame, dst);
  }
  static void group_body(ctx_t c, std::uint32_t node, const void *nd, std::uint64_t g,
                         void *frame, const tile_out *out) {
    __sg_ph_group_body(c, node, nd, g, frame, out);
  }
  static void write(ctx_t c, std::uint32_t node, void *dst, const void *src) {
    __sg_ph_write(c, node, dst, src);
  }
  static void run_layer(ctx_t c, std::uint32_t layer, std::uint64_t g, const void *nd,
                        void *frame) {
    __sg_ph_run_layer(c, layer, g, nd, frame);
  }
};

} // namespace sg::detail::dev

#endif
