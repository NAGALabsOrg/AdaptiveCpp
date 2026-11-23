// SPDX-License-Identifier: BSD-2-Clause
// Evaluation of the expression a group body returns: every reduce and scan
// stage into scratch, then the write-out of the output elements.
#ifndef HIPSYCL_SG_DETAIL_DEV_GROUP_EVAL_HPP
#define HIPSYCL_SG_DETAIL_DEV_GROUP_EVAL_HPP

#include "hipSYCL/sg/detail/dev/drivers.hpp"
#include "hipSYCL/sg/detail/dev/layout.hpp"
#include "hipSYCL/sg/group_expr.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace sg::detail::dev {

struct group_env {
  char *scratch;
  sycl::group<1> grp;
  std::uint64_t lid, W;
  sycl::group<1> group() const { return grp; }
  void sync() const { sycl::group_barrier(grp); }
  std::uint32_t uniform(std::uint64_t v) const {
    return sycl::group_broadcast(grp, static_cast<std::uint32_t>(v));
  }
};

template <class P = ph, class E>
void evaluate_group_expr(const E &e, const void *nd, ctx_t ctx, std::uint32_t node, void *frame,
                         const tile_out *out) {
  const node_rec &n = node_at(ctx, node);
  group_env env{static_cast<char *>(frame) + n.scratch_offset, nd_of(nd).get_group(),
                nd_of(nd).get_local_id(0), nd_of(nd).get_local_range(0)};
  e.stage(env, 0);
  const std::uint64_t total = sg::detail::prod3(e.dims());
  for (std::uint64_t k = env.lid; k < total; k += env.W) {
    const auto v = e.read(env, 0, k);
    P::write(ctx, node, reinterpret_cast<void *>(tile_address(*out, k)), &v);
  }
}

} // namespace sg::detail::dev

#endif
