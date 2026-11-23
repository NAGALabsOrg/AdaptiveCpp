// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_KERNEL_PREPARER_HPP
#define HIPSYCL_SG_DETAIL_KERNEL_PREPARER_HPP

#include "hipSYCL/sg/detail/launch_plan.hpp"
#include "hipSYCL/sg/detail/tree_store.hpp"
#include "hipSYCL/sg/kernel.hpp"

#include <memory>
#include <vector>

namespace sg::detail {

struct prepared_kernel {
  std::shared_ptr<kernel_tree> tree;
  launch_plan plan;
};

class kernel_prep_service {
public:
  virtual ~kernel_prep_service() = default;
  virtual prepared_kernel prepare(node_record &node, const std::vector<slot_binding> &resolved,
                                  const device_limits &limits) = 0;
};

class kernel_preparer final : public kernel_prep_service {
public:
  explicit kernel_preparer(tree_store &store) : store_(store) {}

  prepared_kernel prepare(node_record &node, const std::vector<slot_binding> &resolved,
                          const device_limits &limits) override {
    std::shared_ptr<kernel_tree> tree;
    if (node.key)
      tree = store_.find(node.builder_type, *node.key);
    if (!tree) {
      prepare_context ctx;
      ctx.resolve_dim = [&](std::uint64_t enc) -> std::uint64_t {
        for (std::size_t s = 0; s < node.slots.size() && s < resolved.size(); ++s)
          for (int d = 0; d < 3; ++d)
            if (node.slots[s].extent[d] == enc)
              return resolved[s].extent[d];
        throw error{errc::internal, "a deferred extent could not be resolved"};
      };
      prepare_context_scope pcs(&ctx);
      kernel_handle_impl impl(node.slots);
      builder_scope bs(&impl);
      kernel_handle kh(impl);
      node.builder(kh);
      tree = impl.finish();
      if (node.key)
        store_.insert(node.builder_type, std::move(node.key), tree);
    }
    for (const sink &s : tree->defs.sinks) {
      const sycl::range<3> total = tree->defs.nodes[s.node].total_shape();
      for (int d = 0; d < 3; ++d)
        if (s.slot >= resolved.size() || resolved[s.slot].extent[d] != total[d])
          throw error{errc::shape_mismatch,
                      "evaluate: the definition's shape differs from the result's shape"};
    }
    std::vector<std::size_t> sizes;
    for (const closure_entry &c : tree->closures)
      sizes.push_back(c.bytes.size());
    prepared_kernel out;
    out.plan = plan_launch(tree->durable, tree->defs, tree->order, sizes, limits);
    out.tree = std::move(tree);
    return out;
  }

private:
  tree_store &store_;
};

} // namespace sg::detail

#endif
