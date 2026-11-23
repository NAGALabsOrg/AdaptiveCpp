// SPDX-License-Identifier: BSD-2-Clause
// The recorded kernel (kernel_tree), the recorder the builder talks to
// (kernel_handle_impl) and the store of trees by (builder type, key).
#ifndef HIPSYCL_SG_DETAIL_TREE_STORE_HPP
#define HIPSYCL_SG_DETAIL_TREE_STORE_HPP

#include "hipSYCL/glue/llvm-sscp/composition.hpp"
#include "hipSYCL/sg/detail/closure.hpp"
#include "hipSYCL/sg/detail/composition_builder.hpp"
#include "hipSYCL/sg/detail/def_graph.hpp"
#include "hipSYCL/sg/detail/durable_graph.hpp"
#include "hipSYCL/sg/detail/probe.hpp"
#include "hipSYCL/sg/detail/scope.hpp"

#include <map>
#include <memory>
#include <vector>

namespace sg::detail {

struct kernel_tree {
  def_graph defs;
  std::vector<closure_entry> closures;
  durable_graph durable;
  std::vector<int> order;
  std::unique_ptr<hipsycl::glue::sscp::composition_config> config;
};

class tree_store {
public:
  std::shared_ptr<kernel_tree> find(std::uint64_t builder_type, const key_holder &key) const {
    auto it = entries_.find(builder_type);
    if (it == entries_.end())
      return nullptr;
    for (const entry &e : it->second)
      if (e.key->equals(key))
        return e.tree;
    return nullptr;
  }
  void insert(std::uint64_t builder_type, std::unique_ptr<key_holder> key,
              std::shared_ptr<kernel_tree> tree) {
    entries_[builder_type].push_back({std::move(key), std::move(tree)});
  }

private:
  struct entry {
    std::unique_ptr<key_holder> key;
    std::shared_ptr<kernel_tree> tree;
  };
  std::map<std::uint64_t, std::vector<entry>> entries_;
};

class kernel_handle_impl {
public:
  explicit kernel_handle_impl(const std::vector<array_slot> &slots) : slots_(slots) {
    tree_ = std::make_shared<kernel_tree>();
  }

  const def_graph &defs() const { return tree_->defs; }

  // body_then_links: the body closure, then one closure per link.
  int add_definition(def_node d, std::vector<closure_entry> body_then_links) {
    if (body_then_links.size() != d.links.size() + 1)
      throw error{errc::internal, "add_definition: closure count does not match the links"};
    d.closure = static_cast<int>(tree_->closures.size());
    tree_->closures.push_back(std::move(body_then_links[0]));
    for (std::size_t k = 0; k < d.links.size(); ++k) {
      d.links[k].closure = static_cast<int>(tree_->closures.size());
      tree_->closures.push_back(std::move(body_then_links[k + 1]));
    }
    tree_->defs.nodes.push_back(std::move(d));
    return static_cast<int>(tree_->defs.nodes.size() - 1);
  }

  void add_sink(sink s) {
    if (s.slot >= slots_.size() || slots_[s.slot].mode == role_mode::read)
      throw error{errc::builder_misuse, "evaluate: the target is not a result"};
    if (s.node < 0 || s.node >= static_cast<int>(tree_->defs.nodes.size()))
      throw error{errc::builder_misuse, "evaluate: unknown definition"};
    for (const sink &o : tree_->defs.sinks)
      if (o.node == s.node && o.slot == s.slot)
        throw error{errc::builder_misuse, "evaluate: a definition is evaluated once per result"};
    const array_slot &slot = slots_[s.slot];
    s.kind = slot.mode == role_mode::inplace ? result_kind::inplace : result_kind::plain;
    s.op_symbol = slot.op_symbol;
    check_shape(s);
    tree_->defs.sinks.push_back(std::move(s));
  }

  std::shared_ptr<kernel_tree> finish() {
    validate(tree_->defs);
    tree_->durable = build_durable_graph(tree_->defs);
    tree_->order = upstream_order(tree_->durable);
    tree_->config = std::make_unique<hipsycl::glue::sscp::composition_config>(
        make_composition_config(tree_->durable, tree_->defs, tree_->order));
    return std::move(tree_);
  }

  static kernel_handle_impl *&current_slot() noexcept {
    static thread_local kernel_handle_impl *c = nullptr;
    return c;
  }
  static kernel_handle_impl *current() noexcept { return current_slot(); }

private:
  void check_shape(const sink &s) const {
    const def_node &d = tree_->defs.nodes[s.node];
    const array_slot &slot = slots_[s.slot];
    const sycl::range<3> total = d.total_shape();
    for (int k = 0; k < 3; ++k) {
      std::uint64_t e = slot.extent[k];
      if (e & (1ull << 63)) {
        prepare_context *c = prepare_context::current();
        if (!c || !c->resolve_dim)
          return; // checked again after preparation
        e = c->resolve_dim(e);
      }
      if (e != total[k])
        throw error{errc::shape_mismatch,
                    "evaluate: the definition's shape differs from the result's shape"};
    }
  }

  const std::vector<array_slot> &slots_;
  std::shared_ptr<kernel_tree> tree_;
};

struct builder_scope {
  explicit builder_scope(kernel_handle_impl *k) : prev_(kernel_handle_impl::current_slot()) {
    kernel_handle_impl::current_slot() = k;
  }
  ~builder_scope() { kernel_handle_impl::current_slot() = prev_; }
  kernel_handle_impl *prev_;
};

inline void record_sink(std::uint32_t slot, int definition_index) {
  kernel_handle_impl *k = kernel_handle_impl::current();
  if (!k)
    throw error{errc::builder_misuse, "a result can only be assigned inside a kernel builder"};
  sink s;
  s.node = definition_index;
  s.slot = slot;
  k->add_sink(std::move(s));
}

} // namespace sg::detail

#endif
