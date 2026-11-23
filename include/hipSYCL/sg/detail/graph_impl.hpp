// SPDX-License-Identifier: BSD-2-Clause
// The graph runtime: recording, ordering, memory, kernel and host-work issue.
#ifndef HIPSYCL_SG_DETAIL_GRAPH_IMPL_HPP
#define HIPSYCL_SG_DETAIL_GRAPH_IMPL_HPP

#include "hipSYCL/sg/detail/dependency.hpp"
#include "hipSYCL/sg/detail/device_ops.hpp"
#include "hipSYCL/sg/detail/kernel_preparer.hpp"
#include "hipSYCL/sg/detail/launcher.hpp"
#include "hipSYCL/sg/detail/memory_manager.hpp"
#include "hipSYCL/sg/detail/role_check.hpp"
#include "hipSYCL/sg/future.hpp"
#include "hipSYCL/sg/graph_options.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace sg::detail {

struct graph_services {
  std::unique_ptr<device_ops> ops;
  std::unique_ptr<memory_service> memory;
  std::unique_ptr<kernel_prep_service> prep;
  std::unique_ptr<launch_service> launch;
};

class graph_impl {
public:
  graph_impl(const graph_options &, graph_services s) : svc_(std::move(s)) {}
  graph_impl(const graph_impl &) = delete;
  graph_impl &operator=(const graph_impl &) = delete;
  ~graph_impl() {
    try {
      recording_.clear();
      wait();
    } catch (...) {
    }
  }

  std::size_t add_handle(std::uint64_t type, std::size_t elem_bytes) {
    return svc_.memory->add_handle(type, elem_bytes);
  }

  std::size_t add_promise(std::shared_ptr<void> state, std::function<bool()> is_set) {
    promises_.push_back({std::move(state), std::move(is_set)});
    return promises_.size() - 1;
  }

  node_record &begin_scope(const sycl::device &dev) {
    if (scope_)
      throw error{errc::builder_misuse, "submit was called inside another submit scope"};
    scope_ = std::make_unique<node_record>();
    scope_->owner = this;
    scope_->device = svc_.ops->add_device(dev);
    set_current_scope(scope_.get());
    return *scope_;
  }
  void end_scope() {
    set_current_scope(nullptr);
    std::unique_ptr<node_record> n = std::move(scope_);
    if (!n)
      return;
    if (n->defines != 1)
      throw error{errc::builder_misuse,
                  n->defines == 0 ? "submit needs exactly one define_kernel or define_host_work"
                                  : "submit takes only one define_kernel or define_host_work"};
    recording_.push_back(std::move(n));
  }
  void abort_scope() noexcept {
    set_current_scope(nullptr);
    scope_.reset();
  }

  void run() {
    std::vector<std::unique_ptr<node_record>> nodes = std::move(recording_);
    recording_.clear();
    for (auto &n : nodes)
      run_node(*n);
  }

  void wait() {
    svc_.ops->wait_all();
    svc_.launch->release_finished();
    svc_.memory->release_finished();
    staged_.clear();
    host_ops_.clear();
    next_wait_ = 0;
  }

private:
  struct promise_entry {
    std::shared_ptr<void> state;
    std::function<bool()> is_set;
  };
  struct slot_work {
    bool producer = false, reader = false;
    role_effect effect;
    std::size_t count = 0;
  };

  const device_desc &describe(int dev) {
    if (static_cast<std::size_t>(dev) >= descs_.size())
      descs_.resize(dev + 1);
    if (!descs_[dev])
      descs_[dev] = svc_.ops->describe(dev);
    return *descs_[dev];
  }

  std::uint64_t promise_value(std::size_t p) {
    return static_cast<promise_base *>(promises_[p].state.get())->as_u64();
  }

  void await_futures(const node_record &n) {
    for (std::size_t p : n.futures_read) {
      while (!promises_[p].is_set()) {
        while (next_wait_ < host_ops_.size() && svc_.ops->finished(host_ops_[next_wait_]))
          ++next_wait_;
        if (promises_[p].is_set())
          break;
        if (next_wait_ >= host_ops_.size())
          throw error{errc::future_unset,
                      "a future was read but its promise is never set by issued work"};
        svc_.ops->wait({host_ops_[next_wait_]});
        ++next_wait_;
      }
    }
  }

  static interval bytes_of_slot(const array_slot &s) {
    return {s.offset * s.elem_bytes, (s.offset + slot_count_of(s)) * s.elem_bytes};
  }

  static std::vector<op_id> merged(std::vector<op_id> a, const std::vector<op_id> &b) {
    a.insert(a.end(), b.begin(), b.end());
    std::sort(a.begin(), a.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
    a.erase(std::remove(a.begin(), a.end(), op_id{0}), a.end());
    return a;
  }

  void migrate(std::size_t handle, const migration &m) {
    const handle_state &h = svc_.memory->handle(handle);
    const std::size_t bytes = h.count * h.elem_bytes;
    std::vector<op_id> deps = merged(
        m.wait_for, tracker_.resolve({access{handle, {0, bytes}, access_mode::read, -1}}));
    op_id last = 0;
    if (m.plan.kind == copy_kind::direct) {
      const copy_step &st = m.plan.steps[0];
      last = svc_.ops->copy(st.exec_device, m.to.ptr, m.from.ptr, bytes, deps);
    } else if (m.plan.kind == copy_kind::staged_via_host) {
      auto buf = std::make_shared<std::vector<std::byte>>(bytes);
      staged_.push_back(buf);
      const op_id down =
          svc_.ops->copy(m.plan.steps[0].exec_device, buf->data(), m.from.ptr, bytes, deps);
      last = svc_.ops->copy(m.plan.steps[1].exec_device, m.to.ptr, buf->data(), bytes, {down});
    }
    svc_.memory->commit_migration(handle, m, last);
    tracker_.commit(last, {access{handle, {0, bytes}, access_mode::write, -1}});
    if (last)
      copy_ops_for_node_.push_back(last);
  }

  void run_node(node_record &n) {
    await_futures(n);

    std::vector<array_slot> resolved = n.slots;
    std::vector<slot_binding> bindings(resolved.size());
    for (std::size_t s = 0; s < resolved.size(); ++s)
      for (int d = 0; d < 3; ++d) {
        std::uint64_t e = resolved[s].extent[d];
        if (e & (1ull << 63))
          e = promise_value(n.futures_read.at(e & ~(1ull << 63)));
        resolved[s].extent[d] = e;
        bindings[s].extent[d] = e;
      }

    const std::vector<role_effect> effects = check_roles(table_, resolved, n.device, next_group_);
    copy_ops_for_node_.clear();
    std::vector<op_id> wait_for;

    // memory: producers allocate, readers migrate
    for (std::size_t s = 0; s < resolved.size(); ++s) {
      const array_slot &sl = resolved[s];
      const std::size_t count = slot_count_of(sl);
      const int place_dev = sl.is_host ? -1 : n.device;
      const bool reuse = sl.mode == role_mode::inplace && sl.init == init_kind::reuse;
      if (sl.mode == role_mode::read || reuse) {
        if (auto m = svc_.memory->for_reader(sl.handle, place_dev))
          migrate(sl.handle, *m);
      } else {
        placement p = svc_.memory->for_producer(sl.handle, place_dev, count);
        wait_for = merged(wait_for, p.wait_for);
      }
    }

    // initializer fills
    for (std::size_t s = 0; s < resolved.size(); ++s) {
      if (!effects[s].needs_fill)
        continue;
      const array_slot &sl = resolved[s];
      const std::size_t count = slot_count_of(sl);
      const handle_state &h = svc_.memory->handle(sl.handle);
      const std::size_t bytes = count * sl.elem_bytes;
      const std::vector<op_id> deps = merged(
          wait_for, tracker_.resolve({access{sl.handle, {0, bytes}, access_mode::write, -1}}));
      const op_id f = svc_.ops->fill(n.device, h.mem.ptr, sl.init_bits, count, deps);
      tracker_.commit(f, {access{sl.handle, {0, bytes}, access_mode::write, -1}});
      svc_.memory->note_use(sl.handle, f);
      copy_ops_for_node_.push_back(f);
    }

    // accesses and dependencies
    std::vector<access> accesses;
    for (std::size_t s = 0; s < resolved.size(); ++s) {
      const array_slot &sl = resolved[s];
      access a;
      a.handle = sl.handle;
      a.bytes = bytes_of_slot(sl);
      a.mode = sl.mode == role_mode::read ? access_mode::read
               : sl.mode == role_mode::produce ? access_mode::write
                                               : access_mode::inplace;
      a.group = effects[s].group;
      accesses.push_back(a);
      const handle_state &h = svc_.memory->handle(sl.handle);
      bindings[s].address = reinterpret_cast<std::uint64_t>(h.mem.ptr) + a.bytes.begin;
    }
    std::vector<op_id> deps = merged(tracker_.resolve(accesses), wait_for);
    deps = merged(deps, copy_ops_for_node_);

    op_id op = 0;
    placement toolchain;
    if (n.kind == node_record::kind_t::kernel) {
      const device_limits limits = describe(n.device).limits;
      prepared_kernel pk = svc_.prep->prepare(n, bindings, limits);
      toolchain = svc_.memory->acquire_toolchain(n.device, pk.plan.toolchain_bytes);
      deps = merged(deps, toolchain.wait_for);
      std::vector<std::uint64_t> result_addr(pk.tree->order.size(), 0);
      for (std::size_t pos = 0; pos < pk.tree->order.size(); ++pos) {
        const durable_layer &l = pk.tree->durable.layers[pk.tree->order[pos]];
        if (l.result_backed)
          result_addr[pos] = bindings.at(l.slot).address;
      }
      op = svc_.launch->launch(n.device, pk, bindings, result_addr, toolchain, deps);
    } else {
      if (n.host_table) {
        for (std::size_t s = 0; s < resolved.size(); ++s)
          if (resolved[s].is_host && s < n.host_table->address.size()) {
            n.host_table->address[s] = reinterpret_cast<void *>(bindings[s].address);
            for (int d = 0; d < 3; ++d)
              n.host_table->extent[s][d] = resolved[s].extent[d];
          }
      }
      op = svc_.ops->host_task(n.host_fn, deps);
      host_ops_.push_back(op);
    }

    tracker_.commit(op, accesses);
    for (std::size_t s = 0; s < resolved.size(); ++s) {
      const array_slot &sl = resolved[s];
      svc_.memory->note_use(sl.handle, op);
      if (sl.mode != role_mode::read && !(sl.mode == role_mode::inplace && sl.init == init_kind::reuse))
        svc_.memory->note_produced(sl.handle, slot_count_of(sl), op);
      else if (sl.mode == role_mode::inplace)
        svc_.memory->note_produced(sl.handle, slot_count_of(sl), op);
    }
    if (n.kind == node_record::kind_t::kernel)
      svc_.memory->release_toolchain(n.device, toolchain, op);
  }

  graph_services svc_;
  dependency_tracker tracker_;
  handle_table table_;
  int next_group_ = 0;
  std::vector<promise_entry> promises_;
  std::vector<std::unique_ptr<node_record>> recording_;
  std::unique_ptr<node_record> scope_;
  std::vector<std::optional<device_desc>> descs_;
  std::vector<op_id> host_ops_;
  std::size_t next_wait_ = 0;
  std::vector<std::shared_ptr<std::vector<std::byte>>> staged_;
  std::vector<op_id> copy_ops_for_node_;
};

} // namespace sg::detail

#endif
