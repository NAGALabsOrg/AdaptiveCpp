// SPDX-License-Identifier: BSD-2-Clause
// The graph runtime. A submit records a node (its roles and, for a kernel, the
// kernel graph its builder described). The first run prepares every kernel for
// its launch values, orders its tiles and composes it; later runs rebind
// addresses and launch values, recompute only what a changed launch value
// reaches and launch the same composed kernel.
#ifndef HIPSYCL_SG_DETAIL_GRAPH_IMPL_HPP
#define HIPSYCL_SG_DETAIL_GRAPH_IMPL_HPP

#include "hipSYCL/glue/llvm-sscp/composition.hpp"
#include "hipSYCL/sg/detail/backing_plan.hpp"
#include "hipSYCL/sg/detail/dependency.hpp"
#include "hipSYCL/sg/detail/device_ops.hpp"
#include "hipSYCL/sg/detail/kernel_preparer.hpp"
#include "hipSYCL/sg/detail/launch_plan.hpp"
#include "hipSYCL/sg/detail/launcher.hpp"
#include "hipSYCL/sg/detail/memory_manager.hpp"
#include "hipSYCL/sg/detail/role_check.hpp"
#include "hipSYCL/sg/detail/tile_graph.hpp"
#include "hipSYCL/sg/graph_options.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sg::detail {

struct graph_services {
  std::unique_ptr<device_ops> ops;
  std::unique_ptr<memory_service> memory;
  std::unique_ptr<launch_service> launch;
  std::function<std::size_t()> pool_slabs; // slabs taken from the devices; empty without pools
};

// What the runtime did, for tests and measurements.
struct run_stats {
  std::size_t prepares = 0;           // kernels prepared (first run or a changed launch value)
  std::size_t unit_bodies = 0;        // unit bodies evaluated
  std::size_t compositions = 0;       // composition configurations made
  std::size_t launches = 0;           // kernel launches issued
  std::size_t launch_value_reads = 0; // launch values read on the host
};

inline std::uint64_t task_list_hash(const std::vector<std::string> &tasks) {
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&](unsigned char c) {
    h ^= c;
    h *= 1099511628211ull;
  };
  for (const std::string &t : tasks) {
    for (char c : t)
      mix(static_cast<unsigned char>(c));
    mix(0);
  }
  return h;
}

inline std::int64_t decode_launch_value(const void *p, const launch_slot_rec &l) {
  if (l.is_float) {
    if (l.bytes == 4) {
      float f;
      std::memcpy(&f, p, 4);
      return static_cast<std::int64_t>(f);
    }
    double d;
    std::memcpy(&d, p, 8);
    return static_cast<std::int64_t>(d);
  }
  switch (l.bytes) {
  case 1: { std::int8_t v; std::memcpy(&v, p, 1); return v; }
  case 2: { std::int16_t v; std::memcpy(&v, p, 2); return v; }
  case 4: { std::int32_t v; std::memcpy(&v, p, 4); return v; }
  default: { std::int64_t v; std::memcpy(&v, p, 8); return v; }
  }
}

class graph_impl {
public:
  graph_impl(const graph_options &o, graph_services s) : opts_(o), svc_(std::move(s)) {}
  graph_impl(const graph_impl &) = delete;
  graph_impl &operator=(const graph_impl &) = delete;
  ~graph_impl() {
    try {
      wait();
    } catch (...) {
    }
  }

  std::size_t add_handle(std::uint64_t type, std::size_t elem_bytes) {
    return svc_.memory->add_handle(type, elem_bytes);
  }

  bool built() const noexcept { return built_; }
  const run_stats &stats() const noexcept { return stats_; }
  std::size_t pool_slab_count() const { return svc_.pool_slabs ? svc_.pool_slabs() : 0; }
  const std::vector<std::unique_ptr<node_record>> &nodes() const noexcept { return nodes_; }

  node_record &begin_scope(const sycl::device &dev) {
    open_scope(node_record::kind_t::kernel, svc_.ops->add_device(dev));
    scope_->graph.group_size_cap = static_cast<std::int64_t>(
        opts_.group_size_cap != 0 ? opts_.group_size_cap : describe(scope_->device).limits.max_work_group_size);
    return *scope_;
  }
  node_record &begin_host_scope() {
    if (host_device_ < 0)
      host_device_ = svc_.ops->add_device(sycl::device{});
    open_scope(node_record::kind_t::host_work, host_device_);
    return *scope_;
  }
  void end_scope() {
    set_current_scope(nullptr);
    std::unique_ptr<node_record> n = std::move(scope_);
    if (!n)
      return;
    if (n->kind == node_record::kind_t::host_work && !n->host_fn)
      throw error{errc::builder_misuse, "submit<sg::host> needs host work: call run on the host handle"};
    check_node_roles(*n);
    nodes_.push_back(std::move(n));
    states_.emplace_back();
  }
  void abort_scope() noexcept {
    set_current_scope(nullptr);
    scope_.reset();
  }

  void run() {
    built_ = true;
    for (auto &h : table_)
      h.inplace_group = -1;
    for (std::size_t i = 0; i < nodes_.size(); ++i)
      run_node(*nodes_[i], states_[i]);
  }

  void wait() {
    svc_.ops->wait_all();
    svc_.launch->release_finished();
    svc_.memory->release_finished();
    staged_.clear();
  }

private:
  struct kernel_state {
    std::unique_ptr<prepared_kernel> prepared;
    tile_graph tg;
    std::vector<std::int64_t> launch_values;
    std::shared_ptr<hipsycl::glue::sscp::composition_config> config;
    std::size_t toolchain_bytes = 0;
  };

  void open_scope(node_record::kind_t kind, int device) {
    if (scope_)
      throw error{errc::builder_misuse, "submit was called inside another submit scope"};
    if (built_)
      throw error{errc::build, "the graph is already built: submit is not allowed after the first run"};
    scope_ = std::make_unique<node_record>();
    scope_->owner = this;
    scope_->kind = kind;
    scope_->device = device;
    set_current_scope(scope_.get());
  }

  const device_desc &describe(int dev) {
    if (static_cast<std::size_t>(dev) >= descs_.size())
      descs_.resize(dev + 1);
    if (!descs_[dev])
      descs_[dev] = svc_.ops->describe(dev);
    return *descs_[dev];
  }

  static void check_node_roles(const node_record &n) {
    for (std::size_t a = 0; a < n.slots.size(); ++a)
      for (std::size_t b = a + 1; b < n.slots.size(); ++b) {
        const array_slot &x = n.slots[a], &y = n.slots[b];
        if (x.handle != y.handle || (x.mode == role_mode::read && y.mode == role_mode::read))
          continue;
        throw error{errc::build, "handle " + std::to_string(x.handle) +
                                     " is used by conflicting roles in one submit: a handle is a source or a "
                                     "result of one kernel, not both"};
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

  // What a copy that makes the handle readable on another device waits for besides the handle's producer.
  std::vector<op_id> reader_deps(std::size_t handle) {
    const handle_state &h = svc_.memory->handle(handle);
    return tracker_.resolve({access{handle, {0, h.count * h.elem_bytes}, access_mode::read, -1}});
  }

  void migrate(std::size_t handle, const migration &m) {
    const handle_state &h = svc_.memory->handle(handle);
    const std::size_t bytes = h.count * h.elem_bytes;
    std::vector<op_id> deps = merged(m.wait_for, tracker_.resolve({access{handle, {0, bytes}, access_mode::read, -1}}));
    op_id last = 0;
    if (m.plan.kind == copy_kind::direct) {
      const copy_step &st = m.plan.steps[0];
      last = svc_.ops->copy(st.exec_device, m.to.ptr, m.from.ptr, bytes, deps);
    } else if (m.plan.kind == copy_kind::staged_via_host) {
      auto buf = std::make_shared<std::vector<std::byte>>(bytes);
      staged_.push_back(buf);
      const op_id down = svc_.ops->copy(m.plan.steps[0].exec_device, buf->data(), m.from.ptr, bytes, deps);
      last = svc_.ops->copy(m.plan.steps[1].exec_device, m.to.ptr, buf->data(), bytes, {down});
    }
    svc_.memory->commit_migration(handle, m, last);
    tracker_.commit(last, {access{handle, {0, bytes}, access_mode::write, -1}});
    if (last)
      copy_ops_for_node_.push_back(last);
  }

  // The number in a handle, read by the host once the work that wrote it is done.
  std::int64_t read_launch_value(const node_record &n, const launch_slot_rec &l) {
    const array_slot &s = n.slots.at(l.handle);
    const handle_state &h0 = svc_.memory->handle(s.handle);
    if (!h0.produced)
      throw error{errc::never_produced, "handle " + std::to_string(s.handle) +
                                            " is read as a launch value but nothing has produced it"};
    if (auto m = svc_.memory->for_reader(s.handle, -1, reader_deps(s.handle)))
      migrate(s.handle, *m);
    const handle_state &h = svc_.memory->handle(s.handle);
    const std::size_t bytes = h.count * h.elem_bytes;
    svc_.ops->wait(tracker_.resolve({access{s.handle, {0, bytes}, access_mode::read, -1}}));
    ++stats_.launch_value_reads;
    return decode_launch_value(h.mem.ptr, l);
  }

  void prepare_kernel(node_record &n, kernel_state &ks, const std::vector<std::int64_t> &values) {
    if (ks.prepared && ks.launch_values == values)
      return;
    auto pk = std::make_unique<prepared_kernel>(kernel_preparer::prepare(n.graph, values, ks.prepared.get()));
    tile_graph tg = build_tile_graph(*pk);
    stats_.unit_bodies += static_cast<std::size_t>(pk->unit_bodies_evaluated);
    ++stats_.prepares;
    if (!ks.config || ks.config->tasks != pk->graph.tasks) {
      ks.config = std::make_shared<hipsycl::glue::sscp::composition_config>();
      ks.config->tasks = pk->graph.tasks;
      ks.config->unique_hash = task_list_hash(pk->graph.tasks);
      ++stats_.compositions;
    }
    ks.prepared = std::move(pk);
    ks.tg = std::move(tg);
    ks.launch_values = values;
    ks.toolchain_bytes = 0;
  }

  void run_node(node_record &n, kernel_state &ks) {
    const bool kernel = n.kind == node_record::kind_t::kernel;
    std::vector<array_slot> resolved = n.slots;
    std::vector<slot_binding> bindings(resolved.size());
    copy_ops_for_node_.clear();

    eval_env env;
    std::vector<std::int64_t> values;
    if (kernel) {
      for (const launch_slot_rec &l : n.graph.launches)
        values.push_back(read_launch_value(n, l));
      prepare_kernel(n, ks, values);
      env.launch = ks.launch_values.data();
      env.launch_count = ks.launch_values.size();
      const prepared_kernel &pk = *ks.prepared;
      env.count_of = [&pk](std::int64_t view, int dim) -> std::int64_t {
        const view_node &v = pk.views.at(static_cast<std::size_t>(view));
        if (v.grid_rank > 0)
          return dim < 0 ? prod3(v.grid) : v.grid[dim];
        return dim < 0 ? prod3(v.shape) : v.shape[dim];
      };
    }
    for (std::size_t s = 0; s < resolved.size(); ++s) {
      for (int d = 0; d < 3; ++d) {
        resolved[s].extent[d] =
            d < resolved[s].ext.rank ? static_cast<std::uint64_t>(resolved[s].ext.v[d].evaluate(env)) : 1;
        bindings[s].extent[d] = resolved[s].extent[d];
      }
    }

    const std::vector<role_effect> effects = check_roles(table_, resolved, n.device, next_group_);

    // memory: producers allocate, readers migrate. A request carries what its users already wait for:
    // the writes of the handle for a producer (the initializer fill and the kernel wait for them), the
    // reads of the handle for a migration copy.
    for (std::size_t s = 0; s < resolved.size(); ++s) {
      const array_slot &sl = resolved[s];
      const std::size_t count = slot_count_of(sl);
      const int place_dev = sl.is_host ? -1 : n.device;
      const bool reuse = sl.mode == role_mode::inplace && sl.init == init_kind::reuse;
      if (sl.mode == role_mode::read || reuse) {
        if (auto m = svc_.memory->for_reader(sl.handle, place_dev, reader_deps(sl.handle)))
          migrate(sl.handle, *m);
      } else {
        svc_.memory->for_producer(
            sl.handle, place_dev, count,
            tracker_.resolve({access{sl.handle, {0, count * sl.elem_bytes}, access_mode::write, -1}}));
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
      const std::vector<op_id> deps = tracker_.resolve({access{sl.handle, {0, bytes}, access_mode::write, -1}});
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
    std::vector<op_id> deps = merged(tracker_.resolve(accesses), copy_ops_for_node_);

    op_id op = 0;
    placement toolchain;
    if (kernel) {
      const device_limits limits = describe(n.device).limits;
      if (ks.toolchain_bytes == 0)
        ks.toolchain_bytes = encode_launch(*ks.prepared, ks.tg, bindings, 0, limits).toolchain_bytes;
      toolchain = svc_.memory->acquire_toolchain(n.device, ks.toolchain_bytes, deps);
      const launch_plan plan = encode_launch(*ks.prepared, ks.tg, bindings,
                                             reinterpret_cast<std::uint64_t>(toolchain.mem.ptr), limits);
      op = svc_.launch->launch(n.device, plan, toolchain, ks.config.get(), deps);
      ++stats_.launches;
    } else {
      if (n.host_table) {
        host_slot_table bound = *n.host_table;
        for (std::size_t s = 0; s < resolved.size(); ++s)
          if (resolved[s].is_host && s < bound.address.size()) {
            bound.address[s] = reinterpret_cast<void *>(bindings[s].address);
            for (int d = 0; d < 3; ++d)
              bound.extent[s][d] = resolved[s].extent[d];
          }
        op = svc_.ops->host_task(
            [table = n.host_table, bound = std::move(bound), fn = n.host_fn] {
              *table = bound;
              fn();
            },
            deps);
      } else {
        op = svc_.ops->host_task(n.host_fn, deps);
      }
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
    if (kernel)
      svc_.memory->release_toolchain(n.device, toolchain, op);
  }

  graph_options opts_;
  graph_services svc_;
  dependency_tracker tracker_;
  handle_table table_;
  int next_group_ = 0;
  bool built_ = false;
  int host_device_ = -1;
  run_stats stats_;
  std::vector<std::unique_ptr<node_record>> nodes_;
  std::vector<kernel_state> states_;
  std::unique_ptr<node_record> scope_;
  std::vector<std::optional<device_desc>> descs_;
  std::vector<std::shared_ptr<std::vector<std::byte>>> staged_;
  std::vector<op_id> copy_ops_for_node_;
};

} // namespace sg::detail

#endif
