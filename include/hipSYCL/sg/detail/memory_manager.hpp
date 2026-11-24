// SPDX-License-Identifier: BSD-2-Clause
// Handle memory: one home per handle, capacity-based reuse, migration plans.
// Fit and placement never depend on whether operations have finished.
#ifndef HIPSYCL_SG_DETAIL_MEMORY_MANAGER_HPP
#define HIPSYCL_SG_DETAIL_MEMORY_MANAGER_HPP

#include "hipSYCL/sg/detail/copy_plan.hpp"
#include "hipSYCL/sg/detail/dependency.hpp"
#include "hipSYCL/sg/detail/pool.hpp"

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace sg::detail {

struct block {
  int device = -1; // -1: host
  void *ptr = nullptr;
  std::size_t capacity = 0;
  pool_block alloc;
};

struct placement {
  block mem;
};

struct migration {
  copy_plan plan;
  block from, to;
  std::vector<op_id> wait_for;
};

struct handle_state {
  std::uint64_t type = 0;
  std::size_t elem_bytes = 0, count = 0;
  bool produced = false;
  block mem;
  place home;
  std::vector<op_id> users;
  op_id last_producer = 0;
};

class memory_service {
public:
  virtual ~memory_service() = default;
  virtual std::size_t add_handle(std::uint64_t type, std::size_t elem_bytes) = 0;
  virtual const handle_state &handle(std::size_t) const = 0;
  // deps of the allocating requests: the operations the work that will use the memory already depends on
  virtual placement for_producer(std::size_t h, int device, std::size_t count, const std::vector<op_id> &deps) = 0;
  virtual std::optional<migration> for_reader(std::size_t h, int device, const std::vector<op_id> &deps) = 0;
  virtual void commit_migration(std::size_t h, const migration &, op_id copy_op) = 0;
  virtual void note_use(std::size_t h, op_id) = 0;
  virtual void note_produced(std::size_t h, std::size_t count, op_id) = 0;
  virtual placement acquire_toolchain(int device, std::size_t bytes, const std::vector<op_id> &deps) = 0;
  virtual void release_toolchain(int device, const placement &, op_id last_user) = 0;
  virtual void release_finished() = 0;
};

class memory_manager final : public memory_service {
public:
  struct services {
    std::function<pool &(int)> pool_of;
    std::function<void *(std::size_t)> host_alloc;
    std::function<void(void *)> host_free;
    std::function<void(const std::vector<op_id> &)> wait_ops;
    std::function<place(int)> place_of;
  };

  explicit memory_manager(services s) : s_(std::move(s)) {}
  ~memory_manager() override {
    for (handle_state &h : handles_)
      try {
        release_block(h.mem, h.users);
      } catch (...) {
      }
    release_finished();
  }

  void release_finished() override {
    for (const deferred_host &d : deferred_)
      s_.host_free(d.ptr);
    deferred_.clear();
  }

  std::size_t add_handle(std::uint64_t type, std::size_t elem_bytes) override {
    handle_state h;
    h.type = type;
    h.elem_bytes = elem_bytes;
    h.home = place{};
    handles_.push_back(std::move(h));
    return handles_.size() - 1;
  }
  const handle_state &handle(std::size_t i) const override { return handles_.at(i); }

  placement for_producer(std::size_t i, int device, std::size_t count, const std::vector<op_id> &deps) override {
    handle_state &h = handles_.at(i);
    const std::size_t bytes = count * h.elem_bytes;
    placement p;
    if (bytes == 0) {
      p.mem.device = device;
      return p;
    }
    if (h.mem.ptr && h.mem.device == device && bytes <= h.mem.capacity) {
      p.mem = h.mem;
      return p;
    }
    p = allocate_on(device, bytes, deps, "handle " + std::to_string(i));
    release_block(h.mem, h.users);
    h.mem = p.mem;
    h.home = s_.place_of(device);
    h.users.clear();
    return p;
  }

  std::optional<migration> for_reader(std::size_t i, int device, const std::vector<op_id> &deps) override {
    const handle_state &h = handles_.at(i);
    const place target = s_.place_of(device);
    const std::size_t bytes = h.count * h.elem_bytes;
    if (h.home == target || bytes == 0 || !h.mem.ptr)
      return std::nullopt;
    std::vector<op_id> copy_deps = deps;
    if (h.last_producer != 0)
      copy_deps.push_back(h.last_producer);
    placement a = allocate_on(device, bytes, copy_deps, "handle " + std::to_string(i));
    migration m;
    m.from = h.mem;
    m.to = a.mem;
    m.plan = plan_copy(h.home, target, interval{0, bytes});
    if (h.last_producer != 0)
      m.wait_for.push_back(h.last_producer);
    std::sort(m.wait_for.begin(), m.wait_for.end());
    m.wait_for.erase(std::unique(m.wait_for.begin(), m.wait_for.end()), m.wait_for.end());
    return m;
  }

  void commit_migration(std::size_t i, const migration &m, op_id copy_op) override {
    handle_state &h = handles_.at(i);
    std::vector<op_id> users = h.users;
    users.push_back(copy_op);
    release_block(m.from, users);
    h.mem = m.to;
    h.home = s_.place_of(m.to.device);
    h.users = {copy_op};
  }

  void note_use(std::size_t i, op_id op) override { handles_.at(i).users.push_back(op); }
  void note_produced(std::size_t i, std::size_t count, op_id op) override {
    handle_state &h = handles_.at(i);
    h.produced = true;
    h.count = count;
    h.last_producer = op;
    h.users.push_back(op);
  }

  placement acquire_toolchain(int device, std::size_t bytes, const std::vector<op_id> &deps) override {
    return allocate_on(device, bytes, deps, "the toolchain memory of a launch");
  }
  void release_toolchain(int device, const placement &p, op_id last_user) override {
    (void)device;
    release_block(p.mem, {last_user});
  }

private:
  placement allocate_on(int device, std::size_t bytes, const std::vector<op_id> &deps, const std::string &what) {
    placement p;
    p.mem.device = device;
    if (bytes == 0)
      return p;
    if (device < 0) {
      p.mem.ptr = s_.host_alloc(bytes);
      p.mem.capacity = bytes;
    } else {
      try {
        pool_block b = s_.pool_of(device).allocate(bytes, deps);
        p.mem.ptr = b.ptr;
        p.mem.capacity = b.nodes * pool::node_bytes;
        p.mem.alloc = std::move(b);
      } catch (const error &e) {
        if (e.code() != errc::allocation_refused)
          throw;
        throw error{errc::allocation_refused, what + ": the device refused " + std::to_string(bytes) +
                                                  " bytes (" + e.what() + ")"};
      }
    }
    return p;
  }
  void release_block(const block &b, const std::vector<op_id> &users) {
    if (!b.ptr)
      return;
    if (b.device < 0)
      deferred_.push_back(deferred_host{b.ptr, users});
    else
      s_.pool_of(b.device).release(b.alloc, users);
  }

  struct deferred_host {
    void *ptr;
    std::vector<op_id> users;
  };

  services s_;
  std::vector<handle_state> handles_;
  std::vector<deferred_host> deferred_;
};

} // namespace sg::detail

#endif
