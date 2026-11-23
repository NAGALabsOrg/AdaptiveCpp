#pragma once
#include "hipSYCL/sg/detail/graph_impl.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace sgt {
using namespace sg::detail;

struct mock_ops final : device_ops {
  std::vector<std::string> log;
  struct copy_call { int dev; void *dst; const void *src; std::size_t bytes; std::vector<op_id> deps; };
  struct fill_call { int dev; std::size_t count; std::vector<std::byte> pattern; std::vector<op_id> deps; };
  struct launch_call { int dev; kernel_launch_desc desc; std::vector<op_id> deps; };
  struct host_call { std::vector<op_id> deps; };
  std::vector<copy_call> copies;
  std::vector<fill_call> fills;
  std::vector<launch_call> launches;
  std::vector<host_call> hosts;
  std::vector<op_id> waited;
  int wait_all_calls = 0;
  op_id next = 0;
  bool run_host_tasks = true;
  std::set<op_id> unfinished;
  device_limits limits{1024, 48 * 1024, 0xffffffffull};

  int add_device(const sycl::device &) override { return 0; }
  device_desc describe(int) override {
    device_desc d;
    d.total_memory = 1 << 30;
    d.max_single_alloc = 1 << 30;
    d.limits = limits;
    d.context = 1;
    return d;
  }
  void *allocate(int, std::size_t) override { return nullptr; }
  void deallocate(int, void *) override {}
  op_id copy(int dev, void *dst, const void *src, std::size_t bytes, const std::vector<op_id> &deps) override {
    log.push_back("copy");
    copies.push_back({dev, dst, src, bytes, deps});
    return ++next;
  }
  op_id fill(int dev, void *, std::vector<std::byte> pattern, std::size_t count,
             const std::vector<op_id> &deps) override {
    log.push_back("fill");
    fills.push_back({dev, count, std::move(pattern), deps});
    return ++next;
  }
  op_id host_task(std::function<void()> fn, const std::vector<op_id> &deps) override {
    log.push_back("host_task");
    hosts.push_back({deps});
    if (run_host_tasks && fn)
      fn();
    return ++next;
  }
  op_id launch(int dev, const kernel_launch_desc &d, const std::vector<op_id> &deps) override {
    log.push_back("launch");
    launches.push_back({dev, d, deps});
    return ++next;
  }
  bool finished(op_id id) override { return !unfinished.count(id); }
  void wait(const std::vector<op_id> &ids) override {
    log.push_back("wait");
    for (op_id i : ids) {
      waited.push_back(i);
      unfinished.erase(i);
    }
  }
  void wait_all() override {
    log.push_back("wait_all");
    ++wait_all_calls;
  }
};

struct mock_memory final : memory_service {
  std::vector<std::string> log;
  std::vector<handle_state> handles;
  std::vector<char> arena = std::vector<char>(1 << 16);
  std::map<std::size_t, migration> forced_migration;
  std::size_t add_handle(std::uint64_t type, std::size_t elem) override {
    handle_state h;
    h.type = type;
    h.elem_bytes = elem;
    handles.push_back(h);
    return handles.size() - 1;
  }
  const handle_state &handle(std::size_t i) const override { return handles.at(i); }
  placement for_producer(std::size_t h, int dev, std::size_t count) override {
    log.push_back("for_producer " + std::to_string(h));
    placement p;
    p.mem.device = dev;
    p.mem.ptr = arena.data() + 4096 * h;
    p.mem.capacity = count * handles[h].elem_bytes;
    handles[h].mem = p.mem;
    return p;
  }
  std::optional<migration> for_reader(std::size_t h, int) override {
    log.push_back("for_reader " + std::to_string(h));
    auto it = forced_migration.find(h);
    if (it == forced_migration.end())
      return std::nullopt;
    return it->second;
  }
  void commit_migration(std::size_t h, const migration &, op_id) override {
    log.push_back("commit_migration " + std::to_string(h));
  }
  void note_use(std::size_t h, op_id op) override {
    log.push_back("note_use " + std::to_string(h) + " " + std::to_string(op));
  }
  void note_produced(std::size_t h, std::size_t count, op_id op) override {
    log.push_back("note_produced " + std::to_string(h) + " " + std::to_string(op));
    handles[h].produced = true;
    handles[h].count = count;
  }
  placement acquire_toolchain(int dev, std::size_t) override {
    log.push_back("acquire_toolchain");
    placement p;
    p.mem.device = dev;
    p.mem.ptr = arena.data() + 60000;
    return p;
  }
  void release_toolchain(int, const placement &, op_id op) override {
    log.push_back("release_toolchain " + std::to_string(op));
  }
  int released = 0;
  void release_finished() override { ++released; }
};

struct mock_prep final : kernel_prep_service {
  int calls = 0;
  std::vector<std::vector<slot_binding>> seen;
  prepared_kernel prepare(node_record &, const std::vector<slot_binding> &r, const device_limits &) override {
    ++calls;
    seen.push_back(r);
    prepared_kernel p;
    p.tree = std::make_shared<kernel_tree>();
    return p;
  }
};

struct mock_launch final : launch_service {
  std::vector<std::vector<op_id>> deps;
  int released = 0;
  device_ops *ops = nullptr;
  op_id launch(int dev, const prepared_kernel &, const std::vector<slot_binding> &,
               const std::vector<std::uint64_t> &, const placement &,
               const std::vector<op_id> &d) override {
    deps.push_back(d);
    kernel_launch_desc kd;
    return ops ? ops->launch(dev, kd, d) : 0;
  }
  void release_finished() override { ++released; }
};

struct rig {
  mock_ops *ops;
  mock_memory *mem;
  mock_prep *prep;
  mock_launch *launch;
  std::unique_ptr<sg::detail::graph_impl> impl;
  rig() {
    graph_services s;
    auto o = std::make_unique<mock_ops>();
    auto m = std::make_unique<mock_memory>();
    auto p = std::make_unique<mock_prep>();
    auto l = std::make_unique<mock_launch>();
    ops = o.get();
    mem = m.get();
    prep = p.get();
    launch = l.get();
    launch->ops = ops;
    s.ops = std::move(o);
    s.memory = std::move(m);
    s.prep = std::move(p);
    s.launch = std::move(l);
    impl = std::make_unique<sg::detail::graph_impl>(sg::graph_options{}, std::move(s));
  }
};
} // namespace sgt
