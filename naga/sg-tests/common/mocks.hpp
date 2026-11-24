#pragma once
#include "hipSYCL/sg/detail/bound_block.hpp"
#include "hipSYCL/sg/detail/graph_impl.hpp"
#include "hipSYCL/sg/graph.hpp"

#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace sgt {
using namespace sg::detail;

struct aligned_arena {
  std::size_t bytes;
  char *base;
  explicit aligned_arena(std::size_t n) : bytes(n), base(static_cast<char *>(std::aligned_alloc(256, n))) {
    std::fill(base, base + n, 0);
  }
  ~aligned_arena() { std::free(base); }
  aligned_arena(const aligned_arena &) = delete;
};

// A device whose queue only logs.
struct mock_ops final : device_ops {
  std::vector<std::string> log;
  struct copy_call { int dev; void *dst; const void *src; std::size_t bytes; std::vector<op_id> deps; op_id id; };
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
    copies.push_back({dev, dst, src, bytes, deps, next + 1});
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

// Handle memory in one 256-byte aligned arena; 64 KiB per handle.
struct mock_memory final : memory_service {
  static constexpr std::size_t per_handle = 1 << 16;
  std::vector<std::string> log;
  std::vector<handle_state> handles;
  aligned_arena arena{1 << 24};
  std::size_t toolchain_at = 1 << 23;
  std::map<std::size_t, migration> forced_migration;
  std::size_t add_handle(std::uint64_t type, std::size_t elem) override {
    handle_state h;
    h.type = type;
    h.elem_bytes = elem;
    handles.push_back(h);
    return handles.size() - 1;
  }
  const handle_state &handle(std::size_t i) const override { return handles.at(i); }
  placement for_producer(std::size_t h, int dev, std::size_t count, const std::vector<op_id> &) override {
    log.push_back("for_producer " + std::to_string(h));
    placement p;
    p.mem.device = dev;
    p.mem.ptr = arena.base + per_handle * h;
    p.mem.capacity = count * handles[h].elem_bytes;
    handles[h].mem = p.mem;
    return p;
  }
  std::optional<migration> for_reader(std::size_t h, int, const std::vector<op_id> &) override {
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
  placement acquire_toolchain(int dev, std::size_t bytes, const std::vector<op_id> &) override {
    log.push_back("acquire_toolchain");
    placement p;
    p.mem.device = dev;
    p.mem.ptr = arena.base + toolchain_at;
    toolchain_at += (bytes + 255) / 256 * 256;
    if (toolchain_at > arena.bytes)
      toolchain_at = 1 << 23;
    return p;
  }
  void release_toolchain(int, const placement &, op_id op) override {
    log.push_back("release_toolchain " + std::to_string(op));
  }
  int released = 0;
  int wait_alls_at_release = -1;
  const mock_ops *ops = nullptr;
  void release_finished() override {
    ++released;
    wait_alls_at_release = ops ? ops->wait_all_calls : -1;
  }
};

// Records every launch with its plan and composition.
struct mock_launch final : launch_service {
  std::vector<std::vector<op_id>> deps;
  std::vector<launch_plan> plans;
  std::vector<kernel_launch_desc> descs;
  std::vector<const hipsycl::glue::sscp::composition_config *> configs;
  int released = 0;
  device_ops *ops = nullptr;
  std::unique_ptr<launcher> real;
  op_id launch(int dev, const launch_plan &p, const placement &t,
               const hipsycl::glue::sscp::composition_config *cfg, const std::vector<op_id> &d) override {
    deps.push_back(d);
    plans.push_back(p);
    configs.push_back(cfg);
    descs.push_back(make_launch_desc(p, reinterpret_cast<std::uintptr_t>(t.mem.ptr), cfg));
    if (!real)
      real = std::make_unique<launcher>(*ops);
    return real->launch(dev, p, t, cfg, d);
  }
  void release_finished() override { ++released; }
};

struct rig {
  mock_ops *ops;
  mock_memory *mem;
  mock_launch *launch;
  sg::detail::graph_impl *impl;
  std::unique_ptr<sg::graph> g;
  explicit rig(sg::graph_options o = {}) {
    graph_services s;
    auto op = std::make_unique<mock_ops>();
    auto m = std::make_unique<mock_memory>();
    auto l = std::make_unique<mock_launch>();
    ops = op.get();
    mem = m.get();
    launch = l.get();
    launch->ops = ops;
    mem->ops = ops;
    s.ops = std::move(op);
    s.memory = std::move(m);
    s.launch = std::move(l);
    auto owned = std::make_unique<sg::detail::graph_impl>(o, std::move(s));
    impl = owned.get();
    g = std::make_unique<sg::graph>(std::move(owned));
  }
};
} // namespace sgt
