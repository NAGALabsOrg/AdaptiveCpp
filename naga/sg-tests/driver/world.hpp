#pragma once
#include "hipSYCL/sg/detail/dev/drivers.hpp"
#include "hipSYCL/sg/detail/dev/layout.hpp"

#include <boost/test/unit_test.hpp>

#include <sycl/sycl.hpp>

#include <cstdlib>
#include <cstring>
#include <vector>

namespace sgd {
using namespace sg::detail::dev;

// A synthetic bound block on the device: tickets, task records, wait lists and
// the flags, with no views. The test's policy decides what a task does.
struct world {
  sycl::queue q;
  std::vector<ticket_rec> tickets;
  std::vector<task_rec> tasks;
  std::vector<std::uint32_t> waits;
  std::uint32_t n_flags = 1;
  std::byte *block = nullptr;
  std::uint32_t *flags = nullptr;
  std::int64_t *out = nullptr;
  std::size_t out_count = 0;
  std::vector<sg::detail::view_node> views;
  std::int64_t *indices = nullptr;

  explicit world(std::size_t outputs) : q(sycl::default_selector_v), out_count(outputs) {
    out = sycl::malloc_device<std::int64_t>(outputs, q);
    q.memset(out, 0, outputs * sizeof(std::int64_t)).wait();
  }
  ~world() {
    sycl::free(block, q);
    sycl::free(flags, q);
    sycl::free(out, q);
    sycl::free(indices, q);
  }

  // A view descriptor for the block; returns its index in the view section.
  int add_view(const sg::detail::view_node &v) {
    views.push_back(v);
    return static_cast<int>(views.size()) - 1;
  }

  // An int64 table in device memory, which a view can take as its elements.
  void upload_indices(const std::vector<std::int64_t> &values) {
    indices = sycl::malloc_device<std::int64_t>(values.size(), q);
    q.memcpy(indices, values.data(), values.size() * sizeof(std::int64_t)).wait();
  }

  // One ticket holding the given tasks; returns the flag of its last task.
  std::uint32_t add_ticket(const std::vector<task_rec> &list) {
    ticket_rec t{};
    t.first_task = static_cast<std::uint32_t>(tasks.size());
    t.n_tasks = static_cast<std::uint32_t>(list.size());
    for (task_rec r : list) {
      r.flag = n_flags++;
      tasks.push_back(r);
    }
    tickets.push_back(t);
    return tasks.back().flag;
  }

  // A task record that waits on the given flags.
  task_rec task(std::int64_t id, const std::vector<std::uint32_t> &on) {
    task_rec r{};
    r.tile_idx = id;
    r.first_wait = static_cast<std::uint32_t>(waits.size());
    r.n_waits = static_cast<std::uint32_t>(on.size());
    for (std::uint32_t f : on)
      waits.push_back(f);
    r.out_global = reinterpret_cast<std::uint64_t>(out);
    return r;
  }

  void upload() {
    block_sizes s;
    s.tickets = tickets.size() * sizeof(ticket_rec);
    s.tasks = tasks.size() * sizeof(task_rec);
    s.waits = waits.size() * sizeof(std::uint32_t);
    s.views = views.size() * sizeof(sg::detail::view_node);
    block_header h{};
    h.magic = block_magic;
    h.n_tickets = static_cast<std::uint32_t>(tickets.size());
    h.n_tasks = static_cast<std::uint32_t>(tasks.size());
    h.n_flags = n_flags;
    const std::uint64_t bytes = lay_out(h, s);
    flags = sycl::malloc_device<std::uint32_t>(n_flags, q);
    q.memset(flags, 0, n_flags * sizeof(std::uint32_t)).wait();
    h.flags = reinterpret_cast<std::uint64_t>(flags);
    std::vector<std::byte> host(bytes);
    std::memcpy(host.data(), &h, sizeof(h));
    if (!tickets.empty())
      std::memcpy(host.data() + h.off_tickets, tickets.data(), s.tickets);
    if (!tasks.empty())
      std::memcpy(host.data() + h.off_tasks, tasks.data(), s.tasks);
    if (!waits.empty())
      std::memcpy(host.data() + h.off_waits, waits.data(), s.waits);
    if (!views.empty())
      std::memcpy(host.data() + h.off_views, views.data(), s.views);
    block = static_cast<std::byte *>(sycl::aligned_alloc_device(256, bytes, q));
    q.memcpy(block, host.data(), bytes).wait();
  }

  template <class P> void run(std::size_t native_groups, std::size_t group_size, std::size_t local_bytes = 64) {
    const std::byte *b = block;
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<std::byte, 1> local{sycl::range<1>(local_bytes), h};
      h.parallel_for(sycl::nd_range<1>(native_groups * group_size, group_size), [=](sycl::nd_item<1> it) {
        run_native_group<P>(b, it, static_cast<void *>(&local[0]));
      });
    });
    q.wait_and_throw();
  }

  std::vector<std::int64_t> read_out() {
    std::vector<std::int64_t> v(out_count);
    q.memcpy(v.data(), out, out_count * sizeof(std::int64_t)).wait();
    return v;
  }
  std::vector<std::uint32_t> read_flags() {
    std::vector<std::uint32_t> v(n_flags);
    q.memcpy(v.data(), flags, n_flags * sizeof(std::uint32_t)).wait();
    return v;
  }
};
} // namespace sgd
