// SPDX-License-Identifier: BSD-2-Clause
// device_ops over one out-of-order AdaptiveCpp_retargetable queue; every
// operation is submitted with AdaptiveCpp_retarget naming the device and its
// context.
#ifndef HIPSYCL_SG_DETAIL_SYCL_DEVICE_OPS_HPP
#define HIPSYCL_SG_DETAIL_SYCL_DEVICE_OPS_HPP

#include "hipSYCL/sg/detail/dev/generic_kernel.hpp"
#include "hipSYCL/sg/detail/device_ops.hpp"
#include "hipSYCL/sg/error.hpp"

#include <sycl/sycl.hpp>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sg::detail {

class sycl_device_ops final : public device_ops {
public:
  ~sycl_device_ops() override {
    try {
      if (queue_)
        queue_->wait();
    } catch (...) {
    }
  }

  int add_device(const sycl::device &d) override {
    std::lock_guard<std::mutex> lock(m_);
    for (std::size_t i = 0; i < devices_.size(); ++i)
      if (devices_[i] == d)
        return static_cast<int>(i);
    devices_.push_back(d);
    contexts_.emplace_back(d);
    return static_cast<int>(devices_.size() - 1);
  }

  device_desc describe(int dev) override {
    const sycl::device &d = devices_.at(dev);
    device_desc o;
    o.total_memory = d.get_info<sycl::info::device::global_mem_size>();
    if (d.get_backend() == sycl::backend::omp) {
      const long pages = sysconf(_SC_PHYS_PAGES), ps = sysconf(_SC_PAGE_SIZE);
      if (pages > 0 && ps > 0)
        o.total_memory = static_cast<std::size_t>(pages) * static_cast<std::size_t>(ps);
    }
    o.max_single_alloc =
        std::min<std::size_t>(d.get_info<sycl::info::device::max_mem_alloc_size>(), o.total_memory);
    if (d.has(sycl::aspect::AdaptiveCpp_free_memory))
      o.free_memory = d.get_info<sycl::info::device::AdaptiveCpp_free_memory>();
    o.limits.max_work_group_size = d.get_info<sycl::info::device::max_work_group_size>();
    o.limits.local_mem_bytes = d.get_info<sycl::info::device::local_mem_size>();
    o.limits.max_groups = d.get_backend() == sycl::backend::cuda ? 0x7fffffffull : 0xffffffffull;
    o.context = static_cast<std::uint64_t>(dev) + 1;
    return o;
  }

  void *allocate(int dev, std::size_t bytes) override {
    void *p = sycl::malloc_device(bytes, devices_.at(dev), contexts_.at(dev));
    if (!p)
      throw error{errc::allocation_refused,
                  "the device refused an allocation of " + std::to_string(bytes) + " bytes"};
    return p;
  }
  void deallocate(int dev, void *p) override { sycl::free(p, contexts_.at(dev)); }

  op_id copy(int exec_dev, void *dst, const void *src, std::size_t bytes,
             const std::vector<op_id> &deps) override {
    return submit(exec_dev, deps, [&](sycl::handler &h) { h.memcpy(dst, src, bytes); });
  }

  op_id fill(int dev, void *dst, std::vector<std::byte> pattern, std::size_t count,
             const std::vector<op_id> &deps) override {
    if (dev < 0 || count == 0 || pattern.empty())
      return host_task([] {}, deps);
    if (pattern.size() > 16)
      throw error{errc::limit_exceeded, "a fill pattern is at most 16 bytes"};
    std::array<unsigned char, 16> pat{};
    const std::size_t psz = pattern.size();
    std::memcpy(pat.data(), pattern.data(), psz);
    unsigned char *out = static_cast<unsigned char *>(dst);
    return submit(dev, deps, [=](sycl::handler &h) {
      constexpr std::size_t group = 128;
      const std::size_t rounded = (count + group - 1) / group * group;
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(rounded), sycl::range<1>(group)),
                     [=](sycl::nd_item<1> it) {
                       const std::size_t i = it.get_global_id(0);
                       if (i >= count)
                         return;
                       for (std::size_t b = 0; b < psz; ++b)
                         out[i * psz + b] = pat[b];
                     });
    });
  }

  op_id host_task(std::function<void()> fn, const std::vector<op_id> &deps) override {
    return submit(0, deps, [&](sycl::handler &h) { h.async_host(fn); });
  }

  op_id launch(int dev, const kernel_launch_desc &d, const std::vector<op_id> &deps) override {
    if (d.data_args.size() != 1)
      throw error{errc::internal, "the generic kernel takes exactly one data argument"};
    const std::uintptr_t block = d.data_args[0];
    return submit(dev, deps, [&](sycl::handler &h) {
      if (d.items) {
        item_kernel k{block, {hipsycl::glue::sscp::hide_composition_pointer(d.config)}};
        constexpr std::size_t group = 128;
        const std::size_t rounded = (static_cast<std::size_t>(d.n_items) + group - 1) / group * group;
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(rounded), sycl::range<1>(group)), k);
        return;
      }
      sycl::local_accessor<std::byte, 1> frame{sycl::range<1>(d.local_bytes), h};
      generic_kernel k{block, {hipsycl::glue::sscp::hide_composition_pointer(d.config)}, frame};
      h.parallel_for(sycl::nd_range<1>{sycl::range<1>(d.groups * d.work_group_size),
                                       sycl::range<1>(d.work_group_size)},
                     k);
    });
  }

  bool finished(op_id id) override {
    sycl::event e = event_of(id);
    return e.get_info<sycl::info::event::command_execution_status>() ==
           sycl::info::event_command_status::complete;
  }
  void wait(const std::vector<op_id> &ids) override {
    for (op_id id : ids)
      event_of(id).wait();
  }
  void wait_all() override {
    if (queue_)
      queue_->wait_and_throw();
  }

private:
  sycl::event event_of(op_id id) {
    std::lock_guard<std::mutex> lock(m_);
    auto it = events_.find(id);
    if (it == events_.end())
      throw error{errc::internal, "unknown operation id"};
    return it->second;
  }

  template <class F> op_id submit(int dev, const std::vector<op_id> &deps, F &&body) {
    std::vector<sycl::event> evs;
    sycl::queue *q;
    sycl::property_list props;
    {
      std::lock_guard<std::mutex> lock(m_);
      if (!queue_)
        queue_ = std::make_unique<sycl::queue>(
            sycl::property_list{sycl::property::queue::AdaptiveCpp_retargetable{}});
      q = queue_.get();
      for (op_id d : deps) {
        auto it = events_.find(d);
        if (it != events_.end())
          evs.push_back(it->second);
      }
      const int use = dev < 0 ? 0 : dev;
      props = sycl::property_list{sycl::property::command_group::AdaptiveCpp_retarget{
          contexts_.at(use), devices_.at(use)}};
    }
    sycl::event e = q->submit(props, [&](sycl::handler &h) {
      h.depends_on(evs);
      body(h);
    });
    std::lock_guard<std::mutex> lock(m_);
    const op_id id = ++next_;
    events_.emplace(id, e);
    return id;
  }

  std::mutex m_;
  std::vector<sycl::device> devices_;
  std::vector<sycl::context> contexts_;
  std::unique_ptr<sycl::queue> queue_;
  std::map<op_id, sycl::event> events_;
  op_id next_ = 0;
};

} // namespace sg::detail

#endif
