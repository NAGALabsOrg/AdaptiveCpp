// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_LAUNCHER_HPP
#define HIPSYCL_SG_DETAIL_LAUNCHER_HPP

#include "hipSYCL/sg/detail/bound_block.hpp"
#include "hipSYCL/sg/detail/device_ops.hpp"
#include "hipSYCL/sg/detail/kernel_preparer.hpp"
#include "hipSYCL/sg/detail/memory_manager.hpp"

#include <cstring>
#include <memory>
#include <vector>

namespace sg::detail {

class launch_service {
public:
  virtual ~launch_service() = default;
  virtual op_id launch(int device, const prepared_kernel &, const std::vector<slot_binding> &,
                       const std::vector<std::uint64_t> &result_address_by_layer,
                       const placement &toolchain, const std::vector<op_id> &deps) = 0;
  virtual void release_finished() = 0;
};

class launcher final : public launch_service {
public:
  explicit launcher(device_ops &ops) : ops_(ops) {}

  op_id launch(int device, const prepared_kernel &k, const std::vector<slot_binding> &by_slot,
               const std::vector<std::uint64_t> &result_address_by_layer,
               const placement &toolchain, const std::vector<op_id> &deps) override {
    const launch_plan &p = k.plan;
    if (p.grid == 0)
      return ops_.host_task([] {}, deps);

    auto staging = std::make_shared<std::vector<std::byte>>(build_bound_block(
        *k.tree, p, by_slot, result_address_by_layer,
        reinterpret_cast<std::uint64_t>(toolchain.mem.ptr)));
    kept_.push_back({staging, k.tree});

    std::byte *block = static_cast<std::byte *>(toolchain.mem.ptr);
    const op_id u = ops_.copy(device, block, staging->data(), p.bound_block_bytes, deps);
    std::vector<op_id> after{u};
    if (p.flag_words > 0)
      after.push_back(ops_.fill(device, block + p.flags_byte_offset,
                                std::vector<std::byte>(4, std::byte{0}), p.flag_words, deps));

    kernel_launch_desc d;
    d.groups = p.grid;
    d.work_group_size = p.work_group_size;
    d.local_bytes = p.local_bytes;
    std::memcpy(&d.ctx, staging->data(), sizeof(d.ctx));
    d.config = k.tree->config.get();
    return ops_.launch(device, d, after);
  }

  void release_finished() override { kept_.clear(); }

private:
  struct keep {
    std::shared_ptr<std::vector<std::byte>> bytes;
    std::shared_ptr<kernel_tree> tree;
  };
  device_ops &ops_;
  std::vector<keep> kept_;
};

} // namespace sg::detail

#endif
