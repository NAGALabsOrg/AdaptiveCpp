// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_LAUNCHER_HPP
#define HIPSYCL_SG_DETAIL_LAUNCHER_HPP

#include "hipSYCL/sg/detail/bound_block.hpp"
#include "hipSYCL/sg/detail/device_ops.hpp"
#include "hipSYCL/sg/detail/launch_plan.hpp"
#include "hipSYCL/sg/detail/memory_manager.hpp"

#include <cstring>
#include <memory>
#include <vector>

namespace sg::detail {

class launch_service {
public:
  virtual ~launch_service() = default;
  // Uploads the plan's block into the toolchain region, clears the flags, launches.
  virtual op_id launch(int device, const launch_plan &, const placement &toolchain,
                       const hipsycl::glue::sscp::composition_config *config,
                       const std::vector<op_id> &deps) = 0;
  virtual void release_finished() = 0;
};

class launcher final : public launch_service {
public:
  explicit launcher(device_ops &ops) : ops_(ops) {}

  op_id launch(int device, const launch_plan &p, const placement &toolchain,
               const hipsycl::glue::sscp::composition_config *config, const std::vector<op_id> &deps) override {
    if (!p.items && p.groups == 0)
      return ops_.host_task([] {}, deps);
    if (p.items && p.n_items == 0)
      return ops_.host_task([] {}, deps);

    auto staging = std::make_shared<std::vector<std::byte>>(p.block);
    kept_.push_back(staging);

    std::byte *block = static_cast<std::byte *>(toolchain.mem.ptr);
    const op_id u = ops_.copy(device, block, staging->data(), staging->size(), deps);
    std::vector<op_id> after{u};
    if (p.flag_words > 0)
      after.push_back(ops_.fill(device, block + p.flags_byte_offset, std::vector<std::byte>(4, std::byte{0}),
                                p.flag_words, deps));
    return ops_.launch(device, make_launch_desc(p, reinterpret_cast<std::uintptr_t>(block), config), after);
  }

  void release_finished() override { kept_.clear(); }

private:
  device_ops &ops_;
  std::vector<std::shared_ptr<std::vector<std::byte>>> kept_;
};

} // namespace sg::detail

#endif
