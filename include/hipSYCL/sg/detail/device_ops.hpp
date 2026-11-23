// SPDX-License-Identifier: BSD-2-Clause
// The operations the graph runtime needs from devices, behind an interface so
// the runtime can be tested with mocks.
#ifndef HIPSYCL_SG_DETAIL_DEVICE_OPS_HPP
#define HIPSYCL_SG_DETAIL_DEVICE_OPS_HPP

#include "hipSYCL/glue/llvm-sscp/composition.hpp"
#include "hipSYCL/sg/detail/dependency.hpp"
#include "hipSYCL/sg/detail/dev/abi.hpp"
#include "hipSYCL/sg/detail/launch_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace sg::detail {

struct device_desc {
  std::size_t total_memory = 0, max_single_alloc = 0;
  std::optional<std::size_t> free_memory;
  device_limits limits;
  std::uint64_t context = 0;
};

struct kernel_launch_desc {
  std::size_t groups = 0, work_group_size = 1, local_bytes = 256;
  dev::launch_ctx ctx{};
  const hipsycl::glue::sscp::composition_config *config = nullptr;
};

class device_ops {
public:
  virtual ~device_ops() = default;
  virtual int add_device(const sycl::device &) = 0;
  virtual device_desc describe(int dev) = 0;
  virtual void *allocate(int dev, std::size_t bytes) = 0;
  virtual void deallocate(int dev, void *) = 0;
  virtual op_id copy(int exec_dev, void *dst, const void *src, std::size_t bytes,
                     const std::vector<op_id> &deps) = 0;
  virtual op_id fill(int dev, void *dst, std::vector<std::byte> pattern, std::size_t count,
                     const std::vector<op_id> &deps) = 0;
  virtual op_id host_task(std::function<void()>, const std::vector<op_id> &deps) = 0;
  virtual op_id launch(int dev, const kernel_launch_desc &, const std::vector<op_id> &deps) = 0;
  virtual bool finished(op_id) = 0;
  virtual void wait(const std::vector<op_id> &) = 0;
  virtual void wait_all() = 0;
};

} // namespace sg::detail

#endif
