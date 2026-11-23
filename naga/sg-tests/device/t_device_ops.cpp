#include "common.hpp"

#include "hipSYCL/sg/detail/sycl_device_ops.hpp"

using namespace sg::detail;

BOOST_AUTO_TEST_SUITE(sycl_device_ops_suite)

BOOST_AUTO_TEST_CASE(add_device_dense_and_stable) {
  sycl_device_ops ops;
  const sycl::device d = sgt::dev();
  const int a = ops.add_device(d);
  BOOST_CHECK_EQUAL(a, 0);
  BOOST_CHECK_EQUAL(ops.add_device(d), 0);
  if (auto two = sgt::two_devices()) {
    const int b = ops.add_device(two->second);
    BOOST_CHECK(b == 0 || b == 1);
    BOOST_CHECK_EQUAL(ops.add_device(two->second), b);
  }
}

BOOST_AUTO_TEST_CASE(describe_reports_device_numbers) {
  sycl_device_ops ops;
  const sycl::device d = sgt::dev();
  const int i = ops.add_device(d);
  const device_desc desc = ops.describe(i);
  BOOST_CHECK(desc.total_memory > 0);
  BOOST_CHECK(desc.max_single_alloc > 0 && desc.max_single_alloc <= desc.total_memory);
  BOOST_CHECK_EQUAL(desc.limits.max_work_group_size, d.get_info<sycl::info::device::max_work_group_size>());
  BOOST_CHECK_EQUAL(desc.limits.local_mem_bytes, d.get_info<sycl::info::device::local_mem_size>());
  BOOST_CHECK_EQUAL(desc.free_memory.has_value(), d.has(sycl::aspect::AdaptiveCpp_free_memory));
  BOOST_CHECK_EQUAL(desc.limits.max_groups,
                    d.get_backend() == sycl::backend::cuda ? 0x7fffffffull : 0xffffffffull);
}

BOOST_AUTO_TEST_CASE(allocate_copy_fill_host_task) {
  sycl_device_ops ops;
  const int i = ops.add_device(sgt::dev());
  void *p = ops.allocate(i, 256);
  BOOST_REQUIRE(p != nullptr);
  std::vector<unsigned char> host(256, 7), back(256, 0);
  const op_id up = ops.copy(i, p, host.data(), 256, {});
  const op_id f = ops.fill(i, p, {std::byte{0xAB}, std::byte{0xCD}}, 64, {up});
  const op_id down = ops.copy(i, back.data(), p, 256, {f});
  ops.wait({down});
  BOOST_CHECK(ops.finished(down));
  for (int k = 0; k < 128; ++k)
    BOOST_CHECK_EQUAL(back[k], (k % 2) ? 0xCD : 0xAB);
  for (int k = 128; k < 256; ++k)
    BOOST_CHECK_EQUAL(back[k], 7);
  bool ran = false;
  const op_id h = ops.host_task([&] { ran = true; }, {down});
  ops.wait({h});
  BOOST_CHECK(ran);
  ops.wait_all();
  ops.deallocate(i, p);
}

BOOST_AUTO_TEST_CASE(copy_respects_dependencies) {
  sycl_device_ops ops;
  const int i = ops.add_device(sgt::dev());
  int *p = static_cast<int *>(ops.allocate(i, 4096 * 4));
  std::vector<int> a(4096, 1), b(4096, 2), out(4096, 0);
  const op_id c1 = ops.copy(i, p, a.data(), 4096 * 4, {});
  const op_id c2 = ops.copy(i, p, b.data(), 4096 * 4, {c1});
  const op_id c3 = ops.copy(i, out.data(), p, 4096 * 4, {c2});
  ops.wait({c3});
  for (int v : out)
    BOOST_REQUIRE_EQUAL(v, 2);
  ops.deallocate(i, p);
}

BOOST_AUTO_TEST_SUITE_END()
