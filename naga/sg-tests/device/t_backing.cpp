#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include <vector>

using namespace sgt;

BOOST_AUTO_TEST_SUITE(backing)

BOOST_AUTO_TEST_CASE(marked_bands_in_local_memory_give_the_matrix_product) {
  if (test_device().get_info<sycl::info::device::max_work_group_size>() < 1024) {
    BOOST_TEST_MESSAGE("SKIPPED: the device's largest work-group is below 1024");
    return;
  }
  std::vector<int> a(128 * 128), b(128 * 128);
  for (int i = 0; i < 128; ++i)
    for (int k = 0; k < 128; ++k) {
      a[i * 128 + k] = i;
      b[i * 128 + k] = k;
    }
  sg::graph g;
  auto ah = g.create_mem_handle<int>();
  auto bh = g.create_mem_handle<int>();
  auto ch = g.create_mem_handle<int>();
  host_write(g, ah, a);
  host_write(g, bh, b);
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k6(kh, ah, bh, ch, false); });
  std::vector<int> c;
  host_read(g, ch, 128 * 128, c);
  g.run();
  g.wait();
  BOOST_CHECK_EQUAL(c[3 * 128 + 5], 1920);
  BOOST_CHECK_EQUAL(c[127 * 128 + 127], 2064512);
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j)
      BOOST_REQUIRE_EQUAL(c[i * 128 + j], 128 * i * j);
}

BOOST_AUTO_TEST_CASE(private_values_of_the_chain_a_to_e_give_d_equal_3i_plus_1) {
  sg::graph g;
  auto x = g.create_mem_handle<int>();
  auto b = g.create_mem_handle<int>(), c = g.create_mem_handle<int>(), d = g.create_mem_handle<int>(),
       f = g.create_mem_handle<int>();
  host_write(g, x, iota_i(16));
  g.submit(test_device(), [&](sg::kernel_handle &kh) { k8(kh, x, b, c, d, f); });
  std::vector<int> od;
  host_read(g, d, 16, od);
  g.run();
  g.wait();
  for (int i = 0; i < 16; ++i)
    BOOST_CHECK_EQUAL(od[i], 3 * i + 1);
}

BOOST_AUTO_TEST_SUITE_END()
