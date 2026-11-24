#include "../common/check.hpp"
#include "../common/graphs.hpp"

#include <vector>

using namespace sgt;

namespace {

template <class F, class G> void two_steps(sg::kernel_handle &kh, H<float> x_in, H<float> y_out, F f, G g) {
  auto x = kh.source(x_in, sg::shape<1>({64}));
  auto items = kh.work_items({64});
  auto a = kh.map_items(items, [=](auto item) {
    auto v = x[item.id()];
    return item.compute(v, f);
  });
  auto first = a.named("first");
  auto b = kh.map_items(items, [=](auto item) {
    auto v = first[item.id()];
    return item.compute(v, g);
  });
  kh.evaluate(y_out, b);
}

template <class F, class G> std::vector<float> run_two(F f, G g, float start) {
  sg::graph gr;
  auto x = gr.create_mem_handle<float>();
  auto y = gr.create_mem_handle<float>();
  host_write(gr, x, iota_f(64, start));
  gr.submit(test_device(), [&](sg::kernel_handle &kh) { two_steps(kh, x, y, f, g); });
  std::vector<float> out;
  host_read(gr, y, 64, out);
  gr.run();
  gr.wait();
  return out;
}

} // namespace

BOOST_AUTO_TEST_SUITE(jit_counter)

BOOST_AUTO_TEST_CASE(one_kernel_launched_twice_counts_once_and_a_second_kernel_counts_again) {
  sycl::queue q(test_device());
  float *data = sycl::malloc_shared<float>(64, q);
  const std::uint64_t base = sg::jit_compile_count();
  auto first = [=](sycl::id<1> i) { data[i[0]] = 3.f * static_cast<float>(i[0]); };
  q.parallel_for(sycl::range<1>(64), first).wait();
  q.parallel_for(sycl::range<1>(64), first).wait();
  const std::uint64_t after_first = sg::jit_compile_count();
  auto second = [=](sycl::id<1> i) { data[i[0]] = 5.f + static_cast<float>(i[0]); };
  q.parallel_for(sycl::range<1>(64), second).wait();
  const std::uint64_t after_second = sg::jit_compile_count();
  sycl::free(data, q);
  BOOST_CHECK_EQUAL(after_first - base, 1u);
  BOOST_CHECK_EQUAL(after_second - after_first, 1u);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(composition_identity)

BOOST_AUTO_TEST_CASE(same_task_list_with_other_data_adds_no_compilation_and_another_order_adds_one) {
  auto add_one = [](float v) { return v + 1.f; };
  auto times_two = [](float v) { return v * 2.f; };

  const auto first = run_two(add_one, times_two, 0.f);
  const std::uint64_t after_first = sg::jit_compile_count();
  for (int i = 0; i < 64; ++i)
    BOOST_REQUIRE_EQUAL(first[i], 2.f * (i + 1.f));

  const auto again = run_two(add_one, times_two, 100.f);
  const std::uint64_t after_again = sg::jit_compile_count();
  for (int i = 0; i < 64; ++i)
    BOOST_REQUIRE_EQUAL(again[i], 2.f * (100.f + i + 1.f));
  BOOST_CHECK_EQUAL(after_again, after_first);

  const auto swapped = run_two(times_two, add_one, 0.f);
  const std::uint64_t after_swapped = sg::jit_compile_count();
  for (int i = 0; i < 64; ++i)
    BOOST_REQUIRE_EQUAL(swapped[i], 2.f * i + 1.f);
  BOOST_CHECK_EQUAL(after_swapped, after_again + 1);
}

BOOST_AUTO_TEST_SUITE_END()
