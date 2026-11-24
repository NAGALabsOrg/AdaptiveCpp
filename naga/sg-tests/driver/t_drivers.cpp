#include "world.hpp"

#include <chrono>
#include <cstdlib>
#include <string>

using namespace sgd;

namespace {

// out[k] = out[k-1] + 1 (forward) or out[k] = out[k+1] + 1 (reverse), written by item 0
struct chain_policy {
  static void dispatch(const std::byte *block, std::uint32_t rec, std::uint64_t, const void *nd, void *) {
    const sycl::nd_item<1> &it = *static_cast<const sycl::nd_item<1> *>(nd);
    if (it.get_local_id(0) != 0)
      return;
    const task_rec &r = tasks_of(block)[rec];
    std::int64_t *out = reinterpret_cast<std::int64_t *>(r.out_global);
    const std::int64_t k = r.tile_idx;
    if (r.run_kind == 0)
      out[k] = (k == 0 ? 0 : out[k - 1]) + 1;
    else
      out[k] = (k == r.partial_stride - 1 ? 0 : out[k + 1]) + 1;
  }
};

struct pair_policy {
  static void dispatch(const std::byte *block, std::uint32_t rec, std::uint64_t, const void *nd, void *) {
    const sycl::nd_item<1> &it = *static_cast<const sycl::nd_item<1> *>(nd);
    if (it.get_local_id(0) != 0)
      return;
    const task_rec &r = tasks_of(block)[rec];
    std::int64_t *out = reinterpret_cast<std::int64_t *>(r.out_global);
    const std::int64_t g = r.tile_idx;
    const std::int64_t groups = r.partial_stride;
    if (r.run_kind == 0)
      out[g] = g + 1;
    else
      out[groups + g] = (g > 0 ? out[g - 1] : 0) + out[g];
  }
};

// The view table of the data-chosen chain: element 4 is the index map the task reads through.
constexpr int index_map_view = 4;

// out[k] = out[k-1] + 1, where out[k-1] is read through a view whose tile is chosen by kernel data,
// so the only wait is the device visitor's wait on that tile's flag inside the read
struct data_chosen_chain_policy {
  static void dispatch(const std::byte *block, std::uint32_t rec, std::uint64_t, const void *nd, void *local) {
    const sycl::nd_item<1> &it = *static_cast<const sycl::nd_item<1> *>(nd);
    if (it.get_local_id(0) != 0)
      return;
    const task_rec &r = tasks_of(block)[rec];
    std::int64_t *out = reinterpret_cast<std::int64_t *>(r.out_global);
    const std::int64_t k = r.tile_idx;
    const sg::detail::view_node *views = views_of(block);
    device_visitor visitor{flags_of(block)};
    std::int64_t previous = 0;
    read_view(views, index_map_view, k, 0, static_cast<const std::byte *>(local), visitor,
              reinterpret_cast<std::byte *>(&previous));
    out[k] = previous + 1;
  }
};

void run_forward_chain(std::size_t n, std::size_t groups_launched) {
  world w(n);
  std::vector<std::uint32_t> flag_of(n);
  for (std::size_t k = 0; k < n; ++k) {
    std::vector<std::uint32_t> on;
    if (k > 0)
      on.push_back(flag_of[k - 1]);
    task_rec r = w.task(static_cast<std::int64_t>(k), on);
    r.run_kind = 0;
    flag_of[k] = w.add_ticket({r});
  }
  w.upload();
  const auto t0 = std::chrono::steady_clock::now();
  w.run<chain_policy>(groups_launched, 8);
  const auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  BOOST_CHECK_LT(secs, 60.0);
  const auto out = w.read_out();
  for (std::size_t k = 0; k < n; ++k)
    BOOST_REQUIRE_EQUAL(out[k], static_cast<std::int64_t>(k) + 1);
  const auto f = w.read_flags();
  BOOST_CHECK_EQUAL(f[0], groups_launched);
}

} // namespace

BOOST_AUTO_TEST_SUITE(driver_suite)

BOOST_AUTO_TEST_CASE(a_chain_of_4096_tickets_each_waiting_on_the_one_before) { run_forward_chain(4096, 4096); }

BOOST_AUTO_TEST_CASE(a_chain_with_more_groups_launched_than_tickets_ends_the_extra_groups) {
  run_forward_chain(100, 130);
}

BOOST_AUTO_TEST_CASE(a_chain_whose_logical_order_is_the_reverse_of_the_ticket_order) {
  const std::size_t n = 4096;
  world w(n);
  // ticket t carries logical index n - 1 - t and waits on logical index n - t, the ticket before it
  std::vector<std::uint32_t> flag_of(n);
  for (std::size_t t = 0; t < n; ++t) {
    const std::int64_t k = static_cast<std::int64_t>(n - 1 - t);
    std::vector<std::uint32_t> on;
    if (t > 0)
      on.push_back(flag_of[t - 1]);
    task_rec r = w.task(k, on);
    r.run_kind = 1;
    r.partial_stride = static_cast<std::uint32_t>(n);
    flag_of[t] = w.add_ticket({r});
  }
  w.upload();
  w.run<chain_policy>(n, 8);
  const auto out = w.read_out();
  for (std::size_t k = 0; k < n; ++k)
    BOOST_REQUIRE_EQUAL(out[k], static_cast<std::int64_t>(n - k));
}

BOOST_AUTO_TEST_CASE(a_chain_of_4096_tickets_each_reading_the_one_before_through_a_tile_chosen_by_data) {
  const std::size_t n = 4096;
  world w(n);

  // ticket k reads tile k - 1; ticket 0 gets -1, which is outside and reads the fill value 0
  std::vector<std::int64_t> chosen(n);
  for (std::size_t k = 0; k < n; ++k)
    chosen[k] = static_cast<std::int64_t>(k) - 1;
  w.upload_indices(chosen);

  // view 0: the chosen indices as elements
  sg::detail::view_node indices_source;
  indices_source.kind = static_cast<std::uint32_t>(sg::detail::vk::source);
  indices_source.elem_bytes = 8;
  indices_source.shape_rank = 1;
  indices_source.shape[0] = static_cast<std::int64_t>(n);
  indices_source.base = reinterpret_cast<std::uint64_t>(w.indices);
  w.add_view(indices_source);

  // view 1: the indices cut into tiles of one
  sg::detail::view_node indices_tiles;
  indices_tiles.kind = static_cast<std::uint32_t>(sg::detail::vk::tile);
  indices_tiles.parent = 0;
  indices_tiles.elem_bytes = 8;
  indices_tiles.grid_rank = 1;
  indices_tiles.shape_rank = 1;
  indices_tiles.grid[0] = static_cast<std::int64_t>(n);
  w.add_view(indices_tiles);

  // view 2: the results, one tile of one element per ticket, with flags from slot 1 on
  sg::detail::view_node results;
  results.kind = static_cast<std::uint32_t>(sg::detail::vk::result);
  results.elem_bytes = 8;
  results.grid_rank = 1;
  results.shape_rank = 1;
  results.grid[0] = static_cast<std::int64_t>(n);
  results.base = reinterpret_cast<std::uint64_t>(w.out);
  results.tile_elems = 1;
  results.wait_flags = 1;
  results.flag_base = 1;
  results.segs = 1;
  w.add_view(results);

  // view 3: the result tiles as elements
  sg::detail::view_node elements;
  elements.kind = static_cast<std::uint32_t>(sg::detail::vk::join);
  elements.parent = 2;
  elements.elem_bytes = 8;
  elements.shape_rank = 1;
  elements.shape[0] = static_cast<std::int64_t>(n);
  w.add_view(elements);

  // view 4: the elements picked by the chosen indices
  sg::detail::view_node picked;
  picked.kind = static_cast<std::uint32_t>(sg::detail::vk::index_map);
  picked.parent = 3;
  picked.parent2 = 1;
  picked.elem_bytes = 8;
  picked.oob = static_cast<std::uint32_t>(sg::detail::oob_kind::fill);
  w.add_view(picked);

  // every ticket has an empty wait list
  for (std::size_t k = 0; k < n; ++k) {
    task_rec r = w.task(static_cast<std::int64_t>(k), {});
    w.add_ticket({r});
  }
  w.upload();

  const auto t0 = std::chrono::steady_clock::now();
  w.run<data_chosen_chain_policy>(n, 8);
  const auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  BOOST_CHECK_LT(secs, 60.0);

  const auto out = w.read_out();
  for (std::size_t k = 0; k < n; ++k)
    BOOST_REQUIRE_EQUAL(out[k], static_cast<std::int64_t>(k) + 1);
}

BOOST_AUTO_TEST_CASE(two_tasks_per_group_with_the_second_waiting_on_the_previous_group) {
  const std::size_t groups = 1024;
  world w(2 * groups);
  std::vector<std::uint32_t> first_flag(groups);
  for (std::size_t g = 0; g < groups; ++g) {
    task_rec a = w.task(static_cast<std::int64_t>(g), {});
    a.run_kind = 0;
    a.partial_stride = static_cast<std::uint32_t>(groups);
    std::vector<std::uint32_t> on;
    if (g > 0)
      on.push_back(first_flag[g - 1]);
    task_rec b = w.task(static_cast<std::int64_t>(g), on);
    b.run_kind = 1;
    b.partial_stride = static_cast<std::uint32_t>(groups);
    // the flag of the first task is the one the next group's second task waits on
    const std::uint32_t before = w.n_flags;
    w.add_ticket({a, b});
    first_flag[g] = before;
  }
  w.upload();
  w.run<pair_policy>(groups, 8);
  const auto out = w.read_out();
  for (std::size_t g = 0; g < groups; ++g) {
    BOOST_REQUIRE_EQUAL(out[g], static_cast<std::int64_t>(g) + 1);
    BOOST_REQUIRE_EQUAL(out[groups + g], static_cast<std::int64_t>(2 * g + 1));
  }
}

BOOST_AUTO_TEST_CASE(the_chain_passes_with_one_openmp_thread) {
  const char *threads = std::getenv("OMP_NUM_THREADS");
  const sycl::device d = sycl::queue{}.get_device();
  if (d.get_backend() != sycl::backend::omp || !threads || std::string(threads) != "1") {
    BOOST_TEST_MESSAGE("runs only on omp with OMP_NUM_THREADS=1");
    return;
  }
  run_forward_chain(4096, 4096);
}

BOOST_AUTO_TEST_SUITE_END()
