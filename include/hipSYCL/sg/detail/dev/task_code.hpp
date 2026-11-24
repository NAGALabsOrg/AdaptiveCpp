// SPDX-License-Identifier: BSD-2-Clause
// The device code of one map: a function per (run function, input element
// types). Item i of unit u reads element i of each input run through the views,
// calls the user function and stores its result. It returns at once when its
// task number is not the one being run; units are packed in the group in unit
// order and items beyond a shorter run stay idle.
#ifndef HIPSYCL_SG_DETAIL_DEV_TASK_CODE_HPP
#define HIPSYCL_SG_DETAIL_DEV_TASK_CODE_HPP

#include "hipSYCL/sg/body.hpp"
#include "hipSYCL/sg/detail/dev/backing.hpp"
#include "hipSYCL/sg/detail/dev/iterators.hpp"
#include "hipSYCL/sg/detail/dev/layout.hpp"
#include "hipSYCL/sg/detail/dev/view_read.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>

namespace sg::detail {

template <class T> struct result_traits {
  static constexpr result_kind kind = result_kind::per_item;
  using value = T;
};
template <class T> struct result_traits<reduced<T>> {
  static constexpr result_kind kind = result_kind::reduced;
  using value = T;
};
template <class T> struct result_traits<binned<T>> {
  static constexpr result_kind kind = result_kind::binned;
  using value = T;
};

template <class Fn, class... A>
SYCL_EXTERNAL [[clang::annotate("sg_body_entry")]] void tile_task(std::uint32_t self, const std::byte *block,
                                                                    std::uint32_t recno, std::uint64_t,
                                                                    const void *ndp, void *mem) {
  using R = std::invoke_result_t<const Fn &, run_item, A...>;
  using traits = result_traits<R>;
  using V = typename traits::value;
  const dev::task_rec &r = dev::tasks_of(block)[recno];
  if (r.task_id != self)
    return;
  const sycl::nd_item<1> &nd = *static_cast<const sycl::nd_item<1> *>(ndp);
  std::byte *local = static_cast<std::byte *>(mem);
  const Fn &fn = *reinterpret_cast<const Fn *>(block + r.closure);
  dev::device_visitor vis{dev::flags_of(block)};

  dev::run_ctx c;
  c.nd = ndp;
  c.ws = local;
  c.gid = static_cast<std::uint32_t>(nd.get_local_id(0));
  c.group_size = static_cast<std::uint32_t>(nd.get_local_range(0));
  c.lanes = r.stride;
  c.n_units = r.n_units;
  c.k = c.gid % r.stride;
  const std::uint32_t unit = c.gid / r.stride;
  c.real_unit = unit < r.n_units;
  c.unit = c.real_unit ? unit : 0;
  c.kind = r.run_kind;
  c.seg = r.seg;
  c.segs = r.segs;
  c.ips = r.ips;
  c.rounds = r.rounds;
  c.run_len = r.run_len;
  c.partial_addr = r.partial_addr;
  c.partial_stride = r.partial_stride;
  c.tile_idx = r.tile_idx;

  for (std::uint32_t round = 0; round < r.rounds; ++round) {
    const std::int64_t base = r.run_kind == 1 ? static_cast<std::int64_t>(r.seg) * r.ips
                              : r.run_kind == 2 ? static_cast<std::int64_t>(round) * r.ips
                                                : 0;
    const std::int64_t remaining = r.run_len - base;
    const std::int64_t share = remaining < 0 ? 0 : (remaining < r.stride ? remaining : r.stride);
    c.round = round;
    c.pos = base + c.k;
    c.valid = c.real_unit ? share : 0;
    c.active = c.real_unit && static_cast<std::int64_t>(c.k) < share;
    c.call_off = 0;

    const dev::sel_rec *sels = dev::sels_of(block) + r.first_sel + static_cast<std::size_t>(c.unit) * r.n_inputs;
    const auto vals = [&]<std::size_t... I>(std::index_sequence<I...>) {
      return std::tuple<A...>{(c.active ? dev::read_window<A>(block, sels[I], c.pos, local, vis) : A{})...};
    }(std::index_sequence_for<A...>{});

    run_item item(&c);
    const R res = std::apply([&](const A &...a) { return fn(item, a...); }, vals);

    const bool last_round = round + 1 == r.rounds;
    const bool owner_segment = r.run_kind != 1 || r.seg + 1 == r.segs;
    if constexpr (traits::kind == result_kind::per_item) {
      if (c.active) {
        const V v = res;
        dev::store_result(r, local, static_cast<std::int64_t>(c.unit) * r.run_len + c.pos, &v);
      }
    } else if constexpr (traits::kind == result_kind::reduced) {
      if (c.real_unit && c.k == 0 && last_round && owner_segment) {
        const V v = res.value;
        dev::store_result(r, local, c.unit, &v);
      }
    } else {
      if (c.real_unit && c.k < r.bins && last_round && owner_segment) {
        const V v = res.value;
        dev::store_result(r, local, static_cast<std::int64_t>(c.unit) * r.bins + c.k, &v);
      }
    }
    sycl::group_barrier(nd.get_group());
  }
}

} // namespace sg::detail

#endif
