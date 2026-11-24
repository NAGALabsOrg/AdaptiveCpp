// SPDX-License-Identifier: BSD-2-Clause
// Work-item maps: the body runs once, symbolically, and records what each item
// reads and computes. An item reads elements of sources, and a result array of
// another work-item set only at its own index.
#ifndef HIPSYCL_SG_DETAIL_ITEM_MAP_HPP
#define HIPSYCL_SG_DETAIL_ITEM_MAP_HPP

#include "hipSYCL/sg/detail/registry.hpp"
#include "hipSYCL/sg/view.hpp"
#include "hipSYCL/sg/work_set.hpp"

#include <source_location>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <algorithm>

namespace sg {

namespace detail {

inline source_loc item_site() { return {"<kernel builder>", 0}; }

inline build_error build_error_11(const std::string &who) {
  return build_error(11, "kh.map_items", item_site(),
                     "kh.map_work cannot take " + who +
                         ": result arrays of kh.map_items never feed work tiles, and result tiles of "
                         "kh.map_work never feed work items");
}

} // namespace detail

// The item a body is run for, once, symbolically.
class item_ctx {
public:
  item_ctx(detail::kgraph *g, detail::item_map_rec *rec, int rank, int first_reg)
      : g_(g), rec_(rec), rank_(rank), next_reg_(first_reg) {}

  item_index id() const {
    const std::int64_t zero[3] = {0, 0, 0};
    return item_index(rank_, zero);
  }

  // fn on the values of this item; fn is a device function and runs no collectives
  template <class... Args> auto compute(Args &&...args) {
    constexpr std::size_t N = sizeof...(Args);
    static_assert(N >= 2, "item.compute takes at least one input and a function");
    auto t = std::forward_as_tuple(std::forward<Args>(args)...);
    return compute_impl(t, std::make_index_sequence<N - 1>{});
  }

  int registers_used() const noexcept { return next_reg_; }

private:
  template <class Tuple, std::size_t... I> auto compute_impl(Tuple &t, std::index_sequence<I...>) {
    constexpr std::size_t F = sizeof...(I);
    using Fn = std::decay_t<std::tuple_element_t<F, Tuple>>;
    using R = std::invoke_result_t<const Fn &, typename std::decay_t<std::tuple_element_t<I, Tuple>>::value_type...>;
    static_assert(sizeof...(I) <= detail::dev::max_item_inputs, "item.compute takes at most four inputs");
    static_assert(std::is_trivially_copyable_v<R> && sizeof(R) <= 16, "item.compute: results are at most 16 bytes");
    const Fn &fn = std::get<F>(t);
    detail::item_task_rec r;
    r.task_symbol = detail::item_task_ref<Fn, typename std::decay_t<std::tuple_element_t<I, Tuple>>::value_type...>().symbol;
    r.task = g_->task_index(r.task_symbol);
    r.closure = detail::capture_closure(fn);
    (r.inputs.push_back(std::get<I>(t).in), ...);
    r.out_reg = next_reg_++;
    r.out_bytes = sizeof(R);
    rec_->tasks.push_back(std::move(r));
    item_val<R> out;
    out.in.kind = detail::item_in_kind::reg;
    out.in.reg = rec_->tasks.back().out_reg;
    out.in.bytes = sizeof(R);
    return out;
  }

  detail::kgraph *g_;
  detail::item_map_rec *rec_;
  int rank_;
  int next_reg_;
};

template <class T> item_val<T> elements<T>::operator[](const item_index &i) const {
  detail::view_rec &v = g_->views[id_];
  const std::string who = v.name.empty() ? "array #" + std::to_string(id_) : "array '" + v.name + "'";
  if (v.kind != detail::vk::source && v.kind != detail::vk::array)
    throw detail::build_error_11(who);
  if (i.rank() != v.ext.rank)
    throw error{errc::shape_mismatch, "kh.map_items: " + who + " has rank " + std::to_string(v.ext.rank) +
                                          " and was read at an index of rank " + std::to_string(i.rank())};
  item_val<T> out;
  out.in.kind = v.kind == detail::vk::array ? detail::item_in_kind::array : detail::item_in_kind::source;
  out.in.view = id_;
  out.in.bytes = sizeof(T);
  for (int d = 0; d < 3; ++d)
    out.in.off[d] = i.offset(d);
  if (v.kind == detail::vk::array)
    for (int d = 0; d < 3; ++d)
      if (i.offset(d) != 0)
        throw build_error(10, "kh.map_items", detail::item_site(),
                          "a work item reads a result array only at its own index: " + who,
                          {{"offset", i.offset(d)}});
  return out;
}

namespace detail {

template <class Body> auto map_items_impl(kgraph &g, const item_set &items, Body &&body, source_loc where) {
  const work_rec &w = g.works[items.id()];
  item_map_rec rec;
  rec.work = items.id();
  rec.where = where;
  int first_reg = 0;
  for (const item_map_rec &m : g.imaps)
    for (const item_task_rec &t : m.tasks)
      first_reg = std::max(first_reg, static_cast<int>(t.out_reg) + 1);
  item_ctx ctx(&g, &rec, w.grid.rank, first_reg);
  auto out = body(ctx);
  using V = decltype(out);
  static_assert(std::is_same_v<V, item_val<typename V::value_type>>, "kh.map_items: the body returns the item's value");
  if (out.in.kind != item_in_kind::reg)
    throw error{errc::builder_misuse, "kh.map_items: the body must return the result of item.compute"};
  using T = typename V::value_type;
  rec.result_bytes = sizeof(T);
  const int map_id = static_cast<int>(g.imaps.size());
  view_rec v;
  v.kind = vk::array;
  v.elem_bytes = sizeof(T);
  v.map = map_id;
  v.ext = w.grid;
  v.slot = -1;
  const int view_id = g.add_view(std::move(v));
  rec.result_view = view_id;
  g.add_imap(std::move(rec));
  return elements<T>(&g, view_id);
}

} // namespace detail

} // namespace sg

#endif
