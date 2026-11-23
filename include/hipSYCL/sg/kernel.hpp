// SPDX-License-Identifier: BSD-2-Clause
// The kernel builder: parallel_for, map_groups, evaluate.
#ifndef HIPSYCL_SG_KERNEL_HPP
#define HIPSYCL_SG_KERNEL_HPP

#include "hipSYCL/sg/body.hpp"
#include "hipSYCL/sg/detail/closure.hpp"
#include "hipSYCL/sg/detail/dev/iterators.hpp"
#include "hipSYCL/sg/detail/probe.hpp"
#include "hipSYCL/sg/detail/registry.hpp"
#include "hipSYCL/sg/detail/tree_store.hpp"
#include "hipSYCL/sg/group_expr.hpp"
#include "hipSYCL/sg/kernel_kinds.hpp"
#include "hipSYCL/sg/mapping_forms.hpp"
#include "hipSYCL/sg/roles.hpp"

#include <sycl/sycl.hpp>

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace sg {

class kernel_handle;

template <class T, int N> class definition {
public:
  int index() const noexcept { return index_; }

private:
  struct not_trivially_copyable {
    not_trivially_copyable() {}
    not_trivially_copyable(const not_trivially_copyable &) {}
  };
  explicit definition(int i) : index_(i) {}
  int index_;
  not_trivially_copyable marker_;
  friend class kernel_handle;
};

template <class T, int N> class evaluation {
public:
  int definition_index() const noexcept { return index_; }

private:
  explicit evaluation(int i) : index_(i) {}
  int index_;
  friend class kernel_handle;
};

template <class Kind, class T, int M, class Positions> class mapped {
public:
  using value_type = T;
  static constexpr int rank = M;
  using kind = Kind;
  using positions_type = Positions;
  mapped(int source, Positions p) : source_(source), positions_(p) {}
  int source() const noexcept { return source_; }
  const Positions &positions() const noexcept { return positions_; }

private:
  int source_;
  Positions positions_;
};

namespace detail {

template <class X> struct is_mapped : std::false_type {};
template <class K, class T, int M, class P> struct is_mapped<mapped<K, T, M, P>> : std::true_type {};
template <class X> inline constexpr bool is_mapped_v = is_mapped<std::decay_t<X>>::value;

template <class M>
using iter_t = group_iterator<typename M::value_type, M::rank, typename M::kind>;

template <class Kind> constexpr link_kind link_kind_of() {
  if constexpr (std::is_same_v<Kind, mapping::item>)
    return link_kind::item;
  else if constexpr (std::is_same_v<Kind, mapping::local>)
    return link_kind::local;
  else
    return link_kind::global;
}

template <int N> sycl::range<3> pad3(const sycl::range<N> &r) {
  sycl::range<3> o(1, 1, 1);
  for (int d = 0; d < N; ++d)
    o[d] = r[d];
  return o;
}

template <int N, class M> link make_link(const M &m, std::vector<closure_entry> &closures) {
  static_assert(std::is_trivially_copyable_v<typename M::positions_type>,
                "sg::map_groups: the positions function must be trivially copyable");
  link l;
  l.kind = link_kind_of<typename M::kind>();
  l.source = m.source();
  const probed_form pf = probe_positions<N>(m.positions());
  l.form = pf.kind;
  l.count = pf.count;
  l.positions_symbol =
      positions_entry_ref<typename M::positions_type, N, M::rank>().symbol;
  closures.push_back(capture_closure(m.positions()));
  return l;
}

} // namespace detail

class kernel_handle {
public:
  explicit kernel_handle(detail::kernel_handle_impl &impl) : impl_(&impl) {}

  // item body
  template <int N, class... A>
  auto parallel_for(const sycl::range<N> &range, const sycl::range<N> &group_shape, A &&...a) {
    static_assert(sizeof...(A) >= 1, "sg::parallel_for: a body is required");
    auto t = std::forward_as_tuple(std::forward<A>(a)...);
    return item_impl<N>(range, group_shape, t, std::make_index_sequence<sizeof...(A) - 1>{});
  }

  // group body
  template <int N, class Map0, class... A,
            std::enable_if_t<detail::is_mapped_v<Map0>, int> = 0>
  auto parallel_for(const sycl::range<N> &group_range, Map0 &&m0, A &&...a) {
    static_assert(sizeof...(A) >= 1, "sg::parallel_for: a body is required");
    auto t = std::forward_as_tuple(std::forward<Map0>(m0), std::forward<A>(a)...);
    return group_impl<N>(group_range, t, std::make_index_sequence<sizeof...(A)>{});
  }

  template <class Kind, class T, int M, class Positions>
  mapped<Kind, T, M, Positions> map_groups(const definition<T, M> &source, Positions positions) {
    static_assert(std::is_trivially_copyable_v<Positions>,
                  "sg::map_groups: the positions function must be trivially copyable");
    return mapped<Kind, T, M, Positions>(source.index(), positions);
  }

  template <class T, int N> evaluation<T, N> evaluate(const definition<T, N> &d) {
    return evaluation<T, N>(d.index());
  }

private:
  template <int N, class Tuple, std::size_t... I>
  auto item_impl(const sycl::range<N> &range, const sycl::range<N> &gs, Tuple &t,
                 std::index_sequence<I...>) {
    constexpr std::size_t B = sizeof...(I);
    using Body = std::decay_t<std::tuple_element_t<B, Tuple>>;
    static_assert(std::is_invocable_v<const Body &, item<N>,
                                      detail::iter_t<std::decay_t<std::tuple_element_t<I, Tuple>>>...>,
                  "sg::parallel_for: the body's arguments must match the mappings");
    using R = std::invoke_result_t<const Body &, item<N>,
                                   detail::iter_t<std::decay_t<std::tuple_element_t<I, Tuple>>>...>;
    static_assert(std::is_trivially_copyable_v<R>, "sg: element types must be trivially copyable");
    const Body &body = std::get<B>(t);
    for (int d = 0; d < N; ++d)
      if (gs[d] == 0 || range[d] % gs[d] != 0)
        throw error{errc::shape_mismatch,
                    "parallel_for: the range " + std::to_string(range[d]) +
                        " is not a multiple of the group shape " + std::to_string(gs[d]) +
                        " in dimension " + std::to_string(d)};
    detail::def_node n;
    n.kind = detail::body_kind::item_body;
    n.body_symbol =
        detail::item_body_entry_ref<Body, N,
                                    detail::iter_t<std::decay_t<std::tuple_element_t<I, Tuple>>>...>()
            .symbol;
    n.store_symbol = detail::store_entry_ref<R>().symbol;
    n.type = detail::type_id<R>();
    n.elem_bytes = sizeof(R);
    n.rank = N;
    n.range = detail::pad3<N>(range);
    n.group_shape = detail::pad3<N>(gs);
    std::vector<detail::closure_entry> closures;
    closures.push_back(detail::capture_closure(body));
    (n.links.push_back(detail::make_link<N>(std::get<I>(t), closures)), ...);
    return definition<R, N>(impl_->add_definition(std::move(n), std::move(closures)));
  }

  template <int N, class Tuple, std::size_t... I>
  auto group_impl(const sycl::range<N> &grid, Tuple &t, std::index_sequence<I...>) {
    // I enumerates the maps; the body is the last element of the tuple
    constexpr std::size_t B = sizeof...(I);
    using Body = std::decay_t<std::tuple_element_t<B, Tuple>>;
    static_assert(
        (!std::is_same_v<typename std::decay_t<std::tuple_element_t<I, Tuple>>::kind, mapping::item> &&
         ...),
        "sg::parallel_for: a group body cannot take an item mapping");
    using E = std::invoke_result_t<const Body &, group<N>,
                                   detail::iter_t<std::decay_t<std::tuple_element_t<I, Tuple>>>...>;
    static_assert(detail::is_expr_v<E>, "sg: a group body returns a group expression");
    static_assert(E::rank <= N, "sg: a group body returns an expression of rank <= the group range's");
    static_assert(std::is_trivially_copyable_v<E> &&
                  std::is_trivially_copyable_v<typename E::value_type>);
    using V = typename E::value_type;
    const Body &body = std::get<B>(t);

    detail::def_node n;
    n.kind = detail::body_kind::group_body;
    std::vector<detail::closure_entry> closures;
    closures.push_back(detail::capture_closure(body));
    (n.links.push_back(detail::make_link<N>(std::get<I>(t), closures)), ...);

    // probe the body once: output shape and scratch need
    std::array<detail::dev::probe_iter_info, B> infos{};
    for_each_info<B>(infos, n, t, std::index_sequence<I...>{});
    detail::probe_scope ps;
    const E e = body(group<N>(sycl::id<N>{}, grid),
                     detail::iter_t<std::decay_t<std::tuple_element_t<I, Tuple>>>(
                         detail::dev::iterator_state{nullptr, 0, static_cast<std::uint32_t>(I), 0,
                                                     0, &infos[I]})...);
    const detail::dims3 od = e.dims();
    n.body_symbol =
        detail::group_body_entry_ref<Body, N,
                                     detail::iter_t<std::decay_t<std::tuple_element_t<I, Tuple>>>...>()
            .symbol;
    n.store_symbol = detail::store_entry_ref<V>().symbol;
    n.type = detail::type_id<V>();
    n.elem_bytes = sizeof(V);
    n.rank = N;
    n.range = detail::pad3<N>(grid);
    n.group_shape = sycl::range<3>(od[0], od[1], od[2]);
    n.scratch_bytes = e.scratch_bytes();
    return definition<V, N>(impl_->add_definition(std::move(n), std::move(closures)));
  }

  template <std::size_t B, class Infos, class Node, class Tuple, std::size_t... I>
  void for_each_info(Infos &infos, const Node &n, Tuple &t, std::index_sequence<I...>) {
    ((infos[I] = make_info(impl_->defs().nodes[std::get<I>(t).source()].group_shape,
                           n.links[I].count)),
     ...);
  }
  static detail::dev::probe_iter_info make_info(const sycl::range<3> &shape, std::size_t count) {
    detail::dev::probe_iter_info i{};
    for (int d = 0; d < 3; ++d)
      i.shape[d] = shape[d];
    i.count = count;
    return i;
  }

  detail::kernel_handle_impl *impl_;
};

} // namespace sg

#endif
