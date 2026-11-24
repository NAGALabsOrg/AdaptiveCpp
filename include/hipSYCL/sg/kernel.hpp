// SPDX-License-Identifier: BSD-2-Clause
// The kernel builder. It runs at submit and records a kernel graph, not code:
// roles on handles, views, work sets, maps over them, marks and sinks. A unit
// body is host code; it runs once per unit at preparation and records what the
// unit picked.
#ifndef HIPSYCL_SG_KERNEL_HPP
#define HIPSYCL_SG_KERNEL_HPP

#include "hipSYCL/sg/body.hpp"
#include "hipSYCL/sg/detail/backing_plan.hpp"
#include "hipSYCL/sg/detail/item_map.hpp"
#include "hipSYCL/sg/detail/registry.hpp"
#include "hipSYCL/sg/detail/repeat.hpp"
#include "hipSYCL/sg/detail/scope.hpp"
#include "hipSYCL/sg/handle.hpp"
#include "hipSYCL/sg/launch_value.hpp"
#include "hipSYCL/sg/roles.hpp"
#include "hipSYCL/sg/view.hpp"
#include "hipSYCL/sg/work_set.hpp"

#include <sycl/sycl.hpp>

#include <cstring>
#include <functional>
#include <source_location>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace sg {

namespace detail {

template <run_kind K, class Fn, class... A> struct unit_result {
  static constexpr run_kind kind = K;
  using fn_type = Fn;
  unit_result(std::vector<input_sel> s, Fn f) : sels(std::move(s)), fn(std::move(f)) {}
  std::vector<input_sel> sels;
  Fn fn;
};

template <class R> struct unit_result_traits;
template <run_kind K, class Fn, class... A> struct unit_result_traits<unit_result<K, Fn, A...>> {
  static constexpr run_kind kind = K;
  using fn_type = Fn;
  using args = std::tuple<A...>;
  using value_result = std::invoke_result_t<const Fn &, run_item, A...>;
};

template <class T> struct is_run_sel : std::false_type {};
template <class T> struct is_run_sel<run_sel<T>> : std::true_type {};

inline const char *run_name(run_kind k) {
  return k == run_kind::plain ? "unit.run" : k == run_kind::distributed ? "unit.run_distributed" : "unit.run_replicated";
}

} // namespace detail

// A unit of a work tile: its tile and its position in the tile.
class unit_base {
public:
  explicit unit_base(detail::unit_ctx &c) : c_(&c) {}
  idx tile() const { return c_->tile; }
  idx id() const { return c_->id; }

protected:
  template <detail::run_kind K, class... Args> auto make(Args &&...args) const {
    auto t = std::forward_as_tuple(std::forward<Args>(args)...);
    return make_impl<K>(t, std::make_index_sequence<sizeof...(Args) - 1>{});
  }

private:
  template <detail::run_kind K, class Tuple, std::size_t... I> auto make_impl(Tuple &t, std::index_sequence<I...>) const {
    static_assert(sizeof...(I) >= 1, "a run needs at least one input and a function");
    static_assert((detail::is_run_sel<std::decay_t<std::tuple_element_t<I, Tuple>>>::value && ...),
                  "the inputs of a run are 1-D runs: pass sg::flatten(tile)");
    constexpr std::size_t F = sizeof...(I);
    using Fn = std::decay_t<std::tuple_element_t<F, Tuple>>;
    static_assert(std::is_trivially_copyable_v<Fn>, "sg: a device function and its captures must be trivially copyable");
    std::vector<detail::input_sel> sels;
    (sels.push_back(std::get<I>(t).sel), ...);
    check_lengths<K>(sels);
    return detail::unit_result<K, Fn, typename std::decay_t<std::tuple_element_t<I, Tuple>>::value_type...>(
        std::move(sels), std::get<F>(t));
  }

  template <detail::run_kind K> void check_lengths(const std::vector<detail::input_sel> &sels) const {
    const std::int64_t len = sels.front().len;
    for (const detail::input_sel &s : sels)
      if (s.len != len)
        throw error{errc::shape_mismatch, std::string(detail::run_name(K)) +
                                              ": the inputs of a run have different lengths: " + std::to_string(len) +
                                              " and " + std::to_string(s.len)};
    if (len > c_->run_len)
      throw build_error(14, detail::run_name(K), detail::item_site(),
                        "an input is longer than the run of the work set",
                        {{"input_length", len}, {"run_length", c_->run_len}});
    if constexpr (K == detail::run_kind::replicated) {
      if (c_->env && c_->env->distributed_segments)
        for (const detail::input_sel &s : sels) {
          const std::int64_t segs = c_->env->distributed_segments(s.view);
          if (segs > 0)
            throw build_error(6, "unit.run_replicated", detail::item_site(),
                              "an input is a distributed result of the unit's own work set",
                              {{"segments", segs}});
        }
    }
  }

  detail::unit_ctx *c_;
};

class unit : public unit_base {
public:
  using unit_base::unit_base;
  template <class... Args> auto run(Args &&...args) const {
    return make<detail::run_kind::plain>(std::forward<Args>(args)...);
  }
};

class segmented_unit : public unit_base {
public:
  using unit_base::unit_base;
  template <class... Args> auto run_distributed(Args &&...args) const {
    return make<detail::run_kind::distributed>(std::forward<Args>(args)...);
  }
  template <class... Args> auto run_replicated(Args &&...args) const {
    return make<detail::run_kind::replicated>(std::forward<Args>(args)...);
  }
};

class kernel_handle {
public:
  explicit kernel_handle(detail::node_record &n) : n_(&n), g_(&n.graph) {}

  // ---- roles ----

  template <class T, int N> elements<T> source(const mem_handle<T> &h, const shape<N> &s) {
    detail::view_value_check<T>{};
    const int slot = detail::register_slot<T>(h, s.extents(), detail::role_mode::read, false, {},
                                              detail::init_kind::none, {});
    detail::view_rec v;
    v.kind = detail::vk::source;
    v.elem_bytes = sizeof(T);
    v.slot = slot;
    v.ext = s.extents();
    return elements<T>(g_, g_->add_view(std::move(v)));
  }

  template <class T> void evaluate(const mem_handle<T> &h, const elements<T> &e) {
    add_sink<T>(h, e, detail::role_mode::produce, {}, detail::init_kind::none, {}, nullptr);
  }
  template <class T, class Op> void evaluate(const detail::inplace_spec<T, Op> &s, const elements<T> &e) {
    std::vector<std::byte> bits;
    if (s.init == detail::init_kind::value)
      bits = detail::bytes_of(s.init_value);
    const std::string sink = detail::sink_task_ref<T, Op>().symbol;
    add_sink<T>(s.handle, e, detail::role_mode::inplace, s.op, s.init, std::move(bits), &sink);
  }

  // the number in a handle, read by the host just before the node launches
  template <class T> sg::launch_value<T> launch_value(const mem_handle<T> &h) {
    static_assert(std::is_arithmetic_v<T>, "kh.launch_value: the handle holds an arithmetic value");
    const int slot = detail::register_slot<T>(h, bdims{1}, detail::role_mode::read, true, {},
                                              detail::init_kind::none, {});
    detail::launch_slot_rec l;
    l.handle = static_cast<std::size_t>(slot);
    l.type = detail::type_id<T>();
    l.bytes = sizeof(T);
    l.is_float = std::is_floating_point_v<T>;
    g_->launches.push_back(l);
    return sg::launch_value<T>(static_cast<std::size_t>(slot), static_cast<std::int64_t>(g_->launches.size()) - 1);
  }

  // ---- views ----

  template <class T, class Oob>
  tiles<T> tile(const elements<T> &x, const dims &shape, const Oob &oob,
                std::source_location loc = std::source_location::current()) {
    const detail::view_rec &p = g_->views[x.id()];
    if (p.kind == detail::vk::array)
      throw detail::build_error_11(p.name.empty() ? "array #" + std::to_string(x.id()) : "array '" + p.name + "'");
    if (p.ext.rank != shape.rank)
      throw error{errc::shape_mismatch, std::string("kh.tile at ") + loc.file_name() + ":" +
                                            std::to_string(loc.line()) + ": a tile of rank " +
                                            std::to_string(shape.rank) + " over elements of rank " +
                                            std::to_string(p.ext.rank)};
    detail::view_rec v;
    v.kind = detail::vk::tile;
    v.parent = x.id();
    v.elem_bytes = sizeof(T);
    v.shape = shape;
    v.grid_rank = shape.rank;
    set_oob<T>(v, oob);
    return tiles<T>(g_, g_->add_view(std::move(v)));
  }
  template <class T> tiles<T> tile(const elements<T> &x, const dims &shape) { return tile(x, shape, fill<T>{T{}}); }

  template <class T> tiles<T> retile(const tiles<T> &t, const dims &shape) {
    detail::view_rec v;
    v.kind = detail::vk::retile;
    v.parent = t.id();
    v.elem_bytes = sizeof(T);
    v.shape = shape;
    v.grid_rank = 1;
    return tiles<T>(g_, g_->add_view(std::move(v)));
  }

  template <class T> elements<T> join(const tiles<T> &t, std::source_location loc = std::source_location::current()) {
    const detail::view_rec &p = g_->views[t.id()];
    if (p.grid_rank != p.shape.rank)
      throw build_error(12, "kh.join", source_loc::from(loc),
                        "tiles are stitched over a grid of their own rank",
                        {{"grid_rank", p.grid_rank}, {"tile_rank", p.shape.rank}});
    detail::view_rec v;
    v.kind = detail::vk::join;
    v.parent = t.id();
    v.elem_bytes = sizeof(T);
    v.ext.rank = p.shape.rank;
    return elements<T>(g_, g_->add_view(std::move(v)));
  }

  template <class T> tiles<T> extend_tiles(const tiles<T> &t, const dims &shape, repeat_t) {
    const detail::view_rec &p = g_->views[t.id()];
    if (shape.rank < p.shape.rank)
      throw error{errc::shape_mismatch, "kh.extend_tiles: the new tile shape has fewer dimensions than the tile"};
    detail::view_rec v;
    v.kind = detail::vk::extend;
    v.parent = t.id();
    v.elem_bytes = sizeof(T);
    v.shape = shape;
    v.grid_rank = p.grid_rank;
    return tiles<T>(g_, g_->add_view(std::move(v)));
  }

  // index tiles from a host function of (tile, slot)
  template <class F> tiles<std::int64_t> define_tiles(const bound_expr &n, const dims &shape, F fn) {
    detail::view_rec v;
    v.kind = detail::vk::table;
    v.elem_bytes = sizeof(std::int64_t);
    v.shape = shape;
    v.grid_rank = 1;
    v.define = true;
    v.define_n = n;
    v.define_fn = [fn](std::int64_t t, std::int64_t s) { return static_cast<std::int64_t>(fn(t, s)); };
    v.table_rank = 1;
    return tiles<std::int64_t>(g_, g_->add_view(std::move(v)));
  }

  // index tiles: each root plus each offset
  tiles<std::int64_t> stencil(std::initializer_list<std::int64_t> offsets, const range &roots) {
    std::vector<std::array<std::int64_t, 3>> o;
    for (std::int64_t x : offsets)
      o.push_back({x, 0, 0});
    return add_stencil(std::move(o), roots);
  }
  tiles<std::int64_t> stencil(std::initializer_list<std::initializer_list<std::int64_t>> offsets, const range &roots) {
    std::vector<std::array<std::int64_t, 3>> o;
    for (const auto &l : offsets) {
      std::array<std::int64_t, 3> a{0, 0, 0};
      int d = 0;
      for (std::int64_t x : l)
        if (d < 3)
          a[d++] = x;
      o.push_back(a);
    }
    return add_stencil(std::move(o), roots);
  }

  // each index replaced by its element; out of range goes to the policy
  template <class I, class Oob, class T>
  tiles<T> index_map(const tiles<I> &map, const Oob &oob, const elements<T> &elems) {
    static_assert(std::is_integral_v<I>, "kh.index_map: the map holds integer indices");
    const detail::view_rec &m = g_->views[map.id()];
    const detail::view_rec &src = g_->views[elems.id()];
    if (src.kind == detail::vk::array)
      throw detail::build_error_11(src.name.empty() ? "array #" + std::to_string(elems.id()) : "array '" + src.name + "'");
    detail::view_rec v;
    v.kind = detail::vk::index_map;
    v.parent = elems.id();
    v.parent2 = map.id();
    v.elem_bytes = sizeof(T);
    v.shape = m.shape;
    v.grid_rank = m.grid_rank;
    set_oob<T>(v, oob);
    return tiles<T>(g_, g_->add_view(std::move(v)));
  }

  // ---- work sets ----

  work_set work_tiles(const bdims &grid, const dims &tile, detail::sint run,
                      std::source_location loc = std::source_location::current()) {
    return work_set(g_, detail::add_tile_work(*g_, grid, tile, run.v, false, 1, 1, source_loc::from(loc)));
  }
  segmented_work_set work_tiles(const bdims &grid, const dims &tile, const segment_items &run,
                                std::source_location loc = std::source_location::current()) {
    return segmented_work_set(g_, detail::add_tile_work(*g_, grid, tile, run.segments * run.items, true,
                                                         run.segments, run.items, source_loc::from(loc)));
  }

  work_set slice_work(const work_set &w, const bdims &offset, const bdims &extent, const bdims &stride,
                      const dims &tile, detail::sint run,
                      std::source_location loc = std::source_location::current()) {
    return work_set(g_, detail::add_slice_work(*g_, w.id(), offset, extent, stride, tile, run.v,
                                               source_loc::from(loc)));
  }

  item_set work_items(const bdims &range, std::source_location loc = std::source_location::current()) {
    return item_set(g_, detail::add_item_work(*g_, range, source_loc::from(loc)));
  }

  // ---- maps ----

  template <class Body> auto map_work(const work_set &w, Body &&body) {
    return map_work_impl<unit>(w, std::forward<Body>(body));
  }
  template <class Body> auto map_work(const segmented_work_set &w, Body &&body) {
    return map_work_impl<segmented_unit>(w, std::forward<Body>(body));
  }

  template <class Body>
  auto map_items(const item_set &items, Body &&body, std::source_location loc = std::source_location::current()) {
    return detail::map_items_impl(*g_, items, std::forward<Body>(body), source_loc::from(loc));
  }

  // ---- marks ----

  template <class Mode = void, class T>
  void mark_local(const result_tiles<T> &r, std::source_location loc = std::source_location::current()) {
    static_assert(std::is_void_v<Mode> || std::is_same_v<Mode, sg::segmented>,
                  "kh.mark_local: the only mode is sg::segmented");
    detail::mark_result(*g_, r.id(), std::is_same_v<Mode, sg::segmented>, source_loc::from(loc));
  }
  template <class T>
  void mark_local(const tiles<T> &array_tiles, const work_set &w,
                  std::source_location loc = std::source_location::current()) {
    detail::mark_array_tiles(*g_, array_tiles.id(), w.id(), source_loc::from(loc));
  }

  // ---- repetition ----

  template <class T, class Fn> result_tiles<T> repeat(const bound_expr &n, const result_tiles<T> &start, Fn &&fn) {
    return detail::record_repeat(*this, *g_, n, start, std::forward<Fn>(fn));
  }

  detail::kgraph &graph() noexcept { return *g_; }

private:
  template <class T, class Oob> static void set_oob(detail::view_rec &v, const Oob &oob) {
    if constexpr (std::is_same_v<Oob, wrap>) {
      v.oob = detail::oob_kind::wrap;
    } else {
      v.oob = detail::oob_kind::fill;
      const T t = static_cast<T>(oob.value);
      std::memcpy(v.fill, &t, sizeof(T));
    }
  }

  tiles<std::int64_t> add_stencil(std::vector<std::array<std::int64_t, 3>> offsets, const range &roots) {
    detail::view_rec v;
    v.kind = detail::vk::table;
    v.elem_bytes = sizeof(std::int64_t);
    v.shape = dims{static_cast<std::int64_t>(offsets.size())};
    v.grid_rank = roots.begin.rank;
    v.stencil = true;
    v.offsets = std::move(offsets);
    v.roots_begin = roots.begin;
    v.roots_end = roots.end;
    v.table_rank = static_cast<std::uint32_t>(roots.begin.rank);
    return tiles<std::int64_t>(g_, g_->add_view(std::move(v)));
  }

  template <class T>
  void add_sink(const mem_handle<T> &h, const elements<T> &e, detail::role_mode mode, std::string op,
                detail::init_kind init, std::vector<std::byte> bits, const std::string *sink_symbol) {
    const detail::view_rec &v = g_->views[e.id()];
    bdims ext;
    ext.rank = v.ext.rank;
    for (int d = 0; d < ext.rank; ++d)
      ext.v[d] = bound_expr::count_of(e.id(), d);
    const int slot = detail::register_slot<T>(h, ext, mode, false, std::move(op), init, std::move(bits));
    detail::sink_rec s;
    s.slot = slot;
    s.view = e.id();
    s.items = v.kind == detail::vk::array;
    if (sink_symbol) {
      s.sink_task = g_->task_index(*sink_symbol);
      if (!s.items)
        throw error{errc::builder_misuse, "kh.evaluate: in-place results take work-item arrays"};
    }
    g_->sinks.push_back(s);
  }

  template <class UnitT, class Body> auto map_work_impl(const work_set &w, Body &&body) {
    using B = std::decay_t<Body>;
    using R = std::invoke_result_t<const B &, UnitT>;
    using traits = detail::unit_result_traits<R>;
    using Fn = typename traits::fn_type;
    using V = typename detail::result_traits<typename traits::value_result>::value;
    constexpr detail::result_kind rk = detail::result_traits<typename traits::value_result>::kind;
    static_assert(std::is_trivially_copyable_v<V> && sizeof(V) <= 16, "sg: results are at most 16 bytes");
    static_assert(std::is_same_v<UnitT, segmented_unit> == (traits::kind != detail::run_kind::plain),
                  "a segmented work set runs run_distributed or run_replicated; a plain one runs run");
    if (w.id() < 0 || g_->works[w.id()].items)
      throw error{errc::builder_misuse, "kh.map_work: the first argument is not a set of work tiles"};

    detail::map_rec m;
    m.work = w.id();
    m.run = traits::kind;
    m.rk = rk;
    m.result_bytes = sizeof(V);
    m.eval = [body = B(body)](detail::unit_ctx &u) {
      UnitT ut(u);
      R r = body(ut);
      detail::unit_eval_out o;
      o.kind = traits::kind;
      o.sels = r.sels;
      o.closure = detail::capture_closure(r.fn);
      return o;
    };
    set_task<Fn>(m, typename traits::args{});
    const int map_id = g_->add_map(std::move(m));

    const detail::work_rec &wr = g_->works[w.id()];
    detail::view_rec v;
    v.kind = detail::vk::result;
    v.map = map_id;
    v.elem_bytes = sizeof(V);
    v.grid_rank = wr.grid.rank;
    const int tile_rank = wr.tile.rank;
    v.shape.rank = tile_rank < 3 && rk != detail::result_kind::reduced ? tile_rank + 1 : tile_rank;
    const int view_id = g_->add_view(std::move(v));
    g_->maps[map_id].result_view = view_id;
    return result_tiles<V>(g_, view_id);
  }

  template <class Fn, class... A> void set_task(detail::map_rec &m, std::tuple<A...>) {
    m.task_symbol = detail::tile_task_ref<Fn, A...>().symbol;
    m.task = g_->task_index(m.task_symbol);
    m.n_inputs = sizeof...(A);
    m.sizing = [](const detail::closure_entry &ce) {
      static_assert((std::is_default_constructible_v<A> && ...), "sg: run inputs are default constructible");
      alignas(Fn) std::byte buf[sizeof(Fn)];
      std::memcpy(buf, ce.bytes.data(), sizeof(Fn));
      const Fn &fn = *reinterpret_cast<const Fn *>(buf);
      detail::dev::sizing_rec rec;
      detail::dev::run_ctx c;
      c.sizing = &rec;
      run_item item(&c);
      (void)fn(item, A{}...);
      detail::sizing_info p;
      p.bins = rec.bins;
      p.workspace_bytes = rec.elem_bytes;
      p.partial_bytes = rec.partial_bytes;
      return p;
    };
  }

  detail::node_record *n_;
  detail::kgraph *g_;
};

} // namespace sg

#endif
