// SPDX-License-Identifier: BSD-2-Clause
// kh.repeat(n, start, fn): one structure applied n times. The function is
// recorded once. At preparation the recorded section is instanced n times, the
// first reading `start` and every later one the result of the one before; n
// changes the bound tables and never the structure.
#ifndef HIPSYCL_SG_DETAIL_REPEAT_HPP
#define HIPSYCL_SG_DETAIL_REPEAT_HPP

#include "hipSYCL/sg/detail/scope.hpp"
#include "hipSYCL/sg/view.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace sg::detail {

template <class Kh, class T, class Fn>
result_tiles<T> record_repeat(Kh &kh, kgraph &g, const bound_expr &n, const result_tiles<T> &start, Fn &&fn) {
  repeat_rec r;
  r.n = n;
  r.start_view = start.id();
  r.vb = static_cast<int>(g.views.size());
  r.wb = static_cast<int>(g.works.size());
  r.mb = static_cast<int>(g.maps.size());
  r.ib = static_cast<int>(g.imaps.size());
  r.eb = g.events.size();

  const view_rec &s = g.views[start.id()];
  view_rec alias;
  alias.kind = vk::retile;
  alias.parent = start.id();
  alias.alias_prev = true;
  alias.elem_bytes = s.elem_bytes;
  alias.grid_rank = s.grid_rank;
  alias.shape.rank = s.shape.rank;
  r.prev_view = g.add_view(std::move(alias));
  result_tiles<T> prev(&g, r.prev_view);

  result_tiles<T> out = fn(kh, prev);
  r.result_view = out.id();
  r.ve = static_cast<int>(g.views.size());
  r.we = static_cast<int>(g.works.size());
  r.me = static_cast<int>(g.maps.size());
  r.ie = static_cast<int>(g.imaps.size());
  r.ee = g.events.size();
  g.repeats.push_back(std::move(r));
  return out;
}

// Id renaming for one instance of a repeated section. References made from
// outside the section to something inside it mean the last instance.
struct section_map {
  const repeat_rec &r;
  std::int64_t iteration;     // instance being built
  std::int64_t last;          // n - 1
  bool from_outside;          // the referring record is not part of the section
  std::int64_t previous_result_view; // result view of the previous instance
  std::int64_t lv() const { return r.ve - r.vb; }
  std::int64_t lw() const { return r.we - r.wb; }
  std::int64_t lm() const { return r.me - r.mb; }
  std::int64_t li() const { return r.ie - r.ib; }

  int view(int old, bool after_section_shift) const {
    if (old < 0)
      return old;
    if (old < r.vb)
      return old;
    if (old < r.ve) {
      const std::int64_t it = from_outside ? last : iteration;
      return static_cast<int>(old + it * lv());
    }
    (void)after_section_shift;
    return static_cast<int>(old + last * lv());
  }
  int work(int old) const {
    if (old < 0)
      return old;
    if (old < r.wb)
      return old;
    if (old < r.we)
      return static_cast<int>(old + (from_outside ? last : iteration) * lw());
    return static_cast<int>(old + last * lw());
  }
  int map(int old) const {
    if (old < 0)
      return old;
    if (old < r.mb)
      return old;
    if (old < r.me)
      return static_cast<int>(old + (from_outside ? last : iteration) * lm());
    return static_cast<int>(old + last * lm());
  }
  int imap(int old) const {
    if (old < 0)
      return old;
    if (old < r.ib)
      return old;
    if (old < r.ie)
      return static_cast<int>(old + (from_outside ? last : iteration) * li());
    return static_cast<int>(old + last * li());
  }
};

inline bound_expr remap_expr(const bound_expr &e, const section_map &m) {
  return e.remap([&](std::int64_t v) { return static_cast<std::int64_t>(m.view(static_cast<int>(v), false)); });
}
inline bdims remap_dims(const bdims &d, const section_map &m) {
  bdims o = d;
  for (int i = 0; i < d.rank; ++i)
    o.v[i] = remap_expr(d.v[i], m);
  return o;
}

// Expands the repeated sections of a recorded graph for the given launch values.
inline kgraph expand_repeats(const kgraph &in, const eval_env &env) {
  if (in.repeats.empty())
    return in;
  kgraph g = in;
  // sections are expanded from the last one to the first so earlier ids stay put
  for (std::size_t ri = g.repeats.size(); ri-- > 0;) {
    const repeat_rec r = g.repeats[ri];
    eval_env e = env;
    std::int64_t n = r.n.evaluate(e);
    if (n < 1)
      n = 1;
    const std::int64_t last = n - 1;
    kgraph out;
    out.group_size_cap = g.group_size_cap;
    out.tasks = g.tasks;
    out.launches = g.launches;
    out.has_tiles = g.has_tiles;
    out.has_items = g.has_items;

    auto rename_view = [&](view_rec v, const section_map &m) {
      v.parent = m.view(v.parent, false);
      v.parent2 = m.view(v.parent2, false);
      if (v.map >= 0 && (v.kind == vk::result || v.kind == vk::array))
        v.map = v.kind == vk::result ? m.map(v.map) : m.imap(v.map);
      v.ext = remap_dims(v.ext, m);
      v.roots_begin = remap_dims(v.roots_begin, m);
      v.roots_end = remap_dims(v.roots_end, m);
      v.define_n = remap_expr(v.define_n, m);
      return v;
    };
    auto rename_work = [&](work_rec w, const section_map &m) {
      w.root = m.work(w.root);
      w.grid = remap_dims(w.grid, m);
      w.offset = remap_dims(w.offset, m);
      w.extent = remap_dims(w.extent, m);
      w.stride = remap_dims(w.stride, m);
      return w;
    };
    auto rename_map = [&](map_rec mp, const section_map &m, bool in_section) {
      mp.work = m.work(mp.work);
      mp.result_view = m.view(mp.result_view, false);
      if (in_section) {
        const int vb = r.vb, ve = r.ve;
        const std::int64_t shift = m.iteration * m.lv();
        mp.remap = [vb, ve, shift](int v) { return v >= vb && v < ve ? static_cast<int>(v + shift) : v; };
      }
      return mp;
    };
    auto rename_imap = [&](item_map_rec mp, const section_map &m) {
      mp.work = m.work(mp.work);
      mp.result_view = m.view(mp.result_view, false);
      for (item_task_rec &t : mp.tasks)
        for (item_input &i : t.inputs)
          i.view = m.view(i.view, false);
      return mp;
    };

    auto emit = [&](const ev_rec &ev, const section_map &m) {
      switch (ev.kind) {
      case ev_kind::view: out.add_view(rename_view(g.views[ev.idx], m)); break;
      case ev_kind::work: out.add_work(rename_work(g.works[ev.idx], m)); break;
      case ev_kind::map:
        out.add_map(rename_map(g.maps[ev.idx], m, !m.from_outside && ev.idx >= r.mb && ev.idx < r.me));
        break;
      case ev_kind::imap: out.add_imap(rename_imap(g.imaps[ev.idx], m)); break;
      }
    };

    // events before the section
    for (std::size_t i = 0; i < r.eb; ++i)
      emit(g.events[i], section_map{r, 0, last, true, -1});
    // the instances; the alias of instance i reads the result of instance i - 1
    for (std::int64_t it = 0; it < n; ++it) {
      section_map m{r, it, last, false, -1};
      for (std::size_t i = r.eb; i < r.ee; ++i) {
        ev_rec ev = g.events[i];
        if (ev.kind == ev_kind::view && ev.idx == r.prev_view) {
          view_rec alias = rename_view(g.views[ev.idx], m);
          alias.parent = it == 0 ? r.start_view : static_cast<int>(r.result_view + (it - 1) * m.lv());
          out.add_view(std::move(alias));
        } else {
          emit(ev, m);
        }
      }
    }
    // events after the section
    for (std::size_t i = r.ee; i < g.events.size(); ++i)
      emit(g.events[i], section_map{r, 0, last, true, -1});

    out.sinks = g.sinks;
    for (sink_rec &s : out.sinks)
      s.view = section_map{r, 0, last, true, -1}.view(s.view, false);
    out.marks = g.marks;
    for (mark_rec &mk : out.marks) {
      section_map m{r, 0, last, true, -1};
      mk.view = m.view(mk.view, false);
      mk.work = m.work(mk.work);
    }
    // the remaining sections now sit later by (n - 1) sections' worth of ids
    out.repeats.clear();
    for (std::size_t k = 0; k < ri; ++k)
      out.repeats.push_back(g.repeats[k]);
    g = std::move(out);
  }
  return g;
}

} // namespace sg::detail

#endif
