// SPDX-License-Identifier: BSD-2-Clause
// Views: elements, tiles, selected tiles and runs. A view is a descriptor; the
// builder only records it, and the toolchain interprets it from the descriptor.
#ifndef HIPSYCL_SG_VIEW_HPP
#define HIPSYCL_SG_VIEW_HPP

#include "hipSYCL/sg/detail/scope.hpp"
#include "hipSYCL/sg/detail/view_desc.hpp"
#include "hipSYCL/sg/error.hpp"
#include "hipSYCL/sg/launch_value.hpp"

#include <array>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <utility>

namespace sg {

template <class V> struct fill {
  V value;
};
template <class V> fill(V) -> fill<V>;
struct wrap {};
struct repeat_t {};
inline constexpr repeat_t repeat{};

// sg::range(0, 1024) or sg::range({0, 0}, {64, 64})
struct range {
  bdims begin, end;
  range(const bound_expr &b, const bound_expr &e) : begin{b}, end{e} {}
  range(std::initializer_list<bound_expr> b, std::initializer_list<bound_expr> e) : begin(b), end(e) {}
};

namespace detail {

struct prep_access {
  const view_node *nodes = nullptr;
  std::size_t count = 0;
};
inline prep_access *&current_prep() noexcept {
  static thread_local prep_access *p = nullptr;
  return p;
}

// The view ids a unit body sees are those of the recorded structure; an
// instance of a repeated section reads the ids of its own copy.
inline const std::function<int(int)> *&current_view_remap() noexcept {
  static thread_local const std::function<int(int)> *f = nullptr;
  return f;
}
inline int remapped_view(int id) {
  const std::function<int(int)> *f = current_view_remap();
  return f && *f ? (*f)(id) : id;
}

template <class T> struct view_value_check {
  static_assert(std::is_trivially_copyable_v<T>, "sg: element types must be trivially copyable");
  static_assert(sizeof(T) <= 16, "sg: views hold elements of at most 16 bytes");
};

inline kgraph &cur_graph() {
  node_record *n = current_scope();
  if (!n)
    throw error{errc::not_in_submit_scope, "a view needs an active submit scope"};
  return n->graph;
}

} // namespace detail

template <class T> class tile_sel;
template <class T> class run_sel;
template <class T> class item_val;
class item_index;

class view_ref {
public:
  view_ref() = default;
  view_ref(detail::kgraph *g, int id) : g_(g), id_(id) {}
  int id() const noexcept { return id_; }
  detail::kgraph *graph() const noexcept { return g_; }
  bool valid() const noexcept { return g_ != nullptr; }

protected:
  detail::kgraph *g_ = nullptr;
  int id_ = detail::no_view;
};

// elements of an array
template <class T> class elements : public view_ref {
  detail::view_value_check<T> check_;

public:
  using value_type = T;
  elements() = default;
  elements(detail::kgraph *g, int id) : view_ref(g, id) {}
  elements named(const std::string &n) const {
    g_->views[id_].name = n;
    return *this;
  }

  // a work item reads its element; defined with the item maps
  item_val<T> operator[](const item_index &i) const;
};

// tiles of elements
template <class T> class tiles : public view_ref {
  detail::view_value_check<T> check_;

public:
  using value_type = T;
  tiles() = default;
  tiles(detail::kgraph *g, int id) : view_ref(g, id) {}
  tiles named(const std::string &n) const {
    g_->views[id_].name = n;
    return *this;
  }

  // the number of tiles in the grid, or the grid's extent in a dimension
  bound_expr count() const { return bound_expr::count_of(id_, -1); }
  bound_expr count(int dim) const { return bound_expr::count_of(id_, dim); }

  tile_sel<T> operator[](const idx &i) const;
};

template <class T> class result_tiles : public tiles<T> {
public:
  result_tiles() = default;
  result_tiles(detail::kgraph *g, int id) : tiles<T>(g, id) {}
  result_tiles named(const std::string &n) const {
    this->g_->views[this->id_].name = n;
    return *this;
  }
};

// One tile of a view, with the box of its elements that is taken.
template <class T> class tile_sel {
public:
  detail::input_sel sel;
  std::int64_t shape[3] = {1, 1, 1};
  int rank = 1;
};

template <class T> class run_sel {
public:
  using value_type = T;
  detail::input_sel sel;
};

template <class T> tile_sel<T> tiles<T>::operator[](const idx &i) const {
  detail::prep_access *p = detail::current_prep();
  if (!p)
    throw error{errc::builder_misuse, "tiles are selected in unit bodies, which run at preparation"};
  const int vid = detail::remapped_view(id_);
  const detail::view_node &n = p->nodes[vid];
  std::int64_t g[3] = {0, 0, 0};
  for (int d = 0; d < i.rank() && d < 3; ++d)
    g[d] = i[d];
  bool inside = true;
  for (int d = 0; d < 3; ++d)
    if (g[d] < 0 || g[d] >= n.grid[d])
      inside = false;
  if (!inside)
    throw error{errc::shape_mismatch, "a tile was selected outside its grid: tile " + std::to_string(g[0]) + "," +
                                          std::to_string(g[1]) + "," + std::to_string(g[2]) +
                                          " of a grid " + std::to_string(n.grid[0]) + "x" +
                                          std::to_string(n.grid[1]) + "x" + std::to_string(n.grid[2])};
  tile_sel<T> t;
  t.sel.view = vid;
  t.sel.tile = detail::flatten3(g, n.grid);
  for (int d = 0; d < 3; ++d) {
    t.sel.off[d] = 0;
    t.sel.ext[d] = n.shape[d];
    t.shape[d] = n.shape[d];
  }
  t.rank = static_cast<int>(n.shape_rank);
  t.sel.len = detail::prod3(n.shape);
  return t;
}

// a box of a tile
template <class T> tile_sel<T> slice(tile_sel<T> t, const idx &offset, const idx &extent) {
  for (int d = 0; d < 3; ++d) {
    const std::int64_t o = d < offset.rank() ? offset[d] : 0;
    const std::int64_t e = d < extent.rank() ? extent[d] : 1;
    if (o < 0 || o + e > t.shape[d])
      throw error{errc::shape_mismatch, "sg::slice: the box reaches outside its tile in dimension " +
                                            std::to_string(d)};
    t.sel.off[d] += o;
    t.sel.ext[d] = e;
  }
  t.sel.len = detail::prod3(t.sel.ext);
  return t;
}

// a tile as a run
template <class T> run_sel<T> flatten(const tile_sel<T> &t) {
  run_sel<T> r;
  r.sel = t.sel;
  r.sel.len = detail::prod3(t.sel.ext);
  return r;
}

// An item's index in a work-item kernel, with a constant offset.
class item_index {
public:
  item_index() = default;
  item_index(int rank, const std::int64_t off[3]) : rank_(rank) {
    for (int d = 0; d < 3; ++d)
      off_[d] = off[d];
  }
  int rank() const noexcept { return rank_; }
  std::int64_t offset(int d) const noexcept { return off_[d]; }
  item_index operator+(std::int64_t k) const {
    item_index r = *this;
    r.off_[0] += k;
    return r;
  }
  item_index operator-(std::int64_t k) const { return *this + (-k); }

private:
  int rank_ = 1;
  std::int64_t off_[3] = {0, 0, 0};
};

// A value an item holds: an element it read or the result of a compute.
template <class T> class item_val {
public:
  using value_type = T;
  detail::item_input in;
  std::string describe;
};

} // namespace sg

#endif
