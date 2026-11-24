// SPDX-License-Identifier: BSD-2-Clause
// What a submit records: the roles on handles and the kernel graph the builder
// described. Nothing here is code; selections and tables are made at preparation.
#ifndef HIPSYCL_SG_DETAIL_SCOPE_HPP
#define HIPSYCL_SG_DETAIL_SCOPE_HPP

#include "hipSYCL/sg/detail/closure.hpp"
#include "hipSYCL/sg/detail/dev/abi.hpp"
#include "hipSYCL/sg/detail/type_id.hpp"
#include "hipSYCL/sg/detail/view_desc.hpp"
#include "hipSYCL/sg/error.hpp"
#include "hipSYCL/sg/launch_value.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace sg {
class kernel_handle;
class host_handle;

// A position: tile coordinates, unit coordinates, item coordinates.
class idx {
public:
  idx() = default;
  idx(std::initializer_list<std::int64_t> l) {
    for (std::int64_t x : l)
      if (rank_ < 3)
        v_[rank_++] = x;
  }
  template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
  idx(I x) : rank_(1) {
    v_[0] = static_cast<std::int64_t>(x);
  }
  std::int64_t operator[](int d) const { return v_[d]; }
  std::int64_t &operator[](int d) { return v_[d]; }
  int rank() const noexcept { return rank_; }
  void set_rank(int r) noexcept { rank_ = r; }

private:
  std::int64_t v_[3] = {0, 0, 0};
  int rank_ = 0;
};

} // namespace sg

namespace sg::detail {

// What a unit body needs to know while it runs at preparation.
struct unit_env {
  // The segment count when the view reads a distributed result of the unit's own set, else 0.
  std::function<std::int64_t(int)> distributed_segments;
};

class unit_ctx {
public:
  idx tile;
  idx id;
  std::int64_t run_len = 1;
  const unit_env *env = nullptr;
};

enum class role_mode : std::uint32_t { read, produce, inplace };
enum class init_kind : std::uint32_t { none, value, reuse };

struct array_slot {
  std::size_t handle = 0;
  std::uint64_t type = 0;
  std::size_t elem_bytes = 0;
  role_mode mode = role_mode::read;
  bool is_host = false;
  bool whole = true;
  std::size_t offset = 0;
  int rank = 1;
  bdims ext;                           // as the builder wrote it
  std::uint64_t extent[3] = {1, 1, 1}; // bound for one run
  std::string op;
  std::string op_symbol; // the registered combine entry, inplace only
  init_kind init = init_kind::none;
  std::vector<std::byte> init_bits;
};

class graph_impl;

struct host_slot_table {
  std::vector<void *> address;
  std::vector<std::array<std::uint64_t, 3>> extent;
};

// One pulled window: a tile of a view and a box of its elements, taken as a run.
struct input_sel {
  int view = no_view;
  std::int64_t tile = 0;
  std::int64_t off[3] = {0, 0, 0};
  std::int64_t ext[3] = {1, 1, 1};
  std::int64_t len = 0;
  int local_copy = -1; // index of the group's local copy of the tile, if marked
};

struct sizing_info {
  std::int64_t bins = 0;
  std::size_t workspace_bytes = 0; // per item
  std::size_t partial_bytes = 0;   // per unit and segment
};

struct unit_eval_out {
  run_kind kind = run_kind::plain;
  std::vector<input_sel> sels;
  closure_entry closure;
};

// A recorded view. Element views describe an array; tile views describe tiles.
struct view_rec {
  vk kind = vk::source;
  int parent = no_view, parent2 = no_view;
  std::uint32_t elem_bytes = 0;
  int slot = -1; // source: the array slot of the node
  int map = -1;  // result and array: the producing map
  bdims ext;     // element views: extent as written; its rank is the rank of the elements
  dims shape;    // tile views: tile shape; its rank is the tile rank
  int grid_rank = 0;
  oob_kind oob = oob_kind::fill;
  std::uint64_t fill[2] = {0, 0};
  // define_tiles
  bool define = false;
  bound_expr define_n;
  std::function<std::int64_t(std::int64_t, std::int64_t)> define_fn;
  // stencil
  bool stencil = false;
  std::vector<std::array<std::int64_t, 3>> offsets;
  bdims roots_begin, roots_end;
  std::uint32_t table_rank = 1;
  std::string name;
  bool alias_prev = false; // repeat: stands for the previous iteration's result
};

struct work_rec {
  bool items = false;
  bool segmented = false;
  bool slice = false;
  int root = -1;  // slice: the work set whose groups are reused
  bdims grid;     // work tiles; slice: unused; items: the range
  dims tile;      // units of one tile; slice: the slice's tile shape
  std::int64_t run = 1;  // run length of one unit
  std::int64_t segs = 1, ips = 1; // segmented: segments and items per segment
  bdims offset, extent, stride;   // slice
  source_loc where;
};

struct map_rec {
  int work = -1;
  std::function<unit_eval_out(unit_ctx &)> eval;
  std::function<sizing_info(const closure_entry &)> sizing;
  std::function<int(int)> remap; // instances of a repeated section: the views their body sees
  std::string task_symbol;
  int task = -1;
  run_kind run = run_kind::plain;
  std::uint32_t result_bytes = 0;
  result_kind rk = result_kind::per_item;
  std::size_t n_inputs = 0;
  std::vector<std::uint32_t> input_bytes;
  int result_view = -1;
  source_loc where;
};

enum class item_in_kind : std::uint32_t { source = 0, array = 1, reg = 2 };

struct item_input {
  item_in_kind kind = item_in_kind::source;
  int view = no_view;    // source or array view
  int reg = -1;          // register slot of an earlier compute
  std::int64_t off[3] = {0, 0, 0};
  std::uint32_t bytes = 0;
};

struct item_task_rec {
  std::string task_symbol;
  int task = -1;
  closure_entry closure;
  std::vector<item_input> inputs;
  int out_reg = 0;
  std::uint32_t out_bytes = 0;
};

struct item_map_rec {
  int work = -1;
  std::vector<item_task_rec> tasks;
  int result_view = -1;
  std::uint32_t result_bytes = 0;
  source_loc where;
};

struct sink_rec {
  int slot = 0;
  int view = no_view; // elements: a join of result tiles, or a result array
  bool items = false;
  int sink_task = -1; // in-place results: the task that combines into the result
};

enum class mark_kind : std::uint32_t { local, segmented, array_tiles };

struct mark_rec {
  mark_kind kind = mark_kind::local;
  int view = no_view; // result tiles
  int work = -1;      // array tiles: the work set
  source_loc where;
};

struct launch_slot_rec {
  std::size_t handle = 0;
  std::uint64_t type = 0;
  std::uint32_t bytes = 0;
  bool is_float = false;
};

enum class ev_kind : std::uint32_t { view, work, map, imap };
struct ev_rec {
  ev_kind kind;
  int idx;
};

struct repeat_rec {
  bound_expr n;
  int start_view = no_view, prev_view = no_view, result_view = no_view;
  int vb = 0, ve = 0, wb = 0, we = 0, mb = 0, me = 0, ib = 0, ie = 0;
  std::size_t eb = 0, ee = 0; // event range
};

struct kgraph {
  std::vector<view_rec> views;
  std::vector<work_rec> works;
  std::vector<map_rec> maps;
  std::vector<item_map_rec> imaps;
  std::vector<ev_rec> events;
  std::vector<std::string> tasks;
  std::vector<sink_rec> sinks;
  std::vector<mark_rec> marks;
  std::vector<launch_slot_rec> launches;
  std::vector<repeat_rec> repeats;
  bool has_tiles = false, has_items = false;
  std::int64_t group_size_cap = 1024;

  int add_view(view_rec v) {
    views.push_back(std::move(v));
    events.push_back({ev_kind::view, static_cast<int>(views.size()) - 1});
    return static_cast<int>(views.size()) - 1;
  }
  int add_work(work_rec w) {
    works.push_back(std::move(w));
    events.push_back({ev_kind::work, static_cast<int>(works.size()) - 1});
    return static_cast<int>(works.size()) - 1;
  }
  int add_map(map_rec m) {
    maps.push_back(std::move(m));
    events.push_back({ev_kind::map, static_cast<int>(maps.size()) - 1});
    return static_cast<int>(maps.size()) - 1;
  }
  int add_imap(item_map_rec m) {
    imaps.push_back(std::move(m));
    events.push_back({ev_kind::imap, static_cast<int>(imaps.size()) - 1});
    return static_cast<int>(imaps.size()) - 1;
  }
  int task_index(const std::string &symbol) {
    for (std::size_t i = 0; i < tasks.size(); ++i)
      if (tasks[i] == symbol)
        return static_cast<int>(i);
    tasks.push_back(symbol);
    return static_cast<int>(tasks.size()) - 1;
  }
};

struct node_record {
  enum class kind_t { kernel, host_work, interop } kind = kind_t::kernel;
  int device = 0;
  graph_impl *owner = nullptr;
  std::vector<array_slot> slots;
  kgraph graph;
  std::function<void()> host_fn;
  std::shared_ptr<host_slot_table> host_table;
  std::uint64_t builder_type = 0;
};

inline node_record *&scope_slot() noexcept {
  static thread_local node_record *s = nullptr;
  return s;
}
inline node_record *current_scope() noexcept { return scope_slot(); }
inline void set_current_scope(node_record *n) noexcept { scope_slot() = n; }

} // namespace sg::detail

#endif
