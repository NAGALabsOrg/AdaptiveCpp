// SPDX-License-Identifier: BSD-2-Clause
// Fixed-layout records shared by the host encoder and the device decoder. The
// bound block holds them all; it is the single data argument of the kernel.
#ifndef HIPSYCL_SG_DETAIL_DEV_ABI_HPP
#define HIPSYCL_SG_DETAIL_DEV_ABI_HPP

#include <cstdint>

namespace sg::detail {

constexpr std::uint32_t max_item_regs = 256;

enum class run_kind : std::uint32_t { plain = 0, distributed = 1, replicated = 2 };
enum class result_kind : std::uint32_t { per_item = 0, reduced = 1, binned = 2 };

} // namespace sg::detail

namespace sg::detail::dev {

constexpr std::uint64_t block_magic = 0x5347424c4f434b31ull;
constexpr std::uint32_t no_local = 0xffffffffu;

// Flag slot 0 is the ticket counter; task flags follow.
struct block_header {
  std::uint64_t magic;
  std::uint32_t n_tickets, n_tasks;
  std::uint32_t n_item_tasks, n_regs_bytes;
  std::uint64_t n_items;      // work-item kernels: the range, flattened
  std::uint64_t items_ext[3]; // work-item kernels: the range's extents
  std::uint32_t items_rank, pad0;
  std::uint64_t off_tickets, off_tasks, off_waits, off_copies, off_views, off_sels, off_item_tasks,
      off_closures, off_tables;
  std::uint64_t flags; // device address of the uint32 flags
  std::uint32_t n_flags, pad1;
};

struct ticket_rec {
  std::uint32_t first_task, n_tasks, first_copy, n_copies;
};

struct copy_rec {
  std::int32_t view;
  std::uint32_t local_off;
  std::int64_t tile;
  std::uint64_t elems;
  std::uint32_t elem_bytes, pad;
};

struct sel_rec {
  std::int32_t view;
  std::int32_t local_off; // byte offset of a local copy of the tile, or -1
  std::int64_t tile;
  std::int64_t off[3];
  std::int64_t ext[3];
  std::int64_t len;
};

struct task_rec {
  std::uint32_t task_id, flag;
  std::uint32_t first_wait, n_waits;
  std::uint32_t first_sel, n_inputs;
  std::uint32_t n_units, run_kind;
  std::uint32_t result_kind, stride; // result kind; lanes per unit
  std::int64_t run_len;              // positions of the run of one unit
  std::uint32_t seg, segs;
  std::uint32_t ips, rounds;
  std::int64_t tile_idx;
  std::uint32_t out_bytes, bins;
  std::uint32_t out_flags, local_out_off;
  std::uint64_t out_global;
  std::int64_t out_tile_elems;
  std::uint64_t eval_addr;
  std::int64_t eval_grid[3], eval_shape[3], eval_extent[3];
  std::uint64_t closure;
  std::uint32_t closure_bytes, partial_stride;
  std::uint64_t partial_addr;
  std::uint32_t unit_grid[3], pad;
};

constexpr std::uint32_t out_global = 1, out_local = 2, out_eval = 4, out_last_only = 8;

constexpr int max_item_inputs = 4;

struct item_input_rec {
  std::uint32_t kind; // 0 source element, 1 array element, 2 register
  std::int32_t view;
  std::uint32_t reg, bytes;
  std::int64_t off[3];
};

struct item_task_rec {
  std::uint32_t task_id, n_inputs;
  std::uint32_t out_reg, out_bytes;
  std::uint32_t closure_bytes, eval, sink, pad;
  std::uint64_t closure;
  std::uint64_t eval_addr;
  std::int64_t eval_extent[3];
  item_input_rec in[max_item_inputs];
};

static_assert(sizeof(ticket_rec) % 8 == 0 && sizeof(copy_rec) % 8 == 0 && sizeof(sel_rec) % 8 == 0 &&
              sizeof(task_rec) % 8 == 0 && sizeof(item_task_rec) % 8 == 0 && sizeof(block_header) % 8 == 0);

} // namespace sg::detail::dev

#endif
