// SPDX-License-Identifier: BSD-2-Clause
// Reading the bound block: one definition shared by the host encoder and the
// device decoder. Every record is addressed from the block's start and every
// field is accessed through its naturally aligned type.
#ifndef HIPSYCL_SG_DETAIL_DEV_LAYOUT_HPP
#define HIPSYCL_SG_DETAIL_DEV_LAYOUT_HPP

#include "hipSYCL/sg/detail/dev/abi.hpp"
#include "hipSYCL/sg/detail/view_desc.hpp"

#include <cstddef>
#include <cstdint>

namespace sg::detail::dev {

using block_t = const std::byte *;

inline const block_header &header_of(block_t b) { return *reinterpret_cast<const block_header *>(b); }

inline const ticket_rec *tickets_of(block_t b) {
  return reinterpret_cast<const ticket_rec *>(b + header_of(b).off_tickets);
}
inline const task_rec *tasks_of(block_t b) { return reinterpret_cast<const task_rec *>(b + header_of(b).off_tasks); }
inline const std::uint32_t *waits_of(block_t b) {
  return reinterpret_cast<const std::uint32_t *>(b + header_of(b).off_waits);
}
inline const copy_rec *copies_of(block_t b) { return reinterpret_cast<const copy_rec *>(b + header_of(b).off_copies); }
inline const view_node *views_of(block_t b) { return reinterpret_cast<const view_node *>(b + header_of(b).off_views); }
inline const sel_rec *sels_of(block_t b) { return reinterpret_cast<const sel_rec *>(b + header_of(b).off_sels); }
inline const item_task_rec *item_tasks_of(block_t b) {
  return reinterpret_cast<const item_task_rec *>(b + header_of(b).off_item_tasks);
}
inline std::uint32_t *flags_of(block_t b) { return reinterpret_cast<std::uint32_t *>(header_of(b).flags); }

// Sizes of the sections, in bytes, as the encoder lays them out in this order.
struct block_sizes {
  std::uint64_t tickets = 0, tasks = 0, waits = 0, copies = 0, views = 0, sels = 0, item_tasks = 0,
                closures = 0, tables = 0;
};

inline std::uint64_t align_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1) / a * a; }

// Fills the header's offsets for the given section sizes; returns the block size.
inline std::uint64_t lay_out(block_header &h, const block_sizes &s) {
  std::uint64_t at = align_up(sizeof(block_header), 16);
  h.off_tickets = at;
  at = align_up(at + s.tickets, 16);
  h.off_tasks = at;
  at = align_up(at + s.tasks, 16);
  h.off_waits = at;
  at = align_up(at + s.waits, 16);
  h.off_copies = at;
  at = align_up(at + s.copies, 16);
  h.off_views = at;
  at = align_up(at + s.views, 16);
  h.off_sels = at;
  at = align_up(at + s.sels, 16);
  h.off_item_tasks = at;
  at = align_up(at + s.item_tasks, 16);
  h.off_closures = at;
  at = align_up(at + s.closures, 16);
  h.off_tables = at;
  at = align_up(at + s.tables, 16);
  return align_up(at, 256);
}

} // namespace sg::detail::dev

#endif
