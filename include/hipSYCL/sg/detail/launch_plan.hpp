// SPDX-License-Identifier: BSD-2-Clause
// Resolves a prepared kernel into the numbers of one launch: the native group
// size, tickets and their task lists, flags, local memory and the one bound block
// that holds everything the device reads.
#ifndef HIPSYCL_SG_DETAIL_LAUNCH_PLAN_HPP
#define HIPSYCL_SG_DETAIL_LAUNCH_PLAN_HPP

#include "hipSYCL/sg/detail/backing_plan.hpp"
#include "hipSYCL/sg/detail/dev/abi.hpp"
#include "hipSYCL/sg/detail/dev/layout.hpp"
#include "hipSYCL/sg/detail/kernel_preparer.hpp"
#include "hipSYCL/sg/detail/tile_graph.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace sg::detail {

struct device_limits {
  std::size_t max_work_group_size = 1024;
  std::size_t local_mem_bytes = 48 * 1024;
  std::size_t max_groups = 0xffffffffull;
};

struct slot_binding {
  std::uint64_t address = 0;
  std::uint64_t extent[3] = {1, 1, 1};
};

inline std::size_t local_bucket(std::size_t bytes) {
  std::size_t b = 256;
  while (b < bytes)
    b *= 2;
  return b;
}

struct launch_plan {
  std::vector<std::byte> block;
  std::uint64_t base = 0; // device address of the toolchain region; the block is at its start
  std::size_t flags_byte_offset = 0, flag_words = 0, toolchain_bytes = 0;
  std::size_t groups = 0, group_size = 1, local_bytes = 256;
  bool items = false;
  std::uint64_t n_items = 0;
  std::vector<std::string> tasks; // the composition: task functions in index order
  std::vector<value_backing> backing;
};

namespace plan_detail {

inline std::size_t a16(std::size_t v) { return (v + 15) / 16 * 16; }
inline std::size_t a256(std::size_t v) { return (v + 255) / 256 * 256; }

template <class T> inline void put(std::vector<std::byte> &out, std::size_t at, const std::vector<T> &v) {
  if (!v.empty())
    std::memcpy(out.data() + at, v.data(), v.size() * sizeof(T));
}

} // namespace plan_detail

inline launch_plan encode_launch(prepared_kernel &p, const tile_graph &tg, const std::vector<slot_binding> &slots,
                                 std::uint64_t base_address, const device_limits &lim) {
  using namespace plan_detail;
  const kgraph &g = p.graph;
  launch_plan lp;
  lp.base = base_address;
  lp.tasks = g.tasks;
  lp.items = p.items;
  lp.n_items = static_cast<std::uint64_t>(p.n_items);

  if (static_cast<std::size_t>(p.native_size) > lim.max_work_group_size)
    throw error{errc::work_group_too_large, "the largest work tile has " + std::to_string(p.native_size) +
                                                " items; the device's work-group limit is " +
                                                std::to_string(lim.max_work_group_size)};
  lp.group_size = static_cast<std::size_t>(std::max<std::int64_t>(1, p.native_size));

  const std::vector<eval_layout> layouts = compute_eval_layouts(p);
  lp.backing = compute_backing(p, tg);

  // ---- local memory: workspace, local results, then the copies of one ticket ----
  std::size_t ws_bytes = 16;
  for (std::size_t m = 0; m < g.maps.size(); ++m) {
    const map_conc &mc = p.maps[m];
    std::size_t need = lp.group_size * a16(std::max<std::size_t>(1, mc.sizing.workspace_bytes));
    need = std::max<std::size_t>(need, static_cast<std::size_t>(mc.units * mc.bins * 4));
    ws_bytes = std::max(ws_bytes, a16(need));
  }
  std::vector<std::size_t> local_res(g.maps.size(), 0);
  std::size_t local_total = ws_bytes;
  for (std::size_t m = 0; m < g.maps.size(); ++m) {
    const value_backing &vb = lp.backing[m];
    if (vb.store_local && !vb.store_global) {
      local_res[m] = local_total;
      local_total += a16(static_cast<std::size_t>(p.maps[m].tile_elems) * g.maps[m].result_bytes);
    }
  }

  // ---- global regions: flags, partials, stored results ----
  const std::size_t block_reserve = 0; // the block is sized below; regions follow it
  (void)block_reserve;
  std::vector<std::size_t> global_res(g.maps.size(), 0), partial_res(g.maps.size(), 0);
  std::size_t region = 0; // relative to the end of the flags
  for (std::size_t m = 0; m < g.maps.size(); ++m) {
    const map_conc &mc = p.maps[m];
    const work_conc &w = p.works[g.maps[m].work];
    if (lp.backing[m].store_global) {
      global_res[m] = region;
      region += a256(static_cast<std::size_t>(mc.tiles * mc.tile_elems) * g.maps[m].result_bytes);
    }
    if (mc.kind != run_kind::plain && mc.tiles > 0) {
      partial_res[m] = region;
      const std::int64_t segs = w.segmented ? w.segs : 1;
      region += a256(static_cast<std::size_t>(mc.tiles * segs * mc.units) * mc.partial_stride);
    }
  }

  // ---- views and tables ----
  std::vector<view_node> views = p.views;
  std::vector<std::int64_t> tables;
  std::vector<std::size_t> table_at(views.size(), 0);
  for (std::size_t v = 0; v < views.size(); ++v) {
    const view_rec &r = g.views[v];
    view_node &n = views[v];
    switch (r.kind) {
    case vk::source:
      n.base = slots.at(static_cast<std::size_t>(r.slot)).address;
      break;
    case vk::table:
      table_at[v] = tables.size();
      tables.insert(tables.end(), p.tables[v].begin(), p.tables[v].end());
      break;
    case vk::result: {
      const std::size_t m = static_cast<std::size_t>(r.map);
      const value_backing &vb = lp.backing[m];
      if (vb.store_local && !vb.store_global) {
        n.base_local = 1;
        n.base = local_res[m];
      } else {
        n.base_local = 0;
      }
      break;
    }
    default:
      break;
    }
  }

  // ---- tickets, tasks, selections ----
  std::vector<dev::ticket_rec> tickets;
  std::vector<dev::task_rec> tasks;
  std::vector<std::uint32_t> waits;
  std::vector<dev::copy_rec> copies;
  std::vector<dev::sel_rec> sels;
  std::vector<std::byte> closures;
  std::vector<std::size_t> closure_at(g.maps.size(), 0);
  for (std::size_t m = 0; m < g.maps.size(); ++m) {
    closure_at[m] = closures.size();
    const closure_entry &ce = p.maps[m].closure;
    closures.insert(closures.end(), ce.bytes.begin(), ce.bytes.end());
    closures.resize(a16(closures.size()), std::byte{0});
  }

  auto marked = [&](int map, int view) {
    for (const mark_rec &mk : g.marks)
      if (mk.kind == mark_kind::array_tiles && mk.view == view && mk.work == g.maps[map].work)
        return true;
    return false;
  };

  std::size_t max_local = local_total;
  for (std::size_t t = 0; t < tg.group_of_ticket.size(); ++t) {
    const int group = tg.group_of_ticket[t];
    dev::ticket_rec tr{};
    tr.first_task = static_cast<std::uint32_t>(tasks.size());
    tr.first_copy = static_cast<std::uint32_t>(copies.size());
    std::map<std::pair<int, std::int64_t>, std::size_t> copy_at;
    std::size_t local_at = local_total;
    // the copies of this ticket first, so the selections can name them
    for (int ti : tg.group_tasks[group]) {
      const task_inst &inst = tg.tasks[ti];
      const map_conc &mc = p.maps[inst.map];
      for (std::int64_t u = 0; u < mc.units; ++u)
        for (std::size_t j = 0; j < mc.n_inputs; ++j) {
          const input_sel &s = mc.sel(inst.tile, u, j);
          if (!marked(inst.map, s.view) || copy_at.count({s.view, s.tile}))
            continue;
          const view_node &vn = views[s.view];
          dev::copy_rec c{};
          c.view = s.view;
          c.tile = s.tile;
          c.elems = static_cast<std::uint64_t>(prod3(vn.shape));
          c.elem_bytes = vn.elem_bytes;
          c.local_off = static_cast<std::uint32_t>(local_at);
          copy_at[{s.view, s.tile}] = local_at;
          local_at += a16(c.elems * c.elem_bytes);
          copies.push_back(c);
        }
    }
    tr.n_copies = static_cast<std::uint32_t>(copies.size()) - tr.first_copy;
    max_local = std::max(max_local, local_at);

    for (int ti : tg.group_tasks[group]) {
      const task_inst &inst = tg.tasks[ti];
      const map_conc &mc = p.maps[inst.map];
      const map_rec &mr = g.maps[inst.map];
      const work_conc &w = p.works[mr.work];
      const value_backing &vb = lp.backing[inst.map];
      dev::task_rec r{};
      r.task_id = static_cast<std::uint32_t>(mr.task);
      r.flag = inst.flag;
      r.first_wait = static_cast<std::uint32_t>(waits.size());
      r.n_waits = static_cast<std::uint32_t>(inst.waits.size());
      waits.insert(waits.end(), inst.waits.begin(), inst.waits.end());
      r.first_sel = static_cast<std::uint32_t>(sels.size());
      r.n_inputs = static_cast<std::uint32_t>(mc.n_inputs);
      r.n_units = static_cast<std::uint32_t>(mc.units);
      r.run_kind = static_cast<std::uint32_t>(mc.kind);
      r.result_kind = static_cast<std::uint32_t>(mc.rk);
      r.stride = static_cast<std::uint32_t>(mc.stride);
      r.run_len = mc.run_used;
      r.seg = static_cast<std::uint32_t>(inst.seg);
      r.segs = static_cast<std::uint32_t>(w.segmented ? w.segs : 1);
      r.ips = static_cast<std::uint32_t>(w.ips);
      r.rounds = static_cast<std::uint32_t>(mc.rounds);
      r.tile_idx = inst.tile;
      r.out_bytes = mr.result_bytes;
      r.bins = static_cast<std::uint32_t>(mc.bins);
      r.out_tile_elems = mc.tile_elems;
      const bool consumed = vb.store_global || vb.store_local;
      if (vb.store_global) {
        r.out_flags |= dev::out_global;
        r.out_global = base_address + 0; // fixed below, once the region base is known
      }
      if (vb.store_local && !vb.store_global) {
        r.out_flags |= dev::out_local;
        r.local_out_off = static_cast<std::uint32_t>(local_res[inst.map]);
      }
      (void)consumed;
      if (layouts[inst.map].evaluated) {
        const eval_layout &l = layouts[inst.map];
        r.out_flags |= dev::out_eval;
        r.eval_addr = slots.at(static_cast<std::size_t>(l.slot)).address;
        for (int d = 0; d < 3; ++d) {
          r.eval_grid[d] = l.grid[d];
          r.eval_shape[d] = l.shape[d];
          r.eval_extent[d] = l.extent[d];
        }
      }
      r.closure = closure_at[inst.map]; // relative to the closure section until the layout is known
      r.closure_bytes = static_cast<std::uint32_t>(mc.closure.bytes.size());
      r.partial_stride = mc.partial_stride;
      for (int d = 0; d < 3; ++d)
        r.unit_grid[d] = static_cast<std::uint32_t>(w.tile[d]);
      for (std::int64_t u = 0; u < mc.units; ++u)
        for (std::size_t j = 0; j < mc.n_inputs; ++j) {
          const input_sel &s = mc.sel(inst.tile, u, j);
          dev::sel_rec o{};
          o.view = s.view;
          o.tile = s.tile;
          o.len = s.len;
          for (int d = 0; d < 3; ++d) {
            o.off[d] = s.off[d];
            o.ext[d] = s.ext[d];
          }
          const auto it = copy_at.find({s.view, s.tile});
          o.local_off = it == copy_at.end() ? -1 : static_cast<std::int32_t>(it->second);
          sels.push_back(o);
        }
      tasks.push_back(r);
    }
    tr.n_tasks = static_cast<std::uint32_t>(tasks.size()) - tr.first_task;
    tickets.push_back(tr);
  }

  if (max_local > lim.local_mem_bytes)
    throw error{errc::local_memory_exceeded, "a native group needs " + std::to_string(max_local) +
                                                 " bytes of local memory; the device offers " +
                                                 std::to_string(lim.local_mem_bytes)};
  lp.local_bytes = local_bucket(max_local);
  lp.groups = tickets.size();

  // ---- work-item tasks ----
  std::vector<dev::item_task_rec> item_tasks;
  std::vector<std::size_t> item_closure_at;
  if (p.items) {
    std::vector<std::int64_t> view_reg(g.views.size(), -1);
    std::vector<std::int64_t> view_last_task(g.views.size(), -1);
    auto reg_bytes = [](int reg) { return static_cast<std::uint32_t>(reg) * 16u; };
    for (const item_map_rec &im : g.imaps) {
      for (const item_task_rec &t : im.tasks) {
        dev::item_task_rec r{};
        r.task_id = static_cast<std::uint32_t>(t.task);
        r.n_inputs = static_cast<std::uint32_t>(t.inputs.size());
        r.out_reg = reg_bytes(t.out_reg);
        r.out_bytes = t.out_bytes;
        if (r.out_reg + 16 > max_item_regs)
          throw error{errc::limit_exceeded, "a work item needs more than " + std::to_string(max_item_regs / 16) +
                                                " intermediate values"};
        r.closure_bytes = static_cast<std::uint32_t>(t.closure.bytes.size());
        item_closure_at.push_back(closures.size());
        closures.insert(closures.end(), t.closure.bytes.begin(), t.closure.bytes.end());
        closures.resize(a16(closures.size()), std::byte{0});
        for (std::size_t k = 0; k < t.inputs.size(); ++k) {
          const item_input &in = t.inputs[k];
          dev::item_input_rec &o = r.in[k];
          o.bytes = in.bytes;
          for (int d = 0; d < 3; ++d)
            o.off[d] = in.off[d];
          if (in.kind == item_in_kind::source) {
            o.kind = 0;
            o.view = in.view;
          } else if (in.kind == item_in_kind::array) {
            o.kind = 2;
            const std::int64_t reg = view_reg[static_cast<std::size_t>(in.view)];
            if (reg < 0)
              throw error{errc::internal, "a work item reads an array no earlier map made"};
            o.reg = static_cast<std::uint32_t>(reg);
          } else {
            o.kind = 2;
            o.reg = reg_bytes(in.reg);
          }
        }
        item_tasks.push_back(r);
      }
      if (!im.tasks.empty()) {
        view_reg[static_cast<std::size_t>(im.result_view)] = reg_bytes(im.tasks.back().out_reg);
        view_last_task[static_cast<std::size_t>(im.result_view)] =
            static_cast<std::int64_t>(item_tasks.size()) - 1;
      }
    }
    for (const sink_rec &s : g.sinks) {
      if (!s.items)
        continue;
      const std::int64_t last = view_last_task[static_cast<std::size_t>(s.view)];
      if (last < 0)
        throw error{errc::internal, "a result array has no map that makes it"};
      const std::uint64_t addr = slots.at(static_cast<std::size_t>(s.slot)).address;
      if (s.sink_task < 0) {
        dev::item_task_rec &r = item_tasks[static_cast<std::size_t>(last)];
        r.eval = 1;
        r.eval_addr = addr;
      } else {
        dev::item_task_rec r{};
        r.task_id = static_cast<std::uint32_t>(s.sink_task);
        r.n_inputs = 1;
        r.sink = 1;
        r.eval_addr = addr;
        r.in[0].kind = 2;
        r.in[0].reg = static_cast<std::uint32_t>(view_reg[static_cast<std::size_t>(s.view)]);
        item_closure_at.push_back(closures.size());
        item_tasks.push_back(r);
      }
    }
  }

  // ---- layout ----
  dev::block_header h{};
  h.magic = dev::block_magic;
  h.n_tickets = static_cast<std::uint32_t>(tickets.size());
  h.n_tasks = static_cast<std::uint32_t>(tasks.size());
  h.n_item_tasks = static_cast<std::uint32_t>(item_tasks.size());
  h.n_regs_bytes = max_item_regs;
  h.n_items = lp.n_items;
  for (int d = 0; d < 3; ++d)
    h.items_ext[d] = static_cast<std::uint64_t>(p.items_ext[d]);
  h.items_rank = static_cast<std::uint32_t>(p.items_rank);

  dev::block_sizes sz;
  sz.tickets = tickets.size() * sizeof(dev::ticket_rec);
  sz.tasks = tasks.size() * sizeof(dev::task_rec);
  sz.waits = waits.size() * sizeof(std::uint32_t);
  sz.copies = copies.size() * sizeof(dev::copy_rec);
  sz.views = views.size() * sizeof(view_node);
  sz.sels = sels.size() * sizeof(dev::sel_rec);
  sz.item_tasks = item_tasks.size() * sizeof(dev::item_task_rec);
  sz.closures = closures.size();
  sz.tables = tables.size() * sizeof(std::int64_t);
  const std::uint64_t block_bytes = dev::lay_out(h, sz);

  lp.flag_words = static_cast<std::size_t>(p.n_flags);
  lp.flags_byte_offset = static_cast<std::size_t>(block_bytes);
  const std::size_t flags_end = a256(lp.flags_byte_offset + lp.flag_words * 4);
  lp.toolchain_bytes = flags_end + region;
  h.flags = base_address + lp.flags_byte_offset;
  h.n_flags = static_cast<std::uint32_t>(lp.flag_words);

  // addresses that needed the layout
  for (dev::task_rec &r : tasks) {
    r.closure += h.off_closures;
  }
  {
    std::size_t k = 0;
    for (std::size_t t = 0; t < tg.group_of_ticket.size(); ++t)
      for (int ti : tg.group_tasks[tg.group_of_ticket[t]]) {
        const task_inst &inst = tg.tasks[ti];
        dev::task_rec &r = tasks[k++];
        if (r.out_flags & dev::out_global)
          r.out_global = base_address + flags_end + global_res[inst.map];
        if (p.maps[inst.map].kind != run_kind::plain)
          r.partial_addr = base_address + flags_end + partial_res[inst.map];
      }
  }
  for (std::size_t i = 0; i < item_tasks.size(); ++i) {
    dev::item_task_rec &r = item_tasks[i];
    r.closure = h.off_closures + (i < item_closure_at.size() ? item_closure_at[i] : 0);
  }
  for (std::size_t v = 0; v < views.size(); ++v) {
    if (static_cast<vk>(views[v].kind) == vk::table)
      views[v].table = base_address + h.off_tables + table_at[v] * sizeof(std::int64_t);
    else if (static_cast<vk>(views[v].kind) == vk::result && !views[v].base_local) {
      const std::size_t m = static_cast<std::size_t>(g.views[v].map);
      views[v].base = lp.backing[m].store_global ? base_address + flags_end + global_res[m] : 0;
    }
  }

  lp.block.assign(static_cast<std::size_t>(block_bytes), std::byte{0});
  std::memcpy(lp.block.data(), &h, sizeof(h));
  put(lp.block, h.off_tickets, tickets);
  put(lp.block, h.off_tasks, tasks);
  put(lp.block, h.off_waits, waits);
  put(lp.block, h.off_copies, copies);
  put(lp.block, h.off_views, views);
  put(lp.block, h.off_sels, sels);
  put(lp.block, h.off_item_tasks, item_tasks);
  put(lp.block, h.off_closures, closures);
  put(lp.block, h.off_tables, tables);
  return lp;
}

} // namespace sg::detail

#endif
