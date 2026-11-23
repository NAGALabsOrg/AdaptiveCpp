#pragma once
#include "hipSYCL/sg/detail/dev/drivers.hpp"
#include "hipSYCL/sg/detail/dev/group_eval.hpp"
#include "hipSYCL/sg/detail/dev/layout.hpp"

#include <boost/test/unit_test.hpp>

#include <sycl/sycl.hpp>

#include <cstring>
#include <vector>

namespace sgd {
using namespace sg::detail::dev;

constexpr std::size_t log_entries = 4096;
constexpr std::size_t log_bytes = log_entries * 32;
constexpr std::size_t off_ctr = log_bytes;
constexpr std::size_t off_ctx = off_ctr + 64;
constexpr std::size_t off_layers = off_ctx + 64;
constexpr std::size_t off_nodes = off_layers + 1024;
constexpr std::size_t off_links = off_nodes + 4096;
constexpr std::size_t off_flags = off_links + 4096;
constexpr std::size_t off_out = off_flags + 1024;
constexpr std::size_t capacity = off_out + 65536;

struct log_entry {
  std::uint32_t tag, a, b, pad;
  std::uint64_t g, l, x;
};

enum tag : std::uint32_t {
  t_positions = 1, t_item_value, t_prep, t_run_source, t_resolve_tree, t_global_resolve, t_body,
  t_group_body, t_write, t_run_layer
};

// The mock policy keeps its log in the block the launch_ctx lives in.
inline std::uint32_t *log_counter(ctx_t c) {
  return reinterpret_cast<std::uint32_t *>(const_cast<char *>(reinterpret_cast<const char *>(c)) - 64);
}
inline log_entry *log_base(ctx_t c) {
  return reinterpret_cast<log_entry *>(const_cast<char *>(reinterpret_cast<const char *>(c)) - 64 -
                                       log_bytes);
}
inline void record(ctx_t c, std::uint32_t t, std::uint32_t a, std::uint32_t b, std::uint64_t g,
                   std::uint64_t l, std::uint64_t x = 0) {
  sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ctr{*log_counter(c)};
  const std::uint32_t i = ctr.fetch_add(1);
  if (i < log_entries)
    log_base(c)[i] = log_entry{t, a, b, 0, g, l, x};
}

// Mode 0: every native group runs owner. Mode 1: every native group calls
// resolve_layer. Mode 2: native group with g == 0 calls resolve_layer, the one
// with g == 1 marks flag 0 done.
template <int Mode> struct mock {
  static void positions(ctx_t c, std::uint32_t node, std::uint32_t link, std::uint64_t g,
                        std::uint32_t slot, std::uint64_t *src) {
    *src = slot + 100u * link;
    record(c, t_positions, node, link, g, slot, *src);
  }
  static void item_value(ctx_t c, std::uint32_t node, std::uint32_t link, std::uint64_t srcg,
                         std::uint64_t l, void *, void *) {
    record(c, t_item_value, node, link, srcg, l);
  }
  static void prep(ctx_t c, std::uint32_t node, std::uint32_t link, std::uint64_t srcg, const void *,
                   void *) {
    record(c, t_prep, node, link, srcg, 0);
  }
  static void run_source(ctx_t c, std::uint32_t node, std::uint32_t link, std::uint64_t srcg,
                         const void *, void *, const tile_out *out) {
    record(c, t_run_source, node, link, srcg, 0, out->base);
  }
  static void resolve_tree(ctx_t c, std::uint32_t node, std::uint32_t link, std::uint64_t srcg,
                           const void *, void *) {
    record(c, t_resolve_tree, node, link, srcg, 0);
  }
  static void global_resolve(ctx_t c, std::uint32_t node, std::uint32_t link, std::uint64_t srcg,
                             const void *, void *) {
    record(c, t_global_resolve, node, link, srcg, 0);
  }
  static void body(ctx_t c, std::uint32_t node, std::uint64_t g, std::uint64_t l, void *, void *dst) {
    const std::uint32_t flag = *flag_ptr(c, 0, g);
    record(c, t_body, node, flag, g, l, reinterpret_cast<std::uint64_t>(dst));
  }
  static void group_body(ctx_t c, std::uint32_t node, const void *, std::uint64_t g, void *,
                         const tile_out *out) {
    record(c, t_group_body, node, 0, g, 0, out->base);
  }
  static void write(ctx_t c, std::uint32_t node, void *dst, const void *src) {
    std::uint8_t *d = static_cast<std::uint8_t *>(dst);
    const std::uint8_t *s = static_cast<const std::uint8_t *>(src);
    for (std::uint64_t i = 0; i < node_at(c, node).elem_bytes; ++i)
      d[i] = s[i];
    record(c, t_write, node, 0, 0, 0, reinterpret_cast<std::uint64_t>(dst));
  }
  static void run_layer(ctx_t c, std::uint32_t layer, std::uint64_t g, const void *nd, void *frame) {
    const std::uint32_t root = layer_at(c, layer).root_node;
    if constexpr (Mode == 0) {
      owner<mock<Mode>>(c, root, layer, g, nd, frame);
    } else if constexpr (Mode == 1) {
      resolve_layer<mock<Mode>>(c, root, layer, g, nd, frame);
    } else {
      const std::uint32_t first = uniform(nd, g == 0 ? 1u : 0u);
      if (first) {
        resolve_layer<mock<Mode>>(c, root, layer, 0, nd, frame);
      } else if (local_id_of(nd) == 0) {
        sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            f{*flag_ptr(c, layer, 0)};
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
        f.store(flag_done, sycl::memory_order::release);
      }
    }
    record(c, t_run_layer, layer, 0, g, 0);
  }
};

struct world {
  sycl::queue q;
  char *d;
  std::vector<layer_rec> layers;
  std::vector<node_rec> nodes;
  std::vector<link_rec> links;
  std::vector<std::uint32_t> flags;
  std::vector<std::size_t> flag_base;
  world() : q(sycl::default_selector_v), d(sycl::malloc_device<char>(capacity, q)) {
    q.memset(d, 0, capacity).wait();
  }
  ~world() { sycl::free(d, q); }
  std::uint64_t out_addr(std::size_t bytes_in) const {
    return reinterpret_cast<std::uint64_t>(d + off_out + bytes_in);
  }
  std::size_t add_layer(std::uint64_t groups, std::uint32_t root, std::uint64_t backing_bytes_off,
                        std::uint64_t extent0) {
    layer_rec l{};
    l.groups = groups;
    l.root_node = root;
    l.extent[0] = extent0;
    l.extent[1] = 1;
    l.extent[2] = 1;
    l.backing = out_addr(backing_bytes_off);
    std::size_t first = flags.size();
    flag_base.push_back(first);
    flags.resize(first + groups, 0);
    l.flags = reinterpret_cast<std::uint64_t>(d + off_flags) + 4 * first;
    if (!layers.empty())
      l.range_offset = layers.back().range_offset + layers.back().groups;
    layers.push_back(l);
    return layers.size() - 1;
  }
  std::size_t add_node(std::uint64_t groups, std::uint64_t gs, std::uint32_t body_kind,
                       std::uint64_t elem = 4) {
    node_rec n{};
    for (int k = 0; k < 3; ++k) {
      n.range[k] = 1;
      n.group_shape[k] = 1;
      n.grid[k] = 1;
    }
    n.grid[0] = groups;
    n.group_shape[0] = gs;
    n.range[0] = body_kind == 0 ? groups * gs : groups;
    n.group_elems = gs;
    n.elem_bytes = elem;
    n.body_kind = body_kind;
    n.link_first = static_cast<std::uint32_t>(links.size());
    nodes.push_back(n);
    return nodes.size() - 1;
  }
  void add_local_link(std::size_t node, std::uint32_t source, std::uint64_t count,
                      std::uint64_t src_gs, std::uint64_t slots_off, std::uint64_t slot_bytes,
                      std::uint64_t child_off) {
    link_rec l{};
    l.kind = 1;
    l.source = source;
    l.count = count;
    l.slot_bytes = slot_bytes;
    l.slots_offset = slots_off;
    l.child_frame_offset = child_off;
    l.src_grid[0] = l.src_grid[1] = l.src_grid[2] = 1;
    l.src_group_shape[0] = src_gs;
    l.src_group_shape[1] = l.src_group_shape[2] = 1;
    l.src_elem_bytes = 4;
    links.push_back(l);
    nodes[node].link_count++;
  }
  void upload() {
    launch_ctx c{};
    c.layers = reinterpret_cast<std::uint64_t>(d + off_layers);
    c.nodes = reinterpret_cast<std::uint64_t>(d + off_nodes);
    c.links = reinterpret_cast<std::uint64_t>(d + off_links);
    c.num_layers = static_cast<std::uint32_t>(layers.size());
    q.memcpy(d + off_ctx, &c, sizeof(c));
    if (!layers.empty())
      q.memcpy(d + off_layers, layers.data(), layers.size() * sizeof(layer_rec));
    if (!nodes.empty())
      q.memcpy(d + off_nodes, nodes.data(), nodes.size() * sizeof(node_rec));
    if (!links.empty())
      q.memcpy(d + off_links, links.data(), links.size() * sizeof(link_rec));
    if (!flags.empty())
      q.memcpy(d + off_flags, flags.data(), flags.size() * 4);
    q.wait();
  }
  const launch_ctx *ctx() const { return reinterpret_cast<const launch_ctx *>(d + off_ctx); }

  template <int Mode> void run(std::size_t native_groups, std::size_t W, std::size_t frame_bytes = 1024) {
    const launch_ctx *c = ctx();
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<std::byte, 1> frame{sycl::range<1>(frame_bytes), h};
      h.parallel_for(sycl::nd_range<1>(native_groups * W, W), [=](sycl::nd_item<1> it) {
        run_native_group<mock<Mode>>(c, it, static_cast<void *>(&frame[0]));
      });
    });
    q.wait_and_throw();
  }

  std::vector<log_entry> log() {
    std::uint32_t n = 0;
    q.memcpy(&n, d + off_ctr, 4).wait();
    n = std::min<std::uint32_t>(n, log_entries);
    std::vector<log_entry> out(n);
    if (n)
      q.memcpy(out.data(), d, n * sizeof(log_entry)).wait();
    return out;
  }
  std::vector<std::uint32_t> read_flags() {
    std::vector<std::uint32_t> f(flags.size());
    if (!f.empty())
      q.memcpy(f.data(), d + off_flags, f.size() * 4).wait();
    return f;
  }
};

inline std::size_t count_tag(const std::vector<log_entry> &l, std::uint32_t t) {
  std::size_t n = 0;
  for (const auto &e : l)
    n += e.tag == t;
  return n;
}
} // namespace sgd
