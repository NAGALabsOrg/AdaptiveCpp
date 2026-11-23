#include "bench_stats.hpp"

#include "hipSYCL/sg/sg.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using G1 = sg::group_iterator<int, 1, sg::mapping::global>;
using L1 = sg::group_iterator<int, 1, sg::mapping::local>;
using I1 = sg::group_iterator<int, 1, sg::mapping::item>;
using clock_type = std::chrono::steady_clock;

int chain_layers = 30;

sycl::device device() { return sycl::device{sycl::default_selector_v}; }

sg::graph_options options() {
  sg::graph_options o;
  o.pool_reservation_bytes = std::size_t(256) << 20;
  return o;
}

void submit_read(sg::graph &g, sg::mem_handle<int> h, std::size_t n, std::shared_ptr<std::vector<int>> out) {
  g.submit(device(), [&](sg::submit_handle gh) {
    sg::host_array<const int, 1> a(h, sg::shape<1>(sycl::range<1>(n)));
    gh.define_host_work([a, out, n] {
      for (std::size_t i = 0; i < n; ++i)
        (*out)[i] = a.data()[i];
    });
  });
}

struct runner {
  sg::graph g{options()};
  std::function<void()> submit;
  std::function<bool()> check;
  bool ok = true;

  double run() {
    submit();
    const auto t0 = clock_type::now();
    g.run();
    g.wait();
    const double ms = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
    ok = check() && ok;
    return ms;
  }
};

template <class Build>
void single_result(runner &r, std::size_t groups, std::size_t key, Build build, int (*expected)(std::size_t)) {
  const std::size_t n = 4 * groups;
  auto h = r.g.create_mem_handle<int>();
  auto out = std::make_shared<std::vector<int>>(n);
  r.submit = [&r, h, n, key, build, out] {
    r.g.submit(device(), [&](sg::submit_handle gh) {
      sg::array<sg::result<int>, 1> o(h, sg::shape<1>(sycl::range<1>(n)));
      gh.define_kernel(std::make_tuple(key), [o, build, n](sg::kernel_handle &kh) { o = kh.evaluate(build(kh, n)); });
    });
    submit_read(r.g, h, n, out);
  };
  r.check = [out, n, expected] {
    for (std::size_t i = 0; i < n; ++i)
      if ((*out)[i] != expected(i))
        return false;
    return true;
  };
}

inline auto leaf(sg::kernel_handle &kh, std::size_t n) {
  return kh.parallel_for(sycl::range<1>(n), sycl::range<1>(4), [](sg::item<1> it) { return int(it.id()[0]); });
}

std::unique_ptr<runner> make_runner(const std::string &name, std::size_t groups) {
  auto r = std::make_unique<runner>();
  if (name == "plain_item") {
    single_result(*r, groups, groups,
                  [](sg::kernel_handle &kh, std::size_t n) {
                    return kh.parallel_for(sycl::range<1>(n), sycl::range<1>(4),
                                           [](sg::item<1> it) { return 3 * int(it.id()[0]) + 1; });
                  },
                  sgt::bench::expected_plain_item);
  } else if (name == "ten_layers") {
    single_result(*r, groups, groups,
                  [](sg::kernel_handle &kh, std::size_t n) {
                    auto d = leaf(kh, n);
                    for (int k = 0; k < 10; ++k) {
                      auto m = kh.map_groups<sg::mapping::global>(d, [](sycl::id<1> g) { return sg::single<1>{g}; });
                      d = kh.parallel_for(sycl::range<1>(n), sycl::range<1>(4), m, [](sg::item<1> it, G1 a) {
                        return a[0][sycl::id<1>(it.local_id()[0])] * 2 % 1000;
                      });
                    }
                    return d;
                  },
                  sgt::bench::expected_ten_layers);
  } else if (name == "chain_30") {
    single_result(*r, groups, groups,
                  [](sg::kernel_handle &kh, std::size_t n) {
                    auto d = leaf(kh, n);
                    for (int k = 0; k < chain_layers; ++k) {
                      auto m = kh.map_groups<sg::mapping::global>(d, [](sycl::id<1> g) { return sg::single<1>{g}; });
                      d = kh.parallel_for(sycl::range<1>(n), sycl::range<1>(4), m, [](sg::item<1> it, G1 a) {
                        return a[0][sycl::id<1>(it.local_id()[0])] + 1;
                      });
                    }
                    return d;
                  },
                  [](std::size_t i) { return static_cast<int>(i) + chain_layers; });
  } else if (name == "layers_8_16_32") {
    single_result(*r, 32, 0,
                  [](sg::kernel_handle &kh, std::size_t) {
                    auto l1 = leaf(kh, 32);
                    auto m1 = kh.map_groups<sg::mapping::global>(
                        l1, [](sycl::id<1>) { return sg::block<1>{sycl::id<1>(0), sycl::range<1>(8)}; });
                    auto l2 = kh.parallel_for(sycl::range<1>(64), sycl::range<1>(4), m1, [](sg::item<1> it, G1 a) {
                      const sycl::id<1> l(it.local_id()[0]);
                      return a[0][l] + a[a.size() - 1][l];
                    });
                    auto m2 = kh.map_groups<sg::mapping::global>(
                        l2, [](sycl::id<1>) { return sg::block<1>{sycl::id<1>(0), sycl::range<1>(16)}; });
                    return kh.parallel_for(sycl::range<1>(128), sycl::range<1>(4), m2, [](sg::item<1> it, G1 a) {
                      const sycl::id<1> l(it.local_id()[0]);
                      return a[0][l] + a[a.size() - 1][l];
                    });
                  },
                  sgt::bench::expected_layers_8_16_32);
  } else if (name == "a_to_f") {
    auto hb = r->g.create_mem_handle<int>(), hc = r->g.create_mem_handle<int>(), hd = r->g.create_mem_handle<int>(),
         hf = r->g.create_mem_handle<int>();
    auto ob = std::make_shared<std::vector<int>>(16), oc = std::make_shared<std::vector<int>>(16),
         od = std::make_shared<std::vector<int>>(16), of = std::make_shared<std::vector<int>>(16);
    runner *rp = r.get();
    r->submit = [rp, hb, hc, hd, hf, ob, oc, od, of] {
      rp->g.submit(device(), [&](sg::submit_handle gh) {
        const sycl::range<1> shape(16);
        sg::array<sg::result<int>, 1> sb(hb, sg::shape<1>(shape)), sc(hc, sg::shape<1>(shape)),
            sd(hd, sg::shape<1>(shape)), sf(hf, sg::shape<1>(shape));
        gh.define_kernel(std::make_tuple(0), [sb, sc, sd, sf](sg::kernel_handle &kh) {
          const sycl::range<1> r(16), gs(4);
          auto A = leaf(kh, 16);
          auto mA = kh.map_groups<sg::mapping::local>(A, [](sycl::id<1> g) { return sg::single<1>{g}; });
          auto B = kh.parallel_for(r, gs, mA, [](sg::item<1> it, L1 a) { return a[0][sycl::id<1>(it.local_id()[0])] + 1; });
          auto C = kh.parallel_for(r, gs, mA, [](sg::item<1> it, L1 a) { return a[0][sycl::id<1>(it.local_id()[0])] * 2; });
          auto mB = kh.map_groups<sg::mapping::local>(B, [](sycl::id<1> g) { return sg::single<1>{g}; });
          auto mC = kh.map_groups<sg::mapping::local>(C, [](sycl::id<1> g) { return sg::single<1>{g}; });
          auto D = kh.parallel_for(r, gs, mB, mC, [](sg::item<1> it, L1 b, L1 c) {
            const sycl::id<1> l(it.local_id()[0]);
            return b[0][l] + c[0][l];
          });
          auto mE = kh.map_groups<sg::mapping::global>(
              C, [](sycl::id<1> g) { return sg::single<1>{sycl::id<1>(3 - g[0])}; });
          auto E = kh.parallel_for(r, gs, mE, [](sg::item<1> it, G1 c) { return c[0][sycl::id<1>(it.local_id()[0])] + 1000; });
          auto mF = kh.map_groups<sg::mapping::item>(E, [](sycl::id<1> g) { return sg::single<1>{g}; });
          auto F = kh.parallel_for(r, gs, mF, [](sg::item<1>, I1 e) { return e[0] + 1; });
          sb = kh.evaluate(B);
          sc = kh.evaluate(C);
          sd = kh.evaluate(D);
          sf = kh.evaluate(F);
        });
      });
      submit_read(rp->g, hb, 16, ob);
      submit_read(rp->g, hc, 16, oc);
      submit_read(rp->g, hd, 16, od);
      submit_read(rp->g, hf, 16, of);
    };
    r->check = [ob, oc, od, of] {
      for (std::size_t i = 0; i < 16; ++i) {
        const auto e = sgt::bench::expected_a_to_f(i);
        if ((*ob)[i] != e.b || (*oc)[i] != e.c || (*od)[i] != e.d || (*of)[i] != e.f)
          return false;
      }
      return true;
    };
  }
  return r;
}

bool has_big_size(const std::string &name) {
  return name == "plain_item" || name == "ten_layers" || name == "chain_30";
}

} // namespace

int main(int argc, char **argv) {
  using namespace sgt::bench;
  const auto parsed = parse_args(std::vector<std::string>(argv + 1, argv + argc));
  if (!parsed) {
    std::cerr << "usage: sg_jit_bench --case <" << "plain_item|ten_layers|layers_8_16_32|chain_30|a_to_f"
              << "> [--warm-runs N] [--warmup-runs N] [--big-groups N]\n";
    return 2;
  }
  const args &a = *parsed;
  chain_layers = a.layers;
  bool ok = true;
  result res{a.case_name, true, 0.0, 0.0, std::nullopt, false};
  try {
    auto small = make_runner(a.case_name, 4);
    res.cold_ms = small->run();
    std::vector<double> warmup, warm;
    for (int i = 0; i < a.warmup_runs; ++i)
      warmup.push_back(small->run());
    for (int i = 0; i < a.warm_runs; ++i)
      warm.push_back(small->run());
    res.warm_ms = median(warm);
    res.rejit = rejit_detected(warmup, res.warm_ms);
    ok = small->ok;
    if (has_big_size(a.case_name)) {
      auto big = make_runner(a.case_name, a.big_groups);
      big->run();
      std::vector<double> times;
      for (int i = 0; i < a.warm_runs; ++i)
        times.push_back(big->run());
      res.big_ms = median(times);
      ok = ok && big->ok;
    }
  } catch (const std::exception &e) {
    std::cerr << "sg_jit_bench: " << e.what() << '\n';
    ok = false;
  }
  res.ok = ok;
  std::cout << format_result_line(res) << std::endl;
  return ok ? 0 : 1;
}
