// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s

#include <algorithm>
#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

template <class T> using aref =
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>;

template <class T>
std::vector<T> fetch(sycl::queue &q, const T *device, std::size_t count) {
  std::vector<T> host(count);
  q.memcpy(host.data(), device, count * sizeof(T)).wait_and_throw();
  return host;
}

bool report(const char *name, bool ok) {
  std::cout << name << (ok ? ": PASSED\n" : ": FAILED\n");
  return ok;
}

bool pure_index(sycl::queue &q) {
  int *out = sycl::malloc_device<int>(64, q);
  q.parallel_for(sycl::nd_range<1>{64, 16}, [=](sycl::nd_item<1> item) {
    int p = 4 * static_cast<int>(item.get_local_id(0)) + 7;
    sycl::group_barrier(item.get_group());
    out[item.get_global_id(0)] = p;
  });
  const auto r = fetch(q, out, 64);
  sycl::free(out, q);
  bool ok = true;
  for (int i = 0; i < 64; ++i)
    ok = ok && r[i] == 4 * (i % 16) + 7;
  return report("pure_index", ok);
}

bool global_load(sycl::queue &q) {
  int *in = sycl::malloc_device<int>(64, q);
  int *out = sycl::malloc_device<int>(64, q);
  std::vector<int> init(64);
  for (int i = 0; i < 64; ++i)
    init[i] = 1000 + i;
  q.memcpy(in, init.data(), 64 * sizeof(int));
  q.parallel_for(sycl::nd_range<1>{64, 16}, [=](sycl::nd_item<1> item) {
    const size_t gid = item.get_global_id(0);
    int v = in[gid];
    sycl::group_barrier(item.get_group());
    in[item.get_group(0) * 16 + (item.get_local_id(0) + 1) % 16] = -1;
    out[gid] = v;
  });
  const auto r = fetch(q, out, 64);
  sycl::free(in, q);
  sycl::free(out, q);
  bool ok = true;
  for (int i = 0; i < 64; ++i)
    ok = ok && r[i] == 1000 + i;
  return report("global_load", ok);
}

bool uniform_load(sycl::queue &q) {
  int *p = sycl::malloc_device<int>(1, q);
  int *out = sycl::malloc_device<int>(16, q);
  const int five = 5;
  q.memcpy(p, &five, sizeof(int));
  q.parallel_for(sycl::nd_range<1>{16, 16}, [=](sycl::nd_item<1> item) {
    int v = *p;
    sycl::group_barrier(item.get_group());
    if (item.get_local_id(0) == 0)
      *p = 99;
    out[item.get_local_id(0)] = v;
  });
  const auto r = fetch(q, out, 16);
  const auto pointee = fetch(q, p, 1);
  sycl::free(p, q);
  sycl::free(out, q);
  bool ok = pointee[0] == 99;
  for (int i = 0; i < 16; ++i)
    ok = ok && r[i] == 5;
  return report("uniform_load", ok);
}

bool cmpxchg_counter(sycl::queue &q) {
  int *counter = sycl::malloc_device<int>(1, q);
  int *out = sycl::malloc_device<int>(64, q);
  q.memset(counter, 0, sizeof(int));
  q.parallel_for(sycl::nd_range<1>{64, 16}, [=](sycl::nd_item<1> item) {
    aref<int> ref{*counter};
    int old = ref.load();
    while (!ref.compare_exchange_weak(old, old + 1)) {
    }
    int x = old;
    sycl::group_barrier(item.get_group());
    out[item.get_global_id(0)] = x;
  });
  auto r = fetch(q, out, 64);
  const auto end = fetch(q, counter, 1);
  sycl::free(counter, q);
  sycl::free(out, q);
  std::sort(r.begin(), r.end());
  bool ok = end[0] == 64;
  for (int i = 0; i < 64; ++i)
    ok = ok && r[i] == i;
  return report("cmpxchg_counter", ok);
}

bool fetch_or_bits(sycl::queue &q) {
  unsigned *mask = sycl::malloc_device<unsigned>(1, q);
  unsigned *out = sycl::malloc_device<unsigned>(16, q);
  q.memset(mask, 0, sizeof(unsigned));
  q.parallel_for(sycl::nd_range<1>{16, 16}, [=](sycl::nd_item<1> item) {
    const unsigned bit = 1u << item.get_local_id(0);
    unsigned x = aref<unsigned>{*mask}.fetch_or(bit);
    sycl::group_barrier(item.get_group());
    out[item.get_local_id(0)] = x;
  });
  const auto r = fetch(q, out, 16);
  const auto end = fetch(q, mask, 1);
  sycl::free(mask, q);
  sycl::free(out, q);
  bool ok = end[0] == 0xFFFFu;
  for (unsigned i = 0; i < 16; ++i)
    ok = ok && (r[i] & (1u << i)) == 0;
  return report("fetch_or_bits", ok);
}

int main() {
  sycl::queue q{sycl::property::queue::in_order{}};
  bool ok = true;
  ok &= pure_index(q);
  ok &= global_load(q);
  ok &= uniform_load(q);
  ok &= cmpxchg_counter(q);
  ok &= fetch_or_bits(q);
  return ok ? 0 : 1;
}

// CHECK: pure_index: PASSED
// CHECK-NEXT: global_load: PASSED
// CHECK-NEXT: uniform_load: PASSED
// CHECK-NEXT: cmpxchg_counter: PASSED
// CHECK-NEXT: fetch_or_bits: PASSED
