// The first line of every executable's log: which backend, platform and device the
// cell ran on. The runner reads it and fails a cell whose device is not of its backend.
#pragma once

#include <cstdio>
#include <string>

#include <sycl/sycl.hpp>

namespace sgt {

inline std::string no_spaces(std::string s) {
  for (char &c : s)
    if (c == ' ' || c == '\t' || c == '\n')
      c = '_';
  return s.empty() ? std::string("none") : s;
}

inline const char *backend_name(sycl::backend b) {
  switch (b) {
  case sycl::backend::omp: return "omp";
  case sycl::backend::cuda: return "cuda";
  case sycl::backend::ocl: return "ocl";
  case sycl::backend::level_zero: return "ze";
  default: return "other";
  }
}

// `with_device` false: a cell that runs without a device.
inline void print_cell_line(bool with_device) {
  if (!with_device) {
    std::printf("sg-cell backend=host platform=none device=none vendor=none\n");
    std::fflush(stdout);
    return;
  }
  const sycl::device d = sycl::queue{}.get_device();
  std::printf("sg-cell backend=%s platform=%s device=%s vendor=%s\n", backend_name(d.get_backend()),
              no_spaces(d.get_platform().get_info<sycl::info::platform::name>()).c_str(),
              no_spaces(d.get_info<sycl::info::device::name>()).c_str(),
              no_spaces(d.get_info<sycl::info::device::vendor>()).c_str());
  std::fflush(stdout);
}

} // namespace sgt
