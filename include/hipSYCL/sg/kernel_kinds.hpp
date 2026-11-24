// SPDX-License-Identifier: BSD-2-Clause
// The work kinds of graph::submit: a device kernel by default, host work, and
// interop, which is declared and rejected as not designed.
#ifndef HIPSYCL_SG_KERNEL_KINDS_HPP
#define HIPSYCL_SG_KERNEL_KINDS_HPP

namespace sg {

struct host {};
struct interop {};

} // namespace sg

#endif
