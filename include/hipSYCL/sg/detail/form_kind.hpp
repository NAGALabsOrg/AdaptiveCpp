// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_DETAIL_FORM_KIND_HPP
#define HIPSYCL_SG_DETAIL_FORM_KIND_HPP

#include <cstdint>

namespace sg {
enum class form_kind : std::uint32_t { single, block, strided, stencil, ids, index_map };
}

#endif
