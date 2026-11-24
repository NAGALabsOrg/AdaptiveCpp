// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_SG_ERROR_HPP
#define HIPSYCL_SG_ERROR_HPP

#include <array>
#include <cstdint>
#include <source_location>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sg {

enum class errc {
  role_conflict,
  size_mismatch,
  type_mismatch,
  never_produced,
  inplace_group_mismatch,
  builder_misuse,
  shape_mismatch,
  allocation_refused,
  local_memory_exceeded,
  work_group_too_large,
  limit_exceeded,
  not_in_submit_scope,
  build,
  internal
};

inline const char *to_string(errc c) noexcept {
  switch (c) {
  case errc::role_conflict: return "role_conflict";
  case errc::size_mismatch: return "size_mismatch";
  case errc::type_mismatch: return "type_mismatch";
  case errc::never_produced: return "never_produced";
  case errc::inplace_group_mismatch: return "inplace_group_mismatch";
  case errc::builder_misuse: return "builder_misuse";
  case errc::shape_mismatch: return "shape_mismatch";
  case errc::allocation_refused: return "allocation_refused";
  case errc::local_memory_exceeded: return "local_memory_exceeded";
  case errc::work_group_too_large: return "work_group_too_large";
  case errc::limit_exceeded: return "limit_exceeded";
  case errc::not_in_submit_scope: return "not_in_submit_scope";
  case errc::build: return "build";
  case errc::internal: return "internal";
  }
  return "internal";
}

class error : public std::runtime_error {
public:
  error(errc code, const std::string &message)
      : std::runtime_error(message), code_{code} {}
  errc code() const noexcept { return code_; }

private:
  errc code_;
};

struct source_loc {
  const char *file = "";
  int line = 0;
  static source_loc from(const std::source_location &l) { return {l.file_name(), static_cast<int>(l.line())}; }
};

struct named_number {
  std::string name;
  std::int64_t value;
};

// One rejection of the language: its number in the catalogue, the construct
// that owns it and an explanation.
class build_error : public error {
public:
  build_error(int rule, const std::string &construct, source_loc where, const std::string &explanation,
              std::vector<named_number> numbers = {})
      : error(errc::build, render(construct, where, explanation, numbers)), rule_(rule),
        construct_(construct), where_(where), numbers_(std::move(numbers)) {}
  int rule() const noexcept { return rule_; }
  const std::string &construct() const noexcept { return construct_; }
  source_loc where() const noexcept { return where_; }
  const std::vector<named_number> &numbers() const noexcept { return numbers_; }

  static std::string render(const std::string &construct, source_loc where,
                            const std::string &explanation,
                            const std::vector<named_number> &numbers) {
    std::string s = construct + " at " + where.file + ":" + std::to_string(where.line) + ": " + explanation;
    if (!numbers.empty()) {
      s += " (";
      for (std::size_t i = 0; i < numbers.size(); ++i) {
        if (i)
          s += ", ";
        s += numbers[i].name + " = " + std::to_string(numbers[i].value);
      }
      s += ")";
    }
    return s;
  }

private:
  int rule_;
  std::string construct_;
  source_loc where_;
  std::vector<named_number> numbers_;
};

struct rule_info {
  int number;
  const char *construct;
  const char *meaning;
};

inline const std::array<rule_info, 15> &rejection_catalogue() {
  static const std::array<rule_info, 15> c = {{
      {1, "kh.work_tiles", "work tile above the device cap"},
      {2, "kh.work_tiles", "native shape above 3-D or a 3-D tile with a run"},
      {3, "kh.index_map", "tile dependency cycle"},
      {4, "kh.mark_local", "plain mark_local on a distributed result"},
      {5, "kh.mark_local<sg::segmented>", "segmented mark_local on a non-distributed result"},
      {6, "unit.run_replicated", "run_replicated on a distributed result of its own set"},
      {7, "kh.map_items", "work tiles and work items in one kernel"},
      {8, "launch_value", "launch value used as structure"},
      {9, "kh.mark_local", "mark_local of array tiles given non-array tiles"},
      {10, "kh.map_items", "work item reading a result array at another index"},
      {11, "kh.map_work", "result arrays fed to work tiles or result tiles fed to work items"},
      {12, "kh.join", "join with grid rank differing from tile rank"},
      {13, "kh.slice_work", "slice_work outside its work set"},
      {14, "unit.run", "run input longer than the run"},
      {15, "sg::interop", "submit<sg::interop>"},
  }};
  return c;
}

} // namespace sg

#endif
