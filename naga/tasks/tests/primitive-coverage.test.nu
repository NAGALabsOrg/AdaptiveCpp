use std/assert
use ../primitive-coverage.nu [absent-names, wanted-names]

const root = (path self | path dirname | path dirname | path dirname | path dirname)

def summary [] { open --raw ($root | path join docs naga-internal brainstorm language-pass summary.md) }
def program [] { open --raw ($root | path join naga sg-tests compile all_primitives.cpp) }

def test-the-program-uses-every-name [] {
  assert equal (absent-names (program) (summary)) []
}

def test-the-names-of-the-tables-are-listed [] {
  let names = (wanted-names (summary))
  for n in [
    kh.source kh.tile .count kh.retile kh.join kh.extend_tiles kh.define_tiles kh.stencil kh.index_map
    kh.work_tiles sg::segment_items kh.slice_work kh.map_work kh.work_items kh.map_items item.id item.compute
    kh.mark_local kh.mark_local<sg::segmented> unit.run_distributed unit.run_replicated kh.repeat kh.evaluate
    unit.tile unit.id unit.run sg::flatten sg::slice sg::reduce sg::inclusive_scan sg::tile_bin sg::repeat sg::fill
    sg::graph g.create_mem_handle g.submit sg::host sg::interop kh.launch_value sg::shape sg::inplace g.run
  ] {
    assert ($n in $names) $"($n) is not among the wanted names"
  }
}

def test-a-program-without-the-stencil-reports-exactly-it [] {
  let without = (program | str replace --all "kh.stencil" "kh.stencils_are_gone")
  assert equal (absent-names $without (summary)) ["kh.stencil"]
}

def test-the-program-compiles [] {
  let out = ($nu.temp-dir | path join sg-all-primitives.o)
  let src = ($root | path join naga sg-tests compile all_primitives.cpp)
  ^acpp --acpp-targets=generic -std=c++20 -c -o $out $src
  assert ($out | path exists)
  rm --force $out
}

test-the-program-uses-every-name
test-the-names-of-the-tables-are-listed
test-a-program-without-the-stencil-reports-exactly-it
test-the-program-compiles
print "primitive-coverage tests: all passed"
