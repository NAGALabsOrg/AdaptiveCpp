use std/assert
use ../sg-conformance.nu [findings, collect-commits]

const manifest = "# m

| unit | files | interfaces |
|---|---|---|
| C5 | a/five.hpp, a/five.cpp | thing@a/five.hpp |
| C9 | a/nine.hpp | none |

## Deviations

- no entries
"

def reader [] { {|p| if $p == "a/five.hpp" { "struct thing {};" } else { "" } } }

def test-a-clean-fixture-has-no-findings [] {
  let c = [{sha: "a1", subject: "C5: x", files: ["a/five.hpp"]}]
  assert equal (findings $manifest ["a/five.hpp" "a/nine.hpp"] $c (reader)) []
}

def test-an-unlisted-new-file-is-one-finding [] {
  let f = (findings $manifest ["a/five.hpp" "a/other.hpp"] [] (reader))
  assert equal ($f | length) 1
  assert equal ($f | first | get name) "a/other.hpp"
  assert equal ($f | first | get kind) "unlisted-file"
}

def test-a-commit-touching-another-units-file-is-one-finding [] {
  let c = [{sha: "b2", subject: "C5: x", files: ["a/nine.hpp"]}]
  let f = (findings $manifest [] $c (reader))
  assert equal ($f | length) 1
  assert ($f | first | get name | str contains "C5")
  assert ($f | first | get name | str contains "a/nine.hpp")
}

def test-a-subject_without-a-unit-id-is-one-finding [] {
  let c = [{sha: "c3", subject: "fix things", files: []}]
  assert equal (findings $manifest [] $c (reader) | length) 1
}

def test-a-file-in-two-rows-is-one-finding [] {
  let m = ($manifest | str replace "a/nine.hpp" "a/five.cpp")
  let f = (findings $m [] [] (reader))
  assert equal ($f | length) 1
  assert equal ($f | first | get kind) "duplicate-file"
}

def test-a-missing-interface-needs-a-deviation [] {
  let gone = {|p| "" }
  let f = (findings $manifest [] [] $gone)
  assert equal ($f | length) 1
  assert equal ($f | first | get kind) "missing-interface"
  let excused = ($manifest | str replace "- no entries" "- thing is declared elsewhere: moved")
  assert equal (findings $excused [] [] $gone) []
}

def test-collect-commits-reads-a-real-repository [] {
  let dir = ($nu.temp-dir | path join $"sg-conf-(random uuid)")
  mkdir $dir
  git -C $dir init --quiet
  git -C $dir config user.email t@example.invalid
  git -C $dir config user.name t
  "x" | save ($dir | path join one.txt)
  git -C $dir add one.txt
  git -C $dir commit --quiet -m "C1: base"
  let base = (git -C $dir rev-parse HEAD | str trim)
  "y" | save ($dir | path join two.txt)
  git -C $dir add two.txt
  git -C $dir commit --quiet -m "C9: two"
  let c = (collect-commits $dir $base)
  rm --recursive --force $dir
  assert equal ($c | length) 1
  assert equal ($c | first | get subject) "C9: two"
  assert equal ($c | first | get files) ["two.txt"]
}

test-a-clean-fixture-has-no-findings
test-an-unlisted-new-file-is-one-finding
test-a-commit-touching-another-units-file-is-one-finding
test-a-subject_without-a-unit-id-is-one-finding
test-a-file-in-two-rows-is-one-finding
test-a-missing-interface-needs-a-deviation
test-collect-commits-reads-a-real-repository
print "sg-conformance tests: all passed"
