use std/assert
use ../doc-consistency.nu [findings, required-ids]

const summary = "# summary

## The language

| primitive | meaning |
|---|---|
| kh.source(array) | an input |
| kh.stencil(offsets, roots) | index tiles |

Rules:
- a rule

## The graph around a kernel (D15-D17)

```cpp
g.run();
```

## Decisions
"

def ids-line [] { required-ids | str join " " }

def clean-01 [] { $"kh.source kh.stencil\n(ids-line)\n" }
def clean-02 [] { "g.run\n" }
def clean-03 [] { "terms\n" }

def test-a-clean-fixture-has-no-findings [] {
  assert equal (findings $summary (clean-01) (clean-02) (clean-03)) []
}

def test-a-primitive-missing-from-01-is-one-finding [] {
  let d01 = ((clean-01) | str replace "kh.stencil" "kh.nothing")
  let f = (findings $summary $d01 (clean-02) (clean-03))
  assert equal ($f | length) 1
  assert equal ($f | first | get name) "kh.stencil"
  assert equal ($f | first | get kind) "missing-name"
}

def test-a-graph-name-missing-from-02-is-one-finding [] {
  let f = (findings $summary (clean-01) "nothing\n" (clean-03))
  assert equal ($f | length) 1
  assert equal ($f | first | get name) "g.run"
  assert equal ($f | first | get where) "02"
}

def test-a-superseded-term-is-one-finding [] {
  let f = (findings $summary (clean-01) (clean-02) $"(clean-03)\nuses parallel_for here\n")
  assert equal ($f | length) 1
  assert equal ($f | first | get name) "parallel_for"
  assert equal ($f | first | get where) "03"
}

def test-every-superseded-term-is-banned [] {
  for t in [parallel_for map_groups group_expr mapping durable_graph promise future "structural hash" "key tuple"] {
    let f = (findings $summary (clean-01) (clean-02) $"(clean-03)\n($t)\n")
    assert equal ($f | length) 1 $"($t) was not found"
  }
}

def test-a-missing-id-is-one-finding [] {
  let d01 = ((clean-01) | str replace " W11 " " ")
  let f = (findings $summary $d01 (clean-02) (clean-03))
  assert equal ($f | length) 1
  assert equal ($f | first | get name) "W11"
  assert equal ($f | first | get kind) "missing-id"
}

def test-an-id-found-in_any_of-the-three-documents-counts [] {
  let d01 = ((clean-01) | str replace " D9 " " ")
  let f = (findings $summary $d01 (clean-02) "D9\n")
  assert equal $f []
}

test-a-clean-fixture-has-no-findings
test-a-primitive-missing-from-01-is-one-finding
test-a-graph-name-missing-from-02-is-one-finding
test-a-superseded-term-is-one-finding
test-every-superseded-term-is-banned
test-a-missing-id-is-one-finding
test-an-id-found-in_any_of-the-three-documents-counts
print "doc-consistency tests: all passed"
