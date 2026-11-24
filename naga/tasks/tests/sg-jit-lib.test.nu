use std/assert
use ../sg-jit-lib.nu *

const ok_line = "sg_chain_bench n=30 ok=1 cold_ms=11043.2"
const bad_line = "sg_chain_bench n=30 ok=0 cold_ms=11043.2"
const begin = ";---------------- Begin AdaptiveCpp IR dump --------------"
const end = ";----------------- End AdaptiveCpp IR dump ---------------"

def row [backend: string, n: int, rep: int, status: string, cold: any, lines: int] {
  {
    label: x, backend: $backend, n: $n, rep: $rep, status: $status
    cold_ms: $cold, ir_functions: 2, ir_lines: $lines, log: ""
  }
}

def test-classify-exit-codes [] {
  assert equal (classify-run 139 "") "crash"
  assert equal (classify-run 124 "") "timeout"
  assert equal (classify-run 0 $ok_line) "pass"
  assert equal (classify-run 1 $bad_line) "fail"
  assert equal (classify-run 0 $bad_line) "fail"
  assert equal (classify-run 2 "") "error"
  assert equal (classify-run 0 "") "error"
}

def test-parse-line [] {
  let r = (parse-bench-line $ok_line)
  assert equal $r.n 30
  assert equal $r.ok 1
  assert equal $r.cold_ms 11043.2
  assert error {|| parse-bench-line "garbage" }
}

def test-ir-metrics-picks-largest-section [] {
  let dump = ([
    $begin
    "; x"
    "define void @a() {"
    "}"
    $end
    $begin
    "; header"
    "define void @b() {"
    "  ret void"
    "}"
    "define void @c() {"
    $end
  ] | str join "\n")
  assert equal (ir-metrics $dump) {functions: 2, lines: 5}
  assert equal (ir-metrics "") {functions: 0, lines: 0}
}

def test-cold-ratio-is-the-ratio-of-medians [] {
  let rows = [
    (row ze 10 1 pass 1000.0 2000)
    (row ze 10 2 pass 1100.0 2000)
    (row ze 10 3 pass 900.0 2000)
    (row ze 30 1 pass 3200.0 5000)
    (row ze 30 2 pass 3000.0 5000)
    (row ze 30 3 pass 3100.0 5000)
  ]
  let r = (chain-ratios $rows | first)
  assert equal $r.backend "ze"
  assert ((($r.cold_ratio) - 3.1 | math abs) < 0.000001)
}

def test-ir-ratio-is-the-ratio-of-line-counts [] {
  let rows = [
    (row omp 10 1 pass 1000.0 2000)
    (row omp 30 1 pass 3000.0 5000)
  ]
  let r = (chain-ratios $rows | first)
  assert equal $r.ir_ratio 2.5
}

def test-a-failed-length-gives-no-ratio [] {
  let rows = [
    (row ocl 10 1 pass 1000.0 2000)
    (row ocl 30 1 crash null 0)
  ]
  let r = (chain-ratios $rows | first)
  assert equal $r.cold_ratio null
  assert equal $r.ir_ratio null
}

def test-aggregate-status-is-the-first-that-is-not-pass [] {
  let a = (aggregate-reps [
    (row cuda 10 1 pass 1000.0 100)
    (row cuda 10 2 timeout null 100)
  ])
  assert equal ($a | first | get status) "timeout"
}

def test-report-shape [] {
  let text = (build-report [
    (row ze 10 1 pass 1000.0 2000)
    (row ze 30 1 pass 3000.0 5000)
  ])
  let lines = ($text | lines)
  assert equal ($lines | first) "# sg chain measurement"
  assert ("## ze" in $lines)
  assert ("## ratios 30 over 10" in $lines)
  assert ($lines | any {|l| $l == "| ze | 3.00 | 2.50 |" })
}

def test-cache-env-is-a-record-for-every-backend [] {
  for b in [omp cuda ocl ze] {
    assert equal (cache-env $b | describe | str starts-with "record") true
  }
  assert equal (cache-env omp) ({x: 0} | reject x)
  assert equal (cache-env cuda) {CUDA_CACHE_DISABLE: "1"}
  assert equal (cache-env ocl) {NEO_CACHE_PERSISTENT: "0"}
  assert equal (cache-env ze) {NEO_CACHE_PERSISTENT: "0"}
  assert equal ({A: "1"} | merge (cache-env omp)) {A: "1"}
}

def test-mask-of-each-backend [] {
  assert equal (mask-of omp) "omp"
  assert equal (mask-of cuda) "omp;cuda"
  assert equal (mask-of ocl) "ocl"
  assert equal (mask-of ze) "ze"
}

test-cache-env-is-a-record-for-every-backend
test-mask-of-each-backend
test-classify-exit-codes
test-parse-line
test-ir-metrics-picks-largest-section
test-cold-ratio-is-the-ratio-of-medians
test-ir-ratio-is-the-ratio-of-line-counts
test-a-failed-length-gives-no-ratio
test-aggregate-status-is-the-first-that-is-not-pass
test-report-shape
print "sg-jit-lib tests: all passed"
