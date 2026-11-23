use std/assert
use ../sg-jit-lib.nu *

const ok_line = "sg_jit_bench case=chain_30 ok=1 cold_ms=11043.2 warm_ms=0.412 big_ms=3.870 rejit=0"
const bad_line = "sg_jit_bench case=chain_30 ok=0 cold_ms=11043.2 warm_ms=0.412 big_ms=3.870 rejit=0"
const begin = ";---------------- Begin AdaptiveCpp IR dump --------------"
const end = ";----------------- End AdaptiveCpp IR dump ---------------"

def test-classify-exit-codes [] {
  assert equal (classify-run 124 "") "infeasible"
  assert equal (classify-run 139 "") "crash"
  assert equal (classify-run 134 "") "crash"
  assert equal (classify-run 0 $ok_line) "pass"
  assert equal (classify-run 1 $bad_line) "fail"
  assert equal (classify-run 0 $bad_line) "fail"
  assert equal (classify-run 2 "") "error"
  assert equal (classify-run 0 "") "error"
}

def test-parse-line [] {
  let r = (parse-bench-line $ok_line)
  assert equal $r.case "chain_30"
  assert equal $r.ok 1
  assert equal $r.cold_ms 11043.2
  assert equal $r.warm_ms 0.412
  assert equal $r.big_ms 3.87
  assert equal $r.rejit 0
  let na = (parse-bench-line "sg_jit_bench case=a_to_f ok=1 cold_ms=5.0 warm_ms=1.000 big_ms=NA rejit=1")
  assert equal $na.big_ms null
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

def test-layer-call-sites-counts [] {
  let dump = ([
    "  call void @__sg_drv_owner.l3(ptr %0) #7"
    "  call void @__sg_drv_compute.l3(ptr %0) #7"
    "  call void @__sg_drv_resolve_layer.l2(ptr %0) #8"
    "  call void @__sg_drv_prep.n0(ptr %0) #7"
    "attributes #7 = { noinline }"
    "attributes #8 = { convergent }"
  ] | str join "\n")
  assert equal (layer-call-sites $dump) {total: 3, noinline: 2}
}

def row [backend: string, case: string, rep: int, status: string, cold: any, warm: any, big: any, lines: int] {
  {
    label: x, backend: $backend, case: $case, rep: $rep, status: $status
    cold_ms: $cold, warm_ms: $warm, big_ms: $big, rejit: 0
    ir_functions: 2, ir_lines: $lines, log: ""
  }
}

def test-aggregate-median-and-status [] {
  let a = (aggregate-reps [
    (row cuda ten_layers 1 pass 1000.0 10.0 null 100)
    (row cuda ten_layers 2 pass 1200.0 20.0 null 100)
    (row cuda ten_layers 3 pass 1100.0 10.0 null 100)
  ])
  assert equal ($a | length) 1
  assert equal ($a | first | get status) "pass"
  assert equal ($a | first | get compile_ms) 1090.0
  assert equal ($a | first | get warm_ms) 10.0
  let b = (aggregate-reps [
    (row cuda ten_layers 1 pass 1000.0 10.0 null 100)
    (row cuda ten_layers 2 crash null null null 100)
  ])
  assert equal ($b | first | get status) "crash"
}

def test-report-shape [] {
  let before = (aggregate-reps [
    (row ze chain_30 1 infeasible null null null 900)
    (row ze ten_layers 1 pass 2000.0 10.0 5.0 100)
  ])
  let after = (aggregate-reps [
    (row ze chain_30 1 pass 3000.0 20.0 8.0 300)
    (row ze ten_layers 1 pass 1000.0 10.0 5.0 100)
  ])
  let text = (build-report $before $after)
  let lines = ($text | lines)
  assert equal ($lines | first) "# sg layer calls: before/after"
  assert ("## ze" in $lines)
  assert ("## verdicts" in $lines)
  assert ($lines | any {|l| $l starts-with "| case | status before | status after |" })
  assert ($lines | any {|l| $l == "chain_30 on ze: infeasible -> pass" })
  assert ($lines | any {|l| $l == "ir growth chain_30/ten_layers on ze: before 9.00 after 3.00" })
  let ten = ($lines | where {|l| $l starts-with "| ten_layers |" } | first)
  assert ($ten | str contains "| 0.50 |")
  let chain = ($lines | where {|l| $l starts-with "| chain_30 |" } | first)
  assert ($chain | str contains "| infeasible | pass | n/a | 2980 | n/a |")
  assert ($lines | any {|l| $l == "rejit flagged rows: 0" })
}

test-classify-exit-codes
test-parse-line
test-ir-metrics-picks-largest-section
test-layer-call-sites-counts
test-aggregate-median-and-status
test-report-shape
print "sg-jit-lib tests: all passed"
