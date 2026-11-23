# Pure helpers of the sg layer-call measurement: classify runs, parse bench
# lines, measure IR dumps, aggregate repetitions and render the report.

const ir_begin = ";---------------- Begin AdaptiveCpp IR dump --------------"
const ir_end = ";----------------- End AdaptiveCpp IR dump ---------------"

def num [v: any] {
  if $v == null { return null }
  if ($v | describe) == "string" {
    let s = ($v | str trim)
    if ($s | is-empty) { return null }
    return ($s | into float)
  }
  $v | into float
}

def med [xs: list] {
  if ($xs | is-empty) { return null }
  let s = ($xs | sort)
  let n = ($s | length)
  if ($n mod 2) == 1 {
    $s | get (($n - 1) // 2)
  } else {
    (($s | get ($n // 2 - 1)) + ($s | get ($n // 2))) / 2
  }
}

export def classify-run [exit_code: int, bench_line: string] {
  if $exit_code == 124 {
    "infeasible"
  } else if $exit_code >= 128 {
    "crash"
  } else if $exit_code == 0 and ($bench_line | str contains "ok=1") {
    "pass"
  } else if $exit_code == 1 or ($exit_code == 0 and ($bench_line | str contains "ok=0")) {
    "fail"
  } else {
    "error"
  }
}

export def parse-bench-line [line: string] {
  let toks = ($line | str trim | split row ' ')
  if ($toks | first) != "sg_jit_bench" or ($toks | length) != 7 {
    error make {msg: $"malformed bench line: ($line)"}
  }
  let kv = ($toks | skip 1 | each {|t|
    let p = ($t | split row '=')
    if ($p | length) != 2 { error make {msg: $"malformed bench line: ($line)"} }
    {k: ($p | get 0), v: ($p | get 1)}
  })
  let get = {|k|
    let m = ($kv | where {|e| $e.k == $k })
    if ($m | is-empty) { error make {msg: $"malformed bench line: ($line)"} }
    $m | first | get v
  }
  let big = (do $get "big_ms")
  {
    case: (do $get "case")
    ok: (do $get "ok" | into int)
    cold_ms: (do $get "cold_ms" | into float)
    warm_ms: (do $get "warm_ms" | into float)
    big_ms: (if $big == "NA" { null } else { $big | into float })
    rejit: (do $get "rejit" | into int)
  }
}

export def ir-metrics [dump: string] {
  let ls = ($dump | lines)
  let n = ($ls | length)
  let begins = ($ls | enumerate | where {|e| $e.item == $ir_begin } | each {|e| $e.index })
  let ends = ($ls | enumerate | where {|e| $e.item == $ir_end } | each {|e| $e.index })
  mut best_lines = 0
  mut best_start = -1
  mut best_end = -1
  for b in $begins {
    let later = ($ends | where {|e| $e > $b })
    let e = (if ($later | is-empty) { $n } else { $later | first })
    let count = $e - $b - 1
    if $count > $best_lines {
      $best_lines = $count
      $best_start = $b + 1
      $best_end = $e
    }
  }
  if $best_lines == 0 {
    return {functions: 0, lines: 0}
  }
  let defs = ($ls | slice $best_start..<$best_end | where {|l| $l | str starts-with "define " } | length)
  {functions: $defs, lines: $best_lines}
}

export def layer-call-sites [dump: string] {
  let ls = ($dump | lines)
  let groups = ($ls
    | parse --regex '^attributes #(?<k>\d+) = \{(?<body>.*)\}'
    | where {|r| $r.body =~ '\bnoinline\b' }
    | each {|r| $r.k })
  let calls = ($ls | where {|l| $l =~ 'call ' and $l =~ '@__sg_drv_(owner|compute|resolve_layer)\.l\d+\(' })
  let marked = ($calls | where {|l|
    let m = ($l | parse --regex ' #(?<k>\d+)\s*$')
    ($m | is-not-empty) and (($m | first | get k) in $groups)
  })
  {total: ($calls | length), noinline: ($marked | length)}
}

export def aggregate-reps [rows: table] {
  let keys = ($rows | select backend case | uniq)
  $keys | each {|k|
    let reps = ($rows | where {|r| $r.backend == $k.backend and $r.case == $k.case })
    let nonpass = ($reps | each {|r| $r.status } | where {|s| $s != "pass" })
    let status = (if ($nonpass | is-empty) { "pass" } else { $nonpass | first })
    let passing = ($reps | where {|r| $r.status == "pass" })
    let compile = ($passing | each {|r|
      let d = (num $r.cold_ms) - (num $r.warm_ms)
      if $d < 0 { 0.0 } else { $d }
    })
    let warms = ($passing | each {|r| num $r.warm_ms })
    let bigs = ($passing | each {|r| num ($r | get -o big_ms) } | where {|x| $x != null })
    let first_rep = ($reps | first)
    {
      backend: $k.backend
      case: $k.case
      status: $status
      compile_ms: (med $compile)
      warm_ms: (med $warms)
      big_ms: (med $bigs)
      ir_functions: (num ($first_rep | get -o ir_functions))
      ir_lines: (num ($first_rep | get -o ir_lines))
      rejit: ($reps | each {|r| (num ($r | get -o rejit)) | default 0 } | math max)
    }
  }
}

def find-row [t: table, b: string, c: string] {
  let m = ($t | where {|r| $r.backend == $b and $r.case == $c })
  if ($m | is-empty) { null } else { $m | first }
}

def status-of [row: any] {
  if $row == null { "n/a" } else { $row.status }
}

def pass-val [row: any, col: string] {
  if $row == null or $row.status != "pass" { return null }
  $row | get -o $col
}

def fixed [v: any, d: int] {
  if $v == null { "n/a" } else { $v | into float | into string --decimals $d }
}

def whole [v: any] {
  if $v == null { "n/a" } else { $v | into float | math round | into int | into string }
}

def ratio [before: any, after: any] {
  if $before == null or $after == null or $before == 0 { "n/a" } else { fixed ($after / $before) 2 }
}

def count-cell [row: any, col: string] {
  if $row == null { return "n/a" }
  let v = ($row | get -o $col)
  if $v == null or $v == 0 { "n/a" } else { $v | into float | math round | into int | into string }
}

def ir-ratio [t: table, b: string] {
  let c = (find-row $t $b "chain_30")
  let ten = (find-row $t $b "ten_layers")
  if $c == null or $ten == null { return "n/a" }
  let cl = ($c | get -o ir_lines)
  let tl = ($ten | get -o ir_lines)
  if $cl == null or $tl == null or $cl == 0 or $tl == 0 { "n/a" } else { fixed ($cl / $tl) 2 }
}

export def build-report [before: table, after: table] {
  let case_order = [plain_item ten_layers layers_8_16_32 chain_30 a_to_f]
  let backends = ([omp cuda ocl ze] | where {|b|
    ($before | any {|r| $r.backend == $b }) or ($after | any {|r| $r.backend == $b })
  })
  mut out = ["# sg layer calls: before/after" ""]
  for b in $backends {
    $out = ($out | append [$"## ($b)" ""])
    $out = ($out | append "| case | status before | status after | compile ms before | compile ms after | compile ratio | kernel ms before | kernel ms after | kernel ratio | big kernel ms before | big kernel ms after | big kernel ratio | ir lines before | ir lines after | ir functions before | ir functions after |")
    $out = ($out | append ("|" + (1..16 | each {|_| " --- |" } | str join "")))
    for c in $case_order {
      let rb = (find-row $before $b $c)
      let ra = (find-row $after $b $c)
      if $rb == null and $ra == null { continue }
      let cb = (pass-val $rb compile_ms)
      let ca = (pass-val $ra compile_ms)
      let kb = (pass-val $rb warm_ms)
      let ka = (pass-val $ra warm_ms)
      let bb = (pass-val $rb big_ms)
      let ba = (pass-val $ra big_ms)
      let cells = [
        $c (status-of $rb) (status-of $ra)
        (whole $cb) (whole $ca) (ratio $cb $ca)
        (fixed $kb 3) (fixed $ka 3) (ratio $kb $ka)
        (fixed $bb 3) (fixed $ba 3) (ratio $bb $ba)
        (count-cell $rb ir_lines) (count-cell $ra ir_lines)
        (count-cell $rb ir_functions) (count-cell $ra ir_functions)
      ]
      $out = ($out | append ("| " + ($cells | str join " | ") + " |"))
    }
    $out = ($out | append "")
  }
  $out = ($out | append ["## verdicts"])
  let ze_8 = (find-row $before ze layers_8_16_32)
  let ze_8a = (find-row $after ze layers_8_16_32)
  if $ze_8 != null or $ze_8a != null {
    $out = ($out | append $"layers_8_16_32 on ze: (status-of $ze_8) -> (status-of $ze_8a)")
  }
  for b in [cuda ocl ze] {
    let x = (find-row $before $b chain_30)
    let y = (find-row $after $b chain_30)
    if $x != null or $y != null {
      $out = ($out | append $"chain_30 on ($b): (status-of $x) -> (status-of $y)")
    }
  }
  for b in [cuda ocl ze] {
    let present = ([chain_30 ten_layers] | any {|c| (find-row $before $b $c) != null or (find-row $after $b $c) != null })
    if $present {
      $out = ($out | append $"ir growth chain_30/ten_layers on ($b): before (ir-ratio $before $b) after (ir-ratio $after $b)")
    }
  }
  let flagged = (($before | append $after) | where {|r| (num ($r | get -o rejit) | default 0) > 0 } | length)
  $out = ($out | append $"rejit flagged rows: ($flagged)")
  $out | str join "\n"
}
