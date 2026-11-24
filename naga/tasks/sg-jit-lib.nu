# Pure helpers of the sg chain measurement: classify runs, parse bench lines, measure
# IR dumps, aggregate repetitions and compute the 30-over-10 ratios.

const ir_begin = ";---------------- Begin AdaptiveCpp IR dump --------------"
const ir_end = ";----------------- End AdaptiveCpp IR dump ---------------"

export def mask-of [backend: string] {
  match $backend {
    "omp" => "omp"
    "cuda" => "omp;cuda"
    "ocl" => "ocl"
    "ze" => "ze"
  }
}

# The environment that switches a backend's driver cache off; a record for every backend.
export def cache-env [backend: string] {
  match $backend {
    "cuda" => {CUDA_CACHE_DISABLE: "1"}
    "ocl" | "ze" => {NEO_CACHE_PERSISTENT: "0"}
    _ => ({x: 0} | reject x)
  }
}

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

# pass, crash (killed by a signal), timeout (124), fail (the result line says ok=0), error
export def classify-run [exit_code: int, bench_line: string] {
  if $exit_code == 124 {
    "timeout"
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
  if ($toks | first) != "sg_chain_bench" or ($toks | length) != 4 {
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
  {
    n: (do $get "n" | into int)
    ok: (do $get "ok" | into int)
    cold_ms: (do $get "cold_ms" | into float)
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

# One row per (backend, n): the status (the first that is not pass), the median cold time and the
# median line count of the final IR over the passing repetitions.
export def aggregate-reps [rows: table] {
  let keys = ($rows | select backend n | uniq)
  $keys | each {|k|
    let reps = ($rows | where {|r| $r.backend == $k.backend and $r.n == $k.n })
    let nonpass = ($reps | each {|r| $r.status } | where {|s| $s != "pass" })
    let status = (if ($nonpass | is-empty) { "pass" } else { $nonpass | first })
    let passing = ($reps | where {|r| $r.status == "pass" })
    {
      backend: $k.backend
      n: $k.n
      status: $status
      cold_ms: (med ($passing | each {|r| num $r.cold_ms }))
      ir_lines: (med ($passing | each {|r| num $r.ir_lines }))
      ir_functions: (med ($passing | each {|r| num $r.ir_functions }))
    }
  }
}

# For each backend: median cold_ms(30) / median cold_ms(10) and ir_lines(30) / ir_lines(10).
export def chain-ratios [rows: table] {
  let agg = (aggregate-reps $rows)
  $agg | get backend | uniq | each {|b|
    let ten = ($agg | where {|r| $r.backend == $b and $r.n == 10 })
    let thirty = ($agg | where {|r| $r.backend == $b and $r.n == 30 })
    let usable = ($ten | is-not-empty) and ($thirty | is-not-empty) and ($ten | first | get status) == "pass" and ($thirty | first | get status) == "pass"
    {
      backend: $b
      cold_ratio: (if $usable { (($thirty | first | get cold_ms) / ($ten | first | get cold_ms)) } else { null })
      ir_ratio: (if $usable { (($thirty | first | get ir_lines) / ($ten | first | get ir_lines)) } else { null })
    }
  }
}

def fixed [v: any, d: int] {
  if $v == null { "n/a" } else { $v | into float | into string --decimals $d }
}

export def build-report [rows: table] {
  let agg = (aggregate-reps $rows)
  mut out = ["# sg chain measurement" ""]
  for b in ($agg | get backend | uniq) {
    $out = ($out | append [$"## ($b)" "" "| n | status | cold ms | ir lines | ir functions |" "| --- | --- | --- | --- | --- |"])
    for r in ($agg | where {|r| $r.backend == $b } | sort-by n) {
      $out = ($out | append $"| ($r.n) | ($r.status) | (fixed $r.cold_ms 1) | (fixed $r.ir_lines 0) | (fixed $r.ir_functions 0) |")
    }
    $out = ($out | append "")
  }
  $out = ($out | append ["## ratios 30 over 10" "" "| backend | cold ratio | ir lines ratio |" "| --- | --- | --- |"])
  for r in (chain-ratios $rows) {
    $out = ($out | append $"| ($r.backend) | (fixed $r.cold_ratio 2) | (fixed $r.ir_ratio 2) |")
  }
  $out | str join "\n"
}
