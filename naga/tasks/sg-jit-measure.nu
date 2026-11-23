# Measure sg_jit_bench over backends and cases with every persistent cache cold.
#
#   nu naga/tasks/sg-jit-measure.nu <label> [--backends cuda,ocl,ze]
#       [--cases plain_item,ten_layers,layers_8_16_32,chain_30,a_to_f] [--reps 3] [--timeout 1200]
#
# Each run gets a fresh, empty ACPP_APPDB_DIR and the drivers' on-disk caches switched
# off; the user's ~/.acpp and the drivers' cache directories are never read or written.
# Results go to ~/logs/acpp-fork/sg-jit-<label>-<timestamp>/results.csv.

use sg-jit-lib.nu [classify-run, parse-bench-line, ir-metrics, aggregate-reps]

const all_backends = [omp cuda ocl ze]
const all_cases = [plain_item ten_layers layers_8_16_32 chain_30 a_to_f]

def mask-of [backend: string] {
  match $backend {
    "omp" => "omp"
    "cuda" => "omp;cuda"
    "ocl" => "ocl"
    "ze" => "ze"
  }
}

def cache-env [backend: string] {
  match $backend {
    "cuda" => {CUDA_CACHE_DISABLE: "1"}
    "ocl" | "ze" => {NEO_CACHE_PERSISTENT: "0"}
    _ => {}
  }
}

def run-bench [log: path, env_vars: record, seconds: int, case: string] {
  try {
    let r = (do --ignore-errors {
      with-env $env_vars {
        ^timeout $"($seconds)s" sg_jit_bench --case $case o+e>| tee { save --force $log }
      }
    } | complete)
    if $r.exit_code < 0 { 128 - $r.exit_code } else { $r.exit_code }
  } catch {|e|
    if ($e.msg | str contains "core dumped") { 139 } else { 255 }
  }
}

def main [
  label: string
  --backends: string = "cuda,ocl,ze"
  --cases: string = "plain_item,ten_layers,layers_8_16_32,chain_30,a_to_f"
  --reps: int = 3
  --timeout: int = 1200
] {
  let backends = ($backends | split row ',')
  let cases = ($cases | split row ',')
  if ($backends | any {|b| $b not-in $all_backends }) or ($cases | any {|c| $c not-in $all_cases }) {
    print -e "sg-jit-measure: unknown backend or case"
    exit 1
  }
  if (which sg_jit_bench | is-empty) {
    print -e "sg-jit-measure: sg_jit_bench not found"
    exit 1
  }

  let dir = ($env.HOME | path join logs acpp-fork $"sg-jit-($label)-(date now | format date '%Y%m%d-%H%M%S')")
  mkdir $dir
  let csv = ($dir | path join results.csv)
  print $"sg-jit-measure: results in ($dir)"

  mut rows = []
  for backend in $backends {
    for case in $cases {
      for rep in 1..$reps {
        let stem = $"($backend)-($case)-($rep)"
        let log = ($dir | path join $"($stem).log")
        let dump = ($dir | path join $"($stem).final.ll")
        let appdb = ($dir | path join $"appdb-($stem)")
        mkdir $appdb
        let env_vars = ({
          ACPP_APPDB_DIR: $appdb
          ACPP_VISIBILITY_MASK: (mask-of $backend)
          ACPP_S2_DUMP_IR_FINAL: $dump
        } | merge (cache-env $backend))
        print $"sg-jit-measure: ($stem) start (date now | format date '%H:%M:%S')"
        let code = (run-bench $log $env_vars $timeout $case)
        let text = (if ($log | path exists) { open --raw $log | ansi strip } else { "" })
        let line = ($text | lines | where {|l| $l starts-with "sg_jit_bench case=" } | last 1 | get -o 0 | default "")
        let status = (classify-run $code $line)
        let parsed = (if $line != "" { try { parse-bench-line $line } catch { null } } else { null })
        let ir = (if ($dump | path exists) { ir-metrics (open --raw $dump) } else { {functions: 0, lines: 0} })
        rm --recursive --force $appdb
        print $"sg-jit-measure: ($stem) exit ($code) ($status)"
        $rows = ($rows | append {
          label: $label
          backend: $backend
          case: $case
          rep: $rep
          status: $status
          cold_ms: (if $parsed == null { null } else { $parsed.cold_ms })
          warm_ms: (if $parsed == null { null } else { $parsed.warm_ms })
          big_ms: (if $parsed == null { null } else { $parsed.big_ms })
          rejit: (if $parsed == null { null } else { $parsed.rejit })
          ir_functions: $ir.functions
          ir_lines: $ir.lines
          log: $log
        })
        $rows | to csv | save --force $csv
        if $status != "pass" { break }
      }
    }
  }

  print ""
  print (aggregate-reps $rows | table --expand)
}
