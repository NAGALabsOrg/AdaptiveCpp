# Measure sg_chain_bench over backends and chain lengths with every persistent cache cold.
#
#   nu naga/tasks/sg-jit-measure.nu <label> [--backends omp,cuda,ocl,ze] [--lengths 10,20,30]
#       [--reps 3] [--timeout 1800]
#
# Each run gets a fresh, empty ACPP_APPDB_DIR, the drivers' on-disk caches switched off and
# ACPP_S2_DUMP_IR_FINAL set; the user's ~/.acpp and the drivers' cache directories are never
# read or written. Results go to ~/logs/acpp-fork/sg-jit-<label>-<timestamp>/results.csv and
# the ratios median cold_ms(30)/cold_ms(10) and ir_lines(30)/ir_lines(10) are printed per backend.

use sg-jit-lib.nu [classify-run, parse-bench-line, ir-metrics, aggregate-reps, chain-ratios, mask-of, cache-env]

const all_backends = [omp cuda ocl ze]

def run-bench [log: path, env_vars: record, seconds: int, n: int] {
  try {
    let r = (do --ignore-errors {
      with-env $env_vars {
        ^timeout $"($seconds)s" sg_chain_bench --n $n o+e>| tee { save --force $log }
      }
    } | complete)
    if $r.exit_code < 0 { 128 - $r.exit_code } else { $r.exit_code }
  } catch {|e|
    if ($e.msg | str contains "core dumped") { 139 } else { 255 }
  }
}

def main [
  label: string
  --backends: string = "omp,cuda,ocl,ze"
  --lengths: string = "10,20,30"
  --reps: int = 3
  --timeout: int = 1800
] {
  let backends = ($backends | split row ',')
  let lengths = ($lengths | split row ',' | each {|l| $l | into int })
  if ($backends | any {|b| $b not-in $all_backends }) {
    print -e "sg-jit-measure: unknown backend"
    exit 1
  }
  if (which sg_chain_bench | is-empty) {
    print -e "sg-jit-measure: sg_chain_bench not found"
    exit 1
  }

  let dir = ($env.HOME | path join logs acpp-fork $"sg-jit-($label)-(date now | format date '%Y%m%d-%H%M%S')")
  mkdir $dir
  let csv = ($dir | path join results.csv)
  print $"sg-jit-measure: results in ($dir)"

  mut rows = []
  for backend in $backends {
    for n in $lengths {
      for rep in 1..$reps {
        let stem = $"($backend)-($n)-($rep)"
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
        let code = (run-bench $log $env_vars $timeout $n)
        let text = (if ($log | path exists) { open --raw $log | ansi strip } else { "" })
        let line = ($text | lines | where {|l| $l starts-with "sg_chain_bench n=" } | last 1 | get -o 0 | default "")
        let status = (classify-run $code $line)
        let parsed = (if $line != "" { try { parse-bench-line $line } catch { null } } else { null })
        let ir = (if ($dump | path exists) { ir-metrics (open --raw $dump) } else { {functions: 0, lines: 0} })
        rm --recursive --force $appdb
        print $"sg-jit-measure: ($stem) exit ($code) ($status)"
        $rows = ($rows | append {
          label: $label
          backend: $backend
          n: $n
          rep: $rep
          status: $status
          cold_ms: (if $parsed == null { null } else { $parsed.cold_ms })
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
  print (chain-ratios $rows | table --expand)
}
