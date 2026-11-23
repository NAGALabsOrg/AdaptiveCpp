# Render the before/after comparison of two sg-jit-measure runs.
#
#   nu naga/tasks/sg-jit-report.nu <before.csv> <after.csv> [--out <path>]

use sg-jit-lib.nu [aggregate-reps, build-report]

const header = "label,backend,case,rep,status,cold_ms,warm_ms,big_ms,rejit,ir_functions,ir_lines,log"

def load-results [path: path] {
  if not ($path | path exists) {
    print -e $"sg-jit-report: ($path) does not exist"
    exit 1
  }
  if (open --raw $path | lines | first) != $header {
    print -e $"sg-jit-report: ($path) lacks the results.csv header"
    exit 1
  }
  open $path
}

def main [before: path, after: path, --out: path] {
  let b = (load-results $before)
  let a = (load-results $after)
  let text = (build-report (aggregate-reps $b) (aggregate-reps $a))
  let target = (if $out == null { $after | path dirname | path join report.md } else { $out })
  $text | save --force $target
  print $text
}
