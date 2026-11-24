# Render the table and the 30-over-10 ratios of one sg-jit-measure run.
#
#   nu naga/tasks/sg-jit-report.nu <results.csv> [--out <path>]

use sg-jit-lib.nu [build-report]

const header = "label,backend,n,rep,status,cold_ms,ir_functions,ir_lines,log"

def main [results: path, --out: path] {
  if not ($results | path exists) {
    print -e $"sg-jit-report: ($results) does not exist"
    exit 1
  }
  if (open --raw $results | lines | first) != $header {
    print -e $"sg-jit-report: ($results) lacks the results.csv header"
    exit 1
  }
  let text = (build-report (open $results))
  let target = (if $out == null { $results | path dirname | path join report.md } else { $out })
  $text | save --force $target
  print $text
}
