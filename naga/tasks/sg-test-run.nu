# Run the sg graph toolchain test cells (package naga/sg-tests, env sg-tests).
#
#   nu naga/tasks/sg-test-run.nu <host|omp|cuda|ocl|ze|lit|all> ...
#
#   host -> sg_host_tests, sg_compiler_tests
#   omp  -> sg_driver_tests, sg_tests   (ACPP_VISIBILITY_MASK=omp)
#   cuda -> sg_driver_tests, sg_tests   (omp;cuda)
#   ocl  -> sg_driver_tests, sg_tests   (ocl)
#   ze   -> sg_driver_tests, sg_tests   (ze)
#   lit  -> lit over share/naga-sg-tests/lit
# Every cell runs even if an earlier one failed; each command runs under `timeout 1800s` (exit 124 is reported as "timeout"). Each
# command's output is teed to its own log under ~/logs/acpp-fork/sg-tests-<timestamp>/
# so the final table can name it. The exit code is non-zero if any cell failed.

def cells [] {
  [
    {group: host, mask: "omp", exes: [sg_host_tests sg_compiler_tests]}
    {group: omp, mask: "omp", exes: [sg_driver_tests sg_tests]}
    {group: cuda, mask: "omp;cuda", exes: [sg_driver_tests sg_tests]}
    {group: ocl, mask: "ocl", exes: [sg_driver_tests sg_tests]}
    {group: ze, mask: "ze", exes: [sg_driver_tests sg_tests]}
  ]
}

def run-logged [log: path, mask: string, bin: string, ...args: string] {
  try {
    let r = (do --ignore-errors {
      with-env {ACPP_VISIBILITY_MASK: $mask} {
        ^timeout 1800s $bin ...$args o+e>| tee { save --force $log }
      }
    } | complete)
    $r.exit_code
  } catch {|e|
    if ($e.msg | str contains "core dumped") { 139 } else { 255 }
  }
}

def boost-summary [log: path] {
  let found = (open --raw $log | ansi strip | lines | where {|l| $l starts-with '*** ' })
  if ($found | is-empty) { "-" } else { $found | last }
}

def main [...groups: string] {
  let want = (if ($groups | is-empty) or ($groups | any {|g| $g == "all" }) {
    [host omp cuda ocl ze lit]
  } else { $groups })

  let dir = ($env.HOME | path join logs acpp-fork $"sg-tests-(date now | format date '%Y%m%d-%H%M%S')")
  mkdir $dir
  print $"sg-test-run: logs in ($dir)"

  mut rows = []
  for c in (cells) {
    if $c.group not-in $want { continue }
    for exe in $c.exes {
      let log = ($dir | path join $"($c.group)-($exe).log")
      print $"sg-test-run: [($c.group)] ($exe) start (date now | format date '%H:%M:%S')"
      let code = (run-logged $log $c.mask $exe "--log_level=test_suite")
      print $"sg-test-run: [($c.group)] ($exe) exit ($code)"
      $rows = ($rows | append {cell: $c.group, exe: $exe, exit: (if $code == 124 { "timeout" } else { $code }), summary: (boost-summary $log), log: $log})
    }
  }

  if "lit" in $want {
    let log = ($dir | path join "lit.log")
    let suite = ($env.CONDA_PREFIX | path join share naga-sg-tests lit)
    print $"sg-test-run: [lit] start (date now | format date '%H:%M:%S')"
    let code = (run-logged $log "omp" "lit" $suite "-v")
    print $"sg-test-run: [lit] exit ($code)"
    $rows = ($rows | append {cell: lit, exe: lit, exit: (if $code == 124 { "timeout" } else { $code }), summary: "-", log: $log})
  }

  print ""
  print ($rows | table --expand)
  let failed = ($rows | where {|r| $r.exit != 0 } | length)
  print $"sg-test-run: ($rows | length) runs, ($failed) failed"
  if $failed > 0 { exit 1 }
}
