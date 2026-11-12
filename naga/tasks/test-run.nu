# Run AdaptiveCpp's test suite cells built by test-build.nu.
#
#   nu naga/tasks/test-run.nu [omp] [cuda] [ocl] [ze] [sanitizers]   (no args = all)
#
# The matrix mirrors acpp-toolchain's linux-64 CI (suite.nu run generic ...):
#   omp        -> rt_tests, sycl_tests, pstl_tests, pcuda_tests
#   omp;cuda   -> sycl_tests            (the RTX 4090)
#   ocl        -> sycl_tests            (CI's OpenCL is a CPU runtime; here it is Intel Graphics)
#   ze         -> sycl_tests            (Level Zero, Intel Graphics; not covered by CI)
#   sanitizers -> sanitizer-smoke.nu    (address, thread)
# Every executable gets CI's `--log_level=test_suite` so each test names itself
# in the log. EVERY cell runs even if an earlier one failed; each has a hard
# timeout so a hang is reported rather than blocking. The final table lists
# mask, executable, exit code, Boost summary line and log path; the exit code
# of this script is non-zero if any cell failed.
const TIMEOUT_SECONDS = 1800

def cells [] {
  [
    {group: omp, mask: "omp", exes: [rt_tests sycl_tests pstl_tests pcuda_tests]}
    {group: cuda, mask: "omp;cuda", exes: [sycl_tests]}
    {group: ocl, mask: "ocl", exes: [sycl_tests]}
    {group: ze, mask: "ze", exes: [sycl_tests]}
  ]
}

# Run one command line under timeout, teeing all output to $log; returns the exit code.
def run-logged [log: path, mask: string, bin: string, ...args: string] {
  let r = (do --ignore-errors {
    with-env {ACPP_VISIBILITY_MASK: $mask} {
      ^timeout $"($TIMEOUT_SECONDS)s" $bin ...$args o+e>| tee { save --force $log }
    }
  } | complete)
  $r.exit_code
}

def boost-summary [log: path, code: int] {
  if $code == 124 { return $"TIMEOUT after ($TIMEOUT_SECONDS)s" }
  let found = (open --raw $log | ansi strip | lines | where {|l| $l starts-with '*** ' })
  if ($found | is-empty) { "-" } else { $found | last }
}

def main [...groups: string] {
  let root = ($env.FILE_PWD | path join .. .. | path expand)
  let build = ($root | path join .pixi naga-tests)
  let want = (if ($groups | is-empty) or ($groups | any {|g| $g == "all" }) {
    [omp cuda ocl ze sanitizers]
  } else { $groups })

  let dir = ($env.ACPP_TEST_LOG_DIR? | default ($env.HOME | path join logs acpp-fork $"tests-(date now | format date '%Y%m%d-%H%M%S')"))
  mkdir $dir
  print $"test-run: logs in ($dir)"

  mut rows = []
  for c in (cells) {
    if $c.group not-in $want { continue }
    for exe in $c.exes {
      let bin = ($build | path join $exe)
      let log = ($dir | path join $"($c.group)-($exe).log")
      print $"test-run: [($c.mask)] ($exe) start (date now | format date '%H:%M:%S')"
      mut code = 0
      if not ($bin | path exists) {
        "executable not built; run test-build\n" | save --force $log
        $code = 127
      } else {
        $code = (run-logged $log $c.mask $bin "--log_level=test_suite")
      }
      print $"test-run: [($c.mask)] ($exe) exit ($code)"
      $rows = ($rows | append {mask: $c.mask, exe: $exe, exit: $code, summary: (boost-summary $log $code), log: $log})
    }
  }

  if "sanitizers" in $want {
    let log = ($dir | path join "sanitizers.log")
    let work = ($dir | path join sanitizer-work)
    mkdir $work
    let script = ($env.FILE_PWD | path join sanitizer-smoke.nu)
    print $"test-run: sanitizers start (date now | format date '%H:%M:%S')"
    let code = (run-logged $log "omp" "nu" $script "--workdir" $work)
    print $"test-run: sanitizers exit ($code)"
    let last = (open --raw $log | ansi strip | lines | where {|l| $l =~ 'sanitizer smoke:' } | last 1 | get -o 0 | default "-")
    $rows = ($rows | append {mask: "-", exe: "sanitizer-smoke", exit: $code, summary: $last, log: $log})
  }

  print ""
  print ($rows | table --expand)
  let failed = ($rows | where exit != 0 | length)
  print $"test-run: ($rows | length) cells, ($failed) failed"
  if $failed > 0 { exit 1 }
}
