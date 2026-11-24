# Run the sg graph toolchain test cells (package naga/sg-tests, env sg-tests).
#
#   nu naga/tasks/sg-test-run.nu <host|omp|cuda|ocl|ze|lit|all> ...
#
#   host -> sg_host_tests, sg_compiler_tests
#   omp  -> sg_driver_tests (also with OMP_NUM_THREADS=1), sg_tests, the rerun cases at adaptivity levels 1 and 2
#   cuda -> sg_driver_tests, sg_tests, the rerun cases at levels 1 and 2   (omp;cuda)
#   ocl  -> same                                                            (ocl)
#   ze   -> same                                                            (ze)
#   lit  -> lit over share/naga-sg-tests/lit
#
# The first line of every executable's log is
#   sg-cell backend=<b> platform=<p> device=<d> vendor=<v>
# A run whose log lacks it, or whose device is not of its cell's backend, fails and the
# message names the cell, the expected backend and the device found.
# Every cell runs even if an earlier one failed; each command runs under `timeout 1800s`
# (exit 124 is reported as "timeout"). Every device run gets a fresh empty ACPP_APPDB_DIR.
# Logs go to ~/logs/acpp-fork/sg-tests-<timestamp>/. The exit code is non-zero if any cell failed.

const cpu_names = '(?i)(cpu|xeon|core\(tm\)|ryzen|epyc|threadripper)'

export def expected-device [cell: string] {
  match $cell {
    "host" => {backend: "host", vendor: null, gpu: false, what: "no device"}
    "omp" => {backend: "omp", vendor: null, gpu: false, what: "the host CPU"}
    "cuda" => {backend: "cuda", vendor: '(?i)nvidia', gpu: true, what: "an NVIDIA GPU through CUDA"}
    "ocl" => {backend: "ocl", vendor: '(?i)intel', gpu: true, what: "an Intel GPU through OpenCL"}
    "ze" => {backend: "ze", vendor: '(?i)intel|^pci:32902$', gpu: true, what: "an Intel GPU through Level Zero"}
    _ => { error make {msg: $"unknown cell ($cell)"} }
  }
}

export def parse-cell-line [line: string] {
  $line | parse --regex '^sg-cell backend=(?<backend>\S+) platform=(?<platform>\S+) device=(?<device>\S+) vendor=(?<vendor>\S+)$'
}

# The verdict of one executable's log for the cell it ran in.
export def cell-verdict [cell: string, exe: string, log_text: string] {
  let want = (expected-device $cell)
  let first = ($log_text | ansi strip | lines | where {|l| ($l | str trim) != "" } | first 1 | get -o 0 | default "")
  let parsed = (parse-cell-line $first)
  if ($parsed | is-empty) {
    return {
      ok: false
      device: "none"
      message: $"cell ($cell) \(($exe)\): the log does not start with 'sg-cell backend=<b> platform=<p> device=<d> vendor=<v>'; expected backend ($want.backend) \(($want.what)\)"
    }
  }
  let got = ($parsed | first)
  let wrong_backend = $got.backend != $want.backend
  let wrong_vendor = ($want.vendor != null) and not ($got.vendor =~ $want.vendor)
  let on_cpu = $want.gpu and ($got.device =~ $cpu_names)
  if $wrong_backend or $wrong_vendor or $on_cpu {
    return {
      ok: false
      device: $got.device
      message: $"cell ($cell) \(($exe)\) expected backend ($want.backend) on ($want.what) but found device ($got.device) \(backend=($got.backend), platform=($got.platform), vendor=($got.vendor)\)"
    }
  }
  {ok: true, device: $got.device, message: ""}
}

def cells [] {
  [
    {group: host, mask: "omp", runs: [
      {exe: sg_host_tests, args: [], env: {}}
      {exe: sg_compiler_tests, args: [], env: {}}
    ]}
    ...(["omp" "cuda" "ocl" "ze"] | each {|g|
      {group: $g, mask: (if $g == "cuda" { "omp;cuda" } else { $g }), runs: [
        {exe: sg_driver_tests, args: [], env: {}}
        {exe: sg_tests, args: [], env: {}}
        {exe: sg_tests, args: ["--run_test=reruns"], env: {ACPP_ADAPTIVITY_LEVEL: "1"}}
        {exe: sg_tests, args: ["--run_test=reruns"], env: {ACPP_ADAPTIVITY_LEVEL: "2"}}
        ...(if $g == "omp" { [{exe: sg_driver_tests, args: [], env: {OMP_NUM_THREADS: "1"}}] } else { [] })
      ]}
    })
  ]
}

const run_limit_secs = 1800
const kill_grace_secs = 60

# A run that was ended by the limit (exit 124, or 137 after the grace period's kill) and ran at least
# as long as the limit is a timeout; any other exit code stays as it is.
export def classify-exit [code: int, elapsed_secs: float, limit_secs: int] {
  if ($code == 124) or (($code == 137 or $code == -9) and $elapsed_secs >= $limit_secs) { "timeout" } else { $code }
}

# One limited run: SIGTERM at the limit, SIGKILL after the grace period, so a process that blocks or
# ignores SIGTERM is ended too. The result is the exit code, or "timeout" when the limit ended it.
export def run-limited [log: path, mask: string, env_vars: record, limit_secs: int, grace_secs: int, bin: string, ...args: string] {
  let started = (date now)
  let code = (try {
    let r = (do --ignore-errors {
      with-env ({ACPP_VISIBILITY_MASK: $mask} | merge $env_vars) {
        ^timeout $"--kill-after=($grace_secs)s" $"($limit_secs)s" $bin ...$args o+e>| tee { save --force $log }
      }
    } | complete)
    $r.exit_code
  } catch {|e|
    if ($e.msg | str contains "core dumped") { 139 } else { 255 }
  })
  classify-exit $code (((date now) - $started | into int) / 1000000000.0) $limit_secs
}

def run-logged [log: path, mask: string, env_vars: record, bin: string, ...args: string] {
  run-limited $log $mask $env_vars $run_limit_secs $kill_grace_secs $bin ...$args
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
  mut n = 0
  for c in (cells) {
    if $c.group not-in $want { continue }
    for r in $c.runs {
      $n = $n + 1
      let tag = ($r.env | transpose k v | each {|e| $"($e.k)=($e.v)" } | str join "," )
      let name = ([$r.exe ...($r.args | each {|a| $a | str replace --all --regex '[^A-Za-z0-9_]' '' })] | str join "-")
      let log = ($dir | path join $"($c.group)-($n)-($name).log")
      let appdb = ($dir | path join $"appdb-($n)")
      mkdir $appdb
      print $"sg-test-run: [($c.group)] ($r.exe) ($tag) start (date now | format date '%H:%M:%S')"
      let code = (run-logged $log $c.mask ($r.env | merge {ACPP_APPDB_DIR: $appdb}) $r.exe "--log_level=test_suite" ...$r.args)
      rm --recursive --force $appdb
      let v = (cell-verdict $c.group $r.exe (open --raw $log))
      if not $v.ok { print -e $"sg-test-run: ($v.message)" }
      print $"sg-test-run: [($c.group)] ($r.exe) exit ($code)"
      let first_line = (open --raw $log | ansi strip | lines | where {|l| ($l | str trim) != "" } | first 1 | get -o 0 | default "")
      $rows = ($rows | append {
        cell: $c.group, exe: $r.exe, env: $tag
        exit: (if $code == "timeout" { "timeout" } else if $code == 0 and not $v.ok { "wrong device" } else { $code })
        device: $first_line, summary: (boost-summary $log), log: $log
      })
    }
  }

  if "lit" in $want {
    let log = ($dir | path join "lit.log")
    let suite = ($env.CONDA_PREFIX | path join share naga-sg-tests lit)
    print $"sg-test-run: [lit] start (date now | format date '%H:%M:%S')"
    let code = (run-logged $log "omp" {} "lit" $suite "-v")
    print $"sg-test-run: [lit] exit ($code)"
    $rows = ($rows | append {cell: lit, exe: lit, env: "", exit: (if $code == 124 { "timeout" } else { $code }), device: "-", summary: "-", log: $log})
  }

  print ""
  print ($rows | table --expand)
  let failed = ($rows | where {|r| $r.exit != 0 } | length)
  print $"sg-test-run: ($rows | length) runs, ($failed) failed"
  if $failed > 0 { exit 1 }
}
