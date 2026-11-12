# check-devices.nu: run `acpp-info -l` and fail unless OpenMP, CUDA, OpenCL and
# Level Zero each loaded with at least one device.
#
# Output format (src/tools/acpp-info/acpp-info.cpp list_backends):
#   Loaded backend 0: OpenMP
#     Found device: AdaptiveCpp OpenMP host device
def main [] {
  let out = (^acpp-info -l | complete)
  print $out.stdout
  if $out.stderr != "" { print --stderr $out.stderr }
  if $out.exit_code != 0 { error make {msg: $"acpp-info -l exited with ($out.exit_code)"} }

  # Group the "Found device" lines under the backend line above them.
  mut current = ""
  mut found = {}
  for l in ($out.stdout | lines) {
    let b = ($l | parse --regex '^Loaded backend \d+: (?<name>.+?)\s*$')
    if ($b | is-not-empty) {
      $current = ($b | first | get name)
      $found = ($found | upsert $current 0)
    } else if ($l =~ '^\s+Found device:') and $current != "" {
      $found = ($found | upsert $current (($found | get $current) + 1))
    }
  }

  let missing = (["OpenMP" "CUDA" "OpenCL" "Level Zero"] | where {|n| (($found | get -o $n | default 0) < 1) })
  if ($missing | is-not-empty) {
    error make {msg: $"missing backend(s) or no devices: ($missing | str join ', ')"}
  }
  print "check-devices: OpenMP, CUDA, OpenCL and Level Zero all report devices"
}
