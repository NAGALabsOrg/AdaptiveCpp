#!/usr/bin/env nu
# write-build-cfg.nu: clang config files for the FRESHLY BUILT clang, in the
# BUILD tree's bin dir (<build>/bin), so every host-side invocation of the
# bootstrapped clang gets the isolation the conda compiler activation gives the
# outer compiler: sysroot, gcc toolchain and the activation flags.
#
# Derived from acpp-toolchain's shared/write-build-cfg.nu, reduced to linux-64
# (DEVIATION: windows and macOS branches removed) plus the i386 cfgs below.
#
# Why: the LLVM_ENABLE_RUNTIMES child (compiler-rt, libomp) is a SEPARATE cmake
# driven by <build>/bin/clang, and ninja-spawned configures never see the outer
# flags. build-linux.nu strips CFLAGS/CXXFLAGS/CPPFLAGS from the env, so the
# child only gets what RUNTIMES_CMAKE_ARGS hands it. A cfg file next to the clang
# follows EVERY invocation, whatever configures it.
#
# Mechanism (clang docs, "Configuration files"): for the driver mode of the
# invoked name clang loads <triple>-<driver>.cfg from its own directory, where
# <triple> is the effective target. We write <T>-clang.cfg (C), <T>-clang++.cfg
# (C++), <T>-clang-cpp.cfg (preprocessor), and NEVER a plain clang.cfg /
# <arch>.cfg: device-target invocations (spir64, nvptx64, amdgcn) must not load anything.
#
# Runs BEFORE build-linux.nu strips the activation env. Absolute paths only, one
# option per line. Not the install-time activation/install-cfg.nu (that one
# writes the relocatable <CFGDIR>-relative files shipped in $PREFIX/bin).
#
# Usage: nu write-build-cfg.nu <build-dir>

source build-lib.nu

# Flag units never written: gcc-only; clang parses it and ignores it, printing
# -Wignored-optimization-argument on every single compile.
const DROP_UNITS = [
  "-fno-merge-constants"
]

# Options whose argument is a separate word; the pair stays on one cfg line.
const TWO_WORD_OPTIONS = ["-isystem" "-I" "-L" "-include" "-iquote" "-idirafter" "-isysroot" "-stdlib++-isystem" "/I" "/imsvc" "-imsvc" "/external:I"]

# An environment variable as a string; unset -> "".
def env-or-empty [name: string] {
  $env | get -o $name | default "" | into string
}

# Split a flag string into units, one per cfg line.
def flag-units [flags: string] {
  let words = ($flags | split row -r '\s+' | where {|w| $w != "" })
  mut out = []
  mut i = 0
  while $i < ($words | length) {
    let w = ($words | get $i)
    if ($w in $TWO_WORD_OPTIONS) and ($i + 1) < ($words | length) {
      $out = ($out | append $"($w) ($words | get ($i + 1))")
      $i = $i + 2
    } else {
      $out = ($out | append $w)
      $i = $i + 1
    }
  }
  $out
}

# Drop the units clang rejects or warns about; remove duplicates, keep order.
def keep-units [units: list<string>] {
  $units | where {|u| not ($u in $DROP_UNITS) } | uniq
}

# Write one cfg file; returns the path.
def write-cfg [bin: string, triple: string, driver: string, header: string, lines: list<string>] {
  let path = ($bin | path join $"($triple)-($driver).cfg")
  ([$header] ++ $lines | str join "\n") + "\n" | save --force $path
  $path
}

# Same sysroot choice as conda-toolchain-args: CONDA_BUILD_SYSROOT, else the
# conda layout under BUILD_PREFIX.
def linux-cfgs [bin: string] {
  let triple = (linux-triple)
  let bp = (env-or-empty BUILD_PREFIX)
  let sysroot_env = (env-or-empty CONDA_BUILD_SYSROOT)
  let derived = (if $bp != "" { [$bp $triple "sysroot"] | path join } else { "" })
  let sysroot = (if ($sysroot_env != "" and ($sysroot_env | path exists)) {
    $sysroot_env
  } else if ($derived != "" and ($derived | path exists)) {
    $derived
  } else {
    ""
  })
  if $sysroot == "" or $bp == "" {
    error make {msg: "write-build-cfg: no conda sysroot or BUILD_PREFIX (CONDA_BUILD_SYSROOT unset and the BUILD_PREFIX layout is absent)"}
  }

  let cflags = (env-or-empty CFLAGS)
  let cxxflags = (env-or-empty CXXFLAGS)
  let cppflags = (env-or-empty CPPFLAGS)
  let ldflags = (env-or-empty LDFLAGS)
  let iso = [$"--sysroot=($sysroot)" $"--gcc-toolchain=($bp)"]
  let ld = (keep-units (flag-units $ldflags))
  let c_lines = $iso ++ (keep-units ((flag-units $cflags) ++ (flag-units $cppflags))) ++ $ld
  let cxx_lines = $iso ++ (keep-units ((flag-units $cxxflags) ++ (flag-units $cppflags))) ++ $ld
  let cpp_lines = $iso ++ (keep-units (flag-units $cppflags))

  let note = $"# Build-tree clang config for ($triple), written by write-build-cfg.nu. Not shipped."
  [
    {path: (write-cfg $bin $triple "clang" $"($note) C driver." $c_lines), token: $sysroot}
    {path: (write-cfg $bin $triple "clang++" $"($note) C++ driver." $cxx_lines), token: $sysroot}
    {path: (write-cfg $bin $triple "clang-cpp" $"($note) Preprocessor driver." $cpp_lines), token: $sysroot}
  ]
  # DEVIATION (fork fix, handoff.md 24-31): i386 cfgs, see i386-cfgs below.
  | append (i386-cfgs $bin $iso $sysroot)
}

# DEVIATION (fork fix, handoff.md 24-31; from the fork's former naga/build/write-build-cfg.nu).
# -m32 changes the effective triple to i386-conda-linux-gnu (compiler-rt probes and
# builds an i386 variant). Without a cfg for that triple the just-built clang
# searches the system, which must never be used. Only the isolation lines: with
# the conda sysroot in force the 32-bit probe gives the true answer for this
# toolchain (it has no 32-bit glibc, so i386 is dropped, as on CI).
def i386-cfgs [bin: string, iso: list<string>, sysroot: string] {
  let t = "i386-conda-linux-gnu"
  let note = $"# Build-tree clang config for ($t), written by write-build-cfg.nu. Not shipped. Isolation only."
  [
    {path: (write-cfg $bin $t "clang" $"($note) C driver." $iso), token: $sysroot}
    {path: (write-cfg $bin $t "clang++" $"($note) C++ driver." $iso), token: $sysroot}
    {path: (write-cfg $bin $t "clang-cpp" $"($note) Preprocessor driver." $iso), token: $sysroot}
  ]
}

def main [build: string] {
  let bin = ($build | path join "bin")
  mkdir $bin

  let written = (linux-cfgs $bin)

  # Fail fast: every expected file exists and carries its sysroot token.
  for w in $written {
    if not ($w.path | path exists) { error make {msg: $"write-build-cfg: expected ($w.path) was not written"} }
    if ($w.token != "") and not (open --raw $w.path | str contains $w.token) {
      error make {msg: $"write-build-cfg: ($w.path) does not contain '($w.token)'"}
    }
  }

  for w in $written {
    print $"--- ($w.path)"
    print (open --raw $w.path)
  }
  print $"write-build-cfg: wrote ($written | length) file\(s) into ($bin)"
}
