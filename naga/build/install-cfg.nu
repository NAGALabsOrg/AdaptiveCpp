#!/usr/bin/env nu
# install-cfg.nu: write clang's triple-named config files into $PREFIX/bin.
#
# Clang looks for <default-triple>-<driver>.cfg beside itself, so shipping
# these lets a bare `clang` invocation find --sysroot/-isystem/-L/-rpath with NO
# activation script and NO environment variable. The files are
# <CFGDIR>-relative (see cfg/*.tmpl), so they survive conda's prefix relocation.
# Linux only: the full template (clang/clang++ end in
# --sysroot=<CFGDIR>/../<triple>/sysroot).
#
# Usage: nu install-cfg.nu <triple>
def main [triple: string] {
  let cfg_dir = ($env.FILE_PWD | path join "cfg")
  let bindir = ($env.PREFIX | path join "bin")
  mkdir $bindir

  let drivers = [
    {tmpl: "clang.cfg.tmpl", out: $"($triple)-clang.cfg"}
    {tmpl: "clang++.cfg.tmpl", out: $"($triple)-clang++.cfg"}
    {tmpl: "clang-cpp.cfg.tmpl", out: $"($triple)-clang-cpp.cfg"}
  ]

  for d in $drivers {
    open --raw ($cfg_dir | path join $d.tmpl)
      | str replace --all "@TRIPLE@" $triple
      | save --force ($bindir | path join $d.out)
    print $"install-cfg: wrote ($d.out)"
  }
}
