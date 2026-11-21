# render-install.nu: render + install the naga-acpp compiler activation scripts (linux-64) from the
# VENDORED conda-forge ctng-compiler-activation templates. A Nushell port of acpp-toolchain's
# shared/activation/render-install.sh (linux-64 slice); pinned ref in vendor/.../PINNED_REF.
# Output is byte-identical to the bash renderer's, except that the shellcheck gate is not run here.
#
# Env: NAGA_SOURCE_DIR (this directory), PREFIX, PKG_NAME (rattler-build), ACTIVATION_SIDE = clang | clangxx.

def subst [files: list<string>, from: string, to: string] {
  for f in $files { open --raw $f | str replace --all $from $to | save --force $f }
}

def main [] {
  let side = $env.ACTIVATION_SIDE
  let vendor = ($env.NAGA_SOURCE_DIR | path join "vendor" "ctng-compiler-activation")
  let work = (mktemp --directory)
  for f in [activate-gcc.sh activate-g++.sh deactivate-gcc.sh deactivate-g++.sh] {
    open --raw ($vendor | path join $f) | save --force ($work | path join $f)
  }

  let chost = "x86_64-conda-linux-gnu"
  let cbuild = $chost
  let cflags = "-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe"
  let cxxflags = "-fvisibility-inlines-hidden -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -ffunction-sections -pipe"
  let ldflags = "-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--disable-new-dtags -Wl,--gc-sections -Wl,--allow-shlib-undefined"
  let ldflags_ld = "-O2 --sort-common --as-needed -z relro -z now --disable-new-dtags --gc-sections --allow-shlib-undefined"
  let debug_cflags = "-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-all -fno-plt -Og -g -Wall -Wextra -fvar-tracking-assignments -ffunction-sections -pipe"
  let debug_cxxflags = "-fvisibility-inlines-hidden -fmessage-length=0 -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-all -fno-plt -Og -g -Wall -Wextra -fvar-tracking-assignments -ffunction-sections -pipe"

  # canonical @VAR@ substitution matrix, build_scripts.sh order
  let all = ["activate-gcc.sh" "activate-g++.sh" "deactivate-gcc.sh" "deactivate-g++.sh"] | each {|f| $work | path join $f }
  for s in [
    ["@UNAME_KERNEL_RELEASE@" "0"] ["@IS_WIN@" "0"] ["@MACHINE@" "x86_64"] ["@CMAKE_SYSTEM_NAME@" "Linux"]
    ["@MESON_SYSTEM@" "linux"] ["@MESON_FAMILY@" "x86_64"] ["@CBUILD@" $cbuild] ["@CHOST@" $chost]
    ["@CPPFLAGS@" "-DNDEBUG -D_FORTIFY_SOURCE=2 -O2"] ["@DEBUG_CPPFLAGS@" "-D_DEBUG -D_FORTIFY_SOURCE=2 -Og"]
    ["@CFLAGS@" $cflags] ["@DEBUG_CFLAGS@" $debug_cflags] ["@CXXFLAGS@" $cxxflags] ["@DEBUG_CXXFLAGS@" $debug_cxxflags]
    ["@LDFLAGS@" $ldflags] ["@LDFLAGS_LD@" $ldflags_ld] ["@EXE_EXT@" ""] ["@LIBRARY_PREFIX@" ""]
    ["@CONDA_BUILD_CROSS_COMPILATION@" ""]
  ] { subst $all $s.0 $s.1 }

  # clang variants: copies, then -fno-merge-constants removal in the activate scripts
  let w = {|f| $work | path join $f }
  open --raw (do $w "activate-gcc.sh") | save --force (do $w "activate-clang.sh")
  open --raw (do $w "activate-g++.sh") | save --force (do $w "activate-clang++.sh")
  open --raw (do $w "deactivate-gcc.sh") | save --force (do $w "deactivate-clang.sh")
  open --raw (do $w "deactivate-g++.sh") | save --force (do $w "deactivate-clang++.sh")
  subst [(do $w "activate-clang.sh") (do $w "activate-clang++.sh")] " -fno-merge-constants" ""

  # compiler-specific _tc_activation entries (what bash yields after line continuations)
  let q = {|s| "\"" + $s + "\"" }
  let clang_extra = " " + ([
    $"CC,($chost)-clang" $"CPP,($chost)-clang-cpp" $"OBJC,($chost)-clang" $"CC_FOR_BUILD,($cbuild)-clang"
    $"CPP_FOR_BUILD,($cbuild)-clang-cpp" $"OBJC_FOR_BUILD,($cbuild)-clang" $"CLANG,($chost)-clang"
    "ac_cv_func_malloc_0_nonnull,yes" "ac_cv_func_realloc_0_nonnull,yes"
  ] | each {|e| (do $q $e) + " " } | str join "")
  let clangxx_base = " " + ([
    $"CXX,($chost)-clang++" $"OBJCXX,($chost)-clang++" $"CXX_FOR_BUILD,($cbuild)-clang++"
    $"OBJCXX_FOR_BUILD,($cbuild)-clang++" $"CLANGXX,($chost)-clang++"
  ] | each {|e| (do $q $e) + " " } | str join "")
  # ACPP DELTA: SYCL environment on the CXX side; ${...} expands at activation time
  let clangxx_extra = $clangxx_base + " " + ([
    "ACPP_TARGETS,${ACPP_TARGETS:-generic}"
    $"ACPP_CLANG,${CONDA_PREFIX}/bin/($chost)-clang++"
    $"ACPP_CPU_CXX,${CONDA_PREFIX}/bin/($chost)-clang++"
  ] | each {|e| (do $q $e) + " " } | str join "")

  # the bash renderer's `find -name "*activate-clang.sh"` also matches deactivate-clang.sh
  let ac = (do $w "activate-clang.sh")
  let axx = (do $w "activate-clang++.sh")
  let c_files = [$ac (do $w "deactivate-clang.sh")]
  let cxx_files = [$axx (do $w "deactivate-clang++.sh")]
  subst $c_files "@C_EXTRA@" $clang_extra
  subst $c_files "@AR@" $"($chost)-ar"
  subst $c_files "@NM@" $"($chost)-nm"
  subst $c_files "@RANLIB@" $"($chost)-ranlib"
  subst $cxx_files "@CXX_EXTRA@" $clangxx_extra

  let prefix = $env.PREFIX
  let name = $env.PKG_NAME
  mkdir ($prefix | path join "etc" "conda" "activate.d") ($prefix | path join "etc" "conda" "deactivate.d") ($prefix | path join "bin")
  match $side {
    "clang" => {
      open --raw $ac | save --force ($prefix | path join "etc" "conda" "activate.d" $"activate-($name).sh")
      open --raw (do $w "deactivate-clang.sh") | save --force ($prefix | path join "etc" "conda" "deactivate.d" $"deactivate-($name).sh")
      ^ln -sf clang ($prefix | path join "bin" $"($chost)-clang")
      ^ln -sf clang-cpp ($prefix | path join "bin" $"($chost)-clang-cpp")
    }
    "clangxx" => {
      open --raw $axx | save --force ($prefix | path join "etc" "conda" "activate.d" $"activate-($name).sh")
      open --raw (do $w "deactivate-clang++.sh") | save --force ($prefix | path join "etc" "conda" "deactivate.d" $"deactivate-($name).sh")
      ^ln -sf clang++ ($prefix | path join "bin" $"($chost)-clang++")
    }
    _ => { error make {msg: $"unknown ACTIVATION_SIDE '($side)'"} }
  }
  rm --recursive --force $work
}
