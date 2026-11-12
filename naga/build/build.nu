# build.nu: the LLVM + AdaptiveCpp build behind `pixi install` (linux-64, local).
#
# Sources are fetched by rattler-build into $SRC_DIR (llvm-project, AdaptiveCpp =
# this working tree, OpenCL-Headers, OpenCL-CLHPP). The CMake binary dir lives
# OUTSIDE the work dir ($SRC_DIR/../build_dir) so ninja state survives between
# builds and rebuilds are incremental.
#
# Invoked in place as `nu <SRC_DIR>/AdaptiveCpp/naga/build/build.nu` so that
# `source lib.nu` resolves beside this file.

source lib.nu

def main [] {
  # Empty values forwarded from the recipe env mean "unset".
  for v in [CCACHE_DIR CCACHE_MAXSIZE CCACHE_BASEDIR CCACHE_NOHASHDIR ACPP_BUILD_DIR] {
    if ($env | get -o $v | default "") == "" { hide-env --ignore-errors $v }
  }
  let src = $env.SRC_DIR
  let prefix = $env.PREFIX
  let build = ($env.ACPP_BUILD_DIR? | default ($src | path join ".." "build_dir"))
  mkdir $build

  $env.CMAKE_PREFIX_PATH = ($build + ":" + ($env.CMAKE_PREFIX_PATH? | default ""))
  $env.CCACHE_COMPILERCHECK = "content"

  # Capture the cmake arguments (they embed the conda flags for the OUTER
  # configure), then STRIP those flags from the environment: the runtimes child
  # cmake (spawned mid-build by ninja) seeds its flags from env CFLAGS/CXXFLAGS.
  # The outer build sees identical flags via outer-flag-args; the child starts
  # clean and gets exactly what RUNTIMES_CMAKE_ARGS hands it.
  let args = (stage-args $src $prefix $build)
  # Diagnostic: the flags and find roots the outer configure receives.
  print "--- outer configure flag args"
  $args | where {|a| $a =~ '^-D(CMAKE_(C|CXX)_FLAGS|CMAKE_[A-Z]+_LINKER_FLAGS|CMAKE_FIND_ROOT_PATH=)' } | each {|a| print $a } | ignore

  # Clang config files for the freshly built clang in <build>/bin. Must run
  # HERE: the activation env is still intact, and the loop below strips it.
  ^nu ($env.FILE_PWD | path join "write-build-cfg.nu") $build

  # LDFLAGS deliberately STAYS in the env: the SPIRV-translator inner
  # ExternalProject env-seeds its linker flags and needs conda's -L$PREFIX/lib.
  for v in [CFLAGS CXXFLAGS CPPFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS] {
    hide-env --ignore-errors $v
  }

  ^cmake ($src | path join "llvm-project" "llvm") -G Ninja ...$args -B $build

  # The LLVM dylib must exist BEFORE AdaptiveCpp's SPIRV-LLVM-Translator
  # ExternalProject builds: its inner cmake links libLLVM.so directly, so the
  # outer ninja has no rule for it and high job counts race ahead of the link.
  ^cmake --build $build --target LLVM --parallel (cpu-count)
  ^cmake --build $build --parallel (cpu-count)

  cd $build
  ^cmake --install $build

  # HOLLOW-RUNTIMES GUARD: the runtimes child-configure fails SOFT; a broken
  # child compiler yields a green build that ships no clang_rt libs.
  let rt_glob = ($prefix | path join "lib" "clang" "**" "libclang_rt.asan*")
  if ((glob $rt_glob | length) == 0) {
    dump-runtimes-logs $build
    error make {msg: "compiler-rt runtimes are HOLLOW (no asan artifacts installed) - child configure evidence dumped above"}
  }

  # Clang config files (<triple>-clang{,++,-cpp}.cfg in bin/): bare clang then
  # finds --sysroot/-isystem/-L/-rpath with no activation script.
  ^nu ($env.FILE_PWD | path join "install-cfg.nu") (linux-triple)

  if (which ccache | is-not-empty) { ^ccache --show-stats }
}
