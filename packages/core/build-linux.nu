# build-linux.nu: the ONE LLVM+AdaptiveCpp build behind every naga-acpp package (linux-64).
#
# Derived from acpp-toolchain's shared/build-stage.nu, reduced to linux-64 (acpp-toolchain
# is being retired; other platforms get their own scripts when ported). The cmake
# arguments are in build-lib.nu. LLVM is built as a dylib (libLLVM.so) and every
# tool links it. The LLVM source is AMD's llvm-project snapshot (version suffix `git`,
# so the soname is libLLVM.so.23.0git), the commit TheRock's comgr is built on.
#
# DEVIATIONS from acpp-toolchain's shared/build-stage.nu (each marked in place):
#   - sources come from fetch-sources.nu, not from rattler-build's source section;
#   - NAGA_BUILD_DIR (set by pixi-build-naga) replaces ACPP_BUILD_DIR;
#   - the staging-cache diet is dropped (the package input globs do that job);
#   - ccache is pointed at the durable build dir;
#   - windows and macOS branches are removed.

source build-lib.nu
# DEVIATION: source cache (see header).
source fetch-sources.nu

def main [] {
  # Empty values forwarded from the recipe env mean "unset".
  for v in [CCACHE_DIR CCACHE_MAXSIZE CCACHE_BASEDIR CCACHE_NOHASHDIR] {
    if ($env | get -o $v | default "") == "" { hide-env --ignore-errors $v }
  }
  if ($env.NAGA_BUILD_DIR? | default "") == "" {
    error make {msg: "build-linux: NAGA_BUILD_DIR is not set (the package needs `build-dir` in its pixi-build-naga config)"}
  }

  # DEVIATION: rattler-build's sanitized env does not forward the caller's ccache
  # settings, so the cache lives beside the durable build tree.
  if ($env.CCACHE_DIR? | default "") == "" { $env.CCACHE_DIR = ($env.NAGA_BUILD_DIR | path join "ccache") }
  if ($env.CCACHE_MAXSIZE? | default "") == "" { $env.CCACHE_MAXSIZE = "30G" }
  if (which ccache | is-not-empty) { ^ccache --zero-stats }

  # DEVIATION: the sources are the fetch-sources.nu layout, not $SRC_DIR.
  let src = (prepare-sources)
  # TheRock's tree: used by the licence guard and the ROCm runtime deploy below.
  let rocm = ($src | path join "rocm-dist")

  # License-drift guard: package outputs ship the VENDORED texts from
  # shared/licenses/, so assert they are byte-identical to the licenses in the
  # actually extracted sources. A pin bump that changes a license fails HERE.
  for pair in [
    [($src | path join "shared" "licenses" "llvm-LICENSE.TXT"), ($src | path join "llvm-project" "LICENSE.TXT")]
    [($src | path join "shared" "licenses" "AdaptiveCpp-LICENSE"), ($src | path join "AdaptiveCpp" "LICENSE")]
    [($src | path join "shared" "licenses" "rocm" "hip-LICENSE.md"), ($src | path join "rocm-dist" "share" "doc" "hip" "LICENSE.md")]
    [($src | path join "shared" "licenses" "rocm" "rocr-LICENSE.md"), ($src | path join "rocm-dist" "share" "doc" "rocr" "LICENSE.md")]
    [($src | path join "shared" "licenses" "rocm" "amd_comgr-LICENSE.txt"), ($src | path join "rocm-dist" "share" "doc" "amd_comgr" "LICENSE.txt")]
    [($src | path join "shared" "licenses" "rocm" "rocprofiler-register-LICENSE.md"), ($src | path join "rocm-dist" "share" "doc" "rocprofiler-register" "LICENSE.md")]
  ] {
    if ((open --raw $pair.0 | str replace --all "\r" "") != (open --raw $pair.1 | str replace --all "\r" "")) {
      error make {msg: $"vendored license ($pair.0) differs from source tree ($pair.1): update packages/core/licenses/"}
    }
  }

  let prefix = $env.PREFIX
  # DEVIATION: one CMake build tree per host prefix, <NAGA_BUILD_DIR>/trees/<hash of $PREFIX> (was a
  # single ACPP_BUILD_DIR or $SRC_DIR/../build_dir). Pixi uses different work dirs, hence host
  # prefixes, for install, build and publish and for each package-config change; CMake caches
  # absolute prefix paths. Per-prefix trees keep each context incremental and never build against
  # a stale prefix. The prefix is stamped inside its tree; a tree whose stamped work dir
  # (.pixi/bld/naga-acpp-core/<key>) no longer exists is deleted at the start of each build.
  let trees = ($env.NAGA_BUILD_DIR | path join "trees")
  mkdir $trees
  let tree_of = {|p| $trees | path join ($p | hash sha256 | str substring 0..11) }
  # adopt the former single tree (<NAGA_BUILD_DIR>/build + host-prefix.txt) as its prefix's tree
  let legacy = ($env.NAGA_BUILD_DIR | path join "build")
  let legacy_stamp = ($env.NAGA_BUILD_DIR | path join "host-prefix.txt")
  if ($legacy | path exists) and ($legacy_stamp | path exists) {
    let legacy_prefix = (open --raw $legacy_stamp | str trim)
    let dest = (do $tree_of $legacy_prefix)
    if not ($dest | path exists) {
      mv $legacy $dest
      $legacy_prefix | save --force ($dest | path join "host-prefix.txt")
    }
    rm --force $legacy_stamp
  }
  let build = (do $tree_of $env.PREFIX)
  for t in (ls $trees | where type == dir | get name) {
    if $t == $build { continue }
    let s = ($t | path join "host-prefix.txt")
    if ($s | path exists) {
      let work_dir = (open --raw $s | str trim | path dirname)
      if not ($work_dir | path exists) { rm --recursive --force $t }
    }
  }
  mkdir $build
  $env.PREFIX | save --force ($build | path join "host-prefix.txt")

  $env.CMAKE_PREFIX_PATH = ($build + ":" + ($env.CMAKE_PREFIX_PATH? | default ""))
  $env.CCACHE_COMPILERCHECK = "content"

  # Capture the cmake arguments (they embed the conda flags for the OUTER
  # configure), then STRIP those flags from the environment: the runtimes child
  # cmake (spawned mid-build by ninja) seeds its flags from env CFLAGS/CXXFLAGS,
  # the pollution vector behind the system-header failure.
  let args = (stage-args $src $prefix $build)

  # Clang config files for the freshly built clang in <build>/bin: sysroot,
  # gcc toolchain and the activation flags. This covers the runtimes child and
  # anything else the bootstrapped clang compiles for the host. It must run HERE:
  # the activation env is still intact, and the hide-env loop below strips it.
  ^nu ($src | path join "shared" "write-build-cfg.nu") $build
  # LDFLAGS deliberately STAYS in the env: the SPIRV-translator inner
  # ExternalProject env-seeds its linker flags and NEEDS conda's -L$PREFIX/lib.
  for v in [CFLAGS CXXFLAGS CPPFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS] {
    hide-env --ignore-errors $v
  }

  ^cmake ($src | path join "llvm-project" "llvm") -G Ninja ...$args -B $build

  # The LLVM dylib must exist BEFORE AdaptiveCpp's SPIRV-LLVM-Translator
  # ExternalProject builds: its inner cmake links the file directly, so the outer
  # ninja has no rule for it and high job counts race ahead of the link.
  ^cmake --build $build --target LLVM --parallel (cpu-count)
  ^cmake --build $build --parallel (cpu-count)

  cd $build
  ^cmake --install $build

  # HOLLOW-RUNTIMES GUARD: the runtimes child-configure fails SOFT: a broken
  # child compiler yields a technically green build that ships no clang_rt libs.
  # Fail HERE instead, with the child's configure evidence dumped.
  if ((glob ($prefix | path join "lib" "clang" "**" "libclang_rt.asan*") | length) == 0) {
    dump-runtimes-logs $build
    error make {msg: "compiler-rt runtimes are HOLLOW (no asan artifacts installed): child configure evidence dumped above"}
  }

  # Unversioned .so symlinks are dev-package files elsewhere in conda; the
  # versioned sonames are what the runtime needs.
  for f in [libLLVM.so libLTO.so libRemarks.so libclang.so libclang-cpp.so] {
    let p = ($prefix | path join "lib" $f)
    if ($p | path exists) { rm $p }
  }
  # NB: NO triple-prefixed driver symlinks here. They are activation-package files;
  # a base package carrying the triplet poisons plain runtime environments.

  # ROCm runtime subset: ONLY these files come from TheRock's tarball (rocm-dist),
  # copied with the tarball's OWN layout so the discovered subdirs resolve under
  # the prefix (ACPP_HIP_ROOT=.). Nothing from its lib/llvm (no libLLVM,
  # libclang-cpp, clang, lld or tools), no .kpack/, no HIP/HSA headers: libamd_comgr
  # NEEDs libLLVM.so.23.0git and libclang-cpp.so.23.0git (RPATH $ORIGIN first) and
  # binds to OUR copies in the same lib/.
  let libdir = ($prefix | path join "lib")
  for stem in [
    {n: amdhip64, req: true} {n: hsa-runtime64, req: true} {n: amd_comgr, req: true}
    {n: hiprtc, req: true} {n: hiprtc-builtins, req: true} {n: rocm_kpack, req: true}
    {n: rocprofiler-register, req: true}
  ] {
    let matches = (glob ($rocm | path join "lib" $"lib($stem.n).so*"))
    if ($matches | is-empty) {
      if $stem.req { error make {msg: $"rocm deploy: no lib($stem.n).so* under ($rocm)/lib"} }
      continue
    }
    ^cp -a ...$matches $libdir
    print $"rocm deploy: lib($stem.n): ($matches | length) files"
  }
  let sysdeps = ($rocm | path join "lib" "rocm_sysdeps")
  if not ($sysdeps | path exists) { error make {msg: $"rocm deploy: no rocm_sysdeps at ($sysdeps)"} }
  ^cp -a $sysdeps $libdir
  let bitcode = ($rocm | path join "lib" "llvm" "amdgcn" "bitcode")
  if not ($bitcode | path exists) { error make {msg: $"rocm deploy: no device bitcode at ($bitcode)"} }
  let bitcode_dest = ($prefix | path join "lib" "llvm" "amdgcn")
  mkdir $bitcode_dest
  ^cp -a $bitcode $bitcode_dest
  print $"rocm deploy: bitcode -> ($bitcode_dest)/bitcode"

  # Clang config files (<triple>-clang{,++,-cpp}.cfg in bin/): bare clang then
  # finds --sysroot/-isystem/-L/-rpath with no activation script.
  ^nu ($src | path join "shared" "activation" "install-cfg.nu") (linux-triple) true

  # DEVIATION: the staging-cache diet (deleting the source trees from $SRC_DIR)
  # is dropped: there is no staging cache, and the trees live in the durable source cache.

  if (which ccache | is-not-empty) { ^ccache --show-stats }
}
