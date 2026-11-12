# lib.nu: cmake argument assembly for the local LLVM+AdaptiveCpp build
# (linux-64 only). Trimmed from acpp-toolchain's shared/build-stage-lib.nu.
# A library on purpose: it defines NO `main`; build.nu and write-build-cfg.nu
# `source` it.
#
# Local differences from the packaging build: ROCm/HIP backend is not built
# (WITH_ROCM_BACKEND=OFF, LLVM_TARGETS_TO_BUILD=X86;NVPTX); no staging cache.

def cpu-count [] { $env.CPU_COUNT? | default (sys cpu | length | into string) | into int }

# Link jobs are memory-bound, not CPU-bound: LLVM links are ~1-2 GB each.
def link-jobs [] {
  let mem_gb = (((sys mem | get total | into int) / 1073741824) | math round)
  [1 ([($mem_gb // 4) (cpu-count)] | math min)] | math max
}

def linux-triple [] { "x86_64-conda-linux-gnu" }

def common-args [src: string, prefix: string, build: string] {
  [
    "-DCMAKE_BUILD_TYPE=Release"
    $"-DCMAKE_INSTALL_PREFIX=($prefix)"
    ...(if (which ccache | is-empty) { [] } else { ["-DCMAKE_C_COMPILER_LAUNCHER=ccache" "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"] })
    # ROCm is off, so no AMDGPU/SPIRV targets (those exist in packaging only for
    # TheRock's libamd_comgr). OpenCL / Level Zero use the SPIRV-LLVM-Translator
    # ExternalProject, which the fork builds itself.
    "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX"
    # compiler-rt and openmp are RUNTIMES (the LLVM 23 way), built by a child
    # cmake driven by the just-built clang.
    "-DLLVM_ENABLE_RUNTIMES=compiler-rt;openmp"
    # Flat lib/ for libomp and compiler-rt: AMD's tree hard-codes this OFF, upstream defaults it ON on Linux.
    "-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF"
    "-DLLVM_BUILD_TOOLS=ON"
    "-DCLANG_BUILD_TOOLS=ON"
    "-DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF"
    "-DLLVM_ENABLE_RTTI=ON"
    "-DLLVM_ENABLE_EH=ON"
    "-DLLVM_EXTERNAL_PROJECTS=AdaptiveCpp"
    $"-DLLVM_EXTERNAL_ADAPTIVECPP_SOURCE_DIR=($src)/AdaptiveCpp"
    "-DLLVM_ADAPTIVECPP_LINK_INTO_TOOLS=ON"
    # conda supplies the vendor runtimes from their own packages. A relative
    # vendor root makes each vendor's app-config value $ACPP_RT_LIB_DIR-relative
    # instead of baking in the build prefix.
    "-DACPP_CUDA_ROOT=."
    "-DACPP_OCL_ROOT=."
    "-DACPP_ZE_ROOT=."
    "-DACPP_LIBOMP_ROOT=lib"
    # The host JIT defaults to svml on x86_64 (intel-cmplr-lib-rt in lib/).
    "-DACPP_SVML_ROOT=lib"
    "-DWITH_CUDA_BACKEND=ON"
    "-DWITH_CPU_BACKEND=ON"
    "-DWITH_ROCM_BACKEND=OFF"
    "-DACPP_COMPILER_FEATURE_PROFILE=full"
    # The fork refuses LLVM newer than its tested range unless told otherwise.
    "-DACPP_EXPERIMENTAL_LLVM=ON"
    $"-DCUDAToolkit_ROOT=($prefix)"
    # The fork clones its SPIR-V translator itself; pin the commit.
    "-DLLVMSPIRV_COMMIT=a5fa639ff3f550780c34615830f9949bed4cba06"
    "-DOPENMP_ENABLE_LIBOMPTARGET=OFF"
    # libomp must not write into the source tree, and must not install the
    # libgomp/libiomp5 alias symlinks (libgomp.so.1 would collide with gcc's).
    "-DLIBOMP_COPY_EXPORTS=OFF"
    "-DLIBOMP_INSTALL_ALIASES=OFF"
    "-DCOMPILER_RT_BUILD_BUILTINS=ON"
    "-DCOMPILER_RT_BUILD_SANITIZERS=ON"
    "-DCOMPILER_RT_BUILD_PROFILE=ON"
    "-DCOMPILER_RT_BUILD_LIBFUZZER=ON"
    $"-DLLVM_PARALLEL_LINK_JOBS=(link-jobs)"
    "-DLLVM_INCLUDE_BENCHMARKS=OFF"
    "-DLLVM_INCLUDE_EXAMPLES=OFF"
    "-DLLVM_INCLUDE_TESTS=OFF"
    "-DLLVM_ENABLE_ZLIB=FORCE_ON"
    "-DLLVM_ENABLE_ZSTD=FORCE_ON"
    # Optional LLVM / LLDB features: FORCE_ON (LLVM_ENABLE_*) / ON (LLDB_ENABLE_*)
    # make a missing library fail the configure. LLDB's option macro knows only
    # On/Off/Auto: FORCE_ON there silently turns the feature OFF.
    "-DLLVM_ENABLE_LIBXML2=FORCE_ON"
    "-DLLDB_ENABLE_LIBXML2=ON"
    "-DLLDB_ENABLE_LZMA=ON"
    # python scripting against the Python LIMITED API (abi3).
    "-DLLDB_ENABLE_SWIG=ON"
    "-DLLDB_ENABLE_PYTHON=ON"
    "-DLLDB_ENABLE_PYTHON_LIMITED_API=ON"
    "-DLLDB_EMBED_PYTHON_HOME=OFF"
    "-DLLDB_ENABLE_TREESITTER=ON"
  ]
}

def conda-toolchain-args [] {
  let sysroot = ($env.CONDA_BUILD_SYSROOT? | default "")
  let bp = ($env.BUILD_PREFIX? | default "")
  let derived = (if $bp != "" { [$bp (linux-triple) "sysroot"] | path join } else { "" })
  let chosen = (if ($sysroot != "" and ($sysroot | path exists)) {
    $sysroot
  } else if ($derived != "" and ($derived | path exists)) {
    $derived
  } else {
    ""
  })

  if $chosen == "" {
    print "WARNING: no conda sysroot found (CONDA_BUILD_SYSROOT unset and BUILD_PREFIX layout absent); compiler-rt will use system headers"
    return []
  }
  print $"compiler-rt runtimes toolchain: sysroot=($chosen) gcc-toolchain=($bp)"

  # PSEUDO-CROSS: the runtimes are configured by a SEPARATE child cmake driven
  # by the just-built clang. It never receives the outer CMAKE_{C,CXX}_FLAGS and
  # seeds its flags from env CFLAGS/CXXFLAGS (conda's gcc-shaped flags, which
  # build.nu strips). Everything it needs goes through RUNTIMES_CMAKE_ARGS:
  #   --sysroot       C library headers (glibc 2.28)
  #   --gcc-toolchain C++ stdlib headers (libstdc++)
  #   COMPILER_TARGET clang's --target; must match the sysroot triple
  #   CMAKE_FIND_ROOT_PATH* mirrors the compiler activation's CMAKE_ARGS; the
  #   list separator inside is `|` (LLVM's ExternalProject LIST_SEPARATOR).
  # Deliberately NOT set: CMAKE_SYSTEM_NAME=Linux (flips the child into full
  # cross mode).
  let triple = (linux-triple)
  let flags = $"--sysroot=($chosen) --gcc-toolchain=($bp)"
  let host_prefix = ($env.PREFIX? | default "")
  [
    ([
      $"-DCMAKE_SYSROOT=($chosen)"
      $"-DCMAKE_C_COMPILER_TARGET=($triple)"
      $"-DCMAKE_CXX_COMPILER_TARGET=($triple)"
      $"-DCMAKE_ASM_COMPILER_TARGET=($triple)"
      $"-DCMAKE_C_FLAGS=($flags)"
      $"-DCMAKE_CXX_FLAGS=($flags)"
      "-DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER"
      "-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY"
      "-DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY"
      $"-DCMAKE_FIND_ROOT_PATH=($host_prefix)|($chosen)"
    ] | str join ";" | $"-DRUNTIMES_CMAKE_ARGS=($in)")
  ]
}

# The outer configure must receive the conda flags EXPLICITLY once build.nu
# strips them from the env. Replicates cmake's env-seeding: CFLAGS->CMAKE_C_FLAGS,
# CXXFLAGS->CMAKE_CXX_FLAGS, LDFLAGS->all three *_LINKER_FLAGS, plus -pthread
# (an explicit cache value overrides _INIT, so it is folded in here).
def outer-flag-args [] {
  let cflags = ($env.CFLAGS? | default "")
  let cxxflags = ($env.CXXFLAGS? | default "")
  let base_ld = ($env.LDFLAGS? | default "")
  let ldflags = ([$base_ld "-pthread"] | str join " " | str trim)
  [
    $"-DCMAKE_C_FLAGS=($cflags)"
    $"-DCMAKE_CXX_FLAGS=($cxxflags)"
    $"-DCMAKE_EXE_LINKER_FLAGS=($ldflags)"
    $"-DCMAKE_SHARED_LINKER_FLAGS=($ldflags)"
    $"-DCMAKE_MODULE_LINKER_FLAGS=($ldflags)"
  ]
}

# Dump the runtimes/builtins child-configure failure evidence. CMake >= 3.26
# writes try_compile evidence to CMakeFiles/CMakeConfigureLog.yaml, not
# CMakeError.log.
def dump-runtimes-logs [build: string] {
  let root = ($build | path join "runtimes")
  for f in (glob $"($root)/**/CMakeConfigureLog.yaml") {
    print $"===== child configure evidence: ($f) ====="
    let text = (open --raw $f | lines)
    $text | first 500 | str join "\n" | print
    print $"===== ... tail of ($f) ====="
    $text | last 200 | str join "\n" | print
  }
}

def linux-args [src: string, prefix: string] {
  [
    "-DLLVM_ENABLE_PROJECTS=clang;lld;lldb;clang-tools-extra;bolt;polly"
    "-DLLVM_ENABLE_PEDANTIC=OFF"
    # With ccache as the compiler launcher LLVM warns about precompiled headers.
    "-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON"
    "-DCOMPILER_RT_BUILD_XRAY=ON"
    "-DCOMPILER_RT_BUILD_MEMPROF=ON"
    "-DCOMPILER_RT_BUILD_ORC=ON"
    # One dylib every tool links against.
    "-DLLVM_BUILD_LLVM_DYLIB=ON"
    "-DLLVM_LINK_LLVM_DYLIB=ON"
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    # libedit / curses (+panel): lldb line editing and the `gui` command.
    "-DLLVM_ENABLE_LIBEDIT=FORCE_ON"
    "-DLLDB_ENABLE_LIBEDIT=ON"
    "-DLLDB_ENABLE_CURSES=ON"
    "-DCMAKE_INSTALL_RPATH=$ORIGIN/../lib"
    $"-DLLVM_HOST_TRIPLE=(linux-triple)"
    $"-DLLVM_DEFAULT_TARGET_TRIPLE=(linux-triple)"
    "-DWITH_LEVEL_ZERO_BACKEND=ON"
    "-DWITH_OPENCL_BACKEND=ON"
    "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX"
    $"-DCUDA_DEVICE_LIBS_PATH=($prefix)/nvvm/libdevice"
    $"-DOpenCL_LIBRARY=($prefix)/lib/libOpenCL.so"
    $"-DOpenCL_INCLUDE_DIR=($prefix)/include"
    $"-DFETCHCONTENT_SOURCE_DIR_OCL-HEADERS=($src)/OpenCL-Headers"
    $"-DFETCHCONTENT_SOURCE_DIR_OCL-CXX-HEADERS=($src)/OpenCL-CLHPP"
  ] ++ (conda-toolchain-args)
}

# The complete cmake argument list. Reads the conda flags from the environment
# (CFLAGS, CXXFLAGS, LDFLAGS, CMAKE_ARGS), so it must run BEFORE build.nu
# strips them. Order: activation CMAKE_ARGS (authoritative, first), common,
# outer flags, linux (later -D wins).
def stage-args [src: string, prefix: string, build: string] {
  let activation_args = ($env.CMAKE_ARGS? | default "" | split row -r '\s+' | where {|a| $a != "" })
  $activation_args
    | append (common-args $src $prefix $build)
    | append (outer-flag-args)
    | append (linux-args $src $prefix)
}
