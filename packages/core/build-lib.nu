# build-lib.nu: the cmake argument assembly behind build-linux.nu. Derived from
# acpp-toolchain's shared/build-stage-lib.nu, reduced to the linux-64 arguments
# (acpp-toolchain is being retired; other platforms get their own scripts when
# they are ported). A library on purpose: it defines NO `main`.

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
    # ccache when present
    ...(if (which ccache | is-empty) { [] } else { ["-DCMAKE_C_COMPILER_LAUNCHER=ccache" "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"] })
    "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX"
    # compiler-rt and openmp are RUNTIMES (the LLVM 23 way: LLVM_ENABLE_PROJECTS=openmp
    # is a fatal error, and =compiler-rt is deprecated). They are built by the
    # bootstrapping runtimes build, a child cmake driven by the just-built clang, and
    # installed into this prefix.
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
    # The host JIT defaults to svml (intel-cmplr-lib-rt in lib/).
    "-DACPP_SVML_ROOT=lib"
    "-DWITH_CUDA_BACKEND=ON"
    "-DWITH_CPU_BACKEND=ON"
    "-DWITH_ROCM_BACKEND=OFF"
    "-DACPP_COMPILER_FEATURE_PROFILE=full"
    # The fork refuses LLVM newer than its tested range unless told otherwise;
    # this build is LLVM 23 (AMD snapshot).
    "-DACPP_EXPERIMENTAL_LLVM=ON"
    $"-DCUDAToolkit_ROOT=($prefix)"
    # SPIR-V translator: NAGALabsOrg/SPIRV-LLVM-Translator naga/amd-llvm-23, Khronos
    # main 7d4b128d + the AdaptiveCpp address-space patch.
    "-DLLVMSPIRV_COMMIT=a5fa639ff3f550780c34615830f9949bed4cba06"
    "-DOPENMP_ENABLE_LIBOMPTARGET=OFF"
    # libomp must not write into the source tree (exports/), and must not
    # install the libgomp/libiomp5 alias symlinks (libgomp.so.1 would collide
    # with the gcc runtime's in every environment).
    "-DLIBOMP_COPY_EXPORTS=OFF"
    "-DLIBOMP_INSTALL_ALIASES=OFF"
    # compiler-rt: builtins plus the sanitizer runtimes.
    "-DCOMPILER_RT_BUILD_BUILTINS=ON"
    "-DCOMPILER_RT_BUILD_SANITIZERS=ON"
    "-DCOMPILER_RT_BUILD_PROFILE=ON"
    "-DCOMPILER_RT_BUILD_LIBFUZZER=ON"
    $"-DLLVM_PARALLEL_LINK_JOBS=(link-jobs)"
    "-DLLVM_INCLUDE_BENCHMARKS=OFF"
    "-DLLVM_INCLUDE_EXAMPLES=OFF"
    "-DLLVM_INCLUDE_TESTS=OFF"
    # FileCheck, not, count and the other LLVM utilities ship in naga-acpp-llvm-dev.
    "-DLLVM_INSTALL_UTILS=ON"
    # AMD's llvm-project creates bin/flang and bin/amdflang when flang is not a
    # built project: a Fortran driver name with no front end, colliding with conda-forge's flang.
    "-DCLANG_LINK_FLANG=OFF"
    "-DLLVM_ENABLE_ZLIB=FORCE_ON"
    "-DLLVM_ENABLE_ZSTD=FORCE_ON"
    # Optional LLVM / LLDB features: FORCE_ON (LLVM_ENABLE_*) / ON (LLDB_ENABLE_*)
    # make a missing library fail the configure instead of silently disabling the
    # feature. LLDB's option macro knows only On/Off/Auto.
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

# The runtimes child cmake (compiler-rt, libomp) is a SEPARATE configure driven by
# the just-built clang; it never receives the outer CMAKE_{C,CXX}_FLAGS. It gets
# its sysroot and gcc toolchain through RUNTIMES_CMAKE_ARGS (pseudo-cross:
# --sysroot finds glibc 2.28 headers, --gcc-toolchain the libstdc++ headers,
# COMPILER_TARGET makes the target explicit). `|` is the embedded list separator.
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

# The outer configure receives the conda flags EXPLICITLY once build-linux.nu
# strips them from the env. Replicates cmake's env seeding: CFLAGS->CMAKE_C_FLAGS,
# CXXFLAGS->CMAKE_CXX_FLAGS, LDFLAGS->all three *_LINKER_FLAGS (+ -pthread).
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

# Dump the runtimes/builtins child-configure failure evidence into the log: the
# console only shows "ABI info - failed"; the WHY lives in CMakeConfigureLog.yaml.
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
  ([
    "-DLLVM_ENABLE_PROJECTS=clang;lld;lldb;clang-tools-extra;bolt;polly"
    "-DLLVM_ENABLE_PEDANTIC=OFF"
    # With ccache as the compiler launcher LLVM warns about precompiled headers.
    "-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON"
    "-DCOMPILER_RT_BUILD_XRAY=ON"
    "-DCOMPILER_RT_BUILD_MEMPROF=ON"
    "-DCOMPILER_RT_BUILD_ORC=ON"
    # One dylib every tool links against: the seam the whole partition rests on.
    "-DLLVM_BUILD_LLVM_DYLIB=ON"
    "-DLLVM_LINK_LLVM_DYLIB=ON"
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    "-DLLVM_ENABLE_LIBEDIT=FORCE_ON"
    "-DLLDB_ENABLE_LIBEDIT=ON"
    "-DLLDB_ENABLE_CURSES=ON"
    "-DCMAKE_INSTALL_RPATH=$ORIGIN/../lib"
    $"-DLLVM_HOST_TRIPLE=(linux-triple)"
    $"-DLLVM_DEFAULT_TARGET_TRIPLE=(linux-triple)"
  ] ++ (linux-x86-backend-args $src $prefix)
    ++ (conda-toolchain-args))
}

# The full backend set. ROCm: TheRock's multi-arch tarball is a BUILD input found
# through hip_ROOT; its runtime subset is copied into the prefix afterwards and
# ACPP_HIP_ROOT=. tells the fork the prefix IS the HIP root.
def linux-x86-backend-args [src: string, prefix: string] {
  [
    "-DWITH_LEVEL_ZERO_BACKEND=ON"
    "-DWITH_OPENCL_BACKEND=ON"
    "-DWITH_ROCM_BACKEND=ON"
    "-DACPP_HIP_ROOT=."
    $"-Dhip_ROOT=($src)/rocm-dist"
    # SPIRV: TheRock's libamd_comgr.so.3 needs LLVMInitializeSPIRV* from our libLLVM.
    "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU;SPIRV"
    $"-DCUDA_DEVICE_LIBS_PATH=($prefix)/nvvm/libdevice"
    $"-DOpenCL_LIBRARY=($prefix)/lib/libOpenCL.so"
    $"-DOpenCL_INCLUDE_DIR=($prefix)/include"
    $"-DFETCHCONTENT_SOURCE_DIR_OCL-HEADERS=($src)/OpenCL-Headers"
    $"-DFETCHCONTENT_SOURCE_DIR_OCL-CXX-HEADERS=($src)/OpenCL-CLHPP"
  ]
}

# The complete cmake argument list. Reads the conda flags from the environment
# (CFLAGS, CXXFLAGS, LDFLAGS, CMAKE_ARGS), so it must run BEFORE build-linux.nu strips them.
def stage-args [src: string, prefix: string, build: string] {
  let flag_args = (outer-flag-args)

  # The compiler activation package's own cmake arguments (triplet binutils,
  # install layout, CMAKE_FIND_ROOT_PATH* carrying the sysroot), passed verbatim, first.
  let activation_args = ($env.CMAKE_ARGS? | default "" | split row -r '\s+' | where {|a| $a != "" })

  # The ROCm tree is outside the activation's find roots, so add it as a root.
  let rocm = ($src | path join "rocm-dist")
  let activation_args = (if ($rocm | path exists) {
    $activation_args | each {|a|
      if ($a | str starts-with "-DCMAKE_FIND_ROOT_PATH=") { $"($a);($rocm)" } else { $a }
    }
  } else { $activation_args })

  ($activation_args
    | append (common-args $src $prefix $build)
    | append $flag_args
    | append (linux-args $src $prefix))
}
