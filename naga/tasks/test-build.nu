# Build AdaptiveCpp's OWN test suite (tests/) against the INSTALLED default
# environment: the in-tree tests/ is configured directly (no copy), with the
# env's clang/clang++ (what CI's pkg-generic wrapper gets from the compiler
# activation) and the env's AdaptiveCpp cmake config. The build tree lives
# under .pixi/ so nothing lands in the source tree.
#
# Options mirror shared/tests/suite/pkg-generic of acpp-toolchain (linux-64):
# ACPP_TARGETS=generic, WITH_PCUDA_TESTS=ON, WITH_PSTL_TESTS=ON, Release.
# Targets: rt_tests sycl_tests pstl_tests pcuda_tests (CI's ACPP_SUITE_EXES).
#
#   pixi run test-build
def main [] {
  let root = ($env.FILE_PWD | path join .. .. | path expand)
  let prefix = $env.CONDA_PREFIX
  let src = ($root | path join tests)
  let build = ($root | path join .pixi naga-tests)
  mkdir $build

  let args = [
    -G Ninja
    -S $src
    -B $build
    -DCMAKE_BUILD_TYPE=Release
    $"-DCMAKE_C_COMPILER=($prefix)/bin/clang"
    $"-DCMAKE_CXX_COMPILER=($prefix)/bin/clang++"
    $"-DCMAKE_PREFIX_PATH=($prefix)"
    $"-DAdaptiveCpp_DIR=($prefix)/lib/cmake/AdaptiveCpp"
    -DACPP_TARGETS=generic
    -DWITH_PSTL_TESTS=ON
    -DWITH_PCUDA_TESTS=ON
  ]
  print $"test-build: start (date now | format date '%H:%M:%S')"
  ^cmake ...$args
  let jobs = (sys cpu | length | into string)
  ^cmake --build $build --parallel $jobs --target rt_tests sycl_tests pstl_tests pcuda_tests
  print $"test-build: end (date now | format date '%H:%M:%S')"
}
