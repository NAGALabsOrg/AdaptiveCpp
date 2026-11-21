# Compile and install one C++ program with the toolchain's clang++ from the host prefix.
def main [] {
  let bin = ($env.PREFIX | path join "bin")
  mkdir $bin
  ^($env.PREFIX | path join "bin" "clang++") ($env.NAGA_SOURCE_DIR | path join "hello.cpp") -o ($bin | path join "naga-acpp-check-consumer")
}
