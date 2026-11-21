# check-packages.nu: unit checks of the sliced naga-acpp package set, against the installed
# `default` environment's conda-meta records and files (and three scratch environments).
#
#   pixi run check-packages
#
# 1 partition  2 contents  3 dependencies  4 linking  5 symlinks  6 build paths
# 7 activation 8 bare toolchain  9 export round trip  10 name collisions
# Every check prints PASS or FAIL lines; the task exits non-zero if any FAIL was printed.
# The baseline of the core's file list is the part 1 conda-meta record saved under the logs dir.

const SLICES = [naga-acpp-runtime naga-acpp naga-acpp-compiler-rt naga-acpp-llvm-dev naga-acpp-tools naga-acpp-lldb]
const VENDORS = [naga-acpp-runtime-cuda naga-acpp-runtime-level-zero naga-acpp-runtime-ocl naga-acpp-runtime-rocm]
const NO_PROVIDER = '^(libc|libm|libdl|libpthread|librt|libutil|libresolv|libnsl|libanl|ld-linux-x86-64)\.so|^libcuda\.so\.1$'
# bundled OpenCL-Headers/CLHPP installs (conda-forge owns those paths)
const UNSHIPPED = '^(include/CL/|share/cmake/OpenCLHeadersCpp/|share/pkgconfig/OpenCL-)'
# outside the partition contract: shipped by at most one package, need not be shipped at all
const PYCACHE = '/__pycache__/'

def pass [name: string] { print $"PASS  ($name)"; 0 }
def fail [name: string, detail: string] { print $"FAIL  ($name): ($detail)"; 1 }
def verdict [name: string, problems: list<string>] {
  if ($problems | is-empty) { pass $name } else {
    print $"FAIL  ($name): ($problems | length) problem\(s\)"
    $problems | first 25 | each {|p| print $"        ($p)" }
    1
  }
}

def records [prefix: string] {
  glob $"($prefix)/conda-meta/*.json" | each {|f| open $f } | where {|r| ($r | get -o files) != null }
}

def main [] {
  let root = $env.PWD
  let env_dir = ($root | path join ".pixi" "envs" "default")
  let baseline = (glob ($env.HOME | path join "logs" "acpp-fork" "fork-splits-*" "baseline" "naga-acpp-core-*.json") | sort | last)
  let recs = (records $env_dir)
  let by_name = ($recs | reduce -f {} {|r, acc| $acc | insert $r.name $r })
  let ours = ([naga-acpp-llvm] ++ $SLICES ++ $VENDORS)
  mut failed = 0

  # 1. partition
  let core_files = (open $baseline | get files)
  let claims = ($ours | each {|n|
    let r = ($by_name | get -o $n)
    if $r == null { [] } else { $r.files | each {|f| {path: $f, pkg: $n} } }
  } | flatten | group-by path)
  let missing_pkgs = ($ours | where {|n| ($by_name | get -o $n) == null })
  mut problems = ($missing_pkgs | each {|n| $"package not installed: ($n)" })
  for f in $core_files {
    let c = ($claims | get -o $f | default [])
    let shipped_by = ($c | get -o pkg | default [])
    if ($f =~ $PYCACHE) {
      if ($c | length) > 1 { $problems = ($problems | append $"pyc claimed twice: ($f) by ($shipped_by | str join ',')") }
    } else if ($f =~ $UNSHIPPED) {
      if not ($c | is-empty) { $problems = ($problems | append $"unshipped file claimed: ($f) by ($shipped_by | str join ',')") }
    } else if ($c | is-empty) { $problems = ($problems | append $"unassigned: ($f)") } else if ($c | length) > 1 {
      $problems = ($problems | append $"claimed twice: ($f) by ($shipped_by | str join ',')")
    }
  }
  $failed = $failed + (verdict "1 partition (every core file in exactly one package or the unshipped list)" $problems)

  # 2. contents (the shipped package_contents tests, linux-64)
  let contents = {
    naga-acpp-runtime: {
      exists: [bin/opt bin/llc bin/lld lib/libLLVM.so.23.0git lib/libclang-cpp.so.23.0git lib/libomp.so
        lib/hipSYCL/librt-backend-omp.so lib/hipSYCL/llvm-to-backend/libllvm-to-backend.so lib/hipSYCL/llvm-to-backend/libllvm-to-host.so
        lib/hipSYCL/llvm-to-backend/libllvm-to-spirv.so lib/hipSYCL/ext/llvm-spirv/bin/llvm-spirv etc/AdaptiveCpp/acpp-toolchain.json]
      absent: [lib/libamdhip64.so lib/libhsa-runtime64.so lib/liblldb.so lib/hipSYCL/librt-backend-cuda.so lib/hipSYCL/librt-backend-ze.so
        lib/hipSYCL/librt-backend-ocl.so lib/hipSYCL/librt-backend-hip.so lib/hipSYCL/llvm-to-backend/libllvm-to-ptx.so
        lib/hipSYCL/llvm-to-backend/libllvm-to-amdgpu.so lib/hipSYCL/ext/llvm-spirv/lib/libLLVMSPIRVLib.a etc/conda/activate.d/acpp-runtime-activate.sh]
    }
    naga-acpp: {
      exists: [bin/clang bin/acpp bin/acpp-info bin/llvm-ar bin/x86_64-conda-linux-gnu-clang.cfg bin/x86_64-conda-linux-gnu-clang++.cfg
        bin/x86_64-conda-linux-gnu-clang-cpp.cfg include/AdaptiveCpp/sycl/sycl.hpp lib/cmake/AdaptiveCpp/adaptivecpp-config-version.cmake lib/clang/23/include/omp.h]
      absent: [bin/clang-tidy bin/lldb bin/llvm-config]
    }
    naga-acpp-compiler-rt: { exists: [], absent: [lib/clang/23/include/omp.h bin/clang], globs: [lib/clang/23/lib/*/libclang_rt.asan* lib/clang/23/lib/*/libclang_rt.profile*] }
    naga-acpp-llvm-dev: {
      exists: [bin/llvm-config bin/llvm-as lib/cmake/llvm/LLVMConfig.cmake lib/cmake/clang/ClangConfig.cmake lib/hipSYCL/ext/llvm-spirv/lib/libLLVMSPIRVLib.a]
      absent: [bin/clang bin/acpp bin/clangd bin/lldb bin/clang-tidy]
    }
    naga-acpp-tools: { exists: [bin/clang-format bin/clang-tidy bin/clangd bin/run-clang-tidy bin/scan-build bin/llvm-bolt share/clang-doc/index.js], absent: [] }
    naga-acpp-lldb: { exists: [bin/lldb bin/lldb-server], absent: [] }
    naga-acpp-runtime-cuda: { exists: [lib/hipSYCL/librt-backend-cuda.so lib/hipSYCL/llvm-to-backend/libllvm-to-ptx.so], absent: [] }
    naga-acpp-runtime-level-zero: { exists: [lib/hipSYCL/librt-backend-ze.so], absent: [] }
    naga-acpp-runtime-ocl: { exists: [lib/hipSYCL/librt-backend-ocl.so], absent: [] }
    naga-acpp-runtime-rocm: {
      exists: [lib/hipSYCL/librt-backend-hip.so lib/hipSYCL/llvm-to-backend/libllvm-to-amdgpu.so lib/llvm/amdgcn/bitcode/ockl.bc]
      globs: [lib/libamdhip64.so* lib/libhsa-runtime64.so* lib/libamd_comgr.so* lib/librocm_kpack.so*]
      absent: []
    }
  }
  $problems = []
  for n in ($contents | columns) {
    let r = ($by_name | get -o $n)
    if $r == null { continue }
    let files = $r.files
    let spec = ($contents | get $n)
    for e in $spec.exists { if not ($e in $files) { $problems = ($problems | append $"($n): missing ($e)") } }
    for a in $spec.absent { if $a in $files { $problems = ($problems | append $"($n): must not ship ($a)") } }
    for g in ($spec | get -o globs | default []) {
      let re = ("^" + ($g | str replace --all '.' '\.' | str replace --all '*' '[^/]*') + "$")
      if ($files | where {|f| $f =~ $re } | is-empty) { $problems = ($problems | append $"($n): no file matches ($g)") }
    }
  }
  # the three clang cfg files, and the bare-name lib/*.so* split
  $failed = $failed + (verdict "2 contents (package_contents tests)" $problems)

  # 3. dependencies
  let all_pkgs = ($SLICES ++ $VENDORS)
  let sibling = "naga-acpp-runtime ==0.dev0"
  let expect = {
    naga-acpp-runtime: { must: [intel-cmplr-lib-rt libnuma libedit libxml2-16 libzlib zstd libgcc libstdcxx _x86_64-microarch-level "naga-acpp-llvm ==23.0git"], mustnot: [cuda ocl-icd level-zero python] }
    naga-acpp: { must: [$sibling python libgcc-devel_linux-64 libstdcxx-devel_linux-64 sysroot_linux-64 binutils_impl_linux-64 libgcc libstdcxx _x86_64-microarch-level "naga-acpp-llvm ==23.0git"], mustnot: [cuda ocl-icd level-zero] }
    naga-acpp-compiler-rt: { must: ["naga-acpp ==0.dev0" _x86_64-microarch-level libgcc], mustnot: [cuda ocl-icd level-zero python] }
    naga-acpp-llvm-dev: { must: [$sibling "naga-acpp ==0.dev0" "naga-acpp-tools ==0.dev0" libxml2-devel libedit libzlib zstd _x86_64-microarch-level], mustnot: [cuda ocl-icd level-zero] }
    naga-acpp-tools: { must: [$sibling python libzlib zstd _x86_64-microarch-level], mustnot: [cuda ocl-icd level-zero] }
    naga-acpp-lldb: { must: [$sibling python libedit libtree-sitter ncurses liblzma libxml2-16 libzlib _x86_64-microarch-level], mustnot: [cuda ocl-icd level-zero] }
    naga-acpp-runtime-cuda: { must: [$sibling cuda-cudart cuda-nvvm-tools _x86_64-microarch-level], mustnot: [ocl-icd level-zero python], cuda_below_13: true }
    naga-acpp-runtime-level-zero: { must: [$sibling level-zero _x86_64-microarch-level], mustnot: [cuda ocl-icd python] }
    naga-acpp-runtime-ocl: { must: [$sibling ocl-icd ocl-icd-system _x86_64-microarch-level], mustnot: [cuda level-zero python] }
    naga-acpp-runtime-rocm: { must: [$sibling _x86_64-microarch-level libgcc], mustnot: [cuda ocl-icd level-zero python] }
  }
  $problems = []
  for n in ($expect | columns) {
    let r = ($by_name | get -o $n)
    if $r == null { continue }
    let deps = $r.depends
    let names = ($deps | each {|d| $d | split row " " | first })
    let want = ($expect | get $n)
    for m in $want.must {
      let ok = (if ($m | str contains " ") { $m in $deps } else { $m in $names })
      if not $ok { $problems = ($problems | append $"($n): missing dependency ($m); has ($deps | str join ' | ')") }
    }
    if ($want | get -o cuda_below_13 | default false) {
      let cudart = ($deps | where {|d| ($d | split row " " | first) == "cuda-cudart" })
      let bounded = ($cudart | any {|d| $d =~ '<(12\.[0-9.a-z]*|13(\.0)?(a0)?)(,|$| )' })
      if not $bounded { $problems = ($problems | append $"($n): cuda-cudart has no upper bound below 13: ($cudart | to nuon)") }
    }
    for x in $want.mustnot {
      if ($names | any {|d| $d | str starts-with $x }) { $problems = ($problems | append $"($n): unexpected dependency ($x)") }
    }
  }
  $failed = $failed + (verdict "3 dependencies" $problems)
  let old = (glob ($baseline | path dirname | path join "naga-acpp-local-*.json") | first | open | get depends | each {|d| $d | split row " " | first } | uniq | sort)
  let new = ($all_pkgs | each {|n| $by_name | get -o $n | default {depends: []} | get depends | each {|d| $d | split row " " | first } } | flatten | uniq | sort)
  print $"      old naga-acpp-local dependency names not in the new set: ($old | where {|d| not ($d in $new) } | to nuon)"
  print $"      new dependency names not in the old set: ($new | where {|d| not ($d in $old) } | to nuon)"

  # 4. linking
  let readelf = ($env_dir | path join "bin" "llvm-readelf")
  let providers = ($recs | each {|r| $r.files | each {|f| {base: ($f | path basename), pkg: $r.name} } } | flatten | group-by base | items {|k, v| {base: $k, pkgs: ($v | get pkg | uniq)} } | reduce -f {} {|x, acc| $acc | insert $x.base $x.pkgs })
  let closure = {|name|
    mut seen = [$name]
    mut todo = [$name]
    while not ($todo | is-empty) {
      let cur = ($todo | first)
      $todo = ($todo | skip 1)
      for d in ($by_name | get -o $cur | default {depends: []} | get depends) {
        let dn = ($d | split row " " | first)
        if ($dn in ($by_name | columns)) and not ($dn in $seen) { $seen = ($seen | append $dn); $todo = ($todo | append $dn) }
      }
    }
    $seen
  }
  $problems = []
  for n in $all_pkgs {
    let r = ($by_name | get -o $n)
    if $r == null { continue }
    let clo = (do $closure $n)
    let elf = ($r.paths_data.paths | where path_type == "hardlink" | get _path | where {|f|
      let p = ($env_dir | path join $f)
      (ls -l $p | get 0.size) > 4b and ((open --raw $p | into binary | bytes at 0..<4) == 0x[7f454c46])
    })
    let bad = ($elf | par-each {|f|
      let out = (^$readelf --dynamic -W ($env_dir | path join $f) | complete | get stdout)
      let needed = ($out | lines | where {|l| $l | str contains "(NEEDED)" } | each {|l| $l | parse --regex '\[(?<v>[^\]]*)\]' | get 0.v })
      $needed | where {|lib| not ($lib =~ $NO_PROVIDER) } | where {|lib|
        let pk = ($providers | get -o $lib | default [])
        not ($pk | any {|p| $p in $clo })
      } | each {|lib| $"($n): ($f) needs ($lib), no provider in its run closure" }
    } | flatten)
    $problems = ($problems | append $bad)
  }
  $failed = $failed + (verdict "4 linking (every DT_NEEDED resolves in the run closure; glibc and libcuda.so.1 excepted)" $problems)

  # 5. symlinks
  $problems = []
  for r in ($recs | where {|r| $r.name in $ours }) {
    for p in ($r.paths_data.paths | where path_type == "softlink") {
      let full = ($env_dir | path join $p._path)
      if not ($full | path exists) { $problems = ($problems | append $"($r.name): dangling ($p._path)") }
    }
  }
  $failed = $failed + (verdict "5 symlinks (none dangles)" $problems)

  # 6. build paths
  # FAIL: a pixi work-dir / host-prefix path in a file with no prefix_placeholder record (outside ELF
  # debug sections; static archives not scanned). INFO only: paths into the durable build tree.
  let needles = [".pixi/bld/" "host_placehold"]
  let info_needle = ".build/naga-acpp-core"
  mut infos = []
  $problems = []
  for r in ($recs | where {|r| $r.name in $all_pkgs }) {
    let exempt = ($r.paths_data.paths | where {|p| ($p | get -o prefix_placeholder) != null } | get _path)
    let cands = ($r.paths_data.paths | where {|p| $p.path_type == "hardlink" and not ($p._path in $exempt) and not ($p._path | str ends-with ".a") } | get _path)
    let hits = ($cands | par-each {|f|
      let p = ($env_dir | path join $f)
      let b = (open --raw $p | into binary)
      let is_elf = (($b | length) > 4 and ($b | bytes at 0..<4) == 0x[7f454c46])
      let has = ($needles | any {|nd| ($b | bytes index-of ($nd | into binary)) != -1 })
      let has_info = (($b | bytes index-of ($info_needle | into binary)) != -1)
      if not $has { (if $has_info { {pkg: $r.name, file: $f, where: "INFO"} } else { null }) } else if not $is_elf { {pkg: $r.name, file: $f, where: "non-ELF content"} } else {
        let secs = (^$readelf -S -W $p | complete | get stdout | lines | each {|l| $l | parse --regex '\]\s+(?<name>\.z?debug\S*)\s+\S+\s+[0-9a-f]+\s+(?<off>[0-9a-f]+)\s+(?<size>[0-9a-f]+)' } | flatten | each {|s| {off: ($"0x($s.off)" | into int), size: ($"0x($s.size)" | into int)} } | sort-by off)
        mut pieces = []
        mut cur = 0
        for s in $secs { $pieces = ($pieces | append ($b | bytes at $cur..<($s.off))); $cur = $s.off + $s.size }
        $pieces = ($pieces | append ($b | bytes at $cur..<($b | length)))
        let rest = ($pieces | bytes collect)
        if ($needles | any {|nd| ($rest | bytes index-of ($nd | into binary)) != -1 }) { {pkg: $r.name, file: $f, where: "ELF outside .debug sections"} } else if $has_info { {pkg: $r.name, file: $f, where: "INFO"} } else { null }
      }
    } | compact)
    $infos = ($infos | append ($hits | where where == "INFO"))
    $problems = ($problems | append ($hits | where where != "INFO" | each {|h| $"($h.pkg): ($h.file): ($h.where)" }))
  }
  print $"      INFO: ($infos | length) file\(s\) carry a durable build-tree path \(.build/naga-acpp-core\), by package: ($infos | group-by pkg | items {|k, v| {pkg: $k, n: ($v | length)} } | to nuon)"
  # the core's own record must register the placeholder of the toolchain json
  let core_art = (glob ($root | path join ".pixi" "artifacts-v0" "naga-acpp-core" "*" "*.conda") | each {|f| {f: $f, m: (ls $f | get 0.modified)} } | sort-by m | last | get f)
  let core_paths = (^pixi exec -s unzip -s zstd -s tar -s nushell nu naga/tasks/conda-paths.nu $core_art | from json | get paths)
  let toolchain_json = ($core_paths | where _path == "etc/AdaptiveCpp/acpp-toolchain.json" | first)
  if ($toolchain_json | get -o prefix_placeholder) == null { $problems = ($problems | append $"naga-acpp-core: no prefix_placeholder record for etc/AdaptiveCpp/acpp-toolchain.json \(($core_art | path basename)\)") }
  $failed = $failed + (verdict "6 build paths (no pixi work-dir path in files without a placeholder record; ELF debug sections and static archives exempt)" $problems)

  # 7. activation round trip and compile-link-run in an environment containing the activation
  let tmp = (mktemp --directory)
  let act = '
set -e
export CONDA_PREFIX="$PREFIX_UNDER_TEST"
for side in clang clangxx; do
  var=CC; [ $side = clangxx ] && var=CXX
  before="$(printenv $var || echo __UNSET__)"
  . "$CONDA_PREFIX/etc/conda/activate.d/activate-naga-acpp-${side}_linux-64.sh"
  [ -n "$(printenv $var)" ] || { echo "$var not set"; exit 1; }
  printenv $var | grep -q x86_64-conda-linux-gnu || { echo "$var not triple-prefixed"; exit 1; }
  . "$CONDA_PREFIX/etc/conda/deactivate.d/deactivate-naga-acpp-${side}_linux-64.sh"
  after="$(printenv $var || echo __UNSET__)"
  [ "$before" = "$after" ] || { echo "$var not restored: $before != $after"; exit 1; }
done
echo "activation round-trip OK"
'
  let cc = ('
set -e
cd "$1"
echo "int main(void){return 0;}" > t.c
$CC $CFLAGS $LDFLAGS t.c -o t_c
./t_c
echo "#include <iostream>
int main(){std::cout << 1 << std::endl; return 0;}" > t.cpp
$CXX $CXXFLAGS $LDFLAGS t.cpp -o t_cxx
./t_cxx
echo "compile-link-run OK"
')
  let rt = (^pixi run -e check-activation bash -c $"PREFIX_UNDER_TEST=\"$CONDA_PREFIX\"; ($act)" | complete)
  print $rt.stdout
  $failed = $failed + (if $rt.exit_code == 0 { pass "7a activation round trip" } else { fail "7a activation round trip" $rt.stderr })
  let cr = (^pixi run -e check-activation bash -c $cc _ $tmp | complete)
  print $cr.stdout
  $failed = $failed + (if $cr.exit_code == 0 { pass "7b compile, link, run C and C++ with the activation" } else { fail "7b compile, link, run C and C++ with the activation" $cr.stderr })

  # 8. bare toolchain: only naga-acpp-toolchain, no activation
  let bare_src = ($tmp | path join "bare.cpp")
  "#include <iostream>\nint main(){std::cout << \"bare\" << std::endl; return 0;}\n" | save --force $bare_src
  let sycl_src = ($tmp | path join "sycl.cpp")
  "#include <sycl/sycl.hpp>\nint main(){ sycl::queue q; int* p = sycl::malloc_shared<int>(4, q); q.parallel_for(sycl::range<1>(4), [=](sycl::id<1> i){ p[i] = (int)i[0] * 2; }).wait(); int ok = 1; for (int i = 0; i < 4; i++) ok &= p[i] == i * 2; sycl::free(p, q); return ok ? 0 : 1; }\n" | save --force $sycl_src
  let bare_script = '
set -e
cd "$1"
clang++ -v bare.cpp -o bare 2> v.txt || { cat v.txt; exit 1; }
grep -i "Configuration file" v.txt
./bare
ACPP_VISIBILITY_MASK=omp acpp -O2 sycl.cpp -o sycl_omp
ACPP_VISIBILITY_MASK=omp ./sycl_omp
echo "bare toolchain OK"
'
  let bare = (^pixi run -e check-bare bash -c $bare_script _ $tmp | complete)
  print $bare.stdout
  $failed = $failed + (if $bare.exit_code == 0 { pass "8 bare toolchain (clang++ hello, acpp SYCL on OpenMP)" } else { fail "8 bare toolchain" $"($bare.stdout)($bare.stderr)" })

  # 9. export round trip
  # the consumer is installed into its own environment (same build context as the default install)
  let b = (^pixi install -e check-consumer | complete)
  print $b.stderr
  if $b.exit_code != 0 { $failed = $failed + (fail "9 export round trip" $"install failed: ($b.stderr)") } else {
    let consumer_env = ($root | path join ".pixi" "envs" "check-consumer")
    let rec = (records $consumer_env | where name == "naga-acpp-check-consumer" | first)
    let want = ["naga-acpp-runtime ==0.dev0" "naga-acpp-llvm ==23.0git"]
    let lacking = ($want | where {|w| not ($w in $rec.depends) })
    $failed = $failed + (verdict "9 export round trip (consumer host-depending only on naga-acpp)" ($lacking | each {|w| $"depends lacks ($w); has ($rec.depends | str join ' | ')" }))
  }

  # 10. name collisions against conda-forge feedstock-outputs
  let names = ($ours ++ [naga-acpp-clang_linux-64 naga-acpp-clangxx_linux-64 naga-acpp-toolchain] | uniq)
  $problems = []
  for name in $names {
    let url = $"https://raw.githubusercontent.com/conda-forge/feedstock-outputs/main/outputs/($name | str substring 0..0)/($name | str substring 1..1)/($name | str substring 2..2)/($name).json"
    let status = (http get --full --allow-errors $url | get status)
    if $status != 404 { $problems = ($problems | append $"($name): feedstock-outputs probe returned ($status), expected 404") }
  }
  $failed = $failed + (verdict "10 name collisions (fail closed)" $problems)

  # 11. an environment with only naga-acpp-llvm-dev (plus cmake and ninja): find_package(LLVM) and find_package(Clang) configure
  let cm_src = ($tmp | path join "cm-src")
  mkdir $cm_src
  "cmake_minimum_required(VERSION 3.24)\nproject(x LANGUAGES C)\nfind_package(LLVM CONFIG REQUIRED)\nfind_package(Clang CONFIG REQUIRED)\nmessage(STATUS \"found LLVM ${LLVM_PACKAGE_VERSION} and Clang\")\n" | save --force ($cm_src | path join "CMakeLists.txt")
  # the calling shell may carry another pixi environment's bin dir; CMake must see only the check environment
  let clean_path = ($env.PATH | where {|p| not ($p | str contains ".pixi/envs/") })
  let leaks = [CMAKE_PREFIX_PATH CMAKE_ARGS CONDA_BUILD_SYSROOT CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LDFLAGS_LD DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_CPPFLAGS CXX CPP CC_FOR_BUILD CXX_FOR_BUILD]
  let cm = (with-env {PATH: $clean_path, CC: "clang"} {
    for v in $leaks { hide-env --ignore-errors $v }
    ^pixi run -e check-llvm-dev cmake -S $cm_src -B ($tmp | path join "cm-bld") -G Ninja | complete })
  print $cm.stdout
  $failed = $failed + (if $cm.exit_code == 0 { pass "11 find_package(LLVM) and find_package(Clang) configure with naga-acpp-llvm-dev alone" } else { fail "11 find_package(LLVM)/find_package(Clang)" $"($cm.stdout)($cm.stderr)" })

  rm --recursive --force $tmp
  if $failed > 0 { error make {msg: $"check-packages: ($failed) check\(s\) failed"} }
  print "check-packages: all checks passed"
}
