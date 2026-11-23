# Upstreaming triage: baseline

Pins (every record file states them): U `27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc`, N `0e2bac974c40554f62286cbffe1f06c87cc9caa9`, B `9ec394efa584ced004d644af2194d2133e111d26`. Recorded 2026-11-22 (lookup date for every item below). Status: first run, stops before Review 1; every boundary, kind and keep-in-fork reason in these records is a proposal until Jack decides at Review 1.

## Pins

```pins
U = 27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc  upstream/develop  2026-11-19T21:12:24Z  "[CI] Set ACPP_VISIBILITY_MASK for Metal lit tests (#2333)"
N = 0e2bac974c40554f62286cbffe1f06c87cc9caa9  naga/develop committed tip, start of naga/upstreaming  2026-11-22T21:35:36Z  "cbs: do not recompute side-effecting or memory-reading values in later sub-CFGs (AdaptiveCpp#2264)"
B = 9ec394efa584ced004d644af2194d2133e111d26  merge-base of U and N  2026-10-12T22:29:47+03:00  "[Metal] Emulate atomic64 (#2250)"
M = 350667e47d62407ad9bef5977bdab5a632ae8e87  textual merge of N into U: `git merge-tree --write-tree --name-only --no-messages U N`, exit 1 (conflicts), 6 conflicted paths
```

Re-check 2026-11-23: `git fetch upstream` left `upstream/develop` at U (0 commits beyond U).

M is a tree object only (no branch or ref). Its conflicted paths carry conflict markers (`<<<<<<< U-sha` .. `>>>>>>> N-sha`) in M.

## Divergence counts at the pins

```counts
commits ahead (in N, not in U)    = 304   (303 non-merge + 1 merge: 6a9c3a19420ff13b0316c083773f134b6d7fba25 "Merge branch 'develop' into naga/develop", 2026-10-15)
commits behind (in U, not in N)   = 22
files changed B..N                = 371
files changed B..U                = 66
files changed U..N (direct diff)  = 412
paths in the difference set       = 370   (151 M, 218 A, 1 D); +20113 / -1582 lines (git diff --numstat U M)
conflicted paths                  = 6
```

## Difference set (P2)

Definition: the paths whose content differs between U and M (`git diff --no-renames --name-status U M`), excluding `docs/naga-internal/upstreaming/` (absent at N; excluded by rule). Block format, one line per (directory, status, flag): `<status> <flag> <dir>/ <names...>`, status A/M/D, flag `C` = conflicted in the textual merge, `-` = not. Directory `./` is the repository root.

One path of B..N is absent from this set because U already carries it identically: `tests/compiler/cbs/issue2224.cpp` (see Absorbed commits).

```difference-set
M - ./ .gitignore
M C ./ CMakeLists.txt
A - ./ pixi.lock pixi.toml
M - .github/workflows/ linux-lit.yml
A - .pixi/ config.toml
A - .project/ bank.txt
A - .project/hooks/ post-commit
M - bin/ acpp
D - cmake/ FindCUDA.cmake
A - cmake/ acpp-config-merge.cmake acpp-installed-configs.cmake acpp-vendor-install.cmake acpp-vendor-rpath.cmake discovery.cmake fetch.cmake
M - cmake/ adaptivecpp-config.cmake.in hipsycl-config.cmake.in opensycl-config.cmake.in
A - cmake/discovery/ clspv.cmake common.cmake cuda.cmake hip-layout.cmake hip.cmake metal.cmake nvhpc.cmake ocl.cmake vk.cmake ze.cmake
A - cmake/install/linux/common/ clspv.cmake core.cmake cuda.cmake hip.cmake nvhpc.cmake ocl.cmake omp.cmake vk.cmake ze.cmake
A - cmake/install/linux/x86_64/ core.cmake
A - cmake/install/macos/common/ clspv.cmake omp.cmake vk.cmake
A - cmake/install/windows/common/ clspv.cmake cuda.cmake hip.cmake ocl.cmake omp.cmake ze.cmake
A - cmake/options/common/ core.cmake
A - cmake/options/linux/aarch64/ clspv.cmake core.cmake cuda.cmake hip.cmake nvhpc.cmake ocl.cmake vk.cmake ze.cmake
A - cmake/options/linux/common/ clspv.cmake core.cmake cuda.cmake hip.cmake nvhpc.cmake ocl.cmake vk.cmake ze.cmake
A - cmake/options/linux/x86_64/ clspv.cmake core.cmake cuda.cmake hip.cmake nvhpc.cmake ocl.cmake vk.cmake ze.cmake
A - cmake/options/macos/arm64/ clspv.cmake core.cmake vk.cmake
A - cmake/options/macos/common/ clspv.cmake core.cmake vk.cmake
A - cmake/options/windows/aarch64/ clspv.cmake core.cmake cuda.cmake ocl.cmake vk.cmake ze.cmake
A - cmake/options/windows/common/ clspv.cmake core.cmake cuda.cmake hip.cmake ocl.cmake vk.cmake ze.cmake
A - cmake/options/windows/x86_64/ clspv.cmake core.cmake cuda.cmake hip.cmake ocl.cmake vk.cmake ze.cmake
A - config/common/ clspv.json core.json vk.json
A - config/linux/common/ core.json cuda.json hip.json nvhpc.json ocl.json ze.json
A - config/linux/common/app/ clspv.cfg core.cfg cuda.cfg hip.cfg
A - config/linux/common/deploy/ clspv.json core.json cuda.json hip.json nvhpc.json ocl.json omp.json vk.json ze.json
A - config/linux/x86_64/ core.json
A - config/linux/x86_64/app/ core.cfg
A - config/linux/x86_64/deploy/ core.json
A - config/macos/common/ core.json
A - config/macos/common/app/ clspv.cfg core.cfg
A - config/macos/common/deploy/ clspv.json core.json metal.json omp.json vk.json
A - config/windows/common/ core.json cuda.json hip.json ocl.json ze.json
A - config/windows/common/app/ clspv.cfg core.cfg cuda.cfg hip.cfg ocl.cfg ze.cfg
A - config/windows/common/deploy/ clspv.json core.json cuda.json hip.json ocl.json omp.json vk.json ze.json
M - doc/ deployment.md env_variables.md extensions.md install-cuda.md install-llvm.md install-rocm.md installing.md runtime-spec.md using-acpp.md
A - docs/naga-internal/ 01-kernel-language.md 02-graph-api.md 03-vocabulary.md 04-implementation-research.md README.md
A - docs/naga-internal/history/ 2026-11-02-graph-api-design.md 2026-11-02-graph-api-discussion-notes.md
M - include/hipSYCL/common/ config.hpp.in small_vector.hpp
M C include/hipSYCL/common/ settings.hpp
M - include/hipSYCL/compiler/llvm-to-backend/ LLVMToBackend.hpp
M C include/hipSYCL/compiler/llvm-to-backend/ Utils.hpp
M - include/hipSYCL/compiler/llvm-to-backend/host/ LLVMToHost.hpp
M - include/hipSYCL/compiler/sscp/ DynamicFunctionSupport.hpp
A - include/hipSYCL/compiler/utils/ FunctionPointerArgument.hpp
M - include/hipSYCL/compiler/utils/ LLVMUtils.hpp ProcessFunctionAnnotationsPass.hpp
M - include/hipSYCL/glue/llvm-sscp/ fcall_specialization.hpp jit.hpp
M - include/hipSYCL/runtime/ adaptivity_engine.hpp code_object_invoker.hpp dag_manager.hpp dag_node.hpp dag_unbound_scheduler.hpp data.hpp event.hpp executor.hpp hardware.hpp hints.hpp inorder_executor.hpp inorder_queue.hpp instrumentation.hpp kernel_cache.hpp multi_queue_executor.hpp operations.hpp queue_completion_event.hpp runtime.hpp signal_channel.hpp
A - include/hipSYCL/runtime/ async_host_executor.hpp
M - include/hipSYCL/runtime/cuda/ cuda_event.hpp cuda_queue.hpp
M - include/hipSYCL/runtime/generic/ host_timestamped_event.hpp timestamp_delta_instrumentation.hpp
M - include/hipSYCL/runtime/hip/ hip_event.hpp hip_queue.hpp
M - include/hipSYCL/runtime/metal/ metal_allocator.hpp metal_backend.hpp metal_event.hpp metal_hardware_manager.hpp metal_queue.hpp
M - include/hipSYCL/runtime/omp/ omp_backend.hpp omp_code_object.hpp omp_hardware_manager.hpp
M C include/hipSYCL/runtime/omp/ omp_queue.hpp
M - include/hipSYCL/runtime/serialization/ serialization.hpp
M - include/hipSYCL/sycl/ extensions.hpp handler.hpp jit.hpp queue.hpp usm_query.hpp
M - include/hipSYCL/sycl/libkernel/sscp/ group_functions.hpp
M C include/hipSYCL/sycl/libkernel/sscp/builtins/detail/ reduction.hpp
M - include/hipSYCL/sycl/libkernel/sscp/builtins/detail/ scan_generic.hpp scan_hiplike.hpp
A - naga/tasks/ check-devices.nu check-packages.nu conda-paths.nu sanitize.cpp sanitizer-smoke.nu test-build.nu test-run.nu
A - packages/ README.md
A - packages/activations/ render-install.nu
A - packages/activations/clang/ pixi.toml
A - packages/activations/clangxx/ pixi.toml
A - packages/activations/vendor/ctng-compiler-activation/ NOTES PINNED_REF activate-g++.sh activate-gcc.sh deactivate-g++.sh deactivate-gcc.sh
A - packages/checks/consumer/ build.nu hello.cpp pixi.toml
A - packages/core/ build-lib.nu build-linux.nu fetch-sources.nu pins.toml pixi.toml write-build-cfg.nu
A - packages/core/activation/ install-cfg.nu
A - packages/core/activation/cfg/ clang++.cfg.tmpl clang-cpp.cfg.tmpl clang.cfg.tmpl
A - packages/core/licenses/ AdaptiveCpp-LICENSE llvm-LICENSE.TXT
A - packages/core/licenses/rocm/ amd_comgr-LICENSE.txt hip-LICENSE.md rocprofiler-register-LICENSE.md rocr-LICENSE.md
A - packages/core/patches/ 0012-llvm23-dllexport-double-annotation.patch 0013-llvm23-ompkinds-undef-helper-macros.patch 0014-llvm23-instrumentor-include-vfs.patch 0015-compiler-rt-eprintf-dont-define.patch 0016-llvm23-amd-missing-llvm-abi.patch 0017-compiler-rt-profile-wshadow.patch 0018-llvm23-libllvm-comgr-libstdcxx-compat.patch 0019-llvm23-openmp-ftn-upper-memspace.patch 0020-SLP-limit-second-pass-to-register-VF.patch
A - packages/extras/mutex/ pixi.toml
A - packages/extras/runtime-cuda/ pixi.toml
A - packages/extras/runtime-level-zero/ pixi.toml
A - packages/extras/runtime-ocl/ pixi.toml
A - packages/extras/runtime-rocm/ pixi.toml
A - packages/slices/compiler-rt/ pixi.toml
A - packages/slices/lldb/ pixi.toml
A - packages/slices/llvm-dev/ pixi.toml
A - packages/slices/naga-acpp/ pixi.toml
A - packages/slices/runtime/ pixi.toml
A - packages/slices/tools/ pixi.toml
M - src/ CMakeLists.txt
M - src/common/ CMakeLists.txt filesystem.cpp settings.cpp
M - src/compiler/ AdaptiveCppLlvmPasses.cpp CMakeLists.txt
M - src/compiler/cbs/ IRUtils.cpp PHIsToAllocas.cpp SubCfgFormation.cpp SyncDependenceAnalysis.cpp UniformityAnalysis.cpp VectorShapeTransformer.cpp
M - src/compiler/llvm-to-backend/ AddressSpaceInferencePass.cpp LLVMToBackend.cpp Utils.cpp
M C src/compiler/llvm-to-backend/ CMakeLists.txt
M - src/compiler/llvm-to-backend/amdgpu/ LLVMToAmdgpu.cpp
M - src/compiler/llvm-to-backend/clspv/ AddrSpaceCastRemovalPass.cpp LLVMToCLSPV.cpp
M - src/compiler/llvm-to-backend/host/ HostKernelWrapperPass.cpp LLVMToHost.cpp
M - src/compiler/llvm-to-backend/metal/ Emitter.cpp LLVMToMetal.cpp
M - src/compiler/llvm-to-backend/ptx/ LLVMToPtx.cpp
M - src/compiler/llvm-to-backend/spirv/ LLVMToSpirv.cpp
M - src/compiler/reflection/ FunctionNameExtractionPass.cpp
M - src/compiler/sscp/ AggregateArgumentExpansionPass.cpp DynamicFunctionSupport.cpp HcfRegistrationPass.cpp StdAtomicRemapperPass.cpp TargetSeparationPass.cpp
M - src/compiler/sscp/pcuda/ FreeKernelCall.cpp
M - src/libkernel/sscp/ CMakeLists.txt
M - src/runtime/ CMakeLists.txt adaptivity_engine.cpp backend.cpp backend_loader.cpp dag.cpp dag_builder.cpp dag_direct_scheduler.cpp dag_node.cpp dag_submitted_ops.cpp dag_unbound_scheduler.cpp data.cpp error.cpp inorder_executor.cpp kernel_cache.cpp kernel_configuration.cpp operations.cpp settings.cpp
A - src/runtime/ async_host_executor.cpp
M - src/runtime/cuda/ cuda_event.cpp cuda_queue.cpp
M - src/runtime/hip/ hip_event.cpp hip_queue.cpp
M - src/runtime/hw_model/ memcpy.cpp
M - src/runtime/metal/ metal_allocator.cpp metal_backend.cpp metal_code_object.cpp metal_hardware_manager.cpp metal_queue.cpp
M - src/runtime/ocl/ ocl_code_object.cpp ocl_queue.cpp
M - src/runtime/omp/ omp_backend.cpp omp_code_object.cpp omp_event.cpp omp_hardware_manager.cpp omp_queue.cpp
M - src/runtime/pcuda/ pcuda_device_topology.cpp pcuda_runtime_api.cpp
M - src/runtime/serialization/ serialization.cpp
M - src/runtime/ze/ ze_code_object.cpp
M - src/tools/acpp-appdb-tool/ CMakeLists.txt
M - src/tools/acpp-hcf-tool/ CMakeLists.txt
M - src/tools/acpp-info/ CMakeLists.txt
M - src/tools/acpp-pcuda-pp/ CMakeLists.txt
A - tests/compiler/cbs/ atomic_const_across_barrier.cpp issue2264.cpp memory_result_across_barrier.cpp split_loop_live_load.cpp
A - tests/compiler/sscp/ dynamic_function_indexed.cpp preinline_convergent.cpp
A - tests/compiler/sscp/dynamic-function-cross-tu/ cross_tu.cpp lit.local.cfg second_tu.cpp
M - tests/pcuda/ atomic.cpp
M - tests/sycl/ extensions.cpp queue.cpp
M - tests/sycl/group_functions/ group_functions_reduce.cpp group_functions_scan.cpp
```

Upstream movement behind the six conflicted paths (commits in B..U touching them, `git log B..U -- path`):

| conflicted path | upstream commits since B |
|---|---|
| `CMakeLists.txt` | 16b96a8a "[Windows] Fix deployment manifest paths (#2290)"; bea19449 "[SSCP][Windows] Avoid MSVC CRT libraries (#2269)" |
| `include/hipSYCL/common/settings.hpp` | 92f1911d "Fix parsing of string settings containing whitespace (#2316)" |
| `include/hipSYCL/compiler/llvm-to-backend/Utils.hpp` | bce257b0 "[SSCP] Hide symbols in libraries using LLVM (#2286)"; bea19449 |
| `include/hipSYCL/runtime/omp/omp_queue.hpp` | b5535e79 "[PCUDA][OpenMP] Fix PCUDA kernel launch not running in worker thread (#2263)" |
| `include/hipSYCL/sycl/libkernel/sscp/builtins/detail/reduction.hpp` | beee5ba1 "[SSCP] Fix workgroup reduction precision (#2301)" |
| `src/compiler/llvm-to-backend/CMakeLists.txt` | bce257b0; f4ff3681 "[VK] Create pass to legalize int widths for clspv (#2272)"; bea19449; d795f824 "[Metal] Add integers legalization pass (#2256)" |

## Commits only in N (grouping hints)

303 non-merge commits, `git log --no-merges B..N`. The working-theme labels are the Research Lead's grouping hints derived from subjects; they are not dispositions (two commits mix themes and are labelled by their first-listed theme: `058d037d` mixes sscp-libkernel and linked-llvm, `e38ab468` mixes sscp-libkernel and an alloca fix, `edf83999` mixes stdpar-atomics, llvm22-23 and cmake-link-deps, `819977ca` mixes vector-math, linked-llvm and relocatable-config, `4efddac5` mixes async-host and dynamic-functions, `6b99e906` mixes spirv-translator-fork and dynamic-functions). Counts per theme: relocatable-config 202, async-host 20, linked-llvm 9, pixi 9, vector-math 7, dynamic-functions 7, small-vector 6, pcuda 6, sscp-libkernel 5, llvm22-23 5, naga-docs 4, project-tooling 4, cbs 3, cmake-link-deps 3, ze-cuda-port 3, iwyu-xcode26 2, spirv-translator-fork 2, opencl 2, tool-invocation 2, kernel-cache-lock 1, convergent 1 (sum 303).

```commit-hints
## relocatable-config
395ce904 doc: drop links to removed documents
b85c0918 doc: remove fork-added documents
c90da94d Drop harness-only code and working-note comments
63eb427c r10: owned tools resolve from their real bundled location; toolchain-mode config defaults empty
c22cd245 r10: Windows pre-loads bundled vendor DLLs by full path
e97b59b8 r10: JIT tools resolve env, config, bundled, then fail clearly
e714cddb r10: vendor directories resolve bundled-first
a4fb8170 r10: settings resolvers - env, then bundled, then config
4da8ce24 r10: app config always names each vendor's bundled location
8f5f7177 r10: acpp --acpp-deploy takes a vendor list; bundled vendors always deploy
8e46c6ec r10: deploy manifest grouped per vendor
109964de r10: remove the deployment-strategy readers; manifest and vendor rpaths always generated
4726d312 r10: per-vendor ACPP_BUNDLE_<STEM> replaces the deployment strategy
01e98ddb r10: drop comments left stale by the install-prefix and harness removals
63ec862c r10: fix the HIP prefix variable name in the PCUDA header generation
42b44701 r10: remove dead HIPSYCL_INSTALL_PREFIX
7b28e5bd r10: remove devops/verify harnesses, devops/hooks and the fork pixi project
a507183c llvm-to-host: read the jit_host_* settings at JIT time instead of empty compile-time macros (the options rename left them undefined); fix source-obligations
e4abbc38 windows: llvm-spirv under bindir; harness for relative managed roots on conda layouts
3bcacc9f doc: -DCLANG_INCLUDE_PATH and clang-include-path as clang's resource directory
185335ac clang-include-path is clang's resource directory; upstream's -DCLANG_INCLUDE_PATH is honoured
fa4f61b1 discovery: CUDA honours upstream's -DCUDA_DEVICE_LIBS_PATH
497b4bf0 discovery: a Windows OpenCL import library without its DLL warns and is not found
5df85137 discovery: a found-but-incomplete vendor warns and is not found, as upstream's FindCUDA did
48903913 doc: upstream's install and usage docs describe the fork
66d33e0b doc: deployment.md describes the fork's deploy
5162cd0a doc: source obligations, the walkthrough and pixi's comments match the tree
dbac9c64 doc: the configuration model catches up with the 2026-09-26 slices
7a6d8695 toolchain mode: llc/opt/lld follow CMAKE_INSTALL_BINDIR too
bac28352 deploy: static archives stay behind; svml and amath are nonpermissive
9a4f24bc toolchain mode: libomp and clang are the prefix's own, not copies
ee6c2556 config: acpp-bindir is defined on every platform, like acpp-libdir
2f37ccea install: every destination follows GNUInstallDirs
8b556439 options: managed takes one root override per vendor, ACPP_<VENDOR>_ROOT
1d27d8e2 deploy: Windows HIP ships and deploys the SDK's own DLLs
35ae553f windows: HIP through AMD's HIP SDK (x86_64)
8f3c8184 deploy: a HIP distribution without rocm_sysdeps, or without every optional runtime library, installs and deploys
5ef7fd32 discovery: HIP accepts classic ROCm, ROCm 7.2+ and TheRock layouts
02270e72 build: upstream's legacy link lines leave the root; its -D names feed the options model
bcacdab0 build: backend enablement comes from cmake/discovery; the legacy vendor finds are gone
186e32c1 driver: the installed config defines the plugin capability keys; config keys lose the default- prefix
e7385638 doc: source obligations as implemented; the walkthrough's plan is done
259281ef doc: the configuration model, as implemented
b45beb96 doc: the configuration model, as implemented (part 4)
773bf7c3 doc: the configuration model, as implemented (part 3)
3fb10e3f doc: the configuration model, as implemented (part 2)
07c7cb41 doc: the configuration model, as implemented (part 1)
36928e07 cmake: multipass link lines add no rpath; the app builder owns it
00b81539 devops: install rules and manifest rows stay in sync; the shipped matrix
b01a06ff cmake: applications get the full strategy's install rpath from add_sycl_to_target
25ba6c77 driver: deploy from the merged manifest; the app config check
345d99a6 devops: a test for the driver's config resolution
d401853d driver: read the installed toolchain config; {{ }} resolution (part 1)
fdca58f4 runtime: Windows reaches shipped vendor DLLs through the app config
3e231e37 compiler: tool paths come from settings, not baked macros
1450b815 runtime: the app config lives beside the runtime; string settings keep spaces
db88b424 cmake: backends reach shipped vendors through $ORIGIN
80bc348f cmake: vendor install rules for every unit
c2f53661 cmake: vendor install rules (helper, cuda, omp); targets default
279f6c1a cmake: the root wires discovery, options and the installed configs
ebc22aa5 cmake: generate the installed toolchain config, manifest and app config
762c0080 cmake: one config merge, shared by the build and the harnesses
f881819b config: the toolchain config drops the dead keys; libomp-name
0331d9b2 config: the application config is a template, per unit
2393fd07 devops: goldens and strategy harness for copy-row manifests
bd6a5f3c config: manifests are copy rows only; OMP is its own vendor manifest
16dac257 cmake: two strategies; shipped vendors; ACPP_RT_LIB_DIR (part 2: harnesses)
9be9fbd4 cmake: two strategies; shipped vendors; ACPP_RT_LIB_DIR (part 1: options + harness updates in progress)
b7e8e562 doc: walkthrough — the implementation plan
d0611abc doc: walkthrough — the vendor sweep
295effaa doc: walkthrough — the opens, the variations, OMP as a vendor
46fc000b doc: the 2026-09-25 design walkthrough
2daeb9ea fix: path model review
a28925ba doc: obligations the path model creates
abc199ca doc: the path model in the configuration model
144326fb devops: the strategy flips only app-config rows
f4ed7977 devops: goldens for the path model
c8960e03 devops: harnesses assert the path model
b2437ab5 config: path model for macos
fc4d4caf config: path model for windows
aaf30b3d config: path model for common and linux
1c058587 cmake: two knobs for the remaining vendors
e33bd1e9 cmake: two knobs; per-vendor install roots and subdirs (core, cuda, hip, nvhpc, ocl)
d4f6281e cmake: discovery prefixes are common ancestors; CLANG_EXECUTABLE_PATH
51c9fff4 doc: manifests are for app deployment only; vendor install rules live in cmake
9d7a19e3 config: clang and its headers are hip's, not core's
d60d44cb doc: obligations the reflect review surfaced
479becae doc: the ownership rule
798753fd devops: core harnesses cover both build modes
8a591dc2 config: core manifests split by build mode
8b2aa085 cmake: core options follow ownership, not strategy
b4ed28c4 doc: the prolog order and the OpenCL version floor
4be591ab cmake: ocl discovery is FindOpenCL 2.1 against the machine; the header fetch follows it
23c072fb doc: the prolog and fidelity rules; opencl, level zero and omp obligations corrected
97b06bb7 core: the omp flows fold into core
b72a3e4f cmake: the OpenCL headers fetch in the prolog; ocl discovery through FindOpenCL
d66dd8e2 doc: the common tiers and the merge
2cd89422 devops: the merge harness and the tier proof
1495fa45 config: common tiers for the configuration and manifests
1617ecab cmake: common tiers for the options
4caf3ba0 doc: vk and clspv across the platform axes
49594f13 devops: vk and clspv in the platform harnesses
ccf6d0b4 config: vk and clspv configuration and manifests for the remaining platforms
cb5086fe cmake: vk and clspv options for the remaining platforms
2a17a2cf doc: vk and clspv units in the configuration model and obligations
63185ad6 devops: harnesses for the vk and clspv options and discovery
3f3ffc77 config: vk and clspv configuration and manifests
c9997926 cmake: vk and clspv discovery and options
3b532d82 compiler: llvm-to-clspv reads clspv from the configuration
02eb71b8 doc: the macos platform in the configuration model and obligations
6d7d1833 devops: harnesses for the macos arm64 options and metal discovery
ce6b2b79 config: macos arm64 configuration and manifests
a33d53c2 cmake: macos arm64 options; metal discovery; no opencl on apple
6db7ef5f doc: the windows aarch64 axis in the configuration model and obligations
26479cf2 devops: harness for the windows aarch64 options
847d5747 config: windows aarch64 configuration and manifests
23a32f8c cmake: windows aarch64 options
ca02fc5d doc: the windows platform in the configuration model and obligations
4a7f6333 devops: harness for the windows x86_64 options
0edf17b5 config: windows x86_64 configuration and manifests
54a7b24f cmake: windows x86_64 options; dll discovery for the loader-only vendors
7a507427 doc: the architecture axis in the configuration model and obligations
6e287bd3 devops: harness for the linux aarch64 options
0050200b config: linux aarch64 configuration and manifests
a156a030 cmake: linux aarch64 options
510689b8 doc: omp rows in the source obligations, the omp flows in the configuration model
28dac097 devops: harness for the omp options
27267f01 config: omp configuration
a79941ce cmake: omp options
a62c7f28 doc: ze rows in the source obligations, the ze unit in the configuration model
b8af1f22 devops: harnesses for the ze options and discovery
977912f2 config: ze configuration and manifest
d7c76bf1 cmake: ze discovery and options
54a418ec doc: ocl rows in the source obligations, the ocl unit in the configuration model
9f6fb159 devops: harnesses for the ocl options and discovery
413218a5 config: ocl configuration and manifest
55acc36f cmake: ocl discovery and options
2a14d06d doc: hip rows in the source obligations, the hip unit in the configuration model
9a0568b2 devops: harnesses for the hip options and discovery
28951b9b compiler: llvm-to-amdgpu reads the device library directory from the configuration
b6479205 config: hip configuration and manifest
f67fd917 cmake: hip discovery and options
1419023f doc: the nvhpc unit in the configuration model and its obligations
24b54fca devops: harnesses for the nvhpc options and discovery
9c99a2a4 cmake, config: the HPC SDK runtime as its own vendor unit; the nvcxx flow moves to it
447a3230 doc: cuda rows in the source obligations, the vendor unit in the configuration model
2489bffb devops: harnesses for the cuda options and discovery
34d432fe compiler: llvm-to-ptx reads the libdevice directory from the configuration
d5f2b04e config: cuda configuration and manifest
a770fd8c cmake: cuda discovery and options
a485720f doc: correct three lines the review caught
5088c2fe cmake: drop the config install rules nothing generates
9da0047a doc: the source obligations the configuration model creates
5e14320e doc: the configuration model, as settled
b598fa7d doc: delete the overhaul's design notes and ledger
941f1542 devops: delete the harnesses that tested the flat generation
b4cf5707 config: delete the flat configuration and deploy manifests
0612f964 cmake: delete the flat options, the emitter and the guards
c5fc5fea cmake: hoist discovery ahead of the options, split by build mode
08b61360 cmake: the bundled deployment strategy becomes managed
7814d684 cmake: the LLVM unit's deploy path follows the build mode
a6de2a63 cmake: the linked build puts LLVM at the toolchain root
32e6cb3d cmake: provenance is single-sided; declarations name their discovery input
78f3ede9 config: core's entries and deploy manifest to the settled schema
9a0c60d5 cmake: a resource has a toolchain side and an application side
2746c3c6 config: no vendor zone; vendor libraries deploy beside our own
76fb7a6f doc: how a deployed application finds its vendor libraries
f77cc691 cmake: discovery overrides a default, never an override
14fce81f config: core, split by platform and architecture
3e5a9fcf doc: how an application finds its configuration
595a75fb devops: cmake -P verification harnesses for the config and emitter
5c26ccda cmake: the manifest fill drops no rows
f2373749 build: drop the absolute link-path rpath, give the last two tools the origin idiom
10549e71 build: delete ACPP_HOST_FORCE_MCPU_TARGET and its compiler-flag check
99feb6b9 build: delete the switches that achieved relocatability by emptying values
a2287106 cmake: the ZE loader, clspv and hipcc discoveries move above the options
329b9c75 cmake: the default strategy expands entry references too
48cb064b cmake: ACPP_CPU_CXX records the host compiler it was built with
e77be35f cmake: the emitter copies and fills, and the source files are wired in
ba13821d cmake: the tool name entries carry the platform executable suffix
f295d9b1 cmake: the builder-overridable options as guarded plain sets
11ede2d0 config: deploy destinations are relative to the deployment root
63aafa3b config: the deploy manifests as source files
33dea987 doc: correct two entries the implementation caught
6a6d2564 config: the toolchain configuration as source files
3b08d3f2 doc: close the overhaul's last four opens
485c3ecb doc: settle the overhaul's open questions
823a8648 doc: respecify the relocatable overhaul around deployment strategies
814bd8af doc: correct the compile-definition count in the ledger
80bc9297 cmake: fail configurations this tree does not qualify
4a1b5cbf cmake: declare every configurable path in one file
6283a72c doc: specify the relocatable toolchain overhaul
b797df89 build: stop compiling in two paths that nothing reads
85c1b297 doc: ledger of path resolution and the knobs that change it
36f47bc6 sscp: record bitcode module identifiers relative to the build directory
5e6e90df build: keep build-machine paths out of installed artifacts
2a875365 cmake: let the caller configure the SPIRV translator sub-build
7c42b2e5 cmake: allow the SPIRV translator to be pinned to a commit
6a8e3df1 Revert "cmake: allow a local SPIRV-LLVM-Translator source directory"
83c05f44 acpp: make the backend library directories overridable
39ae2b1b cmake: allow the configuration files to omit environment paths
02415454 cmake: allow a local SPIRV-LLVM-Translator source directory
## vector-math
e97197c8 Android: Linux options model, no libmvec, upstream JIT tool-path names
e3128903 r10: vector math defaults come from the root detection block (x86 Linux svml, aarch64 libmvec, macOS/Windows none)
f3b50f3c compiler: the vector math default is not a build machine's to make
4521fc8b compiler: the vector math library is chosen when an application runs
085171d9 host jit: ask the loader for libmvec, and nothing else
fb0f3e2f host jit: the libmvec loader lookup is glibc-only, guard it as such
4b2a12ed host jit: locate libmvec through the dynamic loader, link by SONAME
## async-host
96cf135f ocl: revert host-task deferral; drop the async_host parking test (SYCL work inside async_host bodies is not supported)
a2e2e236 tests: async_host dependent test submits its kernel behind the dependent; skip backends without deferral
4efddac5 ocl: defer operations that depend on host tasks instead of parking the queue; sscp/reflection: resolve function pointers forwarded through constructors (Darwin C1->C2); regression tests for both
1b335c2a dag_node: initialise execution lane and index
3b9930f7 runtime: queue::wait and async_host wait for deferred submissions on in-order queues
263b06d4 tests: async_host free deadlock and retargeted dependency on a held-back kernel
ecec188c runtime: only queues that can defer a submission follow unsubmitted requirements by deferring
b04d8349 runtime: defer requirements not issued to a backend queue (async_host) on CUDA and HIP
c292f241 hip: defer submission on foreign requirements, and use hipLaunchHostFunc
eee9932b cuda: defer submission on any incomplete foreign requirement
f9d3657f doc: document async_host, and test it
53df214f sycl: add async_host to queue and handler
642c4880 runtime: run host operations on threads owned by the runtime
de36f0f9 doc: specify when an operation is issued, and where errors surface
302cdcd0 tests: retarget to the host device with an allocation the host can reach
03045049 runtime: issue an operation after its requirements where a backend needs it
ebd5dda4 cuda: use cudaLaunchHostFunc instead of cudaStreamAddCallback
1346acea doc: document the retarget context, and correct the API reference
4ca7c773 sycl: give a retargeted command group an executor for its device
3c84beeb sycl: resolve a retargeted command group against the retarget context
## llvm22-23
f78b88b9 llvm-to-backend: raw_ostream takes no std::endl
edf83999 stdpar: convert float/pointer atomics to iN for SSCP builtins; spirv: drop SPV_INTEL_fast_composite on LLVM 23; cmake: find_dependency(Threads) in package configs
a750ab07 compiler: port src/compiler to LLVM 23 (AMD 0bace190), no deprecated API
6cd0e257 LLVM 23: accept in version gate
c488b6c1 Support LLVM 22
## linked-llvm
819977ca Default libmvec on Linux; deploy the LLVM DLL on Windows; Windows libomp app dir; stale comments
058d037d libkernel: use the backends sub-group id/count in work-group reduce and scan; llvm-to-backend: link the LLVM DLL on Windows
4c1aa400 compiler: define LLVM_ADAPTIVECPP_LINK_INTO_TOOLS on the object library
50b49ce4 runtime: link libomp from the runtimes sub-build's own binary dir
d5d79e91 compiler: export getAdaptiveCppPluginInfo from the LLVM DLL on Windows when linked into tools
ea48e343 runtime: give rt-backend-omp an omp.h-only include dir (never the clang resource dir)
c5584b49 runtime: depend on the outer-build target `openmp`, not `omp`
969913ec runtime: build rt-backend-omp against the libomp built by this LLVM build
0ccbc11a discovery: the linked-mode guard reads LLVM_ADAPTIVECPP_LINK_INTO_TOOLS
## dynamic-functions
0d2a1ef4 sscp: dynamic-function host handling: exact annotation matching, weak (ELF/Mach-O) and weak_odr+alternatename (COFF) stand-ins, per-stand-in tags against ICF; tests
825a146b sscp: on COFF, dynamic-function dummies get their own name and /alternatename (real definitions win); test a cross-TU definition
664cadf9 sscp: put host-side dynamic-function placeholders in a COMDAT on COFF; test with a second TU
6964145a reflection: sanitize registered function names on MSVC ABI to match the device module
cf576880 sscp/reflection: match lifetime markers by intrinsic id
6c45e6aa sscp/reflection: resolve function-pointer arguments through unoptimised parameter spills
f815ee1e jit: indexed call sequences; link dynamic-function definitions from other TUs
## cbs
0e2bac97 cbs: do not recompute side-effecting or memory-reading values in later sub-CFGs (AdaptiveCpp#2264)
22422eee cbs: restore split-loop live values from their work-item state array
681c6c5e [CBS] Fix loop-carried state repair in split nested loops (#2274)
## sscp-libkernel
e38ab468 round 7: keep alloca align/size in address-space and aggregate passes; race-free inclusive-first group scans; bounded wg_reduce
b8bc54f4 [SSCP] spirv: declare OpenCL C 2.0 so OpenCL runtimes read generic atomics with 2.0 builtins
a0972d64 tests: fractional and negative float min/max group reductions and scans
1a678052 [SSCP] min/max identities match sycl::known_identity
362b077d [SSCP] wg_reduce: broadcast the result as its own type, not as integer bits through float memory
## iwyu-xcode26
3f96b940 runtime/metal: include what you use (Xcode 26 libc++)
f8cf4da5 runtime: include what you use (Xcode 26 libc++)
## small-vector
479d1d0b Reapply "common: auto_small_vector is a plain std::vector under libc++ too"
ec80ae12 Reapply "common: plain std::vector for small_vector under libc++"
eff8d5bb Revert "common: plain std::vector for small_vector under libc++"
3e92d027 Revert "common: auto_small_vector is a plain std::vector under libc++ too"
1b6ea024 common: auto_small_vector is a plain std::vector under libc++ too
a189a1fd common: plain std::vector for small_vector under libc++
## pcuda
933ba248 pcuda: launch MSVC-mangled kernels by their sanitized name
40c79d96 pcuda: fail cleanly when a kernel's info is not found
038a3b60 Revert "pcuda: temporary trace of the shared-allocation path"
ceece56d pcuda: temporary trace of the shared-allocation path
efdef0fa pcuda: null-check backend/allocator lookups; fix 0-based device ids
9d4573e7 tests/pcuda: rename atomic test cases that shadowed pcuda's atomic functions under MSVC delayed template parsing
## opencl
779f6400 ocl: treat a failed USM prefetch as an ignored hint (Intel CPU OpenCL returns CL_INVALID_VALUE)
5106a326 ocl: build SPIR-V with -cl-std from the device's OpenCL C versions so runtimes use 2.0+ builtins
## convergent
3fc30594 llvm-to-backend: propagate convergent before pre-inline simplification
## kernel-cache-lock
b32bce5b runtime: build code objects outside the kernel cache lock
## tool-invocation
fcd14e4c r10: drop the JIT tool invocation timeout
3f1ba463 llvm-to-backend: time out tool invocations and suppress Windows error dialogs so failures surface as errors
## ze-cuda-port
7b89cf09 cuda: pass the toolkit include subdir in the CUDA cxx flags
a33fbb1f ze: find the Level Zero loader library portably
42835dcb ze: use <cstdint> instead of glibc's internal <bits/stdint-uintn.h>
## cmake-link-deps
5962a7f8 llvm-to-backend: link the dynamic loader library
37f65935 llvm-to-backend: call try_retrieve_settings_variable through common::settings
8b7855e9 runtime: make the threads dependency a usage requirement
## pixi
6d19f58f check envs: rely on the system bash; drop unused helper
a8bf3663 packages: split the toolchain into pixi-build-naga packages (linux-64)
68b26dd5 naga/build: install LLVM utils, no flang links, drop unversioned .so symlinks
7b113b83 pixi tests: build and run the AdaptiveCpp suite (omp, cuda, ocl, ze) plus sanitizer smoke
842e15dc pixi workspace: local naga-acpp-local build (rattler-build, linux-64)
6b99e906 org rename: SPIRV-LLVM-Translator from NAGALabsOrg; fix comment quoting [skip ci]
2ade4976 pixi: one verify task runs every harness; the in-tree build tasks are gone
06db7796 build: add Boost and tasks for the test suite
e4fe5446 build: add a pixi source package for development
## spirv-translator-fork
38ac1833 llvm-to-backend: clone the SPIR-V translator with core.longpaths=true
7dbf3a60 llvm-spirv: fetch the NAGA translator fork (AMD LLVM 23 base)
## naga-docs
03fa09b8 docs(naga-internal): correct the progress-model claim in the research document
40ac91a8 docs(naga-internal): implementation research findings and proposals
d96a9b7c docs(naga-internal): graph cycle and conditional nodes
32359d14 docs(naga-internal): kernel language spec, graph API and vocabulary
## project-tooling
32aa1547 project bank: acpp-fork uses the acpp-fork-v2 bank (concise extraction)
2fc5ef40 devops: ignore python bytecode cache; drop accidentally committed .pyc
d44fda11 project bank: set up acpp-fork
47d18287 devops: retain each commit into the acpp-fork bank
```

## Absorbed commits (P3)

Test used: `git cherry U N` marks a commit with `-` when an upstream commit in U has the same patch-id; independently, each of the 303 commits was reverse-applied to U (`git diff-tree -p --binary C | git apply --cached --reverse --check`, index only) to find commits whose whole effect is already in U.

```absorbed
681c6c5e0abbdb20adbf0308bf4fc880ff6e98d5 == 31db2e83903c1db9f50b03b620eb0a09224cd0eb  patch-id 4bafb826d8059c029c0e27fef06942549652aad4 (git patch-id --stable)  "[CBS] Fix loop-carried state repair in split nested loops (#2274)", upstream commit dated 2026-10-16
```

Consequence: `tests/compiler/cbs/issue2224.cpp` and the `SubCfgFormation.cpp` hunks of that commit are in U, so they are not in the difference set.

Commits whose reverse-apply to U also succeeds but which no upstream commit carries (they only delete or revert content that exists nowhere in U, so there is nothing for U to contain): `6a8e3df1` (Revert "cmake: allow a local SPIRV-LLVM-Translator source directory"), `b4cf5707` (config: delete the flat configuration and deploy manifests), `941f1542` (devops: delete the harnesses that tested the flat generation), `b598fa7d` (doc: delete the overhaul's design notes and ledger), `eff8d5bb` (Revert "common: plain std::vector for small_vector under libc++"), `b85c0918` (doc: remove fork-added documents). Not absorbed; listed so the count 7 of the reverse-apply test is accounted for.

N-only commit that refers to upstream work not in U: `0e2bac97` fixes upstream issue AdaptiveCpp#2264 ("atomic_ref fetch_add returns incorrect result with OMP backend", state OPEN at lookup, https://github.com/AdaptiveCpp/AdaptiveCpp/issues/2264); no upstream commit carries this fix.

## Overlapping upstream work (P3)

Queries run at the lookup date against `AdaptiveCpp/AdaptiveCpp`: `gh pr list --state open --limit 300 --json number,title,headRefName,baseRefName,updatedAt,isDraft,url,files,changedFiles,author` (87 open PRs, all base `develop`) and `git for-each-ref refs/remotes/upstream` after `git fetch upstream --prune` (87 branches besides `develop` and `HEAD`). Overlap is by path: a PR or branch is listed when its changed files include a path of the difference set. Granularity is the path, not the hunk, so a listed item may touch other hunks of the same file. PR link form: `https://github.com/AdaptiveCpp/AdaptiveCpp/pull/<n>`. #1478 changes 132 files but the API returned 100, so its list may be incomplete. 37 open PRs do not overlap and are not listed.

```open-prs-overlapping
#2326 open upd=2026-11-18 @will-saunders-ukaea "Inline keywords and anonymous namespaces." :: include/hipSYCL/sycl/libkernel/sscp/builtins/detail/reduction.hpp
#2323 open upd=2026-11-16 @Oblomov "OpenCL profiling" :: src/runtime/ocl/ocl_queue.cpp
#2307 open upd=2026-11-11 @amavliut "[SSCP] Mark wrappers of convergent builtins as convergent before ..." :: src/compiler/llvm-to-backend/LLVMToBackend.cpp src/compiler/sscp/TargetSeparationPass.cpp tests/compiler/sscp/preinline_convergent.cpp
#2305 open upd=2026-11-05 @amavliut "[CBS] Fix values carried across barriers in loops" :: src/compiler/cbs/SubCfgFormation.cpp
#2304 open upd=2026-11-10 @bezmi "Fix std:: math remapping gaps for the CUDA and HIP JIT" :: include/hipSYCL/compiler/llvm-to-backend/Utils.hpp src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp src/compiler/llvm-to-backend/ptx/LLVMToPtx.cpp
#2297 open upd=2026-11-09 @amavliut "[SSCP][OpenMP] Sub-group sizes > 1 on the host through hierarchical ..." :: doc/env_variables.md include/hipSYCL/compiler/llvm-to-backend/host/LLVMToHost.hpp include/hipSYCL/runtime/omp/omp_hardware_manager.hpp include/hipSYCL/sycl/libkernel/sscp/builtins/detail/reduction.hpp src/compiler/cbs/IRUtils.cpp src/compiler/cbs/SubCfgFormation.cpp src/compiler/llvm-to-backend/LLVMToBackend.cpp src/compiler/llvm-to-backend/host/LLVMToHost.cpp src/compiler/sscp/TargetSeparationPass.cpp src/runtime/omp/omp_hardware_manager.cpp src/runtime/omp/omp_queue.cpp tests/compiler/sscp/preinline_convergent.cpp
#2293 open upd=2026-11-05 @1sgtpepper "[NFC][doc] Document missing ROCm device-library troubleshooting" :: doc/install-rocm.md
#2291 open upd=2026-11-05 @amavliut "[CBS] Pin only the innermost local id as contiguous in the ..." :: src/compiler/cbs/SubCfgFormation.cpp
#2289 open upd=2026-10-27 @EwanC "[VK] Add backend interop for native objects and events" :: include/hipSYCL/sycl/extensions.hpp tests/sycl/extensions.cpp
#2284 open upd=2026-11-18 @carbotaniuman "[CBS] Fix OpenMP load after barrier" :: src/compiler/cbs/SubCfgFormation.cpp
#2283 open upd=2026-11-05 @sbalint98 "Remove HIPSYCL_INSTALL_PREFIX" :: include/hipSYCL/common/config.hpp.in src/common/CMakeLists.txt src/common/filesystem.cpp src/runtime/CMakeLists.txt src/runtime/backend_loader.cpp
#2282 open upd=2026-11-19 @sbalint98 "[SSCP] Fix known identity for group algorithms" :: include/hipSYCL/sycl/libkernel/sscp/group_functions.hpp tests/sycl/group_functions/group_functions_reduce.cpp
#2270 open upd=2026-11-12 @tneval "[SSCP|SPIR-V] Use JIT reflection to optimize lowering of ..." :: src/runtime/ocl/ocl_queue.cpp
#2268 open upd=2026-10-21 @sbalint98 "Use install prefix from cmake CLI when installing llvm-spirv" :: src/compiler/llvm-to-backend/CMakeLists.txt
#2259 open upd=2026-11-18 @Arthur031221 "Fix acpp dry-run filter dropping -I/-D values given as separate args" :: bin/acpp
#2242 open upd=2026-10-20 @resetius "[CI] Do not run workflows twice for in-repo pull requests" :: .github/workflows/linux-lit.yml
#2232 open upd=2026-11-09 @nilsfriess "Fix runtime destruction issues due to allocation map by ..." :: include/hipSYCL/runtime/runtime.hpp
#2211 open upd=2026-09-09 @sbalint98 "[CI] Fix Docker image building for PRs and cleanup llvm.sh ..." :: .github/workflows/linux-lit.yml
#2200 draft upd=2026-08-25 @koparasy "[SSCP][prototype] Implement sycl_khr_free_function_kernels" :: include/hipSYCL/runtime/kernel_cache.hpp include/hipSYCL/sycl/extensions.hpp include/hipSYCL/sycl/handler.hpp src/runtime/kernel_cache.cpp
#2170 draft upd=2026-07-28 @sbalint98 "Proof of concept for using clang tooling to expose all amdgcn ..." :: src/compiler/CMakeLists.txt src/compiler/llvm-to-backend/CMakeLists.txt src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp src/tools/acpp-hcf-tool/CMakeLists.txt
#2166 draft upd=2026-07-29 @sbalint98 "Implement infrastructure to support LLVM intrinsics for the SSCP ..." :: src/compiler/CMakeLists.txt src/compiler/llvm-to-backend/CMakeLists.txt src/compiler/llvm-to-backend/LLVMToBackend.cpp src/compiler/sscp/TargetSeparationPass.cpp src/tools/acpp-hcf-tool/CMakeLists.txt
#2160 open upd=2026-10-19 @sbalint98 "Set proper rpath property for tools" :: src/compiler/llvm-to-backend/CMakeLists.txt src/tools/acpp-hcf-tool/CMakeLists.txt
#2131 open upd=2026-08-04 @tneval "Propagate CMAKE_INSTALL_RPATH to llvm-spirv configuration" :: src/compiler/llvm-to-backend/CMakeLists.txt
#2108 open upd=2026-11-05 @illuhad "Bump version to 26.10" :: CMakeLists.txt
#2038 open upd=2026-06-17 @EkremA "Device compatibility for exception throwing code using assert" :: src/compiler/AdaptiveCppLlvmPasses.cpp src/compiler/CMakeLists.txt src/compiler/sscp/TargetSeparationPass.cpp
#1986 open upd=2026-11-19 @al42and "Support LLVM 22" :: .github/workflows/linux-lit.yml CMakeLists.txt doc/install-llvm.md doc/installing.md src/compiler/AdaptiveCppLlvmPasses.cpp src/compiler/llvm-to-backend/AddressSpaceInferencePass.cpp src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp src/compiler/llvm-to-backend/host/LLVMToHost.cpp src/compiler/llvm-to-backend/metal/Emitter.cpp src/compiler/llvm-to-backend/metal/LLVMToMetal.cpp src/compiler/llvm-to-backend/ptx/LLVMToPtx.cpp src/compiler/sscp/HcfRegistrationPass.cpp
#1948 draft upd=2026-04-25 @EwanC "[Prototype] cl_khr_unified_svm implementation of SYCL SVM" :: src/runtime/CMakeLists.txt
#1946 open upd=2026-11-05 @jabruniessner "Add tracing framework" :: CMakeLists.txt include/hipSYCL/sycl/handler.hpp include/hipSYCL/sycl/queue.hpp src/runtime/CMakeLists.txt src/runtime/dag_node.cpp
#1943 draft upd=2025-12-19 @sbalint98 "Implement GROMACS specific builtins for the SSCP flow" :: src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp
#1937 draft upd=2025-12-03 @illuhad "[SSCP][llvm-to-ptx] DO-NOT-MERGE Cap targeted PTX and SM version ..." :: include/hipSYCL/compiler/llvm-to-backend/Utils.hpp src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp src/compiler/llvm-to-backend/ptx/LLVMToPtx.cpp
#1916 open upd=2025-09-25 @MarkusBuettner "Add checks for zero-sized nd_range kernels" :: include/hipSYCL/sycl/handler.hpp
#1884 open upd=2025-08-06 @thomasRoglin "KnownGroupSize type should be size_t instead of int" :: include/hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp src/compiler/llvm-to-backend/LLVMToBackend.cpp src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp src/compiler/llvm-to-backend/host/HostKernelWrapperPass.cpp src/compiler/llvm-to-backend/ptx/LLVMToPtx.cpp src/compiler/llvm-to-backend/spirv/LLVMToSpirv.cpp
#1796 draft upd=2025-05-06 @sbalint98 "RFC: First approach to add target specific intriniscs for gfx" :: src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp
#1778 open upd=2025-04-19 @KipHamiltons "Fix Apple builds and update the docs for them." :: doc/installing.md src/runtime/CMakeLists.txt
#1777 open upd=2025-04-21 @KipHamiltons "Add note about homebrew distribution to installation docs" :: doc/installing.md
#1670 draft upd=2024-12-23 @nullr0ute "Allow to override the default install locations" :: CMakeLists.txt src/common/CMakeLists.txt src/compiler/CMakeLists.txt src/compiler/llvm-to-backend/CMakeLists.txt src/libkernel/sscp/CMakeLists.txt src/runtime/CMakeLists.txt src/tools/acpp-appdb-tool/CMakeLists.txt src/tools/acpp-hcf-tool/CMakeLists.txt src/tools/acpp-info/CMakeLists.txt
#1562 open upd=2024-08-10 @SidneyCogdill "Remove AdaptiveCpp header directories from implicit include directories" :: cmake/adaptivecpp-config.cmake.in
#1533 open upd=2024-08-06 @normallytangent "[CI] Release AdaptiveCpp Packages" :: CMakeLists.txt
#1478 draft upd=2025-08-28 @keldu "Reworking the includes in the hipSYCL include folder to relate ..." :: include/hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp include/hipSYCL/compiler/llvm-to-backend/Utils.hpp include/hipSYCL/glue/llvm-sscp/jit.hpp include/hipSYCL/runtime/adaptivity_engine.hpp include/hipSYCL/runtime/code_object_invoker.hpp include/hipSYCL/runtime/cuda/cuda_queue.hpp include/hipSYCL/runtime/dag_node.hpp include/hipSYCL/runtime/data.hpp include/hipSYCL/runtime/generic/host_timestamped_event.hpp include/hipSYCL/runtime/inorder_executor.hpp include/hipSYCL/runtime/kernel_cache.hpp include/hipSYCL/runtime/multi_queue_executor.hpp include/hipSYCL/runtime/omp/omp_code_object.hpp include/hipSYCL/runtime/operations.hpp include/hipSYCL/runtime/queue_completion_event.hpp include/hipSYCL/runtime/serialization/serialization.hpp include/hipSYCL/sycl/handler.hpp
#1362 open upd=2024-02-15 @fodinabor "[CMake] Add ACPP_HAS_* values to CMake integration." :: cmake/adaptivecpp-config.cmake.in cmake/hipsycl-config.cmake.in cmake/opensycl-config.cmake.in
#1342 open upd=2024-02-13 @nilsfriess "Add test case for #1276" :: bin/acpp
#1234 open upd=2023-11-14 @nilsfriess "[SYCL2020] Make vec class `constexpr`-constructible" :: .github/workflows/linux-lit.yml
#1095 open upd=2024-04-07 @fxzjshm "Add SSCP MUSA backend" :: CMakeLists.txt src/CMakeLists.txt src/compiler/llvm-to-backend/CMakeLists.txt src/libkernel/sscp/CMakeLists.txt src/runtime/CMakeLists.txt src/runtime/backend_loader.cpp src/runtime/settings.cpp
#1094 open upd=2024-07-31 @ThijsWithaar "Enable CPack" :: CMakeLists.txt src/compiler/llvm-to-backend/CMakeLists.txt
#966 open upd=2023-04-06 @DieGoldeneEnte "rewrite of group algorithms" :: tests/sycl/group_functions/group_functions_reduce.cpp tests/sycl/group_functions/group_functions_scan.cpp
#912 open upd=2023-01-18 @illuhad "[SSCP] Add timing debug output for S2 compilation" :: src/runtime/cuda/cuda_queue.cpp src/runtime/hip/hip_queue.cpp
#807 open upd=2022-08-10 @fodinabor "Refine ROCm clang detection." :: CMakeLists.txt
#778 open upd=2022-11-23 @DieGoldeneEnte "implement known_identity" :: tests/sycl/group_functions/group_functions_scan.cpp
#618 open upd=2022-02-07 @al42and "Minor improvements around sycl::backend" :: src/runtime/serialization/serialization.cpp src/runtime/settings.cpp
#505 open upd=2021-03-20 @fodinabor "Add retrieving CPU name for OpenMP host device." :: src/runtime/omp/omp_hardware_manager.cpp
```

Upstream branches that carry commits not in U (`git rev-list U..branch` > 0) and whose changes since `merge-base(U, branch)` include a path of the difference set: 47 of the 87. Of the other 40: 9 are fully merged into U and 31 carry unmerged commits that touch no difference-set path. A branch whose work was squash-merged still shows as unmerged here. Branch link form: `https://github.com/AdaptiveCpp/AdaptiveCpp/tree/<name>`. Format: `<name> ahead=<commits not in U> last=<date of tip> overlap=<number of difference-set paths touched>`.

```upstream-branches-overlapping
free-function-poc ahead=3 last=2026-10-30 overlap=6
feature/bump-version-26.06 ahead=2 last=2026-10-16 overlap=1
revert-1972-feature/khr-prefetch-host ahead=1 last=2026-09-25 overlap=3
feature/khr-prefetch-host ahead=3 last=2026-09-24 overlap=3
feature/auto-pch ahead=1 last=2026-09-24 overlap=1
sbalint98/fix-llvmsh-ci-2 ahead=4 last=2026-09-09 overlap=1
hotfix/al2-128bit-types ahead=2 last=2026-09-07 overlap=1
hotfix/acpp-treat-headers-as-source ahead=1 last=2026-08-27 overlap=1
rocm7.2-llvm22-support ahead=1 last=2026-08-21 overlap=1
exception_to_assert ahead=16 last=2026-06-05 overlap=3
feature/acpp-module-file-endings ahead=1 last=2026-05-18 overlap=1
llvm22-support ahead=2 last=2026-03-02 overlap=4
feature/sscp-cap-ptx-sm-version ahead=1 last=2025-12-03 overlap=3
ci/llvm ahead=1 last=2025-11-20 overlap=2
ci/macos-sysroot ahead=3 last=2025-10-10 overlap=1
empty_kernel_submission ahead=8 last=2025-05-27 overlap=1
feature/flush ahead=1 last=2025-05-23 overlap=8
test/win-ocl ahead=2 last=2025-05-22 overlap=1
doc/clarify-llvm-versions ahead=1 last=2025-04-22 overlap=2
ci/link-into-llvm ahead=2 last=2025-04-14 overlap=1
fix/cbs-alloca-widening ahead=3 last=2025-03-31 overlap=1
experimental/pcfeature ahead=69 last=2025-03-15 overlap=28
feature/hostslm ahead=7 last=2025-02-20 overlap=13
issue1713/tina ahead=3 last=2025-02-16 overlap=2
make-boost-optional ahead=3 last=2025-02-09 overlap=1
individual-adaptivity-levels ahead=1 last=2025-01-15 overlap=1
manually-set-pic-pie ahead=1 last=2025-01-01 overlap=1
cleanup-higher-dim-ids ahead=2 last=2024-12-18 overlap=1
feature/use-group-algorithms-in-scan ahead=15 last=2024-12-17 overlap=8
feature/scan-chunked-v2 ahead=15 last=2024-12-02 overlap=2
feature/scan-chunked ahead=13 last=2024-11-27 overlap=1
feature/enable-sscp-macos ahead=2 last=2024-11-14 overlap=2
macos-ci-dbg2 ahead=6 last=2024-07-03 overlap=2
hotfix/fix-ocl-header-commits ahead=1 last=2024-06-19 overlap=1
ci-debugging-branch ahead=11 last=2024-05-23 overlap=2
macos-ci-dbg ahead=4 last=2024-05-10 overlap=2
feature/cmake-feature-flags ahead=2 last=2024-02-15 overlap=3
experimental/jit-cache ahead=3 last=2024-01-07 overlap=5
try-more-robust-cmake-rocm-search ahead=1 last=2023-12-21 overlap=1
feature/sync-elision-ignore-stack-load-stores ahead=1 last=2023-12-04 overlap=1
feature/object-pool ahead=5 last=2023-06-01 overlap=4
feature/selective-pointer-wrapping ahead=6 last=2023-01-25 overlap=1
feature/sscp-jit-timer ahead=3 last=2023-01-18 overlap=2
feature/generic-llvm-sscp-amdgpu ahead=118 last=2022-11-30 overlap=31
feature/flexible-filesystem-fallback ahead=1 last=2022-08-02 overlap=1
feature/pointer-future ahead=3 last=2022-07-03 overlap=4
sycl/1.2.1 ahead=4 last=2020-08-21 overlap=5
```

## Upstream context (X1)

All items looked up 2026-11-22. Quotes are verbatim (hard line wraps removed).

**Release schedule and feature freeze** (source: `CONTRIBUTING.md` at U, lines 21-30, file last changed by 1c81b7f7 on 2026-10-12):
- Cadence: "AdaptiveCpp follows a regular release schedule with 4-month cadence: `year.02` (released at the end of February), `year.06` (released at the end of June), `year.10` (released at the end of October)".
- Freeze rule, last week: "In the last week prior to release, no new features should be added and focus should be on bug fixes."
- Freeze rule, last two weeks: "In the last two weeks prior to release, new features should only be added for components of the stack that are still under heavy development (e.g. the SSCP compiler or stdpar support)."
- Observed releases (source: `git tag --sort=-creatordate` and `gh release list -R AdaptiveCpp/AdaptiveCpp`): v25.10.0 2025-11-05 (latest), v25.02.0 2025-05-09, v24.10.0 2024-12-20, v24.06.0 2024-07-29, v24.02.0 2024-03-11, v23.10.0 2023-10-30. No tag exists after v25.10.0; `CMakeLists.txt` at U still sets `ACPP_VERSION_MAJOR 25` / `ACPP_VERSION_MINOR 10` (lines 15-16); PR #2108 "Bump version to 26.10" (illuhad, opened 2026-06-23) is open. The nominal slots 26.02, 26.06 and 26.10 have passed without a release.
- Next release date: not announced in any source found. The next nominal slot after the lookup date is the end of February 2027 (`year.02`), by cadence only.
- Milestone (source: `gh api repos/AdaptiveCpp/AdaptiveCpp/milestones/7`, lookup 2026-11-23): the milestone `26.10.0` is open with 13 open and 8 closed items, no due date, updated 2026-11-19T20:05:03Z; PRs #2273, #2282, #2307 and #1986 carry it. This disagrees with the cadence above (the end-of-October 2026 slot passed without a tag); both are recorded and not reconciled.

**Sequencing expectations** (source: `CONTRIBUTING.md` at U):
- Line 3: "When in doubt about how to implement something or how to integrate with the overall project strategy, please just open an issue for discussion."
- Line 19: "Please files PRs against the `develop` branch."
- Lines 32-37: AI-assisted contributions are accepted if the contributor takes responsibility; "Please do **not** include AI tools in your commit history. Every commit should be wholly owned by humans."

**Maintainer statements** (source: https://github.com/AdaptiveCpp/AdaptiveCpp/discussions/1981 "The SYCL onboarding problem — and an attempt to fix it", opened by jackm97 on 2026-02-27). Each quote was checked verbatim against the comment body at the link on the lookup date.
- illuhad, 2026-02-27, https://github.com/AdaptiveCpp/AdaptiveCpp/discussions/1981#discussioncomment-15948251: "What would definitely be great is if we could offer binaries similarly to oneAPI which users can just download and run."
- same comment: "Since AdaptiveCpp has runtime-loaded backends, the way we've always envisioned this is to have a core binary (core components + CPU backend) and then optional packages for the various backends, such that dependencies to complex stacks like ROCm are isolated."
- jackm97, 2026-09-04, https://github.com/AdaptiveCpp/AdaptiveCpp/discussions/1981#discussioncomment-18292426 (an offer, not a maintainer statement): "If there's interest for setting up `pixi build` upstream so users can simply run `pixi run install` to build locally and perform a global user install, let me know. I can submit a PR that adds the pixi manifest".
- illuhad, 2026-09-09, https://github.com/AdaptiveCpp/AdaptiveCpp/discussions/1981#discussioncomment-18370984, reply in that thread: "What infrastructure is needed for the packaging? If it's just github, would you be willing to upstream?"
- same comment: "I think OpenCL is sufficient for Intel; it's currently better tested and optimized than the Level Zero backend anyway."

**Stance by kind** (what the sources above do and do not say):
- bug fix, performance, other (feature, documentation, test, refactor, tooling/CI, vendored third-party): no known maintainer statement beyond CONTRIBUTING's PR-against-develop and open-an-issue-when-in-doubt rules. Stance: unknown.
- build-system rework: no known maintainer statement. Stance: unknown. (Open upstream work on install prefixes and rpaths is listed above: #2283, #2268, #2160, #2131, #1670.)
- pixi workspace: illuhad invited upstreaming of the packaging work on 2026-09-09 ("would you be willing to upstream?") and in February described binary distribution with a core package plus optional backend packages as the envisioned model. No reply to the `pixi build` manifest offer was found in the thread at lookup. Stance: interest in packaging; pixi manifests specifically unanswered.
- LLVM-version support: no maintainer statement found. `CMakeLists.txt` at U states the policy in code: line 481 "LLVM versions greater than 21 are not yet tested/supported, use -DACPP_EXPERIMENTAL_LLVM=ON if you still wish to try your luck." and line 486 "This LLVM seems to be a development version; only LLVM releases are tested and supported. Use -DACPP_EXPERIMENTAL_LLVM=ON if you still wish to try your luck." Open PR #1986 "Support LLVM 22" (al42and, opened 2026-03-02) is the upstream work on the next version. Stance: unknown beyond that policy text.
