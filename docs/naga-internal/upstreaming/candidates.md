# Upstreaming triage: candidates

Pins: U `27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc`, N `0e2bac974c40554f62286cbffe1f06c87cc9caa9`, B `9ec394efa584ced004d644af2194d2133e111d26`; merge tree M `350667e47d62407ad9bef5977bdab5a632ae8e87` (see baseline.md). Status: first run, before Review 1. Every candidate, boundary, kind and flag below is a proposal; open questions are the unresolved items U01-U12 in decisions.md.

How to read the records:
- A candidate owns paths and hunks of the difference set (baseline.md). Hunk coordinates are new-file `start,count` of `git diff --no-renames -U0 U M -- <path>`. In an assignment block, `paths` are whole paths (`dir/**` = every difference-set path under `dir`), `hunks` are explicit hunks of a split path, and `rest` marks the candidate that owns every hunk of that path not listed elsewhere. A path appears either whole or split, never both.
- Assignment basis: each hunk of a modified path went to the working theme of the N-only commits that last touched its lines (`git blame` on a scratch commit whose tree is M, parents U and N); deletion-only hunks by pickaxe on the deleted lines (45 hunks), or by adjacency to the neighbouring hunk (16 hunks, listed in decisions.md U01). Added paths went by the themes of the commits that touched them.
- Size is the diff of the owned paths and hunks, `git diff --numstat U M`: files, lines added / removed. A split path counts as a file of each owner.
- Kinds are the plan's six; `other` carries a subtype. "secondary" lists other kinds present in the change.
- `requires` edges carry one line of evidence as `M:<path>:<line>` (the line in the tree M) or a command. `pending` names an upstream-pending state. `seam` names changes this candidate relies on that are in a keep-in-fork decision or in an unresolved hunk.
- `applicability` is the trial merge of the candidate and its prerequisites onto U (method in the section at the end).
- `open upstream PRs by path` lists open PRs (baseline.md) whose files include a path the candidate owns; it is path-level and is input for the conversation criteria, not a dependency.
- Conversation flag, horizon and next step are decided after Review 1 (units C1, O1, N1) and are not set here.

## Candidate index

| id | title | kind | files | lines + / - | large |
|---|---|---|---|---|---|
| C01 | Relocatable toolchain configuration | build-system rework | 173 | +6630 / -1206 | yes |
| C02 | async_host and deferred issue of operations with foreign requirements | other: feature | 29 | +1251 / -54 | yes |
| C03 | Support LLVM 22 and 23 | LLVM-version support | 23 | +224 / -58 | no |
| C04 | Linked-into-LLVM build fixes | build-system rework | 4 | +121 / -2 | no |
| C05 | Dynamic functions across translation units and object formats | other: feature | 17 | +966 / -37 | no |
| C06 | CBS: values carried across barriers and split loops | bug fix | 5 | +409 / -9 | no |
| C07 | SSCP group reduce and scan correctness | bug fix | 6 | +326 / -86 | no |
| C08 | Keep alloca alignment and size in address-space and aggregate passes | bug fix | 3 | +20 / -3 | no |
| C09 | Include what you use for Xcode 26 libc++ | bug fix | 53 | +106 / -3 | no |
| C10 | small_vector as std::vector under libc++ | bug fix | 1 | +15 / -0 | no |
| C11 | pcuda robustness and MSVC kernel names | bug fix | 4 | +85 / -34 | no |
| C12 | OpenCL: SPIR-V built for the device's OpenCL C version | bug fix | 2 | +81 / -0 | no |
| C13 | OpenCL: failed USM prefetch is an ignored hint | bug fix | 1 | +16 / -4 | no |
| C14 | stdpar: float and pointer atomics through integer builtins | bug fix | 1 | +31 / -17 | no |
| C15 | SSCP: propagate convergent before pre-inline simplification | bug fix | 2 | +139 / -0 | no |
| C16 | Kernel cache: build code objects outside the cache lock | performance | 1 | +84 / -50 | no |
| C17 | JIT tool invocations: exit-code messages and Windows error dialogs | bug fix | 7 | +28 / -15 | no |
| C18 | Level Zero: include `<cstdint>` instead of a glibc-internal header | bug fix | 1 | +1 / -1 | no |
| C19 | Threads as a usage requirement and in package configs | build-system rework | 4 | +20 / -3 | no |
| C20 | pixi workspace and packages | pixi workspace | 50 | +7553 / -0 | yes |

Totals check: file counts add up to 387 for the candidates plus 20 for keep-in-fork = 407 entries, because the 30 split paths count once per owner (340 whole paths + 67 owner entries of split paths); lines add up with the keep-in-fork decisions (keep-in-fork.md: +2007 / -0) to the difference set's +20113 / -1582.

## C01 Relocatable toolchain configuration

- title: Relocatable toolchain configuration: option and discovery model, install layout, driver and run-time configuration resolution
- intent: Replace baked-in build-machine paths with a layered options, discovery and installed-configuration model (GNUInstallDirs install layout, `$ORIGIN` rpaths, an installed toolchain, application and deploy configuration read by the `acpp` driver and the runtime, run-time resolution of the LLVM tools and of the host JIT's vector math library), so that an installed AdaptiveCpp can be relocated and its vendor libraries bundled or deployed.
- kind: build-system rework
- secondary: other: documentation (doc/ edits for the model), other: refactor (removal of the legacy per-vendor finds and flat option emitter)
- size: files 173, lines +6630 / -1206
- large: yes. Proposed split (not decided), each part also a coherent intent: (a) options, discovery and configuration merge (`cmake/options/**`, `cmake/discovery/**`, `cmake/discovery.cmake`, `cmake/acpp-config-merge.cmake`, `config/**`, root `CMakeLists.txt` wiring); (b) install layout, vendor install rules and rpaths (`cmake/install/**`, `cmake/acpp-vendor-*.cmake`, `cmake/acpp-installed-configs.cmake`, install/rpath hunks of the `src/**/CMakeLists.txt` files and `src/tools/**`); (c) driver configuration and deploy (`bin/acpp`, `doc/` edits); (d) run-time settings and tool resolution (`include/hipSYCL/common/settings.hpp`, `src/common/**`, `src/runtime/backend_loader.cpp`, `Utils.cpp` and the `llvm-to-*` tool-path hunks, `LLVMToHost` vector math). Platform axes (Windows, macOS, Android) and vendor units (HIP SDK, NVHPC, Vulkan, clspv) are spread through (a)-(d).
- merged cycle: the host JIT vector-math change (default `libmvec`, loader lookup, `jit_host_*` run-time settings; working theme `vector-math`) and this candidate depend on each other, so they are one candidate. Evidence: `M:CMakeLists.txt:542` sets `DEFAULT_VEC_MATH_LIB`, consumed at `M:cmake/options/linux/common/core.cmake:129` (`set(ACPP_VECTOR_MATH_LIB "${DEFAULT_VEC_MATH_LIB}")`); the JIT consumes the installed key written from `M:config/common/core.json:90` (`"@ACPP_VECTOR_MATH_LIB@"`) via `M:src/compiler/llvm-to-backend/host/LLVMToHost.cpp:320` (`try_retrieve_settings_variable`).
- requires: none
- pending: maintainer decision on the configuration model itself: it replaces upstream's install-prefix and rpath handling, on which open upstream PRs #2283, #2268, #2160, #2131 and #1670 (draft) also work (baseline.md, overlapping PRs). Needs a maintainer answer before the PR; no unmerged upstream change is needed in U.
- seams: C01 owns two unresolved hunks that carry a fork-identity line: `M:src/compiler/llvm-to-backend/CMakeLists.txt:272` is `GIT_REPOSITORY https://github.com/NAGALabsOrg/SPIRV-LLVM-Translator` (U has `AdaptiveCpp/SPIRV-LLVM-Translator`, `U:src/compiler/llvm-to-backend/CMakeLists.txt:207`), in hunk `@268,7`; `@244,15` defines `LLVMSPIRV_GIT_TAG`/`LLVMSPIRV_GIT_SHALLOW` (`M:...CMakeLists.txt:250-254`). Resolution: rework needed on this candidate (restore the upstream repository URL before offering), unless Jack moves the fork-URL lines to keep-in-fork at Review 1 (unresolved U03).
- applicability: conflicting files: `CMakeLists.txt`, `include/hipSYCL/common/settings.hpp`, `src/compiler/llvm-to-backend/CMakeLists.txt`. Their conflict hunks overlap upstream commits 16b96a8a/bea19449 (Windows manifest and CRT), 92f1911d (settings whitespace parsing), bce257b0/f4ff3681/d795f824 (see baseline.md).
- open upstream PRs by path: #2304 #2297 #2293 #2283 #2268 #2259 #2170 #2166 #2160 #2131 #2108 #2038 #1986 #1948 #1946 #1943 #1937 #1884 #1796 #1778 #1777 #1670 #1562 #1533 #1362 #1342 #1095 #1094 #807

```assignment C01
paths
  bin/**
  cmake/discovery/**
  cmake/install/**
  cmake/options/**
  config/**
  src/common/**
  src/libkernel/**
  src/tools/**
  cmake/FindCUDA.cmake
  cmake/acpp-config-merge.cmake
  cmake/acpp-installed-configs.cmake
  cmake/acpp-vendor-install.cmake
  cmake/acpp-vendor-rpath.cmake
  cmake/discovery.cmake
  cmake/fetch.cmake
  doc/deployment.md
  doc/env_variables.md
  doc/install-cuda.md
  doc/install-rocm.md
  doc/using-acpp.md
  include/hipSYCL/common/config.hpp.in
  include/hipSYCL/common/settings.hpp
  include/hipSYCL/compiler/llvm-to-backend/host/LLVMToHost.hpp
  src/CMakeLists.txt
hunks
  doc/install-llvm.md @94,1
  src/compiler/CMakeLists.txt @19,3 @27,1 @229,3
  src/compiler/llvm-to-backend/spirv/LLVMToSpirv.cpp @20,1 @354,9 @364,8
  src/runtime/CMakeLists.txt @117,3 @136,3 @157,9 @187,4 @209,9 @229,3 @250,7 @272,3 @277,3 @296,8 @358,7 @521,0 @549,8 @607,3
rest
  CMakeLists.txt
  cmake/adaptivecpp-config.cmake.in
  cmake/hipsycl-config.cmake.in
  cmake/opensycl-config.cmake.in
  doc/installing.md
  src/compiler/llvm-to-backend/CMakeLists.txt
  src/compiler/llvm-to-backend/Utils.cpp
  src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp
  src/compiler/llvm-to-backend/clspv/LLVMToCLSPV.cpp
  src/compiler/llvm-to-backend/host/LLVMToHost.cpp
  src/compiler/llvm-to-backend/ptx/LLVMToPtx.cpp
  src/runtime/backend_loader.cpp
```

## C02 async_host and deferred issue of operations with foreign requirements

- title: async_host and deferred issue of operations whose requirements belong to another backend queue
- intent: Let a queue run host operations on threads owned by the runtime (`async_host`), and issue an operation to a backend queue only after its requirements from other queues or host tasks have completed (host-function based deferral on CUDA and HIP, waiting in in-order queues), with retargeted command groups resolved against a retarget context.
- kind: other: feature
- secondary: bug fix (deadlocks fixed along the way: `1b335c2a`, `263b06d4`), other: documentation (doc/extensions.md, doc/runtime-spec.md), other: test
- size: files 29, lines +1251 / -54
- large: yes (Research Lead's proposal). Proposed split: (a) deferral on foreign requirements (`include/hipSYCL/runtime/{cuda,hip}/**`, `src/runtime/{cuda,hip}/**`, `inorder_executor`, `inorder_queue.hpp`, `operations.hpp`, `event.hpp`, `doc/runtime-spec.md`); (b) the `async_host` extension (`async_host_executor.*`, `sycl/{queue,handler,extensions}.hpp`, `doc/extensions.md`, tests); (c) the retarget context (`hints.hpp`, `usm_query.hpp`, handler/queue retarget hunks).
- requires: none
- pending: none. Open upstream PRs by path overlap (below) but none is needed.
- seams: none
- applicability: clean
- open upstream PRs by path: #2289 #2283 #2232 #2200 #1948 #1946 #1916 #1778 #1670 #1478 #1095 #912 #618

```assignment C02
paths
  include/hipSYCL/runtime/cuda/**
  include/hipSYCL/runtime/hip/**
  src/runtime/cuda/**
  src/runtime/hip/**
  doc/runtime-spec.md
  include/hipSYCL/runtime/async_host_executor.hpp
  include/hipSYCL/runtime/event.hpp
  include/hipSYCL/runtime/executor.hpp
  include/hipSYCL/runtime/hints.hpp
  include/hipSYCL/runtime/inorder_queue.hpp
  include/hipSYCL/runtime/operations.hpp
  include/hipSYCL/runtime/runtime.hpp
  include/hipSYCL/sycl/extensions.hpp
  include/hipSYCL/sycl/handler.hpp
  include/hipSYCL/sycl/queue.hpp
  include/hipSYCL/sycl/usm_query.hpp
  src/runtime/async_host_executor.cpp
hunks
  src/runtime/CMakeLists.txt @36,1
rest
  doc/extensions.md
  include/hipSYCL/runtime/inorder_executor.hpp
  src/runtime/dag_direct_scheduler.cpp
  src/runtime/dag_node.cpp
  src/runtime/inorder_executor.cpp
  src/runtime/serialization/serialization.cpp
  tests/sycl/extensions.cpp
```

## C03 Support LLVM 22 and 23

- title: Support LLVM 22 and 23
- intent: Build and run AdaptiveCpp against LLVM 22 and LLVM 23: version-gate bump, LLVM API changes in the compiler passes and backends (`LLVMUtils.hpp` compatibility helpers, `createBr`, `getFirstNonPHI` and related), SPIR-V `SPV_INTEL_fast_composite` handling, CI matrix entry for clang 22, installation documentation.
- kind: LLVM-version support
- secondary: none
- size: files 23, lines +224 / -58
- large: no
- requires: none
- pending: two. (1) Maintainer decision on LLVM 23: U's gate refuses development versions (`U:CMakeLists.txt:481` "LLVM versions greater than 21 are not yet tested/supported" and `:486` "This LLVM seems to be a development version; only LLVM releases are tested and supported"), and this candidate raises the supported maximum to 23 (`M:CMakeLists.txt:490`), where the fork builds against an AMD snapshot (`packages/core/pins.toml`, commit 0bace190). (2) Open upstream PR #1986 "Support LLVM 22" (al42and, updated 2026-11-19) changes 12 of the same paths; whichever lands second must rebase.
- seams: none
- applicability: clean
- open upstream PRs by path: #2307 #2305 #2304 #2297 #2291 #2284 #2242 #2211 #2170 #2166 #2108 #2038 #1986 #1946 #1943 #1937 #1884 #1796 #1778 #1777 #1670 #1533 #1234 #1095 #1094 #807

```assignment C03
paths
  .github/**
  src/compiler/llvm-to-backend/metal/**
  include/hipSYCL/compiler/utils/LLVMUtils.hpp
  src/compiler/cbs/IRUtils.cpp
  src/compiler/cbs/PHIsToAllocas.cpp
  src/compiler/cbs/SyncDependenceAnalysis.cpp
  src/compiler/cbs/UniformityAnalysis.cpp
  src/compiler/cbs/VectorShapeTransformer.cpp
  src/compiler/llvm-to-backend/host/HostKernelWrapperPass.cpp
  src/compiler/sscp/HcfRegistrationPass.cpp
  src/compiler/sscp/TargetSeparationPass.cpp
hunks
  CMakeLists.txt @490,2
  doc/installing.md @55,3 @59,2
  src/compiler/AdaptiveCppLlvmPasses.cpp @42,1 @44,3
  src/compiler/llvm-to-backend/AddressSpaceInferencePass.cpp @160,4
  src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp @207,5
  src/compiler/llvm-to-backend/host/LLVMToHost.cpp @34,1 @351,1 @353,1
  src/compiler/llvm-to-backend/ptx/LLVMToPtx.cpp @95,4 @241,1 @243,1
  src/compiler/llvm-to-backend/spirv/LLVMToSpirv.cpp @57,9 @78,1
  src/compiler/sscp/pcuda/FreeKernelCall.cpp @12,2 @190,1 @199,1 @270,1
rest
  doc/install-llvm.md
  src/compiler/cbs/SubCfgFormation.cpp
```

## C04 Linked-into-LLVM build fixes

- title: Linked-into-LLVM build fixes: plugin export, object-library definition, LLVM DLL linking, libomp from the runtimes build
- intent: Make AdaptiveCpp built as an LLVM external project (`LLVM_ADAPTIVECPP_LINK_INTO_TOOLS`) work when LLVM is a DLL on Windows and when openmp is a runtime of the LLVM build: export `getAdaptiveCppPluginInfo` from the LLVM DLL, define the macro on the object library, link the LLVM DLL and match its `/Zc:dllexportInlines-` flag, and build `rt-backend-omp` against the libomp the same LLVM build produces.
- kind: build-system rework
- secondary: none
- size: files 4, lines +121 / -2
- large: no
- requires: none. Checked: every `ACPP_*` and `LLVM_*` name used in the libomp hunk (`M:src/runtime/CMakeLists.txt:376-458`) is defined in U (`ACPP_LLVM_COMPONENT`, `U:CMakeLists.txt:81-84`) or in the hunk itself (`ACPP_BUILDTREE_LIBOMP_DIR`, `M:src/runtime/CMakeLists.txt:411`); `LLVM_ADAPTIVECPP_LINK_INTO_TOOLS` exists at U (`U:CMakeLists.txt:82-89`).
- pending: none
- seams: none. Disposition is contested (unresolved U09): the use case is the fork's own build pipeline.
- applicability: clean
- open upstream PRs by path: #2283 #2268 #2170 #2166 #2160 #2131 #2038 #1986 #1948 #1946 #1778 #1670 #1095 #1094

```assignment C04
hunks
  src/compiler/llvm-to-backend/CMakeLists.txt @58,5 @67,7 @131,7
rest
  src/compiler/AdaptiveCppLlvmPasses.cpp
  src/compiler/CMakeLists.txt
  src/runtime/CMakeLists.txt
```

## C05 Dynamic functions across translation units and object formats

- title: Dynamic functions: definitions in other translation units, indexed call sequences, host stand-ins on ELF, Mach-O and COFF, function-pointer argument reflection
- intent: Make the dynamic-function extension work when a definition lives in another translation unit or is selected by index, give host-side stand-ins distinct, ICF-safe identities on every object format, and resolve function-pointer arguments and lifetime markers in the reflection pass.
- kind: other: feature
- secondary: bug fix (COFF stand-ins, MSVC name sanitising), other: test, other: documentation (doc/extensions.md)
- size: files 17, lines +966 / -37
- large: no
- requires: none. Checked: the hunks use no helper added by C03 (`LLVMUtils.hpp` helpers are not referenced from them).
- pending: none
- seams: none
- applicability: clean
- open upstream PRs by path: #2307 #2297 #2289 #2166 #1884 #1478

```assignment C05
paths
  include/hipSYCL/compiler/sscp/**
  include/hipSYCL/glue/**
  src/compiler/reflection/**
  tests/compiler/sscp/dynamic-function-cross-tu/**
  include/hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp
  include/hipSYCL/compiler/utils/FunctionPointerArgument.hpp
  include/hipSYCL/compiler/utils/ProcessFunctionAnnotationsPass.hpp
  include/hipSYCL/sycl/jit.hpp
  src/compiler/sscp/DynamicFunctionSupport.cpp
  tests/compiler/sscp/dynamic_function_indexed.cpp
  tests/sycl/queue.cpp
hunks
  doc/extensions.md @393,25
  tests/sycl/extensions.cpp @1529,110
rest
  src/compiler/llvm-to-backend/LLVMToBackend.cpp
```

## C06 CBS: values carried across barriers and split loops

- title: CBS: do not recompute side-effecting or memory-reading values in later sub-CFGs; restore split-loop live values
- intent: Fix wrong results from the OpenMP CBS transformation when a side-effecting or memory-reading value is used after a barrier (an atomic fetch_add in a loop, loads across barriers, live values of split loops).
- kind: bug fix
- secondary: other: test (four lit tests under `tests/compiler/cbs/`)
- size: files 5, lines +409 / -9
- large: no
- requires: none
- pending: none. Commit `0e2bac97` fixes upstream issue #2264 (open at lookup); the earlier `681c6c5e` is already in U and is not part of this candidate. Open upstream PRs #2305, #2297, #2291 and #2284 change `SubCfgFormation.cpp` too (path overlap).
- seams: none
- applicability: clean
- open upstream PRs by path: #2305 #2297 #2291 #2284

```assignment C06
paths
  tests/compiler/cbs/**
hunks
  src/compiler/cbs/SubCfgFormation.cpp @662,4 @679,3 @704,2 @1140,6 @1148,6
```

## C07 SSCP group reduce and scan correctness

- title: SSCP group reduce and scan: known identities, wg_reduce broadcast type, race-free scans, sub-group id and count
- intent: Correct SSCP group reductions and scans: min/max identities match `sycl::known_identity`, `wg_reduce` broadcasts the result as its own type instead of integer bits through float memory, inclusive-first scans are race-free, and the backend's sub-group id and count are used; with tests for fractional and negative float min/max.
- kind: bug fix
- secondary: other: test
- size: files 6, lines +326 / -86
- large: no
- requires: none
- pending: none. Open upstream PR #2282 "[SSCP] Fix known identity for group algorithms" (sbalint98, updated 2026-11-19) changes `group_functions.hpp` and `group_functions_reduce.cpp`, which this candidate also changes (same intent for the identities); U's commit beee5ba1 (#2301) changed `reduction.hpp`, which conflicts with this candidate.
- seams: none
- applicability: conflicting files: `include/hipSYCL/sycl/libkernel/sscp/builtins/detail/reduction.hpp` (upstream beee5ba1 "[SSCP] Fix workgroup reduction precision (#2301)")
- open upstream PRs by path: #2326 #2297 #2282 #966 #778

```assignment C07
paths
  include/hipSYCL/sycl/libkernel/**
  tests/sycl/group_functions/**
```

## C08 Keep alloca alignment and size in address-space and aggregate passes

- title: Keep alloca alignment and size when rewriting address spaces and expanding aggregate arguments
- intent: Preserve the alignment and size of allocas in the address-space inference, clspv address-space-cast removal and aggregate-argument expansion passes.
- kind: bug fix
- secondary: none
- size: files 3, lines +20 / -3
- large: no
- requires: none
- pending: none
- seams: none
- applicability: clean
- open upstream PRs by path: #1986

```assignment C08
paths
  src/compiler/llvm-to-backend/clspv/AddrSpaceCastRemovalPass.cpp
  src/compiler/sscp/AggregateArgumentExpansionPass.cpp
rest
  src/compiler/llvm-to-backend/AddressSpaceInferencePass.cpp
```

## C09 Include what you use for Xcode 26 libc++

- title: Add missing standard includes for Xcode 26 libc++ in the runtime and the Metal backend
- intent: Add the direct includes that code in `include/hipSYCL/runtime`, `src/runtime` and the Metal backend relies on transitively, so the runtime builds with the libc++ of Xcode 26.
- kind: bug fix (a build failure; see unresolved U11 on the kind)
- secondary: none
- size: files 53, lines +106 / -3
- large: no
- requires: none
- pending: none
- seams: none
- applicability: conflicting files: `include/hipSYCL/runtime/omp/omp_queue.hpp` (upstream b5535e79 "[PCUDA][OpenMP] Fix PCUDA kernel launch not running in worker thread (#2263)" changes the include block)
- open upstream PRs by path: #2297 #2283 #2200 #1946 #1478 #1095 #618 #505

```assignment C09
paths
  include/hipSYCL/runtime/generic/**
  include/hipSYCL/runtime/metal/**
  include/hipSYCL/runtime/omp/**
  include/hipSYCL/runtime/serialization/**
  src/runtime/hw_model/**
  src/runtime/metal/**
  src/runtime/omp/**
  include/hipSYCL/runtime/adaptivity_engine.hpp
  include/hipSYCL/runtime/code_object_invoker.hpp
  include/hipSYCL/runtime/dag_manager.hpp
  include/hipSYCL/runtime/dag_node.hpp
  include/hipSYCL/runtime/dag_unbound_scheduler.hpp
  include/hipSYCL/runtime/data.hpp
  include/hipSYCL/runtime/hardware.hpp
  include/hipSYCL/runtime/instrumentation.hpp
  include/hipSYCL/runtime/multi_queue_executor.hpp
  include/hipSYCL/runtime/queue_completion_event.hpp
  include/hipSYCL/runtime/signal_channel.hpp
  src/runtime/adaptivity_engine.cpp
  src/runtime/backend.cpp
  src/runtime/dag.cpp
  src/runtime/dag_builder.cpp
  src/runtime/dag_submitted_ops.cpp
  src/runtime/dag_unbound_scheduler.cpp
  src/runtime/data.cpp
  src/runtime/error.cpp
  src/runtime/kernel_cache.cpp
  src/runtime/kernel_configuration.cpp
  src/runtime/operations.cpp
  src/runtime/settings.cpp
hunks
  include/hipSYCL/runtime/inorder_executor.hpp @23,1
  include/hipSYCL/runtime/kernel_cache.hpp @29,1
  src/runtime/backend_loader.cpp @31,2
  src/runtime/dag_direct_scheduler.cpp @25,3
  src/runtime/dag_node.cpp @20,3
  src/runtime/inorder_executor.cpp @21,1 @390,0
  src/runtime/serialization/serialization.cpp @20,1
```

## C10 small_vector as std::vector under libc++

- title: `small_vector` and `auto_small_vector` are plain `std::vector` under libc++
- intent: Avoid the double free at runtime teardown on macOS caused by sbo's allocator keeping its small buffer inside the allocator object, by using `std::vector` for both aliases when `_LIBCPP_VERSION` is defined.
- kind: bug fix
- secondary: none
- size: files 1, lines +15 / -0
- large: no
- requires: none
- pending: unclear whether still needed: U's commit 96c7d17a "[small_vector] Work around container implementation of libc++ 22 (#2273)" (2026-11-18) changes `include/hipSYCL/common/sbo/small_vector.hpp` for libc++ 22. Unresolved U05.
- seams: none
- applicability: clean
- open upstream PRs by path: none

```assignment C10
paths
  include/hipSYCL/common/small_vector.hpp
```

## C11 pcuda robustness and MSVC kernel names

- title: pcuda: null-check backend and allocator lookups, fail cleanly on missing kernel info, launch MSVC-mangled kernels by their sanitised name, 0-based device ids
- intent: Make pcuda report errors instead of crashing (null lookups, missing kernel information), use 0-based device ids, launch kernels whose MSVC-mangled names were sanitised, and rename test cases that shadowed pcuda's atomic functions under MSVC.
- kind: bug fix
- secondary: other: test (`tests/pcuda/atomic.cpp`)
- size: files 4, lines +85 / -34
- large: no
- requires: none
- pending: none
- seams: none
- applicability: clean
- open upstream PRs by path: none

```assignment C11
paths
  src/runtime/pcuda/**
  tests/pcuda/**
rest
  src/compiler/sscp/pcuda/FreeKernelCall.cpp
```

## C12 OpenCL: SPIR-V built for the device's OpenCL C version

- title: OpenCL: build SPIR-V with the device's OpenCL C version and declare OpenCL C 2.0 for generic atomics
- intent: Pass `-cl-std` derived from the device's OpenCL C versions when building SPIR-V and declare OpenCL C 2.0 in the generated module, so OpenCL runtimes read generic atomics with the 2.0 builtins.
- kind: bug fix
- secondary: none
- size: files 2, lines +81 / -0
- large: no
- requires: none
- pending: none
- seams: none
- applicability: clean
- open upstream PRs by path: #1884

```assignment C12
paths
  src/runtime/ocl/ocl_code_object.cpp
rest
  src/compiler/llvm-to-backend/spirv/LLVMToSpirv.cpp
```

## C13 OpenCL: failed USM prefetch is an ignored hint

- title: OpenCL: a failed USM prefetch is ignored with a warning
- intent: Treat a failed `clEnqueueMigrateMemINTEL` prefetch as an ignorable hint (Intel CPU OpenCL returns `CL_INVALID_VALUE`), keeping a marker event so the operation still completes.
- kind: bug fix
- secondary: none
- size: files 1, lines +16 / -4
- large: no
- requires: none
- pending: none
- seams: none
- applicability: clean
- open upstream PRs by path: #2323 #2270

```assignment C13
paths
  src/runtime/ocl/ocl_queue.cpp
```

## C14 stdpar: float and pointer atomics through integer builtins

- title: stdpar: convert float and pointer atomics to integer atomics for the SSCP builtins
- intent: Remap `std::atomic` operations on floating-point and pointer types in the stdpar SSCP flow to the integer builtins, which are the only ones SSCP provides.
- kind: bug fix
- secondary: none
- size: files 1, lines +31 / -17
- large: no
- requires: none
- pending: none
- seams: none
- applicability: clean
- open upstream PRs by path: none

```assignment C14
paths
  src/compiler/sscp/StdAtomicRemapperPass.cpp
```

## C15 SSCP: propagate convergent before pre-inline simplification

- title: SSCP: propagate `convergent` to wrappers before the pre-inline simplification
- intent: Mark wrappers of convergent builtins convergent before the pre-inline simplification pipeline so it cannot move or duplicate them across control flow; with a lit test.
- kind: bug fix
- secondary: other: test
- size: files 2, lines +139 / -0
- large: no
- requires: none
- pending: open upstream PR #2307 "[SSCP] Mark wrappers of convergent builtins as convergent before ..." (amavliut, updated 2026-11-11) changes `LLVMToBackend.cpp` and adds `tests/compiler/sscp/preinline_convergent.cpp`, the same file this candidate adds; it may be the same fix. Overlap, not a requirement; to be examined before opening a PR.
- seams: none
- applicability: clean
- open upstream PRs by path: #2307 #2297 #2166 #1884

```assignment C15
paths
  tests/compiler/sscp/preinline_convergent.cpp
hunks
  src/compiler/llvm-to-backend/LLVMToBackend.cpp @36,2 @65,46 @420,1
```

## C16 Kernel cache: build code objects outside the cache lock

- title: Kernel cache: build code objects outside the cache lock
- intent: Compile and load code objects without holding the kernel cache's lock, with futures for concurrent requests for the same object.
- kind: performance
- secondary: bug fix (lock held across JIT compilation); see unresolved U11 on the kind
- size: files 1, lines +84 / -50
- large: no
- requires: none
- pending: none
- seams: none
- applicability: clean
- open upstream PRs by path: #2200 #1478

```assignment C16
rest
  include/hipSYCL/runtime/kernel_cache.hpp
```

## C17 JIT tool invocations: exit-code messages and Windows error dialogs

- title: JIT tool invocations: report the exit code of a failed tool and suppress Windows error dialogs in child processes
- intent: Make a failing `opt`, `llc`, `lld`, `clang`, `clspv` or `llvm-spirv` invocation report its exit code, and prevent a child that cannot load a DLL from blocking the JIT on a modal Windows error dialog.
- kind: bug fix
- secondary: none
- size: files 7, lines +28 / -15
- large: no
- requires: C01 (symbol, via an entangled hunk, see unresolved U02). Evidence: `suppressChildErrorDialogs` and `describeToolExitCode` are defined at `M:src/compiler/llvm-to-backend/Utils.cpp:58` and `:73` inside hunk `@53,58`, which is assigned to C01 (it also holds C01's tool resolution); this candidate calls them at `Utils.cpp:266`, `:275`, `LLVMToAmdgpu.cpp:360`, `:549`, `LLVMToCLSPV.cpp:359`, `LLVMToHost.cpp:560`, `:568`, `:576`, `LLVMToPtx.cpp:222`, `:260`, `LLVMToSpirv.cpp:399`; the declaration `M:include/hipSYCL/compiler/llvm-to-backend/Utils.hpp:406` is owned here.
- pending: none
- seams: none
- applicability (with C01): conflicting files: `CMakeLists.txt`, `include/hipSYCL/common/settings.hpp`, `include/hipSYCL/compiler/llvm-to-backend/Utils.hpp`, `src/compiler/llvm-to-backend/CMakeLists.txt`. `Utils.hpp` is a whole-path of this candidate and carries the conflict with upstream bce257b0 (`ACPP_DEFAULT_VISIBILITY` on the declarations).
- open upstream PRs by path: #2304 #2297 #2170 #1986 #1943 #1937 #1884 #1796 #1478

```assignment C17
paths
  include/hipSYCL/compiler/llvm-to-backend/Utils.hpp
hunks
  src/compiler/llvm-to-backend/Utils.cpp @19,2 @266,1 @275,1
  src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp @360,1 @549,1
  src/compiler/llvm-to-backend/clspv/LLVMToCLSPV.cpp @359,1
  src/compiler/llvm-to-backend/host/LLVMToHost.cpp @560,1 @568,1 @576,1
  src/compiler/llvm-to-backend/ptx/LLVMToPtx.cpp @222,1 @260,1
  src/compiler/llvm-to-backend/spirv/LLVMToSpirv.cpp @399,1
```

## C18 Level Zero: include `<cstdint>`

- title: Level Zero: include `<cstdint>` instead of glibc's internal `<bits/stdint-uintn.h>`
- intent: Remove the dependence of `ze_code_object.cpp` on a glibc-internal header.
- kind: bug fix
- secondary: none
- size: files 1, lines +1 / -1
- large: no
- requires: none
- pending: none
- seams: none
- applicability: clean
- open upstream PRs by path: none

```assignment C18
paths
  src/runtime/ze/**
```

## C19 Threads as a usage requirement and in package configs

- title: Declare Threads as a public usage requirement of `acpp-rt` and as a dependency of the installed package configs
- intent: Link `Threads::Threads` publicly from `acpp-rt` (and name it explicitly for `rt-backend-omp`), and call `find_dependency(Threads)` in the installed `adaptivecpp`, `hipsycl` and `opensycl` package configs, so consumers get the imported target.
- kind: build-system rework
- secondary: bug fix (consumers of the installed config lacked `Threads::Threads`); see unresolved U11
- size: files 4, lines +20 / -3
- large: no
- requires: none. Checked and ruled out: C02's `async_host_executor.cpp` uses `std::thread` (`M:src/runtime/async_host_executor.cpp:12`, `:68`) but links `Threads::Threads` already at U (`U:src/runtime/CMakeLists.txt:64`, PRIVATE), so C02 does not need this change.
- pending: none
- seams: none
- applicability: clean
- open upstream PRs by path: #2283 #1948 #1946 #1778 #1670 #1562 #1362 #1095

```assignment C19
hunks
  cmake/adaptivecpp-config.cmake.in @9,3
  cmake/hipsycl-config.cmake.in @14,3
  cmake/opensycl-config.cmake.in @11,3
  src/runtime/CMakeLists.txt @65,7 @532,4
```

## C20 pixi workspace and packages

- title: pixi workspace: pixi-build packages for the toolchain, their build scripts, environments and tasks
- intent: Provide a pixi workspace that builds the AdaptiveCpp toolchain from source as conda packages (core build, slices, runtime vendor extras, compiler activation, consumer check), with tasks to build and run the test suite and sanitizer smoke tests against the installed environment.
- kind: pixi workspace
- secondary: other: vendored third-party (`packages/activations/vendor/ctng-compiler-activation`, conda-forge's compiler activation scripts, pinned by `PINNED_REF`), other: tooling/CI (`naga/tasks`, `.pixi/config.toml`)
- size: files 50, lines +7553 / -0 (of which `pixi.lock` is +3824)
- large: yes. Proposed split (not decided): (a) workspace manifest, lock and tasks (`pixi.toml`, `pixi.lock`, `naga/tasks/**`, `.pixi/config.toml`, `.gitignore` hunk); (b) package slices and vendor extras (`packages/slices/**`, `packages/extras/**`, `packages/checks/**`); (c) core build and compiler activation (`packages/core/**` except KF3, `packages/activations/**`).
- requires: C01, C03, C04. Evidence: C01: `M:packages/core/build-lib.nu:41-44` passes `-DACPP_CUDA_ROOT=.`, `-DACPP_OCL_ROOT=.`, `-DACPP_ZE_ROOT=.`, `-DACPP_LIBOMP_ROOT=lib`, options that exist only in the model of `M:cmake/options/common/core.cmake:221-231` (`ACPP_<STEM>_ROOT`). C03: `M:packages/core/build-lib.nu:52-53` ("this build is LLVM 23 (AMD snapshot)", `-DACPP_EXPERIMENTAL_LLVM=ON`). C04: `M:packages/core/build-lib.nu:27` (`-DLLVM_ENABLE_RUNTIMES=compiler-rt;openmp`) and `:37` (`-DLLVM_ADAPTIVECPP_LINK_INTO_TOOLS=ON`), handled by `M:src/runtime/CMakeLists.txt:378` (`"openmp" IN_LIST LLVM_ENABLE_RUNTIMES`).
- pending: maintainer decision: whether upstream wants a pixi manifest. Discussion #1981: Jack offered to submit it on 2026-09-04 and illuhad replied on 2026-09-09 asking what infrastructure the packaging needs and whether it could be upstreamed; no answer on the `pixi build` offer was found (baseline.md, maintainer statements).
- seams: relies on keep-in-fork decision KF3 (patches to the pinned LLVM snapshot and `pins.toml`). Evidence: `M:packages/core/fetch-sources.nu:70` opens `pins.toml` and `:71` names the `patches` directory. Resolution: rework needed (make the source pins and patch set an input of the package instead of files in `packages/core`), unless Jack moves KF3 into this candidate at Review 1. The package and channel names (`naga-acpp-*`, channel `https://prefix.dev/jackm97/naga-labs`, backend `pixi-build-naga`) are fork identity: contested, fallback keep-in-fork (unresolved U07).
- applicability (with C01, C03, C04): conflicting files: `CMakeLists.txt`, `include/hipSYCL/common/settings.hpp`, `src/compiler/llvm-to-backend/CMakeLists.txt` (all from C01)
- open upstream PRs by path: none

```assignment C20
paths
  .pixi/**
  naga/**
  packages/activations/**
  packages/checks/**
  packages/core/activation/**
  packages/core/licenses/**
  packages/extras/**
  packages/slices/**
  packages/README.md
  packages/core/build-lib.nu
  packages/core/build-linux.nu
  packages/core/fetch-sources.nu
  packages/core/pixi.toml
  packages/core/write-build-cfg.nu
  pixi.lock
  pixi.toml
hunks
  .gitignore @4,2
```

## Dependency map (D1)

Edges, `A -> B` = A requires B:

```requires
C17 -> C01 symbol
C20 -> C01 option
C20 -> C03 llvm-version
C20 -> C04 mechanism
```

Every other candidate requires no candidate. Topological order (prerequisites first): C01, C03, C04, then C17 and C20, with C02, C05-C16, C18 and C19 free; the graph is acyclic. Merged cycle: the host JIT vector-math change into C01 (see C01).

Upstream-pending edges: C01 (maintainer decision on the configuration model), C03 (maintainer decision on LLVM 23 as a development version; PR #1986), C20 (maintainer decision on a pixi manifest), C07 and C15 (overlap with open PRs #2282 and #2307 that may carry the same fix; overlap, not a requirement).

Seams: C01 (fork-URL lines in hunks `@268,7`, `@244,15`), C20 (KF3, source pins and patches). All other candidates: none.

## Applicability (D2) method

For a candidate X with prerequisite closure P (`C17 -> {C17, C01}`, `C20 -> {C20, C01, C03, C04}`, otherwise `{X}`): select every hunk of every path of the difference set whose owner is in P (whole paths contribute all their hunks), apply the selected hunks of `git diff --no-renames -U0 U M` to U's file content by old-side coordinates (no git apply heuristics), and check that every hunk's removed lines equal U's lines at the old range. The result is `clean` when all selected hunks match U and none of them carries a conflict marker line (`<<<<<<<`); otherwise the files whose selected hunks carry markers are listed as conflicting. A sanity run applies all hunks of all 370 paths and compares the result with M: 0 mismatches. Keep-in-fork decisions are not prerequisites.

| candidate | closure | hunks match U | conflicting files |
|---|---|---|---|
| C01 | C01 | yes | `CMakeLists.txt`, `include/hipSYCL/common/settings.hpp`, `src/compiler/llvm-to-backend/CMakeLists.txt` |
| C02 | C02 | yes | none |
| C03 | C03 | yes | none |
| C04 | C04 | yes | none |
| C05 | C05 | yes | none |
| C06 | C06 | yes | none |
| C07 | C07 | yes | `include/hipSYCL/sycl/libkernel/sscp/builtins/detail/reduction.hpp` |
| C08 | C08 | yes | none |
| C09 | C09 | yes | `include/hipSYCL/runtime/omp/omp_queue.hpp` |
| C10 | C10 | yes | none |
| C11 | C11 | yes | none |
| C12 | C12 | yes | none |
| C13 | C13 | yes | none |
| C14 | C14 | yes | none |
| C15 | C15 | yes | none |
| C16 | C16 | yes | none |
| C17 | C17, C01 | yes | `CMakeLists.txt`, `include/hipSYCL/common/settings.hpp`, `include/hipSYCL/compiler/llvm-to-backend/Utils.hpp`, `src/compiler/llvm-to-backend/CMakeLists.txt` |
| C18 | C18 | yes | none |
| C19 | C19 | yes | none |
| C20 | C20, C01, C03, C04 | yes | `CMakeLists.txt`, `include/hipSYCL/common/settings.hpp`, `src/compiler/llvm-to-backend/CMakeLists.txt` |
