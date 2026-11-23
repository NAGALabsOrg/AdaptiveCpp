# Upstreaming triage: candidates

Pins: U `27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc`, N `0e2bac974c40554f62286cbffe1f06c87cc9caa9`, B `9ec394efa584ced004d644af2194d2133e111d26`; merge tree M `350667e47d62407ad9bef5977bdab5a632ae8e87` (see baseline.md). Status: Review 1 applied (decisions.md, 2026-11-23); conversation flags, horizons and next steps are drafted for Review 2 and are proposals until Jack decides there.

How to read the records:
- A candidate owns paths and hunks of the difference set (baseline.md). Hunk coordinates are new-file `start,count` of `git diff --no-renames -U0 U M -- <path>`. In an assignment block, `paths` are whole paths (`dir/**` = every difference-set path under `dir`), `hunks` are explicit hunks of a split path, and `rest` marks the candidate that owns every hunk of that path not listed elsewhere. A path appears either whole or split, never both.
- A hunk that serves two dispositions is split by line: `@start,count:<new lines>^<old lines>` where the new lines are the 1-based indexes among the hunk's added lines, the old lines (after `^`) are the indexes among its removed lines, and `*` means all. The pieces of a hunk partition its lines. A piece applied alone puts its added lines before the removed lines it does not own.
- Assignment basis: each hunk of a modified path went to the working theme of the N-only commits that last touched its lines (`git blame` on a scratch commit whose tree is M, parents U and N); deletion-only hunks by pickaxe, or by adjacency to the neighbouring hunk (Review 1 U01: accepted). Review 1 then amended the boundaries (decisions.md).
- Size is the diff of the owned paths and hunks (files, lines added / removed). A split path counts as a file of each owner.
- Kinds are the plan's six; `other` carries a subtype. "secondary" lists other kinds present in the change.
- `requires` edges carry one line of evidence as `M:<path>:<line> <token>`: the line in the tree M must contain the token. `pending` names an upstream-pending state. `seam` names changes this candidate relies on that are in a keep-in-fork decision.
- `applicability` is the trial merge of the candidate and its prerequisites onto U (method at the end).
- `open upstream PRs by path` lists open PRs (baseline.md) whose files include a path the candidate owns; it is path-level input, not a dependency.
- Conversation flags (C1): a candidate is flagged when at least one of five criteria fires, and the fields `flag inputs`, `conversation`, `question`, `venue` and `offer` record it: (a) kind is build-system rework or pixi workspace; (b) it adds a feature, option or extension, or changes a default (`adds`); (c) it is flagged large; (d) its disposition was contested (`contested`); (e) unmerged upstream work overlaps it (`overlap`). Criterion (e) is applied to open PRs whose intent or content overlaps the candidate's, as read in their descriptions or diffs, and to PRs the candidate depends on; a bare path overlap does not fire it, and those PRs are listed under `open upstream PRs by path` only.
- Horizon (O1) is one of: now, after conversation, after next release. Next step (N1) is `type | owner | done-condition`, with the type one of: open conversation, prepare PR, seam rework, wait.

## Candidate index

| id | title | kind | large | conversation | horizon |
|---|---|---|---|---|---|
| C01 | Relocatable toolchain configuration | build-system rework | yes | yes | after conversation |
| C02 | async_host and deferred issue of operations with foreign requirements | other: feature | yes | yes | after conversation |
| C03 | LLVM 23 support | LLVM-version support | no | yes | after conversation |
| C04 | Linked-into-LLVM build fixes | build-system rework | no | yes | after conversation |
| C05 | Dynamic functions across translation units and object formats | other: feature | no | yes | after conversation |
| C06 | CBS: values carried across barriers and split loops | bug fix | no | yes | after conversation |
| C07 | SSCP group scans and sub-group ids in work-group reduce | bug fix | no | yes | after conversation |
| C08 | Keep alloca alignment, size and name in address-space and aggregate passes | bug fix | no | no | now |
| C09 | Include what you use for Xcode 26 libc++ | bug fix | no | no | now |
| C11 | pcuda robustness and MSVC kernel names | bug fix | no | no | now |
| C12 | OpenCL: SPIR-V built for the device's OpenCL C version | bug fix | no | no | now |
| C13 | OpenCL: failed USM prefetch is an ignored hint | bug fix | no | no | now |
| C14 | stdpar: float and pointer atomics through integer builtins | bug fix | no | no | now |
| C15 | SSCP: propagate convergent before pre-inline simplification | bug fix | no | yes | after conversation |
| C16 | Kernel cache: build code objects outside the cache lock | performance | no | no | now |
| C17 | JIT tool invocations: exit-code messages and Windows error dialogs | bug fix | no | no | now |
| C18 | Level Zero: include `<cstdint>` instead of a glibc-internal header | bug fix | no | no | now |
| C19 | Link dependencies: Threads as a public requirement | bug fix | no | no | now |
| C20 | pixi workspace and packages | pixi workspace | yes | yes | after conversation |

C10 (small_vector under libc++) became keep-in-fork KF5 at Review 1; its id is retired. Counts: 19 candidates (9 flagged for a conversation, 10 not), plus the keep-in-fork decisions in keep-in-fork.md.

## C01 Relocatable toolchain configuration

- title: Relocatable toolchain configuration: option and discovery model, install layout, driver and run-time configuration resolution
- intent: Replace baked-in build-machine paths with a layered options, discovery and installed-configuration model (GNUInstallDirs install layout, `$ORIGIN` rpaths, an installed toolchain, application and deploy configuration read by the `acpp` driver and the runtime, run-time resolution of the LLVM tools and of the host JIT's vector math library), so that an installed AdaptiveCpp can be relocated and its vendor libraries bundled or deployed.
- kind: build-system rework
- secondary: other: documentation (doc/ edits for the model), other: refactor (removal of the legacy per-vendor finds and flat option emitter)
- size: files 173, lines +6596 / -1205
- large: yes (kept at Review 1, U12). Proposed split, to be decided in the conversations: (a) options, discovery and configuration merge (`cmake/options/**`, `cmake/discovery/**`, `cmake/discovery.cmake`, `cmake/acpp-config-merge.cmake`, `config/**`, root `CMakeLists.txt` wiring); (b) install layout, vendor install rules and rpaths (`cmake/install/**`, `cmake/acpp-vendor-*.cmake`, `cmake/acpp-installed-configs.cmake`, install/rpath hunks of the `src/**/CMakeLists.txt` files and `src/tools/**`); (c) driver configuration and deploy (`bin/acpp`, `doc/` edits); (d) run-time settings and tool resolution (`include/hipSYCL/common/settings.hpp`, `src/common/**`, `src/runtime/backend_loader.cpp`, `Utils.cpp` and the `llvm-to-*` tool-path hunks, `LLVMToHost` vector math). Platform axes (Windows, macOS, Android) and vendor units (HIP SDK, NVHPC, Vulkan, clspv) are spread through (a)-(d); whether the parts stand alone has not been checked.
- merged cycle: the host JIT vector-math change (default `libmvec`, loader lookup, `jit_host_*` run-time settings; working theme `vector-math`) and this candidate depend on each other, so they are one candidate. Evidence: `M:CMakeLists.txt:542` sets `DEFAULT_VEC_MATH_LIB`, consumed at `M:cmake/options/linux/common/core.cmake:129` (`set(ACPP_VECTOR_MATH_LIB "${DEFAULT_VEC_MATH_LIB}")`); the JIT consumes the installed key written from `M:config/common/core.json:90` (`"@ACPP_VECTOR_MATH_LIB@"`) via `M:src/compiler/llvm-to-backend/host/LLVMToHost.cpp:320` (`try_retrieve_settings_variable`).
- requires: none
- pending: maintainer decision on the configuration model itself: it replaces upstream's install-prefix and rpath handling, on which open upstream PRs #2283, #2268, #2160, #2131 and #1670 (draft) also work. Needs a maintainer answer before the PR; no unmerged upstream change is needed in U.
- seams: none. The fork-identity line in `src/compiler/llvm-to-backend/CMakeLists.txt` (hunk `@268,7`, the `NAGALabsOrg` translator URL) was split out at Review 1 (U03) into KF4; the upstream URL stays in this candidate's version of that hunk.
- applicability: conflicting files: `CMakeLists.txt`, `include/hipSYCL/common/settings.hpp`, `src/compiler/llvm-to-backend/CMakeLists.txt`. The conflicts overlap upstream commits 16b96a8a/bea19449 (Windows manifest and CRT), 92f1911d (settings whitespace parsing), bce257b0/f4ff3681/d795f824 (see baseline.md).
- open upstream PRs by path: #2304 #2297 #2293 #2283 #2268 #2259 #2170 #2166 #2160 #2131 #2108 #2038 #1986 #1948 #1946 #1943 #1937 #1884 #1796 #1778 #1777 #1670 #1562 #1533 #1362 #1342 #1095 #1094 #807
- flag inputs: adds=option,default; contested=no; overlap=#2283,#2268,#2160,#2131,#1670 (install prefix and rpath work; judged by title, bodies not read)
- conversation: yes (criteria a,b,c,e)
- question: Do the maintainers want installs to be relocatable through an options, discovery and installed-configuration model that replaces build-machine paths (GNUInstallDirs layout, `$ORIGIN` rpaths, an installed toolchain and application config read by `acpp` and the runtime, run-time libmvec lookup, new `ACPP_<VENDOR>_ROOT` options and a changed vector-math default)? How should it be split, and how does it relate to #2283, #2268, #2160, #2131 and #1670?
- venue: new issue (CONTRIBUTING.md line 3: open an issue when in doubt about design), opened together with C20's reply in discussion #1981 as conversation 1
- offer: the whole candidate on U, in the parts agreed in the issue
- horizon: after conversation
- next step: open conversation | owner: Jack | done: the issue is open together with the reply in discussion #1981 (C20), with the recorded question, and the maintainers' answer is recorded in decisions.md

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
  src/compiler/llvm-to-backend/CMakeLists.txt @200,21 @268,7:1-4,6-7^2-3
  src/compiler/llvm-to-backend/Utils.cpp @23,12:9-12 @53,58:25-30,32-58^*
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
- large: yes (kept at Review 1, U12). Proposed split, to be decided in the conversations: (a) deferral on foreign requirements (`include/hipSYCL/runtime/{cuda,hip}/**`, `src/runtime/{cuda,hip}/**`, `inorder_executor`, `inorder_queue.hpp`, `operations.hpp`, `event.hpp`, `doc/runtime-spec.md`); (b) the `async_host` extension (`async_host_executor.*`, `sycl/{queue,handler,extensions}.hpp`, `doc/extensions.md`, tests); (c) the retarget context (`hints.hpp`, `usm_query.hpp`, handler/queue retarget hunks).
- requires: none
- pending: none. Open upstream PRs by path overlap (below) but none is needed.
- seams: none
- applicability: clean
- open upstream PRs by path: #2289 #2283 #2232 #2200 #1948 #1946 #1916 #1778 #1670 #1478 #1095 #912 #618
- flag inputs: adds=feature; contested=no; overlap=none (path-only overlap with #2289, #2232, #2200, #1946, #1916 and others; their descriptions do not share this intent for #2200 and #1478, the rest not read)
- conversation: yes (criteria b,c)
- question: Is an `async_host` queue and handler extension, with deferred issue of operations whose requirements belong to other queues or host tasks (host-function deferral on CUDA and HIP, waits on in-order queues, a retarget context), wanted in AdaptiveCpp, and how should it be split?
- venue: new issue
- offer: the candidate with its documentation (doc/extensions.md, doc/runtime-spec.md) and tests, in the parts agreed in the issue
- horizon: after conversation
- next step: open conversation | owner: Jack | done: the issue is open with the recorded question and the maintainers' answer is recorded in decisions.md

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

## C03 LLVM 23 support

- title: LLVM 23 support: compatibility helpers, compiler ports and the version gate
- intent: Build AdaptiveCpp against LLVM 23: the version gate accepts 23, the compiler passes and backends use compatibility helpers (`LLVMUtils.hpp`: `getPointerTo`, `getFirstNonPHI`, `moveBefore`, `insertBefore`, `getBranchCondition`, `createBr`, `createCondBr`, `clearModuleInlineAsm`) for the LLVM API changes (split branch instructions, `splitBlockBefore`, iterator insertion points, module inline asm), the SPIR-V extension list drops `SPV_INTEL_fast_composite` on 23, and the installation table names 23.
- kind: LLVM-version support
- secondary: none
- size: files 14, lines +177 / -42
- large: no
- requires: none
- pending: open upstream PR #1986 "Support LLVM 22" (al42and, updated 2026-11-19). Its LLVM 22 changes (plugin header move, `--enable-unsafe-fp-math` guards, lifetime-intrinsic arguments, data layouts, CI entry, install documentation) are the same as the fork's hunks held in KF6, and the LLVM 23 build needs them; they arrive in U if #1986 lands. The maintainer decision on LLVM 23 as a development version (`U:CMakeLists.txt:481` and `:486`) is asked in C20's conversation (Review 1, U08).
- seams: KF6 (see the seams block); resolution: the pending edge on #1986 (Review 1, U08), revisit trigger "#1986 merges".
- applicability: clean
- open upstream PRs by path: #2307 #2305 #2297 #2291 #2284 #2166 #2108 #2038 #1986 #1946 #1884 #1778 #1777 #1670 #1533 #1095 #1094 #807
- flag inputs: adds=none; contested=no; overlap=#1986 (same intent: LLVM 22 changes this candidate builds on; read)
- conversation: yes (criteria e)
- question: Does upstream want LLVM 23 accepted in the version gate once #1986 has landed, given the rule that development versions are not supported (`CMakeLists.txt:481`, `:486`)? Asked together with C20's question.
- venue: C20's conversation (existing discussion #1981), joined; comment on PR #1986 once it has merged
- offer: the LLVM 23 helpers and ports on top of U with #1986
- horizon: after conversation
- next step: wait | owner: Research Lead | done: C20's conversation (opened first) has the maintainers' answer on LLVM 23 and PR #1986 has merged

```assignment C03
paths
  include/hipSYCL/compiler/utils/LLVMUtils.hpp
  src/compiler/cbs/IRUtils.cpp
  src/compiler/cbs/PHIsToAllocas.cpp
  src/compiler/cbs/SyncDependenceAnalysis.cpp
  src/compiler/cbs/UniformityAnalysis.cpp
  src/compiler/cbs/VectorShapeTransformer.cpp
  src/compiler/llvm-to-backend/host/HostKernelWrapperPass.cpp
  src/compiler/sscp/TargetSeparationPass.cpp
hunks
  CMakeLists.txt @490,2
  doc/installing.md @55,3 @59,2
  src/compiler/sscp/HcfRegistrationPass.cpp @12,2 @98,2
  src/compiler/llvm-to-backend/spirv/LLVMToSpirv.cpp @57,9 @78,1
  src/compiler/sscp/pcuda/FreeKernelCall.cpp @12,2 @190,1 @199,1 @270,1
rest
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
- seams: none. Disposition contested (Review 1, U09): kept as a candidate.
- applicability: clean
- open upstream PRs by path: #2283 #2268 #2170 #2166 #2160 #2131 #2038 #1986 #1948 #1946 #1778 #1670 #1095 #1094
- flag inputs: adds=none; contested=yes; overlap=none
- conversation: yes (criteria a,d)
- question: Is building AdaptiveCpp linked into the LLVM tools (`LLVM_ADAPTIVECPP_LINK_INTO_TOOLS`, openmp as an LLVM runtime, a Windows LLVM DLL) a configuration upstream supports, so that these fixes belong upstream?
- venue: draft PR as the venue itself
- offer: the candidate as a draft PR on U
- horizon: after conversation
- next step: open conversation | owner: Jack | done: the draft PR is open with the recorded question and the maintainers' answer is recorded in decisions.md

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
- flag inputs: adds=feature; contested=no; overlap=none (path-only overlap with #2307, #2297, #2289, #2166, #1884, #1478; not read)
- conversation: yes (criteria b)
- question: Is extending the dynamic-function extension (definitions in other translation units, indexed call sequences, host stand-ins for ELF, Mach-O and COFF) wanted, and does the extension's interface change need design discussion first?
- venue: new issue
- offer: the candidate with its lit tests and doc/extensions.md text
- horizon: after conversation
- next step: open conversation | owner: Jack | done: the issue is open with the recorded question and the maintainers' answer is recorded in decisions.md

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
- pending: none. Commit `0e2bac97` fixes upstream issue #2264 (open at lookup); the earlier `681c6c5e` is already in U and is not part of this candidate.
- seams: none
- applicability: clean
- open upstream PRs by path: #2305 #2297 #2291 #2284
- flag inputs: adds=none; contested=no; overlap=#2305 (appears to fix the same lookup-order bug in `fixSingleSubCfgValues` that hunk `@1140,6` fixes; PR body read, diff not compared line by line)
- conversation: yes (criteria e)
- question: PR #2305 appears to fix the same bug in `fixSingleSubCfgValues` (the alloca found in `RemappedInstAllocaMap` being overwritten by `getLoopStateAllocaForLoad`). Our change also stops recomputing side-effecting and memory-reading values after a barrier and restores split-loop live values, with four lit tests (issue #2264). Should ours go on top of #2305 once it merges, or be folded into it?
- venue: comment on PR #2305
- offer: the candidate's hunks and four lit tests, rebased on #2305
- horizon: after conversation
- next step: open conversation | owner: Jack | done: the comment is posted on #2305 and the answer is recorded in decisions.md

```assignment C06
paths
  tests/compiler/cbs/**
hunks
  src/compiler/cbs/SubCfgFormation.cpp @662,4 @679,3 @704,2 @1140,6 @1148,6
```

## C07 SSCP group scans and sub-group ids in work-group reduce

- title: SSCP group scans and work-group reduce: race-free scans, sub-group id and count from the backend, with tests
- intent: Make SSCP work-group scans race-free (the inclusive scan of each sub-group first, exclusive results shifted and `init` applied once per work-group, every work-item reaching every barrier) and make `wg_reduce` and the scans take the sub-group id and count from the backend instead of deriving them from the maximum sub-group size, with bounded halving in `wg_reduce`; with tests for fractional and negative float min/max reductions and scans.
- kind: bug fix
- secondary: other: test
- size: files 5, lines +302 / -82
- large: no
- requires: none
- pending: open upstream PR #2282 "[SSCP] Fix known identity for group algorithms" (sbalint98, approved, updated 2026-11-19). The tests of this candidate expect the identities that #2282 and KF7 provide (`M:tests/sycl/group_functions/group_functions_scan.cpp:1004`). The identity hunks and the `wg_reduce` broadcast comment are held in KF7 (Review 1, C07 amendment).
- seams: KF7 (see the seams block); resolution: the pending edge on #2282, revisit trigger "#2282 merges".
- applicability: clean
- open upstream PRs by path: #2326 #2297 #2282 #966 #778
- flag inputs: adds=none; contested=no; overlap=#2282 (the tests rely on its identities and rewrite the same test file; read)
- conversation: yes (criteria e)
- question: The scan rewrites and the sub-group id and count change in `wg_reduce` are independent of #2282's identities, but our fractional and negative min/max tests rely on them. Should they come as a separate PR after #2282, and do you want the tests added there?
- venue: comment on PR #2282
- offer: the scan and `wg_reduce` changes and the tests, as a PR on top of #2282
- horizon: after conversation
- next step: open conversation | owner: Jack | done: the comment is posted on #2282 and the answer is recorded in decisions.md

```assignment C07
paths
  include/hipSYCL/sycl/libkernel/sscp/builtins/detail/scan_generic.hpp
  include/hipSYCL/sycl/libkernel/sscp/builtins/detail/scan_hiplike.hpp
  tests/sycl/group_functions/**
rest
  include/hipSYCL/sycl/libkernel/sscp/builtins/detail/reduction.hpp
```

## C08 Keep alloca alignment, size and name in address-space and aggregate passes

- title: Keep alloca alignment, size and name when rewriting address spaces and expanding aggregate arguments
- intent: Preserve the alignment, array size and name (`takeName`) of allocas that the address-space inference and the clspv address-space-cast removal passes replace, and the alignment of the alloca that aggregate-argument expansion creates for a by-value parameter.
- kind: bug fix
- secondary: none
- size: files 3, lines +20 / -3
- large: no
- requires: none
- pending: none
- seams: none
- applicability: clean
- open upstream PRs by path: #1986
- flag inputs: adds=none; contested=no; overlap=none (path-only overlap with #1986)
- conversation: no
- horizon: now
- next step: prepare PR | owner: Research Lead | done: a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
- branch: pr/alloca-alignment-size-name (head 9c902b6d345d)

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
- kind: bug fix (a build failure; kind accepted at Review 1, U11)
- secondary: none
- size: files 53, lines +106 / -3
- large: no
- requires: none
- pending: none
- seams: none
- applicability: conflicting files: `include/hipSYCL/runtime/omp/omp_queue.hpp` (upstream b5535e79 "[PCUDA][OpenMP] Fix PCUDA kernel launch not running in worker thread (#2263)" changes the include block)
- open upstream PRs by path: #2297 #2283 #2200 #1946 #1478 #1095 #618 #505
- flag inputs: adds=none; contested=no; overlap=none (path-only overlap with #2297, #2283, #2200, #1946, #1478, #1095, #618, #505; #1478 read, a different intent)
- conversation: no
- horizon: now
- next step: prepare PR | owner: Research Lead | done: a branch on U carries exactly the hunks of this candidate and the conflict in `omp_queue.hpp` is resolved against b5535e79
- branch: pr/xcode26-includes (head bf81c80a6801)

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
- flag inputs: adds=none; contested=no; overlap=none
- conversation: no
- horizon: now
- next step: prepare PR | owner: Research Lead | done: a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
- branch: pr/pcuda-robustness (head a78cf9316230)

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
- flag inputs: adds=none; contested=no; overlap=none (path-only overlap with #1884; not read)
- conversation: no
- horizon: now
- next step: prepare PR | owner: Research Lead | done: a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
- branch: pr/ocl-spirv-cl-std (head 1c14f4cf7268)

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
- flag inputs: adds=none; contested=no; overlap=none (path-only overlap with #2323 and #2270; both read, different intents)
- conversation: no
- horizon: now
- next step: prepare PR | owner: Research Lead | done: a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
- branch: pr/ocl-prefetch-hint (head d91d044e2330)

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
- flag inputs: adds=none; contested=no; overlap=none
- conversation: no
- horizon: now
- next step: prepare PR | owner: Research Lead | done: a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
- branch: pr/stdpar-atomic-int-builtins (head e9c5d084802e)

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
- pending: none. Open PR #2307 (amavliut) makes the same fix; this candidate is an overlap, not a requirement. Compared at the diff level: #2307 marks functions only and also changes stage 1 to `buildO0DefaultPipeline`; this candidate also marks direct call sites and its test puts the barrier behind a user helper (expects one barrier call), where #2307's test calls `item.barrier()` directly (expects two).
- seams: none
- applicability: clean
- open upstream PRs by path: #2307 #2297 #2166 #1884
- flag inputs: adds=none; contested=no; overlap=#2307 (same intent; diff read)
- conversation: yes (criteria e)
- question: Our `propagateConvergence` also marks direct call sites convergent, and our lit test puts the barrier behind a user helper (expects one barrier call), where #2307's test calls `item.barrier()` directly. Do you want the call-site marking and the helper test folded into #2307, or as a follow-up PR?
- venue: comment on PR #2307
- offer: the call-site marking and the helper-barrier test
- horizon: after conversation
- next step: open conversation | owner: Jack | done: the comment is posted on #2307 and the answer is recorded in decisions.md

```assignment C15
paths
  tests/compiler/sscp/preinline_convergent.cpp
hunks
  src/compiler/llvm-to-backend/LLVMToBackend.cpp @36,2 @65,46 @420,1
```

## C16 Kernel cache: build code objects outside the cache lock

- title: Kernel cache: build code objects outside the cache lock
- intent: Compile and load code objects without holding the kernel cache's lock, with futures for concurrent requests for the same object.
- kind: performance (kind accepted at Review 1, U11)
- secondary: bug fix (lock held across JIT compilation)
- size: files 1, lines +84 / -50
- large: no
- requires: none
- pending: none
- seams: none
- applicability: clean
- open upstream PRs by path: #2200 #1478
- flag inputs: adds=none; contested=no; overlap=none (path-only overlap with #2200 (read, a free-function-kernels prototype) and #1478)
- conversation: no
- horizon: now
- next step: prepare PR | owner: Research Lead | done: a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
- branch: pr/kernel-cache-lock (head bd3937a5f568)

```assignment C16
rest
  include/hipSYCL/runtime/kernel_cache.hpp
```

## C17 JIT tool invocations: exit-code messages and Windows error dialogs

- title: JIT tool invocations: report the exit code of a failed tool and suppress Windows error dialogs in child processes
- intent: Make a failing `opt`, `llc`, `lld`, `clang`, `clspv` or `llvm-spirv` invocation report its exit code, and prevent a child that cannot load a DLL from blocking the JIT on a modal Windows error dialog.
- kind: bug fix
- secondary: none
- size: files 7, lines +61 / -15
- large: no
- requires: none. At Review 1 (U02) the two definitions `suppressChildErrorDialogs` and `describeToolExitCode` were split out of C01's tool-path hunk into this candidate (`M:src/compiler/llvm-to-backend/Utils.cpp:58` and `:73`), so the former edge on C01 is gone.
- pending: none
- seams: none
- applicability: conflicting files: `include/hipSYCL/compiler/llvm-to-backend/Utils.hpp` (a whole path of this candidate; upstream bce257b0 adds `ACPP_DEFAULT_VISIBILITY` to the declarations)
- open upstream PRs by path: #2304 #2297 #2170 #1986 #1943 #1937 #1884 #1796 #1478
- flag inputs: adds=none; contested=no; overlap=none (path-only overlap with #2304, #2297, #2170, #1986, #1943, #1937, #1884, #1796, #1478; not read)
- conversation: no
- horizon: now
- next step: prepare PR | owner: Research Lead | done: a branch on U carries exactly the hunks of this candidate and the conflict in `Utils.hpp` is resolved against bce257b0
- branch: pr/jit-tool-errors (head 9c705d333b5d)

```assignment C17
paths
  include/hipSYCL/compiler/llvm-to-backend/Utils.hpp
hunks
  src/compiler/llvm-to-backend/Utils.cpp @19,2 @23,12:1-8 @53,58:1-24,31 @266,1 @275,1
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
- flag inputs: adds=none; contested=no; overlap=none
- conversation: no
- horizon: now
- next step: prepare PR | owner: Research Lead | done: a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
- branch: pr/ze-cstdint (head 5283deac3af0)

```assignment C18
paths
  src/runtime/ze/**
```

## C19 Link dependencies: Threads as a public requirement

- title: Declare Threads as a public usage requirement of `acpp-rt` and in the package configs
- intent: Link `Threads::Threads` publicly from `acpp-rt` (and name it explicitly for `rt-backend-omp`) and call `find_dependency(Threads)` in the installed `adaptivecpp`, `hipsycl` and `opensycl` package configs so consumers get the imported target.
- kind: bug fix (changed from build-system rework at Review 1, U11)
- secondary: none
- size: files 4, lines +20 / -3
- large: no
- requires: none. Checked and ruled out: C02's `async_host_executor.cpp` uses `std::thread` (`M:src/runtime/async_host_executor.cpp:12`, `:68`) but links `Threads::Threads` already at U (`U:src/runtime/CMakeLists.txt:64`, PRIVATE), so C02 does not need this change. The `CMAKE_DL_LIBS` line (`llvm-to-backend/CMakeLists.txt@200,21`) was moved here from C01 at Review 1 (U02) and moved back to C01 at Review 2, because its comment names `getLibMvecDir()`, which is C01's code (`M:src/compiler/llvm-to-backend/CMakeLists.txt:203`).
- pending: none
- seams: none
- applicability: clean
- open upstream PRs by path: #2283 #2268 #2170 #2166 #2160 #2131 #1948 #1946 #1778 #1670 #1562 #1362 #1095 #1094
- flag inputs: adds=none; contested=no; overlap=none (path-only overlap with #2283, #1948, #1946, #1778, #1670, #1562, #1362, #1095; not read)
- conversation: no
- horizon: now
- next step: prepare PR | owner: Research Lead | done: a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
- branch: pr/threads-public-link (head eb21034a25ce)

```assignment C19
hunks
  cmake/adaptivecpp-config.cmake.in @9,3
  cmake/hipsycl-config.cmake.in @14,3
  cmake/opensycl-config.cmake.in @11,3
  src/runtime/CMakeLists.txt @65,7 @532,4
```

## C20 pixi workspace and packages

- title: pixi workspace: pixi-build packages for the toolchain, their build scripts, source pins and patches, environments and tasks
- intent: Provide a pixi workspace that builds the AdaptiveCpp toolchain from source as conda packages (core build with its source pins and patches, slices, runtime vendor extras, compiler activation, consumer check), with tasks to build and run the test suite and sanitizer smoke tests against the installed environment.
- kind: pixi workspace
- secondary: other: vendored third-party (`packages/activations/vendor/ctng-compiler-activation`, conda-forge's compiler activation scripts, pinned by `PINNED_REF`), other: tooling/CI (`naga/tasks`, `.pixi/config.toml`), other: patches to a pinned LLVM snapshot (`packages/core/patches`, moved here from KF3 at Review 1, U06)
- size: files 60, lines +8204 / -0 (of which `pixi.lock` is +3824)
- large: yes (kept at Review 1, U12). Proposed split, to be decided in the conversation: (a) workspace manifest, lock and tasks (`pixi.toml`, `pixi.lock`, `naga/tasks/**`, `.pixi/config.toml`, `.gitignore` hunk); (b) package slices and vendor extras (`packages/slices/**`, `packages/extras/**`, `packages/checks/**`); (c) core build, sources and compiler activation (`packages/core/**`, `packages/activations/**`).
- requires: C01, C03, C04. Evidence is in the requires block; C03 carries the pending edge on #1986, so this candidate waits on it through C03.
- pending: maintainer decision: whether upstream wants a pixi manifest. Discussion #1981: Jack offered to submit it on 2026-09-04 and illuhad replied on 2026-09-09 asking what infrastructure the packaging needs and whether it could be upstreamed; no answer on the `pixi build` offer was found (baseline.md, maintainer statements).
- seams: KF4 (the pinned translator commit lives in the fork's translator repository); see the seams block. Review 2 addresses it by asking in this candidate's conversation whether upstream wants an LLVM 23 branch in AdaptiveCpp/SPIRV-LLVM-Translator, which the fork could supply. The patches and pins (KF3 before Review 1) are part of this candidate now, so that seam is resolved. Renaming the `naga-acpp-*` packages and the channel `https://prefix.dev/jackm97/naga-labs` is preparation for this candidate's PR (Review 1, U07), not a separate disposition.
- applicability (with C01, C03, C04): conflicting files: `CMakeLists.txt`, `include/hipSYCL/common/settings.hpp`, `src/compiler/llvm-to-backend/CMakeLists.txt` (all from C01)
- open upstream PRs by path: none
- flag inputs: adds=none; contested=yes; overlap=none
- conversation: yes (criteria a,c,d)
- question: Would AdaptiveCpp take a pixi workspace that builds the toolchain as conda packages (core build with source pins and patches, slices, runtime vendor extras, compiler activation, consumer check, lock file in the tree), and in what shape? This answers illuhad's question of 2026-09-09 about what infrastructure the packaging needs: it needs the build-system work of the linked C01 issue first. For LLVM 23 builds against a development snapshot: is the development-version rule at `CMakeLists.txt:481` and `:486` something you want to keep or relax (the question C03 rides on), and do you want an LLVM 23 branch in AdaptiveCpp/SPIRV-LLVM-Translator, which we could supply?
- venue: existing discussion #1981 (Jack's offer of 2026-09-04, illuhad's question of 2026-09-09); the reply is posted together with C01's issue as conversation 1 and links it
- offer: the pixi workspace and packages, with the `naga-acpp-*` names and the channel replaced before the PR
- horizon: after conversation
- next step: open conversation | owner: Jack | done: the reply is posted in discussion #1981 together with C01's issue, linking it, with the recorded questions, and the maintainers' answers are recorded in decisions.md

```assignment C20
paths
  .pixi/**
  naga/**
  packages/activations/**
  packages/checks/**
  packages/core/activation/**
  packages/core/licenses/**
  packages/core/patches/**
  packages/extras/**
  packages/slices/**
  packages/README.md
  packages/core/build-lib.nu
  packages/core/build-linux.nu
  packages/core/fetch-sources.nu
  packages/core/pins.toml
  packages/core/pixi.toml
  packages/core/write-build-cfg.nu
  pixi.lock
  pixi.toml
hunks
  .gitignore @4,2
```

## Dependency map (D1)

Edges, `A -> B type evidence` = A requires B; the evidence is `M:<path>:<line> <token>` and the token must be on that line of the tree M:

```requires
C20 -> C01 option M:packages/core/build-lib.nu:41 ACPP_CUDA_ROOT
C20 -> C03 llvm-version M:packages/core/build-lib.nu:52 LLVM
C20 -> C04 mechanism M:packages/core/build-lib.nu:37 LLVM_ADAPTIVECPP_LINK_INTO_TOOLS
```

Upstream-pending edges, `candidate kind number-or-text`; a `pr` edge names an open upstream PR and is checked against GitHub:

```pending
C03 pr 1986 Support LLVM 22
C07 pr 2282 Fix known identity for group algorithms
C01 decision configuration model for relocatable installs
C20 decision pixi manifest in the repository
```

Seams, `candidate keep-in-fork resolution evidence-token | note`; the resolution is `pending` (a named upstream PR carries the change, with its Review 1 reference) or `rework` (the rework needed):

```seams
C03 KF6 pending M:src/compiler/AdaptiveCppLlvmPasses.cpp:43 PassPlugin | the LLVM 23 build needs the LLVM 22 plugin-header move that #1986 also makes (Review 1, U08)
C07 KF7 pending M:tests/sycl/group_functions/group_functions_scan.cpp:1004 known_identity_v | the tests expect the identities that #2282 also makes (Review 1, C07 amendment)
C20 KF4 rework M:packages/core/build-lib.nu:57 LLVMSPIRV_COMMIT | the pinned commit exists only in the fork's translator repository that the URL line in KF4 selects; rework: carry the repository URL as a configure input of the package build, so the upstream URL stays in C01; Review 2: the C20 conversation also asks whether upstream wants an LLVM 23 branch in AdaptiveCpp/SPIRV-LLVM-Translator, which the fork could supply
```

Every other candidate has no requires, no pending edge and no seam. The graph is acyclic; topological order (prerequisites first): C01, C03, C04, then C20, with every other candidate free. Merged cycle: the host JIT vector-math change into C01 (see C01). Review 1 changes to the graph: C17 no longer requires C01 (U02); C01's seam on the translator URL is resolved (U03); C20's seam on KF3 is resolved by moving KF3 into C20 (U06); C03 and C07 gained pending edges and seams on KF6 and KF7. Review 2: the `CMAKE_DL_LIBS` line moved from C19 back to C01 (no edge changes).

## Applicability (D2) method

For a candidate X with prerequisite closure P (`C20 -> {C20, C01, C03, C04}`, otherwise `{X}`): select every part of every hunk of the difference set whose owner is in P (whole paths contribute all their hunks; a split hunk contributes its lines owned by P), apply the selected parts of `git diff --no-renames -U0 U M` to U's file content by old-side coordinates (no git apply heuristics), and check that every touched hunk's removed lines equal U's lines at the old range. The result is `clean` when all touched hunks match U and no selected added line starts with `<<<<<<< `; otherwise the files whose selected lines carry markers are listed as conflicting. A sanity run applies every part of every disposition to all 370 paths and compares the result with M.

| candidate | closure | hunks match U | conflicting files |
|---|---|---|---|
| C01 | C01 | yes | `CMakeLists.txt`, `include/hipSYCL/common/settings.hpp`, `src/compiler/llvm-to-backend/CMakeLists.txt` |
| C02 | C02 | yes | none |
| C03 | C03 | yes | none |
| C04 | C04 | yes | none |
| C05 | C05 | yes | none |
| C06 | C06 | yes | none |
| C07 | C07 | yes | none |
| C08 | C08 | yes | none |
| C09 | C09 | yes | `include/hipSYCL/runtime/omp/omp_queue.hpp` |
| C11 | C11 | yes | none |
| C12 | C12 | yes | none |
| C13 | C13 | yes | none |
| C14 | C14 | yes | none |
| C15 | C15 | yes | none |
| C16 | C16 | yes | none |
| C17 | C17 | yes | `include/hipSYCL/compiler/llvm-to-backend/Utils.hpp` |
| C18 | C18 | yes | none |
| C19 | C19 | yes | none |
| C20 | C20, C01, C03, C04 | yes | `CMakeLists.txt`, `include/hipSYCL/common/settings.hpp`, `src/compiler/llvm-to-backend/CMakeLists.txt` |
