# Upstreaming triage: decisions

Pins: U `27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc`, N `0e2bac974c40554f62286cbffe1f06c87cc9caa9`, B `9ec394efa584ced004d644af2194d2133e111d26`; merge tree M `350667e47d62407ad9bef5977bdab5a632ae8e87` (see baseline.md).

Status: first run, stopped before Review 1. Nothing in this file is decided. Every boundary, kind, large flag and keep-in-fork reason in candidates.md and keep-in-fork.md is a proposal of the Research Lead. The proposals rest on heuristics: commit themes are the Research Lead's own classification of subjects, hunk ownership is by `git blame` on a scratch commit with tree M and by pickaxe for deletion-only hunks, so Review 1 must treat all of them as unchecked claims about intent.

## Review 1

none (not yet held)

## Review 2

none (not yet held)

## Unresolved list (for Review 1)

Each item: what resists the rules, the Research Lead's proposal as recorded, and the options. Jack decides; a dissent by the Research Lead would be one line on the item.

### U01 Hunks assigned by adjacency or pickaxe

- what: 16 hunks have no N-only commit on their lines (deletion-only or context-only) and went to the owner of the neighbouring hunk; 45 deletion-only hunks went to the commit that deleted the lines (pickaxe).
- adjacency hunks, with the owner they went to: `CMakeLists.txt@707,1` C01, `CMakeLists.txt@787,1` C01, `bin/acpp@1008,0` C01, `include/hipSYCL/common/settings.hpp@74,1` C01, `@102,1` C01, `@154,1` C01, `include/hipSYCL/compiler/llvm-to-backend/Utils.hpp@380,1` C17, `include/hipSYCL/runtime/kernel_cache.hpp@387,0` C16, `include/hipSYCL/runtime/omp/omp_queue.hpp@21,1` C09, `src/compiler/llvm-to-backend/CMakeLists.txt@264,2` C01, `@441,1` C01, `@451,2` C01, `@481,0` C01, `src/compiler/sscp/StdAtomicRemapperPass.cpp@307,0` C14, `src/runtime/CMakeLists.txt@272,3` C01, `src/runtime/pcuda/pcuda_device_topology.cpp@40,0` C11.
- options: (a) accept as recorded; (b) Jack names the hunks he wants moved and the Research Lead reassigns them.

### U02 Entangled hunks of modified files left provisionally in C01

- what: six hunks mix lines of two intents; kept whole in C01 for now.
  - `src/compiler/llvm-to-backend/CMakeLists.txt@200,21`: a C19 `CMAKE_DL_LIBS` line next to a C01 anchor.
  - `src/compiler/llvm-to-backend/Utils.cpp@23,12` and `@53,58`: C17 definitions (`suppressChildErrorDialogs`, `describeToolExitCode`) mixed with C01 tool-path resolution. Effect: C17 requires C01 (edge recorded in candidates.md, type symbol).
  - `src/runtime/CMakeLists.txt@229,3`.
  - `src/compiler/llvm-to-backend/CMakeLists.txt@268,7` and `@276,3`: see U03.
- also resolved by content, for Jack's confirmation: `.gitignore@57,5` to KF1; `src/compiler/AdaptiveCppLlvmPasses.cpp@155,9`, `src/compiler/CMakeLists.txt@158,10`, `src/compiler/llvm-to-backend/CMakeLists.txt@58,5` and `src/runtime/CMakeLists.txt@376,83` to C04; `tests/sycl/extensions.cpp@717,177` to C02 and `@1529,110` to C05.
- options: (a) keep in C01 (current; C17 then requires C01); (b) hand-split the `Utils.cpp` hunks so C17's two definitions move to C17 and C17 stands alone; (c) move the definitions to C17 whole and have C01 require C17.

### U03 Fork URL inside a C01 hunk

- what: `src/compiler/llvm-to-backend/CMakeLists.txt@268,7` sets `GIT_REPOSITORY https://github.com/NAGALabsOrg/SPIRV-LLVM-Translator` (U: `AdaptiveCpp/SPIRV-LLVM-Translator`), and `@276,3` carries the generic `LIST_SEPARATOR` and `core.longpaths` lines; `@244,15` (`LLVMSPIRV_GIT_TAG`) belongs with them.
- options: (a) rework before offering: restore the upstream URL in C01 (recorded); (b) move the fork-URL line to a new keep-in-fork decision (fork identity) and keep the generic lines in C01; (c) offer the tag and long-path handling as a separate small candidate.

### U04 Runs of added files that belong to a different intent than the file's owner

- what: 11 runs inside added files, kept whole in the file's dominant candidate (C01) because line-level splitting produced mid-statement fragments: `cmake/discovery.cmake@39,1 @42,1 @70,5` and `cmake/options/common/core.cmake@434,1` and `cmake/options/windows/common/core.cmake@50,1 @62,1`, `config/windows/common/core.json@10,3`, `config/windows/common/deploy/core.json@52,2` (themed C04, linked-into-LLVM); `cmake/options/linux/common/cuda.cmake@53,1 @55,1` and `cmake/options/windows/common/cuda.cmake@46,1` (themed C18, cuda include subdir).
- options: (a) leave whole in C01 (recorded; C01 then also carries the C04 and CUDA-portability intents in these lines); (b) hand-split and move the runs to C04 and a new C21; (c) treat them as part of C01's intent and drop the theme.

### U05 C10 and upstream commit 96c7d17a

- what: C10 makes `small_vector` a `std::vector` under libc++ to avoid a teardown double free; U's `96c7d17a` (#2273, 2026-11-18) works around libc++ 22 in `include/hipSYCL/common/sbo/small_vector.hpp`. Its diff has not been read against C10's.
- options: (a) keep as a candidate and have the Research Lead read both diffs; (b) drop C10 as superseded if #2273 fixes the same failure (becomes an absorbed commit, not keep-in-fork); (c) keep in fork (scope).

### U06 Seam between C20 and KF3

- what: C20 reads `packages/core/pins.toml` and `packages/core/patches/` (KF3, transitional).
- options: (a) move KF3 into C20; (b) rework C20 so source pins and patches are input of the package (recorded); (c) offer C20 without `packages/core` and keep that directory in the fork.

### U07 C20 is contested

- what: a case to offer (Jack's offer in discussion #1981 on 2026-09-04; maintainer asked on 2026-09-09 whether it could be upstreamed) and a case to keep (channel `https://prefix.dev/jackm97/naga-labs`, backend `pixi-build-naga`, package names `naga-acpp-*`, a vendored third-party directory, a lock file of 3824 lines).
- recorded: candidate flagged for conversation, fallback keep-in-fork.
- options: (a) candidate with the conversation (recorded); (b) keep in fork now (fork identity); (c) offer a reduced manifest only (workspace and tasks, not the packages); (d) split as proposed in candidates.md.

### U08 C03 and development LLVM 23

- what: U's gate rejects LLVM above 21 and development builds. C03 raises the maximum to 23 for an AMD snapshot, and open PR #1986 does LLVM 22.
- options: (a) one candidate (recorded) with the maintainer decision as a pending edge; (b) split into LLVM 22 (overlaps #1986) and LLVM 23; (c) keep the LLVM 23 part in the fork (transitional).

### U09 C04 is contested

- what: C04's use case is the fork's own build (AdaptiveCpp linked into an LLVM build with openmp as a runtime), so it could be kept as scope.
- options: (a) candidate (recorded); (b) keep in fork (scope); (c) merge into C03 or C20.

### U10 Keep-in-fork boundaries

- KF1: the `.gitignore@57,5` hunk with `.project/**`; whether the `.gitignore@4,2` hunk (C20) should be there too.
- KF2: whether any `docs/naga-internal/` document is worth offering.
- KF3: transitional for the fork's LLVM snapshot; see U06.
- options: accept as recorded, or name paths to move.

### U11 Kind choices

- C09 (include-what-you-use for Xcode 26): recorded bug fix; alternative other: refactor.
- C16 (code objects built outside the kernel cache lock): recorded performance; alternative bug fix.
- C19 (Threads usage requirement and package configs): recorded build-system rework; alternative bug fix.
- C02 and C05 are `other: feature` with secondary bug fixes.
- options: accept, or name the kind.

### U12 Large flags and splits

- C01, C02 and C20 are flagged large with the proposed splits written in candidates.md; whether to split before or after the conversations is open. C01's four-way split cuts across platform axes and vendor units, so each part may need the others (not checked).
- options: accept the flags and defer the split to the PR stage; accept the proposed splits as separate candidates (the Research Lead then re-runs F1, D1 and D2); name other boundaries.
