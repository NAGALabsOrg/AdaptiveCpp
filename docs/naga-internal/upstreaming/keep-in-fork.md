# Upstreaming triage: keep-in-fork decisions

Pins: U `27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc`, N `0e2bac974c40554f62286cbffe1f06c87cc9caa9`, B `9ec394efa584ced004d644af2194d2133e111d26`; merge tree M `350667e47d62407ad9bef5977bdab5a632ae8e87` (see baseline.md). Status: Review 1 applied (decisions.md, 2026-11-23).

Block format as in candidates.md: `paths` are whole paths (`dir/**` = every difference-set path under `dir`, never the records directory `docs/naga-internal/upstreaming/`, which is outside the difference set), `hunks` are `path @start,count` hunks of a split path (new-file coordinates of `git diff --no-renames -U0 U M -- <path>`), with `:<new lines>^<old lines>` for a hunk split by line.

Reasons are the plan's three: fork identity, transitional, scope. Counts: 6 decisions; fork identity 2 (KF1, KF2), transitional 4 (KF4, KF5, KF6, KF7), scope 0 (none). KF3 (patches and pins) moved into candidate C20 at Review 1 (U06) and its id is retired. A `trigger` is the event after which the decision is revisited.

## KF1 Fork project tooling

- reason: fork identity
- sentence: The project bank and its commit hook (`.project/`) and the `.gitignore` line for them are tooling of this fork's own operation and mean nothing to AdaptiveCpp.
- trigger: none
- size: files 3, lines +55 / -0
- changes:

```assignment KF1
paths
  .project/**
hunks
  .gitignore @57,5
```

## KF2 NAGA internal documents

- reason: fork identity
- sentence: The kernel-language, graph-API, vocabulary and research documents under `docs/naga-internal/` are NAGA Labs' own design documents, not documentation of AdaptiveCpp.
- trigger: none
- size: files 7, lines +1301 / -0
- changes:

```assignment KF2
paths
  docs/naga-internal/**
```

## KF4 The fork's SPIR-V translator repository

- reason: transitional
- sentence: The `GIT_REPOSITORY` line that selects `NAGALabsOrg/SPIRV-LLVM-Translator` points at the fork's branch of the translator for its pinned LLVM 23 snapshot, which exists only for the fork's own environment.
- trigger: none recorded (Review 1, U03)
- size: files 1, lines +1 / -1
- seam: candidate C20 pins a commit of that repository (see the seams block in candidates.md).
- changes:

```assignment KF4
hunks
  src/compiler/llvm-to-backend/CMakeLists.txt @268,7:5^1
```

## KF5 small_vector as std::vector under libc++

- reason: transitional
- sentence: Making `small_vector` and `auto_small_vector` plain `std::vector` under libc++ works around the small-buffer allocator for the fork's macOS toolchain runs, and upstream's commit 96c7d17a (#2273) may make it unnecessary.
- trigger: a macOS libc++ run of upstream develop shows no teardown abort (Review 1, U05)
- size: files 1, lines +15 / -0
- changes:

```assignment KF5
paths
  include/hipSYCL/common/small_vector.hpp
```

## KF6 LLVM 22 changes that PR #1986 also makes

- reason: transitional
- sentence: The CI matrix entry, plugin-header move, `--enable-unsafe-fp-math` guards, lifetime-intrinsic argument fix, data layouts, Metal includes and install documentation for LLVM 22 are the same changes open PR #1986 makes, so the fork keeps them until that PR lands.
- trigger: #1986 merges (Review 1, U08)
- size: files 10, lines +47 / -16
- seam: candidate C03 builds on these changes (see the seams block in candidates.md).
- changes:

```assignment KF6
paths
  .github/**
  src/compiler/llvm-to-backend/metal/**
hunks
  src/compiler/AdaptiveCppLlvmPasses.cpp @42,1 @44,3
  src/compiler/llvm-to-backend/AddressSpaceInferencePass.cpp @160,4
  src/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpu.cpp @207,5
  src/compiler/llvm-to-backend/host/LLVMToHost.cpp @34,1 @351,1 @353,1
  src/compiler/llvm-to-backend/ptx/LLVMToPtx.cpp @95,4 @241,1 @243,1
  src/compiler/sscp/HcfRegistrationPass.cpp @27,1 @29,3 @261,1
rest
  doc/install-llvm.md
```

## KF7 SSCP identities and the wg_reduce broadcast comment

- reason: transitional
- sentence: The min/max identity helpers match what open PR #2282 makes, and the `wg_reduce` broadcast hunk is only a comment on a fix that U already has from #2301, so the fork keeps both until #2282 lands.
- trigger: #2282 merges (Review 1, C07 amendment)
- size: files 2, lines +24 / -4
- seam: candidate C07's tests expect these identities (see the seams block in candidates.md).
- changes:

```assignment KF7
paths
  include/hipSYCL/sycl/libkernel/sscp/group_functions.hpp
hunks
  include/hipSYCL/sycl/libkernel/sscp/builtins/detail/reduction.hpp @112,8
```
