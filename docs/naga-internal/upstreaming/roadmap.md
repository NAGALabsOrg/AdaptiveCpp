# Upstreaming triage: roadmap

Pins: U `27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc` (upstream/develop, 2026-11-19T21:12:24Z, unchanged when fetched again on 2026-11-23), N `0e2bac974c40554f62286cbffe1f06c87cc9caa9`, B `9ec394efa584ced004d644af2194d2133e111d26`.

Status: draft for Review 2. The order, horizons, conversations and next steps below are the Research Lead's proposals; nothing here is decided until decisions.md holds Jack's Review 2 entry. Candidates and keep-in-fork decisions were approved at Review 1 (2026-11-23); their full records are in candidates.md and keep-in-fork.md. This file is meant to be read alone.

In short: 19 candidates, 6 keep-in-fork decisions. 9 candidates need a maintainer conversation first (C01, C02, C03, C04, C05, C06, C07, C15, C20); 10 can go straight to a PR (C08, C09, C11, C12, C13, C14, C16, C17, C18, C19). Candidates are PR-ready against U (no build was run; readiness means the textual trial merge, see candidates.md); the two with conflicts to resolve first are C09 (`omp_queue.hpp`) and C17 (`Utils.hpp`).

## Release timing

Two facts, both recorded, not reconciled (lookup 2026-11-23):
- CONTRIBUTING.md at U (lines 21-30) states the cadence `year.02`, `year.06`, `year.10` (released at the end of February, June and October), no new features in the last week before a release, and in the last two weeks new features only for components under heavy development (for example SSCP or stdpar).
- The GitHub milestone `26.10.0` is open (13 open and 8 closed items, no due date, updated 2026-11-19T20:05:03Z) and PRs #2273, #2282, #2307 and #1986 carry it, while the end-of-October 2026 slot passed without a tag: the latest tag is v25.10.0 (2025-11-05) and PR #2108 "Bump version to 26.10" is open.
- The next release date is not announced in any source found. No candidate has the horizon "after next release"; whether any should is Jack's call at Review 2.

## Order and horizons

Candidates in an order where none precedes one it requires (C20 requires C01, C03 and C04). Conversations come first, so the candidates that need one are listed first; the others follow, smallest and unconflicted first. Horizons: `now` means no conversation flag, no upstream-pending edge and no prerequisite outside `now`; `after conversation` means flagged or dependent on a flagged or pending candidate.

```order
1 C01 after conversation
2 C03 after conversation
3 C04 after conversation
4 C20 after conversation
5 C02 after conversation
6 C05 after conversation
7 C06 after conversation
8 C07 after conversation
9 C15 after conversation
10 C18 now
11 C14 now
12 C13 now
13 C12 now
14 C08 now
15 C11 now
16 C16 now
17 C19 now
18 C17 now
19 C09 now
```

Titles: C01 Relocatable toolchain configuration; C02 async_host and deferred issue of operations with foreign requirements; C03 LLVM 23 support; C04 Linked-into-LLVM build fixes; C05 Dynamic functions across translation units and object formats; C06 CBS: values carried across barriers and split loops; C07 SSCP group scans and sub-group ids in work-group reduce; C08 Keep alloca alignment and size in address-space and aggregate passes; C09 Include what you use for Xcode 26 libc++; C11 pcuda robustness and MSVC kernel names; C12 OpenCL: SPIR-V built for the device's OpenCL C version; C13 OpenCL: failed USM prefetch is an ignored hint; C14 stdpar: float and pointer atomics through integer builtins; C15 SSCP: propagate convergent before pre-inline simplification; C16 Kernel cache: build code objects outside the cache lock; C17 JIT tool invocations: exit-code messages and Windows error dialogs; C18 Level Zero: include `<cstdint>` instead of a glibc-internal header; C19 Link dependencies: Threads and the dynamic loader library; C20 pixi workspace and packages.

## Conversations

Opening order, with the reason: C20 first because it gates C03 (its LLVM 23 question rides on it) and the pixi decision; then the three comments on open PRs, which are time-sensitive (#2282 is approved, #2307 and #2305 are awaiting review); then the design issues; the draft PR last because it needs its branch prepared.

```conversations
1 C20
venue: existing discussion #1981 (Jack's offer of 2026-09-04, illuhad's question of 2026-09-09)
question: Would AdaptiveCpp take a pixi workspace that builds the toolchain as conda packages (core build with source pins and patches, slices, runtime vendor extras, compiler activation, consumer check, lock file in the tree), and in what shape? For LLVM 23 builds against a development snapshot: is the development-version rule at `CMakeLists.txt:481` and `:486` something you want to keep or relax (the question C03 rides on)?
offer: the pixi workspace and packages, with the `naga-acpp-*` names and the channel replaced before the PR
2 C07
venue: comment on PR #2282
question: The scan rewrites and the sub-group id and count change in `wg_reduce` are independent of #2282's identities, but our fractional and negative min/max tests rely on them. Should they come as a separate PR after #2282, and do you want the tests added there?
offer: the scan and `wg_reduce` changes and the tests, as a PR on top of #2282
3 C15
venue: comment on PR #2307
question: Our `propagateConvergence` also marks direct call sites convergent, and our lit test puts the barrier behind a user helper (expects one barrier call), where #2307's test calls `item.barrier()` directly. Do you want the call-site marking and the helper test folded into #2307, or as a follow-up PR?
offer: the call-site marking and the helper-barrier test
4 C06
venue: comment on PR #2305
question: PR #2305 appears to fix the same bug in `fixSingleSubCfgValues` (the alloca found in `RemappedInstAllocaMap` being overwritten by `getLoopStateAllocaForLoad`). Our change also stops recomputing side-effecting and memory-reading values after a barrier and restores split-loop live values, with four lit tests (issue #2264). Should ours go on top of #2305 once it merges, or be folded into it?
offer: the candidate's hunks and four lit tests, rebased on #2305
5 C01
venue: new issue (CONTRIBUTING.md line 3: open an issue when in doubt about design)
question: Do the maintainers want installs to be relocatable through an options, discovery and installed-configuration model that replaces build-machine paths (GNUInstallDirs layout, `$ORIGIN` rpaths, an installed toolchain and application config read by `acpp` and the runtime, run-time libmvec lookup, new `ACPP_<VENDOR>_ROOT` options and a changed vector-math default)? How should it be split, and how does it relate to #2283, #2268, #2160, #2131 and #1670?
offer: the whole candidate on U, in the parts agreed in the issue
6 C02
venue: new issue
question: Is an `async_host` queue and handler extension, with deferred issue of operations whose requirements belong to other queues or host tasks (host-function deferral on CUDA and HIP, waits on in-order queues, a retarget context), wanted in AdaptiveCpp, and how should it be split?
offer: the candidate with its documentation (doc/extensions.md, doc/runtime-spec.md) and tests, in the parts agreed in the issue
7 C05
venue: new issue
question: Is extending the dynamic-function extension (definitions in other translation units, indexed call sequences, host stand-ins for ELF, Mach-O and COFF) wanted, and does the extension's interface change need design discussion first?
offer: the candidate with its lit tests and doc/extensions.md text
8 C04
venue: draft PR as the venue itself
question: Is building AdaptiveCpp linked into the LLVM tools (`LLVM_ADAPTIVECPP_LINK_INTO_TOOLS`, openmp as an LLVM runtime, a Windows LLVM DLL) a configuration upstream supports, so that these fixes belong upstream?
offer: the candidate as a draft PR on U
joined C03
venue: C20's conversation (existing discussion #1981), joined; comment on PR #1986 once it has merged
question: Does upstream want LLVM 23 accepted in the version gate once #1986 has landed, given the rule that development versions are not supported (`CMakeLists.txt:481`, `:486`)? Asked together with C20's question.
offer: the LLVM 23 helpers and ports on top of U with #1986
```

## Next steps

One per candidate: type, owner, done-condition. Types: open conversation, prepare PR, seam rework, wait. The seam on C20 (the translator repository, KF4) is handled as part of its PR preparation after its conversation (rework recorded in candidates.md); the pending edges on C03 (#1986) and C07 (#2282) are in the done-conditions.

```steps
C01 | open conversation | Jack | the issue is open with the recorded question and the maintainers' answer is recorded in decisions.md
C02 | open conversation | Jack | the issue is open with the recorded question and the maintainers' answer is recorded in decisions.md
C03 | wait | Research Lead | C20's conversation (opened first) has the maintainers' answer on LLVM 23 and PR #1986 has merged
C04 | open conversation | Jack | the draft PR is open with the recorded question and the maintainers' answer is recorded in decisions.md
C05 | open conversation | Jack | the issue is open with the recorded question and the maintainers' answer is recorded in decisions.md
C06 | open conversation | Jack | the comment is posted on #2305 and the answer is recorded in decisions.md
C07 | open conversation | Jack | the comment is posted on #2282 and the answer is recorded in decisions.md
C08 | prepare PR | Research Lead | a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
C09 | prepare PR | Research Lead | a branch on U carries exactly the hunks of this candidate and the conflict in `omp_queue.hpp` is resolved against b5535e79
C11 | prepare PR | Research Lead | a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
C12 | prepare PR | Research Lead | a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
C13 | prepare PR | Research Lead | a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
C14 | prepare PR | Research Lead | a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
C15 | open conversation | Jack | the comment is posted on #2307 and the answer is recorded in decisions.md
C16 | prepare PR | Research Lead | a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
C17 | prepare PR | Research Lead | a branch on U carries exactly the hunks of this candidate and the conflict in `Utils.hpp` is resolved against bce257b0
C18 | prepare PR | Research Lead | a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
C19 | prepare PR | Research Lead | a branch on U carries exactly the hunks of this candidate and its textual trial merge is clean
C20 | open conversation | Jack | the comment is posted in discussion #1981 with the recorded questions and the maintainers' answers are recorded in decisions.md
```

## Keep-in-fork list

No next step: a keep-in-fork decision carries none. A trigger is the event after which the decision is revisited.

```keep
KF1 | fork identity | none | The project bank and its commit hook (`.project/`) and the `.gitignore` line for them are tooling of this fork's own operation and mean nothing to AdaptiveCpp.
KF2 | fork identity | none | The kernel-language, graph-API, vocabulary and research documents under `docs/naga-internal/` are NAGA Labs' own design documents, not documentation of AdaptiveCpp.
KF4 | transitional | none recorded (Review 1, U03) | The `GIT_REPOSITORY` line that selects `NAGALabsOrg/SPIRV-LLVM-Translator` points at the fork's branch of the translator for its pinned LLVM 23 snapshot, which exists only for the fork's own environment.
KF5 | transitional | a macOS libc++ run of upstream develop shows no teardown abort (Review 1, U05) | Making `small_vector` and `auto_small_vector` plain `std::vector` under libc++ works around the small-buffer allocator for the fork's macOS toolchain runs, and upstream's commit 96c7d17a (#2273) may make it unnecessary.
KF6 | transitional | #1986 merges (Review 1, U08) | The CI matrix entry, plugin-header move, `--enable-unsafe-fp-math` guards, lifetime-intrinsic argument fix, data layouts, Metal includes and install documentation for LLVM 22 are the same changes open PR #1986 makes, so the fork keeps them until that PR lands.
KF7 | transitional | #2282 merges (Review 1, C07 amendment) | The min/max identity helpers match what open PR #2282 makes, and the `wg_reduce` broadcast hunk is only a comment on a fix that U already has from #2301, so the fork keeps both until #2282 lands.
```

## What Review 2 decides

Conversation flags and questions, the order, the horizons and the next steps above. Points the draft leaves open for Jack: whether any candidate should be "after next release" given the release timing above; the reading of criterion (e) (candidates.md: open PRs of overlapping intent or content, not bare path overlap); C19's `CMAKE_DL_LIBS` line serving code that is in C01; and the seam C20 has on KF4.
