# Upstreaming triage: decisions

Pins: U `27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc`, N `0e2bac974c40554f62286cbffe1f06c87cc9caa9`, B `9ec394efa584ced004d644af2194d2133e111d26`; merge tree M `350667e47d62407ad9bef5977bdab5a632ae8e87` (see baseline.md).

## Review 1

Held 2026-11-23. Jack approved every Research Lead recommendation as presented, with the amendments below. Dissent noted by the Research Lead: none. The records were amended as listed (candidates.md and keep-in-fork.md carry the result).

Resolutions of the unresolved items of the first run:

| item | decision | effect on the records |
|---|---|---|
| U01 adjacency and pickaxe hunks | accept | none |
| U02 entangled hunks in C01 | hand-split: C17 owns `suppressChildErrorDialogs` and `describeToolExitCode` and requires nothing; the `CMAKE_DL_LIBS` line of `llvm-to-backend/CMakeLists.txt@200,21` moves to C19 | hunks `Utils.cpp@23,12`, `Utils.cpp@53,58` and `llvm-to-backend/CMakeLists.txt@200,21` split by line between C01, C17 and C19; the C17 edge on C01 is removed |
| U03 fork URL in a C01 hunk | the `NAGALabsOrg` translator URL line becomes a new keep-in-fork decision (KF4), transitional; the generic `LIST_SEPARATOR`, long-path and tag lines stay in C01 | hunk `llvm-to-backend/CMakeLists.txt@268,7` split by line between KF4 and C01; C01's seam is resolved |
| U04 runs in added files | leave in C01 | none |
| U05 C10 and 96c7d17a | the whole of C10 becomes keep-in-fork (KF5), transitional; trigger: a macOS libc++ run of upstream develop shows no teardown abort | C10 retired |
| U06 seam C20 and KF3 | KF3 (pins and patches) moves into C20 | KF3 retired; C20 owns `packages/core/patches/**` and `packages/core/pins.toml` |
| U07 C20 contested | a candidate flagged for conversation; renaming the `naga-acpp-*` packages and the channel is preparation for its PR | C20 text |
| U08 C03 and LLVM 23 | the LLVM 22 hunks that #1986 also makes go to a keep-in-fork decision (KF6), transitional, trigger "#1986 merges"; C03 becomes the LLVM 23 helpers with an upstream-pending edge on #1986; its gate and development-version question is part of C20's conversation | C03 reduced; KF6 added; pending edge and seam recorded |
| U09 C04 contested | stays a candidate | none |
| U10 keep-in-fork boundaries | accept; `.gitignore@4,2` stays with C20 | none |
| U11 kinds | accept, except C19 becomes bug fix | C19 kind and title |
| U12 large flags and splits | keep the large flags on C01, C02 and C20; the splits are decided in the conversations | none |

Further amendments of the note:
- C07: the identity hunks covered by #2282 go to keep-in-fork (KF7), transitional, trigger "#2282 merges"; the broadcast-type hunk that #2301 already did goes with it. C07 becomes the scans, the sub-group id and count in `wg_reduce`, and our tests, with an upstream-pending edge on #2282.
- C15: stays a candidate, flagged for conversation, venue a comment on PR #2307 offering the call-site marking and the helper-barrier test.

Entry by id (each approved as presented unless it says amended):

| id | entry |
|---|---|
| C01 | approved; amended: the `CMAKE_DL_LIBS` lines and the C17 definitions are out of it (U02), the translator URL line is out of it (U03); large flag kept (U12) |
| C02 | approved; large flag kept (U12) |
| C03 | amended: LLVM 23 helpers only, pending edge on #1986, the LLVM 22 hunks are in KF6 (U08) |
| C04 | approved; stays a candidate (U09) |
| C05 | approved |
| C06 | approved |
| C07 | amended: scans, sub-group id and count, tests; pending edge on #2282; identity and broadcast hunks are in KF7 |
| C08 | approved |
| C09 | approved |
| C10 | amended: whole candidate is keep-in-fork KF5 (U05); id retired |
| C11 | approved |
| C12 | approved |
| C13 | approved |
| C14 | approved |
| C15 | approved; stays a candidate, flagged, venue a comment on PR #2307 |
| C16 | approved |
| C17 | amended: owns the two definitions and requires nothing (U02) |
| C18 | approved |
| C19 | amended: kind bug fix (U11); owns the `CMAKE_DL_LIBS` line (U02) |
| C20 | amended: owns the pins and patches (U06); flagged for conversation, renaming is preparation for its PR (U07); large flag kept (U12) |
| KF1 | approved; `.gitignore@4,2` stays with C20 (U10) |
| KF2 | approved |
| KF3 | amended: moved into C20 (U06); id retired |
| KF4 | new at Review 1 (U03): the translator repository line, transitional |
| KF5 | new at Review 1 (U05): C10, transitional, trigger as above |
| KF6 | new at Review 1 (U08): the LLVM 22 hunks #1986 also makes, transitional, trigger "#1986 merges" |
| KF7 | new at Review 1 (C07 amendment): identities and broadcast hunk, transitional, trigger "#2282 merges" |

Unresolved list after Review 1: none.

## Review 2

none (not yet held)
