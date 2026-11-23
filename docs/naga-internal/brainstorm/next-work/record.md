# Next work on sg: goal-setting record

Plan sg-next-goals. Approval page: summary.md beside this file.

## 1. Process
- Carried out by the Research Lead alone, started 2026-11-23 14:55 UTC (plan locked by Jack: "That works. Go ahead").
- Contacts with Jack during the process: none other than presenting summary.md after the push.
- Statements attributed to Jack come only from writings that predate the start: note sg-redesign-decisions (entries to "LANGUAGE PASS CLOSED", 2026-11-23 ~14:45 UTC), the Research Lead's system prompt and "How you work" (standing guidance), the handoff of 2026-11-23 00:15 UTC (trajectory, scope correction), note toolchain-impl-design.
- Labels: observed = run or inspected in this process (raw output under /tmp/next-work-survey/, commands below); recorded = taken from a cited source.

## 2. Source ledger
| id | source | ref | used by |
|---|---|---|---|
| S1 | docs/naga-internal/brainstorm/language-pass/summary.md (v6) | dd6245df | F1, F2 |
| S2 | note sg-redesign-decisions, D2-D17 | entries to 2026-11-23 14:45 | F1, F2 |
| S3 | plan sg-redesign-brainstorm (contract U1-U11) | plan file, 2026-11-23 | F10 |
| S4 | prototype commits | e7b98d97, fdfea347, 0f305657 | F20-F26 |
| S5 | docs/naga-internal/01-04 | 35ff9985 | F11 |
| S6 | note toolchain-impl-design | entries 2026-11-19 to 2026-11-21 | F12 |
| S7 | note impl-research-findings | entries 2026-11-18 to 2026-11-20 | F13 |
| S8 | standing guidance: system prompt and "How you work" | as of 2026-11-23 | W1-W7 |
| S9 | trajectory: handoff 2026-11-23 00:15 UTC, "Scope correction" | handoff text | T1-T4 |
| S10 | plan sg-core and plan list | ListPlans 2026-11-23 | F21 |
| S11 | logs ~/logs/acpp-fork/sg-* | listed in /tmp/next-work-survey/06*.txt | F22-F25 |

## 3. Guidance (W) and trajectory (T)
| id | item | effect |
|---|---|---|
| W1 | compute is cheap, memory is not | on the next work's content (already in the agreed design); no effect on the choice |
| W2 | one visible step per line | on form: goals and page are plain single statements; no effect on the choice |
| W3 | reason from the problem, not the finished algorithm | on method of the later work; no effect on the choice (goals name no method) |
| W4 | new primitives rather than bent ones | already satisfied by the agreed design; no effect on the choice |
| W5 | don't hedge | on form: goals carry no fallbacks or extra scope; X list states what is left out |
| W6 | finished means committed and pushed | on the goals (G8) and criterion K1 |
| W7 | estimate in wall-clock time limited by builds and runs | criterion K4 |

| id | trajectory item | relation to the chosen work |
|---|---|---|
| T1 | porting the other platforms | follows it: a port needs a working toolchain of the agreed design |
| T2 | CI on the fork | follows it: CI needs tests of the agreed design to run |
| T3 | the pixi conversation | independent |
| T4 | posting the CBS fix upstream | independent, later, when Jack says |

## 4. Agreed design (from S1, S2)
Terms: work tile (one native work-group, fixed, D6/D7); work item (plain range launch item, D13/D14); run (a unit's 1-D items, D7, extra result dimension); segment (a run split across native groups, sg::segment_items, D10/D11); launch value (kh.launch_value(handle), deferred host read, bound only, D15-D17).
| id | status |
|---|---|
| D2 runtime sizes | holds |
| D3 elements and tiles | holds (join equal ranks, oob partial tiles per D13) |
| D4 index maps | holds; waiting per chosen tile per D9 |
| D5 tile_bin | holds in changed form (run collective) |
| D6 work tiles are native groups | holds as amended by D7 |
| D7 names, fixed groups, run dimension | holds |
| D8 ordering with run-time contract | superseded by D9 |
| D9 tile dependency graph, rejected if unorderable | holds |
| D10 segmented runs at construction | holds |
| D11 segmented work sets as a type | holds; its marking API superseded by D12 |
| D12 one kh.mark_local with sg::segmented mode | holds |
| D13 shorter runs, oob tiles, work items, array tiles local | holds |
| D14 work-item results only at own index | holds |
| D15 submit lambda is the builder; graph built once; no key, no hash; launch values replace promise/future | holds |
| D16 any work may produce a launch value; bound only | holds |
| D17 kh.launch_value; submit<kind> | holds |
The rejections the design states: work tile over the device cap; tile dependency cycle (includes a data-chosen pull within its own work set); wrong kh.mark_local mode; run_replicated on a distributed result of its own set; work tiles and work items in one kernel; a launch value used as structure; impure device code (W4 of the language-pass ledger).

## 5. Findings
| id | label | finding | evidence |
|---|---|---|---|
| F1 | observed | No primitive of the agreed design exists in code: work_tiles, map_work, map_items, segment_items, mark_local, slice_work, tile_bin, run_distributed, run_replicated, define_tiles each 0 hits in include/ src/ naga/. | `rg -F -o <name> include src naga`, /tmp/next-work-survey/04-sg-code.txt |
| F2 | observed | The code in the tree implements the superseded language: include/hipSYCL/sg/ (45 files, 5034 lines) with parallel_for (314 hits), map_groups (72), evaluate (80); compiler: CompositionStep.hpp 544 lines, BodyChecker.hpp 352 lines. | same file |
| F3 | observed | The prototype is 3 commits: `git rev-list --count e7b98d97..0f305657` = 2, `e7b98d97^..0f305657` = 3; 135 files, 13488 insertions. | 02-prototype-range.txt |
| F4 | observed | Tests: naga/sg-tests (71 files, 6421 lines), targets sg_host_tests, sg_compiler_tests, sg_driver_tests, sg_tests, sg_jit_bench; cells host, omp, cuda, ocl, ze, lit. | 05-tests.txt |
| F10 | observed | Of the brainstorm's artefacts (S3 U1-U11) only the language work exists, as kernel-language-proposal.md, kernel-model.md and language-pass/; no file for the device summary, requirements, change register, evidence, composition model, evidence trace, side topics, open questions or note (name search: 0 matches for baseline, device-summary, requirements, changes, evidence, side-topics, open-questions). | 08-brainstorm.txt, 08b |
| F11 | observed | Fork docs 01-03 carry "Status: agreed 2026-11-18", before the builder frame (2026-11-19) and the work-tile language (2026-11-23); see section 7. | 07-fork-docs.txt |
| F12 | recorded | toolchain-impl-design: established: builder frame, structure vs bound values, one launch, pool and memory model, tests as a separate package (S6, 2026-11-19 to 2026-11-21); open: the lowering of the work-tile language (its mechanism entries describe layers and claim/shortcut/wait, superseded by D9). | S6 |
| F13 | recorded | impl-research-findings: fork facts that still hold (SSCP flow, dynamic functions and indexed call sequences, annotation channel, CBS inlining, per-backend events, local memory per backend, atomics rules: 32-bit flags, acquire/release); its ordering proposals predate D9. | S7 sections A-T, WW, XX |
| F20 | observed | The three prototype commits landed within 23 seconds (01:13:10 to 01:13:33, 2026-11-23) after days of uncommitted work. | 02-prototype-range.txt |
| F21 | observed | Plan sg-core shows "implementing, 18 continuations"; its mechanism design was rewritten (versions 1, 2, the "no scheduler" correction, the agreed core) while work ran (S6 entries 2026-11-20 20:42 to 2026-11-21 03:24). | ListPlans; S6 |
| F22 | recorded | Compile cost: cuda ten_layers 10445-11450 ms, chain_30 318406-322465 ms cold before the noinline change; after: 1165-1172 ms and 15988-16278 ms. | sg-jit-before/after results.csv |
| F23 | recorded | ze chain_30 and layers_8_16_32 are "crash" in both runs; ocl chain_30 "infeasible". | same |
| F24 | recorded | The last sg_tests omp log stops entering "global_chain_of_30" (no completion line). | sg-tests-20261122-204917/omp-sg_tests.log |
| F25 | observed | The sg-tests environment's OpenCL ICD link (etc/OpenCL/vendors/ocl-icd-system -> /etc/OpenCL/vendors) is dated 2026-11-22 15:23 -0800, after both jit runs (14:36 and 15:14 -0800): their ocl rows ran without the system ICDs. | 09-opencl-vendors.txt, 06b |
| F26 | recorded | 8/16-bit compare-exchange was not portable (cuda misaligned fault, ze no i8/i16 atomics) and host buffers were freed under in-flight copies; both patched in the prototype (e7b98d97). | handoff 2026-11-23 "Portability and plan gaps" |

## 6. Prototype build account (S4)
Built: graph runtime and builder of the superseded language (e7b98d97), composition step and checker with a layer-call noinline policy (fdfea347), test package, bench and measurement tasks (0f305657).
Process faults:
- PF1 Work held uncommitted for days, landed as one checkpoint (F20; e7b98d97, fdfea347, 0f305657).
- PF2 Design settled during implementation (F21; e7b98d97).
- PF3 Effort beyond what was asked: measurement tooling and a bench for a prototype that Jack found far too rigorous (0f305657).
- PF4 A test cell measured the wrong device without the runner noticing (F25; 0f305657 sg-test-run.nu has no device check).
Outcome faults:
- OF1 Quadratic compile cost from per-layer cloning plus global always-inline (F22; fdfea347).
- OF2 OpenMP deep chains infeasible: CBS inlines every barrier-containing callee (F24; fdfea347).
- OF3 Level Zero fails on deep chains (F23; fdfea347).
- OF4 IGC crash on a small kernel, layers_8_16_32 (F23; 0f305657 bench case).
- OF5 Portability gaps in atomics and buffer lifetime (F26; e7b98d97).

## 7. Lessons, backends, docs
Lessons (current status): L1 nesting cost (F22, recorded); L2 barrier-callee inlining on CPU (F24, F13, recorded); L3 forward progress per backend (F13, D9, recorded); L4 Level Zero hang from 20 layers (F23, recorded); L5 IGC crash (F23, recorded); L6 OpenCL on host in the test environment (F25, observed: link present now).
| backend | prototype status | label |
|---|---|---|
| OpenMP (omp) | passes except global_chain_of_30 | recorded (F24) |
| CUDA (omp;cuda) | passes; chain_30 slow before noinline | recorded (F22) |
| OpenCL (ocl) | measured without system ICDs; chain_30 infeasible | recorded (F23, F25) |
| Level Zero (ze) | chain_30 and layers_8_16_32 crash | recorded (F23) |
| host cell | unit tests only | recorded (F4) |
Docs (F11): 01 sections 1-15 all stale (parallel_for, mappings, evaluate, cycle/chain, body rules of the superseded language). 02: 1 holds (purpose); 2 stale (D15); 3 stale (handles on a graph, no size); 4 holds in principle (ordering by roles); 5 holds (inplace) except its form; 6 stale (get_host_handle, realloc); 7 stale (D15); 8 holds (undesigned); 9 holds (graph cycle, deferred); 10 stale (D15); 11 holds; 12 stale (report removed); 13 stale (no host IR); 14 history; 15 partly stale. 03: 1-11 stale (names of the superseded language). 04: 1, 2, 9, 14 hold (fork facts); 3-8, 10-13 stale (proposals predating the builder frame and D9).

## 8. Gaps (code, tests, docs, brainstorm)
- GP1 code: none of the agreed primitives exists (F1); the tree holds the superseded language (F2).
- GP2 code: the lowering of the work-tile language (tile graph ordering, segments, local array tiles, work-item kernels, launch values) is undesigned and unbuilt (F12).
- GP3 tests: none for the agreed language; the runner does not check the device (F4, F25).
- GP4 docs: 01-03 describe the superseded language (F11).
- GP5 brainstorm: composition model, evidence trace, side topics and closing note do not exist (F10).
- GP6 code: the prototype's open faults OF1-OF4 (F22-F24).
Open items from S6/S7: OI1 lowering mechanism under D9 (F12); OI2 compile-time checker coverage of positions functions (language-pass record U16 finding); OI3 graph modification (D15, undesigned).

## 9. Criteria (fixed before candidates)
| id | criterion | basis | judged by |
|---|---|---|---|
| K1 | ends with something committed and pushed that later trajectory work needs | W6, T1, T2 | 2 = a working toolchain of the agreed design; 1 = a design or document only; 0 = neither |
| K2 | retires a risk the prototype showed (OF1-OF5) | F22-F26 | 2 = settled by running on the backends; 1 = argued only; 0 = none |
| K3 | can start without a decision from Jack | D2-D17, F12 | 2 = yes; 1 = needs a small decision; 0 = needs a design conversation |
| K4 | wall-clock cost in builds and runs | W7 | 2 = no toolchain rebuild; 1 = under 5 rebuilds; 0 = more |
Ordering: K1, then K2, then K3, then K4, compared lexicographically on scores. Tie-break: fewer toolchain rebuilds.

## 10. Candidates
| id | candidate | achieves | leaves undone | cost (builds, runs) | rests on |
|---|---|---|---|---|---|
| C1 | implement the agreed language, lowering included, and bring docs 01-03 in line (Jack's consideration) | GP1, GP2, GP3, GP4, GP6 for the new code; OI1, OI2 | GP5 as separate artefacts; OI3 | many rebuilds of naga-acpp-local (~17 min each with ccache) plus four test cells per round (~30-60 min) | F1, F2, F12, F22-F26 |
| C2 | design the lowering of the work-tile language as a document first | GP2 on paper, OI1 | GP1, GP3, GP4, GP6 | none | F12, F13 |
| C3 | rewrite docs 01-03 to the agreed design | GP4 | GP1, GP2, GP3, GP5, GP6, OI1-OI3 | none | F11 |
| C4 | fix the prototype's backend faults in the superseded code | GP6 for code that will be replaced | GP1-GP5, OI1-OI3 | several rebuilds and runs | F2, F22-F24 |
| C5 | finish the brainstorm's remaining artefacts | GP5 | GP1-GP4, GP6 | none | F10 |

## 11. Matrix and choice
| | K1 | K2 | K3 | K4 |
|---|---|---|---|---|
| C1 | 2 (T1, T2, W6) | 2 (F22-F26) | 2 (D2-D17; lowering is the Research Lead's per S6 division of labour) | 0 (W7) |
| C2 | 1 (W6) | 1 (F13) | 2 (F12) | 2 (W7) |
| C3 | 1 (W6) | 0 | 2 (F11) | 2 (W7) |
| C4 | 0 (F2: code to be replaced) | 2 (F22-F24) | 2 | 1 (W7) |
| C5 | 1 (W6) | 0 | 0 (U1-U11 need Jack's confirmation per S3) | 2 (W7) |
The rule picks C1 (alone at K1 = 2).
Why not the others: C2 and C3 rank lower on K1 (a document only) and on K2 (C2 argues, C3 retires nothing); their content is part of C1. C4 ranks 0 on K1: it fixes code the agreed design replaces (F2). C5 ranks lower on K1, K2 and K3; its language parts were superseded by the language pass and its model part is inside C1.
Jack's consideration (implement what was specced, more rigorously) is C1, scored on the same criteria; adopted because it alone scores K1 = 2 and K2 = 2 (F1, F22-F26), not because Jack raised it.

## 12. Chosen goals (C1)
- G1 A program using every primitive of the agreed summary (docs/naga-internal/brainstorm/language-pass/summary.md, v6) compiles with the fork's toolchain. (F1)
- G2 Every kernel in docs/naga-internal/brainstorm/language-pass/kernels.md gives its stated reference result on OpenMP, CUDA, OpenCL on the Intel GPU, and Level Zero. (F1, F23)
- G3 Every program the agreed design rejects fails with an error naming the offending construct and its numbers. (section 4)
- G4 A device function containing a loop, recursion or a write outside itself fails to compile with a diagnostic naming the construct. (F2, OI2)
- G5 On every backend, a chain of 30 dependent work sets runs to its reference result, with cold JIT time and device-code size at most 3.5 times those of a chain of 10. (F22-F24)
- G6 Running one graph twice, changing only handle contents and launch values, gives both reference results with no new JIT compilation on the second run. (D15)
- G7 docs/naga-internal/01-kernel-language.md, 02-graph-api.md and 03-vocabulary.md state the agreed design and contradict nothing in summary v6. (F11)
- G8 All of it is committed and pushed to naga/develop. (W6)

## 13. Rigour (applies: G1-G6 run on changed code, tests and build)
- R1 Each completed unit of the implementation plan is its own pushed commit, and nothing from the work is left uncommitted. (PF1)
- R2 The delivered interfaces match the implementation plan's contract; any difference is recorded with its reason. (PF2)
- R3 Every file the work adds is named by a unit of the implementation plan. (PF3)
- R4 Every test log names the device each cell ran on, and a cell that ran on a device outside its backend fails. (PF4)
- R5 Composed device code for a 30-work-set chain is at most 3.5 times that of a 10-set chain. (OF1)
- R6 The 30-set chain compiles and runs on OpenMP. (OF2)
- R7 Chains of 20 to 30 work sets run on Level Zero. (OF3)
- R8 No kernel of the set crashes a backend compiler. (OF4)
- R9 Kernels combining 8- and 16-bit elements, and twenty back-to-back runs of one graph, give their reference results. (OF5)

## 14. Out of scope
- X1 T1 porting other platforms, X2 T2 CI on the fork, X3 T3 the pixi conversation, X4 T4 the upstream CBS fix: trajectory, not this work.
- X5 OI3 graph modification: undesigned (D15).
- X6 GP5 the brainstorm's remaining artefacts as separate files: superseded by the language pass; the model lives in the implementation.
- X7 C4 fixing the superseded prototype code: replaced by C1.
- X8 Features deferred earlier and absent from summary v6 (kernel cycles and branches, graph cycle, interop, several live graphs).
