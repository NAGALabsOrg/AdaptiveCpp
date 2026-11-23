# Next work on sg: for approval

**Chosen:** implement the agreed language (summary v6), its lowering included, and bring docs 01-03 in line with it. This is your "implement what we specced, more rigorously"; it won on its own score, as the only candidate that ends in a working toolchain your next steps (porting, CI) need, and that settles the prototype's failures by running them. Rejected: a lowering document only, a docs rewrite only, fixing the old prototype code, finishing the old brainstorm files.

**Goals**
- G1 A program using every primitive of the agreed summary (docs/naga-internal/brainstorm/language-pass/summary.md, v6) compiles with the fork's toolchain.
- G2 Every kernel in docs/naga-internal/brainstorm/language-pass/kernels.md gives its stated reference result on OpenMP, CUDA, OpenCL on the Intel GPU, and Level Zero.
- G3 Every program the agreed design rejects fails with an error naming the offending construct and its numbers.
- G4 A device function containing a loop, recursion or a write outside itself fails to compile with a diagnostic naming the construct.
- G5 On every backend, a chain of 30 dependent work sets runs to its reference result, with cold JIT time and device-code size at most 3.5 times those of a chain of 10.
- G6 Running one graph twice, changing only handle contents and launch values, gives both reference results with no new JIT compilation on the second run.
- G7 docs/naga-internal/01-kernel-language.md, 02-graph-api.md and 03-vocabulary.md state the agreed design and contradict nothing in summary v6.
- G8 All of it is committed and pushed to naga/develop.

**Rigour** (one per prototype fault)
- R1 Each completed unit of the implementation plan is its own pushed commit, and nothing from the work is left uncommitted.
- R2 The delivered interfaces match the implementation plan's contract; any difference is recorded with its reason.
- R3 Every file the work adds is named by a unit of the implementation plan.
- R4 Every test log names the device each cell ran on, and a cell that ran on a device outside its backend fails.
- R5 Composed device code for a 30-work-set chain is at most 3.5 times that of a 10-set chain.
- R6 The 30-set chain compiles and runs on OpenMP.
- R7 Chains of 20 to 30 work sets run on Level Zero.
- R8 No kernel of the set crashes a backend compiler.
- R9 Kernels combining 8- and 16-bit elements, and twenty back-to-back runs of one graph, give their reference results.

**Out of scope:** porting, CI, the pixi conversation, the upstream CBS fix (your trajectory); graph modification (undesigned); the old prototype code; deferred features (kernel cycles and branches, graph cycle, interop).

**Asked of you:** approve, or correct any line above by quoting it. Full record: docs/naga-internal/brainstorm/next-work/record.md.
