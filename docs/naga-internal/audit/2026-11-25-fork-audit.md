# Audit of the fork at d888afd9 (2026-11-25)

Status: a record of an audit by the Research Lead for Jack, carried out on 2026-11-25 at head d888afd9 of naga/develop. The audit only looked: nothing in the repository, its remote, its manifests or the installed environments was changed, and no build was started. Everything under "Odd, concerning, broken, worth exploring" is a thing to note for discussion; none of it is decided, planned or started. Scratch listings and the scripts that made them are untracked under .pixi/sg-experiments/fork-audit/ on the machine the audit ran on.

## Evidence labels

- OBSERVED: read in the tree, the git history or a log, with the place given.
- INFERRED: reasoned from observed facts; not run.
- RECORDED: already in a note, the work manifest or a commit message; not verified again.
- NOT EXAMINED: not looked at, with the reason.

## Report of the exploration

Carried out 2026-11-25 at head d888afd9 in two passes. Nothing was changed: three read-only surveys (their scripts and CSVs are untracked scratch under .pixi/sg-experiments/fork-audit/), the Research Lead's own reads, and web reads.

### What was explored

1. History: all NAGA commits by subject (372 since the merge base), the fork point, the one upstream merge, per-area commit lists (cbs, sscp, llvm-to-backend, pcuda, include/hipSYCL/sycl).
2. Tree: every top-level area sized; the diff against 9ec394ef by area and per file for runtime and compiler.
3. Graph API: every header under include/hipSYCL/sg (50) and include/hipSYCL/compiler/sg read in full; CompositionStep, composition.hpp, composition_adaptivity, the LLVMToBackend diff, the jit.hpp diff; docs 01, 02, 03, 04, README, summary v6; naga/work-manifest.md; common/graphs.hpp (K1 to K14), the sg-tests CMakeLists; test coverage by shape through a survey.
4. Runtime: inorder_executor, async_host_executor, adaptivity_engine, kernel_configuration, inorder_queue.hpp, ocl_queue (330 to 529), ze_queue (200 to 529), dag_manager, schedulers (parts), the kernel_cache diff, glue/error.hpp.
5. Build: the root manifest; every file under packages/core except the patches; slices runtime and naga-acpp; extras mutex, runtime-cuda, runtime-rocm; activation clangxx and render-install.nu; checks/consumer; the root CMakeLists.txt in full; cmake/discovery.cmake and cmake/options/common/core.cmake in full; acpp-installed-configs.cmake (head); the installed acpp-toolchain.json and acpp-app.cfg; the full bin/acpp diff; the removed configuration-model document (first 200 lines).
6. Tasks: every script under naga/tasks; which a task runs; the newest logs; the newest sg-jit results.csv.
7. Upstream: the architecture, compilation, SSCP, installing, LLVM and runtime-spec pages; issue #2024; the 22 commits on the local upstream/develop ref; three upstream fixes compared hunk by hunk against HEAD.

### Web research

Yes, limited to upstream's own documentation and one issue, for upstream's stated intent. The repository is the authority for everything about the fork.

### Leads that were checked

- Uncommitted sg work: false.
- "The queue is bypassed in the sg path": false; everything goes through one retargetable out-of-order sycl::queue.
- "About 1,000 lines over 42 runtime files": 1,201/-153 over 42 files in src/runtime plus 620/-51 over 42 runtime headers.
- "User kernels are transpiled into native SYCL kernels": the stale README's wording; what is built is one generic SYCL kernel per launch with task functions composed at JIT.
- Unit table C1 to C32: it is C1 to C39.
- OpenCL on the host through the ICD link: resolved earlier (RECORDED); cells verify their device.
- LLVM mismatch with upstream: confirmed.

### Why the exploring is sufficient

Every top-level area is placed and sized; every area NAGA changed is identified with its size and purpose; the two areas that carry the risk (the graph API and the runtime's ordering of host and device work) were read in full; each inference of the first pass was either confirmed by reading or kept labelled.

NOT EXAMINED, none of which can be settled without a build or hardware not at hand: the per-vendor and per-platform files under cmake/options, cmake/install and config/ beyond core; the Windows, macOS, Android, HIP, Vulkan and Metal paths; the reflection and dynamic-function passes beyond their headers and commit subjects; pcuda beyond commit subjects; the nine LLVM patches; pixi.lock; whether any sg test throws inside host work.

## Findings

### A. The work completed

OBSERVED. Fork point: 02415454 (2026-09-03) on upstream f2600750; upstream merged once at 6a9c3a19 (2026-10-15, base 9ec394ef); 372 commits since; 365 files added, 162 modified, +39,069/-1,715. Phases and purpose:

1. Sep 3-5, runtime: retargeted command groups resolve against their own context and executor; async_host (host work as a first-class operation with an event); operations that need completed requirements are issued only once those complete (CUDA, HIP); code objects built outside the kernel-cache lock with in-flight deduplication.
2. Sep 7-26, relocatable toolchain and the configuration model: no build-machine path in any artifact. cmake/discovery (every find up front, ACPP_DISCOVERED_*), cmake/options (per platform and architecture; vendor units with ACPP_BUNDLE_<V> and ACPP_<V>_ROOT), cmake/install (vendor install rules), config/ templates, three generated files in etc/AdaptiveCpp (acpp-toolchain.json with {{ }} keys, acpp-app.cfg with $ACPP_RT_LIB_DIR, acpp-deploy.json), settings resolved env then bundled then config, the driver rewritten to match (config keys renamed, a single deploy manifest, /etc/AdaptiveCpp dropped). About 5,000 lines in cmake and config, root CMakeLists +343/-359, bin/acpp +400/-201.
3. Sep 28 - Oct 22, portability and compiler: the LLVM 23 port (a750ab07), the NAGA SPIR-V translator fork, libc++/Xcode, Windows/MSVC/COFF, Android; correctness fixes: CBS split-loop state (22422eee), CBS side-effecting recomputation (#2264, 0e2bac97), convergence propagated before pre-inline simplification (3fc30594), work-group reduce and scan (362b077d, 058d037d, e38ab468), min/max identities, OpenCL C 2.0 for SPIR-V atomics, string settings with spaces (1450b815), pcuda null checks and MSVC names; reflection through forwarded function pointers (Darwin C1 to C2); dynamic functions across translation units and on COFF; indexed call sequences (f815ee1e).
4. Nov 12-21, pixi: the workspace, the test tasks, the split into pixi-build-naga packages.
5. Nov 18-24, the graph API: design documents, prototype, language pass (D2-D17), implementation in 39 unit commits; with it deferral on OpenCL and Level Zero, async_host on std::async, the CBS alignment fix (c000e542), Level Zero region copy and status query.

### B. The codebase as a whole

OBSERVED. 1391 tracked files, 228,413 lines. Upstream's four parts stand: the SYCL interface (include/hipSYCL/sycl), the runtime (src/runtime, backend plugins), the compiler (src/compiler: clang frontend action, CBS, SSCP stage 1, llvm-to-backend stage 2) and glue.

A kernel: clang (AdaptiveCpp linked in; the driver passes no plugin flag, "plugin-linked-into-llvm" is true) extracts device IR per translation unit into an HCF object with a random id; at run time the runtime picks a backend queue, the adaptivity engine forms a kernel configuration id, the kernel cache finds or JIT-compiles a code object through llvm-to-backend (PTX, SPIR-V, amdgcn, host library with CBS), and the queue launches it. NAGA's runtime divergence is about 1,800 lines.

The graph API sits above sycl::queue: 50 headers, 7,047 lines, header-only, using the public SYCL API plus four hooks: the composition annotation, the composition step at stage 2 (LLVMToBackend.cpp 349 and 374), the composed-kernel identity in the adaptivity engine, and the purity checker in the frontend (Frontend.hpp 281, 811; it runs for translation units that instantiate an sg_body_entry).

Submit records a node (roles plus a kernel graph: views, work sets, maps, marks, sinks, launch slots). run() is ordinary C++ on the calling thread, per node: read launch values (waits on their producers), prepare if they changed (expand repeats, bind sizes, run every unit body on the host, build the tile dependency graph, order tickets), check roles, allocate or migrate handles, encode the bound block, take toolchain memory from the pool, then three queue operations: block upload, flag fill (a fixed 128-wide kernel), the generic kernel on a 1-D nd_range of groups x native size. Host work goes through handler::async_host. Ordering is depends_on between events. On the device each native group claims a ticket with an atomic, copies marked tiles to local memory, runs its task list and waits on done flags (relaxed poll, one acquire fence). Stage 2 replaces one placeholder with a function calling every task with a constant index; then every non-kernel function is always-inlined (GlobalInliningAttributorPass.cpp 21-42, upstream's), so the final module is one function (results.csv of sg-jit-t5-20261124-212803: ir_functions = 1 at chain 10 and 30).

Design against implementation. Matches: the builder runs at submit and a graph is built once (D15); submit kinds, interop rejected (D17, rule 15); launch values as bound values only, refused as structure at compile time (D16, rule 8); the growing pool with the no-new-ordering reuse rule (D18); tile graph ordering with cycle rejection (D9); the relaxed poll with a fence (D20); segmented sets as their own type and the mark rules (D11, D12); one native size per launch (W8); fifteen rejections. Differs: see D-G2 to D-G6 below; "recomputed" values are realised in local memory (RECORDED deviation); the native shape is always flattened to one dimension, so rule 2 is a language rule only; there is no host IR and no host JIT (04 proposed them).

Tests: upstream suites (sycl 827 cases, rt, pstl, pcuda) through test-omp, test-cuda, test-ocl, test-ze; the sg package: host (structure against mocks), compiler (checker, composition at IR level), driver (the device driver in isolation), device (K1 to K14, collectives, segments, in-place, errors, chains, reruns), lit (24 fail, 3 pass, 1 fail_sg), each device cell verifying its device.

### C. The build system

OBSERVED. One build: packages/core runs build-linux.nu under pixi-build-naga 0.1.0 (script mode, channel prefix.dev/jackm97/naga-labs, source at /home/jack/Projects/naga-build-backend). It fetches pinned sources into the durable .build tree (ROCm/llvm-project 0bace190 with 9 patches, OpenCL headers, TheRock ROCm nightly 10.1.0a20260822), configures LLVM with AdaptiveCpp as an external project linked into the tools, builds libLLVM first, then everything, installs into the host prefix, checks licences (packages/core/licenses, reached through the `shared` symlink), checks compiler-rt is not hollow, removes unversioned .so links, copies the ROCm runtime subset in, and installs clang cfg files. Projects clang, lld, lldb, clang-tools-extra, bolt, polly; runtimes compiler-rt, openmp; targets X86, NVPTX, AMDGPU, SPIRV; backends OpenMP, CUDA, OpenCL, Level Zero, ROCm; ACPP_<V>_ROOT=. so every vendor path is relative to the runtime library.

Packages: core (unpublished); six slices carved by file globs (runtime, naga-acpp, compiler-rt, llvm-dev, tools, lldb); extras (the LLVM-lane mutex naga-acpp-llvm 23.0git, runtime-cuda, -level-zero, -ocl, -rocm); activations clang and clangxx (rendered from vendored conda-forge templates); checks/consumer; the root meta package over the six slices. naga-acpp's strong run exports put the runtime and the LLVM lane on every consumer.

Environments: default (toolchain, four vendor runtimes, test dependencies), dev (core's build and host closure without building), sg-tests (default plus naga-sg-tests and lit), check-activation, check-bare, check-consumer, check-llvm-dev, git. A change under CMakeLists.txt, bin, cmake, config, include, src or packages/core makes pixi rebuild core incrementally (per-host-prefix CMake trees, ccache 30G; about 17 minutes RECORDED), re-carve the slices and reinstall an environment the next time it is used; tests/, naga/, doc/ and docs/ do not trigger it. naga-sg-tests (pixi-build-cmake) rebuilds on its own sources or a new toolchain. Upstream's tests/ are built by the test-build task into .pixi/naga-tests, not as a package.

Task scripts:

- check-devices: acpp-info must list devices on four backends.
- test-build: configure and build the rt, sycl, pstl and pcuda tests against the default environment.
- test-run: cells omp, cuda, ocl, ze, sanitizers; 1800 s limit; logs under ~/logs.
- sanitizer-smoke with sanitize.cpp: address and thread link and run clean, address must fire.
- check-packages: eleven checks (partition, contents, dependencies, linking, symlinks, build paths, activation round trip and compile-link-run, bare toolchain, export round trip, name collisions against conda-forge over the network, find_package with llvm-dev alone).
- conda-paths: helper of check-packages.
- sg-test-run: cells host, omp, cuda, ocl, ze, lit; device line verified; fresh appdb; kill-after.
- sg-jit-measure, sg-jit-lib, sg-jit-report: cold chain measurement.
- doc-consistency, primitive-coverage, sg-conformance: tools of the retired sg-impl plan.
- tests/*.test.nu: tests of the tools.

Known behaviours (RECORDED, seen again): removed source files survive in the installed package (pool_config.hpp); pixi reinstall does not force a rebuild; the ROCm copy happens after install, so a fresh prefix with a kept tree fails.

### D. Odd, concerning, broken, worth exploring

#### Graph API

- **D-G1** OBSERVED + INFERRED. Long-running graphs grow without bound. sycl_device_ops::events_ never erases (sycl_device_ops.hpp 175-178); handle_state::users gains an entry per use and per production every run and is cleared only on reallocation or migration (memory_manager.hpp 149-156); pool::release copies that list onto every 1 KiB node of the block (pool.hpp 84-91), which is what ~memory_manager does at destruction; launcher::kept_ and graph_impl::staged_ are freed only at wait(); every pool allocation scans every node and asks whether each recorded user finished (pool.hpp 135-164). By arithmetic: a 100 MB handle kept for 1,000 runs releases about 2e8 ids, 1.6 GB, at destruction. Not run: the largest number of runs of one graph in any test is 20.
- **D-G2** INFERRED (kernel.hpp 443-458). Preparation calls every tile-map device function once on the host with zero-valued inputs to learn its sizes. A device function that divides or takes a modulo by an integer input traps on the host; one that reads through a captured device pointer faults. No test has such a function.
- **D-G3** OBSERVED. In-place results work only for work-item kernels: kh.evaluate(sg::inplace(...), elements) throws "in-place results take work-item arrays" for joined result tiles (kernel.hpp 390-391); 01 section 6 states no such limit; it is not in the manifest's deviations; no test tries it; backing_plan.hpp 142-144 is dead because of it.
- **D-G4** OBSERVED. Work items read only elements of sources and earlier result arrays (item_map.hpp 88-89): not array tiles, index maps or stencils, which 01 section 4 and D13 allow. Offsets exist only in the first dimension (view.hpp 216); an out-of-range read yields zero silently (range_kernel.hpp 39-43); ranges of rank 2 and 3 are untested.
- **D-G5** OBSERVED. graph::run blocks on the host while a launch value's producer is in flight (graph_impl.hpp 225-238); 02 section 2 says run does not block. A floating-point launch value is accepted but nothing can use one (only integral ones convert to bound expressions); it still forces the host read.
- **D-G6** OBSERVED. Smaller divergences from the documents: the trim of a join over partial tiles is done for a join directly over kh.tile and, for map results, by a 1-D special case (kernel_preparer.hpp 215-236), so 2-D and 3-D partial tiles through a map are not trimmed (untested; view_node::trim is unused); kh.repeat with a count below 1 runs once (repeat.hpp 126-128; counts 0 and 1 untested); kernels with a repeat never reuse selections (kernel_preparer.hpp 135); the W1 row of 01's ledger still says "a native work-group"; rule_info has no number names (RECORDED).
- **D-G7** INFERRED. A graph has no error channel. An exception in host work becomes an application-wide asynchronous error (async_host_executor.cpp 73-87); the graph's queue has no handler, and g.wait() calls wait_and_throw (sycl_device_ops.hpp 138-141), whose default handler prints and calls std::terminate (glue/error.hpp 61-68); the work that depended on the failed node still runs. That the queue falls back to that default handler was not read in queue.hpp.
- **D-G8** OBSERVED. The units of one map are compared by the raw bytes of their closures (kernel_preparer.hpp 466, closure.hpp 24-27), padding included, so captures with padding between them can differ by chance and raise "the units of one map must run the same function with the same captures". Every test captures scalars of one type.
- **D-G9** OBSERVED. The purity checker exempts system headers and any namespace named sg (BodyChecker.hpp 39-49). Standard-library algorithms with loops inside a device function pass, and a lit case asserts it (lit/pass/library_loops_not_reported.cpp); 01 section 9 says the function and everything it calls is loop-free. Calls to system functions without a body (printf, malloc) are not reported. INFERRED: a named function object passed to sg::reduce, inclusive_scan, tile_bin or inplace is called inside sg code and so is not walked; no lit case covers it.
- **D-G10** OBSERVED + RECORDED. Device waits spin without bound; correctness of every multi-tile kernel rests on "a started group keeps running" (04 section 5; D19 on hold; the 96-group threshold unexplained).
- **D-G11** OBSERVED. Nothing multi-device, multi-threaded or with a non-default pool has run on hardware; the staged copy between devices is tested against mocks only. A handle has one home, so alternating readers move it every run; every device has its own context, so any device-to-device copy is staged through the host.
- **D-G12** OBSERVED. Scale characteristics: the bound block and preparation are linear in tickets x units x inputs, re-encoded and re-uploaded every run; the tile graph walks every element of every result-derived input on the host (tile_graph.hpp 211-218) and a data-chosen pull adds an edge from every source tile to each consumer (242-252), which is quadratic; every launch is three device operations; one std::async thread per deferred submission (inorder_executor.cpp 168).
- **D-G13** OBSERVED. Limits not in the documents: elements at most 16 bytes, in-place elements at most 8, item.compute at most 4 inputs, 16 intermediate values per work item, rank at most 3, view chains at most 64 deep.
- **D-G14** OBSERVED. Unused: array_rep; handle_slice and mem_handle::slice (public); view_node::trim; device_limits::max_groups (never enforced); device_desc::free_memory; node_record::builder_type; input_sel::local_copy; prepared_kernel::copies, local_copy_count, imap_tiles; array_slot::op_symbol and offset (never set); map_rec::input_bytes; item_val::describe; dev::out_last_only, no_local; block_magic (written, never checked on the device); launch_plan::tasks and base; most of slab_bitmap's search (tests only); four (void) silences.

#### JIT

- **D-J1** OBSERVED + RECORDED. Cold JIT, chain of 30 over chain of 10: omp 11.57, cuda 4.32, ocl 4.10, ze 4.60; IR lines 2.81. The whole composed kernel is one function after stage 2 (ir_functions = 1), by upstream's always-inline pass, which no NAGA line changes or exempts.
- **D-J2** OBSERVED. The HCF object id is random per compile (TargetSeparationPass.cpp 577-580) and is in the persistent cache key; upstream intends it (#2024). So every rebuild of an application pays the cold JIT again: 29 s to 334 s for a chain of 30.

#### Runtime

- **D-R1** OBSERVED. Level Zero region copy returns unimplemented above 32 bits (ze_queue.cpp 322-335, nothing exercises it); ze_queue::insert_event ignores zeEventHostSignal's result (235-237); with deferral in place, submit_external_wait_for is unreachable on Level Zero and reachable on OpenCL only across contexts (ocl_queue.cpp 415-417), so the bridging worker and its std::async are mostly dead.
- **D-R2** OBSERVED. test-run.nu's limit has no kill-after; sg-test-run.nu's has.

#### Divergence from upstream

- **D-U1** OBSERVED. Vendor development LLVM against upstream's stated support (the version cap raised to 23 in CMakeLists.txt 490, the development check overridden). Upstream is 22 commits ahead; 25 of its 66 changed files are NAGA-modified, among them CMakeLists.txt, llvm-to-backend, SubCfgFormation and the SSCP reduction header. Of three overlapping fixes: #2274 is already in the fork as a cherry-pick (681c6c5e); #2316 (settings) and #2301 (work-group reduction) were fixed by NAGA first in another form (1450b815, 362b077d), so those two files will conflict, and upstream's new tests for #2301 are not in the fork. bin/acpp's configuration keys are renamed throughout.
- **D-U2** OBSERVED. No CI covers anything NAGA added: 16 workflows, one NAGA line.

#### Documents

- **D-D1** OBSERVED. docs/naga-internal/README.md and 04 describe the design before the language pass and read as current; the three configuration-model documents (1,232 lines) were removed on 2026-10-16 and the scheme now lives only in cmake header comments (which are thorough); packages/README.md says a backend override is needed and nothing is published; 17 mentions of the retired acpp-toolchain repo, handoff.md and recipe/; bin/acpp comments cite steps "5a" and "5b" of a plan no longer in the tree.

#### What nothing runs

- **D-T1** OBSERVED. No task runs the three tests/async_host programs, the lit suites under tests/compiler (the alignment fix's regression test has no output on disk), the tool tests other than sg-jit-lib, or doc-consistency, primitive-coverage and sg-conformance. check-packages and check-devices last ran 2026-11-21, the sanitizer smoke 2026-11-22. check-packages reads its baseline, and a file of the retired naga-acpp-local package, from ~/logs outside the repository, and its tenth check needs the network.

#### Build and disk

- **D-B1** OBSERVED. llvm-project is pinned by commit URL without a checksum; ROCm comes from a nightly URL and the archive is deleted after extraction, so a fresh machine depends on AMD keeping that nightly. The installed toolchain's version string is fixed at configure time: the default environment reports 25a68774 and "dirty" at head d888afd9. 83 GB of build directory for the retired naga-acpp-local package and an environment core-inspect no manifest names; .pixi 208 GB, .build 38 GB.

### E. What is done well

- **E1** OBSERVED. The graph API needs four small hooks below the public SYCL API; the runtime divergence is about 1,800 lines.
- **E2** OBSERVED + RECORDED. Composition reuses the fork's own indexed-call mechanism; a kernel's identity is its ordered task list, so reruns with other data, sizes and launch values compile nothing (tested by jit_compile_count).
- **E3** RECORDED. Deferral: one rule on four GPU backends; 0 bad runs in 523 after it, against 2 in 20 and a hang before.
- **E4** OBSERVED. The async_host executor: 106 lines, no pool, the future is the event.
- **E5** RECORDED + OBSERVED. Tests: the upstream suite green on four backends (827 cases each, newest logs 2026-11-24); the sg suite on host, four backends and lit; every device cell proves its device; the runner kills what ignores its limit.
- **E6** OBSERVED. The device code is careful: sub-word combines through the containing word, reduce and scan trees in item order without an identity, the relaxed poll with its reason written beside it, one layout definition shared by host encoder and device decoder with alignment asserted field by field.
- **E7** OBSERVED. Build guards: licence drift, hollow runtimes, FORCE_ON features, flag hygiene for the runtimes child build, per-prefix trees, a keyed source cache; the package split is checked eleven ways and its exports reach consumers with no text on their side.
- **E8** OBSERVED. The configuration model is coherent and explained where it lives: one namespace for discovery, three kinds of value, one option and one root per vendor, relative roots that make the installed toolchain relocatable (the installed acpp-toolchain.json carries no build path).
- **E9** OBSERVED. Two fixes NAGA made in September were fixed by upstream in November (string settings, work-group reduction): the problems were real and found early. Several NAGA compiler fixes have regression tests in upstream's own style (issue2264, overaligned_local_array, split_loop_live_load).
- **E10** OBSERVED. NAGA's lines carry 1 marker of unfinished work; upstream's about 250.
- **E11** OBSERVED. Decisions are written with causes (D2-D20) and the manifest records where the implementation deviates.

## Corrections made during the audit

- The licence-drift guard's error text names the right directory: the paths it compares are reached through the `shared` symlink to packages/core.
- The two copies of the headers in the install (include/AdaptiveCpp/hipSYCL and include/AdaptiveCpp/AdaptiveCpp) come from upstream's own install rule (CMakeLists.txt 894-896) and are not a finding.
- Of the three upstream fixes that overlap NAGA's, two were made independently by NAGA before upstream; the third (#2274) is in the fork as a cherry-pick of upstream's commit.
- A floating-point launch value is not truncated into use: it is accepted and then nothing can use it.

## Discussion after the report (2026-11-25)

Jack and the Research Lead, after the report was given. Nothing here is decided; it is recorded so the discussion can be picked up.

### 1. Launch values are for launch configuration; scalars for device code are kernel constants

Jack: a launch value should not be something a device function can capture. A value produced during a run and used inside device code, for example a scalar added to every element of an array, is better served by tagging a source that is a scalar as a kernel constant (the name is open). Launch values stay what configures a launch.

Research Lead: agreed. As built, a launch value cannot be used in a device function at all: it has no conversion to its type and no accessor, and capturing one captures only its handle index and slot. A floating-point launch value is accepted and unusable. A work item cannot read one shared scalar today, because it reads only at its own index; in a tile kernel a one-element source can be broadcast across the run. A kernel constant should be read by the device from the handle's memory at run time: if the host read it and wrote it into the launch data, `run` would block on it again.

### 2. Not blocking `run` on launch values

Jack: do the launch-value part under the hood as an asynchronous host task inside which the kernel is submitted, with a promise and a future that carry the kernel's event out.

Research Lead: the direction is right, with one snag. Waiting on that future blocks as long as today, because the host task cannot submit the kernel before the launch value's producer has finished. To really not block, later work needs something to depend on that exists at once: the host task waits for the kernel before it returns, so that its own event stands for the kernel, or an event is created up front and filled in later, as the runtime's deferred submission does. Sizes come from launch values too, so the memory of that kernel's results is allocated inside the task, and later nodes that touch those handles cannot be set up before it. What moves off the calling thread is therefore the remainder of that run from that node on, not one kernel. Consequences: the graph's bookkeeping, single-threaded today, would run on another thread; a second `run` would queue behind the remainder of the first; size and role errors would surface asynchronously, which meets the finding that a graph has no error channel. Not checked: whether submitting queue work from inside an `async_host` task is clean in the runtime.

### 3. Compiling kernels off the calling thread

Jack: the same mechanism could take kernel compilation off the calling thread: some kernels can be compiled eagerly and some cannot, several can compile in parallel, and a finished compilation simply lands in the kernel cache, aiming at the best throughput of compilations and submissions. This does not make a single kernel compile faster and was never meant to. A chain of 30 work sets in one kernel is rare; its compile time is still to be improved, on another day.

Research Lead: the code is close to this. A composed kernel's identity is its ordered task list, its group size and its local-memory bucket; the task list is known when the builder returns at submit. The kernel cache already builds code objects outside its lock and shares identical builds in flight, and the CPU and CUDA JIT paths run their tools as separate processes. To check before relying on it: the group size and the local-memory bucket are computed during preparation today, which needs the launch values (they appear to follow from structure alone: inferred); and no way is known to compile a kernel without launching it.

