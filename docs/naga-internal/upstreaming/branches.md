# Upstreaming triage: pr/ branches

Pins: U `27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc`, N `0e2bac974c40554f62286cbffe1f06c87cc9caa9`, B `9ec394efa584ced004d644af2194d2133e111d26`; merge tree M `350667e47d62407ad9bef5977bdab5a632ae8e87` (see baseline.md).

Base B0 = `27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc`, the tip of `upstream/develop` fetched on 2026-11-23 (equal to U; `git ls-remote https://github.com/AdaptiveCpp/AdaptiveCpp.git refs/heads/develop` gave the same SHA; b5535e79 and bce257b0 are ancestors). Develop had not moved, so no resolution was forced by a moved base.

Each branch is B0 plus one candidate's changes as a single commit, pushed to `origin` (NAGALabsOrg/AdaptiveCpp) only. No pull request is open for any of them. Change sets are the candidates' assignment blocks in candidates.md applied to `git diff --no-renames -U0 U M`; nothing was built.

| candidate | branch | base SHA | head SHA | commit subject | conflict resolution |
|---|---|---|---|---|---|
| C08 | pr/alloca-alignment-size-name | 27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc | 9c902b6d345d841fa5a7f9df752dae550035f432 | [SSCP] Keep alloca alignment, size and name in address space passes | none |
| C09 | pr/xcode26-includes | 27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc | bf81c80a680182a3522c25f8017e9941d8215c88 | [Runtime] Include what you use for Xcode 26 libc++ | omp_queue.hpp: include block keeps <mutex> from b5535e79 and adds <memory> |
| C11 | pr/pcuda-robustness | 27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc | a78cf93162306b201d325008957fb7b69758f242 | [PCUDA] Make error handling robust and launch MSVC kernel names | none |
| C12 | pr/ocl-spirv-cl-std | 27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc | 1c14f4cf7268113b344c2d2701b3b023b4aae30c | [OpenCL] Build SPIR-V with the device's OpenCL C version | none |
| C13 | pr/ocl-prefetch-hint | 27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc | d91d044e23305d85789298bfdbf1ad9ef2df3c02 | [OpenCL] Ignore failed USM prefetch hints | none |
| C14 | pr/stdpar-atomic-int-builtins | 27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc | e9c5d084802e26410ede4d3ff6a6d27f26221843 | [stdpar] Map float and pointer atomics to integer SSCP builtins | none |
| C16 | pr/kernel-cache-lock | 27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc | bd3937a5f568f0cad76fe5403b7d10d1d07fb55c | [Runtime] Build code objects outside the kernel cache lock | none |
| C17 | pr/jit-tool-errors | 27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc | 9c705d333b5d68fc47e8eb8ab14f1c973c2f7a63 | [SSCP] Report JIT tool exit codes, suppress Windows error dialogs | Utils.hpp: declarations keep ACPP_DEFAULT_VISIBILITY from bce257b0; describeToolExitCode declared with it |
| C18 | pr/ze-cstdint | 27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc | 5283deac3af0aa85e5955eb2cae3bf9777507bc6 | [L0] Include <cstdint> instead of glibc-internal header | none |
| C19 | pr/threads-public-link | 27ed6400e890cf0ecd2aa0b055d75cc08ac68dfc | eb21034a25cec0d7ba74eae32bb5c3000777596e | [CMake] Make Threads a public requirement of acpp-rt | none |

## Notes on the two resolutions

- C09, `include/hipSYCL/runtime/omp/omp_queue.hpp`: M carries conflict markers in the include block. The branch keeps B0's `#include <mutex>` and adds `#include <memory>` directly before it; the only change to the file is that one added line.
- C17, `include/hipSYCL/compiler/llvm-to-backend/Utils.hpp`: M carries conflict markers around the declaration block. The branch keeps every B0 declaration unchanged (ACPP_DEFAULT_VISIBILITY on each existing declaration, including `getRedistPackageBitcodePath`, and the `#ifdef _WIN32` `getHostCRuntimeImportLibraryPath`) and adds two lines after `getRedistPackageBitcodePath`: the comment `// Text for a failed tool invocation's error message: "with exit code N".` and `ACPP_DEFAULT_VISIBILITY std::string describeToolExitCode(int ReturnCode);`. The candidate's own version put them after `getBitcodePath`; the position differs because the candidate's list had dropped `getRedistPackageBitcodePath`, which B0 still has.
- C17, `src/compiler/llvm-to-backend/Utils.cpp`: only the pieces named in the C17 assignment block are applied (the `<string>` include, the Windows `SetErrorMode` includes, `suppressChildErrorDialogs`, `describeToolExitCode` with the namespace close and reopen around it, and the two call sites). The path-resolution lines stay out.

## Shared files

Two files are changed by two branches, in disjoint line ranges of B0: `include/hipSYCL/runtime/kernel_cache.hpp` (pr/xcode26-includes at line 26, pr/kernel-cache-lock elsewhere) and `src/compiler/llvm-to-backend/spirv/LLVMToSpirv.cpp` (pr/ocl-spirv-cl-std at lines 81 and 237, pr/jit-tool-errors at line 360).
