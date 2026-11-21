# packages/

The naga-acpp toolchain (linux-64) as pixi-build-naga packages, carved from one build.
Building needs `PIXI_BUILD_BACKEND_OVERRIDE=pixi-build-naga=<naga-build-backend>/target/release/pixi-build-naga`
until pixi-build-naga is published. Version `0.dev0`; nothing is published.

| Directory | Package | Kind |
|---|---|---|
| `core/` | `naga-acpp-core` (unpublished) | script: the one LLVM + AdaptiveCpp build |
| `slices/runtime` | `naga-acpp-runtime` | slice |
| `slices/naga-acpp` | `naga-acpp` | slice |
| `slices/compiler-rt` | `naga-acpp-compiler-rt` | slice |
| `slices/llvm-dev` | `naga-acpp-llvm-dev` | slice |
| `slices/tools` | `naga-acpp-tools` | slice |
| `slices/lldb` | `naga-acpp-lldb` | slice |
| `extras/mutex` | `naga-acpp-llvm` (`23.0git`) | meta, weak run export |
| `extras/runtime-{cuda,level-zero,ocl,rocm}` | `naga-acpp-runtime-*` | slices |
| `activations/{clang,clangxx}` | `naga-acpp-{clang,clangxx}_linux-64` | script |
| `checks/consumer` | `naga-acpp-check-consumer` (unpublished) | export round trip |
| `../pixi.toml` | `naga-acpp-toolchain` | meta: the six slices |

Unshipped on purpose: files installed by the bundled OpenCL-Headers/OpenCL-CLHPP (`include/CL/**`,
`share/cmake/OpenCLHeadersCpp`, `share/pkgconfig/OpenCL-*.pc`); conda-forge's `opencl-headers`/`clhpp` own those paths.

Checks: `pixi run check-packages`.
