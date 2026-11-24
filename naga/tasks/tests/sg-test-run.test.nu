use std/assert
use ../sg-test-run.nu [cell-verdict classify-exit run-limited]

const omp_line = "sg-cell backend=omp platform=OpenMP_(platform_0) device=AdaptiveCpp_OpenMP_host_device vendor=the_AdaptiveCpp_project"
const cuda_line = "sg-cell backend=cuda platform=CUDA_(platform_0) device=NVIDIA_GeForce_RTX_4090 vendor=NVIDIA"
const ocl_line = "sg-cell backend=ocl platform=OpenCL_(platform_0) device=Intel(R)_Graphics vendor=Intel(R)_Graphics"
const ze_line = "sg-cell backend=ze platform=Level_Zero_(platform_0) device=Intel(R)_Graphics vendor=pci:32902"
const ze_on_nvidia = "sg-cell backend=ze platform=Level_Zero_(platform_0) device=NVIDIA_GeForce_RTX_4090 vendor=pci:4318"
const ocl_on_nvidia = "sg-cell backend=ocl platform=NVIDIA_CUDA device=NVIDIA_GeForce_RTX_4090 vendor=NVIDIA_Corporation"
const ocl_on_cpu = "sg-cell backend=ocl platform=Intel(R)_OpenCL device=13th_Gen_Intel(R)_Core(TM)_i7-13700K vendor=Intel(R)_Corporation"

def test-wrong-device-names-the-cell-and-the-device [] {
  let v = (cell-verdict ocl sg_tests $"($ocl_on_nvidia)\nRunning 5 test cases")
  assert (not $v.ok)
  assert ($v.message | str contains "cell ocl")
  assert ($v.message | str contains "expected backend ocl")
  assert ($v.message | str contains "NVIDIA_GeForce_RTX_4090")
}

def test-an-ocl-cell-that-ran-on-the-openmp-host-device-fails [] {
  let line = "sg-cell backend=omp platform=OpenMP_(platform_0) device=AdaptiveCpp_OpenMP_host_device vendor=the_AdaptiveCpp_project"
  let v = (cell-verdict ocl sg_tests $"($line)\nRunning 77 test cases...")
  assert (not $v.ok)
  assert ($v.message | str contains "cell ocl")
  assert ($v.message | str contains "expected backend ocl")
  assert ($v.message | str contains "AdaptiveCpp_OpenMP_host_device")
}

def test-a-ze-cell-that-ran-on-the-openmp-host-device-fails [] {
  let line = "sg-cell backend=omp platform=OpenMP_(platform_0) device=AdaptiveCpp_OpenMP_host_device vendor=the_AdaptiveCpp_project"
  let v = (cell-verdict ze sg_tests $"($line)\nRunning 77 test cases...")
  assert (not $v.ok)
  assert ($v.message | str contains "cell ze")
  assert ($v.message | str contains "expected backend ze")
  assert ($v.message | str contains "AdaptiveCpp_OpenMP_host_device")
}

def test-an-opencl-cpu-device-is-not-the-intel-gpu [] {
  let v = (cell-verdict ocl sg_tests $ocl_on_cpu)
  assert (not $v.ok)
  assert ($v.message | str contains "13th_Gen_Intel(R)_Core(TM)_i7-13700K")
}

def test-a-ze-device-with-a-pci-vendor-id-of-intel-passes-and-another-id-fails [] {
  assert (cell-verdict ze sg_tests $ze_line).ok
  let v = (cell-verdict ze sg_tests $ze_on_nvidia)
  assert (not $v.ok)
  assert ($v.message | str contains "cell ze")
  assert ($v.message | str contains "NVIDIA_GeForce_RTX_4090")
}

def test-a-run-ended-by-the-limit-is-a-timeout [] {
  assert equal (classify-exit 124 1800.5 1800) "timeout"
  assert equal (classify-exit 137 1861.0 1800) "timeout"
  assert equal (classify-exit -9 1861.0 1800) "timeout"
  assert equal (classify-exit 137 280.0 1800) 137
  assert equal (classify-exit -9 280.0 1800) (-9)
  assert equal (classify-exit 1 1900.0 1800) 1
  assert equal (classify-exit 0 12.0 1800) 0
}

def test-a-run-that-outlives-a-short-limit-is-ended-and-reported-as-a-timeout [] {
  let log = (mktemp --tmpdir sg-limit-test-XXXX.log)
  let t0 = (date now)
  let code = (run-limited $log omp {} 2 2 nu ...["-c" "sleep 30sec"])
  let secs = ((date now) - $t0 | into int) / 1000000000.0
  rm --force $log
  assert equal $code "timeout"
  assert ($secs < 10.0) $"the run was ended after ($secs) s"
}

def test-a-run-that-ends-by-itself-keeps-its-exit-code [] {
  let log = (mktemp --tmpdir sg-limit-test-XXXX.log)
  let code = (run-limited $log omp {} 20 2 nu ...["-c" "exit 3"])
  rm --force $log
  assert equal $code 3
}

def test-a-log-without-the-first-line-fails [] {
  let v = (cell-verdict ze sg_driver_tests "Running 5 test cases\n*** No errors detected")
  assert (not $v.ok)
  assert ($v.message | str contains "cell ze")
  assert ($v.message | str contains "does not start with")
}

def test-four-correct-lines-pass [] {
  for case in [[cell line]; [omp $omp_line] [cuda $cuda_line] [ocl $ocl_line] [ze $ze_line]] {
    let v = (cell-verdict $case.cell sg_tests $"($case.line)\nRunning 5 test cases")
    assert $v.ok $"($case.cell): ($v.message)"
  }
}

def test-a-leading-blank-line-and-colour-codes-are-ignored [] {
  let v = (cell-verdict cuda sg_tests $"\n(ansi green)($cuda_line)(ansi reset)\n")
  assert $v.ok
}

def test-the-host-cell-expects-no-device [] {
  assert (cell-verdict host sg_host_tests "sg-cell backend=host platform=none device=none vendor=none\n").ok
  assert (not (cell-verdict host sg_host_tests $omp_line).ok)
}

test-wrong-device-names-the-cell-and-the-device
test-an-ocl-cell-that-ran-on-the-openmp-host-device-fails
test-a-ze-cell-that-ran-on-the-openmp-host-device-fails
test-an-opencl-cpu-device-is-not-the-intel-gpu
test-a-ze-device-with-a-pci-vendor-id-of-intel-passes-and-another-id-fails
test-a-run-ended-by-the-limit-is-a-timeout
test-a-run-that-outlives-a-short-limit-is-ended-and-reported-as-a-timeout
test-a-run-that-ends-by-itself-keeps-its-exit-code
test-a-log-without-the-first-line-fails
test-four-correct-lines-pass
test-a-leading-blank-line-and-colour-codes-are-ignored
test-the-host-cell-expects-no-device
print "sg-test-run tests: all passed"
