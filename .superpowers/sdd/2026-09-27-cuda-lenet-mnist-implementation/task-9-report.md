# Task 9 Report: Loss, Metrics, AdamW, and Finite Scanning

## Status

Implemented the Task 9 launchers, kernels, contracts, build wiring, and
real-runtime tests. This Windows workstation has no `nvcc` and no NVIDIA device,
so CUDA compilation and runtime behavior are not verified locally. Static build
dry-runs and policy checks are clean; existing host and Python regressions pass.

## Test-First Evidence

The worktree began clean on `feature/rebuild` at
`c1a6e3d9ab309f7f5d8ac4543ecf9101a5b7952a`.

Tests were added to `tests/operator_tests.cu` before any production source was
created. At that red point:

- `git diff --stat` showed only `tests/operator_tests.cu`, with 347 inserted
  lines.
- A source search for definitions of `LaunchSoftmaxCrossEntropy`,
  `LaunchSoftmax`, `LaunchArgmaxAndCountCorrect`, `LaunchAdamW`, and
  `LaunchFindFirstNonFinite` under `src/` returned no files. The tests therefore
  referenced declared but undefined production launchers.
- `make build/operator_tests` was attempted and stopped before compilation
  because GNU `make` is not installed under that command name.
- `Get-Command nvcc` reported `nvcc=missing`, so a CUDA red link/build and the
  required real-runtime red execution could not be observed locally.

The tests exercise real device buffers and launchers without mocks or CPU/device
fallbacks. They cover:

- logits at `+1000` and `-1000`, finite probabilities, row sums, finite losses,
  and independently computed gradient values;
- actual batch scaling and deterministic mean reduction for batches 1, 17, 128,
  and 129;
- inference softmax through its label-free API;
- smallest-class argmax ties, all per-sample predictions/flags, and exact final
  correct count;
- one and ten AdamW steps against an FP64 reference, host double bias
  corrections converted to FP32 factors, pre-update decay, and zero bias decay;
- zero-count and 257-element finite scans, isolated NaN, `+Inf`, and `-Inf`
  classification, multiple bad values, and smallest reported index;
- invalid dimensions, null pointers, overlap, pointer ranges, invalid AdamW
  scalars, and `count > INT_MAX`.

## Algorithms And Synchronization

### Softmax And Cross-Entropy

Each sample maps to one 256-thread block. A 256-float shared array first holds
logits or `-FLT_MAX` for inactive lanes, then holds exponentials or zero. Fixed
stride 128-to-1 trees perform max and sum reductions, with `__syncthreads()`
after initialization and every reduction stage. Subtracting the row maximum
bounds exponent inputs above by zero and guarantees a positive sum for finite
inputs.

Each valid class lane writes one probability and one
`(probability-one_hot)/batch_size` gradient. Lane zero writes
`log(sum)+max-logit[label]`, avoiding `-log(0)` when a target probability
underflows. A separate one-thread project kernel sums per-sample losses in
ascending order and divides exactly once. Inference uses a separate kernel with
no label argument or label-dependent branch.

### Metrics

Each sample maps to one block with parallel 256-entry shared value/index arrays.
Inactive lanes use `(-FLT_MAX, class_count)`. Every fixed tree comparison keeps
the larger value, or the smaller class index on equality. Lane zero writes the
prediction and exact 0/1 flag. A separate one-thread project kernel sums flags
in ascending sample order and writes the final integer count. No external
reduction or atomics are used.

### AdamW

Each parameter index has one owning thread. It computes:

```text
m' = beta1*m + (1-beta1)*g
v' = beta2*v + (1-beta2)*g*g
p' = p - lr*((m'*inverse_bias1)/(sqrt(v'*inverse_bias2)+epsilon) + decay*p)
```

The local `p` is captured before any write, so decay uses the pre-update
parameter. Bias callers pass zero decay. The caller computes
`1/(1-beta^t)` in double and converts each factor to FP32, including `t=1`.
Arrays are disjoint and every index has one writer, so no barriers or atomics are
needed.

### Finite Scan

A one-thread initialization kernel writes the exactly representable `count`
sentinel. A following same-stream grid-stride kernel classifies values with the
CUDA device `isfinite` built-in and uses `atomicMin` only for bad indices.
Concurrent reports therefore deterministically leave the smallest index. No bad
value leaves the initialized sentinel unchanged. `count == 0` still launches
the initializer and writes zero; `count > INT_MAX` is rejected before launch.

All launches use the caller stream, immediately call `CUDA_KERNEL_CHECK()`, and
perform no host or device synchronization.

## Static And Host Evidence

Fresh local evidence:

- `python -m unittest discover -s tests -p "test_*.py"`: 15 tests passed.
- Existing rebuilt/base host executables run individually: `smoke_tests`,
  `dataset_tests`, `random_tests`, `parameters_tests`, `checkpoint_tests`, and
  `cpu_reference_tests` each reported `status=pass`.
- `mingw32-make -n V=1 build/operator_tests`: exit 0 and emitted compile steps
  for `loss.cu`, `metrics.cu`, `adam.cu`, and `operator_tests.cu`, followed by a
  link command containing all three new objects.
- `git diff --check`: exit 0; only line-ending conversion warnings were emitted.
- Source search found no Thrust, CUB, cuBLAS, cuDNN, `atomicAdd`,
  `cudaDeviceSynchronize`, or `cudaStreamSynchronize` in the new kernels.
- Eight kernel launches are present and eight immediate
  `CUDA_KERNEL_CHECK()` calls are present.

`mingw32-make host-tests` cannot run the aggregate recipe because this make uses
Windows `cmd.exe`, which does not understand the Makefile's POSIX
`set -e; for ...` recipe. The six host binaries were therefore executed
individually as listed above. This is an environment/shell limitation, not a
passing aggregate make result.

## Exact H20 Commands

Run from the repository worktree on the H20 host:

```bash
make build/operator_tests
./build/operator_tests --filter softmax
./build/operator_tests --filter metrics
./build/operator_tests --filter adamw
./build/operator_tests --filter finite_scan
```

All five commands remain required because this workstation cannot compile or
run CUDA code. No device skip or fallback is present in the tests.

## Files

- `src/kernels/loss.cu`: stable training/inference softmax and deterministic
  mean-loss reduction.
- `src/kernels/metrics.cu`: deterministic smallest-index argmax, flags, and
  correct-count reduction.
- `src/kernels/adam.cu`: exact AdamW update and first-non-finite scan.
- `include/layers.h`: validation and exact-equation contract documentation.
- `tests/operator_tests.cu`: Task 9 numerical, edge, and validation tests.
- `Makefile`: new CUDA kernel objects in `operator_tests`.
- This report.

## Self-Review And Concerns

Mutation-oriented review confirmed tests would catch removal of max subtraction,
wrong batch scaling, dividing mean loss more than once, label-dependent
inference, largest-index tie selection, omitted flags/count reduction, decay
using updated parameters, nonzero bias decay, a wrong first-step factor,
non-atomic scan writes, failure to initialize the sentinel, and failure to
classify any one of NaN/`+Inf`/`-Inf`.

The only unresolved concern is environmental: CUDA syntax, SM90 compilation,
device numerical tolerances, launch behavior, and real H20 results are
unverified until the exact commands above run on the H20 host.
