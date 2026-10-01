# NVIDIA H20 Acceptance Guide

Run this procedure on the Ubuntu 18.04 H20 host. It is a target-host checklist,
not evidence that CUDA succeeded on the local Windows development machine.
Use Python 3.6, GCC 7.5, CUDA Toolkit 12.1.x (or a compatible newer CUDA 12.x),
an H20-capable driver, and a compute-capability 9.0 NVIDIA H20.

## Evidence Directory

Start in the repository root and retain one timestamped terminal log:

```bash
set -euo pipefail
mkdir -p build/acceptance weights data
exec > >(tee "build/acceptance/h20-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ
nvcc --version
gcc --version
nvidia-smi
```

Do not mark the manual checklist reviewed until every listed declaration,
kernel, reduction, serializer, and parser has actually been inspected.

## Python And Official Data First

Install the hash-locked Python 3.6 dependencies, compile the scripts/tests, run
the Python suites, and prepare official data before any real-MNIST workflow
case:

```bash
python3.6 -m pip install --require-hashes -r requirements-py36.txt
python3.6 -m py_compile \
  scripts/prepare_mnist.py scripts/check_comments.py \
  tests/test_prepare_mnist.py tests/test_data_interop.py \
  tests/test_check_prohibited.py tests/test_check_comments.py \
  tests/test_documentation.py tests/output_format_tests.py \
  tests/compliance_tests.py
python3.6 -m unittest -v \
  tests.test_prepare_mnist tests.test_data_interop \
  tests.test_check_prohibited tests.test_check_comments \
  tests.test_documentation tests.output_format_tests \
  tests.compliance_tests
python3.6 scripts/prepare_mnist.py --output-dir data
stat -c '%n %s' data/train.bin data/test.bin
```

Expected binary sizes are `47100024` bytes for `data/train.bin` and `7850024`
bytes for `data/test.bin`.

## Clean Verbose Build And Static Gates

```bash
make clean
mkdir -p build
make V=1 CUDA_ARCH=sm_90 2>&1 | tee build/verbose-build.log
bash scripts/check_prohibited.sh source .
bash scripts/check_prohibited.sh build build/verbose-build.log
python3.6 scripts/check_comments.py --root . --checklist docs/comment-review-checklist.md
make host-tests
```

The verbose commands must contain both exact targets:

```text
-gencode=arch=compute_90,code=sm_90
-gencode=arch=compute_90,code=compute_90
```

## CUDA Operator And Workflow Tests

Run operator tests and every workflow case. The official data preparation above
must finish first. The corrected real-MNIST overfit and inference invocations
explicitly receive the training binary:

```bash
./build/operator_tests
./build/workflow_tests --case batch17
./build/workflow_tests --case overfit32 --mnist-train data/train.bin
./build/workflow_tests --case one_epoch_checkpoint
./build/workflow_tests --case evaluate
./build/workflow_tests --case infer --mnist-train data/train.bin
./build/workflow_tests --case cli
```

The overfit case requires final loss at most 20% of initial loss and at least
95% training accuracy. All workflow records must report `status=pass`.

## Memory And Binary Evidence

```bash
compute-sanitizer --tool memcheck --error-exitcode=99 ./build/operator_tests
cuobjdump --list-elf build/lenet_cuda
cuobjdump --dump-ptx build/lenet_cuda
ldd build/lenet_cuda
readelf -d build/lenet_cuda
```

Compute Sanitizer must report zero errors. `cuobjdump` must show native `sm_90`
and `compute_90` PTX. Dynamic dependency output must contain no prohibited
neural-network, inference, math-template, reduction-template, or random library;
the source and verbose-build scans cover header-only dependencies.

## Default Training And 99% Threshold

```bash
./build/lenet_cuda train \
  --train data/train.bin \
  --test data/test.bin \
  --output weights/lenet.bin \
  --epochs 20 \
  --batch-size 128 \
  --seed 1337 \
  --device 0

./build/lenet_cuda evaluate \
  --data data/test.bin \
  --weights weights/lenet.bin \
  --min-accuracy 0.99 \
  --device 0

./build/lenet_cuda infer \
  --data data/test.bin \
  --weights weights/lenet.bin \
  --index 0 \
  --device 0
```

The device record must name the NVIDIA H20 and compute capability 9.0. Final
acceptance requires evaluation `accuracy >= 0.99`, `min_accuracy=0.990000`, and
`status=pass`. Timing is informational: evaluation excludes one warm-up batch,
uses CUDA Events, and has no throughput threshold.

## Manual Comment Review

For every item in `docs/comment-review-checklist.md`, review all applicable
shape/layout, linear-index formula, accumulation dimensions, derivative inputs,
boundary handling, races/atomics, synchronization, numerical stability,
ownership/lifetime/aliasing, byte order, and error behavior. Then change that
item from `[ ]` to `[x]` and run:

```bash
python3.6 scripts/check_comments.py \
  --root . --checklist docs/comment-review-checklist.md --require-reviewed
```

Unchecked entries are expected before this human gate and must not be marked by
automation.

## Troubleshooting

- If `sm_90` is rejected, confirm the selected `nvcc` supports Hopper/H20 and is CUDA 12.1.x or a compatible newer CUDA 12.x.
- For a driver/toolkit mismatch, preserve `nvcc --version` and `nvidia-smi` output and load a compatible cluster module pair.
- For out-of-memory errors, preserve requested/free/total byte diagnostics and check other H20 processes before changing batch size.
- For malformed data, compare the exact file sizes and rerun checksum-verified preparation; never bypass the official count checks for acceptance.
- For non-finite training, preserve the phase, tensor, and index diagnostic; rerun operator tests and Compute Sanitizer before changing kernels.
- If accuracy is below 99%, verify official files, seed, defaults, schedule, augmentation, normalization, and persisted-best checkpoint reload before investigating numerics.
- If a dependency scan fails, inspect the named source/log line and the `ldd`/`readelf` evidence; do not suppress the pattern or add a fallback.

Preserve the complete log, verbose build log, checkpoint, `cuobjdump`, dynamic
dependency, and sanitizer outputs as H20 acceptance evidence.
