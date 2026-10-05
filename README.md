# CUDA LeNet for MNIST

This project trains, evaluates, and runs inference with a modern LeNet using
project-owned C++14/CUDA kernels. Model arithmetic is FP32. Activations are
contiguous NCHW, convolution weights are OIHW, and fully connected weights are
row-major `[out][in]`.

The implementation uses the CUDA Runtime API and CUDA device built-ins only.
cuDNN, cuBLAS, TensorRT, Thrust, CUB, cuRAND, neural-network frameworks, mixed
precision, Tensor Cores, distributed execution, and multi-GPU execution are
prohibited. Every neural-network forward, backward, loss, metric, and optimizer
operation is project-owned. CUDA errors terminate the operation with no CPU fallback.

## Target Environment

The baseline target is Ubuntu 18.04, Python 3.6, GCC 7.5, C++14, CUDA Toolkit 12.1.x,
an H20-capable NVIDIA driver, and an NVIDIA H20 GPU with compute
capability 9.0. A newer CUDA 12.x toolkit is acceptable when it supports H20.
The default build embeds native `sm_90` code and forward-compatible
`compute_90` PTX.

The local Windows development machine has no NVIDIA GPU or CUDA compiler.
Local work can verify Python, host C++, Make dry-runs, documentation, and static
checks only. CUDA compilation, CUDA numerical tests, Compute Sanitizer,
training, timing, and the 99% accuracy requirement must be verified on the H20
host. See `docs/h20-acceptance.md`; this README makes no local CUDA-success or
accuracy claim.

## Dataset

The converter uses MNIST from the pinned Hugging Face revision
`77f3279092a1c1579b2250db8eafed0ad422088c`:

- Train URL: `https://huggingface.co/datasets/ylecun/mnist/resolve/77f3279092a1c1579b2250db8eafed0ad422088c/mnist/train-00000-of-00001.parquet`
- Train SHA-256: `f2c01285a9f89399335b00ee4e8d499dc4e46db5e39c74903ce5618d895eb3bf`
- Test URL: `https://huggingface.co/datasets/ylecun/mnist/resolve/77f3279092a1c1579b2250db8eafed0ad422088c/mnist/test-00000-of-00001.parquet`
- Test SHA-256: `d49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b`

The expected source counts are 60,000 training rows and 10,000 test rows. The
script rehashes cached downloads, validates each 28 x 28 grayscale image and
label, and atomically creates `data/train.bin` and `data/test.bin`.

```bash
python3.6 -m pip install --require-hashes -r requirements-py36.txt
python3.6 scripts/prepare_mnist.py --output-dir data
```

`requirements-py36.txt` pins Python 3.6-compatible NumPy, Pillow, PyArrow,
Requests, Certifi, Charset Normalizer, idna, and urllib3 distributions by
version and SHA-256 hash.

## Network

| Stage | Input | Operation | Output |
|---|---|---|---|
| Input | image bytes | normalize, optional training translation | `N x 1 x 28 x 28` |
| Conv1 | `N x 1 x 28 x 28` | 6 OIHW filters, 5 x 5, valid | `N x 6 x 24 x 24` |
| ReLU1/Pool1 | `N x 6 x 24 x 24` | ReLU, 2 x 2 stride-2 max pool | `N x 6 x 12 x 12` |
| Conv2 | `N x 6 x 12 x 12` | 16 OIHW filters, 5 x 5, valid | `N x 16 x 8 x 8` |
| ReLU2/Pool2 | `N x 16 x 8 x 8` | ReLU, 2 x 2 stride-2 max pool | `N x 16 x 4 x 4` |
| Flatten | `N x 16 x 4 x 4` | zero-copy contiguous reshape | `N x 256` |
| FC1 | `N x 256` | `[out][in]` linear `256 -> 120`, ReLU | `N x 120` |
| FC2 | `N x 120` | `[out][in]` linear `120 -> 84`, ReLU | `N x 84` |
| FC3 | `N x 84` | `[out][in]` linear `84 -> 10` | `N x 10` |
| Loss | `N x 10` | stable softmax cross-entropy | scalar mean |

Images are normalized with `(pixel / 255 - 0.1307) / 0.3081`. Training alone
uses deterministic independent x/y translations in `[-2, 2]` with zero fill.
The final partial batch is supported.

## Build And Checks

Prepare official data before running the complete CUDA test target because the
overfit and inference workflow cases consume real MNIST examples.

```bash
make CUDA_ARCH=sm_90
make host-tests
make python-tests
make compliance
make test
```

`make test` runs host tests, CUDA operator/workflow tests, Python tests, and
static compliance checks. It requires the CUDA toolkit/device and prepares the
official dataset before invoking `workflow_tests --mnist-train data/train.bin`.
Use `make V=1` to retain full compiler and linker commands for inspection.
`make compliance` runs checker unit tests, reconciles the comment inventory,
and performs the source-only scan of Make-compiled production/test inputs for
prohibited dependencies, production CPU fallback wiring, and
implementation-dependent random APIs. A Python 3.6-compatible `shlex` analyzer
cross-checks Make's expected test-source manifest against the forced `all`
recipes, follows each command's ordered `-iquote`/`-I` paths, and traces
tests-owned provenance through direct compilation, relocatable links, and
archives. Response files and shell-obfuscated recipes affecting `lenet_cuda`
fail closed because their inputs cannot be proven. The gate recognizes practical
direct source, header, namespace, API, forced-include, and linker forms; it does
not claim to decode arbitrary preprocessor or shell obfuscation.

To inspect commands without executing them, capture a dry-run and label it as
such. This proves only what Make would invoke; the commands were not executed:

```bash
mkdir -p acceptance
make -Bn V=1 > acceptance/dry-run.log
bash scripts/check_prohibited.sh dry-run acceptance/dry-run.log
```

Successful-build evidence must come from an actual verbose build that ends in
`event=build status=pass target=all`, followed by
`bash scripts/check_prohibited.sh build acceptance/verbose-build.log`. The
build-log scan is intentionally separate from the source-only `make compliance`
gate. Retain actual logs outside `build/` so a later cleanup cannot delete them.

The exact default architecture flags are:

```text
-gencode=arch=compute_90,code=sm_90
-gencode=arch=compute_90,code=compute_90
```

## Train

Create the output directory before training:

```bash
mkdir -p weights
./build/lenet_cuda train \
  --train data/train.bin \
  --test data/test.bin \
  --output weights/lenet.bin \
  --epochs 20 \
  --batch-size 128 \
  --seed 1337 \
  --device 0
```

Defaults and fixed optimizer settings:

- Batch size: `128`; accepted range `[1, 1024]`.
- Epochs: `20`; accepted range `[1, 1000]`.
- Learning rate: `0.001` for epochs 1-12, `0.0001` for 13-17, and `0.00001` for 18-20.
- Beta1: `0.9`.
- Beta2: `0.999`.
- Epsilon: `1e-8`.
- Weight decay: `1e-4` on weights only; biases use zero decay.
- Seed: `1337`; any unsigned 64-bit value is accepted.
- Device: `0`; it must name an available CUDA device.

The 60,000-row training source is deterministically split into 55,000 training
and 5,000 validation rows. Training indices are shuffled each epoch;
validation and test rows are neither shuffled nor augmented. A checkpoint is
replaced only by a strictly greater validation accuracy. The persisted best
checkpoint is reloaded before final test evaluation.

## Evaluate

```bash
./build/lenet_cuda evaluate \
  --data data/test.bin \
  --weights weights/lenet.bin \
  --min-accuracy 0.99 \
  --device 0
```

`--min-accuracy` defaults to `0` and accepts finite values in `[0, 1]`.
Equality passes. Accuracy below the requested threshold prints `status=fail`
and returns exit code `3`. The official acceptance threshold is at least 99%
on all 10,000 test images.

Evaluation includes one unmeasured warm-up batch, then uses CUDA Events around
the remaining forward launches. `mean_forward_ms` and `images_per_second` are
informational; there is no performance acceptance threshold. A one-batch
dataset explicitly reports zero for both timing fields.

## Infer

```bash
./build/lenet_cuda infer \
  --data data/test.bin \
  --weights weights/lenet.bin \
  --index 0 \
  --device 0
```

The index is a nonnegative unsigned integer below the loaded sample count.
Inference prints ten logits, ten probabilities, the smallest-index argmax
prediction, and the label.

`--allow-nonstandard-count` is a test-only option for `train` and `evaluate`.
Production commands enforce 60,000 training and 10,000 evaluation/test rows.

## Output Grammar

Successful records are newline-terminated ASCII `key=value` fields in the
shown order. Device names may contain spaces; `compute_capability` is the final
device field. Representative records are:

```text
event=device index=0 name=NVIDIA H20 compute_capability=9.0
event=epoch epoch=1 train_loss=0.123456 validation_accuracy=0.987600 elapsed_ms=1234.567
event=final_test samples=10000 final_test_accuracy=0.991200 best_epoch=17 validation_accuracy=0.992000
event=evaluate samples=10000 accuracy=0.991200 mean_forward_ms=0.321 images_per_second=398753.875 min_accuracy=0.990000 status=pass
event=infer index=0 logits=0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,0.000000000 probabilities=0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,0.000000000,0.000000000 prediction=0 label=0
```

Process exit codes:

- `0`: success.
- `1`: runtime error.
- `2`: command-line usage error.
- `3`: evaluation accuracy below the requested threshold.

Usage, unreadable paths, unavailable devices, malformed binaries, and invalid
indexes produce a concise `error:` diagnostic and nonzero exit. The executable
also prints its usage summary on command and runtime failures.

## Binary Formats

### Dataset

Each little-endian dataset begins with the 8-byte magic `MNISTC1\0`, followed
by `uint32` version `1`, sample count, rows `28`, and columns `28`. The payload
contains all contiguous row-major `uint8` images, then one `uint8` label in
`[0, 9]` per image. The loader rejects zero counts, wrong dimensions, size
overflow, truncation, trailing bytes, invalid labels, and count mismatches.

### Checkpoint

Each little-endian checkpoint begins with the 8-byte magic `LNETC01\0`, format
version `1`, architecture ID `1`, tensor count `10`, one-based best epoch,
validation accuracy, normalization mean/stddev, and a zero reserved field.
Ten records store a 32-byte NUL-padded name, rank, four dimensions, element
count, and contiguous FP32 values in this order:

| Tensor | Dimensions |
|---|---|
| `conv1.weight` | `[6, 1, 5, 5]` |
| `conv1.bias` | `[6]` |
| `conv2.weight` | `[16, 6, 5, 5]` |
| `conv2.bias` | `[16]` |
| `fc1.weight` | `[120, 256]` |
| `fc1.bias` | `[120]` |
| `fc2.weight` | `[84, 120]` |
| `fc2.bias` | `[84]` |
| `fc3.weight` | `[10, 84]` |
| `fc3.bias` | `[10]` |

Adam state is not stored. Checkpoint loading rejects non-finite values, unknown
or reordered tensors, shape/count mismatches, truncation, and trailing bytes.

## Determinism

Seed `1337` is the default. The project owns SplitMix64 streams, rejection-
sampled Fisher-Yates permutations, Box-Muller He initialization, and translated
sample hashing. It does not use implementation-dependent standard shuffling or
normal distributions. Repeated runs of the same binary on the same platform
and GPU configuration are deterministic. Bit-identical checkpoints across
different compilers, CUDA versions, drivers, or GPU architectures are not
promised.

## Error Handling

Operational CUDA Runtime calls are checked with expression, file, line, numeric
code, and CUDA error text. Every kernel launch is checked with
`cudaGetLastError`; phase/test synchronization boundaries attribute asynchronous
failures. Nonthrowing destructors perform best-effort cleanup: results from
`cudaFree`, `cudaStreamDestroy`, and `cudaEventDestroy` cannot escape those
cleanup paths and are intentionally discarded.
Allocation failures report requested bytes and device free/total memory.
Dataset and checkpoint failures include the path and violated invariant.
Training aborts on non-finite activations, losses, gradients, or parameters.

## Troubleshooting

- **CUDA architecture:** CUDA 12.1.x must support `sm_90`; inspect the two exact `-gencode` flags with `make V=1` and verify the binary with `cuobjdump`.
- **driver/toolkit mismatch:** compare `nvcc --version` with `nvidia-smi`; use an H20-compatible driver and supported CUDA 12.x toolkit.
- **out of memory:** read the requested bytes and free/total values in the allocation error, stop other GPU jobs, or select the intended device. The fixed memory plan allocates before batch loops.
- **malformed data:** rerun the checksum-verified converter; do not edit binary headers or pass a nonofficial count without the test-only flag.
- **non-finite training:** preserve the diagnostic phase/tensor/index, confirm official data and defaults, and run operator plus Compute Sanitizer tests before retrying.
- **accuracy below 99%:** confirm seed `1337`, 20 epochs, the learning-rate schedule, official counts, strict-best checkpoint reload, and no modified augmentation or normalization constants.

The complete target-host procedure and evidence checklist are in
`docs/h20-acceptance.md`.
