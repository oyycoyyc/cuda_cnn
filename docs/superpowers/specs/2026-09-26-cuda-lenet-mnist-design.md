# CUDA LeNet MNIST Training and Inference Design

## 1. Purpose

Build a readable C++/CUDA project that trains and runs a modern LeNet model on the MNIST dataset published at `https://huggingface.co/datasets/ylecun/mnist`.

The GPU implementation must use custom CUDA kernels for every neural-network operation. It may use the CUDA Runtime API for allocation, copies, kernel launches, synchronization, device queries, error reporting, and CUDA Event timing. CUDA language intrinsics and device math functions such as `expf`, `logf`, `sqrtf`, and `isfinite` are allowed. It must not include or call cuDNN, cuBLAS, TensorRT, Thrust, CUB, or another framework implementation of a neural-network operator.

The target compute platform is Ubuntu 18.04 with Python 3.6 and an NVIDIA H20 GPU. The baseline toolchain is CUDA Toolkit 12.1.x, an H20-capable compatible NVIDIA driver, GCC 7.5, and C++14. The build emits native `sm_90` code and `compute_90` PTX. A cluster-provided newer CUDA 12.x toolkit is acceptable if it supports `sm_90`; the acceptance log records `nvcc --version`, `nvidia-smi`, GCC version, and the runtime-reported compute capability. The development machine has no NVIDIA GPU or local CUDA toolchain, so final CUDA compilation, GPU numerical comparison, training, and accuracy acceptance run on the H20 platform.

## 2. Goals

- Download the specified Hugging Face MNIST train and test Parquet files and convert them to a compact, validated binary format.
- Train modern LeNet in FP32 with custom CUDA forward, backward, loss, and optimizer kernels.
- Save the best validation checkpoint and load it for evaluation or inference.
- Reach at least 99% classification accuracy on the official 10,000-image MNIST test split.
- Provide CPU reference operators and CPU-versus-CUDA numerical tests.
- Provide detailed comments that explain non-obvious implementation decisions and mathematical indexing.
- Provide reproducible commands and fixed random seeds.

## 3. Non-Goals

- ResNet or another model architecture.
- Mixed-precision, Tensor Core, distributed, or multi-GPU training.
- Calling NVIDIA math, neural-network, template, or inference libraries beyond the CUDA Runtime API and CUDA built-in/device math functions.
- Matching framework-level peak H20 throughput.
- Implementing a general-purpose tensor or autograd framework.
- Training on the local Windows machine, which has no NVIDIA GPU or CUDA compiler.

## 4. Model Architecture

The model is the common modern ReLU/max-pooling LeNet variant for 28 x 28 MNIST images.

| Stage | Input | Operation | Output |
|---|---|---|---|
| Input | - | Normalized grayscale image | `N x 1 x 28 x 28` |
| Conv1 | `N x 1 x 28 x 28` | 6 filters, 5 x 5, stride 1, no padding | `N x 6 x 24 x 24` |
| ReLU1 | `N x 6 x 24 x 24` | Elementwise ReLU | `N x 6 x 24 x 24` |
| Pool1 | `N x 6 x 24 x 24` | 2 x 2 max pool, stride 2 | `N x 6 x 12 x 12` |
| Conv2 | `N x 6 x 12 x 12` | 16 filters, 5 x 5, stride 1, no padding | `N x 16 x 8 x 8` |
| ReLU2 | `N x 16 x 8 x 8` | Elementwise ReLU | `N x 16 x 8 x 8` |
| Pool2 | `N x 16 x 8 x 8` | 2 x 2 max pool, stride 2 | `N x 16 x 4 x 4` |
| Flatten | `N x 16 x 4 x 4` | Contiguous reshape | `N x 256` |
| FC1 | `N x 256` | Linear 256 -> 120 and ReLU | `N x 120` |
| FC2 | `N x 120` | Linear 120 -> 84 and ReLU | `N x 84` |
| FC3 | `N x 84` | Linear 84 -> 10 | `N x 10` |
| Loss | `N x 10` | Stable softmax cross-entropy | Scalar mean loss |

Tensor activations use contiguous NCHW layout. Convolution weights use OIHW layout. Fully connected weights use `[out_features][in_features]` row-major layout. Labels are integer class IDs in `[0, 9]`.

## 5. Repository Layout

```text
cuda_lenet_mnist/
|-- Makefile
|-- README.md
|-- requirements-py36.txt
|-- scripts/
|   |-- prepare_mnist.py
|   |-- check_prohibited.sh
|   `-- check_comments.py
|-- include/
|   |-- cuda_check.h
|   |-- tensor.h
|   |-- dataset.h
|   |-- checkpoint.h
|   |-- layers.h
|   `-- lenet.h
|-- src/
|   |-- main.cu
|   |-- lenet.cu
|   |-- train.cu
|   |-- dataset.cpp
|   |-- checkpoint.cpp
|   `-- kernels/
|       |-- convolution.cu
|       |-- pooling.cu
|       |-- activation.cu
|       |-- linear.cu
|       |-- loss.cu
|       |-- adam.cu
|       |-- input.cu
|       `-- metrics.cu
|-- tests/
|   |-- cpu_reference.cpp
|   |-- operator_tests.cu
|   `-- test_prepare_mnist.py
`-- docs/
    |-- comment-review-checklist.md
    `-- superpowers/
        `-- specs/
```

The split is intentionally small and layer-oriented. Headers expose shape-aware launcher functions, while raw `__global__` kernels remain in the corresponding `.cu` implementation file.

## 6. Dataset Preparation

### 6.1 Source Files

The Python 3.6 preparation script downloads these files from pinned Hugging Face revision `77f3279092a1c1579b2250db8eafed0ad422088c`:

- `https://huggingface.co/datasets/ylecun/mnist/resolve/77f3279092a1c1579b2250db8eafed0ad422088c/mnist/train-00000-of-00001.parquet`
- `https://huggingface.co/datasets/ylecun/mnist/resolve/77f3279092a1c1579b2250db8eafed0ad422088c/mnist/test-00000-of-00001.parquet`

Expected SHA-256 values are `f2c01285a9f89399335b00ee4e8d499dc4e46db5e39c74903ce5618d895eb3bf` for train and `d49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b` for test. Downloads go to temporary files, are hashed before PyArrow opens them, and are atomically renamed only after verification. Expected production row counts are 60,000 and 10,000. Each decoded image must be grayscale and exactly 28 x 28. Each label must be in `[0, 9]`.

### 6.2 Python 3.6 Dependencies

`requirements-py36.txt` pins versions compatible with Python 3.6:

- `numpy==1.19.5`
- `Pillow==8.4.0`
- `pyarrow==6.0.1`
- `requests==2.27.1`
- `certifi==2021.10.8`
- `charset-normalizer==2.0.12`
- `idna==3.3`
- `urllib3==1.26.18`

The final requirements file pins direct and transitive dependencies with hashes and is installed with `pip install --require-hashes`. The script uses only Python 3.6 syntax and APIs, these packages, and the Python standard library. It does not use CUDA or a neural-network framework.

### 6.3 Binary Dataset Format

The converter writes `data/train.bin` and `data/test.bin` in little-endian order:

| Field | Type | Value |
|---|---|---|
| Magic | 8 bytes | `MNISTC1\0` |
| Version | `uint32` | `1` |
| Sample count | `uint32` | `60000` or `10000` |
| Rows | `uint32` | `28` |
| Columns | `uint32` | `28` |
| Images | `uint8[count][28][28]` | Row-major pixels |
| Labels | `uint8[count]` | Class IDs |

The converter accepts an explicit expected row count. Production conversion passes 60,000 or 10,000; synthetic tests pass their fixture count. The C++ loader verifies magic, version, nonzero sample count, 28 x 28 dimensions, exact file-length/count consistency, and label range. The `train` and `evaluate` commands additionally enforce the official production split counts unless a test-only `--allow-nonstandard-count` flag is supplied.

## 7. Training Data Flow

1. Load the converted training and test files into host memory.
2. Create a deterministic 55,000/5,000 train/validation split from the 60,000 training images using the configured seed and the project-owned random procedure below.
3. Shuffle only training indices at the start of each epoch.
4. Copy each batch of packed `uint8` images and labels to preallocated device input buffers.
5. A normalization/augmentation kernel converts pixels to FP32 with `(pixel / 255 - 0.1307) / 0.3081`.
6. During training only, the same kernel applies a deterministic per-sample translation of at most two pixels in each direction, with zero fill. The seed, epoch, and sample index determine the shift so runs are reproducible.
7. Execute the LeNet forward pass, stable softmax cross-entropy, backward pass, and Adam updates entirely on the GPU.
8. Compute validation accuracy after each epoch without shuffling or augmentation.
9. Save model parameters only when validation accuracy is strictly greater than the prior best; an exact tie keeps the earlier checkpoint.
10. After training, reload the best checkpoint and evaluate the untouched 10,000-image test set.

Default hyperparameters:

- Batch size: `128`
- Epochs: `20`
- Optimizer: Adam
- Initial learning rate: `0.001`
- Adam beta1: `0.9`
- Adam beta2: `0.999`
- Adam epsilon: `1e-8`
- Decoupled AdamW weight decay: `1e-4` on weights, not biases
- Learning-rate schedule: epochs 1-12 use `1e-3`, epochs 13-17 use `1e-4`, and epochs 18-20 use `1e-5`
- Seed: `1337`

The final partial batch is supported; kernels receive the actual batch size rather than assuming every batch has 128 samples.

### 7.1 Reproducibility Contract

The project implements SplitMix64 rather than relying on implementation-dependent `std::shuffle` or `std::normal_distribution`. Fisher-Yates permutations use rejection sampling to map random 64-bit values to an unbiased bounded integer. He initialization uses SplitMix64 uniforms and the Box-Muller transform. Translation offsets derive from a SplitMix64 hash of `(seed, epoch, original_sample_index)` and map independently to `[-2, 2]`.

The guarantee is deterministic results for repeated runs of the same binary on the same platform and GPU configuration; bit-identical checkpoints across different compilers, CUDA versions, or GPU architectures are not promised. Tests fix the expected split prefix, first-epoch shuffle prefix, translation offsets, and initialized parameter samples for seed `1337` on the target toolchain.

## 8. CUDA Operator Design

### 8.1 Convolution

The direct forward kernel assigns one CUDA thread to one output element `(n, oc, oh, ow)`. That thread loops over input channels and the 5 x 5 receptive field, accumulates in FP32, and adds one output-channel bias.

Backward input assigns one thread to each input gradient element and gathers all output-gradient contributions whose receptive fields cover that input coordinate. Backward weight assigns one thread to each weight element and accumulates across batch and output spatial positions. Backward bias assigns one thread to each output channel and reduces its output gradients.

This gather-based design avoids write races and `atomicAdd` in the main convolution gradients. It is not intended to maximize H20 occupancy, but the LeNet tensor sizes are small and the implementation is deterministic and auditable.

### 8.2 ReLU

Forward applies `max(0, x)` elementwise. Backward multiplies the incoming gradient by `x > 0`. The pre-activation input remains available until backward completes.

### 8.3 Max Pooling

Forward assigns one thread to each pooled output and stores the winning input offset in a compact index buffer. Ties select the first element in row-major order for deterministic behavior. Backward first zeroes the input-gradient buffer, then scatters each output gradient to its recorded winner. Pooling windows do not overlap for the configured 2 x 2 stride-2 layers, so these writes cannot race.

### 8.4 Fully Connected Layers

Forward assigns one thread to each `(batch, output_feature)` pair and loops over input features. Backward input assigns one thread to each input-gradient element. Backward weight assigns one thread to each weight and sums over the batch. Backward bias reduces over the batch.

### 8.5 Softmax Cross-Entropy

One CUDA block handles one sample. The first kernel subtracts the maximum logit before exponentiation, reduces the exponential sum, writes one loss value for that sample, and writes `(probability - one_hot(label)) / batch_size` as the logits-gradient row. A second project-owned deterministic reduction kernel sums the `N` per-sample losses in fixed order and divides by the actual batch size. The reductions work without an external reduction library and never have multiple blocks racing on the same scalar.

### 8.6 Adam

One thread updates one parameter and its first and second moments using decoupled AdamW. The first update uses global step `t = 1`:

```text
m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
p_t = p_(t-1) - lr * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * p_(t-1))
```

For bias tensors, `weight_decay` is zero. Bias-correction factors are computed on the host in double precision and passed to the launcher as FP32 values. A separate finite-value check kernel reports the first tensor containing `NaN` or `Inf`.

### 8.7 Supporting Kernels

The complete project-owned kernel inventory also includes:

- Input conversion, normalization, and deterministic translation with zero fill in `input.cu`.
- Buffer zeroing where a backward path requires it.
- Label-independent stable softmax for inference.
- Argmax and per-sample correct flags in `metrics.cu`.
- Deterministic reductions for mean loss and correct-count totals.
- Finite-value scanning for activations, losses, gradients, and parameters.

Flatten is a zero-copy shape reinterpretation because `N x 16 x 4 x 4` is already contiguous. Evaluation computes predictions and correct flags on the GPU, reduces the correct count with a project-owned kernel, and copies only the final count to the host.

## 9. Memory Management

- Allocate model parameters, gradients, Adam moments, activations, pooling indices, labels, and work buffers once before training.
- Reuse activation and gradient buffers when lifetimes do not overlap, but prefer explicit buffers where reuse would make backward logic difficult to understand.
- Do not call `cudaMalloc` or `cudaFree` inside the batch loop.
- Use one CUDA stream in the initial implementation to keep ordering explicit.
- Use RAII wrappers for device buffers so early error exits release memory.
- Query free device memory before allocation and print required versus available bytes on failure.

## 10. Parameter Initialization and Checkpoints

Convolution and fully connected weights use deterministic He normal initialization with standard deviation `sqrt(2 / fan_in)`. Biases start at zero.

The checkpoint is a little-endian binary file. Fields are serialized individually; the implementation must not write a compiler-padded C++ struct.

Header fields:

| Field | Type | Required value |
|---|---|---|
| Magic | 8 bytes | `LNETC01\0` |
| Format version | `uint32` | `1` |
| Architecture ID | `uint32` | `1`, meaning the Section 4 modern LeNet |
| Tensor count | `uint32` | `10` |
| Best epoch | `uint32` | One-based epoch number |
| Validation accuracy | `float32` | Value in `[0, 1]` |
| Normalization mean | `float32` | `0.1307` |
| Normalization standard deviation | `float32` | `0.3081` |
| Reserved | `uint32` | `0` |

Each tensor record contains a 32-byte zero-padded ASCII name, `uint32 rank`, four `uint32` dimensions with unused dimensions set to one, `uint64 element_count`, and `element_count` contiguous FP32 values. Canonical order, names, and dimensions are:

| Name | Dimensions |
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

Loading rejects unknown architecture IDs, missing or duplicate tensors, dimension mismatches, truncated data, trailing bytes, and non-finite parameters. Adam state is not stored because resume-training support is outside the required scope.

## 11. Command-Line Interface

```bash
# Download and convert Hugging Face MNIST.
python3.6 scripts/prepare_mnist.py --output-dir data

# Build for H20. CUDA_ARCH may be overridden if the platform reports another target.
make CUDA_ARCH=sm_90

mkdir -p weights

# Train, validate, save the best model, then report final test accuracy.
./build/lenet_cuda train \
  --train data/train.bin \
  --test data/test.bin \
  --output weights/lenet.bin \
  --epochs 20 \
  --batch-size 128 \
  --seed 1337 \
  --device 0

# Evaluate and fail if accuracy is below 99%.
./build/lenet_cuda evaluate \
  --data data/test.bin \
  --weights weights/lenet.bin \
  --min-accuracy 0.99 \
  --device 0

# Infer one indexed test sample and print logits, probabilities, prediction, and label.
./build/lenet_cuda infer \
  --data data/test.bin \
  --weights weights/lenet.bin \
  --index 0 \
  --device 0

# Run deterministic CPU-versus-CUDA operator and gradient tests.
./build/operator_tests
```

Command constraints:

| Command/option | Requirement or range |
|---|---|
| `train --train --test --output` | Required paths; output parent directory must already exist |
| `train --epochs` | Default 20; integer in `[1, 1000]` |
| `train --batch-size` | Default 128; integer in `[1, 1024]` |
| `train --seed` | Default 1337; unsigned 64-bit integer |
| `evaluate --data --weights` | Required paths |
| `evaluate --min-accuracy` | Default 0; finite value in `[0, 1]` |
| `infer --data --weights --index` | Required; index must be within the dataset |
| `--device` | Default 0; must identify an available CUDA device |
| `--allow-nonstandard-count` | Test-only flag accepted by train/evaluate for synthetic or limited datasets |

At startup, the executable selects the requested device, prints its name and compute capability, and rejects an unavailable device. The H20 acceptance run requires compute capability 9.0. A machine with no CUDA device receives a clear error and nonzero exit code. Unknown options, missing required arguments, invalid numeric values, out-of-range indexes, missing output parent directories, and unreadable files also fail with a concise usage message.

Machine-readable summary lines use stable `key=value` fields for epoch, loss, accuracy, elapsed time, and final status. Evaluation performs one unmeasured warm-up batch, then uses CUDA Events around forward kernels for the remaining batches and reports mean forward milliseconds and images per second. Performance is informational and has no acceptance threshold.

## 12. Error Handling

- A `CUDA_CHECK` helper checks every CUDA Runtime call and reports expression, source file, line, CUDA error code, and CUDA error text.
- A kernel-launch helper checks `cudaGetLastError` after launch and synchronizes at test or phase boundaries where asynchronous failures must be attributed accurately.
- Dataset and checkpoint errors include the path and failed invariant.
- Training aborts if loss, activations, gradients, or parameters become non-finite.
- Allocation errors include requested bytes and device free/total memory.
- No error path silently switches to CPU training or inference.

## 13. Commenting and Readability Standard

Comments are a required deliverable, not an optional cleanup step.

Every declaration in a public header, including every type, function, method, and kernel launcher, documents:

- Its purpose.
- Input and output tensor shapes and layouts.
- Ownership and lifetime expectations.
- Whether buffers may alias.
- Preconditions and error behavior.

Every project-owned `__global__` function, without a "trivial kernel" exception, starts with a detailed comment block that explains:

- The mathematical operation.
- How block and thread indices map to tensor coordinates.
- The linear-index formula for each tensor layout.
- Which loop dimensions are accumulated by one thread.
- Boundary handling.
- Whether writes can race and why atomics or synchronization are or are not needed.
- Numerical-stability choices.

Backward kernels additionally document the derivative being implemented and identify which forward values or index buffers they consume. Reduction code documents shared-memory layout and synchronization points. Binary serialization code documents every field and byte-order assumption.

Comments must explain intent and invariants rather than restating obvious assignments. Names, small functions, and consistent tensor-index helpers carry the straightforward parts of the implementation.

`scripts/check_comments.py` verifies that every public declaration and every `__global__` definition has an adjacent documentation block. `docs/comment-review-checklist.md` lists every public declaration, kernel, reduction, binary serializer, and parser. Acceptance requires manually checking off shape/layout, indexing/formula, race/synchronization, boundary, numerical, ownership, and error-behavior explanations as applicable. The automated check enforces presence; the manual checklist enforces substance.

## 14. Testing Strategy

### 14.1 Data and Serialization

- Convert a three-row synthetic Parquet dataset with `--expected-count 3` and verify exact binary header and payload bytes.
- Reject incorrect image dimensions, labels, counts, and truncated files.
- Verify pinned source URLs and SHA-256 values before Parquet deserialization.
- Round-trip a deterministic checkpoint and compare every parameter bit-for-bit.
- Reject malformed checkpoint metadata and tensor dimensions.
- Run `python3.6 -m py_compile scripts/prepare_mnist.py` and `python3.6 -m unittest tests/test_prepare_mnist.py` under the target interpreter.

### 14.2 CPU Reference Operators

Implement straightforward CPU versions of convolution, ReLU, max pooling, fully connected, softmax cross-entropy, and their required gradients. They prioritize clarity rather than speed and are used only by tests.

### 14.3 CUDA Numerical Tests

- Compare each CUDA forward operator against the CPU reference on small deterministic tensors generated from seed `1337`.
- Compare backward-input, backward-weight, and backward-bias outputs against CPU references.
- An ordinary comparison passes when `abs(actual - reference) <= 1e-4 + 1e-4 * abs(reference)` for every element.
- Use centered finite differences with epsilon `1e-3` on the first, center, and last weight of each convolution and fully connected tensor. A gradient passes when `abs(analytic - numeric) <= 1e-2 + 1e-2 * abs(numeric)`.
- Test non-divisible element counts and final partial batches to exercise launch boundaries.
- Verify max-pool tie behavior and stable softmax with large logits.
- Compare one and ten AdamW updates against an FP64 CPU calculation using the exact Section 8.6 equations.
- Verify exact split prefixes, shuffle prefixes, translation offsets, and initialization samples for seed `1337` on the target toolchain.
- Verify the finite-value scanner detects injected `NaN`, positive infinity, and negative infinity.
- Run `compute-sanitizer --tool memcheck ./build/operator_tests` with zero reported memory errors.

### 14.4 Integration and Acceptance

- Run a batch of 17 samples through forward/backward/update and require finite loss, gradients, and parameters.
- Overfit a fixed 32-sample subset for 200 updates; require final loss at most 20% of initial loss and training accuracy at least 95%.
- Run one epoch with 1,024 train and 256 validation samples using `--allow-nonstandard-count`; require a checkpoint that can be reloaded for evaluation.
- Test malformed CLI options, missing files, invalid numeric ranges, missing output parents, unavailable devices, and out-of-range inference indexes; each must return nonzero.
- Train with the documented defaults on H20.
- Run `evaluate --min-accuracy 0.99`; the command must exit zero and report at least 99% on the official test split.
- Run source and verbose-build scans that reject prohibited headers, namespaces, APIs, and linker flags for cuDNN, cuBLAS, TensorRT, Thrust, and CUB.
- Run `ldd` and `readelf -d` to confirm there is no dynamic dependency on cuDNN, cuBLAS, or TensorRT. Thrust/CUB compliance is established by source and verbose-build inspection because they are header libraries.
- Run the automated comment-presence check and complete the manual comment-review checklist.

## 15. Documentation

`README.md` will include:

- Constraint summary and explicit prohibited-library list.
- Ubuntu 18.04, Python 3.6, CUDA 12.1.x baseline, GCC 7.5, compatible driver, and H20 prerequisites.
- Dataset source URLs and conversion commands.
- Network table and tensor layouts.
- Build, test, train, evaluate, and infer commands.
- Default hyperparameters and expected output.
- Checkpoint and dataset formats.
- Troubleshooting for CUDA architecture, driver/toolkit mismatch, out-of-memory, malformed data, non-finite training, and accuracy below 99%.
- A clear statement that CUDA execution was not available on the local development machine and must be verified on the H20 platform.

## 16. Acceptance Criteria

- The project is located at `E:\cuda_lenet_mnist` locally and transfers cleanly to Ubuntu.
- Python preparation code passes compile and unit tests under Python 3.6 with direct and transitive dependencies pinned by version and hash.
- The acceptance log records CUDA, driver, compiler, GPU name, and compute-capability versions.
- `nvcc` builds C++14 native `sm_90` plus `compute_90` PTX using only the CUDA Runtime and permitted CUDA device built-ins from NVIDIA.
- All neural-network forward, backward, loss, and optimizer operations use project-owned CUDA kernels.
- Source/build and dynamic-dependency checks find no prohibited library or header use.
- Operator and gradient tests pass on the H20 platform.
- Compute Sanitizer reports zero memory errors.
- The train/evaluate/infer workflows operate from the documented CLI.
- Final official MNIST test accuracy is at least 99%.
- The automated and manually reviewed evidence shows that source code satisfies the detailed commenting standard in Section 13.
- No ResNet code or architecture remains in the project.
