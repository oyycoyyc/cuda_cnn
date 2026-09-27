# CUDA LeNet MNIST Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild a readable C++14/CUDA project that prepares MNIST, trains and evaluates modern LeNet with project-owned CUDA kernels, and reaches at least 99% official test accuracy on NVIDIA H20.

**Architecture:** CUDA-independent host modules own binary parsing, deterministic random procedures, parameter metadata, checkpoints, CLI parsing, and reporting. Layer-oriented CUDA files expose shape-aware launchers; `LeNet` owns all device buffers and composes those launchers into forward, backward, update, evaluation, and inference workflows without allocating in batch loops. Python is used only for verified dataset preparation and static checks.

**Tech Stack:** C++14, CUDA Runtime API and CUDA device built-ins, CUDA Toolkit 12.1.x, GCC 7.5, GNU Make, Python 3.6, NumPy 1.19.5, Pillow 8.4.0, PyArrow 6.0.1, Requests 2.27.1.

**Spec:** `docs/superpowers/specs/2026-09-26-cuda-lenet-mnist-design.md`

## Global Constraints

- Target Ubuntu 18.04, Python 3.6, GCC 7.5, CUDA Toolkit 12.1.x, C++14, and NVIDIA H20 compute capability 9.0.
- Emit both `sm_90` native code and `compute_90` PTX.
- Use custom CUDA kernels for every neural-network forward, backward, loss, metric, and optimizer operation.
- Use no cuDNN, cuBLAS, TensorRT, Thrust, CUB, cuRAND, neural-network framework, or CPU fallback.
- Use contiguous NCHW activations, OIHW convolution weights, and row-major `[out][in]` linear weights.
- Use only FP32 model arithmetic; no mixed precision, Tensor Cores, distributed execution, or multi-GPU execution.
- Use fixed default seed `1337`; do not use `std::shuffle` or `std::normal_distribution`.
- Allocate all device storage before a training/evaluation loop and use one CUDA stream.
- Serialize dataset and checkpoint fields individually in little-endian order; never serialize compiler-padded structs.
- Every public declaration and every project-owned `__global__` definition must satisfy the documentation rules in spec Section 13 when first introduced.
- Local Windows work may verify Python, host C++, static checks, and documentation only. CUDA compilation, numerical comparison, sanitizer, training, and accuracy claims require the H20 host.
- The existing worktree contains only the design document and is not a Git repository. Task 1 bootstraps version control before implementation commits.

### Locked Constants

- Training Parquet URL: `https://huggingface.co/datasets/ylecun/mnist/resolve/77f3279092a1c1579b2250db8eafed0ad422088c/mnist/train-00000-of-00001.parquet`; SHA-256: `f2c01285a9f89399335b00ee4e8d499dc4e46db5e39c74903ce5618d895eb3bf`; rows: `60000`.
- Test Parquet URL: `https://huggingface.co/datasets/ylecun/mnist/resolve/77f3279092a1c1579b2250db8eafed0ad422088c/mnist/test-00000-of-00001.parquet`; SHA-256: `d49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b`; rows: `10000`.
- Python versions: `numpy==1.19.5`, `Pillow==8.4.0`, `pyarrow==6.0.1`, `requests==2.27.1`, `certifi==2021.10.8`, `charset-normalizer==2.0.12`, `idna==3.3`, and `urllib3==1.26.18`.
- Dataset header: magic `MNISTC1\0`, version `1`, then sample count, rows `28`, columns `28`, all packed images, and all labels.
- Model: Conv `1->6,5x5` gives `N x 6 x 24 x 24`; ReLU; pool gives `N x 6 x 12 x 12`; Conv `6->16,5x5` gives `N x 16 x 8 x 8`; ReLU; pool gives `N x 16 x 4 x 4`; flatten `256`; linear/ReLU `256->120`; linear/ReLU `120->84`; linear `84->10`.
- Defaults: batch size `128`, epochs `20`, learning rate `0.001`, Adam beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`, AdamW weight decay `1e-4` on weights only, and seed `1337`.
- Schedule: epochs `1-12` use `1e-3`, epochs `13-17` use `1e-4`, and epochs `18-20` use `1e-5`.
- Normalization: mean `0.1307`, standard deviation `0.3081`; training translation is independently selected in `[-2, 2]` for each axis with zero fill.
- Checkpoint header: magic `LNETC01\0`, format version `1`, architecture ID `1`, tensor count `10`, one-based best epoch, validation accuracy in `[0,1]`, normalization constants, and reserved zero.
- Checkpoint tensor order and dimensions: `conv1.weight [6,1,5,5]`, `conv1.bias [6]`, `conv2.weight [16,6,5,5]`, `conv2.bias [16]`, `fc1.weight [120,256]`, `fc1.bias [120]`, `fc2.weight [84,120]`, `fc2.bias [84]`, `fc3.weight [10,84]`, `fc3.bias [10]`.
- CLI ranges: epochs `[1,1000]`, batch size `[1,1024]`, finite minimum accuracy `[0,1]`, unsigned 64-bit seed, nonnegative in-range inference index, and an available CUDA device (default `0`).

---

## File Map

| File | Responsibility |
|---|---|
| `Makefile` | C++14 host/CUDA builds, dependency files, test/check/acceptance targets, explicit `sm_90` and `compute_90` generation |
| `.gitignore` | Build, data, weights, logs, Python cache, and downloaded artifact exclusions |
| `requirements-py36.txt` | Direct and transitive Python 3.6 dependencies pinned by version and hash |
| `scripts/prepare_mnist.py` | Verified downloads, Parquet validation, 28x28 grayscale decoding, atomic binary conversion |
| `scripts/check_prohibited.sh` | Source and verbose-build scans for prohibited headers, APIs, namespaces, and linker flags |
| `scripts/check_comments.py` | Public declaration and CUDA kernel documentation-presence checks |
| `scripts/run_h20_acceptance.sh` | Fail-fast H20 environment, build, test, train, evaluation, dependency, and artifact evidence |
| `include/cuda_check.h` | CUDA error reporting and launch checks |
| `include/tensor.h` | Move-only RAII device buffers and allocation accounting |
| `include/dataset.h` | Validated packed MNIST host representation |
| `include/random.h` | SplitMix64, unbiased bounded sampling, split/shuffle, translation, and He initialization primitives |
| `include/parameters.h` | Canonical ten-tensor LeNet parameter schema and host values |
| `include/checkpoint.h` | Checkpoint metadata and load/save API |
| `include/layers.h` | Shape-aware CUDA operator launcher declarations |
| `include/lenet.h` | Model/device-state ownership and complete forward/backward/update API |
| `include/cli.h` | Typed command-line options and strict parser |
| `include/train.h` | Train, evaluate, and infer workflow interfaces |
| `include/reporting.h` | Stable machine-readable summary formatting |
| `src/binary_io.h` | Internal checked little-endian scalar I/O shared by dataset/checkpoint code |
| `src/dataset.cpp` | Dataset parser and official-count enforcement |
| `src/random.cpp` | Deterministic host split, shuffle, and initialization implementation |
| `src/parameters.cpp` | Canonical schema validation and initialized parameter construction |
| `src/checkpoint.cpp` | Atomic checkpoint writer and defensive parser |
| `src/cli.cpp` | CUDA-independent CLI parsing and validation |
| `src/reporting.cpp` | Locale-independent `key=value` output |
| `src/main.cu` | Device selection, command dispatch, usage/runtime exit mapping |
| `src/lenet.cu` | Device storage, model composition, import/export, finite checks |
| `src/train.cu` | Batch packing, scheduling, train/evaluate/infer workflows and CUDA Event timing |
| `src/kernels/*.cu` | Input, activation, pooling, convolution, linear, loss, AdamW, metrics, and finite-scan kernels |
| `tests/test_harness.h` | Dependency-free host/CUDA assertions and named test runner |
| `tests/cpu_reference.{h,cpp}` | Clear CPU forward/backward operators and FP64 AdamW reference |
| `tests/*_tests.cpp` | Python-independent host tests |
| `tests/operator_tests.cu` | CPU-versus-CUDA operator, gradient, boundary, RNG, and finite-scan tests |
| `tests/workflow_tests.cu` | Batch-17, overfit-32, checkpoint-reload, and synthetic workflow tests |
| `tests/test_*.py` | Dataset converter, static checker, interoperability, and documentation tests |
| `docs/comment-review-checklist.md` | Manual semantic review of declarations, kernels, reductions, and binary parsers |
| `docs/h20-acceptance.md` | Target-host prerequisites, exact commands, evidence, and troubleshooting |
| `README.md` | User-facing architecture, formats, commands, constraints, and troubleshooting |

---

### Task 1: Repository and Build/Test Skeleton

**Files:**
- Create: `.gitignore`
- Create: `Makefile`
- Create: `tests/test_harness.h`
- Create: `tests/smoke_tests.cpp`

**Interfaces:**
- Produces: `make host-tests`, `make cuda-tests`, `make test`, `make check`, `make acceptance`, and build directories used by every later task.
- Produces: `TEST_CASE(name)`, `EXPECT_TRUE(value)`, `EXPECT_EQ(expected, actual)`, `EXPECT_NEAR(expected, actual, tolerance)`, and `EXPECT_THROW_CONTAINS(expression, text)`.

- [ ] **Step 1: Initialize version control and write the failing smoke target check**

Run:

```bash
git init
make host-tests
```

Expected: Git initializes successfully and Make fails because `host-tests` is absent.

- [ ] **Step 2: Add the build policy and test harness**

Use these required CUDA flags in `Makefile`:

```make
CUDA_ARCH ?= sm_90
CUDA_COMPUTE := compute_$(patsubst sm_%,%,$(CUDA_ARCH))
CXXFLAGS := -std=c++14 -O2 -Wall -Wextra -Wpedantic
NVCCFLAGS := -std=c++14 -O2 -lineinfo \
  -gencode=arch=$(CUDA_COMPUTE),code=$(CUDA_ARCH) \
  -gencode=arch=$(CUDA_COMPUTE),code=$(CUDA_COMPUTE)
```

The harness must collect named functions, return nonzero on any failed assertion, and print exactly one final `event=test_suite name=<suite> status=<pass|fail>` line.

- [ ] **Step 3: Add a passing host smoke test and aggregate targets**

```cpp
TEST_CASE(smoke) {
  EXPECT_EQ(4, 2 + 2);
}
```

`host-tests` builds and runs all `.cpp` suites. `cuda-tests` builds and runs CUDA suites. `test` depends on both. Compilation emits `.d` dependencies with `-MMD -MP`; `V=1` shows complete commands.

- [ ] **Step 4: Verify the skeleton**

Run:

```bash
make clean
make V=1 host-tests
```

Expected: smoke suite passes, strict C++14 compilation emits no warnings, and `build/` contains objects outside source directories.

- [ ] **Step 5: Commit**

```bash
git add .gitignore Makefile tests/test_harness.h tests/smoke_tests.cpp
git commit -m "build: bootstrap C++ and CUDA test targets"
```

### Task 2: Verified MNIST Preparation

**Files:**
- Create: `requirements-py36.txt`
- Create: `scripts/prepare_mnist.py`
- Create: `tests/test_prepare_mnist.py`

**Interfaces:**
- Produces: `convert_parquet(parquet_path, output_path, expected_count)` and CLI `--output-dir`, plus test-only explicit `--input-parquet --output-bin --expected-count` conversion.
- Produces: exact `MNISTC1\0`, version 1, little-endian files consumed by Task 3.

- [ ] **Step 1: Write failing synthetic-conversion tests**

The primary assertion constructs three Arrow rows and compares all bytes:

```python
expected = (b"MNISTC1\0" + struct.pack("<IIII", 1, 3, 28, 28) +
            image0 + image1 + image2 + bytes(bytearray([0, 5, 9])))
self.assertEqual(expected, open(output_path, "rb").read())
```

Also test wrong image mode/size, missing columns, null image data, malformed encoded bytes, labels outside `[0, 9]`, zero/mismatched counts, checksum mismatch before Parquet open, rehash of cached downloads, and no partial output after failure.

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3.6 -m unittest -v tests.test_prepare_mnist
```

Expected: FAIL because `scripts.prepare_mnist` does not exist.

- [ ] **Step 3: Implement verified download and atomic conversion**

Use the exact revision, URLs, hashes, and production counts from spec Section 6. Download into a same-directory temporary path, stream SHA-256 while writing, reject a mismatch, call `os.replace` only after verification, and hash cached files again before reuse. Decode with `PIL.Image.open`, require mode `L` and size `(28, 28)`, then write all image bytes followed by all label bytes.

The requirement file must contain the nine versions in spec Section 6.2 and a `--hash=sha256:<value>` entry for every distribution accepted on the target. Generate the hashes from files downloaded by Python 3.6 pip on Ubuntu and prove installation with `--require-hashes`; do not copy unverified hash values from secondary sources.

- [ ] **Step 4: Verify Python 3.6 and production-format sizes**

```bash
python3.6 -m pip install --require-hashes -r requirements-py36.txt
python3.6 -m py_compile scripts/prepare_mnist.py tests/test_prepare_mnist.py
python3.6 -m unittest -v tests.test_prepare_mnist
python3.6 scripts/prepare_mnist.py --output-dir data
stat -c '%n %s' data/train.bin data/test.bin
```

Expected sizes: `47100024` and `7850024` bytes.

- [ ] **Step 5: Commit**

```bash
git add requirements-py36.txt scripts/prepare_mnist.py tests/test_prepare_mnist.py
git commit -m "feat: prepare verified MNIST binaries"
```

### Task 3: Defensive Dataset Loader

**Files:**
- Create: `src/binary_io.h`
- Create: `include/dataset.h`
- Create: `src/dataset.cpp`
- Create: `tests/dataset_tests.cpp`
- Create: `tests/dataset_probe.cpp`
- Create: `tests/test_data_interop.py`
- Modify: `Makefile`

**Interfaces:**
- Produces: `MnistDataset LoadMnistDataset(const std::string& path)`.
- Produces: `void RequireDatasetCount(const MnistDataset&, uint32_t expected, bool allow_nonstandard, const std::string& role)`.

```cpp
struct MnistDataset {
  std::uint32_t sample_count;
  std::uint32_t rows;
  std::uint32_t columns;
  std::vector<std::uint8_t> images;
  std::vector<std::uint8_t> labels;
  const std::uint8_t* Image(std::uint32_t index) const;
};
```

- [ ] **Step 1: Write malformed-file and interoperability tests**

Cover valid three-row bytes, `Image(1) == images.data() + 784`, bad magic/version, zero count, non-28 dimensions, count multiplication overflow, truncated header/images/labels, trailing bytes, labels `10` and `255`, unreadable paths, out-of-range image access, and official-count enforcement. Every error assertion checks both the path and failed invariant.

- [ ] **Step 2: Run the failing tests**

```bash
make build/dataset_tests build/dataset_probe
./build/dataset_tests
python3.6 -m unittest -v tests.test_data_interop
```

Expected: compile/link failure because loader symbols are missing.

- [ ] **Step 3: Implement exact checked parsing**

`binary_io.h` supplies `ReadExact`, `ReadU32LE`, `ReadU64LE`, `ReadF32LE`, and matching writers. Floating-point conversion uses `memcpy`. The loader checks expected size `24 + count * 785` before allocations and rejects any unread byte or trailing byte.

- [ ] **Step 4: Verify host and Python/C++ agreement**

```bash
make build/dataset_tests build/dataset_probe
./build/dataset_tests
python3.6 -m unittest -v tests.test_data_interop
```

Expected: both suites pass; the probe reports `count=3 rows=28 columns=28` and the exact fixture checksum/labels.

- [ ] **Step 5: Commit**

```bash
git add Makefile include/dataset.h src/binary_io.h src/dataset.cpp tests/dataset_tests.cpp tests/dataset_probe.cpp tests/test_data_interop.py
git commit -m "feat: load validated MNIST binaries"
```

### Task 4: Reproducibility and Parameter Schema

**Files:**
- Create: `include/random.h`
- Create: `src/random.cpp`
- Create: `include/parameters.h`
- Create: `src/parameters.cpp`
- Create: `tests/random_tests.cpp`
- Create: `tests/parameters_tests.cpp`
- Create: `tests/generate_repro_vectors.py`
- Create: `tests/repro_vectors.txt`
- Modify: `Makefile`

**Interfaces:**
- Produces: `SplitMix64::Next`, rejection-sampled `UniformBounded`, `MakeTrainValidationSplit`, `ShuffledTrainingIndices`, `TranslationOffsetForSample`, `LenetParameterSpecs`, `CreateLenetParameters`, `ValidateLenetParameters`, and `InitializeLenetParameters`.

```cpp
#if defined(__CUDACC__)
#define LENET_HOST_DEVICE __host__ __device__
#else
#define LENET_HOST_DEVICE
#endif
class SplitMix64 {
 public:
  LENET_HOST_DEVICE explicit SplitMix64(std::uint64_t seed);
  LENET_HOST_DEVICE std::uint64_t Next();
  LENET_HOST_DEVICE std::uint64_t UniformBounded(
      std::uint64_t exclusive_upper_bound);
 private:
  std::uint64_t state_;
};
struct DatasetSplit {
  std::vector<std::uint32_t> training_indices;
  std::vector<std::uint32_t> validation_indices;
};
struct TranslationOffset { std::int32_t dx; std::int32_t dy; };
DatasetSplit MakeTrainValidationSplit(std::uint32_t sample_count,
                                      std::uint32_t validation_count,
                                      std::uint64_t seed);
std::vector<std::uint32_t> ShuffledTrainingIndices(
    const std::vector<std::uint32_t>& canonical_indices,
    std::uint64_t seed,
    std::uint32_t one_based_epoch);
LENET_HOST_DEVICE TranslationOffset TranslationOffsetForSample(
    std::uint64_t seed, std::uint32_t one_based_epoch,
    std::uint32_t original_index);
struct ParameterSpec {
  const char* name;
  std::uint32_t rank;
  std::array<std::uint32_t, 4> dimensions;
  std::uint64_t element_count;
  std::uint64_t fan_in;
  bool is_bias;
};
struct ParameterTensor {
  std::string name;
  std::uint32_t rank;
  std::array<std::uint32_t, 4> dimensions;
  std::vector<float> values;
};
using ParameterSet = std::vector<ParameterTensor>;
const std::array<ParameterSpec, 10>& LenetParameterSpecs();
ParameterSet CreateLenetParameters();
void ValidateLenetParameters(const ParameterSet& parameters);
void InitializeLenetParameters(std::uint64_t seed, ParameterSet* parameters);
```

Use standard SplitMix64 and explicit domain separation:

```text
next(state): state += 0x9E3779B97F4A7C15; apply xor-shift/multiply finalizer
derive(seed, domain, a, b): mix(mix(mix(seed xor domain) xor a) xor b)
bounded(bound): reject values below (-bound) mod bound, then return value mod bound
```

Freeze these domains: split `0x53504c49545f5631`, epoch shuffle `0x53485546464c4531`, translation X `0x5452414e535f5831`, translation Y `0x5452414e535f5931`, initialization `0x494e49545f563031`. Epochs are one-based. Translation hashes the original dataset index. Each weight tensor derives an independent stream from seed, domain, canonical tensor ordinal, and fan-in. Box-Muller maps to open `(0,1)`, emits cosine then sine, and discards an unused second value at a tensor boundary.

- [ ] **Step 1: Generate independent golden vectors and write failing C++ tests**

`generate_repro_vectors.py` implements the integer protocol using Python integers masked to 64 bits and writes fixed seed-1337 values for five raw outputs, split prefixes, epoch-1 shuffle prefix, four translation samples, and five initial values from each weight tensor. Commit its generated `repro_vectors.txt`; C++ tests read constants copied from that file, not values computed by C++ production functions.

Also test permutation completeness, no duplicates, 55,000/5,000 sizes, epoch independence, bound one, forced rejection, bias positive zero, finite weights, total parameter count `44,426`, and exact ten-tensor names/shapes from spec Section 10.

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3.6 tests/generate_repro_vectors.py --output tests/repro_vectors.txt
make build/random_tests build/parameters_tests
./build/random_tests
./build/parameters_tests
```

Expected: missing C++ interfaces or golden mismatches.

- [ ] **Step 3: Implement the integer protocol and He initialization**

Fisher-Yates uses `UniformBounded(i + 1)`. The official split shuffles indices `0..59999`, assigns the first 55,000 to training and final 5,000 to validation. Initialization standard deviation is `sqrt(2.0 / fan_in)` and all generated doubles are converted once to FP32. Validation rejects missing, duplicate, unknown, reordered, wrongly ranked, wrongly shaped, or wrongly sized tensors.

- [ ] **Step 4: Verify deterministic vectors and schema**

```bash
make build/random_tests build/parameters_tests
./build/random_tests
./build/parameters_tests
```

Expected: integer-derived values match exactly; initialized samples pass `1e-6` tolerance; reruns from seed `1337` are bit-identical in one binary.

- [ ] **Step 5: Commit**

```bash
git add Makefile include/random.h include/parameters.h src/random.cpp src/parameters.cpp tests/random_tests.cpp tests/parameters_tests.cpp tests/generate_repro_vectors.py tests/repro_vectors.txt
git commit -m "feat: add deterministic LeNet initialization"
```

### Task 5: Checkpoint Serialization

**Files:**
- Create: `include/checkpoint.h`
- Create: `src/checkpoint.cpp`
- Create: `tests/checkpoint_tests.cpp`
- Modify: `Makefile`

**Interfaces:**
- Consumes: canonical `ParameterSet` from Task 4 and little-endian helpers from Task 3.
- Produces: `void SaveCheckpoint(const std::string&, const Checkpoint&)` and `Checkpoint LoadCheckpoint(const std::string&)`.

```cpp
struct CheckpointMetadata {
  std::uint32_t best_epoch;
  float validation_accuracy;
  float normalization_mean;
  float normalization_stddev;
};
struct Checkpoint {
  CheckpointMetadata metadata;
  ParameterSet parameters;
};
```

- [ ] **Step 1: Write round-trip and corruption tests**

Test bit-identical float round-trip, byte-identical load/save, expected total size `178344`, exact 40-byte header, 60-byte tensor metadata, and all malformed cases in spec Section 10: magic, version, architecture, tensor count, epoch, accuracy, normalization, reserved field, names/order/duplicates, rank/dimensions/count, truncation, trailing bytes, and non-finite parameters.

- [ ] **Step 2: Run tests to verify they fail**

```bash
make build/checkpoint_tests
./build/checkpoint_tests
```

Expected: compile/link failure because checkpoint symbols are missing.

- [ ] **Step 3: Implement field-wise atomic serialization**

Write a same-directory temporary file, flush and close it, then replace with `std::rename`. Validate metadata and all tensors before touching an existing destination. Names occupy exactly 32 NUL-padded ASCII bytes; unused dimensions are one. Do not serialize Adam moments or global step.

- [ ] **Step 4: Verify**

```bash
make build/checkpoint_tests
./build/checkpoint_tests
```

Expected: all round-trip and rejection tests pass.

- [ ] **Step 5: Commit**

```bash
git add Makefile include/checkpoint.h src/checkpoint.cpp tests/checkpoint_tests.cpp
git commit -m "feat: serialize validated LeNet checkpoints"
```

### Task 6: CPU References and CUDA Foundation

**Files:**
- Create: `tests/cpu_reference.h`
- Create: `tests/cpu_reference.cpp`
- Create: `tests/cpu_reference_tests.cpp`
- Create: `include/cuda_check.h`
- Create: `include/tensor.h`
- Create: `include/layers.h`
- Create: `tests/operator_tests.cu`
- Modify: `Makefile`

**Interfaces:**
- Produces: CPU convolution, ReLU, pool, linear, softmax cross-entropy, gradients, and FP64 AdamW references.
- Produces: `CUDA_CHECK`, `CUDA_KERNEL_CHECK`, move-only `DeviceBuffer<T>`, and all launcher declarations used by Tasks 7-9.

The locked launcher signatures are:

```cpp
void LaunchNormalizeTranslate(const std::uint8_t* images,
    const std::uint32_t* original_indices, float* output, int batch_size,
    std::uint64_t seed, std::uint32_t one_based_epoch, bool augment,
    cudaStream_t stream);
void LaunchZero(float* values, std::size_t count, cudaStream_t stream);
void LaunchReluForward(const float* input, float* output,
    std::size_t count, cudaStream_t stream);
void LaunchReluBackward(const float* forward_input,
    const float* output_gradient, float* input_gradient,
    std::size_t count, cudaStream_t stream);
void LaunchMaxPoolForward(const float* input, float* output,
    std::uint8_t* winner_offsets, int batch_size, int channels,
    int input_height, int input_width, cudaStream_t stream);
void LaunchMaxPoolBackward(const float* output_gradient,
    const std::uint8_t* winner_offsets, float* input_gradient,
    int batch_size, int channels, int input_height, int input_width,
    cudaStream_t stream);
void LaunchLinearForward(const float* input, const float* weight,
    const float* bias, float* output, int batch_size, int input_features,
    int output_features, cudaStream_t stream);
void LaunchLinearBackward(const float* input, const float* weight,
    const float* output_gradient, float* input_gradient,
    float* weight_gradient, float* bias_gradient, int batch_size,
    int input_features, int output_features, cudaStream_t stream);
void LaunchConvolutionForward(const float* input, const float* weight,
    const float* bias, float* output, int batch_size, int input_channels,
    int input_height, int input_width, int output_channels,
    int kernel_height, int kernel_width, cudaStream_t stream);
void LaunchConvolutionBackward(const float* input, const float* weight,
    const float* output_gradient, float* input_gradient,
    float* weight_gradient, float* bias_gradient, int batch_size,
    int input_channels, int input_height, int input_width,
    int output_channels, int kernel_height, int kernel_width,
    cudaStream_t stream);
void LaunchSoftmaxCrossEntropy(const float* logits,
    const std::uint8_t* labels, float* probabilities,
    float* per_sample_losses, float* mean_loss, float* logits_gradient,
    int batch_size, int class_count, cudaStream_t stream);
void LaunchSoftmax(const float* logits, float* probabilities,
    int batch_size, int class_count, cudaStream_t stream);
void LaunchArgmaxAndCountCorrect(const float* logits,
    const std::uint8_t* labels, std::uint8_t* predictions,
    int* correct_flags, int* correct_count, int batch_size,
    int class_count, cudaStream_t stream);
void LaunchAdamW(float* parameters, const float* gradients,
    float* first_moments, float* second_moments, std::size_t count,
    float learning_rate, float beta1, float beta2, float epsilon,
    float inverse_bias_correction1, float inverse_bias_correction2,
    float weight_decay, cudaStream_t stream);
void LaunchFindFirstNonFinite(const float* values, std::size_t count,
    int* first_bad_index, cudaStream_t stream);
```

- [ ] **Step 1: Write hand-calculated CPU and CUDA-infrastructure tests**

Cover convolution forward/backward, ReLU zero derivative, row-major-first pool tie, linear forward/backward, stable large-logit softmax, one and ten FP64 AdamW updates, CUDA error text, invalid launch detection, move-only ownership, zero-count buffer, and a 257-element copy round-trip.

- [ ] **Step 2: Run tests to verify they fail**

```bash
make build/cpu_reference_tests build/operator_tests
./build/cpu_reference_tests
./build/operator_tests --filter device_buffer
```

Expected: reference and CUDA helper symbols are missing.

- [ ] **Step 3: Implement clear references and RAII**

CPU accumulation order matches CUDA loops. `DeviceBuffer<T>` centralizes `cudaMalloc`/`cudaFree`, never throws from its destructor, reports requested/free/total bytes on allocation failure, and exposes allocation counts to tests. Launchers call `cudaGetLastError` after every kernel but do not synchronize implicitly.

- [ ] **Step 4: Verify on host and H20**

```bash
make build/cpu_reference_tests build/operator_tests
./build/cpu_reference_tests
./build/operator_tests --filter cuda_check
./build/operator_tests --filter device_buffer
```

Expected: host references pass everywhere; CUDA infrastructure passes on H20.

- [ ] **Step 5: Commit**

```bash
git add Makefile include/cuda_check.h include/tensor.h include/layers.h tests/cpu_reference.h tests/cpu_reference.cpp tests/cpu_reference_tests.cpp tests/operator_tests.cu
git commit -m "test: add CPU references and CUDA foundations"
```

### Task 7: Input, ReLU, and Pooling Kernels

**Files:**
- Create: `src/kernels/input.cu`
- Create: `src/kernels/activation.cu`
- Create: `src/kernels/pooling.cu`
- Modify: `include/layers.h`
- Modify: `tests/operator_tests.cu`
- Modify: `Makefile`

**Interfaces:**
- Produces: normalization/translation, FP32 zeroing, ReLU forward/backward, and 2x2 stride-2 max-pool forward/backward launchers.

- [ ] **Step 1: Write failing CPU-versus-CUDA tests**

Check pixels 0/255 against `(pixel / 255 - 0.1307) / 0.3081`, zero-filled deterministic shifts for batches 17 and 257, no augmentation in evaluation, ReLU input `[-2,-0.0,0,3]`, zero derivative at zero, non-divisible element counts, pool tie `[5,5;1,0]` selecting offset zero, and backward writing only winners.

- [ ] **Step 2: Run tests to verify they fail**

```bash
make build/operator_tests
./build/operator_tests --filter input
./build/operator_tests --filter relu
./build/operator_tests --filter maxpool
```

Expected: missing launchers or numerical mismatches.

- [ ] **Step 3: Implement documented kernels**

Input threads map one-to-one to NCHW output elements and hash `(seed, one_based_epoch, original_sample_index)`. Pool forward uses strict `>` over offsets `0,1,2,3`; backward first invokes the project-owned zero kernel, then scatters without atomics because configured windows do not overlap. ReLU backward permits gradient in-place aliasing; other buffers do not alias.

- [ ] **Step 4: Verify on H20**

Run the three filtered commands from Step 2. Expected: exact integer/index results and FP32 comparisons within `1e-4 + 1e-4 * abs(reference)`.

- [ ] **Step 5: Commit**

```bash
git add Makefile include/layers.h src/kernels/input.cu src/kernels/activation.cu src/kernels/pooling.cu tests/operator_tests.cu
git commit -m "feat: add input activation and pooling kernels"
```

### Task 8: Linear and Convolution Kernels

**Files:**
- Create: `src/kernels/linear.cu`
- Create: `src/kernels/convolution.cu`
- Modify: `include/layers.h`
- Modify: `tests/operator_tests.cu`
- Modify: `Makefile`

**Interfaces:**
- Produces: forward, backward-input, backward-weight, and backward-bias launchers for linear and valid convolution.

- [ ] **Step 1: Write failing forward/backward and finite-difference tests**

Linear uses shapes `N=3,in=5,out=4` and `N=17,in=120,out=84`. Convolution uses `N=2,Cin=2,H=7,W=8,Cout=3,K=5`, plus Conv1/Conv2 production shapes. Compare all forward and gradient outputs. For each production weight tensor, centered finite differences use epsilon `1e-3` on first, center, and last values and tolerance `1e-2 + 1e-2 * abs(numeric)`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
make build/operator_tests
./build/operator_tests --filter linear
./build/operator_tests --filter convolution
```

Expected: missing launchers or gradient mismatches.

- [ ] **Step 3: Implement gather-based kernels**

Each forward thread owns one output. Backward-input threads gather all covering output gradients. Backward-weight threads gather over batch/output positions. Backward-bias threads reduce the batch/spatial axes. Main convolution gradients contain no `atomicAdd`; every boundary and linear-index formula is documented adjacent to its kernel.

- [ ] **Step 4: Verify on H20**

Run both filtered commands. Expected: ordinary comparisons and all finite-difference points pass.

- [ ] **Step 5: Commit**

```bash
git add Makefile include/layers.h src/kernels/linear.cu src/kernels/convolution.cu tests/operator_tests.cu
git commit -m "feat: add linear and convolution kernels"
```

### Task 9: Loss, Metrics, AdamW, and Finite Scanning

**Files:**
- Create: `src/kernels/loss.cu`
- Create: `src/kernels/metrics.cu`
- Create: `src/kernels/adam.cu`
- Modify: `include/layers.h`
- Modify: `tests/operator_tests.cu`
- Modify: `Makefile`

**Interfaces:**
- Produces: stable softmax cross-entropy and inference softmax, deterministic mean-loss/correct-count reductions, argmax, AdamW, and first-non-finite scan launchers.

- [ ] **Step 1: Write failing numerical and edge tests**

Cover logits `1000` and `-1000`, probability row sums, `(p-one_hot)/actual_batch_size`, batches 1/17/128/129, smallest-index argmax ties, exact correct counts, one/ten AdamW updates against FP64, zero bias decay, first step `t=1`, and injected NaN/+Inf/-Inf at multiple positions with smallest index returned.

- [ ] **Step 2: Run tests to verify they fail**

```bash
make build/operator_tests
./build/operator_tests --filter softmax
./build/operator_tests --filter metrics
./build/operator_tests --filter adamw
./build/operator_tests --filter finite_scan
```

Expected: missing launchers, unstable outputs, or incorrect update/scanner results.

- [ ] **Step 3: Implement stable project-owned reductions and updates**

One block handles each softmax row after max subtraction. A second deterministic kernel reduces per-sample losses and divides once by actual batch size. Correct-count reduction uses fixed-order project-owned stages. AdamW bias corrections are computed on host in FP64 and passed as FP32; decay uses the pre-update parameter and is zero for biases. The scanner initializes result to `count` and uses `atomicMin` only for reporting the first invalid index.

- [ ] **Step 4: Verify on H20**

Run the four filtered commands. Expected: all finite/stability, exact integer, tolerance, and update assertions pass.

- [ ] **Step 5: Commit**

```bash
git add Makefile include/layers.h src/kernels/loss.cu src/kernels/metrics.cu src/kernels/adam.cu tests/operator_tests.cu
git commit -m "feat: add loss metrics and AdamW kernels"
```

### Task 10: LeNet Composition and Memory Plan

**Files:**
- Create: `include/lenet.h`
- Create: `src/lenet.cu`
- Modify: `tests/operator_tests.cu`
- Modify: `Makefile`

**Interfaces:**
- Consumes: ten canonical parameters and all operator launchers.
- Produces: constructor with maximum batch size, `Forward`, `Backward`, `AdamWStep`, `ExportParameters`, `ImportParameters`, `RequireFinite`, and `RequiredDeviceBytes`.

```cpp
struct AdamWConfig {
  float beta1;
  float beta2;
  float epsilon;
  float weight_decay;
};
class LeNet {
 public:
  LeNet(int maximum_batch_size, std::uint64_t seed, cudaStream_t stream);
  ~LeNet();
  LeNet(const LeNet&) = delete;
  LeNet& operator=(const LeNet&) = delete;
  const float* Forward(const float* normalized_images, int batch_size);
  void Backward(const float* logits_gradient, int batch_size);
  void AdamWStep(std::uint64_t global_step, float learning_rate,
                 const AdamWConfig& config);
  ParameterSet ExportParameters() const;
  void ImportParameters(const ParameterSet& parameters);
  void RequireFinite(const std::string& phase) const;
  std::size_t RequiredDeviceBytes() const;
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
```

- [ ] **Step 1: Write failing storage, full-forward, and full-backward tests**

Assert 44,426 parameters, ten canonical names/shapes, zero biases, required-byte accounting, one pre-allocation memory query, stable addresses/allocation count, rejection above max batch, stage-by-stage `N=2` CPU comparison, final `N=17` logits, zero-copy flatten, all parameter/input gradients, 15 network-level finite-difference points, correct bias/weight decay, and no allocation across 200 updates.

- [ ] **Step 2: Run tests to verify they fail**

```bash
make build/operator_tests
./build/operator_tests --filter lenet_storage
./build/operator_tests --filter lenet_forward
./build/operator_tests --filter lenet_backward
./build/operator_tests --filter lenet_train_step
```

Expected: missing model interface or first unimplemented stage mismatch.

- [ ] **Step 3: Implement fixed-shape composition**

Allocate parameters, gradients, moments, activations, activation gradients, pool winners, labels, losses, metrics, input staging, and scan result once. Preserve pre-activation values until backward. Treat `N x 16 x 4 x 4` as `N x 256` by pointer reinterpretation. Reject invalid actual batch before any launch. Scan activations/loss, gradients, and parameters at their phase boundaries and identify the tensor and index in errors.

- [ ] **Step 4: Verify on H20**

Run all four filtered commands. Expected: CPU/CUDA comparisons and finite differences pass, addresses and allocation count stay fixed.

- [ ] **Step 5: Commit**

```bash
git add Makefile include/lenet.h src/lenet.cu tests/operator_tests.cu
git commit -m "feat: compose complete LeNet GPU model"
```

### Task 11: CLI, Reporting, and Host Batch Flow

**Files:**
- Create: `include/cli.h`
- Create: `src/cli.cpp`
- Create: `include/reporting.h`
- Create: `src/reporting.cpp`
- Create: `tests/cli_tests.cpp`
- Create: `tests/training_data_tests.cpp`
- Create: `tests/reporting_tests.cpp`
- Modify: `Makefile`

**Interfaces:**
- Produces: typed `TrainOptions`, `EvaluateOptions`, `InferOptions`, strict `ParseCli`, stable summary formatters, and host batch packing.

```cpp
enum class Command { kTrain, kEvaluate, kInfer };
struct TrainOptions {
  std::string train_path;
  std::string test_path;
  std::string output_path;
  std::uint32_t epochs;
  std::uint32_t batch_size;
  std::uint64_t seed;
  int device;
  bool allow_nonstandard_count;
};
struct EvaluateOptions {
  std::string data_path;
  std::string weights_path;
  float minimum_accuracy;
  int device;
  bool allow_nonstandard_count;
};
struct InferOptions {
  std::string data_path;
  std::string weights_path;
  std::uint64_t index;
  int device;
};
struct CliOptions {
  Command command;
  TrainOptions train;
  EvaluateOptions evaluate;
  InferOptions infer;
};
bool ParseCli(int argc, const char* const* argv, CliOptions* options,
              std::string* error);
std::string Usage();
struct HostBatch {
  std::vector<std::uint8_t> images;
  std::vector<std::uint8_t> labels;
  std::vector<std::uint32_t> original_indices;
  std::uint32_t size;
};
std::uint32_t PackBatch(const MnistDataset& dataset,
                        const std::vector<std::uint32_t>& order,
                        std::uint32_t offset, std::uint32_t capacity,
                        HostBatch* batch);
void PrintEpochSummary(std::ostream& output, std::uint32_t epoch,
                       float training_loss, float validation_accuracy,
                       double elapsed_ms);
void PrintEvaluationSummary(std::ostream& output, std::uint32_t samples,
                            float accuracy, float mean_forward_ms,
                            float images_per_second, float minimum_accuracy,
                            bool passed);
```

- [ ] **Step 1: Write failing parser, ordering, and exact-output tests**

Cover every default/range in spec Section 11; required paths; duplicate/unknown/cross-command options; trailing numeric text; negative unsigned values; overflow; NaN/Inf; output-parent existence; official counts; and out-of-range inference index. Verify batches for 257 samples are 128/128/1 and preserve image/label/original-index correspondence. Freeze exact lines such as:

```text
event=epoch epoch=1 train_loss=0.123456 validation_accuracy=0.987600 elapsed_ms=1234.567
event=evaluate samples=10000 accuracy=0.991200 mean_forward_ms=0.321 images_per_second=398753.875 min_accuracy=0.990000 status=pass
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
make build/cli_tests build/training_data_tests build/reporting_tests
./build/cli_tests
./build/training_data_tests
./build/reporting_tests
```

Expected: missing parser/reporting/batch symbols.

- [ ] **Step 3: Implement strict host logic**

Use `strtoull`/`strtod` with errno, end-pointer, explicit range, and finite checks. The official split remains 55,000/5,000. Under `--allow-nonstandard-count`, require at least two samples and use validation count `max(1, count / 5)`, enabling a 1,280-row fixture to produce 1,024 training and 256 validation samples. Use classic locale and fixed precision for summaries.

- [ ] **Step 4: Verify host suites**

Run all three commands from Step 2. Expected: each prints one passing suite line.

- [ ] **Step 5: Commit**

```bash
git add Makefile include/cli.h include/reporting.h src/cli.cpp src/reporting.cpp tests/cli_tests.cpp tests/training_data_tests.cpp tests/reporting_tests.cpp
git commit -m "feat: add strict CLI and batch reporting"
```

### Task 12: Train, Evaluate, and Infer Workflows

**Files:**
- Create: `include/train.h`
- Create: `src/train.cu`
- Create: `src/main.cu`
- Create: `tests/workflow_tests.cu`
- Modify: `Makefile`

**Interfaces:**
- Produces: `RunTrain`, `RunEvaluate`, `RunInfer`, and `build/lenet_cuda` commands exactly matching spec Section 11.

```cpp
enum ExitCode {
  kSuccess = 0,
  kRuntimeError = 1,
  kUsageError = 2,
  kAcceptanceFailure = 3
};
int RunTrain(const TrainOptions& options, std::ostream& output,
             std::ostream& error);
int RunEvaluate(const EvaluateOptions& options, std::ostream& output,
                std::ostream& error);
int RunInfer(const InferOptions& options, std::ostream& output,
             std::ostream& error);
float LearningRateForEpoch(std::uint32_t one_based_epoch);
```

- [ ] **Step 1: Write failing workflow tests**

Cover unavailable devices; batch 17 forward/backward/update; global step starting at one; epoch 12/13/17/18 learning rates; finite loss/gradients/parameters; strict-greater checkpoint replacement; fixed 32-sample 200-update overfit; one epoch over 1,024/256 split; checkpoint reload before final test; evaluation warm-up exclusion; threshold equality pass and below-threshold exit 3; inference logits/probabilities/prediction/label; malformed paths/files; and all partial batches.

- [ ] **Step 2: Run tests to verify they fail**

```bash
make build/workflow_tests build/lenet_cuda
./build/workflow_tests --case batch17
./build/workflow_tests --case evaluate
./build/workflow_tests --case infer
```

Expected: missing workflow symbols or nonzero test results.

- [ ] **Step 3: Implement workflows and event timing**

At startup select and print requested device name/capability. Train shuffles only training indices, augments only training, weights epoch loss by actual batch size, validates without shuffle/augmentation, saves only on strict accuracy improvement, reloads the on-disk best checkpoint, then tests. Evaluate performs one unmeasured warm-up batch and times remaining forward launches with CUDA Events; GPU kernels produce final correct count and only that scalar is copied for accuracy. Infer prints ten logits, ten probabilities, prediction, and label. No CUDA failure falls back to CPU.

- [ ] **Step 4: Verify integration on H20**

```bash
./build/workflow_tests --case batch17
./build/workflow_tests --case overfit32
./build/workflow_tests --case one_epoch_checkpoint
./build/workflow_tests --case evaluate
./build/workflow_tests --case infer
```

Expected: loss after overfit is at most 20% of initial loss, training accuracy is at least 95%, and every workflow case passes.

- [ ] **Step 5: Commit**

```bash
git add Makefile include/train.h src/train.cu src/main.cu tests/workflow_tests.cu
git commit -m "feat: add training evaluation and inference workflows"
```

### Task 13: Compliance Checks and Documentation

**Files:**
- Create: `scripts/check_prohibited.sh`
- Create: `scripts/check_comments.py`
- Create: `tests/test_check_prohibited.py`
- Create: `tests/test_check_comments.py`
- Create: `tests/test_documentation.py`
- Create: `docs/comment-review-checklist.md`
- Create: `docs/h20-acceptance.md`
- Create: `README.md`
- Modify: `Makefile`

**Interfaces:**
- Produces: automated source/build policy checks and complete user/acceptance documentation.

- [ ] **Step 1: Write failing checker and documentation tests**

Temporary fixtures must prove each prohibited header, namespace, API, and linker flag is rejected. Comment fixtures cover multiline public declarations, types, methods, launcher declarations, and every `__global__`. Documentation tests require every README item in spec Section 15 and reject placeholder markers.

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3.6 -m unittest -v tests.test_check_prohibited tests.test_check_comments tests.test_documentation
```

Expected: checker modules and documents are absent.

- [ ] **Step 3: Implement scoped scans and complete documentation**

The prohibited scan examines `Makefile`, `include/`, `src/`, tests that compile into binaries, and a supplied verbose-build log; it excludes specifications, README, scanner source, and negative fixtures. The comment checker requires adjacent documentation blocks and reconciles stable declaration/kernel IDs with the checklist. README includes environment, sources/hashes, architecture/layout, formats, all commands/defaults, prohibited libraries, error handling, troubleshooting, and the explicit local-no-CUDA statement.

- [ ] **Step 4: Verify static gates**

```bash
python3.6 -m unittest -v tests.test_check_prohibited tests.test_check_comments tests.test_documentation
python3.6 scripts/check_comments.py --root . --checklist docs/comment-review-checklist.md
bash scripts/check_prohibited.sh source .
make V=1 2>&1 | tee build/verbose-build.log
bash scripts/check_prohibited.sh build build/verbose-build.log
```

Expected: all automated gates pass. The manual checklist remains an explicit human review gate and is checked only after reviewing each listed item.

- [ ] **Step 5: Commit**

```bash
git add Makefile README.md scripts/check_prohibited.sh scripts/check_comments.py tests/test_check_prohibited.py tests/test_check_comments.py tests/test_documentation.py docs/comment-review-checklist.md docs/h20-acceptance.md
git commit -m "docs: add compliance and H20 acceptance guidance"
```

### Task 14: H20 Acceptance and Evidence

**Files:**
- Create: `scripts/run_h20_acceptance.sh`
- Modify: `docs/h20-acceptance.md`
- Modify: `Makefile`

**Interfaces:**
- Consumes: all project binaries, checks, data, and documentation.
- Produces: timestamped acceptance log proving the criteria in spec Section 16.

- [ ] **Step 1: Write a fail-fast acceptance script**

Start with `set -euo pipefail`. Record UTC time and outputs from `nvcc --version`, `gcc --version`, `nvidia-smi`, and the executable’s runtime device report. Run Python compile/tests, clean verbose build, prohibited/comment checks, operator/workflow tests, Compute Sanitizer, data preparation, default training, threshold evaluation, and inference. Preserve pipeline exit codes.

- [ ] **Step 2: Run the complete H20 gate**

```bash
mkdir -p weights build/acceptance
bash scripts/run_h20_acceptance.sh 2>&1 | tee build/acceptance/h20-$(date -u +%Y%m%dT%H%M%SZ).log
```

Expected: every command exits zero and evaluation reports `accuracy>=0.99 status=pass`.

- [ ] **Step 3: Verify architecture and dependency artifacts**

```bash
cuobjdump --list-elf build/lenet_cuda
cuobjdump --dump-ptx build/lenet_cuda
ldd build/lenet_cuda
readelf -d build/lenet_cuda
compute-sanitizer --tool memcheck --error-exitcode=99 ./build/operator_tests
```

Expected: native `sm_90` and `compute_90` PTX are present; no prohibited dynamic dependency appears; sanitizer reports zero errors.

- [ ] **Step 4: Complete manual comment review and final source audit**

Review every checklist entry for shape/layout, formulas, races/synchronization, boundaries, numerical choices, ownership, and error behavior, then mark it `[x]`. Run:

```bash
python3.6 scripts/check_comments.py --root . --checklist docs/comment-review-checklist.md --require-reviewed
if grep -RniE 'resnet|residual block' include src tests README.md Makefile; then exit 1; fi
git status --short
```

Expected: comment review passes, the ResNet scan prints nothing, and only intended acceptance logs/data/weights remain ignored.

- [ ] **Step 5: Commit acceptance automation and reviewed checklist**

```bash
git add Makefile scripts/run_h20_acceptance.sh docs/h20-acceptance.md docs/comment-review-checklist.md
git commit -m "test: automate H20 acceptance evidence"
```

---

## Execution Boundaries

- Tasks 1-5 and the CPU portion of Task 6 can be implemented and verified without an NVIDIA GPU, preferably on the target Ubuntu/Python/GCC versions.
- Tasks 7-10 and CUDA portions of Tasks 6, 12, and 14 require the H20 host.
- Task 11 host tests can run locally; Task 12 workflow behavior is accepted only on H20.
- CUDA-unavailable local work must record tests as not run, never as passing.
- The first H20 pass may expose numerical or accuracy defects. Fix each defect with the systematic-debugging skill and add a regression test before changing production code.
