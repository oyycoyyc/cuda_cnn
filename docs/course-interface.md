# Course train/infer interface

This document specifies the two official executables built from the repository
root: `train` and `infer`. All paths are supplied by the caller; the programs do
not contain default data, label, weight, or model paths.

## Build targets

The default build produces both official executables and the inference runtime:

- `train` compiles the root `main.cu` with `COURSE_TRAIN` and links the course
  training workflow.
- `infer` compiles the same root `main.cu` with `COURSE_INFER` and links
  `course_runtime`, which depends only on host bundle I/O and the forward-only
  course model. It does not link the original trainable LeNet backend, loss,
  metrics, Adam, or the old train/CLI workflow.
- `course_runtime_tests` and the other course tests are optional and exist only
  when configured with `-DCOURSE_BUILD_TESTS=ON`.

Example from a clean build directory:

```sh
cmake -S . -B build -DCOURSE_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Raw data contract

Both programs consume headerless little-endian arrays:

- Images: exactly `N * C * H * W` FP32 (`binary32`) values, packed NCHW.
- Labels: exactly `N` int32 values for training only, each in `[0, 10)`.
- No headers, trailers, padding, index files, or compressed containers.
- `N` is a positive `uint32`. Training requires `N >= 2`; inference requires
  `N >= 1`.
- Only `C=1, H=28, W=28, K=10` are supported. A different geometry is a usage
  error.
- Image values must be finite. Mode-specific bounds are checked before any
  device work: unit `[0, 1]`, byte `[0, 255]`, normalized any finite value.

The two commands are exactly:

```text
train images.f32 labels.i32 N C H W K work_dir
infer images.f32 N C H W K
```

Inference takes no label path. Passing an extra argument is a usage error.

## Environment

| Variable | Used by | Contract |
| --- | --- | --- |
| `COURSE_INPUT_MODE` | train only | `unit`, `byte`, or `normalized`; default `unit`. Never read by inference. |
| `COURSE_EPOCHS` | train only | Integer `1..1000`; default `20`. |
| `COURSE_MODEL_DIR` | infer only | Required non-empty directory containing the published bundle. There is no fallback. |
| `COURSE_INFER_BATCH` | infer only | Integer `1..16384`; default `1024`. |
| `COURSE_PROFILE` | infer only | Exactly `1` enables profiling. Any other value, including an empty value, disables it. |

Training defaults are seed `1337`, batch size `128`, and epochs `20`. The
learning rate is `1e-3` for epochs 1–12, `1e-4` for 13–17, and `1e-5` from 18
onward. AdamW uses `(0.9, 0.999, 1e-8, 1e-4)` with no bias decay. The
training-only validation split is deterministic; no external test split or test
labels are read.

## Model bundle and persisted preprocessing

Training publishes two files in `work_dir`:

- `lenet.bin`: the unchanged checkpoint format holding the canonical ten LeNet
  FP32 tensors.
- `course.meta`: exactly four LF-terminated lines:

```text
course_lenet_v1
input_mode=unit|byte|normalized
shape=1,28,28,10
checkpoint_crc32=XXXXXXXX
```

`XXXXXXXX` is the eight-lowercase-digit IEEE CRC32 of the `lenet.bin` bytes
(same value as Python `zlib.crc32`). The loader validates the metadata spelling,
shape, checkpoint invariants, and CRC. A missing, malformed, mixed, or
CRC-mismatched bundle is a runtime error; there is no guessed recovery.

The persisted mode is the only preprocessing selection at inference. The
on-device transforms are:

- `unit`: `(x - 0.1307F) / 0.3081F`
- `byte`: `(x / 255.0F - 0.1307F) / 0.3081F`
- `normalized`: `x` unchanged, including negative values

No mode passes through a uint8 representation. The `unit` default is a
user-approved project default assumption, not an externally confirmed course
preprocessing requirement. Callers must select the mode that matches their
arrays and persist it during training. Inference does not guess from the input values and
does not perform online preprocessing verification beyond the documented mode
bounds.

## Output and exit status

Successful inference writes exactly `N` decimal class integers in `[0, 10)`,
one per line with a final newline, in input order, and nothing else on stdout.
All diagnostics, profile records, and errors go to stderr. There is no label
comparison, softmax output, or accuracy claim in inference.

- Exit `0`: success.
- Exit `2`: usage error (arguments, environment, shape, or batch bounds).
- Exit `1`: runtime error (unreadable/malformed model or data, CRC mismatch,
  CUDA failure, non-finite output, or output stream failure).

Inference loads and validates the model bundle and every FP32 input value before
the first runtime CUDA allocation or device selection, selects device 0, creates
one stream, one forward-only arena, batch-capacity input/normalized buffers,
all-N prediction storage, and one sticky flag once, initializes the sticky flag
once, enqueues all batches with predictions at global sample offsets, and
performs one final prediction/flag readback and stream synchronization. No
device allocation or device-to-host readback occurs inside the batch loop. All prediction lines are buffered and
published once only after the final validation succeeds; an argument, model,
input, CUDA, or non-finite-output failure writes no prediction byte. A
downstream sink I/O failure is reported, but bytes already accepted by that sink
cannot be rolled back.

## Profiling

`COURSE_PROFILE=1` allocates all timing events before the batch loop. Every
batch, including the first, contributes one ordered record:

```text
event=course_profile_batch index=I batch_samples=S h2d_ms=H prepare_ms=P forward_argmax_ms=F
```

and one aggregate record:

```text
event=course_profile profile_batches=B gpu_h2d_ms=H gpu_prepare_ms=P gpu_forward_argmax_ms=F host_load_model_ms=L host_load_images_ms=I host_cuda_setup_ms=S host_enqueue_batches_ms=E host_readback_sync_ms=R host_write_predictions_ms=W
```

The `gpu_*` values are CUDA event durations summed over every batch. The
`host_*` values are separate monotonic wall-clock buckets. In particular,
`host_enqueue_batches_ms` is host enqueue cost, not device execution time. The
fast path (`COURSE_PROFILE` unset or not exactly `1`) creates no timing events,
does no per-batch synchronization, and emits no profile records.

GPU event diagnostics describe internal device stages only. They are not a
substitute for full-process timing: an external measurement must include
process startup, argument/environment parsing, model and image loading, CUDA
setup, transfers, computation, validation, stdout/stderr, and process exit.
No warmup run is substituted for the measured request, and a full-process wall
time of `>= 100 ms` means the performance target failed even if an internal GPU
stage looks fast. Report full-process and internal timings separately.

## Operational notes

- Use arbitrary caller-owned absolute or relative paths. Do not point the
  programs at personal or hidden data/weight locations.
- `work_dir` may be empty or contain unrelated files; the training publisher
  preserves unrelated entries and only replaces a valid reserved pair.
- Do not overwrite original deployment data or pretrained weights. Choose a
  new model directory for each course run.
- Inference reads only the supplied images and the `COURSE_MODEL_DIR` bundle.
  It does not read labels, fallback weights, Python packages, or a test split.
- The provided tests are optional and Linux-oriented; the default submission
  build does not require the `course/tests` sources.

## Development-only final artifacts

After actual default20epoch training and five successful unprofiled full10000
test processes with at least96% accuracy, the development helper validates the
saved training completion CRC and the current measured binary/model/input hashes:

```sh
python3 scripts/course_package.py --benchmark final-benchmark.json \
  --training-log final-training.log --output course-submission.zip
```

It creates `evaluation.json`, `report`, `result.png` exclusively in the source
root and an explicit-whitelist ZIP; existing output files are never overwritten.
The PNG uses one accuracy/time group and the five-run full-process median. A
missed100ms target is labelled FAILED, not suppressed. Incomplete training,
low accuracy, stale hashes, profiled/inconsistent data or unsafe members fail
before publication. Pillow is required only in the development environment to
generate the PNG; no auto-install is performed, and neither Python nor Pillow
is included or required by the submitted CMake runtime. Model weights, datasets,
tests, helpers and private evidence are excluded from the ZIP. Actual final
metrics and known environmental limits are in `docs/course-interface-results.md`.
