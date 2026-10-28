# Course interface measurement helpers

> **Superseded as primary timing by the 2026-10-27 warm full-request
> amendment.** Preserved as historical cold evidence. Current warm final
> metrics and current root artifacts are in
> [course-warm-benchmark-results.md](course-warm-benchmark-results.md).

Status: all70 helper tests pass on Linux/Python3.12.12 with existing development
Pillow12.1.0 and no skips; the prior27 stdlib-only tests also passed with site
packages disabled.
The new GPU inference path has now been measured on all 10000 images using the
preserved **legacy** weights. Predictions match the original reference exactly;
the full-process 100 ms goal is **not achieved**. Actual default course training
and its full-test five-process acceptance are now complete (see below). Task5 independent review
has approved spec compliance and quality; these measurements are not a final
course-trained-model submission claim.

These helpers are not part of the submission runtime, are not built by CMake,
and do not implement neural-network math. They use only the Python standard
library.

## Fixture conversion

```sh
python3 scripts/course_fixture.py SOURCE.bin OUTPUT_DIR \
  --input-mode unit|byte|normalized
```

`SOURCE.bin` must be an original MNISTC1 v1 file:

- 24-byte header: `MNISTC1\0`, version 1, N, rows 28, columns 28.
- All `N * 784` image bytes followed by all N label bytes; the body is not
  interleaved.
- Exact size `24 + N * 785`; labels must be in `[0, 9]`.

A fresh output directory receives:

- `images.f32`: N x 1 x 28 x 28 little-endian binary32 values.
- `labels.i32`: N little-endian int32 labels.
- `manifest.json`: `schema=course-fixture-v1`, source path/hash, N, shape,
  input mode, and output file names.

Modes:

- `unit`: `f32(pixel / f32(255))` in `[0, 1]`.
- `byte`: exact `f32(pixel)` in `[0, 255]`.
- `normalized`: `f32((f32(unit - f32(0.1307))) / f32(0.3081))`, already in the
  persisted inference domain.

The converter streams bounded image/label chunks. It refuses an existing output
directory without changing it, reads the source read-only, and removes only its
own just-created artifacts if validation fails.

## External benchmark

Fast-path artifact:

```sh
python3 scripts/course_benchmark.py INFER IMAGES LABELS MODEL_DIR OUTPUT.json \
  --count N --batch 1024 --runs 5 --model-source legacy|course_train
```

Separate profiled diagnostic artifact:

```sh
python3 scripts/course_benchmark.py INFER IMAGES LABELS MODEL_DIR PROFILE.json \
  --count N --batch 1024 --runs 1 --model-source legacy|course_train --profile
```

Optional full prediction regression:

```sh
  --reference-predictions N_PREDICTIONS.u8
```

The reference file is exactly N uint8 predictions, never a labels file and
never an inference input.

Inference receives only its official six arguments and the model/batch
environment. Labels are read only by this external checker. Each process is
timed externally with `time.monotonic()` from launch through exit; stdout and
stderr are retained as UTF-8. Existing output JSON is never overwritten.

The JSON schema is `course-benchmark-v1`:

- top level: `schema`, `executable`, `images_path`, `labels_path`, `model_dir`,
  `model_source`, `n`, `batch`, `input_mode`, `profiled`, `runs_requested`,
  `sha256`, `runs`, `summary`;
- `sha256`: `binary`, `model`, `meta`, `images`, `labels`;
- each run: `wall_ms`, `returncode`, `correct`, `stdout`, `stderr`;
- summary: `median_ms`, `max_ms`, `correct`, `accuracy`, `accuracy_pass`,
  `all_wall_below_100ms`;
- optional reference fields: `reference_sha256`, `reference_mismatches`.

Accuracy and the full-wall flag are separate. The CLI fails for invalid
predictions, nonzero exits, accuracy below 0.96, or nonzero reference
mismatches; it does not fail solely because wall time is `>= 100 ms`.
`all_wall_below_100ms` reports that target honestly and independently.

### Profiling ruling

`--profile` does not add or hide a process. The requested `--runs R` are all
profiled diagnostic runs in their own `profiled=true` artifact, and every
stdout/stderr/timing sample is retained. Fast acceptance uses a separate
unprofiled artifact. The intended main-agent sequence is R=5 unprofiled, then
`--profile --runs 1` separately.

## Local test evidence

```text
python -m unittest -v tests.test_course_fixture tests.test_course_benchmark
Ran 27 tests ... OK
```

This local result uses Python 3.12 on Windows with mock subprocesses. It proves
the helper contracts only. It is not H20 evidence. `tests/test_course_benchmark.py`
mocks are harmless processes and do not represent a real CUDA model; actual
bundle validity is ultimately enforced by the compiled inference loader.

## Measured H20 legacy regression

All results below use the new compiled GPU `infer`, explicit persisted input
modes, and full10000 test images. The external checker owns the int32 labels;
`infer` receives only its six official arguments. The reference predictions
were extracted from the preserved original810000-byte snapshot after checking
its SHA256, not regenerated by another inference/warmup run.

The unit-mode trials each contain five fresh-process unprofiled runs. Times in
milliseconds cover subprocess startup through exit, including model/image
loading, CUDA setup, transfers, neural computation, validation, output and
cleanup. All runs returned0, predicted the same10000 classes, matched9911
labels (99.11%), and had0 reference mismatches. No time sample was discarded.

| Batch | All five full-process times / ms | Median / ms | Maximum / ms | All <100ms |
|---|---|---:|---:|---|
|128|1605.926,1507.313,1491.830,1522.712,1519.771|1519.771|1605.926|false|
|1024|1536.751,1493.476,1582.206,1555.140,1504.482|1536.751|1582.206|false|
|4096|1547.490,1564.988,1511.355,1479.635,1518.187|1518.187|1564.988|false|

Full10000 byte and normalized mode correctness checks each also achieved
9911/10000 with0 reference differences (one process each, not a five-run timing
acceptance series). The model bytes are identical; only the explicit metadata
mode and corresponding rawFP32 image representation differ.

Initialization/setup dominates and the small median differences do not justify
changing the approved default batch1024. No larger-batch tuning campaign was
run. These are current measurements, not a paired speedup claim against older
historical benchmarks under potentially different shared-machine load.

### Separate profiling diagnostic

One additional full10000 unit-mode process, batch1024, was explicitly profiled
and stored separately (`profiled=true`, runs1). Its full-process wall time was
1513.449ms. All10 batches, including the first, contributed to the GPU sums:

| GPU event stage | Sum / ms |
|---|---:|
|H2D|1.719|
|FP32 preparation|36.217|
|Forward + label-free argmax|78.942|

| Host wall bucket | ms |
|---|---:|
|Load model|4.083|
|Load images|29.407|
|CUDA setup/allocations/upload/events|1211.914|
|Host enqueue|91.452|
|Final readback/synchronization|25.505|
|Prediction formatting/write/flush|0.603|

Host enqueue can include pageable-copy staging and is not labeled GPU execution.
Host and GPU intervals overlap; they must not be added as independent work.
Host buckets are not exhaustive of process startup/cleanup. External full wall
is authoritative. In particular,78.942ms forward+argmax is **not** a successful
full-process100ms result. Setup timing alone does not establish how much is
native CUDA versus platform-library overhead; no preloads, drivers, clocks,
permissions or timing boundaries were changed.

### Identities and provenance

| Artifact | SHA256 |
|---|---|
|Actual infer binary|`c7c929225d6abfd5b23bae4aead58ac622100a6ede0362b25a1daaadd3ffae0e`|
|Legacy checkpoint|`89e1d79758ab50960ef27d24f51d3aca7d722452c84602f41e3f402d2ae92eb9`|
|Unit course.meta|`ca3461d59afaf460c61814058f6db41090a4de3cbde4cc2c70d2feaca2981159`|
|Unit test images.f32|`bc9f8f501f39b3ca61f899a2f402edf2024bde012b73223431b6bd735e47a165`|
|Test labels.i32, external checker only|`52aa149c856ca033d02e1d2e5cc61033b4c56ed4a56f0edc328206a178c42126`|
|Reference uint8 predictions|`ed3089f6c607def90561bf4ece8b372e3d7f437205b83ca7c5445b8c956a7df3`|

Converter manifests retain the original source hashes, counts and declared
modes. Real test fixtures were produced for unit/byte/normalized (31360000 image
bytes and40000 label bytes each); the60000-sample unit training fixture is
188160000 image bytes and240000 label bytes. Original data/weights were read
only. Per-run raw stdout/stderr, return codes, exact wall values and hashes are
retained in the private experiment benchmark JSON records.

The `unit` input interpretation remains the user-approved project assumption,
not independently confirmed official preprocessing. The leaderboard's timing
boundary remains unknown. This legacy99.11% result is not a claim that the
actual course-trained model meets96%; that separate acceptance is recorded below.

## Actual course-interface default training and final-model acceptance

The official `train images.f32 labels.i32 60000 1 28 28 10 work_dir`
was run in a fresh experiment model directory, with all inherited `COURSE_*`
variables removed. In particular, `COURSE_EPOCHS` was absent: the actual default
20 epochs, seed1337, batch128, original AdamW and learning-rate schedule were
used. Only the supplied training split was consumed (55000 training/5000
validation); the test split was reserved for external post-training evaluation.

A separate one-epoch resource pilot took34.028539s, peak host RSS353096KiB.
Its linear20epoch estimate was680.570789s. The actual independent default20
training completed successfully in594.740494s, peak host RSS369252KiB, no swaps,
and no stdout. All20 ordered epoch records and one final completion are saved:

```text
event=course_train status=pass epochs=20 best_epoch=15 validation_accuracy=0.992399991 input_mode=unit checkpoint_crc32=f473014f
```

Independent strict-best selection from the epoch records confirms epoch15;
checkpoint header/validation float bits, normalization, mode and recomputed
CRC agree with the completion and metadata. Actual retraining produced bytes
identical to the preserved legacy bundle with the same deterministic defaults.
That identity is an observed outcome, not a substitute for training: fresh
launch, full20epoch diagnostics, resource use and completion evidence are kept
separately. No fallback weights or hidden test data were used.

The **course-trained best model** was then measured on all10000 test images in
five fresh **unprofiled** processes, inference batch1024. Every process returned0
and gave9911 correct (99.11%), with identical predictions. The main controller
independently parsed saved raw stdout and counted against both the original
read-only test labels and the converted external-checker labels. `infer`
received no label argument. Current binary/model/meta/image/label SHA256 values
were rechecked after the series; identities match the table above.

| Run | Full-process wall / ms | Correct / 10000 |
|---|---:|---:|
|1|1684.153743|9911|
|2|1641.133869|9911|
|3|1659.597542|9911|
|4|1637.525734|9911|
|5|1657.407840|9911|

- **Accuracy target >=96%: PASS (99.11%).**
- **Full-process median:1657.407840ms; maximum:1684.153743ms.**
- **Full-process <100ms target: FAILED**, not reinterpreted as GPU-only time.
- In the preserved cold artifact, image `Test Time` uses1657.41ms, the median
  of these five actual course-model full-process runs, not a legacy series or
  internal stage. This is the historical cold value; the current warm final
  metric is in
  [course-warm-benchmark-results.md](course-warm-benchmark-results.md).

The timing boundary still includes startup through exit. No paired speedup or
regression claim is made from differences between separate shared-machine
measurement series. Official preprocessing and leaderboard timing remain
unconfirmed.

### Preserved cold submission artifacts and fresh extraction verification

At that cold-acceptance phase, the independently reviewed helper generated the
preserved cold root snapshot (`evaluation.json`, `report`, `result.png`, and
`course-submission.zip`) from the actual course-training log and five-run record
above. Those cold artifacts are preserved under
`.superpowers/sdd/2026-10-27-warm-full-request-benchmark/preserved-cold-artifacts/`
and are not the current root deliverables. The preserved cold image visibly
shows one99.11%/1657.41ms metric group and a red performance FAILED verdict,
with no competing timing number.

The preserved cold ZIP contains exactly34required runtime source/header files
plus3root artifacts; no model/data/tests/tools/Python/private evidence or
symlinks. It is119946bytes compressed and338343bytes unpacked. The900x420 cold
PNG is30620bytes. Cold archive SHA256:
`76da207fb76641745e10a8db6fb5e6063e51ade6b7e5db239d033f742eb977f8`.
After whole-branch review, one consolidated disclosure fix added the observed
five-run range1637.53–1684.15ms and startup-through-exit scope to the same
preserved cold image metric group, plus explicit timing stages and the
unconfirmed-official-preprocessing caveat to the cold report. Four new tests
first failed against the frozen old helper, then passed; all70helper tests pass
without skips. Old artifact versions were preserved. Runtime source members are
byte-identical between archive versions; no retraining, measurement replacement
or numerical change occurred.

A fresh remote extraction built default `train`/`infer` without optional
test/tool/Python files, with0warnings. Official bounded train->infer passed
with labels withheld from the inference path. The rebuilt infer also used the
actual default20-trained model on all10000 images from an images-only working
directory and reproduced the measured predictions exactly. This is functional
verification of the rebuilt binary, not a replacement timing series.

Fresh regressions passed: courseCTest48host/8GPU/9training/15runtime,
originaloperator suite, all12originalworkflow scenarios, nineoldhost suites,
198compliance tests, and Makefile/environment tests. Existing limitations
remain explicit: old `test_prepare_mnist` cannot import missing `pyarrow`
(data_interop passes), and the previous ComputeSanitizer self-disable is not a
valid sanitizer result. No dependencies were installed or platform protections
bypassed. At that phase, the scoped cold disclosure re-review and final backup
remained; no automatic upload/deployment is performed.
