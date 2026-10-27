# Course GPU Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide the course-compatible CUDA `train`/`infer` interface and a verified inference-only LeNet path, then measure accuracy >=96% and the complete-process <100ms objective without redefining either metric.

**Architecture:** Keep the existing application intact. Add host-side course contracts and model handoff, reuse the existing GPU training math, and implement a separate forward-only GPU arena using the same forward kernels. Course targets have their own CMake build/tests; development conversion, external accuracy checking and packaging do not become submission runtime dependencies.

**Tech Stack:** C++14, project-owned FP32 CUDA kernels, CUDA Runtime, CMake >=3.18, CTest; Python standard library for development validation/conversion, existing local Pillow only for result-image packaging.

**Spec:** `docs/superpowers/specs/2026-10-27-course-gpu-interface-design.md` (approved by user).

## Global Constraints

- GPU/CUDA inference is mandatory; no CPU neural-network fallback, new network, mixed precision, fast-math, third-party NN libraries, or driver/preload/quota changes in this iteration.
- Preserve LeNet 1x28x28 -> 6/16 convolutions -> 256/120/84/10 head, FP32 math, and smallest-index argmax ties. Do not edit existing numerical kernels.
- Train interface: `train images.f32 labels.i32 N C H W K work_dir`; infer: `infer images.f32 N C H W K`. N is positive uint32; only C=1,H=W=28,K=10 supported; training N>=2.
- Exact raw input sizes: N*C*H*W*4 image bytes, N*4 label bytes, no headers/trailers; int32 labels [0,10). Never quantize input floats through uint8.
- Approved default: unit floats [0,1] -> (x-0.1307F)/0.3081F on GPU. Explicit training `COURSE_INPUT_MODE=unit|byte|normalized`; persist the mode, never guess from test images or override it at inference.
- `COURSE_MODEL_DIR` is required for infer. No hidden test labels, fallback weights paths, host-specific paths, Python/runtime package installation, or partial prediction output.
- Successful infer stdout: exactly N decimal class integers, one per line. All diagnostics stderr. Usage errors exit2, runtime errors exit1, success exit0.
- Default training: seed1337, batch128, epochs20 (explicit `COURSE_EPOCHS` 1..1000), LR epochs1..12=1e-3,13..17=1e-4,18+=1e-5; AdamW=(0.9,0.999,1e-8,1e-4), no bias decay; train-only validation split.
- Default inference batch1024; explicit optional `COURSE_INFER_BATCH` integer1..16384 supports reproducible capacity tests. No device allocations/readbacks inside the inference batch loop.
- Full-process external time includes startup, loading, CUDA setup, transfers, computation, validation, stdout and exit. GPU event diagnostics are separate; full wall >=100ms means the performance target failed even if a GPU substage passes.
- Do not overwrite original deployment, data or pretrained weights. Work only in the existing isolated workspace and new remote experiment `/gemini/code/cuda_cnn/runtime-output/course-interface-20261027`; follow `docs/server-environment.md`.
- Backup exists at `backup/pre-100ms-20261027` / c4c9265; design draft backup e3a4050. Main agent makes explicit task checkpoint backups after review, not implementer commits; do not reset/discard/stage unrelated work or push/deploy/upload automatically.
- Main agent coordinates/reviews/runs H20; coding subagents use user-requested deepseek-flash. Tests-only handoffs wait for main-agent CUDA RED before production implementation. No local CUDA-success claims.

## Review Focus

1. Bad environment and metadata: unknown modes, missing model directory, mismatched checkpoint/metadata and interrupted publication must fail before inference; tests in Task1/4.
2. Binary boundaries: signed/overflowing dimensions, truncated/trailing payload, NaN/Inf and bad labels must fail safely without writing inputs; tests in Task1.
3. Repeated/tail GPU work: non-finite earlier-batch flags must survive later finite batches; N=1/17/1025 must preserve order, no allocation growth; tests in Task2/4.
4. Separation of roles: infer must work with no readable labels file, ignore training-mode overrides, and write no headers/partial output; tests in Task4/5.
5. Honest artifacts: stale binary/model hashes, mismatched training provenance, excluded archive members, or incomplete measurements must prevent final packaging; tests in Task6.

## File Structure and Dependency Decisions

- `course/interface.h`, `course/io.cpp`: CUDA-free command/data types, strict parsers, float images and int32-label adapter.
- `course/model_io.h/.cpp`: original checkpoint plus strict input-mode metadata, publication and integrity binding.
- `course/tools/model_bundle.cpp`: development-only `course_model_bundle CHECKPOINT OUTPUT_DIR MODE` utility, built only with course tests enabled, to validate/copy an existing checkpoint into a fresh explicit-mode test bundle; never linked into infer or packaged.
- `course/gpu_ops.h/.cu`: FP32 preparation/translation and label-free finite-checking argmax.
- `course/inference.h/.cu`: fixed-capacity inference-only arena and forward chain.
- `course/training.h/.cu`: course training workflow reusing existing LeNet/backward/AdamW; no old RunTrain dependency.
- `course/runtime.h/.cu`, root `main.cu`: infer orchestration, output buffering, diagnostics and macro-selected entry points.
- Root `CMakeLists.txt`: course libraries, train/infer and optional course tests. `COURSE_BUILD_TESTS=OFF` by default; ON explicitly registers nonempty CTest cases, including host tests.
- `course/tests/`: focused C++/CUDA tests and small shared test support (explicit executed-case counts); avoids existing `tests/*_tests` Make wildcards.
- `scripts/course_fixture.py`, `scripts/course_benchmark.py`, `scripts/course_package.py` and matching explicit `tests/test_course_*.py`: development-only helpers.
- `docs/course-interface.md`, `docs/course-interface-results.md`, `docs/evidence/course-interface-20261027/`: usage, measured results and curated evidence; root evaluation.json/report/result.png only after real final validation.
- Keep Makefile/source policy scripts unchanged: existing gates continue to cover the existing Make build. Course CMake tests/build/link/package audits are additional coverage, not falsely described as included in that old source inventory. New course headers/kernels still receive explicit API/ownership/math comments and review.
- Reusable common GPU sources: original activation/pooling/linear/convolution. Training additionally links original lenet/loss/metrics/adam. Infer links no old workflow or training model; optional test targets may link both reference/new models. Common host sources are original parameters/random/checkpoint, not old CLI/reporting/train/dataset workflows.

---

### Task 1: Strict host interface, data contracts and model handoff

**Status:** Complete; all steps verified,48 Linux cases pass, old code reproduces3 failures; reviewed checkpoint `backup/course-interface-task-1-20261027` / `aacad14`. Per-step historical boxes below are superseded by this status and the SDD ledger.

**Files:** Create `course/interface.h`, `course/io.cpp`, `course/model_io.h`, `course/model_io.cpp`, `course/tests/io_tests.cpp`, `course/tests/test_support.h`, `course/tools/model_bundle.cpp`.

**Interfaces (namespace `course`):**
- `enum class InputMode { kUnit, kByte, kNormalized };`
- `InputMode ParseInputMode(const char* value);` nullptr selects unit; empty/unknown names fail, accepted names exactly unit/byte/normalized.
- `struct Shape { std::uint32_t n,c,h,w,k; };`
- `struct TrainOptions { std::string images_path,labels_path,work_dir; Shape shape; InputMode input_mode; std::uint32_t epochs,batch_size; std::uint64_t seed; };`
- `struct InferOptions { std::string images_path,model_dir; Shape shape; std::uint32_t batch_size; bool profile; };`
- `TrainOptions ParseTrainArgs(int argc,const char* const* argv,const char* input_mode,const char* epochs);`
- `InferOptions ParseInferArgs(int argc,const char* const* argv,const char* model_dir,const char* batch,const char* profile);` null optional values select defaults, model_dir cannot be null/empty, profile enabled only exact"1".
- `std::vector<float> ReadImages(const std::string&,const Shape&,InputMode);`
- `std::vector<std::uint8_t> ReadLabels(const std::string&,const Shape&);`
- `struct CourseModel { Checkpoint checkpoint; InputMode input_mode; };`
- `CourseModel LoadCourseModel(const std::string& directory);`
- `void SaveCourseModel(const std::string& directory,const Checkpoint&,InputMode);`

- [ ] Write `io_tests.cpp` first. Assert exact argc9/7, defaults unit/20/128/1337/1024, empty/unknown env rejection, signed/zero/overflow N, dimensions not1/28/28/10, batch0/16385 and epochs0/1001 errors. Use tiny files to prove oversized declarations are rejected without huge allocation.
- [ ] Add file fixtures: ReadImages retains fractional floats exactly, accepts mode bounds, rejects short/long files and NaN/Inf; labels -1/10 rejected before narrowing; normalized finite negative values accepted. Verify no output mutation of input files/symlink aliases in owned temporary directories.
- [ ] Create an owned `build-course-host/` scratch directory, then run `g++ -std=c++14 -O2 -Iinclude -I. course/tests/io_tests.cpp course/io.cpp course/model_io.cpp src/checkpoint.cpp src/parameters.cpp src/random.cpp -o build-course-host/course_io_tests` and `./build-course-host/course_io_tests`; record missing declarations/source RED before implementation. This target does not require CUDA headers; POSIX path/publication checks run on Linux/H20 if unavailable on local Windows, without expanding scope into Windows/CUDA support.
- [ ] Implement the named interfaces. Validate supported scalar representation and checked extents before vectors; validate labels before uint8 conversion. Only ParseTrainArgs consumes training-mode/epoch values; infer parser has no labels or input-mode argument.
- [ ] Define model files `lenet.bin` and `course.meta`. Metadata exact four LF-terminated lines: `course_lenet_v1`, `input_mode=unit|byte|normalized`, `shape=1,28,28,10`, `checkpoint_crc32=XXXXXXXX` (8 lowercase hex). Reject extra/unknown/duplicate fields, invalid CRC and files >256 bytes. Use IEEE CRC32 reflected polynomial0xedb88320, initial/final XOR0xffffffff (same value as Python zlib.crc32). This is accidental-corruption/paired-publication protection, not a security signature.
- [ ] Test/save/load unit and normalized bundles; bind metadata CRC32 to actual checkpoint bytes and use same-directory temporary metadata+atomic rename published last. Refuse input-alias destinations and unrecognized existing files at the reserved lenet.bin/course.meta names, but preserve unrelated files/directories in the caller's work directory (the course does not guarantee an empty /work). A failed/mixed two-file publication must be rejected by loader or preserve the previous valid bundle. No claim of transaction isolation for concurrent training/readers.
- [ ] Implement the development bundle utility with argc4, explicit ParseInputMode, original LoadCheckpoint and SaveCourseModel; refuse nonempty output directory, never alter source checkpoint, and emit diagnostics only on stderr. Direct host compilation uses the same host source dependencies; Task2 registers its optional CMake target.
- [ ] Re-run host tests with nonzero executed-case count; record PASS, changed files and ownership/error contracts. Main reviews and makes a backup checkpoint; implementer does not commit.

### Task 2: GPU input/argmax helpers and forward-only LeNet arena

**Status:** Complete; full optional CMake build and48host/8GPU tests pass on H20, bitwise logits verified, atomic flag PTX inspected; checkpoint `backup/course-interface-task-2-20261027` / `17884b8`. Detailed evidence/remaining minor observations are in the SDD ledger.

**Files:** Create `course/gpu_ops.h/.cu`, `course/inference.h/.cu`, `course/tests/gpu_tests.cu`; create initial `CMakeLists.txt`, register Task1 host tests and new GPU tests.

**Consumes:** Task1 `InputMode`, original `ParameterSet`, original forward launchers and DeviceBuffer accounting.

**Produces (namespace `course`):**
- `void LaunchPrepareInput(const float* packed,const std::uint32_t* original_indices,float* normalized,int batch,InputMode mode,std::uint64_t seed,std::uint32_t epoch,bool augment,cudaStream_t stream);` packed/output disjoint, indices required only for augmentation, epoch>=1 when augmenting.
- `void LaunchArgmax(const float* logits,std::int32_t* predictions,int* bad_output,int batch,int classes,cudaStream_t stream);` sets sticky bad flag on any non-finite logit, no labels, smallest-index ties; caller initializes flag once before the full request.
- `class InferenceModel` with constructor `(int maximum_batch,const ParameterSet&,cudaStream_t)`, destructor, deleted copy, `const float* Forward(const float* normalized,int batch)`, and `std::size_t RequiredDeviceBytes() const`.

- [ ] Add failing GPU tests `float_input_modes_and_translation`, `label_free_argmax_ties_and_sticky_nonfinite`, `inference_logits_bitwise_match_reference`, `inference_tail_repeated_no_allocation`. Cover modes unit/byte/normalized, non-integer values, seed/index-consistent translation and normalized black padding. Tests compare GPU-computed results with known scalar fixtures or reference model, not a production CPU fallback.
- [ ] Main compiles/runs tests-only on H20; capture expected missing new functions/model RED. For direct pre-CMake testing use original NVCC14/O2/sm90 flags and explicit same-toolchain reference objects, without changing production source.
- [ ] Implement preparation with no uint8 roundtrip and original TranslationOffsetForSample. GPU argmax writes int32 and checks all logits in every row; zero the bad flag only at request setup, never inside argmax.
- [ ] Implement one checked/aligned model arena holding only ten parameter payloads and forward activations/winners. Copy validated supplied weights directly; no random initialization, gradient/moment allocation, or whole-arena zeroing. Preserve original forward arithmetic/reduction order; reuse storage only when original launcher alias rules permit it.
- [ ] Test batches1,17,1024 against original LeNet on identical pre-normalized FP32 buffers/weights: memcmp logits and exact predictions. Test capacity bounds, null input, poisoned weights, repeat/tail work, allocation event count unchanged around Forward, and strictly smaller arena than original LeNet at matching capacity.
- [ ] Add CMake >=3.18 CXX/CUDA project: use CXX/CUDA_STANDARD14 REQUIRED, default architectures90 only if caller has not set CMAKE_CUDA_ARCHITECTURES, optimization O2, no fast-math. Common host/forward libraries have no old workflow dependency; reference training backend is separate. Optional tests ON links existing reference LeNet and needed training kernels and builds Task1's course_model_bundle utility; none of those optional sources are required by the default submission build.
- [ ] Run `cmake -S . -B build-course -DCOURSE_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=90`, build `course_io_tests course_gpu_tests`, then `ctest --test-dir build-course -R 'course_(io|gpu)' --output-on-failure`. Require named nonempty test counts, full PASS and warning review; main backs up reviewed task.

### Task 3: Course-only GPU training and persisted preprocessing

**Status:** Complete;48host/8GPU/9training cases pass on H20, including bounded2epoch execution and best-bundle consistency; reviewed checkpoint `backup/course-interface-task-3-20261027` / `a8e0bc3`. Default20epoch/full-test acceptance remains Task6.

**Files:** Create `course/training.h/.cu`, `course/tests/training_tests.cu`; extend CMake with training library/test only.

**Consumes:** Task1 TrainOptions/readers/SaveCourseModel; Task2 preparation/argmax; original LeNet and GPU loss/AdamW; original MakeTrainValidationSplit/ShuffledTrainingIndices.

**Produces:** `int course::RunCourseTrain(const TrainOptions&,std::ostream& diagnostics);` returns0 only after a validated saved best bundle; exceptions are mapped by later entry point.

- [ ] Write failing tests `train_saves_readable_model_without_test_split_files`, `float_training_keeps_fractional_pixels`, `validation_uses_training_only_indices`, `training_hyperparameters_and_schedule`. Tiny N>=2 fixtures live in owned experiment directories, no real mounted output probes.
- [ ] Main verifies tests-only CUDA RED before implementation. Tests must fail for missing training workflow, not absence of a hidden test dataset.
- [ ] Implement training over supplied FP32 images and validated labels only. Allocate reusable packed float/label/index buffers and original LeNet once; GPU preparation/augmentation precedes each Forward. Retain original finite checks after forward/backward/step, checked CUDA boundaries, weighted loss and monotonic global optimizer step.
- [ ] For N60000 split55000/5000; otherwise validation=max(1,N/5). Use original deterministic split/shuffle functions with seed1337. Apply exactly the pinned schedule/AdamW, strictly-improving validation selection, save best weights+mode, reload/validate saved bundle before success. Never call old RunTrain or access an external test set.
- [ ] All logs stderr; completion record `event=course_train status=pass epochs=E best_epoch=B validation_accuracy=A input_mode=M checkpoint_crc32=H` uses classic-locale numbers. No test accuracy claim during training. Default epochs20; bounded tests pass explicit options/COURSE_EPOCHS=1 or a small fixture count and restore environment.
- [ ] Verify one-epoch tiny training changes weights from seeded initialization, bundle roundtrips and remains usable by reference/new forward; LR transition epochs12/13/17/18 and best-selection logic get deterministic tests without20epoch fixture runs. Register CTest `course_training` with explicit executed counts.
- [ ] Main runs new training tests plus existing lenet/operator regressions, reviews for hidden-label access and parameter drift, then backs up the task.

### Task 4: Official executables, buffered predictions and honest profiling

**Files:** Create root `main.cu`, `course/runtime.h/.cu`, `course/tests/runtime_tests.cu`; extend CMake with train/infer and runtime tests; create `docs/course-interface.md`.

**Consumes:** Tasks1–3 exact interfaces.

**Produces:** `int course::RunCourseInfer(const InferOptions&,std::ostream& predictions,std::ostream& diagnostics);` and the two official executables.

- [ ] Add subprocess tests for exact official argv, missing COURSE_MODEL_DIR, malformed/bad CRC models, ignored COURSE_INPUT_MODE at inference, all three input modes, N1/17/1025, no labels present, stdout-only integers in [0,10), stderr diagnostics, and zero stdout on argument/model/non-finite-output failure. Use a fixture whose poisoned early batch is followed by finite batches to catch cleared sticky flags.
- [ ] Main runs tests-only RED against missing train/infer/runtime before implementation; interface absence must be the named failure, not hardware access.
- [ ] Implement macro-selected root main with exclusive COURSE_TRAIN/COURSE_INFER compilation, positional/env parsing, dispatch and exceptions -> exit1/2. Define CMake targets `train` and `infer`; infer does not link old src/train.cu/src/main.cu or original training LeNet backend. Do not change existing Make/CLI behavior.
- [ ] Infer loads/validates model and FP32 input before CUDA, sets device0 without stdout device banners, creates stream, inference arena and input/normalized/output/error buffers once. Initialize sticky error once; enqueue all batches and int32 predictions at global offsets; one final readback of predictions/flag and sync. Only after complete successful validation format all N lines and write once; no labels/softmax/accuracy kernels.
- [ ] `COURSE_PROFILE=1` allocates all timing events before batching. Per batch record upload start/end, normalized-input ready, and forward+argmax complete; after final sync sum GPU H2D/prepare/forward+argmax durations over every batch, including the first. Also report host load_model/load_images/cuda_setup/enqueue_batches/readback_sync/write_predictions wall buckets. Label host enqueue time as enqueue, not GPU execution; default fast path creates no timing events or per-batch synchronization.
- [ ] Document exact env/defaults/mode metadata, fixed shapes and no online preprocessing verification. Document full process vs internal GPU timings, no warmup substitution, and usage with arbitrary caller-provided paths (no personal absolute paths).
- [ ] Build default train/infer and tests from clean CMake directory; run CTest all four suites plus explicit subprocess output assertions. Inspect link commands for forbidden dependencies and training-workflow leakage into infer; main reviews/backups before benchmark helpers.

### Task 5: Reproducible development fixtures, full-test correctness and timing

**Files:** Create `scripts/course_fixture.py`, `scripts/course_benchmark.py`, `tests/test_course_fixture.py`, `tests/test_course_benchmark.py`, `docs/course-interface-results.md`.

**Interfaces:**
- `python3 scripts/course_fixture.py MNISTC1 OUTPUT_DIR --input-mode unit|byte|normalized`: exclusive fresh output directory, produces images.f32/labels.i32 plus manifest.json with exact N/shape/mode/source SHA256; validate original24-byte MNISTC1 v1 format. Python stdlib only, native values written little-endian 32-bit, stream chunks rather than Python lists of all floats.
- `python3 scripts/course_benchmark.py INFER IMAGES LABELS MODEL_DIR OUTPUT_JSON --count N --batch B --runs R --model-source legacy|course_train [--profile] [--reference-predictions PATH]`: external checker owns labels; infer gets only its official six arguments. Default R5,B1024; profile is a separate diagnostic run, not a fast-path acceptance replacement. Optional reference file is exactly N uint8 predictions from the preserved baseline snapshot, not a labels file passed to infer.

- [ ] Write failing stdlib unit tests for converter pixel endpoints/fractions/modes, header/size/label failures, existing-output/alias protection and source immutability. Write benchmark tests using harmless mock executables that reject receiving a labels path, emit wrong/extra/partial/non-integer outputs or exit nonzero; test stale output overwrite protection and all slow samples retained.
- [ ] Run `python3 -m unittest -v tests.test_course_fixture tests.test_course_benchmark` and capture missing-module/function RED; implement helpers minimally and rerun to PASS. No Pillow or numpy in these helpers.
- [ ] Benchmark validates exact N predictions/ranges and computes accuracy externally from int32 labels, times full subprocess startup through exit with monotonic clock, clears COURSE_PROFILE unless requested, sets model-dir/batch only, retains stdout/stderr/exit per run and all times. Record binary/model/meta/input SHA256, model source, count, mode, batch, complete timing scope, median/max, accuracy gate and `all_wall_below_100ms` separately; never pass labels into infer.
- [ ] Pin benchmark JSON schema `course-benchmark-v1`: top-level `schema, executable, images_path, labels_path, model_dir, model_source, n, batch, input_mode, profiled, runs_requested, sha256, runs, summary`; sha256 has binary/model/meta/images/labels; each run has wall_ms/returncode/correct/stdout/stderr; summary has median_ms/max_ms/correct/accuracy/accuracy_pass/all_wall_below_100ms. Successful repeated runs must produce identical predictions; preserve failures without falsely setting a pass summary. Optional reference records reference_sha256/reference_mismatches, requiring0 before claiming legacy regression pass. Paths remain in private evidence and are sanitized out of submitted report text.
- [ ] Main creates fresh remote experiment, records protected-file hashes and source manifest, converts read-only old MNIST splits into runtime-output fixtures; no changes to actual mounts. Original checkpoint becomes a clearly marked legacy diagnostic bundle using course_model_bundle, not by shipping fallback weights. Extract the final10000 uint8 predictions from the preserved810000-byte baseline snapshot into an owned reference file for full prediction regression, never into the inference input or model.
- [ ] With legacy bundle, verify all10000 predictions at batch128/1024/4096 against original reference9911, plus exact raw normalized-buffer Task2 tests. Run five fresh-process fast-path measurements for each feasible batch, one separate profiled run for chosen batch; stop expanding batch trials if initialization plainly dominates. Do not repeat the original36 startup runs.
- [ ] Save true accuracy, correctness diffs, current source/binary/model identities and all stage/whole-process results. If full100ms fails, say so even when GPU substage passes; no false completion claim. Main reviews helper tests, results and unchanged protected hashes, then checkpoints.

### Task 6: Default training acceptance, compliant artifacts and final review

**Files:** Create `scripts/course_package.py`, `tests/test_course_package.py`, root `evaluation.json` after tests; produce root `report`/`result.png` and fresh archive only after real acceptance. Update course usage/results and curate `docs/evidence/course-interface-20261027/`.

**Consumes:** Official targets, validated model bundle, final benchmark JSON and saved course training-completion log. Existing local Pillow is development-only; packaged CMake/runtime must need no Python.

- [ ] Write failing package tests using synthetic owned fixtures: exact JSON, no early final image without training/measurement provenance, binary/model/meta hash mismatch rejected, invalid/missing5-run data rejected, outside-root/symlink/archive member rejection, no overwrite of existing archive. Tests prove wall>100ms results are labelled failed, not rewritten or suppressed.
- [ ] Main runs RED, then implement `python3 scripts/course_package.py --benchmark JSON --training-log LOG --output ZIP`. Require `model_source=course_train`, matching completion CRC/current model+meta/binary hashes and five real successful unprofiled full-test results. Permit a truthful missed100ms result, never pretend performance success; if training incomplete or accuracy<96%, do not produce a final-success package/image.
- [ ] Renderer uses existing development Pillow (clear error if unavailable, no auto-install), creates PNG within4MB/16million pixels, one metric group `Test Accuracy: ...` and `Test Time: ... ms`, with full-process/median-of5 scope. Report lists every run, max, configuration/provenance, preprocessing assumption and unresolved leaderboard timing. No multiple conflicting Test Time values in the image.
- [ ] Explicit source whitelist: CMakeLists.txt, main.cu, necessary course headers/sources (no course/tests or course/tools), required original include/ and original source dependencies, evaluation.json/report/result.png. Exclude data/,build/,weights/,diagnostics/,runtime-output/,Python tools,Dockerfile,symlinks,credentials and personal absolute paths. Validate relative members and compressed<100MB/unpacked<300MB before exclusive archive creation.
- [ ] Main first runs one default-sized training epoch as a resource pilot in a fresh model directory and records extrapolated duration; then performs default20epoch training with only supplied training split. Keep failure/output logs; if resource/session budget prevents completion, mark training/packaging incomplete and stop rather than fabricate results or silently shorten defaults.
- [ ] Main benchmarks the actual course-trained best model on all10000 samples five fresh processes, verifies >=9600 correct and reports full-wall target independently. Run official train->infer contract with no test labels mounted to infer; old pretrained bundle accuracy does not substitute for this model.
- [ ] Run old host/operator/workflow/compliance checks and all course CTests/helper tests. Record old missing-pyarrow and sanitizer restrictions honestly; no dependency installation or tool bypass. Regenerate complete actual CMake build/link evidence and source/protected-file hashes, not just an incremental log missing key targets.
- [ ] Only after actual model validation, generate truthful final image/report/archive, extract to fresh experiment directory, build `train`/`infer` without helper/test/Python files and run bounded train->infer roundtrip plus valid-model infer. Inspect artifact contents and hashes; no automatic platform upload.
- [ ] Fresh whole-branch review of all tasks, main-agent final verification, curated evidence and explicit achieved/not-achieved outcome. Main creates final backup ref without overwriting earlier backup branches or unrelated work; leave original deployment and data intact.

## Plan Self-Review (main agent)

- Spec coverage: contracts/preprocessing/model handoff -> Task1; GPU-only/new arena/math -> Task2; training-only labels/default20epoch -> Task3/6; official output/timing -> Task4; fixtures/full-test accuracy/latency -> Task5/6; submission limits/truthful artifacts -> Task6.
- Review Focus five failure classes each have named tests above; no known unowned class remains.
- Cross-task types all live in namespace course; TrainOptions/InferOptions/mode/shape established in Task1, device helper/model API in Task2, both workflow entry points in Task3/4. Scripts use the same fixed metadata/mode/count contracts.
- Old Make exact-source scan remains meaningful for old targets; course tests are deliberately outside its wildcard and have explicit CMake/CTest coverage. No claim that the old scanner alone audits new CMake code.
- Checkpoint+metadata binding is a local publication/integrity contract, not cryptographic authorization or fully concurrent transactional storage. Each task can be tested before downstream integration.
- Design and this plan are now explicitly approved by the user; implementation proceeds via the preserved subagent/main-agent division. Live task evidence is in `.superpowers/sdd/2026-10-27-course-gpu-interface/progress.md`.

## Execution Handoff

Preserved method: subagent-driven, deepseek-flash implementers, main-agent planning/review/H20 verification. No need to select the method again. After user reviews this plan, execute tasks sequentially with tests-first evidence, task reviews and task checkpoint backups; do not pause between accepted tasks except for a real blocker or a scope change.
