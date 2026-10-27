# Task 4 implementer report: tests/declarations handoff

Status: **F1 buffered-sink fix and 15th direct-API test applied**; awaiting
main-agent independent RED against the frozen old runtime archive, then H20
GREEN of all four suites plus CLI stdout/stderr sink probes. The original
tests-first/GREEN/offset records are retained below; the final sections record
the F1 test and minimal flush correction.
No Task 1/2/3 source or header was changed. No commit, staging, install, SSH, or
GPU run was performed locally.

## Files created

- `course/runtime.h` — exact declaration of the approved inference runtime API.
- `course/tests/runtime_tests.cu` — 15 real-behavior cases, one TU, reusing
  `course/tests/test_support.h` for its single `main`, nonzero-executed rule,
  failure counting, and exclusive scratch root.
- `course/runtime.cu`, root `main.cu`, `CMakeLists.txt` changes, and
  `docs/course-interface.md` — GREEN production implementation (see follow-up).
- `.superpowers/sdd/2026-10-27-course-gpu-interface/task-4-report.md` — this
  report.

No other file was touched. `git status` still lists only the pre-existing
modified/untracked work from Tasks 1–3 plus these two new course files.

## Exact declared API

```cpp
namespace course {
int RunCourseInfer(const InferOptions& options, std::ostream& predictions,
                   std::ostream& diagnostics);
}
```

`course/runtime.h` documents the binding observable contract:

- Required options: non-empty image/model paths; N >= 1; C/H/W/K exactly
  1/28/28/10; batch 1..16384; model bundle valid; raw images satisfy the
  **persisted** input mode; no labels file.
- `std::invalid_argument` for usage/shape/capacity violations, including empty
  model directory or out-of-range direct batch size.
- `std::runtime_error` for missing/malformed model, CRC mismatch, unreadable or
  out-of-mode images, CUDA failures, and non-finite GPU output.
- Model and all FP32 host input validated before the first runtime CUDA
  allocation/device selection. Device 0 is selected without stdout banners.
- Successful stdout is exactly N decimal digits in `[0,10)`, one per line,
  final newline, input order, published only after a successful complete
  readback and non-finite-flag check. Failures write no prediction byte.
- Diagnostics are stderr-only. Successful final record:
  `event=course_infer status=pass samples=N batches=B input_mode=M`.
- When `options.profile` is true, one ordered
  `event=course_profile_batch index=I batch_samples=S h2d_ms=H prepare_ms=P
  forward_argmax_ms=F` record per batch (1-based, including the first), plus
  `event=course_profile profile_batches=B gpu_h2d_ms=... gpu_prepare_ms=...
  gpu_forward_argmax_ms=... host_load_model_ms=... host_load_images_ms=...
  host_cuda_setup_ms=... host_enqueue_batches_ms=... host_readback_sync_ms=...
  host_write_predictions_ms=...`.
- Fast path (`profile == false`) emits neither profile record and must create no
  timing events or per-batch synchronizations. Enqueue is named a host bucket,
  never device execution.

The per-batch profile record is intentional: it is the only black-box way for
the suite to prove that every batch, including the first, participates. The
test also sums the per-batch GPU durations and compares them with the aggregate
record (tolerance 0.01 ms), so omitting the first batch fails both the index
count and the sum check.

## Test coverage map (13 cases)

| Requirement | Case(s) |
| --- | --- |
| Exact official train argv, bounded 1 epoch, own work dir | `train_cli_bounded_epoch_persists_normalized_mode_for_infer` |
| Persisted mode survives train → infer; `COURSE_INPUT_MODE` ignored | same case, plus `infer_uses_persisted_mode_and_ignores_course_input_mode` |
| Train usage argc/N errors exit 2, stdout empty | `train_cli_usage_errors_exit_two_without_stdout` |
| Exact infer argv, missing/empty `COURSE_MODEL_DIR`, extra labels arg | `infer_requires_exact_argv_and_course_model_dir` |
| Missing, malformed, bad-CRC, corrupt-checkpoint models: exit 1, zero stdout | `infer_missing_malformed_and_bad_crc_models_write_no_stdout` |
| All three persisted modes unit/byte/normalized | `infer_uses_persisted_mode_and_ignores_course_input_mode` |
| N=1/17/1025, canonical pure stdout, order, batch 17 invariance | `infer_n1_17_1025_pure_stdout_preserves_input_order` |
| No labels present; labels argument rejected | `infer_needs_no_labels_and_rejects_an_extra_labels_argument` |
| Batch 1..16384 accepted, 0/16385/-1/abc rejected usage-2 | `infer_batch_option_accepts_only_one_through_16384` |
| Profile includes every batch including first; host buckets; no GPU enqueue | `infer_profile_reports_every_batch_including_the_first` |
| Default fast path has no profile records or `_ms=` keys | `infer_default_fast_path_creates_no_profile_records` |
| Valid poison model: early non-finite, later finite, sticky retention | `infer_poisoned_early_batch_fails_while_later_batch_stays_finite` |
| Direct API buffering, success, exception taxonomy | `run_course_infer_direct_buffers_output_and_rejects_bad_options` |
| Direct API poison leaves predictions empty, later finite succeeds | `run_course_infer_direct_keeps_poison_failure_out_of_predictions` |

Every assertion is against a real subprocess or real direct API behavior. There
are no source-text scans, mocks, fake kernels, or test-only production hooks.

### Subprocess mechanics

- The test binary resolves itself with `/proc/self/exe`, takes its directory,
  and requires `train` and `infer` as siblings. This avoids embedded personal
  paths and avoids a test-only environment override.
- A missing sibling fails immediately with
  `missing official entry point 'train' at build-adjacent path ...` before any
  direct runtime call, so the pre-implementation RED is a named absence, not a
  CUDA/hardware failure.
- Children are `fork`/`execve`d with an explicitly built environment. All
  inherited `COURSE_*` variables are removed first; only the case's exact
  overrides are added. The parent environment is never mutated (no `setenv`),
  so "restore env" is structural.
- Stdout and stderr are captured through pipes with `poll`; child exit codes are
  decoded from `waitpid`. Cases assert exact 0/1/2, empty stdout on failures,
  non-empty stderr for failures, and canonical single-digit prediction lines.
- Fixtures are absolute paths below the run's exclusive scratch root; each
  subprocess also runs with a scratch working directory. No `/train`, `/test`,
  mounted dataset, or pretrained path is used.

### Poison fixture validity

- Model: canonical 10-tensor checkpoint, all weights finite `1.0F`, all biases
  `0.0F`, metadata mode `normalized`, published through the real
  `SaveCourseModel`; the test reloads it and checks every parameter is finite.
- Input: first image all `FLT_MAX` (finite, accepted by `ReadImages` in
  normalized mode), second image all `0.0F`. The pair, first-only, and
  second-only files are real raw FP32 files; `ReadImages` is used to prove the
  host input itself is valid and bit-preserved.
- The pair with batch 1 fails exit 1 with empty stdout; first-only fails;
  second-only succeeds with one valid class. This proves the early-batch
  overflow and the later-finite property independently. A per-batch flag clear
  would make the pair succeed and fail the test.
- No model corruption, host-input rejection, CPU NN math, or fake kernel is
  used to manufacture the failure.

## How main should compile the test TU (corrected placement)

`course_runtime`, `train`, and `infer` belong **outside** the optional
`if(COURSE_BUILD_TESTS)` block. Only the test target and registration belong
inside it. The actually applied CMake follows this corrected placement:

```cmake
# Unconditional production targets.
add_library(course_runtime STATIC course/runtime.cu)
target_link_libraries(course_runtime PUBLIC course_host course_forward)
course_enable_host_warnings(course_runtime)

add_executable(train main.cu)
target_compile_definitions(train PRIVATE COURSE_TRAIN)
target_link_libraries(train PRIVATE course_training)

add_executable(infer main.cu)
target_compile_definitions(infer PRIVATE COURSE_INFER)
target_link_libraries(infer PRIVATE course_runtime)

# Inside if(COURSE_BUILD_TESTS) only.
add_executable(course_runtime_tests course/tests/runtime_tests.cu)
target_link_libraries(course_runtime_tests PRIVATE course_runtime course_host)
course_enable_host_warnings(course_runtime_tests)

add_test(NAME course_runtime COMMAND course_runtime_tests)
set_tests_properties(course_runtime PROPERTIES
  PASS_REGULAR_EXPRESSION "executed=15 failures=0 status=pass")
```

`course_runtime_tests` needs `course_host` only for real bundle/checkpoint
fixtures. It must not link `course_training`. Keep all three executables in the
same CMake runtime output directory (the default build root); the sibling
lookup is the acceptance path contract. Do not set a per-test working directory
or binary-directory environment variable instead.

The `.cu` extension is fine for nvcc; the test TU contains no device code or
CUDA headers. `test_support.h` supplies the only `main`, and its runner requires
`executed > 0`, so a filter selecting nothing is a failure.

## RED verification expected from main

1. After adding only the declaration/tests, build the test target:
   `cmake --build <build> --target course_runtime_tests`
   Expected: link failure naming `course::RunCourseInfer(...)` because
   `course/runtime.cu` does not exist yet. This is the named runtime RED.
2. After adding `runtime.cu` but before building `train`/`infer`, run:
   `<build>/course_runtime_tests --filter train_cli_usage`
   Expected: `missing official entry point 'train' at build-adjacent path ...`;
   no CUDA context or device access. The same check works with
   `--filter infer_requires`.
3. Only after both named REDs are recorded should main run the H20 GREEN.

## GREEN acceptance and link audit

- Configure a clean CMake directory with `-DCOURSE_BUILD_TESTS=ON`, build the
  default `train`/`infer`, then run `ctest` for all four suites:
  `course_io_tests`, `course_gpu_tests`, `course_training`, `course_runtime`.
- Run the explicit subprocess assertions already embedded in
  `course_runtime_tests`; they inspect captured stdout/stderr/exit for every
  required case.
- Link audit for infer separation:
  `cmake --build <build> --target infer --verbose > infer-link.txt`, then
  confirm the link line contains no `src/train.cu`, `src/main.cu`,
  `course/training.cu`, `src/lenet.cu`, or
  `src/kernels/{input,loss,metrics,adam}.cu`/archive.
  `nm -C <build>/infer` must not expose `RunCourseTrain`, `LeNet::Forward`
  training symbols, loss/metrics, or Adam symbols.
- The default all-target build compiling training sources is not evidence that
  `infer` links them; the audit must use the infer link command/binary.
- Predictions are checked structurally (count, canonical digits, order, batch
  invariance, zero stdout on failure). No CPU NN math is used. If main later
  wants exact class goldens, derive them only from the reviewed reference GPU
  on the identical inputs and add them as separate reviewed fixtures.

## Local static checks (not CUDA/Linux claims)

- No local CUDA/nvcc build or test execution was possible; no GPU success is
  claimed.
- Host-only syntax/type check of `runtime_tests.cu` passed with MSYS2
  `g++ -std=c++14 -Wall -Wextra -Wpedantic -fsyntax-only` against a temporary
  POSIX declaration shim (the local host is Windows, while the deployment
  target is Linux). This validates C++ syntax/types only, not POSIX runtime
  behavior or CUDA.
- Personal-path scan over the two new files found no `C:\`, `/home/`, user name,
  worktree name, or `/gemini` references.
- `course/runtime.h` includes only `course/interface.h` and `<ostream>`; no
  CUDA or training-header leakage.

## Decisions/risks for main to confirm

1. **Diagnostics record spellings** are now a contract: `samples=`, `batches=`,
   `input_mode=`, `event=course_profile`, `profile_batches=`, per-batch
   `index=`/`batch_samples=`/`h2d_ms=`/`prepare_ms=`/`forward_argmax_ms=`, and
   the six `host_*_ms=` buckets. Implement them exactly or revise tests before
   GREEN.
2. **Per-batch profile lines** were added solely to make first-batch inclusion
   observable. They must be formatted after final sync, not printed inside the
   batch loop.
3. Tests require training stdout empty; global constraints put all diagnostics
   on stderr. Root main should pass `std::cerr` (or equivalent) for train too.
4. Direct `RunCourseInfer` must reject batch sizes outside 1..16384 even though
   `ParseInferArgs` already does for the CLI.
5. The suite is Linux-only by design (`fork`, `poll`, `/proc/self/exe`) and must
   stay under `COURSE_BUILD_TESTS`; the default submission build does not
   include `course/tests`.

## GREEN follow-up after main H20 RED

Main verified the tests-first RED: the TU compiled and the linker failed only
on the missing `course::RunCourseInfer` symbol
(`task-4-red-compile.txt`, `task-4-red-link.txt`). The compile log also exposed
ignored `::write` results. The following objective test defects were corrected
before production code.

### Test corrections

1. The happy training argv now carries all eight positional fields:
   `images labels 4 1 28 28 10 work_dir` (`K=10` is present). The usage cases
   now exercise their intended shapes: argc 8 has images/labels/N/C/H/W/K and
   lacks only `work_dir`; the N=1 case has a complete 9-argument command; the
   extra-argument case has a complete command plus one trailing argument
   (argc 10). None can pass because of an earlier malformed argc/K slot.
2. The N=1 request writes and uses a separate one-image file containing exactly
   the first 784 fixture values. N=17 and N=1025 still use exact-size files, and
   the test now asserts each raw file byte size is exactly
   `samples * 784 * sizeof(float)` before invoking the executable.
3. `PatternImages(kByte)` now uses `(index * 37 + 13) % 2551 / 10.0F`, whose
   inclusive range is `[0, 255.0]`. `ExpectPatternBounds` pins unit/byte bounds
   and fractional-value coverage, and normalized bounds plus negative coverage,
   before the mode fixtures are published.
4. The per-mode case directory (`root/unit`, `root/byte`, `root/normalized`) is
   created before `PublishModel` creates its single `model` child; no temporary
   fixture was moved into a pre-existing unrelated tree.
5. Child diagnostics now use `WriteChildDiagnostic`, which loops over `::write`,
   consumes the returned count, and retries short writes. It uses only
   async-signal-safe calls (no allocation, no formatting, no non-POSIX child
   code) and removes the NVCC/GCC ignored-result warnings without suppression.
6. `runtime.h` no longer claims every possible failure leaves an untouched
   stream. It now narrows the guarantee to argument/model/input/CUDA/non-finite
   failures before publication, while documenting that a downstream sink I/O
   failure is reported but cannot roll back bytes already accepted. The single
   buffered publication behavior is unchanged.

### Production implementation

- `course/runtime.cu` implements the declared `RunCourseInfer`:
  - validates direct options (`N>=1`, fixed geometry, batch 1..16384, non-empty
    paths) before any file/CUDA work;
  - loads and validates the model bundle, then reads all FP32 images in the
    persisted metadata mode, before the first runtime CUDA call;
  - explicitly selects device 0, creates one non-blocking stream, one
    forward-only `InferenceModel`, and the fixed packed/normalized/prediction/
    sticky-flag buffers once;
  - zeroes the sticky flag exactly once, then enqueues every batch (per-batch
    H2D slice, GPU prepare, forward, int32 argmax at the global sample offset)
    with no device allocation, readback, or synchronization in the loop;
  - performs one final predictions+flag device-to-host readback followed by one
    stream synchronization, rejects non-finite output and out-of-range
    predictions, then formats classic-locale decimal lines and publishes stdout
    once;
  - writes stderr diagnostics and, with `COURSE_PROFILE=1`, allocates all four
    events per batch before the loop and reports per-batch plus aggregate GPU
    event sums and separate host wall buckets. Enqueue is labeled host work;
    the default path creates no events, profile records, or per-batch sync.
- Root `main.cu` is macro-exclusive: `COURSE_TRAIN` compiles only the training
  parse/dispatch branch; `COURSE_INFER` compiles only inference. Both map
  `std::invalid_argument` to exit 2 and other standard failures to exit 1,
  write diagnostics to stderr, use classic locales, and never write usage or
  failure text to stdout.
- `CMakeLists.txt` defines `course_runtime`, `train`, and `infer` **outside**
  `COURSE_BUILD_TESTS`. Only `course_runtime_tests` and its registration are
  inside the optional test switch. `infer` links only `course_runtime`
  (-> `course_host` + `course_forward`); it does not link
  `course_reference_training`, `course_training`, original LeNet, loss,
  metrics, Adam, `src/train.cu`, or `src/main.cu`.
- `docs/course-interface.md` documents exact commands/environment/defaults,
  raw array and bundle formats, persisted-mode semantics (including the not
  externally confirmed unit convention), pure stdout/exit taxonomy, fixed
  shapes, no online preprocessing verification, profiling keys, GPU-event
  versus host-wall and full-process timing honesty, no warmup substitution,
  and caller-supplied arbitrary paths with no hidden labels/weights.

### Local static checks performed

These are host-only checks; they are **not** CUDA or Linux build claims. The
Windows syntax pass temporarily stripped the suite's intentional
`#if defined(_WIN32) #error` guard and supplied POSIX/CUDA declarations from
temporary shim headers; no repository source was modified for the check.

```sh
# Test TU syntax/types, with a temporary POSIX declaration shim on Windows.
g++ -std=c++14 -Wall -Wextra -Wpedantic -I <posix-shim> -I. -Iinclude -Isrc \
  -fsyntax-only -x c++ course/tests/runtime_tests.cu

# Runtime host/device API syntax/types against a temporary CUDA declaration
# shim (no device code is linked or executed).
g++ -std=c++14 -Wall -Wextra -Wpedantic -I <cuda-shim> -I. -Iinclude -Isrc \
  -fsyntax-only -x c++ course/runtime.cu

# Infer branch of the root dispatcher (training branch requires real CUDA
# headers, which are absent on the local Windows host).
g++ -std=c++14 -Wall -Wextra -Wpedantic -DCOURSE_INFER -I. -Iinclude -Isrc \
  -fsyntax-only -x c++ main.cu
```

All three commands completed without diagnostics under the local shims. The
runtime/test sources contain no personal absolute paths. `git status` remains
limited to the prior task work plus the Task 4 additions; no Task 1/2/3 source
was edited and no git state was changed.

### Remaining H20 verification (main)

1. Re-run the named-entry-point RED if desired: build only the test target
   before `train`/`infer`, or run `--filter train_cli_usage` before those
   targets exist; the failure must name the missing build-adjacent executable,
   not hardware.
2. Build from a clean CMake directory with `-DCOURSE_BUILD_TESTS=ON`; run all
   four CTest suites plus explicit subprocess assertions. The runtime suite must
   report `executed=15 failures=0 status=pass`.
3. Build the default minimal configuration (`COURSE_BUILD_TESTS=OFF`) to prove
   `train`/`infer` are still produced without `course/tests`.
4. Audit the `infer` link command and binary as described above. Confirm no
   original training workflow, LeNet training archive, loss, metrics, or Adam
   leakage.
5. Run the actual H20 profiling and external wall-time measurements. No local
   timing or 100 ms claim is made.

Local limitations: no NVCC, no CUDA device, and no Linux runtime were available
on this Windows host; all production and test execution claims remain pending
main's H20 run.

## New focused RED test for batch-sensitive input slicing

Main source inspection found that `course/runtime.cu:270-287` treats
`batch_index * batch_size` as both a sample offset and a float offset:
`host_images.data() + offset`, `packed.get() + offset`, and
`normalized.get() + offset` advance by that many floats, while the prediction
pointer correctly advances by that many samples. The old initialized-model
cases produced uniform classes and could not expose the mis-slice.

Added exactly one deterministic case, retaining the existing 13:
`infer_batch_partitions_keep_each_sample_input`.

- Fixture model is real, finite, and format-valid: normalized metadata, hidden
  conv/fc1/fc2 weights all `1.0F` and biases `0.0F`, fc3 weights `1.0F` only on
  the class-0 row, other class rows `0.0F`, and fc3 class-1 bias `1.0e8F`.
- Raw input is a real two-image normalized file: first all `0.0F`, second all
  `1.0F`. `ReadImages` verifies both values are finite and bit-preserved; both
  are valid normalized inputs. No host neural-network arithmetic is used.
- Oracle: correct per-sample slicing yields literal predictions `[1, 0]` for
  both B=1 and B=2. For the all-zero image, class 0 logit is `0` and class 1
  bias is `1e8`, so class 1 wins. For the all-one image, class 0 accumulates
  about `9.68e9`, above `1e8`, so class 0 wins.
- The case runs the official GPU CLI twice with the same file, asserting exact
  `[1, 0]`, completion `batches=2` for B=1 and `batches=1` for B=2, and
  byte-identical stdout between the two capacities.
- With the current mis-sliced runtime, B=1's second batch starts at float 1
  (783 zeros plus one 1), whose class-0 logit stays below the `1e8` class-1
  bias, so it should produce `[1, 1]`; B=2 copies contiguous samples and should
  produce `[1, 0]`. The exact-output and equality assertions therefore fail.
- `CMakeLists.txt` expected CTest count was updated from 13 to 14. This is the
  only CMake change in this handoff.
- `course/runtime.cu`, `main.cu`, `course/runtime.h`, and `docs/course-interface.md`
  were not modified for this RED step. No old source, git state, SSH, subagent,
  install, or credential action occurred.

Local static check: the updated test TU passed the same host-only
`g++ -std=c++14 -Wall -Wextra -Wpedantic -fsyntax-only` check against the
temporary POSIX declaration shim, with 14 registered `TEST_CASE` entries. No
CUDA/H20 run is claimed. Main should build the test and run the focused real-GPU
case against the current runtime to record the expected RED before any
production fix.

## Minimal runtime offset fix after confirmed RED

Main confirmed the focused case failed against the frozen runtime with
`executed=1 failures=1`, the second expected prediction `0` failure at line
1101, and the production hash matching
`task-4-before-offset-fix/manifest.json`.

Only `course/runtime.cu` changed (plus this report and one documentation
clarification). The correction is:

- Added a checked `batch_element_count = batch_size * 784` and allocated
  `packed` and `normalized` to that one-batch capacity instead of the full-N
  `image_element_count`.
- Every batch now uploads to `packed.get()` at pointer zero and prepares into
  `normalized.get()` at pointer zero; `model.Forward` receives
  `normalized.get()` and each batch reuses the same configured capacity.
- The H2D host source is
  `host_images.data() + checked(offset * 784)` where `offset` remains the
  sample offset; the copy size remains
  `current_batch * 784 * sizeof(float)`.
- Prediction storage remains all-N and `device_predictions.get() + offset`
  remains the sample offset, as does the final all-N readback.
- `image_element_count` remains full-N and is still used to validate the host
  image vector length.
- Sticky-flag zeroing once, all-N prediction/one-flag preallocation, the single
  final readback/sync, no allocation/sync/readback/events in the fast loop, and
  every profiling event boundary and duration calculation remain unchanged.
- The literal `[1, 0]` oracle and all prior cases were not weakened.

The documentation now states explicitly that input/normalized buffers are
batch-capacity while predictions are all-N. Local static check after the fix:
the runtime TU again passed the host-only
`g++ -std=c++14 -Wall -Wextra -Wpedantic -fsyntax-only` check against the
temporary CUDA declaration shim with no diagnostics. No NVCC, CUDA, or H20 run
is claimed.

## Formal fix round 1: buffered sink failure (F1/F2/F3)

Review finding F1 was real: the runtime checked `!predictions` immediately after
the single insertion but never flushed, so a small payload accepted into a
buffered sink escaped to a later sink flush/exit while the process still
returned 0 and wrote a false success record.

### Test added first (15 total, existing 14 retained)

`run_course_infer_reports_buffered_sink_failures` is registered at the end of
the TU, after all fork-based subprocess cases and after the existing direct-API
CUDA cases, so no new fork occurs after parent CUDA initialization.

- It uses owned scratch model/image fixtures and direct `RunCourseInfer`.
- Prediction sink: a real buffered `std::ofstream` opened on `/dev/full`; open
  is checked explicitly. The call must throw `std::runtime_error`, and captured
  diagnostics must not contain `event=course_infer status=pass` because the
  prediction flush failure precedes success diagnostics.
- Diagnostics sink: a separate real `/dev/full` `std::ofstream`; the call must
  throw `std::runtime_error`. Predictions may already be validly published
  before this failure, and the test parses them rather than asserting rollback.
- No CPU NN computation, mock kernel, test hook, or dataset write is used.
- `CMakeLists.txt` CTest expected count is now `executed=15`.

Against the old runtime this case fails both `EXPECT_THROWS_AS` checks; main is
expected to link it independently against the frozen archive for the recorded
RED. The existing actual minimal CLI `/dev/full` probe supplies the CLI-level
RED evidence.

### Minimal production correction

Only the publication boundary in `course/runtime.cu` changed:

- After the one prediction payload insertion, call `predictions.flush()` and
  check `!predictions` before assembling/writing any success diagnostics.
  `host_write_predictions_ms` now stops after that flush/check, so its wall
  bucket includes flush cost.
- After the one diagnostics insertion, call `diagnostics.flush()` and check
  `!diagnostics` before returning 0.
- No pipeline, arithmetic, batch-offset fix, sticky flag, readback, allocation,
  or timing-event boundary changed. Validation failures still publish no
  prediction byte; the single payload insertion remains.
- Flushing cannot roll back bytes already accepted by a partial sink; the
  existing honest caveat remains in the header/docs.

Docs F2 now states: the `unit` default is a user-approved project default
assumption, not an externally confirmed course preprocessing requirement.

### Local static checks

```sh
# Runtime source with the temporary CUDA declaration shim.
g++ -std=c++14 -Wall -Wextra -Wpedantic -I <cuda-shim> -I. -Iinclude -Isrc \
  -fsyntax-only -x c++ course/runtime.cu

# Test TU with the temporary POSIX declaration shim (15 TEST_CASE entries).
g++ -std=c++14 -Wall -Wextra -Wpedantic -I <posix-shim> -I. -Iinclude -Isrc \
  -fsyntax-only -x c++ course/tests/runtime_tests.cu
```

Both host-only checks completed without diagnostics. Local results:
`RUNTIME_SYNTAX_OK`, `TEST_SYNTAX_OK`, registered `TEST_CASE` count `15`,
`DIFF_CHECK_OK`. No NVCC, CUDA device, H20, `/dev/full` runtime, or CLI probe
was executed locally. No Task 1/2/3 source, git state, SSH, subagent, or install
action was used.
