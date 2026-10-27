# Task 5 helper implementation report

Status: helper implementation complete and locally GREEN (27 tests). Linux/H20
GREEN and all actual accuracy/timing measurements remain main-agent work. No
H20 accuracy, latency, or performance result is claimed here.

## Files created

- `tests/test_course_fixture.py` (13 tests)
- `tests/test_course_benchmark.py` (14 tests)
- `scripts/course_fixture.py`
- `scripts/course_benchmark.py`
- `docs/course-interface-results.md` (usage/status only; no invented metrics)
- `.superpowers/sdd/2026-10-27-course-gpu-interface/task-5-tests-red.txt`
  (captured missing-module RED)
- `.superpowers/sdd/2026-10-27-course-gpu-interface/task-5-green-local.txt`
  (local 27-test GREEN)

No Task1-4 source, CMake, old source, plan/history document, remote path, or
mounted data was modified. Tests use only the Python standard library and owned
temporary directories.

## RED evidence

Local command (Python 3.12; Linux/H20 owns the authoritative GREEN):

```text
python -m unittest -v tests.test_course_fixture tests.test_course_benchmark
```

Before implementation: `FAILED (errors=2)`, both named missing modules:

```text
ImportError: cannot import name 'course_fixture' from 'scripts'
ImportError: cannot import name 'course_benchmark' from 'scripts'
```

Main independently verified this RED on Linux with system Python 3.10
(`errors=2`, `task-5-red.txt`). Advertised GREEN interpreter is the
`gemini_env` Python 3.12.12.

After implementation, the same local command reports `Ran 27 tests ... OK`.
This is a local stdlib check only; main's Linux run is authoritative.

## Source schema pinned by the fixture tests

Derived from the current producer `scripts/prepare_mnist.py` and reader
`src/dataset.cpp` (and Task5 acceptance text), not guessed:

- 24-byte header: `MNISTC1\0`, little-endian u32 version=1, N, rows=28,
  columns=28.
- Body: all `N * 784` image bytes first, then all N label bytes. The body is
  NOT interleaved. The fixture test `test_labels_are_read_from_the_trailing_block`
  fails an interleaved O(N*785) reader even though the total size matches.
- Exact source size: `24 + N * 785`; zero N, wrong version/shape, truncation,
  trailing bytes, and count/size mismatch are rejected.
- Labels are uint8 class IDs in `[0, 9]`; 10 and above are rejected.

## Implemented fixture API

```python
# scripts/course_fixture.py
convert_mnistc1(source_path, output_dir, input_mode) -> manifest (dict)
main(arguments=None) -> int          # 0 success; argparse usage exits 2
```

- `input_mode` is exactly `unit`, `byte`, or `normalized`.
- Invalid source content/mode -> `ValueError` and no `images.f32` /
  `labels.i32` / `manifest.json` left behind.
- Existing `output_dir` (empty, populated, file, or symlink) ->
  `FileExistsError`, with the existing target byte-for-byte unchanged.
- Source, including a read-only source reached through a symlink, is never
  modified.
- `images.f32` is `<f` binary32 little-endian; `labels.i32` is `<i` int32
  little-endian. Conversion uses bounded chunks and never stores all dataset
  floats in one Python list.
- Manifest must have exactly these keys:

```json
{
  "schema": "course-fixture-v1",
  "source": "<caller path>",
  "source_sha256": "<64 hex>",
  "n": 1,
  "shape": [1, 28, 28, 10],
  "input_mode": "unit|byte|normalized",
  "images": "images.f32",
  "labels": "labels.i32"
}
```

Pixel conversion pinned by the tests (binary32 rounding after each operation):

```text
unit:       f32(p / f32(255))
byte:       f32(p)
normalized: f32((f32(unit - f32(0.1307))) / f32(0.3081))
```

`normalized` therefore emits the persisted inference domain directly and is
negative for pixel 0. It must not do a uint8 roundtrip.

## Implemented benchmark API

```python
# scripts/course_benchmark.py
run_benchmark(
    infer_path, images_path, labels_path, model_dir, output_json,
    count, batch, runs, model_source,
    profile=False, reference_predictions=None,
) -> result (dict)
main(arguments=None) -> int          # 0 pass; 1 run/accuracy/reference fail; 2 usage
```

### Process contract

- Infer is launched with exactly:
  `[images_path, str(count), "1", "28", "28", "10"]`.
  The labels path is never an argument, never an inference input, and is used
  only by this external checker.
- Environment is inherited with `COURSE_PROFILE`, `COURSE_INPUT_MODE`, and
  `COURSE_EPOCHS` removed; only `COURSE_MODEL_DIR` and
  `COURSE_INFER_BATCH` are set. For `profile=True`, `COURSE_PROFILE=1` is set
  for every run in that profiled artifact.
- Timing is external `time.monotonic()` around the whole subprocess, from
  startup through wait/communicate and exit. No mock timer replaces it.
- `runs` requested processes always produce exactly `runs_requested` JSON run
  records. A profiled call is a separate artifact: all R requested runs are
  profiled (`profiled=true`), with no hidden extra process and no unrecorded
  diagnostic run.

### Validation and summary semantics

- A valid run exits 0 and writes exactly N decimal integer lines in `[0, 10)`;
  extra, missing, non-integer, negative, or out-of-range lines are invalid.
- Per-run `correct` is the number of matching samples for an exit-0 valid run;
  invalid output or a nonzero exit forces `correct = 0`.
- All runs must be valid and produce identical parsed predictions. Accuracy is
  computed only from the external int32 labels; `accuracy_pass` is true only
  when all runs are valid, identical, and accuracy is at least 0.96.
  If any condition fails, `summary.correct = 0`, `accuracy = 0.0`, and
  `accuracy_pass = false`.
- All wall samples are retained, including outliers. `median_ms` and `max_ms`
  are over the recorded runs; `all_wall_below_100ms` is true only if every
  retained run is `< 100.0`.
- Nonzero exits and stderr are preserved in the failure JSON. Existing output
  JSON is never overwritten (`FileExistsError`, also for a symlink alias).

### Binding profiling ruling (main, pre-production)

The exact pinned schema has only `runs_requested` and `profiled`, and every
executed process must be preserved. Therefore `--profile` means: the R
requested runs are PROFILED diagnostic runs in a separate `profiled=true`
artifact. It does NOT mean R fast runs plus one unrecorded extra process. Each
profiled run's stdout/stderr/wall time is retained. Main uses an R=5 unprofiled
artifact for fast-path evidence and a separate `--profile --runs 1` artifact for
diagnostics. Profiling neither replaces nor contaminates fast acceptance.

### Pinned JSON schema `course-benchmark-v1`

Top-level exact keys (plus the two optional reference keys when a reference is
supplied):

```text
schema, executable, images_path, labels_path, model_dir, model_source,
n, batch, input_mode, profiled, runs_requested, sha256, runs, summary
reference_sha256, reference_mismatches          # optional
```

- `sha256` exact keys: `binary, model, meta, images, labels`.
  `model` is `model_dir/lenet.bin`; `meta` is `model_dir/course.meta`.
- `input_mode` is read from the persisted `course.meta` line
  `input_mode=unit|byte|normalized`.
- `runs` exact keys: `wall_ms, returncode, correct, stdout, stderr`; stdout and
  stderr are the retained text.
- `summary` exact keys:
  `median_ms, max_ms, correct, accuracy, accuracy_pass, all_wall_below_100ms`.
- `model_source` must be exactly `legacy` or `course_train`.

### Reference file

- Exactly N uint8 bytes, each in `[0, 9]`. Wrong length, a 4*N int32 labels
  file masquerading as a reference, or any byte above 9 -> `ValueError` before
  any infer process starts.
- Records `reference_sha256` and `reference_mismatches`; a nonzero mismatch
  count must make the CLI exit nonzero even when accuracy passes.

## Mock executables

Benchmark tests use harmless real mock processes, never GPU or CPU neural
network code. On POSIX they are executable Python scripts; on Windows they use
a `.cmd` wrapper so the same tests can produce local RED. Each mock records the
exact argv and relevant `COURSE_*` environment, then emits configured
stdout/stderr/exit status or sleeps. Mocks only exercise helper failure and
measurement branches; actual acceptance still requires real compiled GPU infer.

## Implementation limits

- Helpers are development-only and are excluded from the submission runtime and
  CMake targets.
- Benchmark tests use harmless mock processes; they are not a CUDA model and do
  not prove accuracy.
- Local Windows mock timings are contract checks, not H20 measurements.
- Main must run with `TMPDIR` inside a build-owned experiment and record actual
  10,000-sample accuracy and full-process timing separately.

## Main next steps

1. Run the authoritative Linux GREEN with `gemini_env` Python 3.12.12:
   `python3 -m unittest -v tests.test_course_fixture tests.test_course_benchmark`
   and require 27 tests with nonzero pass count.
2. Convert read-only splits into owned temporary fixture directories under the
   fresh experiment; keep `TMPDIR` build-owned. Do not touch actual mounts.
3. Run real unprofiled R=5 artifacts per feasible batch, then one separate
   `--profile --runs 1` diagnostic artifact.
4. Fill `docs/course-interface-results.md` with actual measured accuracy,
   process/GPU timings, hashes, and truthful 100 ms status. The helper suite
   does not prove H20 accuracy or latency.
