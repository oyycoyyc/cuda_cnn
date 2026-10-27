# Task 6 report: package/acceptance helper contract

Status: **GREEN implemented locally** (main Linux GREEN/review still pending).
Main verified the named-missing-helper RED (`task-6-package-red.txt`, errors=1).
`tests/test_course_package.py` and `scripts/course_package.py` now exist. No
root `evaluation.json`, `report`, `result.png`, or archive was created by this
handoff; all package tests use owned temporary repository/fixture trees.

All benchmark values in the tests are synthetic owned fixture data. They are
never actual H20 measurements and must never be copied into docs/results or
used as a submission claim.

## Environment / local checks

- Local development Python: 3.12.7. Local Pillow: 10.4.0 importable and used
  only for renderer tests. Remote Task 6 notes confirm Pillow 12.1.0 in
  `gemini_env`; no installation is performed by the helper or tests.
- Renderer tests use `unittest.skipUnless(..., <visible message>)` when Pillow
  is not importable. Skips are not silent; the availability/version is in the
  skip reason. The missing-Pillow error test runs regardless and requires a
  lazy import inside the renderer.
- `python -m py_compile tests/test_course_package.py` and
  `scripts/course_package.py` pass.
- Historical local RED (helper absent) is preserved in
  `task-6-tests-red-local.txt`; main's Linux RED is `task-6-package-red.txt`.
- Local GREEN: `python -m unittest tests.test_course_package -v` -> **39 tests
  OK** in 6.1 s (log: `task-6-package-green-local.txt`). Rendering and the
  actual symlink rejection ran locally; no test skipped.
- Prior helper regression locally: `tests.test_course_fixture` +
  `tests.test_course_benchmark` -> **27 tests OK** in 3.9 s
  (`task-6-prior-helper-tests-local.txt`).
- `_validate_whitelist(_repository_root())` + `_scan_payload` over all 34 real
  repository sources: clean, 34 files.
- No actual root artifact/archive was produced locally; Windows local execution
  is not a CUDA/H20 acceptance result.
- 39 test methods, with subtest cases covering 73 logical checks. No huge
  fixtures are used for cap tests; archive/image cap boundaries are pure
  function checks.

## Proposed module: `scripts/course_package.py`

Standard library only, except the development-only renderer import. No
runtime/package dependency on Pillow. No neural-network work, no execution of
`train`/`infer`, and no label use by inference: labels are read only by this
external acceptance helper.

### Public constants

```python
PACKAGE_SCHEMA = "course-package-v1"      # helper/result schema, not written to root
EVALUATION_CONTENT = {
    "mode": "image",
    "image_origin": "archive",
    "result_image": "result.png",
}
EVALUATION_JSON_TEXT = (
    '{"mode":"image","image_origin":"archive","result_image":"result.png"}\n')
BENCHMARK_SCHEMA = "course-benchmark-v1"
REQUIRED_N = 10000
REQUIRED_RUNS = 5
REQUIRED_EPOCHS = 20
ACCURACY_THRESHOLD = 0.96
FULL_WALL_LIMIT_MS = 100.0
INPUT_MODES = ("unit", "byte", "normalized")
WHITELIST_FILES = (... exact sorted tuple below ...)
MAX_ARCHIVE_BYTES = 100000000              # strict: compressed must be <
MAX_UNPACKED_BYTES = 300000000             # strict: unpacked must be <
MAX_IMAGE_BYTES = 4000000                  # inclusive: must be <=
MAX_IMAGE_PIXELS = 16000000                # inclusive: must be <=
```

### Exception

```python
class PackageError(ValueError):
    """Invalid or missing provenance/configuration/artifact."""
```

`FileExistsError` (OSError) is reserved for an existing root artifact or output
archive; it must be raised before any write/overwrite. CLI maps
`PackageError`/`OSError` to exit 1 and usage errors to argparse exit 2.

### Functions the tests call directly

```python
def _repository_root():
    """Parent of scripts/, from this file's location; no cwd dependency."""

def validate_member_name(name):
    """Return name unchanged or raise PackageError for an unsafe archive name."""

def check_archive_limits(compressed_bytes, unpacked_bytes):
    """Raise unless compressed < 100,000,000 and unpacked < 300,000,000."""

def check_image_limits(image_bytes, pixels):
    """Raise unless bytes <= 4,000,000 and pixels <= 16,000,000."""

def metric_lines(accuracy, median_ms):
    """Exactly two lines, for both the image and the report:
    ['Test Accuracy: 96.42%',
     'Test Time: 88.10 ms (median of 5 full-process runs)']
    """

def run_package(benchmark_json, training_log, output_zip, root=None):
    """Validate everything, render/report/archive, and return:
    {
      'members': sorted member names,
      'correct': int,
      'accuracy': float,
      'median_ms': float,
      'max_ms': float,
      'performance_target': 'achieved' | 'failed',
    }
    root=None means _repository_root().
    """

def main(arguments=None):
    """CLI: benchmark, training-log, output only. Calls
    run_package(benchmark, training_log, output, root=_repository_root())."""
```

The tests also reference `course_package.__file__`, `EVALUATION_CONTENT`,
`EVALUATION_JSON_TEXT`, `WHITELIST_FILES`, `PackageError`, and the functions
above. No other internal name is part of the test contract.

### CLI (exactly per brief)

```sh
python3 scripts/course_package.py \
  --benchmark BENCHMARK.json --training-log TRAIN.log --output SUBMISSION.zip
```

No repository-root option and no new required arguments. Tests use the
internal `root=` argument only. The output parent directory must already exist;
the helper does not create arbitrary directory trees.

## Evaluation JSON

Exact bytes (single trailing LF, compact key order as shown):

```
{"mode":"image","image_origin":"archive","result_image":"result.png"}
```

Semantically exactly those three keys/values. Tests assert both bytes and
parsed equality.

## Submission whitelist (34 files, sorted)

Derived from the default CMake targets `train`/`infer` and their transitive
local headers only. `course/tests`, `course/tools`, Python tools, docs,
evidence, data, weights, build/runtime-output, CMake test tooling, Dockerfile,
credentials, and symlinks are excluded.

```
CMakeLists.txt
course/gpu_ops.cu
course/gpu_ops.h
course/inference.cu
course/inference.h
course/interface.h
course/io.cpp
course/model_io.cpp
course/model_io.h
course/runtime.cu
course/runtime.h
course/training.cu
course/training.h
include/checkpoint.h
include/cuda_check.h
include/layers.h
include/lenet.h
include/parameters.h
include/random.h
include/tensor.h
main.cu
src/binary_io.h
src/checkpoint.cpp
src/kernels/activation.cu
src/kernels/adam.cu
src/kernels/convolution.cu
src/kernels/input.cu
src/kernels/linear.cu
src/kernels/loss.cu
src/kernels/metrics.cu
src/kernels/pooling.cu
src/lenet.cu
src/parameters.cpp
src/random.cpp
```

Plus generated root artifacts `evaluation.json`, `report`, `result.png`. The
ZIP has exactly these 37 root-level members, sorted, with forward-slash names,
no directory entries, no duplicate/alias names, and no common root folder.

`validate_member_name` rejects: non-string/empty, `.`/`..`, leading `/`,
Windows drive (`C:`), backslash, empty components (`//`), `.`/`..` components,
leading `./`, trailing `/`, and NUL. It returns the unchanged name for every
whitelisted/generated member. Root files must be regular, single-link,
non-symlink files whose real path stays under the repository root.

## Required benchmark input (`course-benchmark-v1`, actual Task 5 schema)

Read the real file, do not guess fields. Required top-level keys:

`schema`, `executable`, `images_path`, `labels_path`, `model_dir`,
`model_source`, `n`, `batch`, `input_mode`, `profiled`, `runs_requested`,
`sha256`, `runs`, `summary`.

- `schema == "course-benchmark-v1"`, `model_source == "course_train"`,
  `profiled is False`, `n == 10000`, `runs_requested == 5`, `len(runs) == 5`.
- `batch` is an int in `[1, 16384]` (not bool); value is reported, not forced to
  1024, because an explicit valid batch is still a full-test result.
- `input_mode` is one of `unit|byte|normalized` and must equal the model meta
  mode and the training completion mode.
- `sha256` has exactly `binary`, `model`, `meta`, `images`, `labels`.

## Training completion provenance

The supplied log may contain epoch/event lines. It must contain exactly one
line beginning `event=course_train `; duplicate completion lines are rejected.
That line is parsed as `key=value` tokens. Required keys:

- `status=pass` (any non-pass rejected)
- `epochs=20` (pilot/1/19 epochs rejected)
- `best_epoch` integer in `[1, 20]`
- `validation_accuracy` finite float in `[0, 1]`
- `input_mode` in the supported modes
- `checkpoint_crc32` exactly eight lowercase hexadecimal digits

The CRC in that line is the training publication's own completion CRC. It must
equal both the model `course.meta` CRC and `zlib.crc32(lenet.bin)` recomputed
by the helper; this detects a mixed/stale model pair even when hashes and log
were edited consistently.

## Model/meta/binary binding

- `model_dir/lenet.bin` and `model_dir/course.meta` must be regular non-symlink
  files.
- `course.meta` must be exactly the four LF-terminated lines from
  `course/model_io.h` in order: `course_lenet_v1`, `input_mode=<mode>`,
  `shape=1,28,28,10`, `checkpoint_crc32=<8 lowercase hex>`. No extra/unknown/
  duplicate/reordered fields; at most 256 bytes.
- Recompute `zlib.crc32(lenet.bin) & 0xffffffff`; it must equal the meta CRC and
  the training completion CRC.
- Recompute SHA-256 for `executable`, `lenet.bin`, `course.meta`, images, and
  labels and require equality with the benchmark's recorded values. This binds
  the package to the exact current measured binary/model/meta files (a rebuild
  or retrain at the recorded paths is therefore rejected as stale). Paths may
  be outside the repository; they are never archived.
- Sizes: images exactly `n*784*4` bytes, labels exactly `n*4` bytes. Labels are
  little-endian int32 in `[0,10)`, read only by this checker.

## Five-run acceptance and summary consistency

For each of the five runs:

- `returncode == 0` (reject bool), stderr retained but not parsed.
- `stdout` is exactly `n` decimal class-integer lines, each in `[0,9]`.
- Recompute `correct` from stdout against labels; it must equal the run's
  recorded `correct`, and every run's prediction vector must be identical.
- `summary.correct` must equal the recomputed correct count; it must be
  `>= 9600` and `summary.accuracy == correct/n` must be `>= 0.96`.
- `summary.accuracy_pass` must equal `accuracy >= 0.96`.
- `summary.median_ms` must equal `statistics.median(run walls)` and
  `summary.max_ms` must equal `max(run walls)` (float tolerance 1e-6 ms).
- `summary.all_wall_below_100ms` must equal `all(wall < 100.0 ...)`.

Any incomplete run list, nonzero exit, malformed/short/long prediction output,
prediction disagreement, label disagreement, forged run correct count, or
fabricated summary field is rejected before any output is written.

`performance_target` is `"achieved"` only when every wall is `< 100.0` ms;
otherwise `"failed"`. A truthful `failed` result is still packaged (accuracy
gate may pass) and is labelled failed in both image and report. It is never
rewritten or suppressed.

## Image rendering

- Lazy `_import_pillow()` inside the renderer only. On ImportError raise
  `PackageError` with a message containing `Pillow`, `development`, and
  `install` (state no automatic installation). No top-level Pillow import.
- Use `Image.new("RGB", ...)`, `ImageFont.load_default(size=24)`, and
  `ImageDraw.Draw(...).text(...)` for each metric line. No system/personal font
  paths. Save PNG.
- Draw exactly the two `metric_lines` strings, each once. No other text may
  contain `Test Accuracy:` or `Test Time:`. The image therefore has one metric
  group and no conflicting timing values; scope is stated as median of five
  full-process runs.
- Validate `<= 4,000,000` file bytes and `<= 16,000,000` pixels before
  publishing. Decimal caps are intentional: the brief says MB, and decimal
  limits satisfy both decimal and MiB interpretations.

## Report (`report`, exact useful lines)

The report is UTF-8 text and must include, at minimum, these exact substrings
(tests pin them):

```
Course submission report
Model source: course_train
Preprocessing assumption: ...
Training completion: event=course_train status=pass epochs=20 ...
Training epochs: 20
Configuration: n=10000 batch=1024 runs=5 profiled=false
Run 1: wall_ms=88.100 correct=9642
...
Run 5: wall_ms=91.500 correct=9642
Median full-process wall: 88.100 ms
Max full-process wall: 91.500 ms
max_ms=91.500
Test Accuracy: 96.42% (9642/10000)
Test Time: 88.10 ms (median of 5 full-process runs)
Accuracy target (>=96%): PASS
Performance target (full-process <100 ms): ACHIEVED   # or FAILED
Leaderboard timing: unresolved; ...
Provenance:
checkpoint_crc32=<completion/meta/model CRC>
binary_sha256=<...>
model_sha256=<...>
meta_sha256=<...>
images_sha256=<...>
labels_sha256=<...>
```

Every run is listed (wall and correct), plus median, max, configuration,
preprocessing assumption, all hashes/CRC, and the unresolved leaderboard timing
scope. The accuracy/time lines are generated consistently with the benchmark;
nothing is invented. For a slow run the report says `FAILED`, not `ACHIEVED`.

## Write ordering / failure safety

1. Resolve root; check `evaluation.json`, `report`, `result.png`, and output ZIP
   with `os.path.lexists` (includes symlink aliases). Any existing path raises
   `FileExistsError` before writes.
2. Validate training log, benchmark, whitelist/tree, model/meta/CRC, hashes,
   inputs, all five runs, and summary. No writes in this phase.
3. Render PNG and build evaluation/report/ZIP bytes in a temporary directory or
   memory; validate image and archive limits.
4. Publish the three root files with `O_CREAT|O_EXCL` (never replace), then the
   ZIP with exclusive creation and no overwrite. If publication fails after
   creating any of this invocation's files, remove only the files created by
   this invocation; never touch preexisting files.

Thus missing/invalid provenance, profiled/stale/incomplete/fabricated data, and
preexisting outputs produce no final root artifacts, image, or archive.

## GREEN implementation decisions

- `scripts/course_package.py` implements the reported API with standard-library
  imports only; Pillow is imported lazily by `render_result_png` and never at
  module import. No auto-install and no system font paths.
- Decimal caps are used: compressed `< 100000000`, unpacked `< 300000000`,
  PNG `<= 4000000` bytes, `<= 16000000` pixels. `check_image_limits` has the
  actual two-argument `(image_bytes, pixels)` API; the renderer supplies
  `width * height` after positive-size and cap checks.
- Strict benchmark JSON decoding rejects duplicate object keys and
  NaN/Infinity. Numeric fields are finite and non-bool; counts are non-bool
  integers.
- Every run must have canonical single-digit `0..9` stdout lines, recomputed
  correct count, identical predictions, and consistent summary/wall fields.
  Any `event=course_profile` evidence in per-run stderr is rejected even when
  `profiled` is falsely false.
- Whitelist validation rejects duplicate/absolute/backslash/traversal names,
  nonregular files, symlinks, hardlinks (`st_nlink != 1`), and real paths
  outside the repository. Payload scan rejects obvious personal absolute paths
  and credential material before publication.
- Report/image contain no repository/model/input absolute paths; report has all
  five walls, max/median, epochs 20, completion CRC, all hashes, input
  assumption, and unresolved leaderboard scope.
- The PNG/report/ZIP are built in memory, caps are checked, then the three root
  files and archive are created with exclusive `O_EXCL` semantics. A failure
  after this invocation created a file removes only that invocation's created
  files; no recursive cleanup and no overwrite.
- Checkpoint bytes are CRC/hash-checked but not decoded; the actual C++ loader
  remains main's responsibility for the real saved bundle. Hash/CRC binding is
  integrity evidence, not cryptographic proof against a fabricated benchmark or
  training log; main's actual run provides that trust.

## Test mapping (39 methods / 73 logical cases)

- Exact JSON, whitelist/graph, repository-root inference.
- Full success with exact 37 ZIP members, exact hashes/CRC, report/provenance,
  metric lines, PNG caps and single metric group.
- Truthful slow >100 ms accepted and marked failed; benchmark file unchanged.
- Accuracy 9600 passes; 9599 rejected.
- Incomplete/pilot training epochs, missing/failed/duplicate completion.
- Completion CRC vs meta vs recomputed checkpoint CRC.
- Stale binary/model/meta hashes and mismatched metadata mode.
- Profiled benchmark, profile events in stderr with a false flag, incomplete
  run set, nonzero run, forged per-run correct, prediction/label disagreement,
  six fabricated summary fields, duplicate/non-finite benchmark JSON, and
  personal-path/credential payloads.
- Legacy model source, wrong sample count, missing/short/modified images/labels.
- Existing archive and each preexisting root artifact preserved.
- Missing whitelist source, symlinked whitelist member, unsafe/alias member
  names, archive/image cap boundaries (pure checks).
- Missing Pillow clear error with no writes; CLI usage exit 2; CLI maps
  arguments and `PackageError` to exit 1.

## Local command log (Windows, synthetic fixtures only)

```
$ python -m py_compile scripts/course_package.py tests/test_course_package.py
compile ok

$ python -m unittest tests.test_course_package -v
Ran 39 tests in 6.113s
OK

$ python -m unittest tests.test_course_fixture tests.test_course_benchmark -v
Ran 27 tests in 3.876s
OK

$ python scripts/course_package.py --help
usage: course_package.py [-h] --benchmark BENCHMARK --training-log
                         TRAINING_LOG --output OUTPUT
```

Limits of this local evidence: Windows/Python 3.12.7/Pillow 10.4.0, not the H20
Linux Python 3.12.12/Pillow 12.1.0 environment; no CUDA, real course-trained
model, real benchmark, actual training, root artifacts, or archive was
exercised. The local symlink test ran (not skipped). Main's Linux GREEN,
actual acceptance packaging, extraction/build/roundtrip, and review remain
pending.
