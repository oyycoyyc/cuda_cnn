# Task 3 report: tests-first handoff for warm result packaging

**Status:** tests-only handoff. `tests/test_course_warm_package.py` is added;
`scripts/course_warm_package.py` deliberately does **not** exist yet and
`scripts/course_package.py` is **unmodified**. Local verification is the exact
named-missing-module RED. Main owns the Linux RED and then the GREEN resume.
No root artifact, ZIP, PNG, source, model, data, measurement record, CMake,
GPU, runtime, docs, or git state was changed.

## Proposed module and public API

```python
# scripts/course_warm_package.py  (TO BE CREATED BY GREEN)
WARM_BENCHMARK_SCHEMA = "course-warm-benchmark-v1"   # same object/value as collector
TIMING_SCOPE = "warm_full_request_h2d_prepare_forward_argmax_d2h_validation"
REQUIRED_N = 10000
MIN_RUNS = 101
MIN_WARMUPS = 1
REQUIRED_EPOCHS = 20
ACCURACY_THRESHOLD = 0.96
WARM_LIMIT_MS = 100.0
HASH_FIELDS = ("binary", "official_infer", "model", "meta", "images", "labels")

def run_warm_package(benchmark_json, training_log, output_zip, root=None): ...
def metric_lines(accuracy, mean_ms, median_ms): ...
def render_result_png(accuracy, mean_ms, median_ms, performance_target,
                      min_ms, max_ms, warmups, runs, n): ...
```

`run_warm_package` returns exactly
`{"members", "correct", "accuracy", "mean_ms", "median_ms", "min_ms",
"max_ms", "performance_target"}`; `performance_target` is `"achieved"` only
when measured **mean < 100 and median < 100**, else `"failed"`.

Shared identities the tests pin (reuse, never fork):

- `course_warm_package.PackageError is course_package.PackageError`
- `course_warm_package.parse_tool_output is
  course_warm_benchmark.parse_tool_output`
- `course_warm_package.WHITELIST_FILES == course_package.WHITELIST_FILES`
  (34 files; archive has exactly 37 members)
- `course_warm_package.WARM_BENCHMARK_SCHEMA` / `TIMING_SCOPE` equal the
  reviewed Task 2 collector constants
- `_repository_root()` resolves from the warm script location, exactly like
  the cold helper
- cold `BENCHMARK_SCHEMA == "course-benchmark-v1"` and `REQUIRED_RUNS == 5`
  remain untouched

`metric_lines` must return exactly two lines:

```text
Test Accuracy: 96.42%
Test Time: mean 21.00 ms; median 20.00 ms
```

## Validation contract pinned by the tests

Order: output refusal first (`course_package._check_output_paths`, so a
preexisting root member or archive raises `FileExistsError` before any
benchmark read), then whitelist, then training log, then the warm JSON.

The JSON must have exactly the reviewed 26 top-level keys. Required:

- schema/timing scope/model source/`input_mode` exact and supported;
  `model_source == "course_train"`; `n == 10000`; `batch` positive in
  `[1, 16384]`; `warmups >= 1`; `runs >= 101`; integral counts never bools
- `status == "pass"`, `errors == []`, `returncode` is int `0` (bool refused),
  `hashes_stable is True`, `predictions_identical_all is True`,
  `process_wall_ms` positive finite
- tool, official infer, images, labels, `model_dir`, `lenet.bin`, `course.meta`
  present and regular; six current SHA-256 values must equal **both**
  `sha256` and `sha256_after`, each with exactly the six keys and lowercase
  64-hex values
- `course.meta` parsed by the cold grammar; its mode equals the JSON mode and
  the training completion mode; training completion CRC equals the meta CRC;
  CRC32 recomputed from the checkpoint bytes equals the meta CRC; epochs `20`
- images exactly `n * 28 * 28 * 4` bytes; labels exactly `n * 4` int32 in
  `[0, 10)`
- raw `stdout`/`stderr` reparsed through `parse_tool_output(stdout, stderr,
  returncode, n, batch, warmups, runs, input_mode)`; stored
  `warmup_records`/`measured_records` must be **exactly equal** to the parsed
  records (including warmups and every per-record field), and the parsed
  completion must match the request counts/batch/mode/status
- stdout is exactly `n` canonical one-per-line class integers; predictions are
  recounted against the original int32 labels; `correct >= 9600`;
  `accuracy/accuracy_pass` recomputed
- summary must be independently recomputed from the **measured** records only:
  `mean_ms`, `median_ms`, `min_ms`, `max_ms` (`statistics`), `correct`,
  `accuracy`, `accuracy_pass`, `mean_below_100ms` (`mean < 100`),
  `median_below_100ms` (`median < 100`). Warmups never enter the statistics;
  no measured outlier may be dropped. All nine stored fields must match.
- accuracy below 0.96, bad provenance, stale hashes, raw/stored mismatch,
  partial sets, cold five-run shape, or a forged summary raise
  `PackageError` **before** any write. A valid measurement whose mean and/or
  median is `>= 100` still publishes truthfully with `"failed"` and CLI exit
  `0`; accuracy failure is a packaging rejection.

## Exact report and image contract

Report (`root/report`, no personal absolute paths):

- `Training completion: <the exact event=course_train line>` and
  `Training epochs: 20`, `Model source: course_train`
- `Configuration: n=10000 batch=1024 warmups=5 runs=101 ...`
- `Timing scope: warm full-request ...` naming H2D, GPU input preparation,
  original forward, argmax, D2H, stream synchronization, host validation
- `Warmups: 5 ... excluded from measured statistics`, then
  `Warmup i: wall_ms=X.XXX ms` for every warmup and
  `Measured i: wall_ms=X.XXX ms` for all 101 measured rounds, in order
- `Mean measured:`, `Median measured:`, `Min measured:`, `Max measured:`
  with three decimals
- one `Test Accuracy: 96.42% (9642/10000)` and the two `metric_lines`, one
  range/honest verdict line
- `Included per measured round:` H2D/prepare/forward/argmax/D2H/synchronize/
  validation; `Excluded from the measured interval:` setup, model/image disk
  loading, CUDA initialization, allocation, weight upload, final stdout
  formatting
- a sentence that the historical cold-process result/scope is unchanged
- `Leaderboard timing: unresolved`, official preprocessing unconfirmed
- `checkpoint_crc32=`, `binary_sha256=` (warm tool), `official_infer_sha256=`,
  `model_sha256=`, `meta_sha256=`, `images_sha256=`, `labels_sha256=`

Image (`root/result.png`, real Pillow draw calls checked with `textbbox`
inside the canvas and size/pixel caps): exactly one `Test Accuracy:` group and
one `Test Time: mean X ms; median Y ms` group, plus
`5 warmups; 101 measured rounds; 10000 images per round`,
`Warm full-request range: ...`, H2D/prepare/forward/argmax/D2H/synchronize/
host-validation scope, and
`Performance target (warm full-request mean and median <100 ms): ACHIEVED`
or `FAILED`. Neither report nor image may carry `startup through exit` or
`median of 5 full-process runs`.

Root publication reuses the cold `_build_archive`/`_publish` safeguards:
exact `evaluation.json` text, three generated root members plus 34 whitelist
files (37 archive members), decimal caps, exclusive creation, rollback of
partial root members, and no overwrite of any preexisting artifact.

## Planned minimal cold-CLI dispatch (GREEN, not done here)

In `scripts/course_package.run_package`, immediately after the existing
`_check_output_paths` call, load the JSON with `_load_json_strict`; if
`schema == "course-warm-benchmark-v1"`, return
`course_warm_package.run_warm_package(benchmark_json, training_log,
output_zip, root=repository_root)`. Every cold schema/path continues through
the untouched 942-line helper. `main` stays as-is (warm success and truthful
warm performance failure both exit `0`; any `PackageError`/`OSError` exits
`1` with a diagnostic). Import must work both as `scripts.course_package` and
as a direct script (sibling-import fallback).

## Tests added

`tests/test_course_warm_package.py`, 51 tests, stdlib `unittest` with the
existing development-only Pillow allowance:

| Class | Tests | Focus |
|---|---:|---|
| `WarmPackageApiTest` | 3 | constants, parser/error/whitelist identity, metric lines, default root |
| `WarmPackageSuccessTest` | 17 | 37 members/exact evaluation, measured-only stats, all 101 times, report scope/identities/no paths, PNG caps/groups/bboxes, achieved outlier, truthful failed, median boundary, W1/R102 minimums, 9600, forged-summary rejects |
| `WarmPackageRejectionTest` | 20 | cold schema/five-run shape, counts, partial/extra records, raw-vs-stored, completion, status/errors/returncode, training, CRC/modes, six stale identities, hash maps, predictions, invariance, forged summary, accuracy, inputs, missing tools, key set, process wall/batch |
| `WarmPackagePublicationTest` | 4 | preexisting archive/root members, publication rollback, immutable benchmark/training/inputs |
| `WarmPackageDispatchTest` | 6 | warm delegation, cold path untouched, output refusal ordering, CLI warm success/truthful failure/validation error |
| `WarmPackageDirectCliTest` | 1 | real copied-script direct CLI with PYTHONPATH removed: clean diagnostic, no traceback, no files |

Fixtures are owned temporary repositories with 10000 synthetic sparse images,
int32 labels, a synthetic checkpoint whose CRC is bound in `course.meta` and
the training log, five warmups, 101 measured records, six before/after hashes,
and all 26 JSON keys. Durations are literals such as `[20]*100 + [121]`
(mean 21, median 20, max 121) and `51*20 + 50*200` (mean 109.1, median 20).
They are arithmetic only, not GPU measurements, and no actual Task 2 number is
copied into expected metrics or root files.

## Commands and local results (win32, Python 3.12.7)

```text
python -m py_compile tests/test_course_warm_package.py
PASS

python -m unittest tests.test_course_warm_package
ImportError: cannot import name 'course_warm_package' from 'scripts'
  (unknown location)
FAILED (errors=1)
```

The second command is the intended RED: one named-missing-module error and no
feature test executed. Fixture scaffolding was additionally self-checked with
a non-functional placeholder module in memory only (not written anywhere): the
26-key fixture round-trips through the real `parse_tool_output`, stored
records equal parsed records, recount is 9642/10000, default measured mean is
12.2277/median 12.0, and the outlier series is 21/20/20/121. No packager logic
was written.

## GREEN implementation and verification (this session)

**Added:** `scripts/course_warm_package.py` (stdlib only, importable as
`scripts.course_warm_package`, as a direct script, and executable with
`python -S -m scripts.course_warm_package`). It aliases the reviewed Task 2
parser/schema/scope and the cold `PackageError`, whitelist, training-log,
`course.meta`/CRC, image/archive-cap, exclusive-write, and publication
helpers; it adds no second implementation of any of those safeguards.

Validation order is: cold `_check_output_paths` (preexisting root member or
archive refused before any benchmark read), cold whitelist, cold training
parser, then the warm evidence. The warm evidence must have exactly the 26
reviewed fields; `course_train`, `N=10000`, `W>=1`, `R>=101`, `status=pass`,
empty `errors`, int (non-bool) closed `returncode=0`, `hashes_stable=true`,
`predictions_identical_all=true`, positive finite `process_wall_ms`, exact six
before/after hash maps matching the current tool/official-infer/model/meta/
images/labels, bound training/`course.meta`/recomputed checkpoint CRC, exact
`n`-byte image and int32 label inputs, raw stdout/stderr reparsed through
`parse_tool_output`, stored records exactly equal to the parsed warmup and
measured records, canonical N-line predictions recounted against the original
labels (`>=9600`), and all nine summary fields independently recomputed from
the measured records only. No warmup or measured outlier is dropped; the
process wall is validated as positive finite only (no new chronology gate, so
the intentionally huge synthetic warmups remain valid test inputs).

`"achieved"` requires **mean and median < 100 ms**; otherwise a
provenance-valid, accurate result is still published truthfully as
`"failed"`. Accuracy below 0.96 is a packaging rejection, matching the cold
contract. Report and image use the exact warm strings pinned by the tests
(single accuracy/time groups, all 101 ordered measured times, warmups,
range, included/excluded stages, six identities, unconfirmed preprocessing,
leaderboard caveat, unchanged historical cold scope, no old cold captions,
no absolute paths).

**Dispatch:** `scripts/course_package.py` gained only the private
`_WARM_BENCHMARK_SCHEMA` literal, a lazy `_warm_package_module()` sibling
import with a direct-script fallback, and a schema probe immediately after
`_check_output_paths` in `run_package`. A warm document delegates to
`run_warm_package(..., root=repository_root)`; every cold path falls through
to the original 942-line behavior. `main`, cold constants, cold validation,
and cold output are unchanged.

**Objective test fixture bug (reported before changing it):** the initial
`test_training_crc_and_mode_bindings_are_rejected` subtests requested
`input_mode="byte"`/`"rgb"` from `_make_fixture`, which consistently rebuilds
`course.meta`, the training completion, and raw stderr from that same mode;
the "benchmark mode mismatch" case was therefore a fully consistent byte-mode
fixture that correctly packaged, and its root artifacts then contaminated the
next shared-root subtest. The two mode cases now keep the unit fixture and
mutate only the benchmark JSON (`byte` and unsupported `rgb`), which is the
intended binding check. Assertions/count/intent are unchanged; 50 tests
remain.

**Commands and exact local results (win32, Python 3.12.7, Pillow 10.4.0):**

```text
python -m py_compile scripts/course_warm_package.py \
  scripts/course_package.py tests/test_course_warm_package.py
exit 0

python -m unittest tests.test_course_warm_package
Ran 50 tests in 14.402s
OK

python -m unittest tests.test_course_benchmark tests.test_course_fixture \
  tests.test_course_package tests.test_course_warm_benchmark
Ran 97 tests in 14.798s
OK

python -m unittest tests.test_course_warm_package \
  tests.test_course_benchmark tests.test_course_fixture \
  tests.test_course_package tests.test_course_warm_benchmark
Ran 147 tests in 28.869s
OK          # no skips; Pillow 10.4.0 was already installed

python -S -m scripts.course_warm_package --help            exit 0
python -S scripts/course_warm_package.py --help            exit 0
python -S -m scripts.course_package --help                 exit 0
python -S scripts/course_package.py --help                 exit 0
```

`report`, `result.png`, and `course-submission.zip` at the repository root
retain their prior lengths and 2026-10-27 mtimes; no root artifact, actual
Task 2 JSON, source, CMake, GPU, runtime, docs, or measurement file was
written.

## Direct-CLI error-class integration fix

Main's independent real direct-script probe
(`task-3-direct-cli-probe.json`) found a class-identity gap that help and
in-process tests could not see. In `python -S scripts/course_package.py`, the
dispatcher runs as `__main__`, while the warm module's direct-script fallback
imports a separate sibling `course_package` module. The warm helper raised
that second copy's `PackageError`, so the original `__main__.main` handler did
not catch it: exit was still `1` with empty stdout and no files, but stderr
was a traceback instead of a `course_package:` diagnostic.

**New real RED/GREEN test:** `WarmPackageDirectCliTest` copies the three
helper scripts and all 34 whitelist files into an owned temporary root,
removes `PYTHONPATH`, writes an owned minimal bad warm JSON
(`{"schema": "course-warm-benchmark-v1"}`) plus a valid owned 20-epoch
training log, and runs the actual direct-script CLI. It asserts exit `1`,
empty stdout, no root members/final archive, a `course_package:` stderr
diagnostic, and no `Traceback`. Against the pre-fix code this test failed
locally with the exact same traceback shape main recorded
(`course_package.PackageError` escaping `__main__.main`).

**Minimal production fix:** the warm-schema branch in `run_package` now keeps
the module object and wraps only `warm_package.PackageError` into the current
dispatcher's `PackageError`:

```python
warm_package = _warm_package_module()
try:
    return warm_package.run_warm_package(...)
except warm_package.PackageError as error:
    raise PackageError(str(error))
```

`FileExistsError`/`OSError` are deliberately not caught or wrapped, no aliases
or global imports were added, and the cold path/API/constants are unchanged.
Validation is unchanged, not relaxed.

**New local results (win32, Python 3.12.7):**

```text
python -m unittest tests.test_course_warm_package.WarmPackageDirectCliTest
pre-fix: FAILED (1 test; stderr traceback, no 'course_package:' diagnostic)
post-fix: Ran 1 test ... OK

python -m unittest tests.test_course_warm_package
Ran 51 tests ... OK

python -m unittest tests.test_course_warm_package tests.test_course_benchmark \
  tests.test_course_fixture tests.test_course_package \
  tests.test_course_warm_benchmark
Ran 148 tests in 27.533s
OK          # no skips
```

## Limits

- These are local win32 (Python 3.12.7) results only. Main's Linux
  Python 3.12.12 run of the same 148 tests is the authoritative regression
  check; no Linux/GPU/CUDA claim is made here.
- All durations, records, and PNG/report metrics in these tests are synthetic
  arithmetic fixtures. No H20 measurement, actual warm artifact, accuracy, or
  performance claim is made or generated.
- The synthetic checkpoint exercises metadata/grammar/CRC binding only; the
  real C++ loader/model validation remains main's actual-data step.
- Root cold artifacts were deliberately not moved or regenerated; actual warm
  publication still requires main's real reviewed data and preservation step.
- Pillow remains the preexisting development-only dependency; nothing was
  installed. Missing Pillow would produce visible skips, as in the cold suite.

## NEEDS_CONTEXT

- GREEN complete locally: warm module + minimal dispatch; 51/51 warm,
  97/97 prior, 148/148 combined win32 PASS, no skips; CLI help exits 0.
- New direct-script RED/GREEN covers main's class-identity probe; dispatch
  now normalizes only warm `PackageError`, leaving FileExists/OSError and the
  cold path untouched. Validation was not relaxed.
- One earlier objective fixture fix: mode-mismatch subtests mutate benchmark
  JSON only; assertions/intent unchanged.
- No root artifact/ZIP/PNG, source, CMake, GPU, runtime, docs, or measurement
  changes; Pillow was already installed.
- Main next: Linux 148 + probe GREEN, independent review, then actual warm
  packaging and fresh extract/build/inference checks on the real model.
- Do not publish or describe synthetic fixture metrics as actual results.
