# Task 2 report: tests-first handoff for the warm collector

**Status:** GREEN implementation plus timeout-retention fix, pending main's
rerun of the actual Linux 97-helper suite. Main's first actual Linux 97 run was
96 PASS and exactly one FAIL (`test_timeout_preserves_partial_output_and_fails`):
the child closed both pipes before sleeping, `subprocess.run` timed out after
EOF, and the exception's stdout/stderr were `None`, so the already-read partial
stderr was lost. No actual H20 W5/R101 series was started. The process lifecycle
now uses one `Popen` child with `communicate(timeout)`, then `kill()` plus a
second `communicate()` on timeout; tests are unchanged. Local win32 post-fix:
27/27 warm, 70/70 prior helpers, 97/97 combined. Supplementary WSL Ubuntu
(Python 3.14.4) validates the targeted timeout test and all 27 warm tests;
main's Linux 3.12.12 97-helper rerun remains authoritative and pending. Actual
H20 remains main-owned and unmeasured. No Task
1/CMake/GPU/old-helper/SSH/install/git-state changes were made.

## Protocol read from the reviewed Task 1 tool

Read from the reviewed snapshot
`.superpowers/sdd/2026-10-27-warm-full-request-benchmark/task-1-review-snapshot/course/tools/warm_benchmark.cu`
(348 lines) rather than guessed.

- CLI is exactly `course_warm_benchmark images.f32 N C H W K warmups runs`
  (`argc == 9`, eight positional arguments after the executable).
- stderr emits exactly W warmup records, then R measured records, then one
  completion record. There is no separate record stream per round.
- Round record:
  `event=course_warm_iteration phase=<warmup|measure> iteration=<one-based within phase> samples=<N> wall_ms=<positive finite> predictions_identical=1`
- Completion:
  `event=course_warm_complete status=pass warmups=<W> runs=<R> samples=<N> batch=<B> input_mode=<unit|byte|normalized>`
- stdout is exactly one final N-line vector of canonical decimal class
  integers in `[0, 10)`, published only after every round passed.
- One warmed process, one final stdout vector; the tool validates each round's
  readback against the first warmup vector internally. The collector must not
  invent per-round stdout copies.

## Pinned collector API

### Constants

```python
WARM_BENCHMARK_SCHEMA = "course-warm-benchmark-v1"
TIMING_SCOPE = "warm_full_request_h2d_prepare_forward_argmax_d2h_validation"
ACCURACY_THRESHOLD = 0.96
DEFAULT_BATCH = 1024
DEFAULT_WARMUPS = 5
DEFAULT_RUNS = 101
FULL_REQUEST_LIMIT_MS = 100.0
PROCESS_TIMEOUT_SECONDS = 600
MODEL_SOURCES = ("legacy", "course_train")
INPUT_MODES = ("unit", "byte", "normalized")
IMAGE_PIXELS = 28 * 28
CLASS_COUNT = 10
SHAPE = (1, 28, 28, 10)
MAX_ROUND_COUNT = 100000
```

`PROCESS_TIMEOUT_SECONDS` must be read at call time (the timeout test patches
the module attribute); it must not be captured as a function default.

### Pure parser (Task 3 may reuse)

```python
parse_tool_output(stdout, stderr, returncode, count, batch, warmups, runs,
                  input_mode)
```

Returns a dict with exactly these keys:

- `warmup_records`: list of exactly `warmups` records
- `measured_records`: list of exactly `runs` records
- `predictions`: list of `count` ints
- `completion`: dict with exactly
  `status`, `warmups`, `runs`, `samples`, `batch`, `input_mode`

Each round record has exactly:
`iteration`, `samples`, `wall_ms`, `predictions_identical`.
`predictions_identical` is the bool `True`; the parser rejects `0` and any
non-`1` value. The parser accepts bounded requests (for example W1/R1/N1), does
not enforce N=10000 or R>=101, and does not require the packager's final gates.

The parser raises `ValueError` for any nonzero process status, missing/extra/
duplicate/misordered round or completion line, wrong phase or one-based
iteration, wrong sample count, extra/missing/duplicate/reordered fields, a
non-positive or non-finite `wall_ms`, a completion count/batch/mode/status
mismatch, or stdout that is not exactly N canonical class lines. Blank protocol
lines and trailing junk are rejected. Token key order is pinned to the reviewed
tool's order.

### Collector

```python
run_benchmark(warm_exe, infer_exe, images, labels, model_dir, output_json,
              count, batch=DEFAULT_BATCH, warmups=DEFAULT_WARMUPS,
              runs=DEFAULT_RUNS, model_source="course_train")
```

Returns the result dict described below and writes it exclusively to
`output_json`. Preflight errors (`ValueError`, `FileExistsError`, `OSError`)
are raised **before launch** and create no result JSON.

Preflight must validate positive counts (rejecting bools), `warmups`/`runs` in
`[1, 100000]`, `model_source`, labels size and `[0, 10)` values, exact images
size `count * 784 * 4`, and the persisted `course.meta` mode/shape. It must
refuse an output that exists or whose normalized real path aliases any of the
six protected identities (warm tool, official infer, images, labels, model,
meta) before launching.

Exactly one child process is permitted. The collector launches it with a single
`subprocess.Popen`, with argv:

```text
[warm_exe, images, str(count), "1", "28", "28", "10", str(warmups), str(runs)]
```

The child environment removes every inherited `COURSE_*` variable and sets only
`COURSE_MODEL_DIR=<model_dir>` and `COURSE_INFER_BATCH=<batch>`. Labels are
never placed in argv or the environment; profile/training/mode/seed variables
cannot leak. The collector reads labels itself only for accuracy.

Hash six files with SHA-256 both before and after the process:
`binary`, `official_infer`, `model`, `meta`, `images`, `labels`. A post-run
read failure stores `null` for that entry. `hashes_stable` is true only when the
entire after-map equals the before-map.

`process_wall_ms` is the actual monotonic wall around the whole single process;
it is never used for inner statistics. Warmup and measured inner durations are
kept exactly as parsed. Summary arithmetic uses **measured records only**:
`statistics.mean`, `statistics.median`, `min`, `max`; warmups are excluded and no
measured outlier is discarded.

## Pinned JSON shape

Top-level keys are exactly those in the controller context:

```text
schema, timing_scope, executable, official_infer, images_path, labels_path,
model_dir, model_source, n, batch, input_mode, warmups, runs, sha256,
sha256_after, hashes_stable, returncode, process_wall_ms, stdout, stderr,
warmup_records, measured_records, predictions_identical_all, status, errors,
summary
```

`sha256` and `sha256_after` each have exactly:
`binary`, `official_infer`, `model`, `meta`, `images`, `labels`.

`summary` has exactly:
`mean_ms`, `median_ms`, `min_ms`, `max_ms`, `correct`, `accuracy`,
`accuracy_pass`, `mean_below_100ms`, `median_below_100ms`.

`status` describes **collection integrity**, not the performance gate. A
closed zero-exit process with the complete protocol, canonical one-vector
stdout, stable hashes, and parsed records gets `status="pass"` and
`errors=[]`, even when `accuracy_pass`, `mean_below_100ms`, or
`median_below_100ms` is false. Those three gates stay independently visible.
Summary metrics remain present for such a valid measurement.

CLI acceptance is stricter: `main(arguments=None)` returns `0` only when
`status == "pass"` **and** `accuracy_pass` **and** `mean_below_100ms` **and**
`median_below_100ms`; otherwise it returns `1`. A valid but slow or inaccurate
measurement still retains its full JSON. Preflight errors print a diagnostic and
return `1` without a result file.

Any post-launch failure (spawn error, timeout, nonzero exit, incomplete or
malformed protocol, non-canonical stdout, unstable hash) produces
`status="fail"`, non-empty `errors`, `predictions_identical_all=false`,
`mean_ms`/`median_ms`/`min_ms`/`max_ms` null, `correct=0`, `accuracy=0.0`, and
all three gates false. Raw `stdout`, `stderr`, `returncode`, and
`process_wall_ms` are retained in the exclusive failure JSON. Failure
`warmup_records`/`measured_records` may be empty; raw stderr still carries any
partial records, and partial records must never be presented as a successful
series. A timeout stores `returncode=null`.

### CLI

```text
course_warm_benchmark warm_exe infer_exe images labels model_dir output_json \
  --count N [--batch 1024] [--warmups 5] [--runs 101] \
  --model-source course_train
```

Six positional paths. `--count` and `--model-source` are required; the other
three flags default to 1024/5/101.

## Test coverage in `tests/test_course_warm_benchmark.py`

- `ParseToolOutputTest`: pinned signature/constants; exact W2/R3/N3 acceptance
  with all records retained; bounded W1/R1/N1 acceptance; rejection of nonzero
  exit, missing/extra/duplicate/misordered/phase-wrong records, every bad round
  field, non-positive/non-finite times, false invariance, bad completions, bad
  canonical stdout, and bad request arguments.
- `WarmCollectorTest`: exact 26-key schema and six hashes before/after; one
  process; exact eight-position argv; explicit model/batch env; hostile
  `COURSE_PROFILE`/`COURSE_INPUT_MODE`/`COURSE_EPOCHS`/`COURSE_SEED`/
  `COURSE_LABELS` isolation; exact one final stdout vector; warmup retention;
  literal mean/median/range `70/3`, `20`, `10`, `40` from measured
  `[10, 20, 40]` with warmups `[1000, 2000]`; accuracy mismatch measured but
  rejected by CLI; slow outlier `300` retained with mean `110`/median `20`;
  CLI success `0`; out-of-range/non-canonical stdout failure; extra record and
  missing completion; process failure retained raw stdout/stderr/exit/wall;
  real narrow-boundary timeout; post-run images mutation/hash failure; mode
  mismatch.
- `WarmPreflightTest`: existing output and six protected-path aliases refused
  before launch; bad images size, labels size/value, missing inputs, bad model
  mode, bad counts/source refused before launch with no JSON and no invocation;
  CLI preflight error returns `1` without launching.

Synthetic mock durations are used only as literal arithmetic inputs. They make
no GPU-performance claim and are not presented as measured H20 data.

## Commands and results (win32, Python 3.12.7)

```text
python -m py_compile tests/test_course_warm_benchmark.py
py_compile PASS

python -m unittest tests.test_course_warm_benchmark
ImportError: cannot import name 'course_warm_benchmark' from 'scripts'
  (unknown location)
FAILED (errors=1)
```

The second command is the expected RED because the proposed module does not
exist. On Linux this is the same named missing-module failure main owns.

To verify the tests themselves, a throwaway reference implementation was placed
outside the workspace in `%TEMP%\opencode\warm-tests-check` and the test file was
copied there. Result:

```text
python -m unittest tests.test_course_warm_benchmark
Ran 27 tests in 2.130s
OK
```

That reference was not added to the repository, was not used for any Linux or
GPU claim, and is not part of this handoff. It was deleted after the check.

At the initial tests-only handoff, `git status --porcelain` showed only the new
test file for this task (the report is under `.superpowers/`). The GREEN work
adds only `scripts/course_warm_benchmark.py`; no Task 1 file, CMake, model,
data, old helper, or results document was edited.

## GREEN implementation and verification (this session)

**Added:** `scripts/course_warm_benchmark.py` (stdlib only, importable as
`scripts.course_warm_benchmark`, executable as a direct script, and usable as
`python -S -m scripts.course_warm_benchmark`).

**Objective test fixture bug (found during GREEN):** on Windows the fixture wrote
`course.meta` through text mode, producing CRLF. The strict metadata grammar
and the real packager require four LF-terminated lines, so the fixture now opens
with `newline="\n"`. This is a platform fixture fix only: 27 tests remain, no
assertion or expected behavior changed, and Linux output is unaffected.

**Verification commands and exact local results (win32, Python 3.12.7):**

```text
python -m py_compile scripts/course_warm_benchmark.py tests/test_course_warm_benchmark.py
exit 0

python -m unittest tests.test_course_warm_benchmark -v
Ran 27 tests in 2.126s
OK

python -m unittest tests.test_course_benchmark tests.test_course_fixture tests.test_course_package
Ran 70 tests in 11.796s
OK

python -m unittest tests.test_course_warm_benchmark tests.test_course_benchmark \
  tests.test_course_fixture tests.test_course_package
Ran 97 tests in 13.929s
OK

python -S -m scripts.course_warm_benchmark --help
exit 0

python -S scripts/course_warm_benchmark.py --help
exit 0
```

**Implementation notes:**

- Exactly one `subprocess.Popen` child; argv is the eight-position tool request
  (`images N C H W K warmups runs`) with `N=count`, `C H W K=1 28 28 10`.
  Success uses one `communicate(timeout)`; timeout uses `kill()` plus a second
  `communicate()` on the same child to drain raw output.
- The child environment removes every inherited `COURSE_*`, then sets only
  `COURSE_MODEL_DIR` and `COURSE_INFER_BATCH`; labels never enter argv/env.
- `PROCESS_TIMEOUT_SECONDS` (600) is read when the call executes, so main can
  inspect the production boundary and tests can patch a narrow boundary.
- Six SHA-256 identities (`binary`, `official_infer`, `model`, `meta`,
  `images`, `labels`) are captured before and after; missing post-run files
  become `null` and force `hashes_stable=false`.
- One final stdout vector is parsed; no per-round stdout streams are invented.
  Raw `stdout`, `stderr`, `returncode`, and actual monotonic `process_wall_ms`
  are retained separately.
- The parser enforces classic single-space records, exact field order/count,
  warmup/measure phase and one-based indices, full N, positive finite inner
  walls, `predictions_identical=1`, the completion status/count/batch/mode, and
  exactly N canonical class lines.
- Statistics use measured records only (`mean`, `median`, `min`, `max`);
  warmups are excluded and every measured outlier remains part of the range.
- Metadata preflight validates the exact four-line header/mode/shape/CRC
  spelling but deliberately does not decode the C++ checkpoint or recompute
  CRC32; the real tool and later packager own those checks. Fixture CRC
  `00000000` is metadata-grammar coverage, not a real-checkpoint claim.
- Result JSON is written exclusively with `allow_nan=False`; serialization
  happens before opening the target, and rollback only removes the file this
  call created. All 26 top-level keys, six hash keys, and nine summary keys are
  present on success and failure.
- Collection failure (nonzero exit, timeout, spawn error, partial/malformed
  protocol, bad stdout, or hash mutation) retains decoded raw output/exit/wall,
  sets `status="fail"`, null timings, zero accuracy, all gates false, and
  non-empty errors. A valid measurement can still have `status="pass"` while
  the CLI returns `1` for accuracy or either 100 ms gate.
- No H20, Linux, accuracy, or timing metric is claimed here; the mock durations
  are synthetic arithmetic only.

## Timeout retention fix after main's first Linux 97 run

**Observed main result:** The first actual Linux 97-helper run was 96 PASS with
exactly one failure, `test_timeout_preserves_partial_output_and_fails`:

```text
AssertionError: 'event=course_warm_iteration' not found in ''
Ran 97 tests in 11.628s
FAILED (failures=1)
```

The `close_streams=True` test mock writes and flushes a partial stderr record,
closes both stdout and stderr, then sleeps for three seconds while the collector
uses a one-second patched timeout. On POSIX Python 3.12, `subprocess.run(...,
timeout=...)` raised `TimeoutExpired` after the pipes had reached EOF but while
the child was still alive; the exception carried `stdout=None` and
`stderr=None`, so the already-read partial stderr was discarded. Main reproduced
this independently in `task-2-timeout-root-cause.txt`:

```text
run_timeout:            stdout="None"  stderr="None"
popen_kill_communicate: stdout="b''"   stderr="b'event=course_warm_iteration proof=partial\n'"
                        returncode=-9
```

CPython's documented/implemented `communicate()` retry path retains bytes read
before the timeout; trusting only the exception payload or waiting again without
killing does not. No actual W5/R101 GPU series was started.

**Binding fix:** "exactly one `subprocess.run`" was an implementation detail;
the real contract is exactly one child process plus raw-output retention. The
collector was changed to:

1. launch exactly one `subprocess.Popen` child with pipes;
2. on success, call `communicate(timeout=PROCESS_TIMEOUT_SECONDS)` once and use
   its returned stdout/stderr plus the actual `child.returncode`;
3. on `TimeoutExpired`, call `child.kill()` then `child.communicate()` again on
   the same child to drain the complete buffered output;
4. never concatenate the `TimeoutExpired` partial payload with the drain result
   (avoids duplicate bytes);
5. compute `process_wall_ms` after the drain, so cleanup through child exit is
   included;
6. keep the same timeout failure status, null timings, false gates, non-empty
   timeout error, and exclusive failure JSON. The post-kill return code is
   `child.returncode` (normally `-9` on POSIX; the tests accept an int or
   `None`, and no substitute code is invented).

No test was changed for this fix, including the `close_streams=True` timeout
reproduction. Timeout remains the dynamically-read `PROCESS_TIMEOUT_SECONDS`
(600 in production); argv/env, six before/after hashes, exclusive JSON,
canonical parser, and measured-only statistics are unchanged.

**Post-fix local checks (win32, Python 3.12.7):**

```text
python -m py_compile scripts/course_warm_benchmark.py
exit 0

python -m unittest tests.test_course_warm_benchmark -v
Ran 27 tests in 2.129s
OK

python -m unittest tests.test_course_benchmark tests.test_course_fixture tests.test_course_package
Ran 70 tests in 11.762s
OK

python -m unittest tests.test_course_warm_benchmark tests.test_course_benchmark \
  tests.test_course_fixture tests.test_course_package
Ran 97 tests in 14.109s
OK

python -S -m scripts.course_warm_benchmark --help
exit 0

python -S scripts/course_warm_benchmark.py --help
exit 0
```

These are win32 results only. Main's POSIX run is the authoritative regression
check for the timeout path; the first Linux 97 run above remains the recorded
failure and the new Linux GREEN is pending.

**Supplementary POSIX check (WSL Ubuntu, Python 3.14.4; not main's 3.12.12):**

```text
python3 -m unittest tests.test_course_warm_benchmark.WarmCollectorTest.test_timeout_preserves_partial_output_and_fails -v
Ran 1 test in 1.002s
OK

python3 -m unittest tests.test_course_warm_benchmark
Ran 27 tests in 1.158s
OK
```

This exercises the kill-plus-drain branch on a real POSIX Python and confirms
the `close_streams=True` partial stderr is retained. It does not replace main's
Linux 97-helper rerun and is not an H20/GPU claim.

## Limits

- Main already verified the Linux named-missing-module RED (`errors=1`, no
  feature tests ran) and recorded the first Linux 97 run as 96 PASS/1 timeout
  FAIL. Main owns the post-fix Linux 97 GREEN run, plus the actual
  H20 N=10000/B=1024/W=5/R=101 measurement and independent review.
- No Linux execution, CUDA build, GPU run, official-infer execution, or actual
  accuracy/timing number is claimed here.
- The timeout test uses a one-second patched boundary with a harmless sleeping
  mock; production still requires the pinned 600-second constant.
- The tests pin exact names and contracts above. If main deliberately chooses
  different public names, update both the report and tests rather than silently
  diverging.

## GREEN handoff

- GREEN added: `scripts/course_warm_benchmark.py`; 27 warm tests kept, with one
  objective LF fixture fix and no assertion/count change.
- Timeout fix: one `Popen` child; success uses one `communicate(timeout)`;
  timeout uses `kill()` then a second `communicate()` on the same child to
  drain full raw output without concatenating exception duplicates.
- Main's first Linux 97 run was 96 PASS/1 timeoutretention FAIL; no GPU series
  started. Post-fix Linux 97 GREEN is still pending.
- Local win32 post-fix: 27/27 warm, 70/70 prior helpers, 97/97 combined;
  `py_compile` PASS; `-S` module and direct-script CLI help exit 0.
- Exact 26-key schema, six before/after hashes, argv/env, raw-output failure
  JSON, parser, and CLI gate semantics are unchanged as pinned above.
- No Linux/GPU/actual W5R101 claim; synthetic durations are arithmetic only.
- Main next: post-fix Linux 97-test GREEN, then actual H20 W5/R101/hash/raw/
  original-label audit and independent review. Do not publish pending results.
