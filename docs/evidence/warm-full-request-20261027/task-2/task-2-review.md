# Task 2 review — warm collector, raw schema, actual 101-round series

**Scope:** supplied scoped diff `61c9e14 -> current uncommitted`, 3 files /
1886 lines: new `scripts/course_warm_benchmark.py` (568 lines), new
`tests/test_course_warm_benchmark.py` (1232 lines), updated
`docs/course-warm-benchmark-results.md` (66 lines). No Task 1, CMake, GPU,
official-infer, or cold-helper change is in the diff. Read-only review; no
suite rerun, no source edit, no git command.

## Verdicts

- **Spec compliance: Approved**
- **Task quality: Approved**

**Critical:** none. **Important:** none. Five minors below; none blocks the
gate, the actual series, or Task 3 reuse.

## Verification actually performed

- Read the brief, controller context (26-key/schema/parser pins), report,
  `controller-task-2-verification.md`, and the full 1886-line diff, then
  inspected the live new files.
- Independently reparsed the stored actual series
  `task-2-course-default20-unit-w5-r101.json`: exactly 26 top-level keys;
  6/6 `sha256` and 6/6 `sha256_after` keys equal; 5 warmup + 101 measured
  records with iterations 1..W and 1..101; every sample 10000; every
  `predictions_identical` true; all 106 inner walls positive finite; 10000
  ASCII-canonical `[0,10)` stdout lines plus one final newline; 107 stderr
  records (5 warmup, 101 measure, 1 completion); `status=pass`, `errors=[]`,
  `returncode=0`.
- Recomputed mean/min/max/median from the 101 measured walls and got exactly
  the stored and audited values: mean `67.82692348514851` (= sum/101),
  median `57.752155`, min `7.550677`, max `638.715843`. Warmup walls and
  `process_wall_ms=9004.474382847548` are separate; warmups are excluded;
  all 106 inner walls sum to `7355.46637`, which is below the process wall.
  The `638.715843` outlier is present in `measured_records` and in the mean.
- Did **not** rerun the 27/70/97 suites (review instruction). The final Linux
  97 PASS / no skips is credited to `controller-task-2-verification.md:8` and
  `task-2-fixed-green.txt`; win32 results remain the implementer's.
- Did **not** recompute the 9911/0.9911 accuracy: this worktree has no
  images/labels fixtures. That comparison remains credited to main's
  `task-2-independent-audit.json`; the stored JSON is internally consistent
  (9911/10000 = 0.9911 = stored accuracy, gates true).

## Strengths (before findings)

1. **Pinned schema is exact in code and in the actual artifact.**
   `_result_record` (`scripts/course_warm_benchmark.py:338-371`) emits the 26
   pinned keys; six hash keys and nine summary keys are enforced by tests
   (`tests/test_course_warm_benchmark.py:29-69`) and confirmed in the actual
   JSON.
2. **One process, correct tool contract, labels isolated.** One `Popen`
   (`scripts/course_warm_benchmark.py:446-448`); argv is the eight positionals
   `images N 1 28 28 10 W R` (`:429-432`); every inherited `COURSE_*` is
   removed and only `COURSE_MODEL_DIR`/`COURSE_INFER_BATCH` are set
   (`:433-438`). Labels are read by the host collector only for accuracy and
   never enter argv/env (`:416`). The success test verifies the real child's
   recorded argv/env and exactly one invocation (`tests/...:619`, `:726-738`).
3. **Protocol and parser strictness match the reviewed C++ tool.** Warmup then
   measured phases, one-based indices, full N samples, positive finite inner
   walls, `predictions_identical=1`, exactly one final completion with
   status/count/batch/mode binding, and exactly N canonical stdout class lines
   (`scripts/...:97-197`). Extra/missing/duplicate/misordered/blank records
   are rejected (`tests/...:433-555`).
4. **Failure retention and exclusive outputs are real.** Timeout uses
   `kill()` then a second `communicate()` on the same child and retains the
   already-buffered partial stderr (`scripts/...:457-470`); no exception
   payload is concatenated. Failures keep raw stdout/stderr/exit/wall, set
   null timings and false gates, and are written with `O_EXCL`
   (`:285-302`, `:509-526`). Existing output and six protected-path aliases
   are refused before launch (`:311-319`, `tests/...:1038-1089`).
5. **Statistical separation is correct.** Summary uses measured records only
   via `statistics.mean/median/min/max` (`scripts/...:376-394`); warmups and
   process wall never enter it. Tests pin `70/3`, `20`, `10`, `40` from
   measured `[10,20,40]` with warmups `[1000,2000]`, and an outlier test pins
   mean `110` / median `20` with `300` retained (`tests/...:619`, `:783`).
   The actual JSON independently reproduces all of this.
6. **Actual-data document fidelity.** Every number and identity in
   `docs/course-warm-benchmark-results.md:15-57` matches the stored JSON
   (mean, median, min, max, 9911/10000, warmup list, process wall, tool/infer/
   model hashes) and the controller verification. The doc is explicit that
   mean and median pass while individual rounds do not all pass, that
   `638.715843 ms` is retained, that warmups and process wall are separate,
   that the timed tool is the development tool (not official infer or a cold
   six-argument subprocess), and that unit input is a project assumption.
7. **Honest tool/model validity boundary.** The collector validates metadata
   grammar, mode, shape, and CRC spelling but not checkpoint contents
   (`scripts/...:238-272`); synthetic `checkpoint_crc32=00000000` is test
   grammar only. The C++ loader and the Task 3 packager own full checkpoint/
   training CRC. The report/doc do not turn the mock fixture into a GPU claim.

## Findings

### Critical
None.

### Important
None.

### Minor

1. **Only `TimeoutExpired` is handled around `communicate()`.**
   `scripts/course_warm_benchmark.py:454-475`: any other exception from
   `communicate()` (e.g. an `OSError` while draining) escapes
   `run_benchmark`, so no exclusive failure JSON is written and no
   `returncode`/raw output is retained. All enumerated failure classes
   (spawn error, timeout, nonzero exit, malformed/partial protocol, bad
   stdout, hash instability) are covered; this is a defensive gap only.
2. **Timeout drain is not itself bounded.**
   `scripts/course_warm_benchmark.py:463-467`: a `kill()` `OSError` is
   swallowed, and the following `communicate()` has no timeout. If the kill
   genuinely failed, the drain could block beyond the pinned 600 s bound.
   Normal self-child kill and already-exited-child cases are safe.
3. **"Canonical" parsing is slightly looser than ASCII decimal.**
   `scripts/course_warm_benchmark.py:52-57` uses `str.isdigit()` + `int()`, so
   non-ASCII digits such as `"١"` would pass; `:61-70` accepts exponent/
   underscore float spellings such as `1e3`/`1_0`. The reviewed tool emits
   plain ASCII decimals, and the actual 10000-line stdout contains only
   `[0-9]`, so there is no impact on this result.
4. **Two failure branches are implemented but untested.**
   No test exercises the `Popen` `OSError` branch
   (`scripts/...:449-452`) or a post-run unreadable protected file yielding
   `sha256_after[name]=None` (`:485-488`). Current failure tests cover nonzero
   exit, timeout, mutation, and malformed protocol
   (`tests/...:919`, `:949`, `:977`, `:841-918`).
5. **Handoff report is stale on the Linux GREEN.**
   `task-2-report.md:3-15` and `:440` still say the post-fix Linux 97 run is
   pending, while `controller-task-2-verification.md:8` and
   `docs/course-warm-benchmark-results.md:56-57` record 97 PASS / no skips
   (70 prior + 27 warm). The final doc is correct; only the older report text
   is inconsistent. This is a documentation-freshness issue, not a code or
   measurement defect.

## Scope notes (explicitly not numbered defects)

- **Checkpoint CRC / model validity** is out of this collector's scope: it
  checks the four-line metadata grammar, persisted mode, shape, and CRC
  spelling only. The fixture's `00000000` CRC is synthetic; the C++ loader
  and Task 3 packager own full checkpoint/training CRC and accuracy
  provenance.
- **Cold root report/PNG/ZIP are unchanged and still show cold scope.** Task 3
  owns strict warm packaging and truthful regeneration after this gate; this
  is not a Task 2 defect.
- **`model_source` is recorded but not bound to model provenance** by this
  collector; the actual run's `course_train` is consistent with the model dir
  and controller evidence, and Task 3's strict packager should bind the
  training/CRC provenance.
- **Mean+median `<100 ms` is not an all-round gate.** The actual doc and this
  review agree: mean and median pass, `638.715843 ms` remains in the range
  and the mean.
- **Original-label accuracy** is not re-derived here because this worktree
  lacks the fixtures. Main's independent audit (`correct=9911`,
  `accuracy=0.9911`, reference match) stands as the accuracy evidence.

## Bottom line

The collector, parser, tests, failure retention, exclusive outputs, hash
binding, statistical separation, the actual H20 series (9004.474383 ms
process wall; 67.82692348514851 ms mean / 57.752155 ms median over 101
samples), and results document meet the Task 2 brief and controller
contract. Actual measurement is a single warmed process with one final
10000-line vector, not
101 fake stdout vectors; mean and median both pass separately from the cold
wall; the slow outlier is retained. Approve the gate; the five minors are
optional hardening/cleanup before or alongside Task 3, and Task 3 still owns
the strict packager, checkpoint CRC/training provenance, and warm-artifact
regeneration.
