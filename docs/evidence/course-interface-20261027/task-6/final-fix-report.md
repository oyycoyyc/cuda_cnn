# Final-review fix wave — tests-first handoff (M1/M2/M3)

Status: NEEDS_CONTEXT — focused tests are ready and RED; production code is
deliberately unchanged in this turn.

Scope: only `tests/test_course_package.py` was edited. Root artifacts
(`report`, `result.png`, `course-submission.zip`, `evaluation.json`), model,
runtime, CUDA, CMake, helpers, whitelist, `max_ms`, and publication mechanics
were not touched. No SSH, installs, git staging/commits, or subagents.

## Tests added (owned synthetic fixtures only)

All four are in `CoursePackageTest` and use existing `_make_fixture` /
`_package` synthetic trees (no real metrics, no fake NN/image placeholder).

1. `test_report_timing_scope_enumerates_every_full_process_stage`
   - Packages `FAST_WALLS` and reads the actual generated root `report`.
   - Asserts case-insensitive presence of `startup`, `model/image loading`,
     `cuda setup`, `transfers`, `computation`, `validation`, `stdout/stderr`,
     `exit`, plus the literal sentence
     `GPU event substages are separate diagnostics, not a substitute for the
     full-process wall`.

2. `test_report_states_official_preprocessing_unconfirmed`
   - Reads the actual generated report and asserts (case-insensitive)
     `official course preprocessing remains unconfirmed`.

3. `test_image_reports_five_run_range_and_scope`
   - Packages `FAST_WALLS` (`min 85.3`, `max 91.5`), spy wraps the real
     `ImageDraw.ImageDraw.text`.
   - Asserts exactly one `Test Accuracy:` and one `Test Time:`; the unchanged
     median text `Test Time: 88.10 ms (median of 5 full-process runs)`;
     literal `5-run full-process range: 85.30 - 91.50 ms`; literal
     `Scope: startup through exit`; no `Test Time: 85.30` /
     `Test Time: 91.50`.
   - Opens the PNG and asserts every captured text bbox is inside the image
     (no clipping), including vertically.

4. `test_slow_image_reports_range_scope_and_failed_verdict`
   - Same spy with `SLOW_WALLS` (`min 148`, `max 152`) and `correct=9670`.
   - Asserts one accuracy / one Test Time; median
     `Test Time: 150.00 ms (median of 5 full-process runs)`; literal
     `5-run full-process range: 148.00 - 152.00 ms`; scope text; `FAILED`;
     no `ACHIEVED`; all text bboxes inside the image.

Support helper `record_image_text(drawn, boxes)` wraps the real
`ImageDraw.ImageDraw.text`, records each string, and stores
`draw.textbbox(...)` when available. It filters draw-only kwargs (`fill`) so
bounds match the real glyph box.

## RED evidence (this turn)

Command:

```
python -m unittest -v tests.test_course_package.CoursePackageTest.test_report_timing_scope_enumerates_every_full_process_stage tests.test_course_package.CoursePackageTest.test_report_states_official_preprocessing_unconfirmed tests.test_course_package.CoursePackageTest.test_image_reports_five_run_range_and_scope tests.test_course_package.CoursePackageTest.test_slow_image_reports_range_scope_and_failed_verdict
```

Result: `Ran 4 tests ... FAILED (failures=4)`, each failing on the missing
literal text (first missing item: `startup` for M1; the caveat for M3;
`5-run full-process range: 85.30 - 91.50 ms` for fast M2; the corresponding
slow range for slow M2). No fixture/setup error.

Full file check: `Ran 43 tests in 6.977s`, `FAILED (failures=4)` — the 39
pre-existing helper tests still pass and the 4 new tests are the only RED.

## Proposed GREEN production change (scripts/course_package.py only)

M1 — append one explicit stage-boundary line in `_report_text` (report is
still UTF-8, deterministic, no paths):

```
Timing scope: full-process wall from startup through exit, including
model/image loading, CUDA setup, transfers, computation, validation, and
stdout/stderr output; GPU event substages are separate diagnostics, not a
substitute for the full-process wall.
```

M3 — append one explicit caveat line (same wave):

```
Preprocessing caveat: official course preprocessing remains unconfirmed.
```

M2 — keep `metric_lines` byte-for-byte unchanged (single accuracy line and
single median `Test Time` line). Extend the renderer so the single metric
group also shows the observed five-run range and scope, computed from
`validated["walls"]` (never hardcoded real metrics):

```
5-run full-process range: {min(walls):.2f} - {max(walls):.2f} ms
Scope: startup through exit (full process)
```

Suggested signature: `render_result_png(accuracy, median_ms,
performance_target, min_ms, max_ms)` (or pass `walls`), with
`run_package` supplying `min(validated["walls"])` /
`max(validated["walls"])`. Increase canvas height from 320 to 420 and keep
generous spacing so the added lines and verdict cannot clip; all drawn
strings remain inside the image. The verdict color/label logic and the
`FAILED`/`ACHIEVED` text stay unchanged.

M4 and all other findings stay untouched (accepted debt), per ruling.

## Post-GREEN verification for main

- `python -m unittest -v tests.test_course_package` → expect 43 tests, 0
  failures, 0 skips (Pillow present).
- Re-run the broader helper set used by Task 6 (existing 66 + 4 new = 70
  helper tests).
- Main owns real remote RED/GREEN, preservation/regeneration of the actual
  root artifacts + ZIP, fresh build/protection checks, and the one scoped
  re-review. Implementer does not regenerate or overwrite the real artifacts.

## GREEN implementation (local, Windows/Python)

Minimal `scripts/course_package.py` edits only; the 4 new tests and all 39
existing package tests are unchanged.

- `render_result_png(..., min_ms, max_ms)` now draws the unchanged
  `metric_lines` group plus exactly one
  `5-run full-process range: MIN - MAX ms` line and one
  `Scope: startup through exit (full process)` line; canvas is 900x420.
- `run_package` supplies `min(validated["walls"])` and
  `max(validated["walls"])`; no actual metric is hardcoded.
- `_report_text` adds the M1 `Timing scope: ... GPU event substages are
  separate diagnostics, not a substitute for the full-process wall.` sentence
  and the M3 `Preprocessing caveat: official course preprocessing remains
  unconfirmed.` line. M4 `max_ms=` remains.

Local commands/results:

```
python -m py_compile scripts/course_package.py tests/test_course_package.py
-> OK

python -m unittest -v <the four M1/M2/M3 tests>
-> Ran 4 tests ... OK

python -m unittest tests.test_course_package
-> Ran 43 tests in 7.139s ... OK

python -m unittest tests.test_course_fixture tests.test_course_benchmark
-> Ran 27 tests in 3.779s ... OK
```

Local helper total: 43 + 27 = 70 tests, all OK (Pillow 10.4 present; no
skips observed in the package file). Limits: this is Windows/Python local
verification only; Linux RED/GREEN, actual republication at a fresh path,
real PNG visual inspection, archive/build/roundtrip/hash checks and the scoped
re-review remain main-owned. No actual root artifacts or ZIP were generated or
touched; synthetic temp fixtures only; no SSH, git staging/commits, subagents,
or installs.
