# Task 6 scoped review: package/acceptance helper and artifact contract

Scope: read-only review of `.superpowers/sdd/2026-10-27-course-gpu-interface/task-6-review.diff`
(1985 lines, four files vs `66b3018`: `scripts/course_package.py` 928 new lines,
`tests/test_course_package.py` 920 new lines / 39 test methods, and the
`docs/course-interface.md` / `docs/course-interface-results.md` updates).
I checked the actual on-disk files (`scripts/course_package.py` 928 lines,
`tests/test_course_package.py` 920 lines / 39 `def test_` methods) against the
diff; no hunk is cut mid-function. I did not re-run tests, run git, or use the
remote host. Controller evidence (`controller-task-6-verification.md`,
`task-6-report.md`, `task-6-all-helpers-green.txt`,
`task-6-final-independent-acceptance.json`, `task-6-default20-summary.json`)
was cross-checked against the diff and docs, not trusted blindly: the green log
really shows `Ran 66 tests ... OK` with 0 skips and 39 `test_course_package`
entries; the acceptance JSON really carries 5 walls, 9911 correct, identical
prediction SHA256, CRC `f473014f`, best epoch 15, 20 completed epochs; the docs
table reproduces those exact numbers and states 100 ms FAILED.

## Strengths

- Provenance chain is genuinely bound, not asserted: training completion must be
  exactly one `event=course_train` line with `status=pass`, `epochs=20`,
  supported `input_mode` and 8-hex CRC (`scripts/course_package.py:300-337`);
  that CRC must equal `course.meta` and a fresh `zlib.crc32(lenet.bin)`
  (`:437-445`); the four meta lines are parsed exactly per `course/model_io.h`
  (`:341-370`, compare `course/model_io.cpp:215-216`); and all five recorded
  SHA256 values are recomputed from the current binary/model/meta/images/labels
  (`:467-485`). A retrain or rebuild at the recorded paths is correctly rejected
  as stale evidence.
- Raw-data truth: predictions are re-parsed strictly (single canonical digit per
  line, exactly N lines, `:373-387`) and `correct` is recomputed against the
  actual little-endian int32 labels (`:456-466`, `:524-535`), then required to be
  identical across runs and equal to `summary.correct`/`summary.accuracy`
  (`:535-570`). `median_ms`, `max_ms` and `all_wall_below_100ms` are recomputed
  from the five walls (`:574-593`), so the six summary fields cannot be
  fabricated independently of the run records.
- Honest failure handling: performance verdict is `achieved` only when every
  wall is `<100.0` ms (`:598-613`, `:771-782`); a slow but accurate result is
  still packaged and labelled FAILED in report, not rewritten or suppressed. The
  slow-result test also proves the benchmark file digest is unchanged
  (`tests/test_course_package.py:1449-1470`). Accuracy gate is exact
  (`>=9600/10000`, `:544-554`).
- Whitelist is the real default CMake graph: I traced the transitive includes of
  `train`/`infer` against `CMakeLists.txt:47-123`; the 34 pinned files cover every
  library, source and header, and nothing from `course/tests`, `course/tools`,
  scripts, data, weights, build, diagnostics, evidence, Dockerfile or docs. Both
  programs need no `course/tests`/`course/tools` sources.
- Publication safety: pre-existing root artifacts/archive are rejected with
  `FileExistsError` before any write via `lexists` (`:842-856`); root artifacts
  and ZIP are created with `O_CREAT|O_EXCL` and fsynced (`:814-840`); a failure
  after this invocation created files removes only those files (`:858-876`).
  Archive members are sorted, forward-slash, non-directory, duplicate-free,
  fixed-date 0644 entries (`:784-811`), and caps are checked before publication
  (`:176-192`, `:187-193` call `check_image_limits(len(png), w*h)` at `:728`).
  Decimal caps are the conservative binding ruling; `check_image_limits` has the
  required `(bytes, pixels)` API.
- Renderer/report meet the artifact contract: lazy Pillow import with a clear
  "development ... install" error and no auto-install (`:695-702`); exactly one
  `Test Accuracy:` and one `Test Time: ... ms (median of 5 full-process runs)`
  line (`:687-691`); report lists every run, median, max, configuration,
  completion line, preprocessing assumption, all five hashes + CRC, PASS accuracy
  and ACHIEVED/FAILED performance, and the unresolved leaderboard timing
  (`:732-781`).
- Test quality beyond happy path: 39 methods / subtests cover exact JSON bytes,
  member set, stale binary/model/meta, metadata CRC recomputed, duplicate and
  non-finite JSON, profiled flag and profile evidence in stderr, missing/short
  runs, nonzero return, forged per-run correct, prediction/label disagreement,
  six fabricated summary fields, outside-root symlink, unsafe/alias member names,
  cap boundaries, existing archive/root artifact preservation, missing Pillow
  with no writes, and CLI exit 2 vs 1. `_assert_no_artifacts` is used across
  rejection cases, which is the "no early image/package" proof.
- Docs are truthful: `docs/course-interface-results.md:195-246` records actual
  default-20 training (594.740494 s), best epoch 15, CRC `f473014f`, five actual
  fresh unprofiled runs (9911/10000, identical predictions), median
  1657.407840 ms, max 1684.153743 ms, accuracy PASS and 100 ms FAILED, plus the
  still-unconfirmed official preprocessing/leaderboard timing and the open
  package/extract gates. The doc median/max match
  `task-6-final-independent-acceptance.json` exactly, and the helper will print
  `Test Time: 1657.41 ms`, matching the doc requirement.

## Findings

### Critical

None.

### Important

None. The binding requirements I probed (current-hash binding, completion/mode/CRC
consistency, strict JSON/schema/raw-prediction/summary/wall checks, profile
isolation, whitelist closure, image/report contract, exclusive publication) are
implemented in code, not just in prose, and the failure paths I could exercise by
reading are correct.

### Minor (non-blocking)

1. No test injects a publication failure after the first root artifact is written:
   `_publish` rollback (`scripts/course_package.py:858-876`) is correct by
   inspection but unexercised, and `tests/test_course_package.py` has no
   `_publish`/`_write_exclusive`/rollback case. Adding a mocked failure at the
   archive write (assert report/evaluation/PNG removed, no preexisting file
   touched) would close the one untested safety path from the review checklist.
2. The slow (>100 ms) case is asserted FAILED only in the report
   (`tests/test_course_package.py:1449-1470`); the image's FAILED verdict
   (`scripts/course_package.py:710-734`) is not directly asserted for a slow
   input (the image-text test uses the fast fixture). Code is correct; coverage
   only.
3. `wall_ms` has only a `0.0` lower bound (`scripts/course_package.py:508`), so a
   nonphysical `0.0` wall set would yield `achieved`. This is not forgery-proofing
   (the binding ruling accepts that hash/CRC evidence is consistency, not
   authentication) and cannot be fully prevented, but a small positive floor
   would catch obviously impossible data.
4. The report labels the input contract "Preprocessing assumption" and describes
   the persisted transform (`:670-685`, `:743-744`) but does not itself repeat
   that official course preprocessing is unconfirmed; that caveat lives only in
   the results doc (`docs/course-interface-results.md:246`). Consider adding one
   sentence to the generated report for standalone honesty.
5. Stale doc status line: `docs/course-interface-results.md:3` still says "27
   stdlib tests", while Task 6 adds 39 and the verified total is 66 (no skips).
6. `_parse_predictions` uses `str.splitlines()` (`scripts/course_package.py:378`),
   which also accepts CRLF and exotic Unicode line separators. This matches
   `scripts/course_benchmark.py:119` and does not affect correctness of the
   recomputed count, but is looser than "one canonical line per sample".
7. The whitelist is an exact pinned tuple but is not derived from CMake at test
   time, so a future required source could be added to the build and silently
   omitted from the ZIP; the planned fresh-extract build is the real guard.
8. `_scan_payload` (`:648-668`) is a pattern heuristic, not credential
   authentication; the controller already states this. Likewise the checkpoint is
   CRC/hash-checked but not fully decoded (`:437-450`), with the C++ loader and
   main's actual header check covering semantic validity.

## Independence and evidence checks

- Named-helper RED precedes implementation and is feature-free (controller
  `task-6-package-red.txt`); Linux GREEN with existing Pillow 12.1.0 is 66 tests,
  0 skips, 39 package tests (`task-6-all-helpers-green.txt` tail).
- Actual default-20 training evidence: fresh model dir, all `COURSE_*` cleared,
  exit 0, 20 ordered epoch records, one completion, CRC `f473014f`, best 15,
  validation 0.992399991, 594.740494 s, retained launch/stdout/stderr/resource
  logs. Byte identity with the legacy bundle is documented as an observed
  deterministic outcome, not as reused training.
- Actual final model: five fresh unprofiled N=10000/B=1024 processes, all 9911
  correct with identical prediction SHA256, current hashes rechecked; independent
  recount against the original labels is recorded. Median 1657.407840 ms, max
  1684.153743 ms, 100 ms FAILED; no legacy 78.942 ms substitution.
- Fresh course CTests 48/8/9/15 and prior operator/12 workflow/9 host/198
  compliance/Makefile/env checks pass; old `prepare_mnist` import failure due to
  missing pyarrow is carried honestly; no invalid Sanitizer claim.
- No CUDA/CMake/original numerical math file appears in the four-file diff; no
  root `evaluation.json`/`report`/`result.png`/archive exists in the workspace
  (confirmed by glob), consistent with "publication awaits this review".

## Cannot verify / remains with main

This gate approves the helper code and its contract evidence only. I cannot
verify and do not claim: actual root `evaluation.json`/`report`/`result.png` or a
submission ZIP exists (all intentionally absent); visible PNG inspection at real
values; fresh extract + default build without Python/tests/tools; bounded
official train->infer roundtrip on the extracted package; valid-model inference
with labels withheld; protected-file hash proof; whole-branch review, final
backup ref, or platform upload. Those are main's next gates per
`controller-task-6-verification.md:19-23`; Task 6 overall acceptance is not
closed by this review, and a separate helper-code approval must not be read as
whole-task or whole-branch acceptance.

## Verdicts

- **SpecCompliance: Approved** for the Task 6 helper/package contract and docs
  update, against the brief bullets and binding global constraints.
- **TaskQuality: Approved** — code is clear, defensive and consistent with the
  repo's Python helper style; tests are dense and negative-case oriented. Minor
  items above are suggested hardening, not merge blockers for this scoped gate.
- **Task 6 overall: Cannot verify from this review alone** (actual artifacts,
  extract/build/roundtrip, protected hashes, whole-branch review and final backup
  remain main's responsibility).
