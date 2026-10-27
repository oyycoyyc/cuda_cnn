# Task 4 fix round 1 scoped re-review (F1 + F2 wording / F3 test)

Scope: read-only. Reviewed `task-4-brief.md`, the appended formal-fix section of
`task-4-report.md` (435-499), the appended controller H20 records
(`controller-task-4-verification.md:29-35`), and `task-4-fix-1.diff` (199 lines).
Diff base checked against `.superpowers/sdd/.../task-4-review-snapshot`: snapshot
`course/runtime.cu` has no flush (only `!predictions` at 367 and
`!diagnostics` at 413), snapshot `docs:94-95` still says "course convention",
snapshot `CMakeLists.txt:162` says `executed=14`, snapshot tests hold 14 cases
with no sink case. Current files therefore differ only by this round's 4-file
change (CMakeLists.txt, course/runtime.cu, course/tests/runtime_tests.cu,
docs/course-interface.md). `main.cu` and `course/runtime.h` are unchanged, which
is consistent because the fix makes the existing promise true rather than
changing it. No subagents, SSH, git command, source edit, or test rerun.

## Finding dispositions

### F1 - Buffered sink failure undetected: ADDRESSED

Original wording checked verbatim against the fixed code:

- "`course/runtime.cu:366-369` checks `!predictions` before any flush" -> now
  `predictions << ...` (366), explicit `predictions.flush()` (369), then
  `!predictions` (370) and the runtime_error (371), all before the diagnostics
  record is even assembled (377+). `host_write_predictions_ms` now closes after
  the flush/check (373).
- "both sinks" -> `diagnostics.flush()` (419) before `!diagnostics` (420),
  throw (421), `return 0` (423); the comment at 416-418 states the
  no-rollback boundary.
- "small payloads fail at exit flush" -> the new test uses a real buffered
  `std::ofstream` on `/dev/full` with an 8-byte payload
  (`runtime_tests.cu:1489-1498`), so insertion succeeds into the stream buffer
  and only the explicit flush can detect the failure. On the frozen pre-flush
  archive this exact assertion failed: `task-4-io-red.txt:1` names
  `runtime_tests.cu:1496 ... did not throw`, `executed=1 failures=1`. On the
  fixed build the case passes and all 15 cases pass
  (`task-4-fix-1-ctest.txt:37`).
- "recorded /dev/full probe (exit 0 + status=pass)" -> fixed CLI probe now
  `returncode 1` with stderr `error: could not write inference predictions`
  and no success record; stderr-sink case `returncode 1` with the four already
  published predictions and no rollback claim
  (`task-4-fix-1-io-probes.json`; summary
  `task-4-fix-1-io-probes.txt:1`). `docs:110-111,122-123` and
  `runtime.h:17-19,43,55` are now truthful as written.

Residual evidence limitation (not a defect, per controller): the frozen-runtime
RED stops at the first expectation, so the diagnostics assertion has no
old-code RED of its own; it is exercised only on the fixed build (GREEN). The
old defect mechanism (no flush) is identical for both sinks, and the new test
would catch a diagnostics-only regression because it runs after the prediction
block.

### F2 - `unit` wording: ADDRESSED

`docs/course-interface.md:94-96` now reads "The `unit` default is a
user-approved project default assumption, not an externally confirmed course
preprocessing requirement." No remaining "course convention" text anywhere in
the changed docs/header/source (grep: only this new sentence matches
"user-approved"). This matches global constraint 7 and the controller's
requirement that official preprocessing stay unconfirmed.

### F3 - Sink-failure test: ADDRESSED

`runtime_tests.cu:1476-1516` adds
`run_course_infer_reports_buffered_sink_failures`:

- real scratch model/images, direct `RunCourseInfer`, no CPU NN math, no test
  hook;
- prediction sink: buffered `/dev/full` ofstream, open checked (1491-1494),
  must throw `std::runtime_error` (1496-1498), and the captured diagnostics
  must not contain `event=course_infer status=pass` (1499-1500), proving the
  flush failure precedes success diagnostics;
- diagnostics sink: separate buffered `/dev/full`, must throw (1511-1513), and
  the four predictions are parsed as valid (1514-1515) with rollback
  deliberately not asserted;
- registered last (1476), after both existing direct-API cases (1350, 1432),
  so no fork follows parent CUDA initialization; the comment at 1477-1478
  records that requirement (its phrasing attributes CUDA init to this case,
  while it is actually the two preceding cases, but the placement is correct
  and harmless).
- Count wiring is consistent: 15 `TEST_CASE`s in the TU, `CMakeLists.txt:162`
  expects `executed=15`, CTest reports `cases=15 executed=15 failures=0`.

## New-breakage scan of the 199-line diff

No Critical and no Important regressions found. The production hunk is exactly
two flush+existing-check insertions; no pipeline, arithmetic, offset, sticky
flag, readback, allocation, profile event boundary, exit mapping, or header
change. Specifically checked:

- Validation-failure paths are untouched and still publish no prediction byte;
  the single payload insertion remains `predictions << prediction_record.str()`
  (366), so flush is not a second payload write and cannot duplicate content.
- `host_write_predictions_ms` (373) now includes sink delivery; that is a true
  wall bucket and the report discloses it (`task-4-report.md:470-471`). No
  profile record key or ordering changed, so the profile tests (15-case suite
  unchanged for cases 1-14) still pass.
- Success path ordering is predictions-flush then diagnostics; on the
  diagnostics-sink failure the throw happens after publication, which is
  exactly the documented non-rollback behavior.
- The new test opens `/dev/full` with `out|binary`; writes are discarded by the
  kernel device, so it is a safe sink probe and adds no data-file write. It
  introduces only a Linux device dependency in a suite that is already
  explicitly Linux-only (`_WIN32` #error); H20 opens it successfully.
- No new compiler warning source (new `<fstream>` include, all locals used);
  the recorded build is warning-free (`task-4-fix-1-build.txt`, controller
  line 33).
- Cosmetic only: the F2 edit leaves an overlong line (docs:97) and the docs
  reflow at 118-119; no content loss.

## Out of scope, carried forward (not fix-loop blockers)

- F4 default `COURSE_INFER_BATCH=1024` still only weakly pinned by batching
  observations.
- F5 `RunProcess` still has no poll/wait timeout (`runtime_tests.cu:280`); the
  fork-after-CUDA ordering requirement is now at least documented by the new
  case comment, but there is still no guard.
- F6 pageable-H2D nuance for `host_enqueue_batches_ms` unchanged.
- Informational: CLI-level exit-1-on-failed-stdout is evidenced by the recorded
  manual probe, not by an automated subprocess case; acceptable for this
  deliverable.

## Verdicts

- Scoped fix verdict: **F1 ADDRESSED, F2 ADDRESSED, F3 ADDRESSED.**
  New Critical/Important breakage: **None.**
- **SpecCompliance: Approved.** (Task 4 brief/global constraints unchanged;
  the previously inaccurate output-failure contract is now enforced both for
  predictions and diagnostics, and the `unit` documentation wording matches
  the approved assumption.)
- **TaskQuality: Approved.** (Two real buffered-ofstream failure paths are
  covered by a discriminating test with frozen-runtime RED and fixed-build
  GREEN; remaining F4/F5/F6 items are pre-existing non-blocking minors for
  main's final acceptance.)

Pending main, not verified here: final hashes and the infer link re-audit; no
20-epoch/10000-sample/100 ms claim; no H20 rerun performed in this review.
