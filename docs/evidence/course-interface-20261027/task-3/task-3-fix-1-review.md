# Task 3 fix round 1 re-review (scoped)

Scope: only the original Important finding and new breakage from the supplied 195-line
`task-3-fix-1.diff`. Files re-read: task-3-brief.md, appended task-3-report.md (131-149),
updated controller-task-3-verification.md (24-26), task-3-fix-1.diff, plus the current
`CMakeLists.txt`, `course/tests/training_tests.cu`, `course/training.h`, `course/training.cu`
and the fix evidence logs. Read-only; no git, SSH, code writes, or suite reruns.

Original finding verbatim: "No multi-epoch end-to-end run. Every workflow case uses epochs = 1,
so epoch >= 2 re-entry (re-shuffle, Forward after a validation Forward, cross-epoch
re-selection and in-run bundle replacement) has no runtime evidence."

## Status: ADDRESSED

- Real two-epoch workflow run added: `two_epoch_training_reselects_and_publishes_consistent_bundle`
  (course/tests/training_tests.cu:550-637) uses N=10, batch 4, `epochs = 2` through the
  ordinary `RunCourseTrain` path (557-562), so epoch 2 now re-enters shuffle, does a new
  `Forward` after epoch 1's validation `Forward`, and re-runs cross-epoch selection.
- Ordering/counts asserted: exactly two `event=course_train_epoch` lines with `epoch=1,2` in
  encounter order and exactly one completion line, no stray `event=course_train` lines
  (572-599).
- Selection is derived independently from the two per-line `val_accuracy` values with the same
  strict `>` rule (603-613), then compared to the saved bundle's `best_epoch` and bit-identical
  `validation_accuracy` (615-619). Because the printed floats round-trip, this oracle is exact
  for either "epoch 2 improves" or "epoch 1 retained"; improvement is not forced, as directed.
- Completion record is parsed from its own line only, and independently checked for
  `epochs=2`, `best_epoch`, `input_mode=unit`, the saved `lenet.bin` CRC32, and bit-identical
  `validation_accuracy` (621-634); no test-accuracy text (635-636).
- Existing eight cases and helpers are untouched; helpers are additive and each used
  (388-433). CTest count updated to 9, matching the nine `TEST_CASE`s
  (`CMakeLists.txt:134`; test cases 471, 550, 639, 673, 695, 783, 873, 931, 960).
- Production code unchanged: `RunCourseTrain` still at `course/training.cu:506`,
  `TrainingStorage` at 522, completion write at 664; header unchanged
  (`course/training.h:107`). Only CMake regex and the test file appear in the diff.
- Report correction applied: fixed arena wording now states twelve preallocated `DeviceBuffer`
  regions plus LeNet's own arena (task-3-report.md:140), matching training.cu:202-231.

## New Critical/Important findings

None found. The diff is additive test code plus a count update: no production source, no
kernel, no CMake topology change, and no weakened existing assertion. The new case cannot pass
a one-epoch (or three-epoch) implementation because the record counts and epoch values are
pinned (592-598), and it cannot pass a non-strict selector when the two accuracies are equal
because the independent oracle keeps epoch 1 (608-611, 617-619).

New Minor notes (non-blocking, not regressions):

- In-run bundle replacement is exercised only when epoch 2 strictly improves, since the
  fixture intentionally does not force improvement; the same `SaveCourseModel` replacement
  path remains covered by the cross-run case (tests 639+). Fine for this coverage goal.
- The test pins epoch-record order and counts but not that the completion line is
  positionally last; the previous epoch<->completion token ambiguity is already fixed by
  per-line parsing (569/583-587).

## Cannot verify

- H20 execution: accepted from task-3-fix-1-build.txt (changed test TU compiled with
  nvcc sm90 and linked, no warnings) and task-3-fix-1-ctest.txt (host 48, GPU 8,
  training 9 executed, all pass; a distinct run - timings differ from task-3-ctest.txt).
  The fix-1 build log is an incremental rebuild of the one changed test TU, which is
  sufficient here because production TUs are unchanged and previously verified; not re-run.
- Which best epoch the H20 fixture actually selected (epoch 1 retained vs epoch 2 reselected)
  is not visible in the logs; the case is written to be correct either way.
- Git-level claim that only these two files changed in the fix round: no git commands
  permitted; the supplied diff is the only change evidence, and the measured anchors above
  are consistent with it.

## Verdicts

- Scoped fix verdict: **ADDRESSED / Approved**
- Final Task 3 spec compliance: **Approved**
- Final Task 3 task quality: **Approved**
