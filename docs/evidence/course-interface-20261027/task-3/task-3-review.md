# Task 3 review: course-only GPU training and persisted preprocessing

Scope: read-only source/spec review of the supplied scoped diff (base 17884b8 -> current
uncommitted `CMakeLists.txt`, `course/training.h`, `course/training.cu`,
`course/tests/training_tests.cu`). Supplied diff was read once as the changed-file set; the
three new files on disk match it (training.h 111, training.cu 671, training_tests.cu 864
lines). No git, test, build, SSH, commit, or edit actions were performed.

## Verdicts

- **Spec compliance: Approved**
- **Task quality: Approved** (one Important runtime-coverage gap, main/Task6-owned; no code
  defect found)

## Strengths

1. **Strict data provenance.** The workflow reads only the two caller paths:
   `ReadImages(options.images_path, ...)` / `ReadLabels(options.labels_path, ...)`
   (course/training.cu:510-512). No `getenv`, `std::cout`/`cerr`, hidden dataset, or old
   `RunTrain`/`train.h` reference exists in the new sources. Labels are the validated uint8
   view produced by Task 1, so nothing is quantized through uint8.
2. **Original math preserved, no kernel edits.** `MakeTrainValidationSplit`
   (training.cu:515-517), `ShuffledTrainingIndices` (541), one `LeNet` instance (523),
   original `Forward`/`Backward`/`AdamWStep` (557/586/590), original loss launcher (559) and
   original argmax/count launcher (348 area). The diff touches no existing numerical source.
3. **Pinned policies are real workflow inputs, not test hooks.** `CourseValidationCount`
   (460) is called at 515-517, `CourseAdamWConfig` (471) at 524, `CourseLearningRate` (475) at
   590, and `BestValidationSelection` (488) at 621. Values/edges match the pin: split
   60000->5000 else max(1,N/5) (460-470), LR 1e-3/1e-4/1e-5 at 12/13/18 (475-486),
   AdamW (0.9,0.999,1e-8,1e-4), strict `>` comparison (497-502).
4. **Correct training loop semantics.** One global step per batch with monotonically
   incremented `global_step` (589-591), sample-weighted epoch loss across batches
   (593-596), validation drawn only from `split.validation_indices` in split order without
   augmentation (609-613), and per-epoch diagnostics (628).
5. **Finite/CUDA boundaries retained.** Preparation output scan (318-323), original
   `LeNet::RequireFinite` after forward/backward/step (557/587/591), probability,
   per-sample-loss, mean-loss and gradient scans (564-577), plus final reload scan (661).
6. **Save/reload contract is genuinely enforced.** Strict improvement exports parameters and
   publishes mode+checkpoint via `SaveCourseModel` (621-626); success requires a reload
   (637) whose mode (639-641), best epoch/accuracy bits (642-647), normalization bits
   (648-654), and full parameter bits (655-659) all match before `return 0`. The completion
   record (664-667) is built from the reloaded bundle and the validated CRC (663), in the
   exact required token order with classic-locale numbers (436-455).
7. **Safe existing work directory.** `EnsureDirectory` creates only when missing and never
   clears (137-166); reserved-pair refusal/replacement comes from the reviewed
   `SaveCourseModel` (`course/model_io.cpp:338-356`). Tests cover missing directory with a
   surviving sibling (tests 427+), pre-populated directory with preserved files (506+), and
   lone junk `lenet.bin` refusal (540+).
8. **Diagnostics only through the supplied stream.** Both record writers write to the
   caller's `std::ostream` only (417-455); no stdout path exists. The happy-path test captures
   an `ostringstream`, checks the CRC against the saved bytes, and asserts no test-accuracy
   text (tests 465-466, completion-record helper near 392-405).
9. **Batch ruling honored.** `ValidateOptions` accepts any positive batch representable as
   `int` and adds no 1024-style cap (training.cu:124-129); the test pins `INT_MAX + 1`
   rejection without allocating (tests 766-779). CLI default 128 remains Task 1's.
10. **Fixtures match the required claims.** N=15/validation=3 (tests 650-655); I re-derived
    the uniform-prediction fractions: validation 1/3; training 2/12 or 1/12; all 3/15, 2/15 or
    1/15, so all three accuracies differ for every possible predicted class, and the saved
    metadata is compared bitwise (729-737). `25.1F`/`25.4F` provably collapse under both
    truncation and rounding (572-573) yet produce different trained parameter bits (622-624),
    with raw FP32 bits preserved (632-633).
11. **Evidence consistent with the code.** Warning-free build (task-3-build.txt), CTest
    48/8/8 pass (task-3-ctest.txt:16,23,30), original operator suite pass
    (task-3-legacy-operator.txt), and the tests-OFF production build compiles
    `course_training` while `course_forward` stays training-free
    (task-3-default-build.txt:105,129; task-3-dependency-audit.txt:1-2). The production CMake
    move of `course_reference_training` is inside the binding ruling; `course_forward` still
    links only `course_reference_forward` + `course_parameters` (CMakeLists.txt:74-80,86-105).

## Findings

### Important (coverage, not a defect; main/Task6-owned)

- **No multi-epoch end-to-end run.** Every workflow case uses the `epochs = 1` default
  (tests 142, and cases at 427/506/540/562/650/740), so epoch >= 2 re-entry (re-shuffle,
  `Forward` after a validation `Forward`, cross-epoch re-selection and in-run bundle
  replacement) has no runtime evidence. Source inspection supports it: `LeNet::Forward`
  invalidates prior state (`src/lenet.cu:138`) and a valid pair may be replaced
  (`course/model_io.cpp:338-356`), and the selection policy is deterministically tested
  (tests 827+). The brief deliberately avoids 20-epoch fixtures, so this is not a spec
  breach; it is the residual runtime risk for the production 20-epoch default. A 2-epoch
  tiny fixture would close it cheaply, but Task 3 is acceptable without it.

### Minor

- **Report inaccuracy (report only, code is permitted).** task-3-report.md:97 says "one
  reusable device arena owns packed ..." while the code preallocates 12 separate
  `DeviceBuffer`s (training.cu:202-231) plus LeNet's own arena. Multiple preallocated
  buffers are allowed; only the wording is wrong (already flagged by the controller).
- **Overstated exception contract in the header.** training.h:191-192 promises
  runtime_error for every file/data/CUDA/bundle failure, but an extreme accepted batch
  (`<= INT_MAX`, training.cu:124-129) can raise `bad_alloc`/`length_error` from the host
  reserve (187-188) or `overflow_error` from a launcher before any CUDA failure. Normal
  positive batches are unaffected.
- **Duplicated metadata parsing.** `ReadValidatedCrc` (training.cu:393-412) re-parses
  `course.meta` even though `LoadCourseModel` just validated it; format knowledge is
  duplicated. A mismatch still surfaces as runtime_error, so behavior is safe.
- **Minor test-strength nits.** The seeded-initialization baseline is taken from
  `InitializeLenetParameters` (tests 470-471) without asserting the pre-training model
  equals it (the equality holds by src/lenet.cu:125-126); and no test observes that the
  optimizer never consumes validation indices, which is established only by source
  (training.cu:541/551).
- **Test targets skip strict host warnings.** CMakeLists.txt:118-124 does not apply
  `course_enable_host_warnings` to the test executables.

## Cannot verify (accepted from supplied evidence)

- Git-level claims that exactly these four files differ from base 17884b8 and that HEAD/index
  are unchanged: no git commands were permitted.
- Tests-first RED ordering (undefined symbols for 5 training API families) and all H20
  outcomes: accepted from task-3-red-compile.txt / task-3-red-link.txt / task-3-build.txt /
  task-3-ctest.txt / task-3-legacy-operator.txt; nothing was re-run.
- Default 20-epoch training, train/infer entry points, full-process timing, and the final
  executable link audit: explicitly outside Task 3 (Task 4/Task 6).

## Conclusion

The task delivers exactly the briefed production workflow and its tests, uses only supplied
FP32/int32 data, preserves the original split/shuffle/LeNet/loss/AdamW math and pinned
schedule, keeps all diagnostics on the caller stream, publishes only through the validated
bundle path, and keeps `course_forward` free of the training backend. Residual risk is
limited to the untested epoch >= 2 path (main/Task6 runtime acceptance) and documentation
nits. **Spec compliance: Approved. Task quality: Approved.**
