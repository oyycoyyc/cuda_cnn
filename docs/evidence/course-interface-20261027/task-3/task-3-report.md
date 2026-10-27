# Task 3 report: training declarations and tests-first handoff

Status: **main verified CUDA RED; GREEN implementation complete locally, awaiting main NVCC/H20 verification.**

Files created/edited:

- `course/training.h`
- `course/tests/training_tests.cu`
- `course/training.cu` (GREEN)
- `CMakeLists.txt` (GREEN training/tests wiring)

## Declarations proposed in `course/training.h`

Production workflow:

```cpp
int RunCourseTrain(const TrainOptions& options, std::ostream& diagnostics);
```

It documents the approved contract: supplied FP32 images/int32 labels only, `ReadImages`/`ReadLabels`, missing work-directory creation without clearing unrelated entries, `SaveCourseModel` reserved-pair refusal, original deterministic split/shuffle with the supplied seed (1337 default), GPU preparation/augmentation before each Forward, weighted loss, monotonic global AdamW step, strict best validation selection, saved bundle reload/validation before returning 0, all diagnostics through the supplied stream, and the exact completion record.

Narrow production-used pure policies needed to test actual schedule/config/best-selection without 20-epoch fixtures:

```cpp
std::uint32_t course::CourseValidationCount(std::uint32_t sample_count);
AdamWConfig course::CourseAdamWConfig();
float course::CourseLearningRate(std::uint32_t one_based_epoch);
class course::BestValidationSelection {
  bool Consider(std::uint32_t one_based_epoch, float validation_accuracy);
  std::uint32_t best_epoch() const noexcept;
  float best_accuracy() const noexcept;
};
```

These are intended to be the policies actually called by `RunCourseTrain`, not test hooks. No old `RunTrain`, `LearningRateForEpoch`, external test split, test dataset, or pretrained weights are used.

## Registered cases (9)

Required cases, exact names:

1. `train_saves_readable_model_without_test_split_files` — real one-epoch training into a missing work directory; sibling file survives; `lenet.bin`/`course.meta` load through `LoadCourseModel`; mode/metadata/normalization/CRC checked; saved weights differ from `InitializeLenetParameters(seed)`; completion record checked; reference LeNet and `InferenceModel` forward logits match bitwise; no test-accuracy claim.
2. `float_training_keeps_fractional_pixels` — two real kByte training runs whose inputs are `25.1F` and `25.4F`. Those values collapse under both uint8 truncation and rounding, yet the resulting saved parameter tensors differ and both bundles work with reference/new forward in persisted kByte mode. This is not a source-file preservation check.
3. `validation_uses_training_only_indices` — N=15, validation=3. Reproduces the original `MakeTrainValidationSplit(N, CourseValidationCount(N), 1337)` oracle, uses identical pixels so predictions are uniform, and distributes labels so that for any uniform predicted class the validation, training-subset, and all-sample accuracies are all different. It recomputes the saved model's validation accuracy on exactly the expected validation indices and requires it to match saved metadata. Also checks deterministic split, disjointness, and exact coverage.
4. `training_hyperparameters_and_schedule` — pins LR transitions at 1..12 / 13..17 / 18+, AdamW `(0.9, 0.999, 1e-8, 1e-4)`, split rule `60000 -> 5000`, `else max(1, N/5)`, and invalid policy inputs.

Additional coverage:

5. `training_replaces_valid_bundle_and_preserves_unrelated_entries` — existing work directory, unrelated file/subdirectory preserved, valid pair replaced on a second run.
6. `training_refuses_malformed_reserved_pair` — lone junk `lenet.bin` produces `std::runtime_error`, the junk file is untouched, and no `course.meta` is created.
7. `training_rejects_invalid_options_before_work` — epochs 0, N 1, bad H, batch 0, batch `INT_MAX+1`, unknown mode, and empty work path throw `std::invalid_argument` and do not create the selected directory.
8. `best_selection_requires_strict_validation_improvement` — first valid accuracy selected; equal/lower values ignored; strict improvement selected; invalid epoch/range/NaN rejected without changing state.
9. `two_epoch_training_reselects_and_publishes_consistent_bundle` — real bounded N=10/batch4 two-epoch run. Requires success, exactly two ordered epoch diagnostics, exactly one completion record with `epochs=2`, and saved metadata/CRC consistent with strict best selection derived independently from the two line-scoped epoch accuracy values. Equal/lower second-epoch accuracy keeps epoch 1; no artificial improvement is forced.

Expected CTest executed count: `executed=9 failures=0 status=pass`.

## Fixtures and resources

- Only tiny owned scratch directories created by `test_support.h` via `MakeScratchDirectory`; no `/train`, `/test`, mounted output, or repo data touched.
- N = 4..15, explicit small `batch_size = 4`, seed `1337`; eight cases use `epochs = 1` and the added coverage case uses `epochs = 2`.
- Raw fixture files are FP32 images and int32 labels written with the existing test support helpers.
- Real GPU training is exercised; no mocks, source-text scans, or CPU network fallback.
- Reference forward uses existing `LeNet`/`ImportParameters`; new forward uses existing `course::InferenceModel`; GPU preparation uses existing `course::LaunchPrepareInput`.

## Static verification performed here

No CUDA compiler or GPU is available in this environment, so no CUDA-success claim is made. Host syntax checks of `training.h`, `training.cu`, and `training_tests.cu` were run with `g++ -std=c++14 -fsyntax-only -Wall -Wextra -Wpedantic` using a temporary CUDA-runtime declaration stub. All produced no diagnostics. This checks C++ syntax/type usage only, not NVCC support or runtime behavior. Main already supplied actual NVCC RED evidence; GREEN NVCC/H20 remains main-owned.

Expected link-time RED until `course/training.cu` exists, with undefined symbols including:

- `course::RunCourseTrain(...)`
- `course::CourseValidationCount(unsigned int)`
- `course::CourseAdamWConfig()`
- `course::CourseLearningRate(unsigned int)`
- `course::BestValidationSelection::Consider(unsigned int, float)`

`MakeTrainValidationSplit` comes from `src/random.cpp`, so the eventual training/test link must include that original split implementation (not `src/train.cu`).

## CMake wiring applied

- `course_reference_training` (original `src/lenet.cu`, loss, metrics, input, AdamW kernels) moved outside `COURSE_BUILD_TESTS` so production training can link it. `course_forward` still does not depend on it.
- `course_training` library added from `course/training.cu` plus `src/random.cpp`; links `course_host`, `course_forward`, `course_reference_training`. No old `src/train.cu`, CLI, dataset, or reporting workflow dependency.
- `course_training_tests` and CTest `course_training` added inside the optional test block, pinned to `executed=9 failures=0 status=pass`.
- `course_checkpoint PUBLIC course_parameters` edge preserved.

## Points for main to confirm during GREEN

1. `RunCourseTrain` honors direct-API `batch_size`; only positive `<= INT_MAX` is required. The old `[1,1024]` wording/limit was removed from the header. Tests use batch 4; the CLI still supplies 128. Large valid capacities may fail normally on device memory.
2. `CourseValidationCount` intentionally rejects N < 2; `RunCourseTrain` already requires N >= 2.
3. Completion record is checked by key/value tokens and its CRC must equal the CRC32 of the saved `lenet.bin`; validation accuracy token must parse back to the saved metadata value bit-for-bit. Epoch diagnostics deliberately use `val_accuracy=`, not the completion keys.
4. Malformed reserved-pair refusal is expected to surface as `std::runtime_error` per the Task 1 bundle contract.
5. The validation fixture was strengthened from the original N=10/class0-or-1 assumption to N=15 with labels covering every class across partitions, so the subset distinction holds for any uniform predicted class. Actual H20 outcome is pending main.

## GREEN implementation (local handoff)

### `course/training.cu`

- `RunCourseTrain` validates direct options (`N>=2`, geometry, epochs 1..1000, positive batch `<= INT_MAX`, known mode), reads only `options.images_path`/`labels_path`, creates a missing work directory without clearing it, and uses `SaveCourseModel`'s reserved-pair rules.
- One `LeNet(max_batch, seed)` plus twelve separately preallocated `DeviceBuffer` regions (packed FP32 images, uint8 labels, original uint32 indices, normalized images, probabilities, per-sample losses, mean loss, logits gradients, predictions, correct flags, correct count, finite-scan result) provide reusable storage. There is no single auxiliary arena; LeNet keeps its own arena.
- Original `MakeTrainValidationSplit(N, CourseValidationCount(N), seed)` and `ShuffledTrainingIndices` are used directly. Training uses `LaunchPrepareInput` with augmentation and original indices; validation uses the same preparation without augmentation.
- Weighted batch loss is accumulated across batches; one global step increments per optimizer update with `CourseLearningRate(epoch)` and `CourseAdamWConfig()`. Finite checks run after preparation, forward, backward, and AdamW step, plus probabilities/loss/gradient scans.
- Strict `BestValidationSelection` improvement exports parameters and publishes mode+CRC via `SaveCourseModel`; after the last epoch the bundle is reloaded, mode/metadata/parameter bits revalidated, imported, and finite-checked.
- Completion record is emitted to the caller's diagnostics stream with classic-locale numbers, using the validated `course.meta` CRC: `event=course_train status=pass epochs=... best_epoch=... validation_accuracy=... input_mode=... checkpoint_crc32=...`. No stdout, no test accuracy.

### `course/training.h`

- Batch contract corrected: positive and representable as `int`; no invented 1024 training cap. Large valid batches may fail normally on memory.
- Pure policies remain the ones the workflow calls. No parallel unused test policy.

### `course/tests/training_tests.cu`

- Validation fixture strengthened to N=15/validation=3 with per-partition label coverage, removing the class-0/1 assumption. Required eight cases stay registered.
- Invalid-options coverage now includes `batch_size = INT_MAX + 1` under `std::invalid_argument`.

### Local commands

```text
g++ -std=c++14 -x c++ -fsyntax-only -Wall -Wextra -Wpedantic \
  -I<temp-cuda-stub> -I. -Iinclude -Isrc course/training.cu
g++ -std=c++14 -x c++ -fsyntax-only -Wall -Wextra -Wpedantic \
  -I<temp-cuda-stub> -I. -Iinclude -Isrc course/tests/training_tests.cu
```

Both local checks were clean. They are syntax/type checks with a declaration-only CUDA stub, not NVCC or H20 claims.

### Concerns / main verification remaining

- Actual NVCC O2/sm90 build and H20 execution of `course_training` are still required.
- Record actual H20 validation-oracle outcome. The fixture is designed to distinguish subsets for any uniform predicted class, but only H20 execution confirms the chosen class/accuracy distribution.
- `course_reference_training` is now a production target; review that moving it outside the optional test block does not alter the baseline parameter/checkpoint PUBLIC edge (it does not).

## Fix round 1: two-epoch coverage

Main's fresh H20 full build/host/GPU/training/operator run passed; independent review approved code/spec with one Important coverage finding: every workflow fixture stopped at one epoch. Added exactly one real bounded two-epoch case; the existing eight cases remain unchanged.

- New case `two_epoch_training_reselects_and_publishes_consistent_bundle`: N=10, batch=4, epochs=2, supplied FP32/int32 fixture only.
- Parses diagnostics line by line (`DiagnosticLines`/`LineStartsWith`), explicitly avoiding a global first-token search. Requires exactly two ordered `event=course_train_epoch` records with `epoch=1,2`, and exactly one completion record.
- Independently applies strict `>` selection to the two per-line `val_accuracy` values, then verifies saved bundle `best_epoch`/`validation_accuracy`, completion `epochs=2`/`best_epoch`/`input_mode=unit`/`validation_accuracy`, and the completion CRC against the saved `lenet.bin` CRC32.
- Equal or lower second-epoch accuracy naturally retains epoch 1; the fixture does not force an improvement. Existing recognized valid-pair replacement and pure strict-comparison policy tests remain.
- CTest `course_training` PASS regex updated to `executed=9 failures=0 status=pass`; exit/count checks are not weakened.
- Report correction: the workflow does not use one auxiliary arena. It uses separately preallocated `DeviceBuffer` storage (twelve regions) plus LeNet's own arena.

Local static check (no CUDA compiler/GPU here):

```text
g++ -std=c++14 -x c++ -fsyntax-only -Wall -Wextra -Wpedantic \
  -I<temp-cuda-stub> -I. -Iinclude -Isrc course/tests/training_tests.cu
```

Clean. H20 execution and the scoped rereview remain pending main; no CUDA-success claim is made here.
