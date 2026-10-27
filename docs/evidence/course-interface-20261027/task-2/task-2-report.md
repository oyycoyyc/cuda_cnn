# Task 2 report: tests/declarations handoff (RED pending)

## Scope delivered
Tests/declarations only, per controller ordering. Created (no Task1 edits, no oldsrc/kernel/public-header edits, no implementations, no CMake):

- `course/gpu_ops.h` — declaration-only `course::LaunchPrepareInput` and `course::LaunchArgmax` with the exact brief signatures and full host contracts (mode formulas, no-uint8, augmentation/padding, sticky flag, overlap/null/dimension validation, no sync).
- `course/inference.h` — declaration-only `course::InferenceModel` with `(int, const ParameterSet&, cudaStream_t)`, destructor, deleted copy, `Forward`, `RequiredDeviceBytes`; pimpl (`std::unique_ptr<Impl>`), fixed one-arena forward-only contract.
- `course/tests/gpu_tests.cu` — one TU, includes existing `course/tests/test_support.h` (owns `main`, scratch root, nonzero executed count). No second harness, no production fallback.

## Test cases (7, explicit names)
1. `float_input_modes_and_translation` — kUnit/kByte/kNormalized no-augment bitwise scalar fixtures; fractional kUnit (1/3, 0.1); kByte 127.5/254.75/128.5 must match full FP32 formula and differ from both uint8 truncation and rounding; normalized pass-through incl. negatives/signed zero; augmented fixtures with pinned original offsets (seed `0x8877665544332211`, epoch 7: index 4 → dx=2,dy=-2; index 5 → dx=-2,dy=0), expected reconstruction via `TranslationOffsetForSample`, normalized black padding in all modes, same-index rows bit-equal, different-index rows differ; non-blocking-stream parity.
2. `input_rejects_invalid_batch_epoch_pointers_and_overlap` — batch 0/-1, augment epoch 0, null packed/output, augment null indices, unknown mode, same/partial overlap; non-augment with null indices and epoch 0 succeeds.
3. `label_free_argmax_ties_and_sticky_nonfinite` — smallest-index ties (incl. all-tied row), FLT_MAX finite not flagged, repeated launches in one request keep caller-zeroed flag 0, NaN at last element of last row detected, sticky across a later finite launch, fresh requests detect NaN/+inf/-inf, fresh finite request clears only via caller zeroing.
4. `argmax_rejects_invalid_shapes_pointers_and_overlap` — batch 0/-1, classes 0/257, three null pointers, predictions overlapping logits (exact + one-element partial), flag overlapping logits/predictions.
5. `inference_logits_bitwise_match_reference` — capacities 1/17/1024: `RequiredDeviceBytes()` strictly smaller than original `LeNet` at matching capacity; one capacity-1024 model vs original `LeNet` fed the same pre-normalized FP32 device buffer and same `ParameterSet`; batches 1/17/1024 with `memcmp` bitwise logits, exact int32 predictions vs host smallest-index argmax of reference logits; no allocation events during Forward; bad flag stays 0.
6. `inference_tail_repeated_no_allocation` — capacity 17 sequence `{17,1,17,3,17,1,16,17,2}` each bitwise vs original LeNet and exact predictions; stable logits pointer; identical consecutive calls; exactly +1 live/event at construction and release to baseline; no allocation/live growth around Forward+argmax.
7. `inference_rejects_invalid_parameters_capacity_and_pointers` — capacities 0/-3, missing/extra tensor, short values, wrong shape, NaN/+inf/-inf poisoned weights; null/zero/over-capacity Forward; full-capacity boundary still valid.

## Pinned decisions (request ruling only if wrong)
- Invalid capacity, schema, or non-finite supplied weights → `std::invalid_argument` (in-memory host API misuse); CUDA/memory failures → `std::runtime_error`.
- kNormalized augmented padding is the normalized black pixel `(0-0.1307)/0.3081`, not raw 0 (matches "normalized black padding").
- `LaunchArgmax` classes accepted in `[1,256]` mirroring the original argmax; flag is never cleared by argmax.
- `Forward` returns the stable arena logits pointer; constructor synchronizes once after weight copy; exactly one DeviceBuffer arena allocation.
- Both launchers reject overlapping buffer extents, including one-element partial overlap, before launch.

## Verification performed locally
- `g++ 14.2 -x c++ -std=c++14 -Wall -Wextra -Wpedantic -fsyntax-only` on both headers and `gpu_tests.cu` against a temporary stub CUDA runtime header: clean, no diagnostics.
- Offsets for the pinned fixture computed with the original host `random.cpp` (`TranslationOffsetForSample`), not guessed.
- No local CUDA compiler/GPU: no CUDA compile or pass claim is made.

## Expected RED for main on H20
Test TU should compile and fail to link only on undefined `course::LaunchPrepareInput`, `course::LaunchArgmax`, and `course::InferenceModel::{ctor,dtor,Forward,RequiredDeviceBytes}` when linked with existing reference objects (`lenet.o`, kernel objects `input/activation/pooling/linear/convolution/loss/metrics/adam`, `parameters.o`, `random.o`). Any other unresolved symbol indicates a test/declaration defect. After main reports that RED, resume for GREEN implementations plus initial `CMakeLists.txt`.

Status: NEEDS_CONTEXT (waiting for main RED).

## GREEN phase (main RED verified: warning-free nvcc compile, link failed only on new course APIs)

### Test/declaration oracle corrections requested by main
1. `course/tests/gpu_tests.cu` missing-index validation: the augment=true/epoch=1 case now passes `nullptr` indices (intended missing-index rejection). The existing augment=true/epoch=0 valid-indices invalid case is retained.
2. Augmented index-consistency fixture: every row is now the same nonconstant source image copied from `PackedFixture(kImagePixels, mode)`. Equal original indices now legitimately select equal translations, so rows 0/1/4 must match exactly; the full per-row expected reconstruction, fractional-mode coverage, black-padding assertions, and distinct-index (`row0 != row2`) memcmp check are unchanged and no tolerance was introduced.
3. `course/gpu_ops.h` doc corrected: `packed` and `original_indices` are read-only, `normalized` is the written output; all remain live through stream work.

### Sources created
- `course/gpu_ops.cu` — `LaunchPrepareInput` kernel: one element per thread; kUnit/kByte/kNormalized exact FP32 order; no uint8 path; original `TranslationOffsetForSample`; padding = normalized black pixel in every mode. `LaunchArgmax` kernel: one thread per row, int32 predictions, strict `>` smallest-index ties, every row/class scanned for NaN/inf, sticky flag only ever written to 1, never cleared. Host validation: batch/classes/mode/nullers, augment index/epoch, overlap by address range, checked extents/grid; no sync/readback.
- `course/inference.cu` — `InferenceModel::{ctor,dtor,Forward,RequiredDeviceBytes}`. Validates capacity, canonical schema, and every supplied weight finite before any device allocation. Queries free memory, then builds exactly one 16-byte-aligned `DeviceBuffer<uint8_t>` arena containing only ten parameter payloads + forward activations + pool winners (ReLU is in-place on its own output, permitted by the original launcher contract). Directs one async parameter upload loop followed by one stream synchronization. Forward runs the unchanged original chain in original order/shapes; no allocations or syncs in Forward; stable logits pointer.
- `CMakeLists.txt` — CMake>=3.18, C++/CUDA standard 14 REQUIRED, `CMAKE_CUDA_ARCHITECTURES=90` only when undefined and before `project()`, fixed O2 (Release and target level), no fast-math. Default build: `course_parameters`, `course_checkpoint`, `course_host` (course I/O only), `course_reference_forward` (activation/pooling/linear/convolution only), `course_forward` (new GPU code; no training dependency). Inside `COURSE_BUILD_TESTS=ON`: separate `course_reference_training` (lenet + input/loss/metrics/adam), `course_io_tests`, `course_gpu_tests`, `course_model_bundle`, and CTest registration of `course_io_tests`/`course_gpu_tests`.

### Static checks performed locally (no CUDA success claim)
- g++ 14.2 `-std=c++14 -Wall -Wextra -Wpedantic -fsyntax-only` on `course/tests/gpu_tests.cu`, `course/gpu_ops.h`, `course/inference.h`: clean.
- g++ 14.2 host-compatible syntax check of `course/inference.cu` against a temporary stub CUDA runtime header: clean, no diagnostics.
- `course/gpu_ops.cu` transformed only for local checking (device keywords removed, launch configuration removed, host/kernel C++ bodies parsed with stub `blockIdx`/`blockDim`/`threadIdx`): clean apart from expected transform-only unused `blocks`/`stream` warnings.
- Host oracle probe using original `random.cpp`: confirms pinned offsets index4=(2,-2), index5=(-2,0), index9=(0,-1), and for all three modes corrected fixtures give row0==row1==row4, row0!=row2, padding present, first padding value -0.424213.
- No nvcc, CMake, or CTest run locally (no local CUDA toolkit/GPU); main owns real H20 build/CTest acceptance.

Status: DONE (GREEN + initial CMake; awaiting main real-GPU acceptance).

## Fix round 1 (open findings from task-2-review.md / controller-task-2-verification.md)

### Critical CMake dependency (brief-required optional tool)
- `CMakeLists.txt`: added `target_link_libraries(course_checkpoint PUBLIC course_parameters)` with a comment. `course_checkpoint` now declares the symbols `src/checkpoint.cpp` actually calls (`ValidateLenetParameters`, `LenetParameterSpecs`, `CreateLenetParameters`) instead of relying on `course_host`'s accidental parameter-before-checkpoint ordering. `course_host` ordering was not changed. Covering test: the required tests-ON `course_model_bundle` build/link that previously returned 2 (`task-2-cmake-build.txt`, `task-2-model-bundle-link.txt`). No CMake source assertions were added.

### Important argmax shared-flag data race
- `course/gpu_ops.cu`: replaced the plain `*bad_output = 1` store with `atomicExch(bad_output, 1)` and a comment covering concurrent row writers, atomic same-address update, constant value, and the never-clearing sticky contract. Math, launch geometry, validation, flags, and all other deferred minor items are unchanged.
- TDD order: the new numerical fixture `label_free_argmax_multiblock_sticky_nonfinite` was added before the kernel change. It uses 1024 rows (four 256-thread blocks), poisons 128 rows spread across every block with NaN/+inf/-inf at varying class positions, verifies the flag becomes 1, then launches an all-finite fixture of the same shape without re-zeroing and verifies the flag remains 1 plus exact per-row predictions. All seven existing GPU cases, including the single-row/first/last/ties cases, are retained.
- Recorded limit: this test is behavioural sticky/multi-block coverage only. It would also pass against the frozen pre-atomic kernel on current hardware, so it is not offered as proof of race freedom. Race freedom for this change rests on the CUDA/C++ memory-model guarantee of the atomic exchange, which main should confirm in the generated PTX/SASS (`atom`/`red` exchange, not a plain `st.global`).

### Local checks and outputs (static only)
- `g++ -x c++ -std=c++14 -Wall -Wextra -Wpedantic -I. -Iinclude -Isrc -I<temp-stub-cuda> -fsyntax-only course/tests/gpu_tests.cu` -> exit 0, no diagnostics. The suite now registers 8 cases (the 7 accepted cases plus `label_free_argmax_multiblock_sticky_nonfinite`).
- Same g++ invocation on `course/inference.cu` -> exit 0, no diagnostics.
- `course/gpu_ops.cu` local transform (device keywords and launch configuration removed, stub `blockIdx`/`blockDim`/`threadIdx`/`atomicExch`) -> only the expected transform-only unused `blocks`/`stream` warnings; no C++ diagnostics, and no plain `*bad_output` store remains.
- No cmake/nvcc/GPU locally, so no configure, build, link, CTest, PTX, or runtime claim is made.

### Outstanding H20 checks before checkpoint
- Fresh `cmake -S . -B build-course -DCOURSE_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=90`; full default build must build `course_model_bundle` and link it, proving the added PUBLIC edge.
- `ctest --test-dir build-course -R 'course_(io|gpu)' --output-on-failure`: expect 8/8 GPU cases and host 48/48.
- Inspect the argmax kernel's generated PTX/SASS for an atomic exchange on `bad_output`; re-run GPU tests after the change.
- Default tests-OFF/override-arch audit and protected-hash/scoped re-review remain main-owned.

Status: DONE (fix round 1; awaiting main real-GPU clean build/CTest and scoped re-review).
