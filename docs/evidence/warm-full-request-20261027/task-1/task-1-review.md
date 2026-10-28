# Task 1 review: optional warm full-request benchmark

Fresh read-only amendment review. Reviewed the task brief, design amendment,
task-1 report, controller verification, and `task-1-review.diff` (1631 lines)
in full, then inspected the actual `CMakeLists.txt`,
`course/tools/warm_benchmark.cu` (348 lines), and
`course/tests/warm_benchmark_tests.cu` (1240 lines). Also checked the unchanged
runtime/API files for concrete integration risks: `course/runtime.cu`,
`course/inference.h`, `course/gpu_ops.h`, `course/interface.h`,
`course/io.cpp`, `course/model_io.h`, `include/tensor.h`,
`include/cuda_check.h`, and `course/tests/test_support.h`.

## Strengths

- The timed round matches the amended boundary: one steady-clock start before
  the per-round sticky-flag reset; every batch is actually copied again from
  host offsets `batch_index * B * 784`; prepare, `Forward`, and global-offset
  `LaunchArgmax` follow in official order; all-N predictions plus the flag are
  copied D2H; exactly one stream synchronization and host flag/range/invariance
  checks precede the clock stop. Model, images, buffers, baseline, and setup
  synchronization are outside the loop; no device allocation, events, disk
  access, or record formatting occurs inside a timed round.
- The baseline is copied from the first real warmup readback, and every later
  round is compared against it. The final stdout is the last actual readback,
  not a cached prediction. Stdout remains unpublished until all rounds pass;
  the passing completion record is written only after stdout and stderr have
  both been flushed successfully, so a stdout sink failure cannot leave a pass
  marker.
- CLI and host ordering are strict: `argc == 9`; `warmups`/`runs` are digit-only
  decimals in `[1, 100000]`; the first seven arguments go through
  `ParseInferArgs` with a null profile input; model and image validation finish
  before `cudaSetDevice(0)`; the persisted model mode is authoritative and no
  labels or training/profile variables are consulted.
- The batch loop is an exact functional match for `RunCourseInfer`'s current
  loop, including offsets, partial-tail handling, one stream, and the
  preallocated buffer layout. The tool links `course_host` plus
  `course_forward` only, and the tests link `course_host` only.
- CMake additions are confined to the existing `if(COURSE_BUILD_TESTS)` block.
  The default graph remains train/infer only. CTest uses the suite regex and a
  300 s timeout, with a 240 s per-child deadline.
- Tests cover literal `[1, 0]` at B=1/B=2 over W=2/R=3 with exact record
  counts and one-based phase indices; N=17/B=4 and N=1025/B=1024 tails against
  the official `infer` oracle; a safe precondition plus finite poison failure
  with no stdout and no pass completion; argc/count/batch/model-metadata/
  CRC/payload/image failures; and persisted-mode behavior with conflicting
  `COURSE_INPUT_MODE`, `COURSE_EPOCHS`, `COURSE_SEED`, and `COURSE_PROFILE`
  values. Host NN arithmetic is not used as an oracle. The child environment
  strips inherited `COURSE_*` variables, and the parent makes no CUDA call
  before fork/exec.

## Critical

None.

## Important

None.

## Minor (non-blocking)

1. The complete request loop is duplicated from `course/runtime.cu` instead of
   sharing one helper. It currently matches exactly, but future runtime changes
   could drift from this benchmark. This is a maintainability note, not a
   current correctness defect.
2. `SplitTokens` collapses repeated whitespace, so the tests pin field order,
   count, and values but not the protocol's exact single-space formatting.
   The implementation currently emits the required single spaces.
3. Optional coverage gaps: byte mode, batch larger than N, N=1, the default
   batch of 1024, and a failure occurring after the first round are not
   separately exercised. Source inspection covers the shared paths, and the
   early-poison probe covers failure publication.
4. `RequireBenchmarkExecutable` checks that the build-adjacent path exists, not
   that it is executable; a non-executable file would surface as exit 127
   rather than the named missing-tool diagnostic.

## Verdict

- **Spec compliance: Approved.**
- **Task quality: Approved.**
- Limitations: the cited H20 NVCC build, five CTest results, link audit,
  default-build check, and `/dev/full` probes were not rerun in this read-only
  review. Real HEAD/index state and the absence of unrelated edits cannot be
  independently proven here without git commands. No actual 10000-image W=5/R=101
  mean or median is claimed by Task 1; that remains Task 2's measurement.
