# Evaluate A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure and reduce full-process evaluation wall time toward less than 1000 ms on all 10,000 MNIST test images without changing predictions or disabling safety checks.

**Architecture:** Add opt-in wall-time diagnostics to the existing evaluation path, measure the unoptimized code, then consolidate model finite-scan result readbacks if the measured stages justify it. Keep all neural-network kernels, batch size 128, finite-scan coverage, state transitions, and output records unchanged.

**Tech Stack:** C++14, project-owned CUDA Runtime kernels, GNU Make, Python unittest, Bash, NVIDIA H20 / CUDA 12.6 on the currently observed remote host.

**Spec:** `docs/superpowers/specs/2026-10-27-evaluate-a-design.md`

## Global Constraints

- Full-process wall time includes process startup, input/checkpoint loading, CUDA initialization, inference, checks, reporting, and process exit; exclude SSH connection setup only.
- Fixed existing checkpoint and all 10,000 test samples; baseline 9911 correct / 99.11% accuracy.
- C++14, FP32, CUDA Runtime API and project-owned kernels only; no cuDNN, cuBLAS, mixed precision, Tensor Cores, CPU fallback, or new third-party dependencies.
- No training, input mutation, checkpoint mutation, removed safety checks, changed CLI exit codes, or changed stdout grammar.
- Keep all device allocations before batch loops and retain one model arena. Expand its result storage from 1 to 58 ints, an increase of 228 bytes.
- Keep batch size 128 in this iteration. If the target is missed, report it rather than silently starting route B or widening scope.
- Keep the original deployed binary untouched; use an isolated local worktree and a separate remote experiment directory under `/gemini/code`.
- Do not stage, delete, overwrite, or commit pre-existing user documents. Do not commit any changes without explicit user permission.
- Design and plan approved by the user; main agent coordinates/reviews/verifies and coding subagents implement.

## Review Focus

- Multiple simultaneous non-finite tensors must report the first tensor in canonical scan order and its smallest bad index, not whichever scan completes first (Task 2).
- Smaller actual batches and idle/forward/backward state changes must not consume stale result slots or scan unused capacity (Task 2).
- Diagnostics must not introduce extra synchronizations or alter normal stdout, inference, training, or failure exit codes (Task 1).
- Repeated finite checks must retain correct results without any extra device allocation, and exact arena-byte accounting must match (Task 2).
- Faster-looking runs must not hide startup, skip samples, change weights, cherry-pick timing, or overwrite the original deployment (Task 3).

---

## File Structure

- `src/train.cu`: optional evaluation timings and diagnostic output, using existing synchronization boundaries.
- `src/lenet.cu`: fixed result-slot arena layout and consolidated finite-scan readback.
- `tests/workflow_tests.cu`: opt-in profile diagnostics and normal-output/accuracy compatibility.
- `tests/operator_tests.cu`: finite-scan diagnostics, state/batch isolation, exact memory budget and allocation invariants.
- `tests/evaluate_snapshot.cu`: tests-only full-dataset output snapshot tool, compiled separately for baseline/candidate comparison; never linked into production.
- `Makefile`: add the snapshot utility to `all` with its own object, dependency file and link rule, and include its source in `COMPLIANCE_TEST_SOURCES`; retain unchanged production linking and test runner lists. This corrects the original manifest-only plan, which conflicts with the existing source-reconciliation gate.
- `docs/comment-review-checklist.md`: reconcile inventory for any added helpers without marking unrelated items reviewed.
- `README.md`: document the optional profiling environment variable and distinguish internal stage timings from process wall time.
- `docs/evaluate-a-results.md`: checked measurements, commands, environment, correctness and remaining limitations.

### Task 1: Opt-in evaluation stage timing and baseline attribution

**Files:** Modify `src/train.cu`, `tests/workflow_tests.cu`, `README.md`, and `docs/comment-review-checklist.md`.

**Interfaces:**
- Consumes existing `RunEvaluate(const EvaluateOptions&, std::ostream&, std::ostream&)` and `EvaluateBatches(...)`.
- Produces private `EvaluationTimings` with double fields `batch_pack_ms`, `normalize_and_check_ms`, `forward_and_check_ms`, `softmax_and_check_ms`, and `metrics_and_readback_ms`, initialized to zero.
- Extend private `EvaluateBatches(...)` with a final `EvaluationTimings* timings = nullptr`; existing training and validation calls remain unchanged.
- Enable profiling only when `std::getenv("CUDA_LENET_PROFILE_EVALUATE")` equals `"1"`. No public CLI option or options-struct change.
- Emit stderr records `event=profile phase=NAME elapsed_ms=VALUE`, fixed three decimals, classic locale, without changing caller stream formatting.
- Phase names: `dataset_load`, `checkpoint_load`, `device_select`, `storage_setup`, `model_setup`, the five batch fields without `_ms`, and `workflow_total`.
- Initialization timing boundaries: before/after dataset load and count validation; checkpoint load; device selection; WorkflowStream plus WorkflowStorage construction; LeNet construction plus parameter import and its existing finite check. `workflow_total` starts at RunEvaluate entry and ends after the normal evaluation summary, before diagnostic printing and destructors; document this explicitly.

- [x] Write workflow case `evaluate_profile`: invoke the executable as a subprocess with profiling disabled, then enabled through `env CUDA_LENET_PROFILE_EVALUATE=1`; capture stdout and stderr separately. Assert identical sample counts, accuracy and status, no profile records on stdout, no profile records when disabled, every expected phase exactly once when enabled, finite nonnegative times, and total no smaller than any single component. Compare only non-timing output fields because forward timing varies.
- [x] Cover a nonstandard one-batch fixture and a 129-sample partial-batch fixture. Preserve the existing one-batch zero forward-timing contract. Assert the normal below-threshold exit remains 3 with profiling enabled.
- [x] Run the new case against the current code and confirm its expected missing-profile-record failure: `make build/workflow_tests && ./build/workflow_tests --case evaluate_profile`.
- [x] Implement the private accumulator and opt-in stderr formatting in `src/train.cu`. Use `std::chrono::steady_clock`; bracket only existing synchronized phases. Add no extra CUDA synchronization. Keep optional timing off in normal execution.
- [x] Reconcile added helper comments/inventory, document the environment variable, and run `./build/workflow_tests --case evaluate_profile`, `./build/workflow_tests --case evaluate`, `./build/workflow_tests --case cli`, and `make compliance PYTHON=python3`.
- [x] On H20, build this instrumentation-only version in the isolated remote experiment. Save one profiled run and at least three unprofiled independent runs, all with the same inputs/checkpoint, before Task 2. If the measured evaluation/checking stages are not significant, stop and report attribution rather than applying speculative changes.

### Task 2: Consolidate model finite-scan results without changing checks

**Files:** Modify `src/lenet.cu` and `tests/operator_tests.cu`. No kernel or public API additions.

**Interfaces:**
- Consumes unchanged `LaunchFindFirstNonFinite(const float*, std::size_t, int*, cudaStream_t)`.
- Retains unchanged `LeNet::RequireFinite(const std::string&) const` semantics, labels, canonical order and one synchronization per call.
- Model arena owns `kMaximumScannedTensors` result ints rather than 1; host `scan_results_` remains the existing fixed array.
- The loop writes tensor i into `scan_result_ + i`; after the loop, one `cudaMemcpyAsync` transfers `count * sizeof(int)` into `scan_results_.data()`, followed by the existing synchronization and ordered diagnostic loop.

- [x] Add test `lenet_finite_readback_preserves_canonical_order`: poison `conv1.weight[21]` and `conv2.weight[19]` together, add a later bad index to the first tensor, and assert `RequireFinite("batch")` names `batch: conv1.weight[21]` repeatedly. Restore finite parameters and assert no error. Use the existing model-import test pattern, which permits non-finite parameter payloads for this diagnostic test.
- [x] Add test `lenet_finite_readback_ignores_inactive_tail_and_stale_slots`: use a capacity-17 input with only a later sample non-finite; Forward(17) must report the first invalid activation, then Forward(1) over a finite first sample must pass. Exercise idle, Forward and Backward finite calls across repeated finite inputs and verify no new device allocation events around the calls.
- [x] Change the exact capacity-17 arena expectation from 1891708 to 1891936 bytes. Run `./build/operator_tests --filter lenet_` and confirm that the new memory-budget expectation fails before the implementation change.
- [x] Update both `RequiredBytes()` and `PlanArena()` to reserve/take 58 ints; consolidate the device-to-host copy in `RequireFinite()` as specified. Preserve every tensor scan, full checks after training optimizer steps, labels and minimum-index logic. Do not cache finite state or change kernels.
- [x] Rebuild and run `./build/operator_tests --filter lenet_`, then full `./build/operator_tests`, `./build/workflow_tests --case evaluate_profile`, `./build/workflow_tests --case evaluate`, and `make compliance PYTHON=python3`. Assert all expected cases actually execute rather than relying on an empty filter run.
- [x] Repeat the H20 profiled and unprofiled runs from Task 1. Keep the change only if correctness passes and unprofiled timing improves; if not, report evidence and revisit A rather than claiming success.

### Task 3: Full verification and reproducible end-to-end evidence

**Files:** Create `tests/evaluate_snapshot.cu`, its focused behavioral tests and `docs/evaluate-a-results.md`; modify `Makefile` to build/scan the snapshot utility without adding it to test runner lists; update the plan checkboxes and README with verified facts only.

**Interfaces:**
- Consumes original deployed `/gemini/code/cuda_cnn/build/lenet_cuda` and isolated candidate binary, plus identical read-only mounted data and original weights.
- Produces raw logs under the remote experiment, summary results in the local project, and a clear achieved/not-achieved verdict for the less-than-1000-ms target.
- Snapshot tool interface: `evaluate_snapshot DATA WEIGHTS OUTPUT BATCH`; require exactly four arguments, batch in [1,128], production dataset count 10000, and a caller-selected writable OUTPUT outside input mounts. Select device 0, load the checkpoint, use the existing normalization/Forward/Softmax/Argmax launchers without augmentation, and synchronize before reading outputs. Write contiguous sample-ordered FP32 logits (10000*10), then FP32 probabilities (10000*10), then uint8 predictions (10000), all to one binary OUTPUT. Buffer snapshots on the host, accept final partial batches, and return nonzero on validation or CUDA errors.

- [x] Before upload/build, use the worktree skill, inspect the working tree, and keep existing untracked user files intact. Copy only reviewed source/test changes to the isolated remote experiment; inspect any remote/local differences first. Do not overwrite the running deployment.
- [x] Record compiler, GPU identification, source revision/diff and SHA-256 of data, weights and both binaries. Keep unchanged weights and inputs throughout comparisons.
- [x] Run host and compliance checks, full operator tests, and the workflow suite with `./build/workflow_tests --mnist-train /gemini/data-1/train.bin`. Do not invoke a data preparation target that would download a second large dataset unnecessarily. Sanitizer was attempted but disabled itself on both versions; no successful sanitizer analysis claimed.
- [x] Build identical snapshot helpers using the dedicated Make rule against separate same-toolchain baseline/candidate objects (`make -j4 all V=1`, C++14/O2/sm90). Run with read-only official inputs at batches 1, 17 and 128. Every snapshot is exactly 810000 bytes; old/new logits, probabilities and predictions match byte-for-byte and accuracy remains 9911/10000. Repeat after full rebuild; keep comparisons separate from timed acceptance runs.
- [x] Benchmark old/new in alternating order, five independent processes each, with profiling disabled and identical command arguments. Measure from immediately before spawning the program until it exits, capture output, and include startup/exit costs. Use a monotonic external timer; also retain `/usr/bin/time` evidence. Do not discard slow runs or force filesystem-cache resets.
- [x] Report all run times, median, maximum, correct/wrong counts, per-stage evidence, speedup versus the contemporaneous baseline, and whether every candidate run is below 1000 ms. Keep unprofiled acceptance results separate from instrumented diagnostic runs.
- [x] Target missed: report the largest remaining device-setup stage and propose further measurement. Do not automatically change batch size, remove checks, enter B, deploy, or commit.

## Status and handoff

- [x] Inspect current evaluation path, finite scanning, test contracts and storage rules.
- [x] Run three fresh original-binary baselines: 2860, 2770 and 2780 ms; all 99.11% accuracy.
- [x] Record the short design and task plan.
- [x] User approves the written design/plan and selects main-agent coordination with subagent implementation.
- [x] Execute Tasks 1–3 in an isolated workspace; individual and whole-worktree reviews have no blocking code findings. First iteration complete, <1000ms objective NOT met. Remote aggregate tests remain blocked by existing pyarrow absence; no merge/deployment performed.

Selected execution method: subagent-driven implementation with main-agent coordination, review and H20 verification. The isolated worktree and remote experiment preserve the original deployment. See this plan's SDD ledger for live evidence and progress.
