# Task 4 independent spec + quality review

Reviewer scope: read-only. Reviewed `task-4-brief.md`, `global-constraints.md`,
`task-4-report.md`, `controller-task-4-verification.md`, and the full
`task-4-review.diff` (base `a8e0bc3` -> current uncommitted, 6 files:
`CMakeLists.txt`, `main.cu`, `course/runtime.h`, `course/runtime.cu`,
`course/tests/runtime_tests.cu`, `docs/course-interface.md`). No git command,
no source edit, no SSH, no subagent, no test/build rerun. H20 evidence is used
exactly as recorded by the controller; nothing here is a new CUDA claim.

Diff fidelity: the diff maps 1:1 onto the current files. Spot check: the
recorded RED `task-4-offset-red.txt` names
`course/tests/runtime_tests.cu:1101 EXPECT_EQ(0, split_predictions[1])`, which
is exactly the diff content for that line. Citations below are current file
lines, not diff lines.

## Verdicts

- **SpecCompliance: Needs fixes.** Every functional checkbox of the brief is
  met, but the delivered public interface documentation promises an
  output-stream-failure exit that the implementation cannot deliver for a
  normally buffered payload (Finding 1), and one documentation sentence
  overstates the approved preprocessing assumption (Finding 2). Both are
  narrow, local fixes.
- **TaskQuality: Needs fixes.** Same contract gap plus its missing test
  coverage; otherwise the design is sound, the batch-offset correction is
  real and discriminating, and the fast-path/profiling structure matches the
  constraints. Remaining items are Minor/informational.

- Critical: none found.
- Important: 1 (F1).
- Minor: 4 (F2-F5), plus 1 informational (F6).

## Cannot verify (requires main / later tasks)

- Task 4 does not and should not include the full 20-epoch training run, the
  full 10000-sample benchmark, or any 100 ms full-process claim (later Tasks
  5/6). No timing value is asserted here.
- I did not rerun `/dev/full`; F1 rests on the recorded probe
  (`task-4-dev-full-probe.json`: `returncode 0`, stderr
  `event=course_infer status=pass samples=4 batches=1 input_mode=unit`) plus
  code reading of the publication path.
- I did not verify exact class goldens against real course data, real H20
  wall-clock bucket magnitudes, or whether the official course preprocessing
  is in fact `unit` (still an approved project assumption).
- Unchanged sources were checked only for named concrete risks (see
  "Unchanged-source checks"); I did not re-review Tasks 1-3 in full.

## Strengths (verified in code, not inferred)

1. **Host-before-CUDA ordering is real.** `ValidateOptions`
   (`course/runtime.cu:196`) runs first; model load (215) and
   `ReadImages` (219-220) plus the host prediction landing buffer (229)
   precede the first CUDA call `cudaSetDevice(0)` (237). Model finiteness is
   enforced by the checkpoint loader (`src/checkpoint.cpp:302`), and input
   finiteness/mode bounds by `ReadImages` (`course/interface.h:95-108`), so
   both bundles are genuinely validated before device selection. No device
   banner is written by any of these paths (`include/cuda_check.h` throws only).
2. **One fixed allocation set, no per-batch device churn.** Stream (238-241),
   forward arena + parameters (241-242), batch-capacity `packed`/`normalized`,
   all-N `device_predictions`, one `int` error flag (243-247). The loop
   (263-301) contains only H2D, prepare, forward, argmax and optional event
   records: no allocation, no readback, no synchronization. Optional events
   are allocated once before the loop (253-256), so the fast path allocates
   none and emits no profile record (376).
3. **Sticky-flag discipline.** One `cudaMemsetAsync` before any argmax (250);
   `LaunchArgmax` documents an accumulate-only sticky flag that is never
   cleared (`course/gpu_ops.h:50-58`); exactly one final D2H of all-N
   predictions plus the flag, then exactly one explicit stream sync (308-314);
   validation of flag and prediction range before any publication (317-327).
4. **Global sample offsets are correct after the fix.** Host source is
   `host_images.data() + CheckedMultiply(offset, 784)` (268-269, 277-281)
   while prediction destination stays `offset` samples (294); buffers are
   reused at pointer zero, matching the batch-capacity allocation. The
   RED->GREEN evidence is a true discriminator: literal `[1,0]` oracle failed
   on the frozen runtime at `runtime_tests.cu:1101` and passes now
   (`task-4-offset-red.txt`, `task-4-offset-green-ctest.txt`).
5. **Buffered, all-or-nothing publication.** All N lines are built in a
   classic-locale buffer and inserted once (359-366) only after full
   validation; a validation/model/input/CUDA/non-finite failure writes no
   prediction byte; the success record is emitted only afterwards (412-415).
   `main.cu` maps `invalid_argument`->2 and other standard failures->1
   (`main.cu:56-63`), imbues classic locales (52-53), and routes train
   diagnostics to `std::cerr` (35).
6. **Profiling honesty structure.** Per-batch records are formatted after the
   final sync, cover index 1..B including the first batch, and use the actual
   stage windows (upload 274-281, prepare 286-291, forward+argmax 293-300);
   aggregate GPU values sum those same per-batch durations (345-356); host
   buckets are separate monotonic walls and enqueue is named
   `host_enqueue_batches_ms` (302, 405-407), never "GPU execution".
   `docs/course-interface.md:127-152` states GPU-event vs full-process
   separation, no warmup substitution, and the >=100 ms failure rule.
7. **Infer separation is real, not just CMake intent.** `infer` links only
   `course_runtime` (`CMakeLists.txt:117-123`); the recorded link line
   (`task-4-link-macro-audit.txt:3`) contains only
   runtime/host/checkpoint/forward/parameters/reference-forward archives and
   no training/LeNet/loss/metrics/Adam/old-workflow object. `runtime.cu` has
   no `ReadLabels`, softmax, or accuracy reference (grep: no matches), so the
   shared-TU unused-helper caveat does not become a hidden label read.
   Train/infer/runtime are unconditional (85-88, 117-123); only the test target
   and `add_test` are inside `COURSE_BUILD_TESTS` (144-162); existing targets
   are untouched (diff is additive).
8. **Env isolation and exit taxonomy are tested end to end.** Infer reads only
   `COURSE_MODEL_DIR`/`COURSE_INFER_BATCH`/`COURSE_PROFILE`
   (`main.cu:37-39`); a conflicting `COURSE_INPUT_MODE`/`COURSE_EPOCHS` pair
   must produce byte-identical stdout (`runtime_tests.cu:928-985`). Fourteen
   real subprocess/direct cases cover exact argv, missing/empty model dir,
   extra labels arg, missing/malformed/bad-CRC/corrupt bundles, all three
   persisted modes, N=1/17/1025 with batch-17 invariance, batch bounds
   1/16384/0/16385/-1/abc, profile-every-batch, fast-path silence, poison
   early/later independence, and direct-API exception taxonomy. Children get
   a cleaned `COURSE_*` environment, an exclusive scratch root, absolute
   fixtures, and pipe capture (119-326 subprocess helpers).
9. **Evidence matches the claims I could check without rerunning:** four
   suites 48/8/9/14 pass (`task-4-offset-green-ctest.txt`), default minimal
   build + bounded roundtrip with labels withheld and `warning_count=0`
   (`task-4-default-roundtrip.txt`), named missing-entry RED
   (`task-4-missing-entry-red.txt`), and no personal/worktree/`/gemini` path
   in any of the six changed files (grep: no matches).

## Findings

### F1 (Important) Buffered stdout sink failure is undetected; a false `status=pass` and exit 0 are produced

Evidence:
- `course/runtime.cu:366` inserts the whole payload into `predictions`, then
  `:367-369` tests `!predictions` and throws. For `std::cout` (stdio-synced)
  and for any buffered `std::ostream`, a small payload is accepted into the
  buffer here; the failing write happens later, at flush/exit, after the return
  value is fixed. The diagnostics counterpart (`:412-415`) has the same shape
  (harmless for `std::cerr`, which is unit-buffered, but not for a buffered
  direct-API sink).
- Documented contract that this violates:
  `docs/course-interface.md:110-111` ("Exit 1: ... output stream failure"),
  `:122-123` ("A downstream sink I/O failure is reported"),
  `course/runtime.h:17-19` (same), `:43` (`runtime_error` for a prediction
  stream failure), `:55` ("Returns 0 only after ... has been written").
- Observed: `task-4-dev-full-probe.json` - actual minimal `infer` N=4 with
  stdout on `/dev/full` returns 0 and writes
  `event=course_infer status=pass samples=4 batches=1 input_mode=unit` to
  stderr, with zero delivered predictions. `task-4-default-roundtrip.txt:2`
  records `dev_full_output_failure_exit_observed=0`.

Independent classification: this is a real contract defect, not a probe
artifact, but it is not Critical - prediction content is correct and no
processing requirement is bypassed; only sink-failure scenarios (full disk,
quota, closed writable sink) are affected. It is Important because the program
can report success while delivering no output, and the failure is
size-dependent (payloads whose `xsputn` stays inside the stdio buffer escape;
large payloads can be caught), so behavior is nondeterministic across N.
Severity is driven by the false pass record, which would mislead exactly the
acceptance flow that reads stdout/stderr/exit.

Minimal fix (either is acceptable; code fix preferred): after the single
insertion, `predictions.flush(); if (!predictions) throw
std::runtime_error(...)` before formatting/writing the success record, and do
the same for `diagnostics`; the "write once" property is preserved. If main
prefers not to promise detectability, then `docs/course-interface.md:110-111`,
`:122-123`, and `course/runtime.h:17-19,43,55` must be narrowed instead
(buffered sink failures may surface only at process-exit flush and do not
change the exit status). Add a direct-API case with a deliberately failing
`std::streambuf` to pin the chosen behavior (see F3).

### F2 (Minor) `docs/course-interface.md:94-96` calls the unit default "the course convention"

Text: "The `unit` default is the course convention; it has not been
independently confirmed against an external deployment corpus." The approved
specification (global constraint 7; design record) is a user-approved project
assumption with official preprocessing still unconfirmed. The second clause
already disclaims confirmation, so the risk is only that a reader infers an
external course-level convention exists. Suggest "the approved project default
(an assumption, not an externally confirmed course convention)". Not a
behavioral defect; no code change.

### F3 (Minor) No test covers sink-failure behavior

`runtime_tests.cu` asserts zero stdout on argument/model/non-finite failures
and full canonical stdout on success, but never exercises a failing
sink, which is why F1 survived the suite. A direct-API case with a streambuf
that returns a short/error result on `xsputn`/`sync` would fail today and pin
the fix. This belongs with F1.

### F4 (Minor) The default `COURSE_INFER_BATCH=1024` is not behaviorally pinned

Observed batching only bounds the default: N=17 -> `batches=1`
(`runtime_tests.cu:991-1062`) and N=1025 -> `batches=2`, which holds for any
default in 513..1024. The only exact check is a self-assertion on a local
constant (`runtime_tests.cu:1181`), not on program behavior. The default
itself is Task 3 code and was reviewed there, so impact is low; an N that
only splits at 1024 (e.g. counting batches for N=1025 *and* asserting the
partial batch shape at a capacity-sensitive boundary, or a profile record for
a 1024-sample first batch) would remove the gap.

### F5 (Minor) Subprocess harness robustness: no timeout; fork-after-CUDA is order-coupled

`RunProcess` waits with `::poll(..., -1)` (`runtime_tests.cu:280`), so a hung
child blocks the suite until CTest's own timeout (recorded 10000000 s in
`task-4-offset-green-ctest.txt` is effectively no backstop) rather than
failing fast. Separately, all fork-based cases are registered before the two
direct-API cases that initialize CUDA in the parent
(`runtime_tests.cu:1349`, `1431`); forking after a CUDA context exists is not
supported by CUDA, so the current order is load-bearing and undocumented. Both
are latent, not present failures: adding a bounded poll/wait timeout and a
one-line note (or an ordering guard) would make future edits safe.

### F6 (Informational) Profile bucket interpretation nuance

`host_enqueue_batches_ms` (`runtime.cu:302`) wraps the enqueue loop, which
issues pageable H2D `cudaMemcpyAsync` from `std::vector<float>` (277-281). On a
non-blocking stream the driver may stage pageable copies synchronously, so for
large batches this "host enqueue" bucket can contain transfer staging time.
The documented claim is still accurate ("host enqueue cost, not device
execution", `docs/course-interface.md:142`; `runtime.h`), and GPU event stages
remain device-side measurements, but readers should not treat the bucket as
pure API overhead.
No change required; optionally note it where the buckets are described.

Also informational, not findings: the aggregate-vs-per-batch GPU sum assertion
in the profile test is self-consistent by construction (both come from the
same computed doubles), so the count/index/`batch_samples` assertions are what
actually prove first-batch inclusion; `task-4-report.md:66` still says
"13 cases" while 14 exist (report metadata drift, controller record is
correct).

## Requirement checklist (brief + global constraints)

| Requirement | Status | Evidence |
| --- | --- | --- |
| Exact `RunCourseInfer(options, predictions, diagnostics)` declaration | Met | `course/runtime.h:79-80`, `runtime.cu:194-196` |
| Macro-exclusive root main, positional/env parse, exit 1/2 | Met | `main.cu:9-63`; no-macro and both-macro compile RED recorded |
| `train`/`infer` default CMake targets; tests optional | Met | `CMakeLists.txt:85-88,117-123,144-162` |
| Infer does not link old workflow/trainable LeNet backend | Met | link line `task-4-link-macro-audit.txt:3`; no `ReadLabels`/softmax/accuracy in `runtime.cu` |
| Host model+input validation before first CUDA allocation/device selection | Met | `runtime.cu:196,215,219-225`; `src/checkpoint.cpp:302`; `interface.h:95-108` |
| Device 0, no stdout banner | Met | `runtime.cu:237`; `cuda_check.h` throws only |
| Buffers/model/stream created once | Met | `runtime.cu:238-247` |
| Sticky flag initialized once; all-N predictions at global offsets; one readback+sync | Met | `runtime.cu:250,268-269,294,308-314,317-327` |
| No allocation/readback/sync/events in fast loop | Met | `runtime.cu:263-301,253-256` |
| Events preallocated, every batch incl. first, honest host/GPU labeling | Met | `runtime.cu:253-256,274-300,345-356,405-407`; `docs:127-152` |
| Only after successful validation format all N lines and publish once | Met | `runtime.cu:317-327,359-366` |
| No labels/softmax/accuracy kernels in inference | Met | grep of `runtime.cu`; `runtime_tests.cu:1119-1145` |
| stdout exactly N digits; all diagnostics stderr; 0/1/2 taxonomy | Met (except F1 sink case) | `docs:103-111`; tests `991-1062,1188-1281` |
| `COURSE_INPUT_MODE` ignored at infer, persisted mode only; env split | Met | `main.cu:31-42`; `runtime_tests.cu:928-985` |
| `COURSE_MODEL_DIR` required, no fallback/hidden labels | Met | `interface.h:78-93`; `runtime_tests.cu:859-889` |
| Poison early batch + later finite batch (sticky retention) | Met | `runtime_tests.cu:1282-1347`; sticky semantics `gpu_ops.h:50-58` |
| Batch capacity reuse must not change per-sample input (offset fix) | Met | `runtime.cu:268-269,277-281`; RED `task-4-offset-red.txt`, GREEN ctest |
| Docs: exact env/defaults/meta, shapes, no online preprocessing verification | Met except F2 wording | `docs:55-99` |
| Docs: full process vs internal GPU timing, no warmup substitution, arbitrary paths | Met | `docs:127-165` |
| Default minimal build produces train/infer without tests/tools; roundtrip | Met | `task-4-default-configure/build/roundtrip.txt` |
| Default build warning-free; all four suites pass | Met (H20-recorded) | `task-4-offset-green-build.txt`/`-ctest.txt` 48/8/9/14 |
| No change to existing Make/CLI behavior | Met | diff touches only CMake additions; Makefile untouched |

## Unchanged-source checks (named concrete risks only)

- `course/gpu_ops.h:44-71`: launcher signatures match the calls; `augment=false`
  makes the null indices/seed/epoch arguments valid; `LaunchArgmax` never
  clears the sticky flag and examines all rows - this is what makes the poison
  test meaningful.
- `course/inference.h:40-70`: `Forward` allocates nothing and does not
  synchronize; capacity must be >= batch (runtime passes the configured batch
  size, and each batch uses `current_batch <= batch_size`); the constructor's
  one internal sync is before the loop.
- `course/interface.h:56-118`: `InferOptions` defaults (batch 1024, profile
  false); `ParseInferArgs` argc/exit contract; `ReadImages` bounds/finiteness
  and "file size checked before allocating".
- `course/model_io.h:24-40`: CRC/format/invariant validation means a malformed
  or mixed bundle fails before CUDA (tests confirm exit 1, empty stdout).
- `include/checkpoint.h:36,51` + `src/checkpoint.cpp:302`: parameter finiteness
  is already enforced inside the host loader, so the composited model
  validation is genuinely before `cudaSetDevice`.
- `include/tensor.h:154-219`: `DeviceBuffer<T>(count)` is element-counted,
  throws `std::runtime_error` on allocation failure, and `cudaMalloc/cudaFree`
  keep their runtime-defined synchronization; destruction order in
  `runtime.cu` (stream declared first -> destroyed last) is safe.
- `course/tests/test_support.h:384-426,436-470`: exclusive scratch root,
  `executed > 0` rule, and the exact `executed=... failures=... status=...`
  line the CTest regex consumes; `EXPECT_*` bind operands once and failures
  are non-fatal counted failures.

## Recommended order for main

1. F1: flush-and-check both sinks (or narrow the contract in header+docs), and
   add the failing-sink direct-API case (F3). Re-run all four suites + the
   `/dev/full` probe; the probe must then show exit 1 with no success record.
2. F2: one-sentence documentation rewording.
3. Optional before acceptance: F4/F5 hardening, F6 note.
