# Warm Full-Request Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure5warmups+101full10000-image repetitions including H2D, GPU
preparation/forward/argmax, D2H, synchronization and validation; publish truthful
mean/median statistics without changing official interfaces or retraining.

**Architecture:** Optional development CUDA tool reuses reviewed GPU components;
stdlib external collector validates raw timing/predictions and hashes. Warm
packager reuses cold package publication safeguards with a new explicit schema.

**Tech Stack:** C++14/CUDA14, existing project-owned FP32 kernels, Python stdlib;
existing development Pillow only for PNG generation.

**Spec:** `docs/superpowers/specs/2026-10-27-warm-full-request-benchmark-design.md`

## Global Constraints

- Same detached isolated workspace; main H20 execution/review/checkpoint refs;
  coding/review subagents `deepseek/deepseek-flash`, no worker SSH/git changes.
- No NN numerical/kernel/architecture/official train-infer argv changes, CPU NN
  fallback, fast-math, mixed precision, library install or platform changes.
- Use actual existing20epoch course model, full10000 images, batch1024; label
  access external-only. Keep original files and all old artifact versions.
-5warmups+101measured rounds for actual acceptance. Recopy/recompute/read back
  every round; time synchronized complete requests, not GPU enqueue/events only.
- Include flag reset, H2D, GPU preparation/forward/argmax, D2H, synchronization,
  prediction-range/finite-flag/invariance checks; exclude/disclose one-time setup,
  input/model disk loading and final formatting. Preserve every duration.
- New schema `course-warm-benchmark-v1`; never relabel cold data. Mean+median
  must both be<100ms for warm-target success; accuracy>=9600/10000 independently.
- Official submission remains34runtime sources+3root artifacts, tests/tools/
  Python/data/models excluded; decimal caps and no personal paths/credentials.
- Main proves RED before implementation, reviews each task, preserves real
  HEAD/index and creates fresh checkpoint refs; no push/deploy/upload.

## Review Focus

- Asynchronous under-timing: end timestamp must follow D2H+stream sync and host
  validation; collect positive host wall for every actual completed request.
- Cached-result shortcut or wrong tail offsets: deterministic zero/one fixture
  at batch1 vs2 and N17/1025 partition checks; all rounds redo whole pipeline.
- Warmup/sample contamination: exactly5phase=warmup and101phase=measure records,
  statistics only measured records, but retain warmup evidence.
- Warm output corruption: any poison/nonfinite/range/invariance failure aborts
  before predictions/final success; collector/packager rejects partial records.
- Stale or relabelled provenance: strictschema/scope/count/raw-record/hash gates,
  no accepting cold JSON or copying old timings into new report.

---

### Task 1: Optional GPU full-request repetition tool

**Status:** Complete; independentSpec/QualityApproved, H20all5CTest48/8/9/15/7
PASS, defaultminimalbuild/link/sinkchecksPASS; backup `61c9e14` /
`backup/warm-benchmark-task-1-20261028`. ActualfullW5R101series belongsTask2.

**Files:** Create `course/tools/warm_benchmark.cu`,
`course/tests/warm_benchmark_tests.cu`; modify only optional part of `CMakeLists.txt`.

**Interfaces:** executable `course_warm_benchmark images.f32 N C H W K warmups runs`.
Required model dir/batch via existing COURSE_MODEL_DIR/COURSE_INFER_BATCH; ignore
training/profile variables and persist model mode. Exactargc9, positive decimal
counts<=100000, warmups>=1/runs>=1 for bounded tests. Actualmain uses5/101.

Stdout exactly N final class lines once after all rounds succeed; no labels.
Each round emits stderr outside its timing window:
`event=course_warm_iteration phase=warmup|measure iteration=I samples=N wall_ms=T predictions_identical=1`.
Indices1-based separately perphase. Final stderr:
`event=course_warm_complete status=pass warmups=W runs=R samples=N batch=B input_mode=MODE`.
Use classiclocale/17significantdigits; retain warmup records too.

- [ ] Tests-first declaration/CLI-contract handoff; main missingtarget/entry RED.
- [ ] Validate all host options/bundle/images before CUDA; explicit device0,
  one model/stream/preallocated buffers, constructor sync before warmup.
- [ ] Each round timestamp before flag reset, all batches H2D offset*784,
  prepare, forward, globalpredictionoffset argmax, allN+flagD2H, one streamsync,
  host flag/range/invariance checks, then timestamp. No allocation/events inside
  timed round; nonfinite anyround fails, no caching/skipping/tinyN substitutes.
- [ ] Real GPU tests: exact2warm/3measure records, literal[1,0] zero/one atB1/B2,
  N17/1025 tails, malformed counts/env/metadata/input fails, earlypoison fails
  before stdout; officialinfer predictions match. CPU NN math prohibited.
- [ ] CMake targets inside COURSE_BUILD_TESTS only, tool links host+forward not
  training; CTest bounded timeout. Main actualbuild/tests+all existing CTests.
- [ ] Independent review and fresh task checkpoint without realindex/HEAD moves.

### Task 2: Warm external collector, raw schema and actual101-round measurement

**Files:** Create `scripts/course_warm_benchmark.py`,
`tests/test_course_warm_benchmark.py`; update usage/results with actual data only.

**Interfaces:** `run_benchmark(warm_exe, infer_exe, images, labels, model_dir,
output_json, count, batch=1024, warmups=5, runs=101, model_source="course_train")`.
CLI same6paths then `--count`, `--batch`, `--warmups`, `--runs`, `--model-source`.
One real toolprocess, labels neverpassed to it, envmodel/batch explicit.

NewJSON records schema, timing_scope=`warm_full_request_h2d_prepare_forward_argmax_d2h_validation`,
paths/count/batch/mode/source/warmups/runs, sha256binary(tool)/official_infer/model/
meta/images/labels, raw stdout/stderr/exit/process_wall_ms, warmup_records,
measured_records (all raw roundfields), predictions_identical_all, summary mean_ms/
median_ms/min_ms/max_ms/correct/accuracy/accuracy_pass/mean_below_100ms/
median_below_100ms. Round records carry iteration/samples/wall_ms/predictions_identical.

- [ ] Stdlib tests first; main namedmissingmodule RED. Harmless processfixtures
  test parsing/errors only, no mock GPU success claims.
- [ ] Validate exactphasecounts/indices/fullN/finitepositive times/successrecord/
  predictioninvariance/actual N stdout labels count. Reject extra/missing/duplicate
  round records and any failure; retain raw failure output in exclusive JSON.
- [ ] Preserve all times and compute mean/median/min/max from measured records,
  not process wall or warmups; hashes before ANDafter processmustmatch. Existing
  output/alias refuses before launching, timeout generous bounded.
- [ ] Main actualH20 existingcourse-trainedmodel/testunit/fullN10000/B1024/W5/R101;
  independently parse rawstderr/compareexpectedpredictions/originaltestlabels/
  currenthashes; recordaccuracy andbothwarmtiminggates separately from coldwall.
- [ ] Independent task review/checkpoint; no primary result numbers until measured.

### Task 3: Warm result packaging and final acceptance

**Files:** Create `scripts/course_warm_package.py`,
`tests/test_course_warm_package.py`; add minimal schema dispatch to
`scripts/course_package.py`; update docs and regenerate root artifacts after gates.

**Interfaces:** Existing packageCLI unchanged; dispatch warm-schema JSON to
`run_warm_package(benchmark_json, training_log, output_zip, root=None)`.
Reuse original whitelist/CRC/training/publication helpers. Preserve old cold
schema/API/tests and its historical package output, no fake cold normalization.

- [ ] Write warm-only ownedfixture tests; main module/schema dispatch RED before
  implementation. Cold70tests remain intact.
- [ ] Require actualcourse_train/default20 completion/current6hashes, supportedmode,
  N10000,W>=1,R>=101, exactrawrecords/warmups/summary independentlyrecomputed,
  predictioncorrect>=9600, successfulsingletoolprocess and allround invariance.
- [ ] PNG oneaccuracy/onetimegroup `Test Time: mean X ms; median Y ms`, explicit
  warmed5/101/full10000/scope/range/gate; reportall101times+warmupcount/included/
  excludedstages/currentidentities/actualtraining/preprocess/leaderboardcaveats.
- [ ] WarmtargetPASS only meanANDmedian<100ms; truthfullypackagefailure otherwise.
  Exclusivearchive/rootfiles and decimalcaps unchanged; preservedoldartifacts.
- [ ] Main tests, independentreview, actualwarmartifactgeneration, freshZIPextract/
  defaultbuild/officialboundedroundtrip/validmodelinference/protectedhashes.
- [ ] Wholeamendmentreview; oneconsolidatedfixwaveifneeded, finalfreshbackupref,
  reportbothwarmedmean/median andscope withoutclaimingcoldstartupgoalchanged.

## Self-review / execution

All5reviewfocuses have task-owned tests. ExistingGPUclass/input/argmax contracts
are consumed unchanged; new counts/protocol/schema are fixed above. Submission
defaultbuild remains independent of newtools. Existingworktree and continuous
subagent-driven execution are explicitly retained from the user's approved
workflow; this timing amendment was confirmed directly, no new method selection.
