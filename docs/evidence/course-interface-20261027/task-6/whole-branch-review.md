# Whole-branch final review — course GPU interface

Reviewer: fresh whole-branch Senior Code Reviewer (read-only).
Scope: `e6301b2..49d79b7` (`backup/course-interface-plan-20261027` to
`backup/course-interface-task-6-20261027`). I read all 12,716 lines of
`.superpowers/sdd/2026-10-27-course-gpu-interface/whole-branch-review.diff` in
contiguous chunks, and reviewed the approved design
(`docs/superpowers/specs/2026-10-27-course-gpu-interface-design.md`), the
binding constraints, the Task-6 controller handoff and the three Task-6 JSON
evidence files. I also independently verified the current root artifact hashes
and the archive member list (37 members; `41e6ef0451b032de33cbd564fd1b58d50063deaf11359df67c1795ef4e45276a`),
and inspected the named `checkpoint`/`layers` boundaries where risk was
concrete. I did not diff binary PNG/ZIP or curated raw prediction JSON, and I
did not rerun any suite or remote measurement.

## Strengths (first)

- **Strict interface discipline.** `ParseTrainArgs`/`ParseInferArgs` enforce
  exact argc (program + 8 / program + 6), unsigned decimal parsing with
  overflow checks, exact supported geometry, N>=2 for training, explicit
  mode/epoch/batch bounds, and no label path for inference
  (`course/io.cpp:131-260`; `course/interface.h`).
- **Raw-I/O safety.** Images and labels use exact byte-size checks before
  allocation, little-endian scalar validation, finite/range checks, full
  int32 label validation before lossless uint8 narrowing, and read-only
  inputs (`course/io.cpp:262-348`). No input is quantized through uint8.
- **Model handoff is coherent and failure-aware.** `lenet.bin` keeps the
  original checkpoint format; `course.meta` is a strict four-line versioned
  file with shape, persisted mode and CRC32. The loader validates metadata,
  checkpoint invariants, all weights, normalization constants and CRC
  (`course/model_io.cpp:1-416`; `course/model_io.h`). Publication uses
  same-directory temporaries and renames metadata last; a mixed pair is
  rejected by CRC. Symlink/hard-link reserved destinations are refused,
  unrelated work-directory entries are preserved, and a valid previous pair
  is the only replaceable case.
- **Inference is genuinely forward-only.** One fixed, checked arena holds the
  ten supplied parameter payloads, forward activations and pool winners; no
  gradients, moments, random initialization, zero-fill or per-batch device
  allocation (`course/inference.cu:1-281`). The original numerical kernels are
  reused without arithmetic changes. GPU tests require bit-identical logits
  against the reference LeNet for batches 1, 17 and 1024.
- **Inference runtime has the right publication boundary.** Host options,
  model bundle and every input value are validated before device selection;
  one device-0 stream, one arena, one set of batch-capacity buffers, all-N
  prediction storage and one sticky flag are allocated once; batches are
  enqueued with global prediction offsets; one D2H readback/synchronization
  follows; validation failures and the sticky non-finite flag produce zero
  prediction bytes; stdout is formatted and flushed once
  (`course/runtime.cu:190-426`). No device allocation or readback occurs in
  the batch loop (`course/runtime.cu:268-305`).
- **Argmax and sticky flag are correct by construction.** Smallest-index ties
  use strict `>`; every logit in every row is scanned; `atomicExch` writes a
  sticky 1 that is zeroed exactly once per request; the host rejects the
  request before publishing anything (`course/gpu_ops.cu:94-129,222-258`;
  `course/runtime.cu:250-321`).
- **Training is supplied-data-only and validation-selected.** The workflow
  reads exactly the caller's images/labels, uses the original LeNet
  forward/backward/AdamW and split/shuffle helpers, uses deterministic
  translation augmentation with normalized-black padding for every mode,
  chooses only by strict validation improvement, saves/reloads/validates the
  best bundle and emits no test accuracy (`course/training.cu:1-671`).
  There is no old `RunTrain` call and no external test split.
- **Build/link separation is real.** `infer` links `course_runtime` and the
  forward-only path; it does not link the trainable LeNet, loss/metrics/Adam,
  old CLI or old workflow (`CMakeLists.txt:119-176`; Task-6 link audit and
  actual fresh build). Default build is tests-free; optional tests/tools are
  explicitly gated (`CMakeLists.txt:127-216`).
- **Tests are unusually strong for this class of change.** Tests-first
  contracts cover host parsing/size/range/metadata/aliases, GPU mode/
  translation/tie/sticky/non-allocation behavior, bounded real training,
  official executable subprocess contracts, sink failures, and helper
  packaging. Recorded evidence: 66 helper tests with no skips (Pillow only in
  development), 48/8/9/15 course CTest groups, 0 warnings, old host/workflow/
  operator/compliance/Makefile suites passing except the documented missing
  `pyarrow` and prior Sanitizer self-disable.
- **Actual acceptance is real and honestly reported.** Fresh default training:
  20 ordered epochs, 594.740494 s, best epoch 15, validation 0.992399991,
  CRC `f473014f`. Five unprofiled full-process final-model runs: all 9911/
  10000 correct, identical predictions, walls 1684.153743 / 1641.133869 /
  1659.597542 / 1637.525734 / 1657.407840 ms (median 1657.407840, max
  1684.153743). Accuracy PASS; the full-process <100 ms goal is explicitly
  FAILED. The byte identity to the preserved legacy checkpoint is presented
  as an observed deterministic-retraining outcome, not as substituted
  provenance.
- **Package evidence is coherent.** 37 members (34 whitelisted sources +
  evaluation.json/report/result.png), 110545 compressed / 327328 unpacked,
  PNG 900x320 / 19932 bytes; fresh extraction default build without Python/
  tests/tools passed, bounded official train->infer passed with labels
  withheld, and the actual final model reproduced the measured predictions
  from an images-only directory. The rebuilt binary hash differs and the
  original timing series was not reassigned to it.
- **No scope violations observed in the diff.** Original sources and the old
  Make workflow are untouched; no CPU NN fallback, fast-math, mixed precision,
  third-party NN library, hidden label path, fallback weights, deployment or
  upload was added. Current root artifacts match the recorded hashes, and the
  real HEAD/index were not part of this review's changes.

## Critical

None. I found no correctness, safety, contract, or package-blocking defect in
the reviewed production or helper code.

## Important

None. The 100 ms failure is a declared performance outcome, not a hidden code
defect; it is treated separately below. The two submission-artifact gaps are
Minor because the submitted numbers are conservative (they overstate time,
not understate it) and the FAILED verdict is not disguised.

## Minor

### M1 — Report does not enumerate the timing stages required by spec §10

- **Where:** `scripts/course_package.py:690-691,743-745,764,770-772`; the
  submitted root `report` says only `Test Time: 1657.41 ms (median of 5
  full-process runs)` and `full-process <100 ms: FAILED`.
- **Why:** the approved spec (§10) requires, while leaderboard timing scope is
  unknown, that the report make explicit **which stages** the reported time
  includes (startup, loading, CUDA setup, transfers, computation, validation,
  output, exit). `docs/course-interface-results.md:240-241` and
  `docs/course-interface.md:147-152` contain that explanation, but those docs
  are not in the 37-member ZIP. A reader of the submitted report alone sees a
  label, not the boundary. The report does correctly say the leaderboard
  scope is unresolved and does not claim 100 ms.
- **How:** add one concise sentence to `_report_text` listing the included
  full-process stages and stating that GPU event substages are separate.
  Regenerate report/PNG/ZIP and recheck hashes.
- **Severity/disposition:** Minor; **must fix before completion** for strict
  submission compliance, despite no false success claim.

### M2 — PNG states the median but not the five-run range

- **Where:** `scripts/course_package.py:687-692,716-718`; the PNG is drawn
  from `metric_lines`, whose only time text is `Test Time: 1657.41 ms
  (median of 5 full-process runs)`. The report lists all five runs and the
  max, but the image does not.
- **Why:** spec §10 says the image's `Test Time` must use the five-run
  full-process median **and state the range**. The image is the primary
  result artifact; omitting the range (1637.53–1684.15 ms) is a disclosure
  gap even though the reported median is conservative and FAILED is shown.
- **How:** extend the image time line (or add a second small line in the same
  single metric group) with `range 1637.53–1684.15 ms; max 1684.15 ms`;
  update `tests/test_course_package.py` expectations; regenerate and recheck.
- **Severity/disposition:** Minor; **must fix before completion**.

### M3 — Report labels unit preprocessing as an assumption but not as officially unconfirmed

- **Where:** `scripts/course_package.py:743-744` emits `Preprocessing
  assumption: caller supplies unit floats ...`; the explicit sentence
  "not independently confirmed official preprocessing" exists only in
  `docs/course-interface-results.md:193-194`, which is not in the ZIP.
- **Why:** the approved design (§2) requires that unit input never be
  presented as confirmed course preprocessing. "Assumption" is honest, and
  the report does not claim official confirmation, but the standalone
  submitted report would be stronger with the explicit caveat.
- **How:** append “official course preprocessing remains unconfirmed” to the
  assumption line while regenerating M1/M2.
- **Severity/disposition:** Minor; **acceptable debt** as written (the word
  “assumption” is present and no confirmation is claimed), but fold into the
  same fix wave.

### M4 — Duplicate max-time field in the report

- **Where:** `scripts/course_package.py:760-761` emits both
  `Max full-process wall: 1684.154 ms` and `max_ms=1684.154`.
- **Why:** cosmetic duplication; no numeric conflict, but it looks like two
  separate metrics and makes the report noisier.
- **How:** remove the extra `max_ms=` line when regenerating.
- **Severity/disposition:** Minor; **acceptable debt**, fix opportunistically.

### M5 — Release flags replace caller-provided release flags

- **Where:** `CMakeLists.txt:26-31` sets `CMAKE_CXX_FLAGS_RELEASE` and
  `CMAKE_CUDA_FLAGS_RELEASE` to `-O2 -DNDEBUG` and adds global `-O2`.
- **Why:** a caller who configures different release flags has them replaced.
  This matches the project's stated O2 baseline and does not affect the
  measured default build, but it is not a transparent CMake interface.
- **How:** if desired, append target options or only default when the cache
  value is unset. No correctness change required.
- **Severity/disposition:** Minor; **acceptable debt** (deferred Task-2 item
  remains valid; default build is the acceptance configuration).

### M6 — Package artifacts are written at final paths, not temp+rename

- **Where:** `scripts/course_package.py:815-839,858-872`. Root artifacts and
  the ZIP are created with `O_EXCL`, written, flushed and fsynced, and removed
  on caught exceptions; a process crash/power loss can still leave a partial
  final artifact.
- **Why:** the model publication path correctly uses temp+rename, but the
  development packaging helper does not. Successful output is independently
  hashed/verified; this is a hardening gap, not an observed defect. The helper
  is not part of the submitted runtime.
- **How:** write to a same-directory temporary and rename, or document the
  crash limitation explicitly.
- **Severity/disposition:** Minor; **acceptable debt** for a development-only
  helper; the successful artifacts were verified.

### M7 — Fresh whitelist build/roundtrip was verified after package generation

- **Where:** spec §10 (`docs/superpowers/specs/2026-10-27-course-gpu-interface-design.md:122`)
  says to verify a fresh whitelist CMake build and train->infer closure
  **before** generating submission artifacts. The controller sequence at
  `controller-task-6-verification.md:23,29,31` generated the root artifacts/
  archive first, then performed the fresh extraction/build/roundtrip.
- **Why:** this is a process-order deviation, not a content defect. The
  extracted default build, bounded official roundtrip and final-model
  images-only inference all passed (`task-6-extracted-roundtrip.json`), and
  no upload/deployment occurred, so no invalid artifact escaped.
- **How:** for future runs, perform the whitelist extraction build first and
  record that ordering; no retrospective artifact change is needed.
- **Severity/disposition:** Minor; **acceptable debt** with the evidence
  caveat stated.

### M8 — Temporary model path is checked, then created without O_EXCL

- **Where:** `course/model_io.cpp:148-166` (`UniqueTemporaryPath` lstat loop)
  and `course/model_io.cpp:382-412` (SaveCheckpoint to that path).
- **Why:** a concurrent process could create the same PID/sequence-derived
  path between the check and write. The PID+sequence scheme and the declared
  sequential course contract make this impractical; the header explicitly
  does not claim transactional isolation from concurrent writers. There is no
  input-alias overwrite because only reserved exact names are validated and
  an existing reserved input file fails the pair check.
- **How:** open temporaries with exclusive create semantics if hardening is
  desired.
- **Severity/disposition:** Minor; **acceptable debt** (documented concurrency
  boundary).

### M9 — Mixed-pair publication can lose the previous valid pair

- **Where:** `course/model_io.cpp:403-411` renames checkpoint first, metadata
  second; a failure between them leaves the new checkpoint with old metadata.
  The loader rejects the mixed pair via CRC, but the previous valid bundle is
  not restored.
- **Why:** the design/header explicitly says no rollback and no transactional
  isolation is claimed; it promises only that a mixed pair is not treated as
  a valid new model. The sequential course workflow is the intended caller.
- **How:** none required for this iteration; a future transactional generation
  scheme could preserve the old pair.
- **Severity/disposition:** Minor; **acceptable debt** (approved
  publication-atomicity ruling).

### M10 — Post-publication sink failure cannot roll back stdout

- **Where:** `course/runtime.cu:359-375` writes and flushes predictions, then
  throws on a flush failure; `course/runtime.h` documents that bytes already
  accepted by the sink cannot be rolled back.
- **Why:** all validation/model/input/CUDA/non-finite failures occur before
  publication (the required zero-output guarantee); arbitrary downstream I/O
  failure is a different class. The narrow wording is honest.
- **How:** none required; retain the documented boundary.
- **Severity/disposition:** Minor; **acceptable debt** (approved ruling).

### M11 — Test harness has no timeout; subprocess poll is unbounded

- **Where:** `course/tests/runtime_tests.cu:281` uses `poll(..., -1)`;
  `CMakeLists.txt` registers course tests without a `TIMEOUT` property. The
  fork-based cases also rely on definition order (`runtime_tests.cu:1477`
  keeps the direct-CUDA sink case last).
- **Why:** a hung child or regression could hang CTest indefinitely rather
  than fail promptly. All recorded runs completed; this is test-infrastructure
  robustness, not production behavior.
- **How:** add a bounded poll/overall deadline and a CTest timeout; optionally
  make ordering explicit via a separate executable.
- **Severity/disposition:** Minor; **acceptable debt** (deferred Task-4 item;
  acceptance suites completed).

### M12 — Packaging helper has no failure-injection/rollback test

- **Where:** `tests/test_course_package.py` exercises successful publication,
  pre-existing paths, missing Pillow and many validation failures, but not a
  mid-publication write/rename failure. `scripts/course_package.py:858-872`
  removal cleanup is therefore not directly covered.
- **Why:** the successful artifact and all validation paths were verified;
  the untested branch is cleanup after an I/O failure in a development helper.
- **How:** if the helper is retained, add a deterministic failing-writer test;
  otherwise accept the documented limitation.
- **Severity/disposition:** Minor; **acceptable debt** (deferred Task-6
  failure-injection item; no production path).

## Explicit spec compliance

### Functional/task requirements — PASS

- GPU CUDA FP32 only; no CPU NN fallback, fast-math, mixed precision,
  third-party NN library, platform preload/bypass, hidden labels, fallback
  weights, old inference workflow or old training workflow in `infer`.
- Fixed LeNet 1x28x28 topology, original numerical kernels/order, FP32, and
  smallest-index argmax ties; bitwise logits checks for batches 1/17/1024.
- Exact `train` 8-position and `infer` 6-position contracts; strict integer/
  shape/batch/epoch/mode parsing; exit 2 usage, exit 1 runtime, exit 0 success.
- Exact raw image/label sizes, overflow checks, finite/range validation,
  lossless label narrowing, read-only inputs, caller-owned work directory,
  no label path in inference.
- Persisted unit/byte/normalized mode; inference uses model metadata only and
  ignores runtime mode/epoch variables; normalization contract enforced via
  the original checkpoint loader.
- Versioned bundle `lenet.bin` + `course.meta`, strict metadata grammar,
  CRC binding, valid-pair-only replacement, symlink/hard-link refusal,
  unrelated-entry preservation, no silent fallback.
- Forward-only arena (no gradients/moments/random init/zero-fill), preallocated
  prediction/flag storage, no per-batch device allocation or readback, one
  final readback/sync, buffered zero-output validation failures, single stdout
  publication.
- Supplied-only training, deterministic split/seed, validation-only best
  selection, best reload/CRC validation, stderr diagnostics, no test accuracy.
- Full-process external timing from process start through exit; profile stages
  separately labeled; five unprofiled final runs; hashes recorded; no warmup
  substitution.
- Packaging whitelist, 37-member archive, size/pixel caps, exact
  `evaluation.json`, report/PNG with one metric group, no data/build/tests/
  tools/Python/symlinks/credentials, `<100 MB`/`<300 MB`.
- Original 67 sources and protected remote hashes remain unchanged according
  to the supplied evidence; real HEAD/index untouched; no upload/deploy.

### Submission disclosure — PARTIAL (M1/M2; M3 optional)

The report/PNG are honest about the failed 100 ms target and do not fabricate
success, but the report does not enumerate included timing stages and the PNG
does not state the five-run range as spec §10 requires. These are small
artifact-text fixes, not code defects.

### Performance objective — FAIL (declared, not hidden)

Functional completion and the user's <100 ms goal are separate. Accuracy
**PASS** (9911/10000, 99.11%). Full-process **FAILED**: median 1657.407840 ms,
max 1684.153743 ms over five unprofiled processes; report and PNG say
`FAILED` and do not substitute internal 78.942 ms GPU work or the legacy
series. The design explicitly permits functional delivery while marking the
performance target unmet.

### Provenance/authentication — limited by design

CRC32/SHA-256 provide consistency, not authentication against wholesale
evidence forgery; the helper docs say so, and I did not treat them as
cryptographic proof. The actual training logs, independent label check and
package extraction are main-agent evidence, not independently rerun here.

## Task quality

- **Task 1 (host contracts/model handoff):** Strong. Exact parsing/size/range
  tests, metadata grammar/CRC tests, valid-pair replacement and unrelated-entry
  preservation, symlink/hard-link cases exercised on Linux. Residual: test-only
  header main, non-POSIX early returns, 256-byte boundary not pinned,
  identical-byte mixed-pair wording; all accepted debt.
- **Task 2 (CMake/build/link):** Strong. 25 commands/24 TUs/7 links, C++14,
  O2, arch 90, no fast-math, 0 warnings; infer link excludes training backend.
  Residual: release-flag override; augmented mapping coupled to documented
  helper; classes-1/overflow branches; internal arena logic_error; accepted.
- **Task 3 (forward-only model/training):** Strong. One arena, no training
  state/random init; bitwise reference comparison; deterministic supplied-only
  training; real one/two-epoch and fractional-byte regression tests. Residual:
  historical report wording, extreme-batch exception taxonomy, duplicate CRC
  token parse, source-only optimizer exclusion, optional warning flags;
  accepted.
- **Task 4 (runtime/CLI):** Strong. Batched global offsets, no per-batch
  allocation/readback, sticky flag, buffered output, sink-failure tests,
  profile first-batch coverage, official subprocess usage/model/exit tests.
  Residual: exact-1024 boundary, unbounded poll/CTest timeout, fork ordering,
  pageable-staging label, self-consistency profile assertions, historical
  drift; accepted.
- **Task 5 (helpers/benchmark):** Strong. Pinned schema, exclusive JSON,
  label isolation, identical-run enforcement, reference uint8 checks,
  separate profile artifact, truthful slow-run flag. Residual: line/
  leading-zero parser leniency, spawn sentinel, conservative mismatch sentinel,
  no post-series hash self-check (main did it), no timeout, unexercised
  defensive branches, mock argv scope, docs asymmetry; accepted.
- **Task 6 (package/training/acceptance):** Strong actual result and package
  verification. Residual: no failure-injection/rollback test, slow-PNG FAILED
  text assertion, zero-wall acceptance, report caveat (M3), splitlines
  leniency, heuristic/full-checkpoint limits, whitelist-literal coupling;
  accepted except M1/M2 artifact disclosure.
- **Cross-cutting:** Old host 9 / operator / workflow 12 / compliance 198 /
  Makefile/env pass; known old `test_prepare_mnist` pyarrow failure and prior
  sanitizer self-disable are recorded honestly rather than claimed green.
  No source-manifest or compliance breakage from the new course sources.

## Deferred minor / ruling dispositions

Legend: **MF** = must fix before completion; **AD** = acceptable debt.

### Task-1 deferred minors

- Header-defined test main → **AD** (each course test target is currently a
  single TU; revisit if multi-TU).
- Non-POSIX early-return cases counted executed → **AD** (acceptance is Linux;
  do not use Windows counts as alias evidence).
- Exact 256-byte metadata boundary not pinned → **AD** (grammar rejects longer
  payloads; generated metadata is far below the cap).
- Mixed-pair wording for identical checkpoint bytes → **AD** (identical bytes
  are a valid prior pair; docs say mixed-byte pairs are rejected, not all
  arbitrary corruption).

### Task-2 deferred minors

- Augmented expectation shares the documented mapping with the kernel →
  **AD** (known-offset fixtures are independent anchors; equal-index fixture
  corrected).
- Classes-1/overflow branch coverage → **AD** (low-risk defensive branches;
  bounds are checked).
- Duplicated O2 release flags override caller release flags → **AD** (matches
  the original O2 baseline; default acceptance build verified; see M5).
- Undocumented internal arena-plan logic_error → **AD** (internal invariant;
  bitwise and capacity tests pass).

### Task-3 deferred minors

- Auxiliary buffers mislabeled as one arena → **AD** (current docs distinguish
  one model arena from batch input/normalized buffers; historical evidence
  wording only).
- Extreme-batch allocation exceptions broader than header → **AD** (CLI batch
  fixed 128; direct API can fail with allocation errors as documented).
- Duplicate validated CRC token parse → **AD** (metadata is parsed/validated
  before this read; duplicate is unreachable).
- Optimizer validation-index exclusion is source-only → **AD** (validation
  oracle separately checks the held-out subset; real run passes).
- Optional CUDA tests lack extra host-warning flags → **AD** (full build audit
  shows 0 warnings).

### Task-4 deferred minors

- Exact default-1024 runtime boundary weakly pinned → **AD** (parser test pins
  1024; N=1025/default is two batches; N=17 is one).
- Subprocess poll unbounded / no CTest timeout → **AD** (see M11).
- Fork/direct-CUDA case ordering coupled → **AD** (sink case is last and
  documented; all subprocess cases precede parent CUDA init).
- Pageable H2D host staging may inflate host enqueue bucket → **AD** (label
  says host enqueue, not GPU execution; honest).
- Profile aggregate/per-batch assertions are self-consistency checks → **AD**
  (first-batch record count is pinned; actual profile recorded separately).
- Older 13-case report historical drift → **AD** (evidence artifact only; not
  runtime behavior).
- Sink F3 closes with F1, not a separate block → **AD** (both buffered sink
  failure cases are covered).

### Task-5 deferred minors

- Prediction parser line/leading-zero leniency → **AD** (development checker
  only; production emits canonical one-digit lines).
- Process-spawn `-1` sentinel overlaps signal 1 → **AD** (both are rejected as
  nonzero; no false pass).
- Conservative reference-mismatch sentinel on invalid runs → **AD** (never
  produces a pass; package rejects nonzero mismatches).
- Script does not self-check post-series hashes → **AD** (main independently
  checked after each real series and final binary/model proof; residual is a
  helper hardening item, see M6 for the analogous publication gap).
- No subprocess timeout → **AD** (actual series completed; add if helper is
  reused under unattended CI).
- Untested defensive branches/nested parent leftovers → **AD** (helper-only
  cleanup paths; successful runs and ordinary failures pass).
- Symlink fixture called read-only without chmod → **AD** (fixture exercises
  read path; no production write-through).
- Mock records argv rather than rejecting labels → **AD** (official CLI extra-
  labels test covers the real contract).
- Doc identity table only unit field asymmetry → **AD** (report includes all
  measured hashes; docs note modes).

### Task-6 deferred minors

- No failure injection/rollback case → **AD** (see M12; successful path and
  all pre-publication validation are covered).
- Slow-image PNG FAILED verdict not asserted by a test → **AD** (renderer is
  deterministic and actual PNG visually inspected by main; add assertion with
  M2 regeneration if touched).
- Zero-wall allowed → **AD** (an external monotonic wall cannot realistically
  be zero; helper is not authentication).
- Report assumption caveat not explicit official-unconfirmed → **AD** (see M3;
  labeled assumption, docs explicit; fold into M1/M2 wave).
- `splitlines` leniency in helpers → **AD** (production writes `\n`; package
  additionally requires canonical lines).
- Heuristic scope/full checkpoint decoding limits → **AD** (helper is not a
  security verifier; the C++ loader and actual runtime provide real model
  validation).
- Stale 27→66 test doc status → **AD** (corrected by main; current docs say
  66).
- Whitelist literal coupled to CMake graph → **AD** (deliberate exact whitelist;
  main traced/extracted build and actual graph pass).

### Rulings explicitly triaged

- Publication atomicity vs arbitrary I/O failure → **AD** (validation failures
  guarantee zero output; downstream sink failure cannot roll back accepted
  bytes; model mixed pair is rejected, no rollback claimed).
- Positive batch API/overflow → **AD** (batch>0, grid/size checks, batch
  bounds; no arbitrary training upper limit beyond int representability).
- Sticky atomic flag → **AD** (atomicExch, zero once per request, never cleared).
- Exact 1024 boundary → **AD** (parser/actual tests; see Task-4).
- Test-harness timeout/fork order → **AD** (see M11; bounded real runs).
- Benchmark post-series hash guard → **AD** (main checks after each real
  series; helper improvement recommended only if reused).
- Helper strictness/rollback/report preprocessing caveat → **AD** for parser
  leniency and rollback test gap; **M1/M2 MF**, **M3 optional AD** as above.
- Reuse detached worktree/preserve dirty baseline, no implementer commits,
  backup refs, deepseek-flash choice, tests-first, strict “1” profile,
  Linux/POSIX host tests, Make-policy preservation, pyarrow/sanitizer
  limitations, evidence retention, per-run scratch root, model publication
  wording, unrelated-entry preservation, error taxonomy, augmentation test
  correction, direct batch size, narrow production-used policies, profile
  schema, keep default 1024, no rerun of 36 startup diagnoses, old tests in
  fresh snapshot, decimal caps, image-cap API → all **AD**; no contrary
  evidence found and no code change required.

## Declined to judge (considered and set aside, with reason)

- **Cryptographic authenticity / malicious whole-evidence forgery:** CRC32 and
  SHA-256 are consistency checks; the helper docs disclaim authentication. Not
  a code defect and outside a local review's ability to prove.
- **Actual remote H20 execution and original run provenance:** I did not SSH
  or rerun; the supplied logs/JSON are internally coherent, but I cannot
  re-attest the machine-level run.
- **Visual PNG layout, clipping and legibility:** main visually inspected the
  PNG; I verified its hash (0c86…), dimensions (900x320) and the renderer
  code, not the rendered pixels.
- **Official course preprocessing and leaderboard timing scope:** unknown by
  design; unit is only the user-approved project assumption. Not judged.
- **Sanitizer status:** prior self-disable is not a valid pass; not claimed.
- **Old `test_prepare_mnist` pyarrow failure:** baseline environment
  limitation; data_interop passes; not attributed to this branch.
- **Windows/MSVC build and non-POSIX alias branches:** declared target is
  Linux/POSIX; the host/build libraries use POSIX headers and the tests
  intentionally skip/reject non-POSIX alias cases. Not judged as a course-
  target defect.
- **Paired speedup/regression claims across shared-machine series:** no such
  claim is made; legacy and course-model series are not presented as a paired
  benchmark.
- **Deployment/upload/push:** none occurred; not part of functional delivery.
- **Git backup refs vs real HEAD/index mechanics:** evidence says real HEAD/
  index untouched; checkpoint refs are backup refs. Not a code question.
- **Exact 256-byte metadata acceptance:** grammar rejects longer payloads;
  accepting a longer but otherwise valid payload is not required.
- **Package helper failure-injection and zero-wall edge:** accepted debt, not
  a production behavior.
- **Final rebuilt binary hash differs from measured binary:** explicitly not
  used to reassign timings; functional verification only. Accepted.

## Cannot verify (separate from declined judgment)

- The actual raw training run, labels and five-process measurements beyond the
  supplied evidence: 20 epochs/594.740494 s, best 15/0.992399991/CRC
  `f473014f`, five 9911-correct runs, median/max. I did not rerun them.
- The actual remote binary/model/meta/image/label hashes at measurement time:
  I verified the current root artifacts match the recorded report/evaluation/
  PNG/ZIP hashes and member list, but not the remote filesystem at run time.
- The original 67 source hashes and 8 protected remote hashes: supplied
  evidence says unchanged; I did not recompute them.
- The 66 helper tests, 48/8/9/15 course CTest groups, old host/operator/
  workflow/compliance/Makefile suites: evidence only; I did not rerun.
- Main's independent parsing/counting of raw predictions against original C1
  labels: evidence only.
- The visual content of the actual PNG: main inspection; I verified hash,
  dimensions, bytes, and renderer text construction, not pixel-level legibility.
- Fresh extracted build/roundtrip and images-only final-model inference:
  evidence (`task-6-extracted-roundtrip.json`, extracted build) only; I did
  not rebuild.
- Official course preprocessing and leaderboard timing: unknown by design.
- Sanitizer validity: no valid result claimed.

## ReadyToDeliver

**Functional code/package: yes** — the implementation, tests, actual default
training, final-model accuracy, package contents, whitelist build graph and
protected-source posture are complete and coherent. No Critical or Important
code finding was found.

**Performance target: no** — full-process median 1657.407840 ms / max
1684.153743 ms; the <100 ms goal is not met. This is the declared outcome, not
a substitution or a hidden test result.

**Overall: with fixes** — apply M1 (report timing-stage sentence) and M2 (PNG
five-run range) before treating the submission artifacts as final; optionally
fold in M3 (explicit official-unconfirmed caveat) and M4 (remove duplicate
`max_ms`). Regenerate report/PNG/ZIP, rerun the affected helper tests, recheck
hashes, and perform the scoped re-review. All other deferred minors are
acceptable debt, with M6/M11/M12 recommended only if the helpers/tests are
reused beyond this submission.
