# Whole warm timing amendment review

BASE `ad3efbf7c0c132aa46771ca7ccb6ffa82c4a767f` -> HEAD
`27876afe5e34d9b351a49992ca0d7041b9214f64` (Task 3 checkpoint), real HEAD
`bf6aae8` unchanged. Scope read: the complete 5858-line
`whole-amendment-review.diff` (read sequentially, no skipped section), the
binding controller context, spec, plan, the live changed sources/docs, the
task-1/2/3 review reports as cross-checks, and the actual root artifacts plus
`task-2-independent-audit.json`, `task-3-publication-acceptance.json`,
`task-3-integrity-verification.json`, CTest/helper/build evidence. Read-only:
no source edit, git command, SSH, subagent, suite rerun or artifact
regeneration. I did not read the old accumulated progress history.

## Verdicts

- **Spec compliance: Approved.** The amended warm protocol, statistics,
  submission artifact and truthful disclosures are implemented and evidenced.
- **Code quality: Approved.** No Critical or Important code/runtime defect.
- **Overall readiness: Needs one docs-only correction, then ready** for the
  remaining controller gates (curated evidence, identity reconciliation, fresh
  backup). No code or submission-artifact fix wave is needed; no rerun or
  republish is needed for the docs correction.

One Important finding exists, but it is documentation-only: the preserved cold
results document still presents the old cold image/ZIP as the current final
deliverable, and the updated interface doc still calls the cold document the
"actual final metrics". The actual `report`, `result.png`, `evaluation.json`
and ZIP are correct and warm. Everything else is Minor.

## Strengths (before findings)

1. **The timer is a correct complete-request host clock, not enqueue or GPU
   events.** `course/tools/warm_benchmark.cu:204` starts `steady_clock` before
   the per-round sticky-flag reset; `:205-241` enqueue flag reset, every H2D
   slice, GPU input preparation, original forward, argmax, both D2H copies and
   the one stream synchronization; `:247-277` performs host flag/range/
   invariance validation; `:278` stops the clock. Record formatting, stderr
   writes, prediction publication and the completion record are after `:278`
   (`:283-327`). No device allocation, event, disk access or output formatting
   occurs inside the measured window. This matches the spec's "before flag
   reset through synchronized readback/validation" boundary.
2. **The measured pipeline is the official GPU pipeline, recomputed every
   round.** `course/tools/warm_benchmark.cu:213-234` reproduces
   `course/runtime.cu:263-301` exactly: same `offset`, `remaining/current_batch`,
   `host_offset = offset * 784`, batch-at-pointer-zero H2D, identical
   `LaunchPrepareInput(..., mode, 0, 0, false, stream)`, `model.Forward`, and
   `LaunchArgmax(logits, device_predictions + offset, ...)`. No cached
   prediction, no CPU NN, no tiny-N substitute. `InferenceModel::Forward`
   (`course/inference.cu:137-172`) only enqueues the original FP32 kernels from
   a preallocated arena; no per-call allocation.
3. **Every round actually transfers and reads back, and all 106 vectors are
   checked.** H2D is repeated from host-resident `host_images` each round
   (`:221-225`); predictions and the bad-output flag are read back and stream
   synchronized each round (`:235-241`); every round is compared against the
   first warmup baseline, and any non-finite/out-of-range/mismatch aborts before
   any prediction or completion record (`:247-281`). Actual JSON has 5 warmup
   and 101 measured records, 107 stderr lines, and `predictions_identical=1`
   throughout.
4. **The actual statistics are proper and truthfully bounded.** From the 101
   stored measured walls I recomputed: mean `67.82692348514851`,
   median `57.752155`, min `7.550677`, max `638.715843`; 9911/10000 correct;
   the `638.715843 ms` outlier is retained, and **8** measured rounds are
   >=100 ms. Mean and median are both <100 ms, while the report/PNG/docs
   explicitly say this does not mean every round was <100 ms and show the
   `7.55-638.72 ms` range. Warmups (93.895522, 113.087978, 57.724755,
   132.007085, 108.231758 ms) are retained separately and excluded; all 101
   measured samples are retained. The whole tool process wall
   (`9004.474382847548 ms`) is separate. I verified all 101 `report` durations
   equal the raw measured values rounded to three decimals (0 mismatches).
5. **Raw schema, provenance and publication are independently strict.** The
   collector (`scripts/course_warm_benchmark.py`) enforces exact phase counts/
   indices/full N/positive finite times/invariance, one tool process, no labels
   argv, explicit model/batch env, and before/after six-file hashes. The packager
   (`scripts/course_warm_package.py:216-353`) reparses raw stdout/stderr with the
   reviewed collector parser, requires the stored warmup/measured records and
   all nine summary fields to equal the recomputed values, recounts predictions
   against int32 labels, enforces N=10000, R>=101, accuracy>=9600, exact
   `timing_scope`/`model_source`, training `epochs=20` and CRC/meta/mode bindings,
   and current six identities. Publication reuses the cold whitelist, decimal
   caps, exclusive `O_EXCL` writes and rollback; the actual root artifacts are
   exclusive and the ZIP is 34 runtime sources + 3 root artifacts only.
6. **Actual artifact fidelity is reproducible from the local files.** Root
   `evaluation.json` 70 B SHA `6d0a18...`, `report` 5824 B SHA `df08e6...`,
   `result.png` 41824 B SHA `679bcd...`, `course-submission.zip` 129468 B SHA
   `c91d68...` all match the publication acceptance JSON. The ZIP has exactly 37
   members / 354330 unpacked bytes, and its `evaluation.json`, `report`,
   `result.png` are byte-identical to the root files. The PNG is 1200x620; I
   visually inspected it: all text fits, one accuracy group, one
   `Test Time: mean 67.83 ms; median 57.75 ms` group, 5 warmups/101 measured/
   10000 per round, range `7.55 - 638.72 ms`, scope and `ACHIEVED` gate visible.
   No helper/tool/test/Python/model/data member, no symlink, no personal path in
   `report`/PNG.
7. **Identity separation and honest boundaries are explicit.** Report and JSON
   identify the measured warm tool `465b53d4...`, the measured official infer
   `c7c92922...`, and the rebuilt verification infer `25515e8d...`; the rebuilt
   binary is never assigned the warm timing. The warm tool's stdout (10000
   lines) is byte-identical to the fresh extracted official infer full-10000
   stdout, and that infer reproduced 9911 correct. The docs disclose unknown
   official preprocessing/leaderboard scope, the unchanged cold 1657.41 ms
   scope, no valid Sanitizer pass, no retraining/retiming for packaging, and the
   corrected controller-only empty-stderr assertion (source emits the documented
   `event=course_infer status=pass ...` line, not empty stderr).
8. **Default source-only build remains independent.** `CMakeLists.txt` keeps
   the warm tool and its test inside `COURSE_BUILD_TESTS` (`:164-181`), links the
   tool to `course_host` + `course_forward` (forward-only, not training), and
   leaves `train`/`infer` untouched. C++14, O2, arch 90, no fast-math. The
   extracted default build log is clean (0 warnings) and builds only the
   official runtime; the full test build produced both warm targets with 0
   warnings. Actual CTest 48/8/9/15/7 and 148 helper tests pass with no skips.

## Findings

### Critical

None.

### Important

**I-1. The preserved cold results document still presents the cold image and
old ZIP as the current "final" deliverable, and the interface doc still points
to it as the "actual final metrics".**
`docs/course-interface.md:195-196` says "Actual final metrics and known
environmental limits are in `docs/course-interface-results.md`", but the
current final metrics are the warm ones in
`docs/course-warm-benchmark-results.md` (also referenced at `:224`). The cold
document then says `docs/course-interface-results.md:242`: "Final image `Test
Time` must use 1657.41 ms ...", and `:252-260` describes the final ZIP as
119946 B / 338343 B unpacked / 900x420 PNG / SHA `76da...`. The actual current
root artifacts are warm: 129468 B / 354330 B / 1200x620 / SHA `c91d68...`, and
the cold versions (including `report` 1828 B, `result.png` 30620 B, ZIP
119946 B) are correctly preserved in
`.superpowers/sdd/2026-10-27-warm-full-request-benchmark/preserved-cold-artifacts/`.
The cold numbers were true for the cold acceptance and must not be rewritten or
relabelled warm. The risk is that a human reader of the current docs reports the
old 1657.41 ms/FAILED metric or checks the old ZIP, contradicting the amended
primary metric. The submission ZIP/root artifacts, report and PNG are themselves
correct, so this is a documentation consistency defect, not a product failure.
Fix: add a "Superseded by the 2026-10-27 warm full-request amendment; preserved
as historical cold evidence" banner to `docs/course-interface-results.md` and
reword `:242`/`:252-260` to say the *preserved cold artifact* uses those values;
change `docs/course-interface.md:195-196` to distinguish historical cold metrics
from the current warm final metrics. Docs-only; no artifact republish, no test
rerun.

### Minor

1. **CMake test target has no explicit dependency on the tool/infer it
   executes.** `CMakeLists.txt:172-175` builds `course_warm_benchmark_tests` but
   never `add_dependencies(... course_warm_benchmark infer)`. A full
   `cmake --build .`/`ctest` flow is fine (both are in ALL and the recorded
   full build passed), but a targeted build of only the test target would report
   "missing ... entry point" even though the sources exist. Fix: add the two
   dependencies; harmless.
2. **PNG omits the explicit warmup-exclusion/excluded-stage statement.**
   `scripts/course_warm_package.py:375-382` shows "5 warmups; 101 measured
   rounds; 10000 images per round", range and "per measured round", but not
   "warmups excluded from statistics" or the excluded setup/load/upload stages;
   the report states both (`_report_text` around `:425-457`). The primary metric
   group, range and gate are correct; optional one-line disclosure improvement.
3. **`REQUIRED_EPOCHS` is unused in the warm module and batch is not pinned to
   1024.** `scripts/course_warm_package.py:37` defines the constant, but the
   20-epoch gate is actually enforced by the shared cold `_parse_training_log`
   (`scripts/course_package.py:327`); `:129-130` accepts batch 1..16384. The
   actual JSON/report is batch 1024 and 20 epochs, so no actual gap; remove the
   dead constant and pin batch only if the amendment intends exactly 1024.
4. **No explicit distinctness assertion between the warm tool and official
   infer identities.** `scripts/course_warm_package.py:162-176,216-237` requires
   both files and binds both hashes, but does not reject identical paths/hashes.
   Actual identities are distinct (`465b...` vs `c7c9...`), and official infer
   cannot emit the warm-only raw protocol, so a false warm claim would also need
   fabricated raw evidence; hardening only.
5. **Cold dispatch reads the benchmark JSON earlier and twice.**
   `scripts/course_package.py:897-916` calls `_load_json_strict` before cold
   whitelist/training validation and then `_validate_benchmark` loads it again.
   Preexisting-output refusal still comes first, warm dispatch is correct, and
   the 97 cold tests pass; only cold error precedence and a TOCTOU/extra-read
   concern, not an acceptance weakening.
6. **Some parser/failure branches are lenient or untested.**
   `scripts/course_warm_benchmark.py:61-73` accepts non-canonical float spellings
   such as `+1`/exponent; `:454-470` handles only `TimeoutExpired`, ignores a
   kill error and does an unbounded post-kill drain; the `Popen` OSError and
   `sha256_after` `None` branches (`:449-452,486-488`) lack direct tests. The
   tool emits canonical records, spawns no children, and the actual run exited
   normally; no actual exposure, hardening only.
7. **The C++ test harness is intentionally loose and leaves some optional
   coverage gaps.** `course/tests/warm_benchmark_tests.cu:128-137` checks only
   path existence for the build-adjacent executable; `:620-663` tolerates
   repeated whitespace rather than the collector's strict single-space grammar.
   Byte mode, batch>N, N=1, default batch 1024 and later-round (post-warmup)
   failure are not separately covered; B1/B2, N17/B4 tail, N1025, poison and
   malformed inputs are. Actual acceptance is unit/B1024/N10000, so no
   load-bearing gap; optional test hardening.
8. **Warm orchestration duplicates the official batch loop.**
   `course/tools/warm_benchmark.cu:213-234` repeats
   `course/runtime.cu:263-301` rather than sharing a helper. I compared them
   today and they are identical in offsets/order/arguments; tests compare warm
   predictions to official infer, but no test pins timing-boundary equivalence.
   This is a deliberate narrow approach (spec rejects a production refactor) and
   a future drift risk, not a current defect.

## Deferred-minor triage (all controller items)

| # | Deferred item | Concrete load-bearing judgment | Action |
|---|---|---|---|
| 1 | Task1 orchestration duplication/drift risk | Current sequence is line-for-line equivalent to `runtime.cu`; only future drift. Tests compare predictions, not the timing boundary. | Non-blocking; optional shared helper/regression test. |
| 2 | Task1 tests parse whitespace flexibly | C++ test parser accepts repeated spaces; the production tool emits canonical single spaces and the Python collector is strict. Actual raw JSON was parsed strictly. | Non-blocking. |
| 3 | Task1 optional byte/B>N/N1/default-batch/later-round-failure coverage | Actual path unit/B1024/N10000 is covered; B1/B2, N17/B4 tail and poison are covered. Missing cases do not affect the actual series; byte path is shared `LaunchPrepareInput` covered by GPU tests. | Non-blocking; optional tests. |
| 4 | Task1 missing tool path existence check only | Tests-first diagnostic only; full build produced the tool; a directory/non-executable path would fail later by name. | Non-blocking. |
| 5 | Task2 non-timeout `communicate` exceptions cleanup | Actual run exited normally; only an exotic OS error could leave a child/no JSON. Timeout path is tested and writes failure JSON. | Non-blocking; optional `try/finally` cleanup. |
| 6 | Task2 swallowed kill error / unbounded drain | Tool spawns no children; SIGKILL is effective; only a pipe-holding grandchild could hang the drain. No actual exposure. | Non-blocking; optional bounded drain. |
| 7 | Task2 decimal lexical flexibility | Parser accepts some non-canonical float spellings; tool emits canonical, packager reparses raw text, hashes/summary still bind. No false acceptance path for the actual artifact. | Non-blocking. |
| 8 | Task2 untested spawn OSError/post-hash `None` branches | Code inspection is straightforward; actual run had both binaries and stable hashes. No actual exposure. | Non-blocking; optional tests. |
| 9 | Task2 historical report header status | Root `report` header is now warm; task-2 report is historical evidence, not a submitted artifact. | Non-blocking (see I-1 for the stale cold docs, which is broader). |
| 10 | Task3 binary vs infer distinctness guard absent | Actual hashes/paths are distinct; official infer cannot emit `course_warm_*` records, so fabricating a false warm pass would require fabricating the whole raw series. | Non-blocking; optional inequality assertion. |
| 11 | Task3 cold JSON validation precedence/read twice | Cold 97 tests pass; preexisting-output refusal still precedes dispatch; only error ordering/extra read. | Non-blocking. |
| 12 | Task3 unused `REQUIRED_EPOCHS`/shared parser 20 gate | 20 epochs is actually enforced by `_parse_training_log`; constant is dead but harmless. | Non-blocking; cosmetic. |
| 13 | Task3 batch flexibility (actual 1024) | Plan permits supported batch range; actual JSON/report is 1024 and disclosed. The timing gate itself is batch-normalized per full request. | Non-blocking. |
| 14 | Task3 PNG warmup exclusion implicit (report explicit) | PNG says 5 warmups/101 measured/per measured round; report says excluded. Primary metric group/range/gate correct. | Non-blocking; optional PNG line. |
| 15 | Task3 broad import fallback | Direct-script CLI test clears `PYTHONPATH` and covers the sibling import; actual normal module import works. Only masks an unexpected ImportError in unusual deployment. | Non-blocking. |
| 16 | Task3 historical report header | Same as #9; current root artifact is warm, task-3 report is evidence. | Non-blocking. |

No deferred item changes the actual measured result, the schema/hash/CRC
bindings, or the published artifact. None is load-bearing for this submission.

## Cross-task wiring and actual-data checks I performed

- **Tool -> collector -> packager protocol is exact**: C++ emits
  `event=course_warm_iteration phase=... iteration=... samples=... wall_ms=...
  predictions_identical=...` and one `event=course_warm_complete ...`; the
  collector parser requires exactly those fields/order/counts; the packager
  reuses that parser and compares stored records exactly. The actual JSON has
  the exact 26 top-level fields, 9 summary fields, 4 round fields, 5/101 records
  and 107 stderr lines.
- **Statistics**: warmups excluded; all 101 measured retained; mean/median/min/
  max recompute from the stored values; 8 rounds >=100 ms; max outlier retained;
  both requested gates pass. The report's 101 values match raw at 3 decimals.
- **Hashes/CRC/provenance**: actual JSON `sha256 == sha256_after` for all six
  identities; report carries all six; model/meta match the actual 20-epoch
  training evidence and `checkpoint_crc32=f473014f`; measured tool `465b...`,
  official infer `c7c9...`, rebuilt verification infer `2551...` distinct.
- **Accuracy/identity separation**: measured warm tool stdout is byte-identical
  to the fresh extracted official infer full-10000 stdout (10000 lines); 9911
  correct is recorded/credited. The rebuilt infer is not the timed binary.
- **Artifacts**: root hashes, PNG dimensions/visual content, ZIP member set/
  sizes/bytes and internal root-file equality independently checked; no
  personal paths or excluded development files in the ZIP.
- **Build/test logs**: extracted default configure/build with 0 warnings;
  bounded train 32/1 epoch -> infer 32 passes with the documented success
  stderr; final model infer 10000 matches warm stdout; CTest 48/8/9/15/7 and
  148 helpers pass with no skips; sanitizer absence is disclosed, not claimed.
- **Controller assertion correction**: the failed acceptance log required empty
  infer stderr; the actual runtime emits `event=course_infer status=pass ...`.
  The corrected probe returns 0 with 32 canonical predictions and that line; no
  product source/ZIP/model/timing was changed for this correction.

## Cannot verify separately

- Actual H20 GPU execution, the 5+101 series, 106-vector equality and the
  recorded mean/median: the H20 paths in the JSON do not exist in this
  worktree. I verified the derived JSON, the audit arithmetic, the actual root
  artifacts and the recorded identities, not the GPU run itself.
- Original C1 test labels / the 9911 recount: labels are remote; credited to
  Task 2 and main. I verified the stored prediction vector and its identity
  with the fresh extracted infer stdout.
- Measured tool/official infer/model/labels file SHA recomputation: those files
  are remote; I verified the recorded six-hash consistency and report/JSON
  equality, not the files themselves.
- Fresh extract/default build/bounded train/infer and protected 67+8 hashes:
  credited to the recorded logs/acceptance JSON; I did not rerun them.
- No valid Sanitizer pass; none is claimed. No retiming/retraining occurred for
  packaging per evidence; I cannot prove absence beyond that evidence.
- I did not regenerate `report`/PNG/ZIP or rerun any suite; those are main's
  verification claims, not mine.

## Final assessment

The amendment is technically coherent and the actual artifact is truthful:
complete-request host timing from flag reset through synchronized readback and
host validation; repeated H2D/prepare/forward/argmax/D2H; no CPU fallback or
labels in CUDA; original FP32 kernels and official interfaces unchanged;
proper 101-sample mean/median with the 638.715843 ms outlier retained; strict
six-identity/CRC/training bindings; exclusive source-only 37-member ZIP; and
fresh build/inference/identity separation disclosed honestly. Mean and median
both pass, but 8 rounds are >=100 ms and the maximum is 638.715843 ms; the
report/PNG/docs disclose this rather than claiming every round is under 100 ms.

**Spec compliance: Approved. Code quality: Approved.** **Readiness: one
docs-only correction (I-1), then final curated evidence/identity reconciliation
and fresh backup.** No source, test, model, measurement or submission-artifact
change is required; still no push/merge/deploy/upload.
