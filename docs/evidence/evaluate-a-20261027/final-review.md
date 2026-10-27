# Whole-worktree final review — evaluate-a

Reviewer: subagent, read-only. Scope: all changes in the retained detached worktree
`C:/Users/26030/.local/share/opencode/worktree/46d211/evaluate-a` against
`bf6aae802b80dfe98809bf9c7f15adefb87efe58`, using
`.superpowers/sdd/2026-10-27-evaluate-a/final-review.diff` (read in full), `progress.md`,
the three `controller-task-*-verification.md` files, the task reviews, the raw benchmark /
snapshot / compliance / hash artifacts, and the on-disk sources. No edits, no commits, no
subagents, no remote access, no suite reruns (main agent owns H20 acceptance).

Working tree matches the package: `git status` shows exactly the seven modified files and six
new files listed below, `git diff --check` is clean (one pre-existing autocrlf warning on
`README.md`), and `HEAD` is still `bf6aae8` with everything uncommitted.

## Outcome status (must not be blurred)

**The startup-inclusive <1000 ms objective is NOT achieved.** Both complete alternating rounds
are retained and I recomputed their summaries from the raw JSON:

| Round | Baseline median / max (ms) | Candidate median / max (ms) | Median reduction | Every candidate < 1000 ms |
|---|---|---|---|---|
| v1 (`final-alternating-benchmark.json`) | 2813.303 / 2856.650 | 2331.904 / 2425.245 | 17.112 % (1.2064x) | No (`all_below_1000=False`) |
| v2 (`…-v2.json`) | 2994.820 / 3098.316 | 2182.433 / 2296.197 | 27.126 % (1.3722x) | No (`all_below_1000=False`) |

Fastest candidate run observed: 2134.411 ms, still >2x the target. `device_select` alone is
1.38-1.72 s. All 20 timed runs processed 10,000 samples with accuracy 0.991100 and exit 0.
No artifact claims otherwise: `README.md:246-251`, `docs/evaluate-a-results.md:3,100,120,157`
state the miss explicitly, and the docs correctly refuse to substitute raw forward time for
wall time. That is the honest position and it is what the design permits
(`docs/superpowers/specs/2026-10-27-evaluate-a-design.md:39-53`).

## What is delivered

- **Task 1** — opt-in `CUDA_LENET_PROFILE_EVALUATE=1` stage timing to stderr only
  (`src/train.cu:43-99,622-719`), plus a subprocess-based workflow case
  (`tests/workflow_tests.cu:215-376,920-975`), README and inventory updates.
- **Task 2** — model finite-scan results expanded from 1 to 58 arena ints and one contiguous
  readback (`src/lenet.cu:26,61-64,283-327,484,532`), exact-budget and semantic regression
  tests (`tests/operator_tests.cu:1590,1654,1999-2073`).
- **Task 3** — tests-owned snapshot comparer (`tests/evaluate_snapshot.cu`), 12 pytest-style
  behavioural tests (`tests/test_evaluate_snapshot.py`), Makefile build/manifest integration,
  results doc, spec/plan/ledger.

## Strengths

1. **Numerical equivalence evidence is genuinely strong.** Same-toolchain old/new snapshots are
   exactly 810,000 bytes and byte-identical at batch 1, 17 and 128 — 100,000 logits, 100,000
   probabilities, 10,000 predictions — and the same SHA-256
   (`33248ba9…51823`) is shared across all batch sizes and across the rebuilt candidate
   (`snapshot-comparison.json`, `snapshot-comparison-rebuilt.json`, `task-3-controller-run.log:20-34`).
   This is real per-sample evidence, not an accuracy-equality inference.
2. **Production change is minimal and surgical.** Only the arena tail size, the slot pointer,
   one copy and comments changed in `src/lenet.cu`; `RequiredBytes()`/`PlanArena()` remain
   mutually guarded by `offset != required_bytes_`, no kernel, normalization, prediction rule
   or public API changed, and there is still exactly one model arena. I confirmed the +228 B
   claim arithmetically at all three capacities (780284→780512, 1891708→1891936,
   71841956→71842184): 57 extra ints, nothing else.
3. **No checks were weakened.** Coverage is unchanged: 10 parameters, 11 activations,
   17 gradients, 20 moments = 58 slots (exactly the new capacity); ordering, smallest-index
   semantics, labels and the single `cudaStreamSynchronize` are preserved; no finite state is
   cached. Tests poison multiple tensors, a later sample, and switch Forward(17)→Forward(1)
   to prove stale tails are not scanned (`tests/operator_tests.cu:1999-2073`).
4. **stdout/exit grammar preservation is verified, not asserted.** The profile stream is
   opt-in, exact-match `1`, classic locale, fixed three decimals, stderr-only, and the new
   subprocess case asserts identical samples/accuracy/status and no `event=profile` on stdout
   or when disabled, while retaining the one-batch zero-timing contract and exit 3
   (`tests/workflow_tests.cu:920-975`). `std::cerr` is tied to `std::cout`, so `2>&1`
   ordering also remains summary-first.
5. **Snapshot helper safety design matches the brief.** Validate-before-work
   (`tests/evaluate_snapshot.cu:421-427`), exclusive `O_CREAT|O_EXCL` (`:242-247`), fsync,
   close and 810,000-byte verification with partial-file removal (`:250-277`), canonical
   `realpath` mount rejection with a `/` boundary (`:135-207`), no CPU fallback, no production
   link (`Makefile:98-99,181-186`), and all 12 rejection/alias/nested-symlink tests executed on
   H20 with zero skips and only build-owned fixtures (`task-3-controller-run.log:3-19`).
6. **Operational honesty.** Original deployment binary (`d06280df…`) appears with identical
   value in the *early* `environment.log:16` and the *late* `final-hashes.log:3`, so the
   "not overwritten" claim has genuine before/after evidence; data and weight hashes match the
   doc table. Remote pyarrow failure and the self-disabling Compute Sanitizer are recorded as
   limitations, never as passes (`docs/evaluate-a-results.md:46-50,82`). Evidence files
   (`.superpowers/sdd/.gitignore` = `*`) cannot be accidentally committed.
7. **Integration is coherent across tasks.** Profile phase names are identical in code, README,
   tests and the results doc; the snapshot tool links the same `lenet.o`/kernels objects, so
   the Task 2 change is inside the compared binaries; `make all`, the compliance manifest and
   the successful-build policy scan agree (198 tests OK / skipped 3, inventory 151/151,
   `compliance scan passed: mode=build evidence=successful-build`).

## Findings

### Critical

None.

### Important

None. In particular, the unmet performance target is an **outcome**, not a code defect, and it
is disclosed everywhere it matters. I found no correctness, safety, scope or evidence-honesty
problem that should block integration of the code as written.

### Minor

**M1. Scan-slot capacity is now exactly saturated with no bound check.**
`src/lenet.cu:26,286-290,484,315-317` — 10 + 11 + 17 + 20 = 58 = `kMaximumScannedTensors`,
zero headroom. The `append` lambda writes `tensors[count++]` unchecked, and the consolidation
makes the same constant load-bearing on the device tail and on a `count * sizeof(int)` bulk
copy into a 58-int host array. All current state widths (30/41/58) are in bounds, and the
arena is zero-initialized, so a slot-mapping bug would surface as a false diagnostic rather
than a silent pass. Pre-existing pattern, but a one-line `if (count == kMaximumScannedTensors)
throw std::logic_error(...)` (or a compile-time assert) is cheap insurance before anyone adds a
59th scanned tensor.

**M2. The snapshot safety tests are not reachable from any automated target.**
`Makefile:69,98-99` — `tests.test_evaluate_snapshot` is absent from `PYTHON_TEST_MODULES` and no
target runs it, while `make all` does build the helper. The 12 tests (including the nested
symlink regression that found a real bug) therefore only run when a human invokes them
explicitly; `make test/check/acceptance` would not notice a regression in the mount/overwrite
guards. This matches the plan's "no runner lists" wording, so it is a coverage/discoverability
gap rather than a violation. Recommend a separate `snapshot-tests` target (kept out of
`all`/`test`) plus one README line.

**M3. `make all` is now unbuildable on a Windows/CUDA host, but still advertises an `.exe`
alias.** `Makefile:98-99,192`, `tests/evaluate_snapshot.cu:1-8` (`#error`, `unistd.h`,
`realpath`). The documented target is Linux/H20 and the local Windows machine has no CUDA
compiler (`README.md:16-26`), so this is not a regression for supported workflows; the
extensionless alias is nevertheless misleading. Gate the rule on `$(OS)` or delete the alias
with a comment.

**M4. The plan tracker was never updated.** `docs/superpowers/plans/2026-10-27-evaluate-a.md:60-82,93-99,107`
are all still `- [ ]`, although the plan's own Task 3 file list asked for checkbox updates and
`progress.md:63,72,91` records the tasks complete. `progress.md` is the real ledger, so nothing
is misrepresented, but the shipped plan now understates completed work. Tick the verified items
(or say explicitly that progress.md supersedes the boxes).

**M5. Results-doc status and evidence pointer will age badly.**
`docs/evaluate-a-results.md:3` still says "final review in progress"; `:141` points raw evidence
at `.superpowers/sdd/...`, which is git-ignored (`*`), so a repo reader cannot open it. Add a
final status sentence and cite the remote experiment paths (already present at `:12-13`) as the
durable evidence location.

**M6. The new tests-owned utility has no usage documentation.**
`README.md:225-251,285-290` documents profiling and evaluation only; the `evaluate_snapshot
DATA WEIGHTS OUTPUT BATCH` interface and its "build-owned output, never under a data mount"
rule exist only in `tests/evaluate_snapshot.cu:34-36,515-533` and the test module. One short
README subsection (or a pointer in the results doc) would make the reproducibility step usable
without reading the source.

**M7. `TextField` still matches a bare substring.**
`tests/workflow_tests.cu:183-196,262-277` — `Field(record, "accuracy")` would resolve
`min_accuracy=` if the evaluate record were ever reordered. It is correct today because
`accuracy=` prints before `min_accuracy=` (`src/reporting.cpp`, unchanged), and it is a
test-only trap inherited from the old helper. Matching `" " + marker` or tokenizing would
remove it.

**M8. Profiling asserts coverage but not the documented order.**
`tests/workflow_tests.cu:354-376` checks one occurrence per phase and total dominance, not the
canonical order promised by `src/train.cu:83-99` and `README.md:236-244`. Low value, optional.

**M9. Phase accounting leaves two small gaps outside any phase.**
`src/train.cu:651-654,692-711` — `CanonicalOrder` (a 10,000-entry allocation) and the
accuracy/summary computation are only inside `workflow_total`. The observed residual is
0.394 ms on the profiled run (phases sum to 2720.916 vs total 2721.310), immaterial to a
2.1-3.0 s goal; worth a sentence or folding into an adjacent phase for future attribution.

## Triage of inherited deferred minors (none silently dropped)

| # | Inherited finding | Disposition |
|---|---|---|
| T1 | Task 1: unconditional `steady_clock` read + 88 B zero-init on the normal path (`src/train.cu:624-626`) | Accept. Tens of ns against a >1 s shortfall; `profile ? now() : {}` is optional tidiness. |
| T2 | Task 1: five repeated `if (profile) { start = now(); }` brackets (`src/train.cu:364-417`) | Accept. Explicit is debuggable; a helper would also need inventory entries. |
| T3 | Task 1: `TextField` substring/order assumption | Accept as M7; fix opportunistically. |
| T4 | Task 1: profile order not tested | Accept as M8; optional. |
| T5 | Task 2: no bound check in `append`, zero headroom | Accept as M1; recommend the one-line guard, not blocking. |
| T6 | Task 2: no poisoned gradient/moment ordering test (`tests/operator_tests.cu:2024-2073`) | Defer. Ordering logic is unchanged and slots 30-57 are exercised on the success path; add when a future change touches moment/gradient scans. |
| T7 | Task 2: no copy-count assertion proving one wide readback | Accept. No copy counter hook exists; code inspection plus the exact 232-byte budget and timing evidence suffice for this change. |
| T8 | Task 2: harness prints no case counts | Accept. RED (`task-2-red.log:1-2`) proves the `lenet_` filter executed the expected cases, and the controller separately ran both new cases by exact name. |
| T9 | Task 2: new semantic tests pass pre-change (only the byte budget is RED-first) | Accept, disclosed. They are preservation guards for a performance-only change; the observable contract (`1891708→1891936`) was driven RED→GREEN. No claim of a semantic RED is made. |
| T10 | Task 3: real mount roots are code-inspected, never write-probed | Accept by design. Probing a read-only mount to prove rejection is exactly the unsafe behaviour the brief forbids; a comment in the test module would record the deliberate choice. |
| T11 | Task 3: Windows/CUDA `make all` breakage | Accept as M3 (unsupported platform, misleading alias only). |
| T12 | Task 3: validate/open TOCTOU, dangling-link message, ignored `unlink`, bind-mount aliases | Accept as M12 below. Low-risk tests-owned tool; `O_EXCL` already closes the symlink race, `fsync`+size check prevent silent acceptance, and `unlink` failure can only leave an unreported residue in a caller-owned directory. |
| T13 | Task 3: no missing-parent / 10001-count / absent-input tests | Defer. The missing-parent guard is simple and read; only 9999 is covered; the malformed-batch test skips when inputs are missing although parsing never reads them. Cheap additions for the next pass. |
| T14 | Task 3: correct count computed and discarded (`tests/evaluate_snapshot.cu:478-483`); `snapshot-comparison-rebuilt.json` has no `correct`/`samples` | Accept with note. `snapshot-comparison.json`'s `correct: 9911` came from controller post-processing, and the rebuilt JSON only records the byte identity. This is traceability, not a false claim: predictions are host-validated against the logits argmax, and evaluation independently reports 9911/10000. Reporting the accumulated count would make each snapshot self-verifying. |
| T15 | Task 3: pageable host staging copies make batch 1 cost ~23.5 s | Accept. Snapshot runs are explicitly excluded from timed acceptance; correctness is unaffected. |
| T16 | Task 3: report metadata drift (`task-3-report.md:12-13,334`) and non-authoritative line refs | Accept. The report is a git-ignored working artifact, not a shipped deliverable; do not cite its line numbers as evidence. |
| T17 | Task 3: duplicated object list in the helper link rule (`Makefile:183-186`) | Accept. Matches the existing `lenet_cuda` style; a shared variable is optional cleanup. |
| T18 | Task 3: `docs/server-environment.md` is new and outside the plan's File Structure | Confirm intent. It contains no secrets, is consistent with `scripts/gemini_env.sh:4-8`, and the spec/plan reference it; keep it as project documentation if deliberate. |
| T19 | Language/style: `docs/evaluate-a-results.md` and `docs/server-environment.md` are Chinese while `README.md` and `docs/h20-acceptance.md` are English | Confirm preference. Content is accurate and tables render; only repo-consistency is affected. |

## Additional minor findings not in the inherited list

**M10. Snapshot-comparison provenance is thinner than the headline claim.**
`tests/evaluate_snapshot.cu:478-483`, `snapshot-comparison-rebuilt.json` — see T14. Also the
package header says "7 files changed, 573 insertions" because `git diff --stat` cannot see the
six new files (helper, tests, results doc, server doc, spec, plan ≈1,270 more lines). Anyone
skimming the header will underestimate the review surface.

**M11. Phase totals are internally consistent but not reconcilable to external wall time by
construction** (`src/train.cu:712-718`): `workflow_total` stops before destructors and process
exit, which is correctly documented; do not later quote it as end-to-end time. No action beyond
keeping the README wording (`README.md:241-244`).

## Recommendations

1. **Do not describe this change as meeting the goal.** Present it as "measured partial
   optimization: candidate faster in 10/10 pairs, median -17 % to -27 %, objective still
   unmet by >1 s, dominant remaining stage `device_select`". The docs already say this; keep it
   that way in the final answer.
2. **Before any future scanned tensor is added**, add the capacity guard/compile-time assert
   (M1) — it is the only latent memory-safety edge this change touches.
3. **Make the snapshot guards regression-visible**: a `snapshot-tests` target outside
   `all`/`test`, plus one README paragraph documenting the tool and the "never write under a
   data mount" rule (M2, M6).
4. **Refresh the paper trail** when the main agent closes the work: plan checkboxes (M4),
   results-doc status and durable evidence paths (M5), and mention that the 12 helper tests are
   a manual target.
5. **Next performance step, for discussion only**: split `SelectAndReportDevice()` into device
   count query, `cudaSetDevice`, and attribute reads to attribute the 1.4 s, then decide between
   a lighter startup path and Route B. Do not widen A silently, and do not move initialization
   outside the measurement window.

## Assessment / ready to integrate

- **Correctness: approved.** Numerical behaviour is proven identical per sample at three batch
  sizes on both binaries (and after a forced rebuild); all finite checks and their diagnostic
  order/labels are intact; the arena grows by exactly 228 bytes with one allocation; stdout,
  exit codes and checkpoint grammar are untouched; the original deployment, data and weights
  are provably unchanged.
- **Safety: approved for a tests-owned tool.** Exclusive creation, canonical mount rejection
  including nested symlinks, validate-before-work, and build-owned-only tests. Residual risks
  (TOCTOU, dangling link, ignored unlink, bind-mount aliases) are bounded and disclosed.
- **Scope: controlled.** Production diff is two files and 184 changed lines; everything else is
  tests, docs or build wiring. The only unplanned file is `docs/server-environment.md` (T18).
- **Evidence: honest.** Every claimed pass is backed by a saved log; every known limitation
  (pyarrow, sanitizer, Nsight, shared-host variance) is labelled as a limitation; both
  benchmark rounds are retained; the goal miss is stated in README and results doc.
- **Verdict: ready to integrate as a measured partial optimization; NOT ready to be reported as
  achieving <1000 ms.** No code-level blockers. The only decision blocker is product-level:
  accept the partial result (with the documented next step) or authorize a follow-up
  investigation of device startup. Main agent owns accept/reject, the fix wave for M1-M6 if
  desired, and the final answer.

## Declined to judge

1. **Actual H20 acceptance and primary review** — explicitly owned by the main agent; I did not
   rebuild or rerun established suites, and I do not re-certify their results.
2. **Internal attribution of the 1.38-1.72 s `device_select` bucket** — only
   `SelectAndReportDevice` is bracketed; driver-level breakdown needs counters that are not
   available (Nsight permission limitation recorded). The code and docs claim only the bucket,
   and I judge only that claim.
3. **Memory safety via Compute Sanitizer** — the tool disabled itself on both original and
   candidate (`CUDA initialized before the Sanitizer`, exit 99), so there is no sanitizer
   result to evaluate. I verified the 232-byte layout and `count <= 58` analytically instead;
   this is not a sanitizer pass.
4. **Runtime rejection of output under a real `/gemini/...` mount** — deliberately never
   probed, and correct that it was not; verified by code reading and simulated mounts only.
   Whether `/gemini/data-2`, `data-3` and `pretrain` exist on this host is also unverified
   (only `data-1` appears in the logs), which is acceptable because absent roots are skipped.
5. **Byte-for-byte identity of the remote candidate source with this worktree** — no source
   hash of the transported remote copy is recorded; line endings were normalized to LF for the
   Linux experiment. Binary hashes and normalized source comparison are recorded, but I will
   not assert more than that.
6. **Which benchmark round is "the" representative number** — shared-host variance is real
   (baseline medians 2813 vs 2995 ms). I reproduce both and judge only that neither meets the
   target and that retaining both is the honest choice.
7. **Whether the Chinese-language docs match the maintainer's preference** — flagged as T19,
   not judged.
8. **Design choice of batch size 128 / Route A versus B** — approved scope; re-litigating it is
   outside a code review, and the docs correctly stop short of Route B.

## Limits of this review

No CUDA compiler or GPU locally, so no compile/link/test execution was performed here; all
execution evidence is the controller's saved artifacts, which I cross-checked against each
other (JSON vs summary logs vs doc tables vs hash logs) rather than re-derived from hardware.
Line references were taken from the current working tree, not from the diff hunk headers.
