# SDD ledger — plan: docs/superpowers/plans/2026-10-27-evaluate-a.md

## Identity and authority

- Approved: user instructed implementation per design, main agent plans/delegates/reviews/verifies; subagents write code.
- Worktree: C:/Users/26030/.local/share/opencode/worktree/46d211/evaluate-a (native-created detached HEAD).
- Base: bf6aae802b80dfe98809bf9c7f15adefb87efe58.
- Implementer model: deepseek/deepseek-flash (verified available and configured for general/explore).
- Spec: docs/superpowers/specs/2026-10-27-evaluate-a-design.md.
- Remote experiment root: /gemini/code/cuda_cnn/runtime-output/evaluate-a-20261027 (create exclusively; original deployment untouched).
- Original checkout and its untracked user documents remain untouched.

## Preflight conflict scan

| Tasks | Shared interface or internal check | Finding / resolution |
|---|---|---|
| 1 | Profiling tests versus stderr/stdout grammar | New records are stderr-only and explicitly opt-in; normal stdout remains unchanged. |
| 2 | 58 result slots versus exact single-arena size | Update both arena budgeting and layout, plus exact-byte test; preserve all scans. |
| 3 | Full workflow suite versus no-training constraint | Fixture-only regression training is bounded test execution, not production training; original weights never change. |
| 1 ↔ 2 | RequireFinite timing and implementation | Measure instrumentation-only baseline before introducing consolidated results; compare stage timings and unprofiled process times. |
| 1 ↔ 3 | Workflow timing diagnostics and final measurement | Diagnostic runs and acceptance runs are separate; external timer measures process startup through exit. |
| 2 ↔ 3 | Arena size and old/new object comparison | Build same-toolchain baseline and candidate snapshots from separate object sets; preserve original binary. |

Ruling: No commits or staging — the plan and existing user instruction prohibit commits without explicit permission; use uncommitted task diff packages rather than commit ranges. Cost if wrong: less granular history, mitigated with saved task snapshots/diffs.

Ruling: Use the configured deepseek/deepseek-flash for coding subagents without model overrides — matches the user's explicit requested division of work. Main agent independently performs acceptance and reviews; no unsupported inference about reviewer model requirements.

Ruling: Controller runs all remote CUDA tests and benchmarks; implementers first deliver tests-only, wait for verified RED evidence, then implement — local machine has no CUDA device/compiler. Cost: one extra handoff per task, required to preserve real TDD evidence.

Ruling: User's latest approval supersedes pending-approval markers in the design and plan, and selects subagent-driven rather than native implementation.

## Tasks

- [x] Task 1: optional stage timing and baseline attribution.
- [x] Task 2: consolidated finite-scan readback, conditional on measured attribution.
- [x] Task 3: snapshot comparison, regression suites and alternating full-process benchmarks.
- [x] Final independent review and main-agent verification (partial performance result; integration intentionally not performed).

## Evidence

- Earlier three deployment baselines: 2860, 2770, 2780 ms; each accuracy 0.991100.
- Task 1 BASE: bf6aae802b80dfe98809bf9c7f15adefb87efe58; no production changes at dispatch.
- Local host baseline: all 9 suites passed. Remote same-toolchain build, finite-scan operator subset and evaluation fixture passed.
- Compared deployed and repository sources: train.cu, lenet.cu, adam.cu, Makefile and workflow_tests.cu match after CRLF/LF normalization; raw byte differences are line endings only.
- Baseline compliance attempt exposed CRLF Bash script transport (remote: `$'\r': command not found`; local Windows additionally GBK decoding errors), before any production change. Remote test log: baseline-compliance.log. This is not a candidate regression.

Ruling: Normalize CRLF to LF only in the isolated remote experiment when transporting text, matching the existing Linux deployment. Leave the original checkout and deployed source unchanged. Re-run remote baseline compliance after normalization; do not expand scope into Windows compliance portability fixes.

- Remote baseline compliance after LF normalization: 198 tests OK, skipped 3; comment inventory 147/147 and source policy gate pass.
- Remote baseline full operator suite and all 11 existing workflow cases passed, including bounded fixture training tests.
- Remote baseline `make python-tests PYTHON=python3`: blocked by pre-existing missing `pyarrow` in `/root/miniconda3` Python 3.12. No dependency installed and no pass claimed.
- Task 1 implementer: ses_e83d0433affeCsiIIL5I6VsYrp (deepseek/deepseek-flash). Tests-only handoff received; remote RED verified: evaluate_profile reports 0 records instead of 11, exit 1. Log task-1-red.log. Implementer resumed for GREEN.
- Local Anaconda Python has the existing pyarrow dependency. All 15 converter/interoperability tests pass using C:/personal_apps/anaconda3/python.exe; no dependency installation required. The remote missing-pyarrow limitation remains documented rather than hidden.
- Additional diagnostic baseline: single-image original-code inference still takes 1.63, 1.64, 1.68 seconds including startup. This suggests a substantial non-batch cost, but does not yet attribute it specifically to CUDA initialization. See baseline-single-infer.log.

Ruling: Preserve this plan's ledger and evidence at finish because changes remain uncommitted by user policy; automatic deletion would destroy the recovery record. Cost: a small ignored scratch directory remains until the user chooses integration/cleanup.

- Task 1 GREEN verified on H20: evaluate_profile/evaluate/cli pass; 198 compliance tests OK (skipped3), inventory151/151. See controller-task-1-verification.md and raw logs.
- Task 1 reviewer dispatched: ses_e83c3ccdfffeYo3MBvvFdlPMZF; read-only review of saved diff/report/controller evidence.
- Task 2 attribution gate satisfied: first profile forward_and_check1193.747ms (~44% workflow); device_select1384.408ms independently above goal. Continued tests show device_select1718.565/1723.233ms and forward_and_check2533.568/1845.446ms, demonstrating substantial load variability. Report all runs, use alternating final comparisons, and do not promise <1000ms from this change.
- Instrumentation-only binary preserved remotely; SHA2560e5d60439515dbae1af1eb555495c00018daaa9d2c79acd0e0bccc674f0be83e. Same-toolchain baseline SHA256d89ab76377be44816c5060055494c2afab0b01b076c689ac2e726ae9a7a937c8.

Task 1: complete (uncommitted diff vs bf6aae8, spec and quality review approved; main-agent H20 verification passed).

Deferred minor findings from Task 1: one unconditional host clock read on normal entry; repeated profile guards; test TextField substring parsing assumes existing output field order; comment mentions phase order without testing order. None affects current correctness or meaningful timing; final review may triage.

- Task2 implementer ses_e83c17a02ffel2Gl1O93vyHXE5: RED and GREEN delivered; H20 main-agent tests pass, compliance198 OK/skipped3, official accuracy0.991100. Verification details in controller-task-2-verification.md.
- Sanitizer unusable on both baseline and candidate: CUDA initialized before sanitizer, tool disables itself (exit99). This is an existing instrumentation/environment limitation, not evidence of an application memory fault or a successful memory check.

Ruling: Correct Task3 Makefile plan defect by building the tests-owned snapshot utility as an additional `all` prerequisite with a dedicated object/link rule, and adding that source to the compliance manifest. The existing analyzer requires exact equality of manifest sources and sources compiled by `all` (analyze_build_graph.py:1029-1055); manifest-only addition would break the gate. Production linking and test runner lists stay unchanged. Cost: `make all` builds one extra test utility; no effect on evaluate runtime.

Task 2: complete (uncommitted task diff vs bf6aae8, spec/quality review approved by ses_e83b87f8fffepEk7Lkdxa8vJ9J; main-agent H20 verification passed).

- Alternating full-process benchmark complete: baseline median2813.303ms max2856.650; candidate median2331.904ms max2425.245; median reduction17.112%, speedup1.2064x. All 10 runs accuracy0.991100. Target <1000ms NOT achieved.
- Candidate aggregate `make -o prepare-data test PYTHON=python3`: all9 host suites, full CUDA operator suite and all12 workflow cases pass; then stops at unchanged missing remote pyarrow. Local15 converter/interoperability tests pass with existing Anaconda environment. Dataset alias is read-only source data symlink, preparation/download target deliberately skipped.
- Task2 deferred minors: existing no explicit bound check in scan append, no poisoned-gradient/moment ordering test, no API-copy-count instrumentation assertion, harness lacks case counts; RED proves lenet_ filter executes the expected cases. Full comparison still pending Task3; benchmark now complete.
- Main-agent review of Task1/2 production code confirms unchanged numerical kernels, exact env gate, preallocated58-slot layout, same-stream scan/copy ordering and unchanged synchronization. Both new cases separately executed after verifying exact registered names in the binary; task-2-explicit-cases.log.
- Task3 implementer dispatched: ses_e83b492f9ffe8vaCJK3tm2Odcy; snapshot utility/behavior tests/Make integration only. Main owns evidence docs and remote acceptance.

- Task3 initial helper builds successfully on H20; no-argument invocation exits2 with usage. The candidate production binary hash remains7309bb46c8c1eab59be049c802ea5a4dc4b4862b53729f376e3d44c28170939a, identical to the timed binary.
- Main-agent pre-GREEN review found two Important issues in Task3: a symlink to a subdirectory beneath a mounted root bypasses lexical-parent stat walking; one rejection test proposes a fresh real mounted output path and is unsafe to execute if validation is buggy. No real mounted output test was executed or uploaded.
- Task3 fix round1/5 opened: original implementer resumed TESTS-ONLY to replace real-mount probes with owned simulated mounts and add a nested symlink regression. Controller will verify safe RED against the initial helper, then delegate the path-validation fix. Production optimization files remain untouched.

- Task3 safe nested RED verified on H20, then realpath fix and corrected ancestor assertions delivered by original implementer. All12 helper tests now GREEN, zero skipped, mounted inputs never used as output probes.
- Snapshot comparison at batches1,17,128: every complete810000-byte old/new snapshot identical, including predictions9911 correct. Final forced-rebuild candidate snapshots identical again; see controller-task-3-verification.md.
- Full successful-build policy evidence regenerated with actual forced rebuild after an incremental log correctly failed for lacking an app linker command. Scanner unchanged. Candidate binary hash changed after rebuild, so a second complete five-pair benchmark was run; both rounds retained.
- Latest v2 baseline/candidate medians2994.820/2182.433ms; v1 medians2813.303/2331.904ms. Neither round reaches1000ms. Original deployment/data/weights checksums unchanged.

Task 3: fix round1/5 (2 addressed, 0 open — unsafe real-mount probes replaced with owned simulated mounts; canonical nested-symlink ancestry fixed with RED/GREEN proof; uncommitted changes).

Task 3: complete (uncommitted task package, spec and quality approved by ses_e83975934ffeaigOoZsgHL4D10; main-agent actual H20 verification and current-binary retiming passed; performance goal still NOT achieved).

- Task3 deferred minors for whole review: actual hardcoded mount defaults are code-inspected rather than write-probed (safety requirement); Linux-only snapshot helper is built by all and therefore unsupported on Windows/CUDA hosts; validation/open TOCTOU under concurrent hostile filesystem mutation; dangling-link diagnostics; ignored unlink errors; missing-parent and10001-count test gaps; extra correct-count kernel result unused in snapshot tool; report line references are not authoritative. No blocker from task reviewer.
- Main local final refresh: all9 host suites and36 converter/interoperability/documentation/output-format tests pass. `git diff --check` passes; Git warns about existing autocrlf policy on README, not a code/test failure.
- Whole-worktree review dispatched: ses_e83915b47ffesiCLz5Ky8yeXmR, package final-review.diff2510lines; no code changes while reviewing.
- Main final verification fresh exit0: full CUDA operator suite, all12 workflow cases, all12 snapshot cases, comment inventory and source/build gates. 43-file normalized local/remote source manifest recorded; current binary and original inputs/deployment hashes unchanged. See controller-final-verification.md and final-verification.log.

Final whole-worktree review complete: ses_e83915b47ffesiCLz5Ky8yeXmR, zero Critical/Important findings, 11 Minor findings plus explicit triage of all inherited minors. Ready only as a measured partial optimization, not as a <1000ms success. Main agent concurs after independent H20 tests and direct production/safety review.

Main dispositions of final-review declined behaviors:
- H20 acceptance/primary review: controller performed actual runs; accepts correctness and measured improvement, rejects any claim that the <1000ms objective is met.
- Internal device_select breakdown: not measured, no internal-driver claim; deferred as the next proposed investigation, not implemented automatically.
- Sanitizer memory safety: unavailable because tool disables itself on both versions; residual validation limitation remains explicit.
- Actual mounted write rejection: deliberately not write-probed under user storage rules; code inspection and safe simulated mount tests only.
- Remote source identity: verified43-file normalized manifest plus fresh same-binary tests and hashes.
- Representative benchmark round: neither treated as universal; both complete5-pair rounds retained and reported, current binary identified separately.
- Chinese documentation: follows user's language and prior project explanation requirements.
- Batch128 / RouteA selection: user-approved design retained, no routeB or batch change performed.

Final minor dispositions: retain fixed58-slot invariant with tests; no future-architecture guard work added. Keep snapshot tests explicit rather than modifying default dataset-dependent test runners; command and limitation documented. Linux-only helper/all limitation documented rather than expanding to Windows CUDA support. TOCTOU/diagnostic/cleanup edge hardening and additional fixtures remain nonblocking follow-up. Plan checkboxes and report status updated; curated evidence exported outside ignored scratch so recovery does not depend solely on this directory.

Finishing skill gate: final fresh `make -o prepare-data test PYTHON=python3` exits2 due to existing missing remote pyarrow (one loader import error), after all host/CUDA/workflow cases pass. Corresponding local Python tests pass; no full remote-suite-green claim. Stop integration: no commit, merge, push, deploy or cleanup. Keep native detached worktree and all evidence for the user's next decision.
