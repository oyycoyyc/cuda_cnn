# Task 2 fix round 1 — scoped independent re-review

Scope: original review findings 1-2 only, plus any new breakage introduced by
`task-2-fix-1.diff` (148 lines). Fix base = `task-2-review-snapshot`; fixed tree =
uncommitted 6 files. Hash check: 3 of 6 files differ from the snapshot
(`CMakeLists.txt`, `course/gpu_ops.cu`, `course/tests/gpu_tests.cu`); `gpu_ops.h`,
`inference.h`, `inference.cu` are byte-identical, so new risk is confined to those
three. Working tree equals snapshot + the supplied diff (only the two replaced flag
lines are removed; everything else is additive). No git, SSH, subagent, code edit or
re-run performed; H20 evidence is read from controller logs, not reproduced.

## Finding 1 (Critical, CMake checkpoint→parameters edge): ADDRESSED

- `CMakeLists.txt:55` now declares
  `target_link_libraries(course_checkpoint PUBLIC course_parameters)`, exactly the
  required edge; `course_host:63` ordering was deliberately left alone, so the
  defect is fixed at its source rather than re-hidden by link order.
- No cycle: `course_parameters` (line 48) has no outgoing link dependencies, and it
  propagates no compile options into consumers (warnings are PRIVATE, line 49), so
  consumers' flags/lists are unaffected.
- Real evidence matches the fix: `task-2-fixed-model-bundle-link.txt` and
  `task-2-fix-1-audit.txt` show the link ending
  `libcourse_host.a libcourse_checkpoint.a libcourse_parameters.a`;
  `task-2-fix-1-build.txt` contains `Built target course_model_bundle` with zero
  warning/error lines in 153 lines of all-target output. The brief's tests-ON
  "builds Task1's course_model_bundle" requirement is now satisfied, and the same
  edge protects Task 4 executables that link `course_host` without touching
  parameter symbols.

## Finding 2 (Important, non-atomic sticky store): ADDRESSED

- `course/gpu_ops.cu:118-124` replaces the plain store with
  `atomicExch(bad_output, 1)`; the surrounding contract is unchanged: only ever
  writes 1, never clears, host zeroes once per request, no kernel-side reader, so
  the constant-value exchange cannot lose an observation and ordering vs the host
  read is still provided by stream completion.
- Code generation is confirmed by the controller's captured PTX
  (`atom.global.exch.b32 %r39, [%rd20], 1;`), which is a real atomic exchange rather
  than a plain `st.global`; the memory-model race is therefore removed, not merely
  masked. Unused return value produces no diagnostic (build log warning-free).
- New coverage `course/tests/gpu_tests.cu:630-692` is labelled honestly as
  behavioural only ("deliberately does not (and cannot) prove race freedom"), which
  is the correct claim for a machine where the pre-atomic stores also produced 1.
  I re-derived its fixtures: 1024 rows = four 256-thread blocks; poison indices
  3,11,...,1019 = exactly 128 rows (asserted), class position `sample % 10` cycles
  0-9 while all three nonfinite kinds are used via `(sample/8)%3`; the follow-up
  finite launch has a unique `10.0` at `sample % 10` against `-1.0`, so the expected
  `sample % 10` predictions are correct, and the flag is read after stream sync.
  CTest shows host48/48, GPU8/8, failures 0.

## New breakage from the 148-line fix: None

No Critical, no Important. The diff touches no public header, no forward math,
launch geometry, validation, arena layout, or CMake default/optional separation; the
four brief-named tests remain at `gpu_tests.cu:267,531,752,813` and the added eighth
case only increases coverage. Optional style nit only: the new test rebuilds its
two row fixtures inline instead of via the existing fixture helpers — cosmetic, no
action requested.

## Cannot verify (main-owned gates, not findings)

Full PTX dump (retained remotely as `logs/task-2-fix-1-ptx.txt`), post-fix protected
local/remote hashes, and the prior default-OFF/minimal-copy and override-86 audits.
Optimization evidence is accepted only at the level the controller claims: the
preserved NVCC `-O2` baseline flag and absence of fast-math, not a ptxas
optimization-level proof.

## Statuses

- Finding 1: **ADDRESSED** (`CMakeLists.txt:55`)
- Finding 2: **ADDRESSED** (`course/gpu_ops.cu:124`)
- New Critical/Important: **None**
- Scoped fix round 1 verdict: **APPROVED**
- **Final Spec Compliance: APPROVED** (brief's previously failing tests-ON
  model_bundle build satisfied; all other brief items were already met)
- **Final Task Quality: APPROVED** (race removed at source, atom confirmed in PTX,
  no regression; original review's deferred Minor items remain open by design:
  duplicated augment oracle, untested `classes==1`/overflow branches, release-flag
  hygiene, undocumented `logic_error` at `inference.cu:231`)
