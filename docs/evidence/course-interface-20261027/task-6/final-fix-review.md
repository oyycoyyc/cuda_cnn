# Scoped final-fix review — M1/M2/M3

Scope: only the one consolidated disclosure-fix wave after the whole-branch
review (`final-fix-report.md`, `final-fix-review.diff`, RED/GREEN endings,
artifact inspection and extracted roundtrip), plus the current root artifacts.
M4–M12 were not re-opened; M4 remains intentionally retained and pinned, M5–M12
are accepted debt per the main ruling. No SSH, reruns, source edits or further
subagents were used.

## Verdict

- **M1 — ADDRESSED.**
- **M2 — ADDRESSED.**
- **M3 — ADDRESSED.**
- **New Critical/Important findings: none.**
- **New functional/package regression: none.**
- One non-blocking evidence-only note: the handoff header
  `final-fix-report.md:3` still says `NEEDS_CONTEXT` from the tests-first turn,
  while its own GREEN section and the main evidence record completion. This is
  a stale evidence status, not a source or artifact defect; **acceptable debt**.
- **ReadyToDeliver FUNCTIONALLY: yes.** The 100 ms performance target remains
  **failed** and is not re-labelled by this disclosure fix.

## M1 — report timing stages (ADDRESSED)

`scripts/course_package.py:774-778` now inserts the report line:

`Timing scope: full-process wall from startup through exit, including model/image
loading, CUDA setup, transfers, computation, validation, and stdout/stderr
output; GPU event substages are separate diagnostics, not a substitute for the
full-process wall.`

The current root `report:20` contains that line verbatim. It explicitly covers
startup through exit, names loading/setup/transfers/computation/validation/
output, and explicitly separates GPU event substages. The original completion,
configuration, all five runs, median, max, accuracy, failed verdict, leaderboard
caveat and provenance remain present; no number was replaced. The new
`tests/test_course_package.py:469-483` test first failed on the old helper
(`final-fix-red.txt`, first missing item `startup`) and passes in GREEN.

## M2 — image range/scope with one primary median (ADDRESSED)

`scripts/course_package.py:687-692` leaves `metric_lines` byte-for-byte
unchanged, so the image has exactly one `Test Time:` primary value:
`1657.41 ms (median of 5 full-process runs)`. `scripts/course_package.py:705-724`
adds exactly one subordinate range line (`5-run full-process range: 1637.53 -
1684.15 ms`) and one scope line (`Scope: startup through exit (full process)`);
`run_package` supplies the observed `min(validated["walls"])` /
`max(validated["walls"])` at `scripts/course_package.py:894-897`, never a
hardcoded real metric. The current root `result.png` is 900x420 / 30620 bytes,
SHA-256 `c5acdf9f294f425efaa840c7010af76b95bf9dac3afc0a4ddf7a66a57bc8b1da`;
I independently viewed it: 99.11%, 1657.41 ms median, 1637.53–1684.15 ms
range, startup-through-exit scope, red `FAILED`, one accuracy line, one
`Test Time:` line, no clipping and no competing timing group. The range is
subordinate to the single median, so the conservative M2 ruling is satisfied.
The fast/slow tests (`tests/test_course_package.py:558-605`) first failed on the
old helper and pass in GREEN, including text-bbox no-clipping checks.

## M3 — official preprocessing unconfirmed (ADDRESSED)

`scripts/course_package.py:752-753` now emits
`Preprocessing caveat: official course preprocessing remains unconfirmed.` in
`_report_text`; the current root `report:5` contains it verbatim. The original
`Preprocessing assumption:` line remains, so the submitted report states both
the condition and that official preprocessing is unconfirmed. The
`tests/test_course_package.py:485-493` test failed on the old report and passes
in GREEN.

## Evidence integrity and scoped regression check

- **RED/GREEN:** `final-fix-red.txt` ends `Ran 43 tests ... FAILED
  (failures=4)` with exactly 39 `... ok` and 4 new-test failures; the failures
  are only the missing M1 stage text, M3 caveat and the two M2 range texts.
  `final-fix-green.txt` ends `Ran 70 tests ... OK`, with 70 `... ok` and no
  skipped lines. This matches the required frozen-old-helper sequence and the
  all-70-helper GREEN (Pillow 12.1.0).
- **Source scope:** the diff changes only `scripts/course_package.py` and
  `tests/test_course_package.py` in source, plus the generated `report` and
  documentation delta. No runtime, CUDA, default, source-graph, whitelist,
  measurement or numerical source changed. `render_result_png` has one internal
  call updated; no orphan three-argument call remains.
- **Artifacts:** `final-fix-artifact-inspection.json` and the current root
  agree: 37 members, 119946 compressed / 338343 unpacked, PNG 900x420 / 30620,
  SHA-256 ZIP `76da207fb76641745e10a8db6fb5e6063e51ade6b7e5db239d033f742eb977f8`,
  report `95cf4d4db06c5ec41fb4e70d3d0e3d964a45c5b06b68825734b0d8ae8dfd5c43`,
  PNG `c5acdf9f294f425efaa840c7010af76b95bf9dac3afc0a4ddf7a66a57bc8b1da`,
  evaluation unchanged at
  `6d0a18a1588af8356b26867ed80266e978df9684d09ba04a336eb4e0993a542a`.
  Independent archive-member comparison against the preserved old ZIP
  (SHA-256 `41e6ef...76a`) shows only `report` and `result.png` differ; all 34
  runtime source members and `evaluation.json` are byte-identical. Old report/
  PNG/ZIP hashes `60f6...`, `0c86...`, `41e6...` are preserved in
  `pre-final-disclosure-fix/`. The new archive is within all caps.
- **Extraction:** `final-fix-extracted-roundtrip.json` records fresh default
  build `pass`, 0 warnings, bounded official train/infer `pass`, and the actual
  final model's full-10000 images-only prediction `match`, with
  `no_retiming_or_retraining_of_finalmodel: true`.
- **Documentation:** `docs/course-interface-results.md:3` now states 70 helper
  tests and its artifact paragraph records the new sizes/SHA and the one
  disclosure-fix narrative. No old measurement was substituted.
- **M4/M7:** M4's duplicate `max_ms` is deliberately retained and remains
  pinned by the existing test; no change made. M7 is no longer a finding: the
  earlier Task-4 fresh minimal build/roundtrip existed, and the regeneration
  occurred after the actual previous-ZIP extraction build, so the final-artifact
  order is resolved per the main qualification.

## Scoped spec compliance

- M1: **PASS** — report explicitly states startup-through-exit stages and
  separate GPU event substages.
- M2: **PASS** — one primary `Test Time:` median, observed five-run min/max
  range, scope, one accuracy group, red `FAILED`, no clipping/conflicting
  timing group.
- M3: **PASS** — official course preprocessing explicitly unconfirmed.
- M4–M12: **not re-adjudicated**; M4 retained by ruling, M5–M12 accepted debt.
- Runtime/package functional behavior: unchanged and unaffected by this wave.

## Scoped task quality

The fix is minimal, tests-first and evidence-backed: the four new assertions
fail on the frozen old helper and pass after the smallest helper change; the
39 pre-existing package tests remain green; all 70 helper tests pass with no
skips. Actual artifacts were preserved before exclusive regeneration, source
members were not altered, the new archive stays within caps, and the fresh
extraction/build/roundtrip/actual-model evidence is intact. The only evidence
blemish is the stale `NEEDS_CONTEXT` handoff header noted above; it does not
affect the source, tests, report, PNG, archive or measured values.

## ReadyToDeliver (functional vs performance)

- **Functional code/package: yes.** The M1/M2/M3 disclosure wave is complete
  and the runtime/package source graph is unchanged; the rebuilt verification
  still passes.
- **Performance: no / still failed.** The actual final-model evidence remains
  99.11% accuracy and full-process median 1657.407840 ms / max 1684.153743 ms;
  the <100 ms target is still explicitly `FAILED`. No retiming or retraining
  occurred in this wave.
- Residual M4–M12 adjudication, final hash/current proof and final backup are
  main-owned as stated.

## Cannot verify

- The actual H20 execution provenance and original measurement logs beyond the
  supplied RED/GREEN, artifact and roundtrip evidence; I did not rerun them.
- The original 67 source hashes and 8 protected remote hashes; not recomputed.
- The future final backup/current-proof closure; not yet performed here.
- Official course preprocessing and leaderboard timing scope; unknown by
  design and correctly left unconfirmed.
- Pixel-level rendering beyond the actual current PNG view, hash, dimensions
  and the no-clipping bbox tests; the visible submitted content was checked.
- Remote filesystem state at regeneration time; current root artifacts and the
  preserved old versions were independently hash-checked.
