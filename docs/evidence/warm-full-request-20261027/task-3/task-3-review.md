# Task 3 review — warm packaging, dispatch, and acceptance helpers

**Scope:** supplied scoped diff `0dc50a2 -> current uncommitted`, 2071 lines,
4 files: new `scripts/course_warm_package.py` (535 lines), new
`tests/test_course_warm_package.py` (1362 lines, 51 tests), minimal schema
dispatch in `scripts/course_package.py` (+25 net), and the warm amendment in
`docs/course-interface.md` (+37 net). Read the brief, controller context,
report, `controller-task-3-verification.md`, design spec, the full diff in two
reads, and inspected the live files. Read-only review: no suite rerun, no
source edit, no git command, no publication.

## Verdicts

- **Spec compliance: Approved**
- **Task quality: Approved**

**Critical:** none. **Important:** none. Six optional minors below; none blocks
the helper gate, actual publication, or the submission.

## Verification actually performed

- **Actual JSON compatibility checked structurally.** The reviewed Task 2 file
  has exactly the 26 top-level keys in `_TOP_LEVEL_FIELDS`, the nine summary
  keys, and the four per-record keys (`iteration`, `samples`, `wall_ms`,
  `predictions_identical`); `schema=course-warm-benchmark-v1`,
  `timing_scope=warm_full_request_h2d_prepare_forward_argmax_d2h_validation`,
  `model_source=course_train`, `input_mode=unit`, `n=10000`, `batch=1024`,
  `warmups=5`, `runs=101`, `status=pass`, `errors=[]`, `returncode=0`,
  `hashes_stable=true`, `predictions_identical_all=true`; 10000 stdout lines,
  107 stderr lines, 6+6 lowercase 64-hex identities.
- **Actual summary arithmetic reproduces independently.** From the 101 stored
  measured walls: mean `67.82692348514851`, median `57.752155`, min
  `7.550677`, max `638.715843` — exactly the stored summary and controller
  numbers. The `638.715843 ms` outlier is present and included; warmups and
  `process_wall_ms=9004.474382847548` are separate.
- **Test evidence credited, not rerun.** `task-3-fixed-green.txt` records Linux
  `Ran 148 tests ... OK` (51 warm + 97 prior), no skips; `task-3-red.txt`
  records the intended named missing-module RED before production code;
  `task-3-direct-cli-probe.json` / `task-3-direct-cli-fixed-probe.json` record
  the pre-fix traceback and the clean `course_package:` exit-1/no-files result.
- **Root artifacts are still the old cold scope**: `report` 1828,
  `result.png` 30620, `course-submission.zip` 119946 bytes, all dated
  2026-10-27 22:33. No warm root artifact exists, consistent with the
  controller's "main publishes after review" statement.

## Strengths (before findings)

1. **Exact reviewed schema, not a guessed one.** Validation requires the
   identical 26/9 key sets (`scripts/course_warm_package.py:103-114`,
   `:276-280`) and aliases the reviewed collector constants/parser/error/
   whitelist (`:29-37`). The tests pin those identities with `assertIs`/`==`
   (`tests/test_course_warm_package.py:381-440`).
2. **Strict provenance is actually independent.** The packager reparses raw
   stdout/stderr through `parse_tool_output` (`scripts/course_warm_package.py:245-249`),
   requires stored warmup and measured records to equal the parsed records
   exactly (`:251-258`), recounts predictions against the original int32 labels
   (`:205-214`, `:260-268`), and recomputes all nine summary fields from the
   measured records only (`:270-330`). Forged summaries, stale identities,
   mismatched modes/CRC, partial record sets, and cold five-run shapes are
   rejected before publication.
3. **Training/model/mode bindings are complete.** The cold training parser owns
   exactly-one completion, `epochs=20`, status, mode, and CRC grammar
   (`scripts/course_package.py:291-339`); the warm module binds training CRC to
   `course.meta`, recomputes CRC32 from `lenet.bin`, and binds benchmark/meta/
   training modes (`scripts/course_warm_package.py:181-191`). Six current
   identities must match both `sha256` and `sha256_after` (`:216-237`).
4. **Statistical contract matches the amendment.** Warmups never enter the
   statistics; no measured sample is filtered; all 101 measured and all warmup
   durations are rendered (`:270-274`, `:430-444`). `"achieved"` requires both
   mean and median `<100 ms` (`:332-334`); a provenance-valid, accurate slow
   result still publishes truthfully as `"failed"` and exits `0`. The outlier
   and mean-boundary tests pin this, including a `121 ms` max with mean
   `21`/median `20` accepted and median exactly `100` failed
   (`tests/test_course_warm_package.py:644-788`).
5. **Publication is exclusive and source-only.** The warm path reuses the cold
   `_check_output_paths`, `_build_archive`, decimal caps, `_publish`, and
   rollback unchanged (`scripts/course_warm_package.py:483-519`); 34 whitelist
   files + 3 generated members = 37, no warm tool/tests/Python/data/model in the
   ZIP. Existing archive/root members are refused before benchmark reading, and
   publication rollback is tested (`tests/test_course_warm_package.py:1117-1203`,
   `:1293-1304`).
6. **Report/PNG obey the warm amendment.** Exactly one accuracy group and one
   `Test Time: mean ... ms; median ... ms` group; image states 5 warmups/101
   rounds/10000 images per round, range, scope, and the mean+median gate; report
   lists every measured and warmup time, training completion, six identities,
   preprocessing caveat, leaderboard unknown, included/excluded stages, and the
   unchanged cold scope. Neither carries the old cold captions
   (`scripts/course_warm_package.py:360-480`; tests `:443-643`).
7. **Tests-first and the real CLI regression are genuine.** The named-module RED
   preceded implementation; the new `WarmPackageDirectCliTest` copies the three
   helpers into an owned root, clears `PYTHONPATH`, runs a real child, and
   asserts exit 1, empty stdout, no files, a `course_package:` diagnostic, and
   no traceback (`tests/test_course_warm_package.py:1204-1260`). The production
   fix wraps only the warm `PackageError`, leaving `FileExistsError`/`OSError`
   and every cold path unchanged (`scripts/course_package.py:897-916`).

## Findings

### Critical
None.

### Important
None.

### Minor

1. **Warm-tool and official-infer identities are hashed but not required to be
   distinct.** `scripts/course_warm_package.py:162-176, 219-237` requires both
   files and binds both hashes, but a document with identical
   `executable`/`official_infer` paths or hashes would pass if all other
   evidence matched. The actual reviewed JSON has distinct identities, so the
   real artifact is unaffected; an `os.path.realpath`/SHA inequality assertion
   would harden the separation requirement.
2. **The cold dispatcher now parses the benchmark JSON before cold whitelist
   and training validation.** `scripts/course_package.py:897-905` adds
   `_load_json_strict` unconditionally, changing cold error precedence for an
   unreadable/duplicate-key JSON and reading the file twice on the cold path.
   Preexisting-output refusal still comes first and the 97 cold tests pass; this
   is an ordering/robustness change, not a behavior regression.
3. **`REQUIRED_EPOCHS = 20` is defined but unused in the warm module**
   (`scripts/course_warm_package.py:37`); epochs are actually enforced by the
   cold `_parse_training_log` (`scripts/course_package.py:327`). Likewise
   `batch` is only range-validated 1..16384 (`:129-130`), not pinned to 1024.
   The actual series is 1024 and the report prints the real batch; remove/use
   the constant if self-documentation matters, and add a B=1024 gate only if
   the amendment intends it.
4. **The PNG does not say warmups are excluded from the statistics.** It says
   "5 warmups; 101 measured rounds" (`scripts/course_warm_package.py:381-382`),
   while the report states exclusion explicitly (`:430-432`). Optional
   disclosure improvement; the primary metric group and range/gate are correct.
5. **Broad import fallbacks are slightly opaque.** The warm module and the
   dispatcher catch any `ImportError` from the `scripts` import and retry a
   top-level sibling (`scripts/course_warm_package.py:21-26`,
   `scripts/course_package.py:889-894`). A genuine dependency failure could be
   masked or resolve a same-named module from another path. The direct-CLI test
   clears `PYTHONPATH`, so the reviewed path is safe; narrowing to the expected
   module-not-found case would be cleaner.
6. **`task-3-report.md:3-5` still opens with the earlier tests-only status**
   ("`scripts/course_package.py` is unmodified") before the later GREEN and
   CLI-fix sections. Documentation freshness only; the controller evidence is
   current.

## Cannot verify here

- **Actual warm root artifacts were not generated** and must not be claimed:
  `report`, `result.png`, and `course-submission.zip` at the root remain the
  old cold artifacts. Actual exclusive warm publication from the real Task 2
  JSON and real training log is main's post-review step.
- The actual tool/model/image/label paths stored in the Task 2 JSON do not exist
  in this review worktree. This review confirmed JSON structure and summary
  arithmetic only; the actual six-file hash/CRC/mode/training check, original
  label recount (9911/10000), and all-106-vector invariance remain credited to
  Task 2's independent audit and main's publication run.
- Fresh ZIP extract, default `COURSE_BUILD_TESTS=OFF` build, bounded
  train/infer, final N=10000 valid-model inference, protected hashes, global
  amendment review, and a fresh backup are main's remaining gates.
- Synthetic fixture durations are arithmetic inputs, not H20/GPU performance
  proof. The actual mean/median/outlier claims rest on the reviewed Task 2
  series, not on these tests.

## Bottom line

The new warm packager implements the amended contract: exact warm schema,
independent raw reparse/recount/recompute, strict training/meta/CRC/mode and
six-hash binding, warmup exclusion with outlier retention, a true
mean-and-median `<100 ms` gate, truthful FAILED packaging, and exclusive
source-only publication through the existing safeguards. The dispatch preserves
the cold API and refusal order, and the real direct-script class-identity defect
is fixed with a regression test. Approve Task 3; the six minors are optional
hardening/documentation cleanup, and actual warm artifact generation, extract/
build/inference, and protected-hash gates still belong to main after this review.
