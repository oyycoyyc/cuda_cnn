# Task 5 independent review — spec compliance and quality

Scope: the five new uncommitted files vs reviewed base `8f1e3e5`, read through
`task-5-review.diff` (1913 lines, read whole). I re-hashed the working tree
against `.superpowers/sdd/2026-10-27-course-gpu-interface/task-5-review-snapshot`
and all five files are byte-identical (scripts/course_fixture.py 199 lines,
scripts/course_benchmark.py 337, tests/test_course_fixture.py 435,
tests/test_course_benchmark.py 728, docs/course-interface-results.md 192), so
the diff is an accurate view and the task added nothing else: those five are
the only files with post-Task5 mtimes, `CMakeLists.txt` has zero references to
the helpers, and the helpers import stdlib only. Read-only review: no subagents,
SSH, git commands, edits, or test reruns.

## Verdicts

- **Spec compliance: Approved.**
- **Task quality: Approved** (minor, non-blocking findings only; main still owns
  acceptance, protected-hash proof and the checkpoint).

## Strengths

- Fixture reads the real MNISTC1 v1 layout — 24-byte header, **all** image bytes
  then all label bytes (`course_fixture.py:130-157`), rejects bad magic/version/
  zero count/shape/exact-size/truncation/trailing bytes/labels >9, and hashes the
  source from the bytes it actually read, so manifest SHA256 cannot drift from
  the consumed file.
- Correct numeric domain: per-mode tables with explicit binary32 rounding
  (`_float32`, `F32_MEAN/F32_STD`, `course_fixture.py:38-63`), no uint8 roundtrip;
  `unit` is correctly rounded (no double-rounding hazard at p/255), and
  `normalized` reproduces `f32((f32(x)-0.1307f)/0.3081f)` so it matches GPU-side
  normalization bit-for-bit — consistent with all three modes giving identical
  predictions.
- Streaming is genuinely bounded (`IMAGE_CHUNK_BYTES`/`LABEL_CHUNK_BYTES`,
  never a dataset-wide float list), little-endian `<f`/`<i` only.
- Exclusive outputs are airtight: `lexists` pre-check plus non-`exist_ok`
  `makedirs` (`course_fixture.py:90,117`), `O_CREAT|O_EXCL` JSON write
  (`course_benchmark.py:141-156`), cleanup on failure, source opened `"rb"`,
  symlink aliases rejected. Tests prove existing dirs/files stay byte-identical.
- Inference argv is exactly the official six arguments (`course_benchmark.py:196-197`);
  labels are external-only and `COURSE_PROFILE`/`INPUT_MODE`/`EPOCHS` are cleared
  while only model-dir/batch are set, so persisted mode cannot be overridden.
- `--profile` matches the binding ruling exactly: all R requested runs are
  profiled in their own `profiled=true` artifact, no hidden extra process, every
  stdout/stderr/wall retained; fast acceptance is a separate call.
- Timing is external `time.monotonic()` around the whole subprocess including
  exit (`:208-238`), no warmup, all samples (including >100 ms outliers) kept,
  median/max recomputed from stored runs, and `all_wall_below_100ms` is strictly
  separate from the accuracy gate.
- Failure semantics are honest: invalid/extra/partial/non-integer/negative/
  out-of-range output, nonzero exit, disagreeing repeat runs, and reference
  mismatches all retain per-run evidence and force `summary.correct/accuracy` to
  0 / `accuracy_pass:false`; failure JSON is still written.
- Results doc is truthful: 15 real fast times, 100 ms gate declared failed, the
  78.942 ms profiled forward explicitly *not* substituted for full wall
  (`docs:166`), legacy-only accuracy, no course-training claim (`docs:191`), no
  host paths, and no mounts/weights overwritten.

## Findings

Critical: none. Important: none.

Minor (all non-blocking):

1. `scripts/course_benchmark.py:119` — `splitlines()` is wider than the pinned
   LF-per-line contract (CR-only, VT/FF, U+2028 split as lines) and `int()`
   accepts zero-padded forms like `007`, so a nonconforming binary could still
   validate. Harmless for the reviewed binary; `split("\n")` + canonical digits
   would be exact.
2. `scripts/course_benchmark.py:220` — the spawn-failure sentinel `returncode=-1`
   is indistinguishable from a POSIX signal-1 death in the JSON; no false pass,
   but reporting is ambiguous.
3. `scripts/course_benchmark.py:295` — when repeat runs disagree,
   `reference_mismatches` is set to `count` as a sentinel, not a measured diff
   count; worth documenting so the field is not read as a measurement.
4. `scripts/course_benchmark.py:16` — `ACCURACY_THRESHOLD = 0.96` is hardcoded
   with no stated provenance (implies Task 6's 96% acceptance); add a comment or
   explicit flag before it becomes an acceptance gate.
5. `scripts/course_benchmark.py:182` — hashes are taken before the series and
   never re-verified afterwards, so a mid-series binary/model swap would go
   unnoticed.
6. `scripts/course_benchmark.py:211` — no timeout around `subprocess.run`; a hung
   infer blocks the checker forever (low risk for interactive use).
7. Untested defensive branches: images-size mismatch (`:185`), malformed
   `course.meta` (`:72-93`), API-level `runs`/`batch`/`model_source` rejection
   (`:166-172`), and the `OSError` spawn path (`:219-223`).
8. `scripts/course_fixture.py:77` — for a nested `OUTPUT_DIR` a rejected
   conversion removes only the leaf, leaving parents created by `makedirs`
   (cosmetic; documented single-level usage is clean).
9. `tests/test_course_fixture.py:359` — the symlink-source test never makes the
   source read-only, so `task-5-report.md:73`'s "read-only source" claim is
   broader than the test (the `"rb"` open does satisfy it).
10. `tests/test_course_benchmark.py:303` — the brief asked for mocks that *reject*
    a labels path; mocks record argv and the test asserts exact/absent argv
    instead. Stronger for this contract, but a wording-level deviation.
11. `docs/course-interface-results.md:173` — the identity table lists only unit
    meta/images hashes; byte/normalized meta+image hashes exist only in private
    evidence. Correct under the sanitization rule, just asymmetric.
12. `docs/course-interface-results.md:88` — exit status deliberately excludes the
    `all_wall_below_100ms` gate (documented at `:166`); an automated wrapper must
    read the JSON flag, not just the exit code.

## Evidence cross-checks performed

Recomputed all 15 fast walls, medians and maxima from
`task-5-measurement-summary.json` and they match the doc table exactly; profile
wall 1513.449 ms and 78.942 ms forward match; `task-5-independent-prediction-audit.json`
has 18 processes x 10000, every run 9911/0, and one identical
`prediction_sha256` equal to the reference file hash `ed3089f6…` (so determinism
holds for byte/normalized R1 too); fixture audit byte counts (31360000/40000,
188160000/240000) and identical source SHA across the three test fixtures match;
`task-5-green-linux.txt` is `Ran 27 tests ... OK` with no skips.

## Cannot verify separately

H20/CUDA measurements, remote-experiment containment, the actual compiled
`infer` behaviour, and protected-file/hash invariance at HEAD are outside this
read-only review; I relied on the controller evidence named in
`controller-task-5-verification.md` for those. Task 6 (full 20-epoch training,
new-model 96% accuracy, packaging) is untouched, and the legacy 99.11% figure is
**not** treated here as course-model acceptance.
