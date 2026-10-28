# Warm full-request evidence index

These development records are excluded from the source-only submission ZIP.
They describe the user-confirmed warm timing amendment, not official leaderboard
timing and not the historical five-process cold-start metric.

## Primary result

Actual H20,20epoch course-trained model, batch1024,5warmups and101measured
full10000-image requests. Mean67.82692348514851ms,median57.752155ms,
min7.550677ms,max638.715843ms, accuracy9911/10000 (99.11%). Every measured
sample, including the maximum, enters the mean/median. Warmups are separate.
Each interval includes H2D, GPU preprocessing/forward/argmax, D2H, sync and
host checks; one-time disk loading/setup/allocation/weight upload/final stdout
formatting are excluded and disclosed.

## Records

- `task-1/`: reviewed optional CUDA benchmark, actual CUDA tests, timing/link
  audit and the measured binary identity. Default official train/infer unchanged.
- `task-2/task-2-course-default20-unit-w5-r101.json`: authoritative actual raw
  series, raw stdout/stderr, all5+101records, full precision, six before/after
  identities and statistics. Never regenerated to improve results.
- `task-2/task-2-independent-audit.json`: main's separate107-record parser,
  recomputed statistics and original-test-label correctness check.
- `task-2/`: collector review,97helper tests and diagnosed/fixed timeout-output
  preservation. Exactly one CUDA child; timeout cleanup does not re-run it.
- `task-3/task-3-review.md`: approved strict warm package/schema dispatch review.
- `task-3/task-3-final-helpers.txt`, `task-3/task-3-final-ctest.txt`: fresh148
  helper tests/no skips and five CUDA groups48/8/9/15/7, all pass.
- `task-3/task-3-publication-acceptance.json`: real37-member ZIP and PNG caps,
  all101report times, fresh extracted source-only default build, bounded official
  train→infer, and actual20epochmodel10000-image inference acceptance.
- `task-3/task-3-integrity-verification.json`: original67local/protected8remote
  files unchanged, current38normalized files matching, six measured identities
  stable, original raw series unchanged, old cold versions preserved.

## Artifact identity

Root `course-submission.zip`:37members,129468compressed bytes,
354330unpacked bytes; SHA256
`c91d680fad2ba4bed3731aef77ffa8d4e638ead1a2f8fc001f8527e0d5e17a19`.
Root PNG:1200x620,41824bytes. It shows99.11%, mean67.83ms/median57.75ms,
W5/R101/N10000, range7.55–638.72ms and the full-request boundary.

The timing belongs to warm-tool SHA256
`465b53d46f12a6f8cc6dde34e5da26e1a4a6e1ac9ec12bf3bdc6047ba2b6d782`,
not to the separately rebuilt verification infer. Official preprocessing and
leaderboard timing remain unconfirmed; there is no valid sanitizer-pass claim.
An unrelated old MNIST-preparation test lacks pyarrow. No install, platform
change, push, merge or deployment was performed for this amendment.

## Closing review and backup

`final-review/whole-amendment-review.md` approves code/spec/artifact integrity.
Its one Important docs-only finding is closed by
`final-review/final-doc-fix-rereview.md`; all optional hardening items were
explicitly triaged as non-blocking. The two-doc fix changed no submitted member,
model or measurement. `final-review/final-statistics-and-artifact-recheck.json`
independently confirms8measured rounds>=100ms, with every sample retained.
Final file/identity checks are in `final-review/final-integrity-verification.json`.
The five execution rulings and their costs are in `execution-rulings.md`.

Final backup reference: `backup/warm-benchmark-final-20261028` (resolve with
`git rev-parse` for its commit). Backup refs use a temporary index and do not
move the real detached HEAD/index. Existing workspace and all original audit
records remain in place; nothing was merged, pushed, deployed or submitted to
the course platform.
