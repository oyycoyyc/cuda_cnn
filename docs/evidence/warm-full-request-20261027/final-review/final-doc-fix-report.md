# Final docs-only fix report (whole-amendment review I-1)

**Status:** I-1 fixed with a docs-only diff in
`docs/course-interface.md` and `docs/course-interface-results.md`. The four
current root artifacts were hashed before and after and are unchanged. No
source, helper, test, CMake, GPU, model, measurement, root artifact, report,
PNG, evaluation, ZIP, git state, install, SSH or subagent action was taken.
No artifact was republished and no test/GPU series was rerun.

## Finding addressed

I-1: `docs/course-interface.md` called `docs/course-interface-results.md`
the location of "Actual final metrics", while that preserved cold document
still presented its historical cold image metric (1657.41 ms) and old ZIP
(119946/338343 bytes, 900x420 PNG, SHA `76da...`) as current final
deliverables. All of those cold numbers were true for the cold acceptance and
remain preserved; only their current-final framing was wrong.

## Sections changed

`docs/course-interface.md` (only the development-artifact paragraph, now
`:195-202`):

- Replaced the "Actual final metrics ... course-interface-results.md" sentence
  with an explicit split: current warm final metrics/root artifacts are in
  `course-warm-benchmark-results.md` (mean67.83ms / median57.75ms,
  5 warmups/101 measured full10000 requests, 99.11%), while the preserved
  historical cold record (1657.41ms cold-process median, FAILED under that
  separate scope) and known environmental limits are in
  `course-interface-results.md`.
- Both relative links are present and resolve to the intended documents.

`docs/course-interface-results.md`:

- Added a top banner (now `:3-7`): **"Superseded as primary timing by the
  2026-10-27 warm full-request amendment. Preserved as historical cold
  evidence."** with a relative link to
  `course-warm-benchmark-results.md` for current metrics/root artifacts.
- Reworded the cold image acceptance item (now `:247-251`) from "Final image
  `Test Time` must use..." to "In the preserved cold artifact, image
  `Test Time` uses1657.41ms..." and pointed to the current warm metric.
- Renamed "Submission artifacts and fresh extraction verification" to
  "Preserved cold submission artifacts and fresh extraction verification"
  (now `:258`) and rewrote the following paragraphs (now `:260-281`) so the
  old root snapshot, image, ZIP and SHA are explicitly **preserved cold
  artifacts from that phase, not current root deliverables**, and are
  preserved under
  `.superpowers/sdd/2026-10-27-warm-full-request-benchmark/preserved-cold-artifacts/`.
- Reframed the closing cold-phase disclosure/backup sentence as historical
  (`:296-297`).
- No historical cold number, table row, training fact, test result, range,
  stage, identity, or ZIP/PNG measurement was changed or relabelled warm.

## Targeted covering check

A stdlib-only Python assertion pass (no docs build, no dependency install)
checked, and printed `doc assertions PASS; four root artifact SHA256 values
unchanged`:

- banner wording and the relative warm-results link in the cold document;
- all five cold run rows (`1684.153743`, `1641.133869`, `1659.597542`,
  `1637.525734`, `1657.407840`, each `9911`), the cold median/max sentence,
  `1657.41ms`, `119946bytes compressed and338343bytes unpacked`,
  `900x420 cold`, `30620bytes`, and cold SHA `76da...`);
- absence of the old current-final phrasings `The final ZIP contains`,
  `Final image \`Test Time\` must use`, and `The image visibly shows`;
- both interface-document links and the explicit current-warm vs preserved-cold
  distinction;
- the warm result document still contains `67.82692348514851`, `57.752155`,
  `9911/10000`, `129468`, `354330`, `1200x620`, and SHA `c91d68...`;
- the four root artifacts' SHA256 values equal the pre-edit baseline:

| Root artifact | Bytes | SHA256 |
|---|---:|---|
| `report` | 5824 | `DF08E6FBA0D4C3E5D5F17BA25BD3CAA87634652319E808AEBFCA63B2086A072F` |
| `result.png` | 41824 | `679BCD4C2DDACB7D1EC0738EF952716D5AA6B6F3E5BEA84108ECE9FB0120956D` |
| `evaluation.json` | 70 | `6D0A18A1588AF8356B26867ED80266E978DF9684D09BA04A336EB4E0993A542A` |
| `course-submission.zip` | 129468 | `C91D680FAD2BA4BED3731AEF77FFA8D4E638EAD1A2F8FC001F8527E0D5E17A19` |

## Limits

- Docs-only fix; these files are not part of the source-only submission ZIP,
  so the 148 helper suite and CTest were deliberately not rerun.
- No Markdown renderer/build tooling was invoked; the check was textual stdlib
  Python over the two docs plus the linked warm document and artifact hashes.
- The review's explicitly nonblocking/deferred minors (CMake test dependency,
  optional PNG disclosure line) were **not** bundled into this fix wave.
- Main owns the fix diff, scoped re-review and final integrity backup; no
  actual warm publication or measurement claim is made here.
