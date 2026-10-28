# Final docs-only fix re-review (I-1)

Scope: `final-doc-fix-review.diff` (140 lines, exactly two docs),
`final-doc-fix-report.md`, `controller-final-doc-fix-verification.txt`, and the
current contents of both changed docs. Re-read I-1 at
`whole-amendment-review.md:119-143`. Read-only: no git/SSH/subagent/test
rerun/source edit. The 5858-line code review is not redone.

## I-1: ADDRESSED

- `docs/course-interface.md:195-202` now separates the **current warm final
  metrics** (67.83 ms mean / 57.75 ms median, 5 warmups/101 measured full-10000
  requests, 99.11%) and current root artifacts, linked to
  `course-warm-benchmark-results.md`, from the **preserved historical cold
  record** (1657.41 ms cold-process median, FAILED under that separate scope),
  linked to `course-interface-results.md` and explicitly "cold evidence, not the
  current timing claim."
- `docs/course-interface-results.md:3-6` adds the required top banner:
  "Superseded as primary timing by the 2026-10-27 warm full-request amendment.
  Preserved as historical cold evidence", with a current-warm link.
- `:247-251` rewords "Final image `Test Time` must use..." to "In the preserved
  cold artifact, image `Test Time` uses1657.41ms..." and points to the current
  warm metric.
- `:258-281` renames the section to "Preserved cold submission artifacts and
  fresh extraction verification", states the old root snapshot/ZIP/image/SHA are
  preserved under
  `.superpowers/sdd/2026-10-27-warm-full-request-benchmark/preserved-cold-artifacts/`
  and "not the current root deliverables." `:296-297` is historical.
- No stale current-final phrasing remains (`Actual final metrics`, `Final image
  \`Test Time\` must use`, `The final ZIP contains`, `The image visibly shows`
  all absent). Cold table rows (1684.153743/1641.133869/1659.597542/1637.525734/
  1657.407840, each 9911), median/max, 119946/338343 bytes, 900x420, 30620
  bytes and SHA `76da...` are preserved.
- Both relative links resolve to existing files. The four root artifacts still
  hash exactly:
  `report` `DF08E6...6A072F` 5824 B, `result.png` `679BCD...20956D` 41824 B,
  `evaluation.json` `6D0A18...A542A` 70 B, ZIP `C91D68...7A19` 129468 B.

## New breakage in this 2-doc diff

None Critical, Important or Minor. Only prose/links/headings changed; tables,
code blocks, cold numbers, training facts and identities are untouched; no
artifact or measurement changed. Relative links are correct from `docs/`; no
broken heading or table structure observed. I did not render Markdown (not
needed for this textual/docs-only diff).

## Readiness

**Spec compliance: Approved. Code quality: Approved.** I-1 is fully resolved
without republishing artifacts. Remaining work is main's current-hash check,
curation and final backup preserving real HEAD/index/user work; no further code
fix wave is warranted. Broader non-blocking minors were deliberately not
extended.
