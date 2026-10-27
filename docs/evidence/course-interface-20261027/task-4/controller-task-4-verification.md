# Main-agent Task4 verification

## Tests-first and initial real build

- First tests-only H20 compile succeeds with ignored-write-result warnings; link fails only on missing RunCourseInfer. Warnings and4 fixture defects were corrected before production acceptance. Initial logs task-4-red-compile.txt/link.txt.
- Fresh tests-ON config builds only runtime test; before official binaries are built, filteredtrainusage fails naming missing official train (no CUDA call). task-4-missing-entry-red.txt.
- Initial all-target build warning-free and all4 suites48host/8GPU/9training/13runtime pass. This masked a genuine offset defect; not claimed as correctness acceptance.

## Batch offset RED/GREEN

- Controller source inspection found sample offset used asfloatpointer offset. Implementer first adds14thcase using finite synthetic model, normalizedzero/one images, exactexpectedclasses1/0 for batch1/2.
- Main confirms runtime/main frozen (task-4-before-offset-fix/manifest.json), compiles newtest witholdruntime and reproduces focusedfailure at secondexpectedclass0: executed1/failures1, exit1. task-4-offset-red.txt.
- Minimal productionfix reuses batch-capacity input/normalized buffers atpointer0, hostsource offset*784, predictiondestinationoffset stillsamples. All14cases now pass on actualH20 alongside48host/8GPU/9training, no warnings: task-4-offset-green-build.txt/ctest.txt.

## Default/source/link/entrypoint acceptance

- Fresh minimal source copy omits all course/tests/tools yet default builds train/infer; bounded official train oneepoch -> savedmodel -> infer4predictions succeeds, labels path withheld. No full20epoch/fulltestaccuracy/timingclaim.
- task-4-default-configure.txt/build.txt/roundtrip.txt and named training/inferstdout/stderr files hold evidence.
- Actual infer link and nm show no course_training/reference_training/originalLeNet/AdamW/loss/metrics/oldworkflows. Root no-macro and both-macro NVCCcompiles fail with intendedpreprocessor errors. task-4-link-macro-audit.txt / infer-link.txt / no-macro-red.txt / both-macros-red.txt. Sharedparameter/forwardtranslationunits can contain unusedhelpers; symbolpresence of sharedReadLabels/initializationhelper alone is not runtime invocation.

## Concrete open review attention

- Outputfailure probe: run actual minimalinfer N4 withstdout=/dev/full. Observed return0 and stderr `event=course_infer status=pass ...`, despitefailed output delivery. task-4-dev-full-probe.json. Runtime checks stream before an explicitflush, so buffered sinkerror can occur after the check; docs/runtime.h promise outputstreamfailure runtime1. Independent reviewer should classify against actualcontract. This is a safe OSsink probe, not a data write or timing result.
- docs/course-interface.md:94 calls unitdefault "the course convention" although approvedspec says user-approvedprojectassumption and officialpreprocessing remainsunconfirmed. Wording should not imply externalcourseconfirmation.
- Actualclass/orderingregression now discriminating; source-basedfastpath/profileinspection still required, not inferred fromstderrabsence alone.

No Task4 completion/checkpoint or full-process100ms/full-course-model success claimed. Current/protected hash proof will follow finalfixes/review.

## Formal fix round1 real H20 results

- Frozen pre-flush (but offset-fixed) runtime archive retained at experiment artifacts/task-4-before-io-fix/libcourse_runtime.a, SHA256 a3ed0b802d6f439497e4ea4cff95bf02bbcbec9ee6cbfda9b0118c44e47f7402.
- Main compiles new15-case test TU, links frozen old runtime with reviewed current dependencies and runs only real buffered-sink case: prediction failure did NOT throw, executed1/failures1, exit1 as expectedRED. This stops at the first assertion; the second diagnostics assertion is not claimed exercised on oldcode. task-4-io-red-build.txt / io-red.txt.
- Fixed all-target build and all4CTest pass48host/8GPU/9training/15runtime, warning-free. Newcase exercises both actual bufferedofstream sink failures; prior14cases unchanged. task-4-fix-1-build.txt / ctest.txt.
- Minimal default source runtime/CMake updated and rebuilt warning-free; normal officialinfer still writes4canonical lines. Actual stdout=/dev/full now exit1 with no successrecord; stderr=/dev/full exit1 with already-valid predictions (no rollback promised). task-4-fix-1-io-probes.txt/json.
- Scoped199line diff adds explicit flush/check after both single payload insertions, test+count15, and accurate defaultunit assumption wording; no CUDA pipeline/math/profile event change. task-4-fix-1.diff. Scopedre-review pending.

Scoped re-review approves F1/F2/F3, spec and quality, no new Critical/Important. Main finalhash/link/source command audit is recorded in task-4-final-verification.json / final-link-audit.txt:21 matching currentfiles;67original,15priorreviewed(non-CMake),8protectedremote unchanged. Actualfixedinfer contains only permittedruntime/host/forwarddependencygraph. Small4batchprofile is an operational diagnostic, not fulltest benchmark evidence. Remaining true coursepreprocess/full20epoch/full10000/timeacceptance belongs to Tasks5/6.
