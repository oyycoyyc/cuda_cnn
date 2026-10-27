# Rulings made during execution

All decisions and their tradeoffs/costs are retained here in chronological order.

- Ruling: Reuse the existing detached linked worktree and preserve dirty baseline — all prior work already backed up; creating/resetting isolation would risk loss — cost if wrong: broader diff, mitigated with task-scoped review packages and checkpoint refs.

- Ruling: Implementers never stage/commit/push; controller records reviewed checkpoint commits on fresh backup refs using a separate index, leaving real HEAD/index unchanged — matches explicit Git-backup requirement without overwriting user staging — cost: non-linear backup refs, documented here.

- Ruling: Use deepseek/deepseek-flash explicitly for coding/review agents and main independent acceptance — user-specified subagent model overrides default tier heuristics — cost if wrong: extra review turns, not permission to weaken tests.

- Ruling: Tests-only first handoff before every production task; main owns remote CUDA RED/GREEN. Task1 may run local host RED itself if portable, otherwise main verifies Linux host RED — local host has no CUDA compiler/device — cost: handoff overhead.

- Ruling: Optional profile strings other than exact"1" disable profiling; reject empty/unknown values for mode/epochs/batch and missing model dir, not for profile — plan's exact1 specification controls general optional-env rejection wording — cost: ignored typo in diagnostic-only switch.

- Ruling: Host contract Linux/POSIX publication tests may run on H20 without CUDA calls; no new Windows/CUDA support work — preserves actual target environment — cost: some new host tests cannot run natively on Windows.

- Ruling: Keep original Make policy/scripts unchanged and new tests under course/tests with explicit CTest registration — avoids implicit wildcard/source-manifest breakage — cost: separate course build/link audit required, not a false old-scan coverage claim.

- Ruling: Known remote pyarrow import error and sanitizer self-disable remain documented baseline limitations; continue approved feature work and verify unaffected/new suites, without dependency installation or re-running old36 diagnoses — user approved plan carrying these limits — cost: complete old aggregate target not green.

- Ruling: Preserve this plan workspace/evidence at finish rather than deleting recovery records while all user-facing changes are checkpointed outside HEAD — user asked Git backups and progress retention — cost: small scratch files remain.

- Ruling: Replace the test harness's fixed scratch root and unconditional startup recursive delete with an exclusively created per-run owned directory before any runtime test — otherwise parallel CTest suites or unrelated pre-existing scratch content could be deleted — cost: small test-only portability branch, no production behavior change.

- Ruling: Model API comments must state the plan's actual publication guarantee (old valid bundle preserved OR mixed pair rejected), not promise rollback for every I/O error after first rename — two-file publication is not a transaction — cost if wrong: may require future transactional generations, not needed by sequential course train/infer contract.

- Ruling: Correct Task1's plan-mandated rejection of every unrelated work-directory artifact — course /work is caller-owned and not guaranteed empty, while the spec only requires protecting inputs/unrelated work. SaveCourseModel must preserve unrelated entries, refusing unrecognized files/symlinks/hardlink aliases at its reserved model filenames; development model_bundle tool may still require a fresh destination. Replace the old extra-artifact refusal test with successful preservation plus separate reserved-filename rejection tests — cost: narrower validation than original plan wording, required for real course compatibility. No human question needed because source rules and approved design resolve it.

- Ruling: File content/value violations throw runtime_error (CLI runtime exit1), argument/env/shape syntax violations remain invalid_argument (usage exit2) — matches public contract and later official CLI error mapping — cost: existing generic exception tests need typed additions, no successful-call behavior change.

- Ruling: Preserve the valid augmentation contract; replace the erroneous throw assertion with the intended null-indices rejection and use identical packed source images for equal-index translation comparisons — task requires equal transform for equal original index, not replacing image data by index — cost: test fixture correction before GREEN; no scope change.

- Ruling: Direct RunCourseTrain honors supplied positive TrainOptions.batch_size, checked for conversion to the existing int-capacity GPU API; CLI stays128. Do not invent a1024 training limit from the separate inference default — no approved training maximum was specified — cost: very large direct-call capacities can fail real memory checks instead of an arbitrary argument cutoff. Tiny4-sample batches are legitimate bounded tests.

- Ruling: Approve narrow production-used split/AdamW/schedule/best-selection policies to test actual training decisions without twenty-epoch fixtures — no new test-only hooks or oldworkflow dependency — cost: a few small declared helper APIs, all used by real workflow.

- Ruling: Runtime output atomicity means validation/model/input/CUDA/nonfinite failures occur before publication; arbitrary output-sink I/O failure cannot promise rollback of already-written stdout bytes — narrow newheader's universal every-failure promise, keep required zero-output validation failures and one-publication design — cost: honest I/O boundary wording, no successfulbehavior change.

- Ruling: Before correcting runtime, add a deterministic batch-sensitive two-image realGPU fixture that compares literalexpected classes and batch1/2 outputs; main must reproduceRED on currentimplementation. Use capacity-sized packed/normalized buffers with pointer0 reused perbatch, hostsource offset*784 and globalpredictionoffset unchanged — this is the approved batching design, not a math change — cost: one additional focused runtimecase and correctbufferlayout. Independent taskreview follows actualfunctionalGREEN, not the masked13casePASS.

- Ruling: Preserve the pinned benchmark schema and every executed process: --profile means the requestedR runs are profiled DIAGNOSTICS in a separate profiled=true artifact, not Rfastcalls plus one unrecorded extra process. Main invokes separate --profile --runs1 after fastR5; all actual stdout/stderr/exit/wall remain in runs. Correct proposed test's mixed/unrecorded-profile assumption before GREEN — otherwise exactschema has nowhere to retain the additionaldiagnostic record and mislabels fast samples — cost: localtestsemanticsclarification, no new CLIflags/schemafields or timing substitution.

- Ruling: Keep approved inference default1024; measuredmedians varyonly~1%against~1.2ssetup and shared-platformnoise, no justified largerbatchdefaultoradditionaltuning — respectsstop-expandingtrialswheninitializationdominates — cost: don'tcherrypickthefastesttinyvariance asoptimization. Do not rerunold36startupdiagnoses or alterplatformlibraries.

- Ruling: Re-run old tests in a fresh owned source snapshot because their fixed build/workflow scratch files overwrite/delete on normal execution — preserve all prior experiment work and still exercise unchanged actual tests — cost: another small build/source directory; current course-specific full build remains in its own fresh build path for complete compilation/link evidence.

- Ruling: Use conservative decimal caps100000000/300000000 archive bytes (strictly below) and4000000 image bytes (inclusive), rather than choosing largerMiB interpretations of courseMB limits — guarantees compliance whichever unit convention the course uses — cost: slightly tighter cap, irrelevant to this tiny source/PNG bundle. Correct synthetic boundary tests and report before implementation.

- Ruling: Package helper image-limit pure API takes `(image_bytes, pixels)` matching actual tests; renderer passes width*height, tests cover positive dimensions at actual image boundary — removes proposed report's accidental3argument mismatch — cost: no behavior/CLI change. Source/log/hash consistency establishes local provenance, not cryptographic proof against a maliciously forged entire evidence set; main actual training/external measurements remain necessary.
