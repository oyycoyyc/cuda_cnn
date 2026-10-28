# Controller rulings (in decision order)

1. Use5fullwarmups+101measured full10000-image requests atbatch1024, from
   before flag reset to after synchronized readback/host validation, with no
   allocation/events inside the interval. Reason: implement the confirmed
   transfers/preparation/readback boundary. Cost: the result may exceed100ms
   and must be reported truthfully; changed boundaries require new measurement.
2. Add an optional development tool reusing InferenceModel/input/argmax rather
   than refactoring official infer. Reason: keep identical GPU math without
   changing the required argv. Cost: a small duplicated orchestration loop
   could drift; literal-class/tail/officialinfer comparison tests mitigate this.
3. Require both warm mean and median<100ms, retain every measured sample and
   disclose range; preserve cold full-wall failure as historical diagnostics.
   Reason: the user asks for mean and median after warmup. Cost: a stricter
   dual-statistic gate; one passing statistic is not sufficient.
4. Reuse the known isolated workspace and existing actual20epoch model; avoid
   installations, preserve originals, and use temporary Git indexes/backup refs.
   Reason: avoid redundant training/worktrees and approval stalls. Cost: clear
   separate ledger/schema/evidence is needed to prevent mixed provenance.
5. Replace the implementation-detail requirement “one subprocess.run” with
   exactly one Popen child using communicate/kill/drain on timeout. Reason:
   independently reproduced Linux run() timeout-after-pipe-EOF lost captured
   diagnostics, violating the real one-child/complete-evidence contract. Cost:
   a small explicit process-lifecycle implementation; no child is relaunched.

No ruling authorizes changing the NN arithmetic, using CPU inference, exposing
labels to the CUDA child, trimming outliers, replacing the measured series,
changing machine configuration, deleting original work, or pushing/deploying.
