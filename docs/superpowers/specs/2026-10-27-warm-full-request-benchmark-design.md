# Warm full-request inference timing amendment

User-confirmed timing amendment,2026-10-27. Supersedes the earlier cold-process
timing acceptance ONLY; model accuracy, CUDA-only execution, official interfaces,
source preservation, training provenance and submission constraints remain.

## Intent and confirmed boundary

The user requires warmup followed by more than100 inference repetitions, with
arithmetic mean and median. They explicitly include every repetition's input
transfer, preprocessing and result readback. One repetition remains the entire
10000-image test request, not one image or one1024-image batch.

Controller defaults:5 full warmups then101 measured full requests, batch1024.
Every repetition redoes H2D, GPU FP32 preparation, original forward+argmax, D2H,
stream synchronization and host output validation. Timing uses host steady clock
from before sticky-flag reset to after synchronized readback/validation. Previous
iteration work must be complete; do not time enqueue only or sum GPU-only events.
All106 prediction vectors must match. Warmup durations are retained separately
and excluded from measured statistics. No slow measured sample is discarded.

Model/image file loading, CUDA initialization, allocation, weight upload and final
stdout formatting are outside the warm per-request interval and disclosed as
such. Input data is host-resident initially, not GPU-resident across repetitions:
each repetition actually transfers all images again. Existing cold five-process
series remains historical diagnostics and is not relabelled as warm measurement.

## Approach

Chosen narrow approach: a development-only CUDA benchmark executable reuses
`course::InferenceModel`, `LaunchPrepareInput` and `LaunchArgmax` with the same
batch offsets/stream order as official inference. It loads/validates once and
preallocates once, then repeats the complete request. This avoids changing the
official argv/output protocol or introducing a risky production session refactor.

Rejected: repeated fresh infer subprocesses (cannot meet warmed CUDA boundary);
GPU-only event timing (omits user-required transfers/readback/host boundary);
cached predictions (not inference). No CPU NN fallback or platform bypass.

The executable takes no labels. An external stdlib collector validates raw round
records/prediction output, owns labels, binds model/meta/binary/official-infer
hashes and computes mean/median/min/max. The tool validates every repetition's
actual readback against its first real prediction vector, including warmups.

## Data and acceptance

New schema `course-warm-benchmark-v1`; do not overload the old cold-process
`course-benchmark-v1`. Save the entire process stdout/stderr/exit/wall,5 warmup
records,101 measured records, all durations, baseline predictions and explicit
per-round prediction-invariance evidence. Main independently parses raw records,
recounts predictions against original labels and rechecks hashes after execution.

The existing actual20epoch course-trained best model remains the source; no
retraining or test-label access inside CUDA tools. Accuracy must remain>=96% on
all10000 images. Report both mean and median. A warm<100ms result requires both
mean and median<100ms; range/max are still disclosed even if outliers exceed100.
Never assert success before actual measurement. This new gate does not turn the
old cold-process~1.66s outcome into a successful cold result.

## Submission artifacts

Keep one primary metric group: `Test Accuracy: ...` and
`Test Time: mean ... ms; median ... ms`, labelled warm full-request timing,
5 warmups/101 measured full10000-image requests and observed range. Report all
measured durations, included/excluded stages, configuration, current identities,
training completion, preprocessing assumption and unknown official leaderboard
scope. Warm-tools/Python/tests/data/model/evidence remain excluded from ZIP.

Old cold helper/schema/tests stay valid for historical reproducibility. Warm
packaging reuses the existing whitelist/exclusive-write safeguards but requires
strict warm provenance and>100 measured rounds. Preserve old actual root artifacts
and ZIP under owned versioned paths before exclusive new publication. Fresh
default extracted build/roundtrip, valid-model inference, review and fresh backup
remain required. Real HEAD/index/deployment/original data/weights stay unchanged.

## Scope/approval

The user's explicit timing correction and boundary confirmation authorize this
amendment; existing uninterrupted execution/subagent model preference is retained.
The counts, tool organization and mean+median gate are controller implementation
choices, not a claimed official course rule. No further per-task approval needed.
