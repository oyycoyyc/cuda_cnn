# SDD ledger — plan: docs/superpowers/plans/2026-09-27-cuda-lenet-mnist-implementation.md

## Setup

- Spec reachable: `docs/superpowers/specs/2026-09-26-cuda-lenet-mnist-design.md`.
- Baseline: `b80751f` (`docs: add CUDA LeNet design and implementation plan`).
- User selected subagent-driven execution and approved `.worktrees/rebuild` after Task 1 creates the build skeleton.
- Git identity is not configured. Commits use command-scoped identity `OpenCode <opencode@localhost>` without changing Git configuration.
- Isolated worktree: `E:\cuda_lenet_mnist\.worktrees\rebuild`, branch `feature/rebuild`, created at `feb2e81` after Task 1 review.
- Worktree baseline: host smoke and permanent Make collision regression both pass under MSYS2 GCC 14.2.0; CUDA/H20 baseline is unavailable locally.

## Preflight Consistency Scan

| Producer / consumer | Shared file or interface | Finding |
|---|---|---|
| Task 1 / Tasks 2-14 | `Makefile`, test harness, build directories | Consistent: Task 1 establishes targets; later tasks extend them sequentially. |
| Task 2 / Task 3 | `MNISTC1\0` binary bytes | Consistent: images are contiguous before labels and header fields match the spec. |
| Task 3 / Task 5 | `src/binary_io.h` | Consistent: checkpoint code reuses checked little-endian scalar helpers. |
| Task 3 / Tasks 11-12 | `MnistDataset`, count enforcement | Consistent: low-level parsing accepts nonzero synthetic counts; workflows enforce role-specific official counts. |
| Task 4 / Task 5 | `ParameterSet`, canonical schema | Consistent: checkpoint validation consumes the sole schema table. |
| Task 4 / Task 7 | host/device SplitMix64 translation | Consistent if integer methods are inline in `random.h`; implementation must not duplicate hash logic in `input.cu`. |
| Task 4 / Task 10 | initialized parameter set | Consistent: model imports the canonical ten tensors. |
| Task 4 / Tasks 11-12 | split, epoch shuffle, original indexes | Consistent: only training indexes reshuffle and translation uses original sample IDs. |
| Task 5 / Tasks 10-12 | checkpoint parameter import/export | Consistent: optimizer state remains intentionally absent. |
| Task 6 / Tasks 7-9 | `layers.h`, `operator_tests.cu`, CUDA helpers | Consistent: launcher signatures are frozen before kernel implementation. |
| Task 7 / Task 10 | input, activation, pooling launchers | Consistent: pre-activations and pool winners remain live through backward. |
| Task 8 / Task 10 | convolution and linear launchers | Consistent: all backward outputs are gather-based and race-free. |
| Task 9 / Tasks 10-12 | loss, metrics, AdamW, finite scan | Consistent: actual batch size and one-based global step flow through all callers. |
| Task 10 / Task 12 | `LeNet` model API and workflow staging | Conflict: Task 10 text says LeNet owns input staging, but its locked API accepts normalized device input and exposes no staging access. See Ruling 3. |
| Task 11 / Task 12 | options, reporting, `HostBatch`, learning schedule | Conflict: Task 11 requires host-only batch tests but lists no dedicated production file, while `src/train.cu` would require NVCC. See Ruling 2. |
| Task 12 / Task 14 | executable commands and stable summaries | Consistent: acceptance calls exactly the implemented CLI. |
| Task 13 / Task 14 | scanners, checklist, README, acceptance guide | Consistent: automated presence checks precede manual semantic review. |
| Task 1 internal | `git init`, RED/GREEN smoke target, commit | Conflict: setup required a baseline repository before SDD artifacts and briefs could be generated. See Ruling 1. |
| Task 2 internal | tests vs converter/download behavior | Consistent; target Python 3.6 hash/install verification is platform-bound. |
| Task 3 internal | malformed tests vs checked parser | Consistent. |
| Task 4 internal | independent vectors vs production protocol | Consistent if generated vectors are committed and C++ tests use fixed constants. |
| Task 5 internal | format tests vs field-wise serializer | Consistent. |
| Task 6 internal | CPU hand fixtures and CUDA RAII | Consistent; host and H20 evidence must remain separately labeled. |
| Task 7 internal | edge tests vs kernel semantics | Consistent. |
| Task 8 internal | CPU comparison and finite differences | Consistent. |
| Task 9 internal | stability/update/scanner tests | Consistent. |
| Task 10 internal | full-network tests vs fixed storage | Consistent after Ruling 3. |
| Task 11 internal | parser/report/batch tests vs host logic | Consistent after Ruling 2. |
| Task 12 internal | workflow tests vs train/evaluate/infer | Consistent. |
| Task 13 internal | negative fixtures vs scoped scanners | Consistent. |
| Task 14 internal | H20 commands vs acceptance claims | Consistent; cannot be completed on the local Windows machine. |

## Rulings

- Ruling 1: Task 1 skips `git init` because the controller created baseline commit `b80751f` so the SDD workspace and task brief could exist. The required RED remains `make host-tests` failing before the Makefile is added. Cost if wrong: the Task 1 report differs from the literal first command but preserves the behavior and repository history.
- Ruling 2: Task 11 may add `include/training_data.h` and `src/training_data.cpp` for `HostBatch` and `PackBatch`, and list them in its commit. This keeps host batch tests CUDA-independent and follows the spec's readability goal. Cost if wrong: two small files beyond the initial repository layout.
- Ruling 3: `LeNet` owns model parameters, moments, activations, gradients, pool indexes, losses, metrics, and scan storage. Each Task 12 workflow owns its packed-input, label, original-index, and normalized-input device staging buffers, allocated once before its loop. Cost if wrong: workflow storage is outside `LeNet`, but there is still no batch-loop allocation and the public model API remains coherent.
- Ruling 4: The OpenCode `task` interface exposes subagent type but no model selector, so implementer/reviewer model choice cannot be set explicitly. Use `general` agents and preserve all other SDD isolation/review controls. Cost if wrong: agent cost/capability cannot be tuned per role.
- Ruling 5: Task 2's phrase “nine versions/packages” is a plan counting error. The binding spec Section 6.2 names exactly eight packages: `numpy`, `Pillow`, `pyarrow`, `requests`, `certifi`, `charset-normalizer`, `idna`, and `urllib3`, with the listed versions. Implement and hash exactly those eight; do not invent a ninth dependency. Cost if wrong: a genuinely required transitive dependency could be omitted, but the pinned Requests dependency closure and PyArrow/NumPy/Pillow set are complete per the spec.
- Ruling 6: Supersede the loss/metrics portion of Ruling 3. Task 10 `LeNet` owns only parameters, gradients, moments, model activations/activation gradients, pool winners, and finite-scan storage reachable through its locked API. Task 12 workflow objects own input/label/index staging plus probabilities, per-sample losses, logits gradients, predictions, correct flags/count, and timing storage, all preallocated before loops. Cost if wrong: workflow-owned loss/metric storage differs from the Task 10 prose list, but avoids inaccessible duplicate buffers while preserving every no-loop-allocation and ownership requirement.
- Ruling 7: Task 11 `ParseCli` validates command syntax, option ownership, required option presence, numeric syntax/ranges, duplicates, and defaults only. Filesystem readability/output-parent checks, loaded dataset official counts, CUDA device availability, and inference index versus loaded count belong to Task 12 workflows. Task 11 adds `include/training_data.h`/`src/training_data.cpp` with `MakeWorkflowSplit` and `PackBatch` so official/nonstandard split and batch correspondence remain host-testable. Cost if wrong: four runtime validation cases move one task later, but the parser remains deterministic and free of filesystem/CUDA side effects.
- Ruling 8: Task 13 makes `cuda-tests` prepare the verified official dataset before running workflow tests and passes `--mnist-train data/train.bin` to the workflow suite. This incorporates the corrected Task 12 H20 ordering for the real-MNIST overfit and inference cases. Cost if wrong: the aggregate CUDA test target requires network/cache access and Python data dependencies, while individual CUDA binaries remain directly runnable for focused work.

## Task Progress

- Task 1: fix round 1/5 (1 addressed, 1 open — executable collision fixed; chronological RED remains invalid; commits `8b3d53b..acc8f24`).
- Task 1: minor (deferred): same-basename collision fixture was temporary, so the Make mapping has no permanent regression test; final review must triage this.
- Task 1: fix round 2/5 (1 addressed, 0 open — reverted Task 1, observed valid missing-target RED, reimplemented fresh, and added permanent collision regression; commits `acc8f24..feb2e81`).
- Task 1: prior deferred minor resolved by `tests/makefile_tests.sh` in `feb2e81`.
- Task 1: complete (commits `b80751f..feb2e81`, review clean).
- Task 2: fix round 1/5 (1 addressed, 0 open — completed NumPy manylinux1 and PyArrow manylinux2010 hashes and added exact lock/temporary cleanup regressions; commits `d40475d..4c86c33`).
- Task 2: complete (commits `feb2e81..4c86c33`, review clean; Python 3.6 runtime and production conversion remain target-only verification).
- Task 3: complete (commits `4c86c33..b586492`, review clean; GCC 7.5 remains target-only verification).
- Task 4: complete (commits `b586492..174d341`, review clean; NVCC/GCC 7.5 remain target-only verification).
- Task 5: fix round 1/5 (1 addressed, 0 open — independently bit-compared all 44,426 payload values; commits `093da88..7b59111`).
- Task 5: complete (commits `174d341..7b59111`, review clean; POSIX rename execution remains target-only verification).
- Task 6: fix round 1/5 (5 Important and 1 Minor addressed, 1 new Important open — strengthened CPU fixtures/contracts, fixed ODR shape, added move-assignment coverage; commits `3e05f8b..d9e611f`).
- Task 6: fix round 2/5 (1 addressed, 0 open — allocation accounting is atomic and macro-independent; commits `d9e611f..05b67a7`).
- Task 6: complete (commits `7b59111..05b67a7`, review clean; CUDA compile/runtime remain H20-only verification).
- Task 7: complete (commits `05b67a7..9f84519`, static review clean; CUDA compile/runtime remain H20-only verification).
- Task 8: fix round 1/5 (1 Critical addressed, 0 open — supplied convolution output-channel stride parameter; commits `58c89d4..c1a6e3d`).
- Task 8: complete (commits `9f84519..c1a6e3d`, static review clean; CUDA compile/runtime remain H20-only verification).
- Task 9: fix round 1/5 (5 Important and 1 Minor addressed, 0 open — fixed shift-stable CE and hardened independent CUDA oracles/validation boundaries; commits `73773d9..a463315`).
- Task 9: complete (commits `c1a6e3d..a463315`, static review clean; CUDA compile/runtime remain H20-only verification).
- Task 10: fix round 1/5 (3 Important addressed, 1 partially open — added execution state, allocation events, flatten probe, byte/tail/nondefault-stream tests; commits `054e081..c115f75`).
- Task 10: fix round 2/5 (1 addressed, 0 open — query/allocation temporal event proof; commits `c115f75..36c2809`).
- Task 10: complete (commits `a463315..36c2809`, static review clean; CUDA compile/runtime remain H20-only verification).
- Task 11: fix round 1/5 (2 Important and 1 Minor addressed, 0 open — isolated report formatting and checked packed-image multiplication; commits `218f05a..eaad2a9`).
- Task 11: complete (commits `36c2809..eaad2a9`, review clean; GCC 7.5 remains target-only verification).
- Task 12: fix round 1/5 (5 Important and 1 Minor addressed, 0 open — removed loop allocations, real-MNIST overfit/infer, discriminating reload hook, CLI end-to-end diagnostics, safe offsets; commits `b075fea..cd1222c`).
- Task 12: complete (commits `eaad2a9..cd1222c`, static review clean; all CUDA workflow/runtime/accuracy evidence remains H20-only).
- Task 13: complete (strict test-first compliance/documentation implementation; 51 Python tests, host suites, Make dry-runs, and source/comment/build-log gates pass locally; Python 3.6 runtime plus all CUDA/H20 evidence remain target-only).
- Task 13: fix round 1/5 (all review issues addressed with fail-closed Make-scoped scans, verified build/dry-run evidence modes, signature-based 147-item comment inventory, production-derived output tests, persistent H20 logs, and accurate cleanup claims; 52 compliance tests and nine host suites pass locally; Python 3.6 runtime and CUDA/H20 execution remain target-only).
- Task 13: fix round 2/5 (3 review findings addressed, 0 open from this round - clean-tree recursive active-header scope, transitive tests-owned production-link provenance, and final-marker/failure-record build-log validation; 25 focused scanner tests, 75 aggregate Python tests, 60 compliance tests, and nine host suites pass locally; Python 3.6 runtime and CUDA/H20 execution remain target-only).
- Task 13 compliance hardening 13.1: complete (recipe parsing is tool-name independent and shell-tokenized, source-bearing commands require one output, Make's test-source manifest exactly reconciles with recipe translation units, and wrapper failures propagate; five focused tests plus Python byte-compilation/3.6 grammar and Bash syntax checks pass; inherited header/provenance/build-log/docs work remains queued for Subtasks 13.2-13.5).
- Task 13 compliance hardening 13.1 fix round 1/5 (5 review findings addressed, 0 open from this round - positional archive inference is limited to recognized archive syntax, source-bearing recipes always require explicit `-o`, tests-owned recipes reject response files and robustly detected shell controls before manifest reconciliation, extra recipe sources fail reconciliation, and malformed/nonzero Make invocations propagate; 13 focused parser/checker tests plus Python byte-compilation/3.6 grammar, Bash syntax, and diff checks pass; Subtasks 13.2-13.5 remain unchanged).
- Task 13 compliance hardening 13.2: complete (active quoted includes follow including-directory, ordered `-iquote`, then ordered `-I` search semantics per translation unit; attached/separated quoted include options are classified exactly, reachable project-local angle headers recurse, unresolved quoted includes and canonical symlink escapes fail closed, and dormant duplicates remain excluded; all five named tests pass without skips, including the Windows junction-backed escape regression; provenance/build-log/docs behavior remains queued for Subtasks 13.3-13.5).
- Task 13 compliance hardening 13.2 fix round 1/5 (2 review findings addressed, 0 open from this round - recursive quoted includes preserve the compiler-selected lexical path and distinct lexical routes to one real header retain independent visited contexts; angle includes ignore source/`-iquote` decoys and select the first ordered `-I` candidate; seven focused tests pass with the real file-symlink integration case skipped for Windows privilege while its deterministic same-identity companion and external junction escape case run; Python byte-compilation/3.6 grammar and diff checks pass; Subtasks 13.3+ remain unchanged).
- Task 13 compliance hardening 13.2 fix round 2/5 (the remaining cycle issue is addressed - visit identity combines the canonical file, canonical lexical containing directory, and include context, terminating equivalent directory-symlink recursion without merging distinct file-symlink directory semantics; nine focused cycle/symlink/order tests pass with two Windows privilege skips while deterministic identity coverage runs; Subtasks 13.3+ remain unchanged).
- Task 13 compliance hardening 13.2 fix round 3/5 (the lexical-context collision is addressed - global visits use the full normalized lexical path and include context while a canonical-real-file DFS stack terminates only current-chain recursion and is cleared in `finally`; ten focused active-header tests pass with two Windows privilege skips while deterministic parent-relative-route and modeled-recursion regressions run; Subtasks 13.3+ remain unchanged).
- Task 13 compliance hardening 13.3: complete (the production graph is rooted at exactly one `build/lenet_cuda[.exe]`, traces suffixless renamed layers independently of compiler/linker names, supports common `ar`/`llvm-ar`/`gcc-ar` creation forms, and rejects reachable tests-owned sources, response files, shell controls, duplicate/missing producers, and escaping artifact inputs without treating include/output operands as artifacts; a test-only `workflow_tests` branch remains isolated; 18 focused provenance tests plus Python byte-compilation/3.6 grammar and diff checks pass; Subtasks 13.4+ remain unchanged).
- Task 13 compliance hardening 13.3a: complete (production-reachable and tests-owned source commands now fail closed on `-Wl,`, direct linker forwarding, paired `-L`/`-l` inputs, bare/compiler/linker response options, residual shell variables, and unquoted globs while preserving current NVCC/gencode flags and quoted/escaped literals; 15 focused analyzer/checker tests pass, with extensionless tokens, general option operands, canonical tests ownership, and expanded archive grammar left to 13.3b/c).
- Task 13 compliance hardening 13.3a fix round 1/5 (forwarded response files in NVCC compiler/linker options now fail closed in production/tests-owned commands, bracket globs accept a literal leading `]` without treating an unclosed bracket as expansion, and shell comments are removed consistently before tokenization/ambiguity analysis without hiding quoted, escaped, or in-word hashes; 21 focused analyzer/checker tests pass and 13.3b/c remain unchanged).
- Task 13 compliance hardening 13.3a fix round 2/5 (the final bracket scanner now preserves partially quoted/escaped members, skips complete POSIX class/collating/equivalence subexpressions before requiring an outer unquoted close, and leaves malformed inner-only or plain unmatched brackets non-expanding; 25 focused analyzer/checker tests pass and 13.3b/c remain unchanged).
