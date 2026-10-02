# Task 13 Compliance Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Task 13 through six bounded, sequentially committed subtasks that close the remaining static-compliance review findings without claiming unavailable CUDA/H20 evidence.

**Architecture:** `scripts/check_prohibited.sh` remains the policy entry point while `scripts/analyze_build_graph.py` owns shell-safe Make recipe parsing, active-input discovery, artifact provenance, and build-log structure. The Makefile emits an expected test-source manifest, focused Python tests exercise each boundary, and documentation describes only guarantees actually enforced by the tools.

**Tech Stack:** Bash 4.4, GNU Make 4.1, Python 3.6 standard library, Python `unittest`, C++14/CUDA Make dry-runs.

**Spec:** `docs/superpowers/specs/2026-09-26-cuda-lenet-mnist-design.md`

## Global Constraints

- Work only in `E:\cuda_lenet_mnist\.worktrees\rebuild` on `feature/rebuild`.
- Preserve the uncommitted round-3 WIP; inspect and complete it rather than reverting it.
- Use test-first development and one non-amended commit per subtask with command-scoped identity `OpenCode <opencode@localhost>`.
- Keep scripts compatible with Ubuntu 18.04, Bash 4.4, GNU Make 4.1, and Python 3.6.
- Production remains CUDA Runtime only: no cuDNN, cuBLAS, TensorRT, Thrust, CUB, cuRAND, other NVIDIA math/framework libraries, or CPU fallback.
- Do not report CUDA compilation, CUDA execution, sanitizer, timing, H20, or accuracy success on the local Windows machine.
- Do not modify unrelated user changes. Each subtask must leave tracked files committed and report any remaining untracked/ignored artifacts.
- Run only the focused tests named by the subtask while implementing; aggregate verification belongs to Subtask 13.6.

---

### Subtask 13.1: Stabilize Recipe Analyzer Foundation

**Files:**
- Create/complete: `scripts/analyze_build_graph.py`
- Modify: `scripts/check_prohibited.sh`
- Modify: `Makefile`
- Test: `tests/test_check_prohibited.py`
- Include: `docs/superpowers/plans/2026-10-02-task-13-compliance-hardening.md`

**Interfaces:**
- Consumes: `make -Bn all` recipe text and `make -s --no-print-directory compliance-test-sources` output.
- Produces: `analyze_build_graph.py source --root ROOT --recipes FILE --manifest FILE`, which prints one normalized active input per line and exits nonzero on malformed or incomplete analysis.

- [x] **Step 1: Audit the inherited WIP before editing**

Run:

```bash
git status --short
git diff -- Makefile scripts/check_prohibited.sh scripts/analyze_build_graph.py tests/test_check_prohibited.py
```

Expected: uncommitted round-3 analyzer work is present; no file is reverted.

- [x] **Step 2: Run the parser/manifest RED tests**

Run:

```bash
python -m unittest -v \
  tests.test_check_prohibited.ProhibitedCheckerTest.test_clang_ccache_quoted_paths_and_iquote_order_are_scanned \
  tests.test_check_prohibited.ProhibitedCheckerTest.test_attached_include_flags_resolve_active_header \
  tests.test_check_prohibited.ProhibitedCheckerTest.test_make_manifest_and_recipe_sources_must_match
```

Expected before stabilization: at least one failure or error from the inherited WIP. If all pass, add a failing case for a wrapper command plus a quoted source/output path before implementation.

- [x] **Step 3: Complete tool-independent tokenization and manifest reconciliation**

Use Python 3.6 `shlex.split(..., posix=True)` for recipe lines. Identify translation units by suffix rather than executable basename, require exactly one supported output for source-bearing recipes, reject response files or shell-control syntax when they affect analysis, and compare the exact normalized `tests/` translation-unit set with the Make manifest.

The Make target must emit only stable records:

```make
compliance-test-sources:
	@for source in $(COMPLIANCE_TEST_SOURCES); do \
	  printf 'test-source=%s\n' "$$source"; \
	done
```

- [x] **Step 4: Verify the focused foundation**

Run the three tests from Step 2, then:

```bash
python -m py_compile scripts/analyze_build_graph.py
bash -n scripts/check_prohibited.sh
```

Expected: all commands pass.

- [x] **Step 5: Commit Subtask 13.1**

Commit subject: `fix: stabilize compliance recipe analysis`

---

### Subtask 13.2: Resolve Active Headers Exactly

**Files:**
- Modify: `scripts/analyze_build_graph.py`
- Test: `tests/test_check_prohibited.py`

**Interfaces:**
- Consumes: per-translation-unit recipe tokens from Subtask 13.1.
- Produces: recursive active-header closure following compiler quote-include order and rejecting unresolved or escaping project-local inputs.

- [ ] **Step 1: Add focused failing tests**

Add/confirm tests for:

```python
def test_active_test_transitive_headers_are_scanned_without_dependency_files(): ...
def test_active_test_unresolved_quoted_include_fails_closed(): ...
def test_active_header_symlink_cannot_escape_repository(): ...
def test_dormant_negative_fixture_is_ignored_but_compiled_test_is_scanned(): ...
```

Also add a duplicate-header test proving resolution order is: including file directory, each `-iquote` directory in command order, then each `-I` directory in command order.

- [ ] **Step 2: Run the header RED tests**

Run the five named tests directly with `python -m unittest -v`.

Expected: the new duplicate-order or boundary case fails before implementation.

- [ ] **Step 3: Implement exact recursive include closure**

Parse separated and attached `-I`/`-iquote` options, preserve their relative ordering within each compiler search class, resolve the complete file through `os.path.realpath`, reject any real path outside the root, recurse only through active includes, and fail on unresolved quoted includes. Angle-bracket system headers may remain unresolved, but any resolved project-local angle include must be scanned.

- [ ] **Step 4: Verify and commit**

Run the five focused tests and Python 3.6 grammar parsing. Commit subject: `fix: resolve active compliance headers exactly`.

---

### Subtask 13.3: Trace Production Artifact Provenance

**Files:**
- Modify: `scripts/analyze_build_graph.py`
- Test: `tests/test_check_prohibited.py`

**Interfaces:**
- Consumes: parsed recipe commands and normalized source/artifact paths.
- Produces: a directed producer/input graph rooted at exactly one `build/lenet_cuda[.exe]`; any reachable `tests/` source causes rejection.

- [ ] **Step 1: Add provenance RED tests**

Cover all of these separately:

```text
ccache clang++ -c tests/cpu_reference.cpp -o build/oracle.o
ld -r build/oracle.o -o build/renamed.o
ar rcs build/libsupport.a build/renamed.o
clang++ build/main.o build/libsupport.a -o build/lenet_cuda
```

Add a positive test where the same test-owned object reaches only `build/workflow_tests`, not `build/lenet_cuda`.

- [ ] **Step 2: Run provenance tests and confirm RED**

Run only `test_clang_ld_and_archive_provenance_reaches_production`, renamed/multilayer tests, response-file rejection, and test-only positive isolation.

- [ ] **Step 3: Implement graph propagation independent of tool basename**

Treat `-o PATH`/attached `-oPATH` as outputs, recognize common archive output/input position, collect artifact-token inputs, reject duplicate producers, and recursively walk only the `lenet_cuda` producer closure. Fail closed for a reachable artifact with no analyzed producer and for reachable response-file or unsupported shell-control input.

- [ ] **Step 4: Verify and commit**

Run the focused provenance tests. Commit subject: `fix: trace production artifact provenance`.

---

### Subtask 13.4: Harden Build Evidence Semantics

**Files:**
- Modify: `scripts/analyze_build_graph.py`
- Modify: `scripts/check_prohibited.sh`
- Test: `tests/test_check_prohibited.py`

**Interfaces:**
- Consumes: an actual or dry-run verbose build log.
- Produces: separate `build` and `dry-run` verdicts; only `build` can report `evidence=successful-build`.

- [ ] **Step 1: Add build-log RED tests**

Require rejection for empty/stale/truncated logs, no compile, no `lenet_cuda` link, missing either exact gencode flag, failure before or after a marker, and marker not final. Require acceptance for `make ... Error 0` and prose containing `error:` that is not a compiler diagnostic.

- [ ] **Step 2: Run the focused build-log tests**

Expected: at least one benign/error-order case fails before implementation.

- [ ] **Step 3: Implement exact evidence rules**

Require a source-bearing `-c` command, exactly a supported `lenet_cuda` link containing both:

```text
-gencode=arch=compute_90,code=sm_90
-gencode=arch=compute_90,code=compute_90
```

For actual mode, reject compiler diagnostics and nonzero Make `Error N`, and require `event=build status=pass target=all` as the final nonempty line. Dry-run mode must print `evidence=commands-not-executed` and never satisfy actual evidence.

- [ ] **Step 4: Verify and commit**

Run only build-log tests plus `bash -n`. Commit subject: `fix: validate compliance build evidence`.

---

### Subtask 13.5: Align Scanner Integration and Documentation

**Files:**
- Modify: `scripts/check_prohibited.sh`
- Modify: `README.md`
- Modify: `docs/h20-acceptance.md`
- Test: `tests/test_documentation.py`
- Test: `tests/compliance_tests.py`

**Interfaces:**
- Consumes: analyzers completed by Subtasks 13.1–13.4.
- Produces: fail-closed source/build scanner CLI and H20 instructions matching its actual guarantees.

- [ ] **Step 1: Add integration/documentation RED tests**

Assert that source mode propagates Make/analyzer/read failures, docs name practical direct/recipe scope rather than arbitrary-obfuscation proof, actual logs live outside `build/`, `make clean` precedes active logging, and the acceptance pipeline uses `set -o pipefail`.

- [ ] **Step 2: Run focused integration tests**

Run `tests.compliance_tests`, `tests.test_documentation`, and scanner failure-injection tests. Expected: documentation or failure-propagation test fails before alignment.

- [ ] **Step 3: Align wrapper and docs**

Keep all temporary cleanup in traps, quote every path, propagate nonzero helper status with its diagnostic, scope source scans to production plus analyzer-returned active test inputs, and document limitations explicitly. Do not describe dry-run output as a successful build.

- [ ] **Step 4: Verify and commit**

Run the focused suites and `git diff --check`. Commit subject: `docs: align compliance scanner guarantees`.

---

### Subtask 13.6: Aggregate Verification and Task 13 Closure

**Files:**
- Modify: `.superpowers/sdd/2026-09-27-cuda-lenet-mnist-implementation/task-13-report.md`
- Modify: `.superpowers/sdd/2026-09-27-cuda-lenet-mnist-implementation/progress.md`
- Modify only if a verification defect is found: files owned by Subtasks 13.1–13.5

**Interfaces:**
- Consumes: all five preceding commits.
- Produces: complete local evidence, a clean Task 13 review package, and an honest H20-only remainder.

- [ ] **Step 1: Run aggregate local verification**

Run:

```bash
make host-tests makefile-tests python-tests compliance
python -m unittest -v tests.test_check_prohibited
bash -n scripts/check_prohibited.sh
python -m py_compile scripts/analyze_build_graph.py scripts/check_comments.py
make -B -n V=1 all
make -B -n V=1 cuda-tests
git diff --check
```

Expected: all locally executable checks pass; dry-runs clearly state commands were not executed.

- [ ] **Step 2: Verify Python 3.6 grammar**

Parse every new/modified Python file with `ast.parse(..., feature_version=(3, 6))`. Expected: no syntax error.

- [ ] **Step 3: Update report and ledger**

Record exact commands/results, all six subtask SHAs, review findings closed, and remaining target-only work. Do not mark semantic comment review or H20 acceptance complete.

- [ ] **Step 4: Commit closure**

Commit subject: `test: close task 13 compliance hardening`.

- [ ] **Step 5: Generate and review the full Task 13 package**

Generate a review package from `cd1222cf61687a12a01e31a33cd6b908313dc0a3` through the closure commit. Accept Task 13 only if the reviewer reports no Critical or Important findings; otherwise create a narrowly scoped follow-up subtask without combining it with Task 14.

## Self-Review

- Spec coverage: source prohibition, no CPU fallback, active test scope, comment gate, real output grammar, build evidence, persistent H20 logging, architecture flags, and honest local/H20 evidence are assigned.
- Placeholder scan: no deferred implementation placeholders are present; target-only CUDA/H20 work is explicitly Task 14 rather than omitted Task 13 work.
- Interface consistency: every analyzer invocation uses `source --root --recipes --manifest` or `build-log --root --log`; Make emits `test-source=PATH`; the success marker remains `event=build status=pass target=all`.
