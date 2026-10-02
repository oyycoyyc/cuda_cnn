# Task 13 Report

## Status

Implemented Task 13 static compliance checks, documentation, output grammar,
Make targets, and corrected H20 data/workflow ordering. No CUDA, H20, training,
timing, sanitizer, or accuracy success is claimed from the local machine.

## Changes

- Added a scoped source/build scanner for prohibited headers, namespaces, API
  families, linker inputs, production CPU fallback references, and prohibited
  standard random APIs. Source scope is `Makefile`, `include/`, `src/`, and C++/
  CUDA files under `tests/`; README, specifications, checker source, Python
  negative fixtures, and other documentation are excluded.
- Added a Python 3.6-compatible comment-presence checker for global public
  types/functions, public methods, multiline launcher declarations, and every
  production `__global__` definition. It reconciles 104 stable declaration/
  kernel IDs with the manual checklist and supports `--require-reviewed`.
- Added 15 manual checklist entries for CUDA checking and binary/data
  serializers/parsers. All 119 entries remain unchecked for the Task 14 human
  semantic review gate.
- Added focused checker/documentation tests plus the requested
  `tests/compliance_tests.py` and `tests/output_format_tests.py` suites.
- Added `all`, `python-tests`, `prepare-data`, and `compliance` Make targets.
  `make test` aggregates host, CUDA, Python, and compliance gates. The default
  build now produces the application and test binaries.
- Corrected aggregate CUDA test ordering: official data preparation precedes
  workflow execution, and `workflow_tests` receives
  `--mnist-train data/train.bin` for its real-MNIST cases.
- Documented environment, pinned data sources/hashes, architecture/layouts,
  binary formats, CLI commands/ranges/defaults, determinism, exact output
  grammar, exit statuses, CUDA Event timing, 99% threshold behavior, errors,
  troubleshooting, and the explicit local-no-CUDA boundary.
- Added an H20 procedure that prepares official data before overfit/inference,
  passes `--mnist-train` to both real-MNIST cases, and records build, sanitizer,
  architecture, dependency, training, evaluation, inference, and manual comment
  evidence.

## TDD Evidence

RED was recorded immediately after adding tests and before adding checker
scripts, documentation, or Make behavior:

```text
python -m unittest -v tests.test_check_prohibited tests.test_check_comments \
  tests.test_documentation tests.output_format_tests tests.compliance_tests
Ran 33 tests
FAILED (failures=43, errors=17)
```

Failures were caused by absent checker CLIs/documents, absent Make compliance
targets, absent official-data workflow ordering, and absent output grammar. The
smallest implementations were then added. Scanner fixture failures also exposed
and corrected case-insensitive `NvInfer.h` matching and per-subtest fixture
isolation before the suite became green.

Final GREEN command on the available local interpreter and toolchain:

```text
$env:BASH = 'C:\personal_apps\msys64\usr\bin\bash.exe'
$env:MAKE = 'C:\personal_apps\msys64\usr\bin\make.exe'
& 'C:\personal_apps\anaconda3\python.exe' -m unittest -v \
  tests.test_prepare_mnist tests.test_data_interop \
  tests.test_check_prohibited tests.test_check_comments \
  tests.test_documentation tests.output_format_tests tests.compliance_tests
Ran 51 tests in 11.003s
OK
```

## Verification

- `C:\personal_apps\anaconda3\python.exe -m unittest ...` for all seven Python
  modules: 51 tests, 0 failures.
- `make PYTHON=C:/personal_apps/anaconda3/python.exe python-tests`: 15 tests,
  0 failures.
- `make PYTHON=python compliance`: 36 tests, 0 failures; comment inventory and
  source policy gates passed.
- `make host-tests makefile-tests`: nine host suites passed; permanent distinct
  host/CUDA same-basename Make regression passed.
- `python -m py_compile` for all changed Python scripts/tests and existing
  related Python files: exit 0.
- Python `ast.parse(..., feature_version=(3, 6))` over the six new Python files:
  `Python 3.6 grammar check passed for 6 files`.
- `bash -n scripts/check_prohibited.sh`: exit 0.
- `bash scripts/check_prohibited.sh source .`:
  `compliance scan passed: mode=source`.
- `python scripts/check_comments.py --root . --checklist
  docs/comment-review-checklist.md`:
  `comment check passed: declarations_and_kernels=104 checklist_items=119`.
- `make -B -n V=1 all | tee build/verbose-build.log` emitted both
  `-gencode=arch=compute_90,code=sm_90` and
  `-gencode=arch=compute_90,code=compute_90`.
- `bash scripts/check_prohibited.sh build build/verbose-build.log`:
  `compliance scan passed: mode=build`.
- `git diff --check`: exit 0 before report creation; rerun before commit.

## Concerns And Target-Only Work

- Python 3.6 is not installed locally. New Python files pass a Python 3.6
  grammar parse, but interpreter/package execution must be repeated on Ubuntu
  with the hash-locked Python 3.6 environment.
- Local `nvcc`, an NVIDIA GPU, and H20 access are unavailable. CUDA compilation,
  CUDA numerical/workflow execution, Compute Sanitizer, `cuobjdump`, dynamic
  dependency inspection, default training, timing, and >=99% accuracy remain
  H20-only and were not run or claimed.
- The manual comment checklist is deliberately unchecked. Task 14 must inspect
  each item semantically and then run `--require-reviewed`.
- `make cuda-tests` now verifies/prepares official MNIST and therefore needs the
  Python data dependencies plus network access or valid cached Parquet files.
  Focused CUDA binaries remain runnable directly without invoking that aggregate
  target.

## Fix Round 1

### Review Issues Addressed

- Relocated the active H20 terminal log to `acceptance/`, outside the directory
  removed by `make clean`, so later build cleanup cannot unlink the evidence.
- Replaced fail-open scanner pipelines with checked `find`, `grep`, and Make
  resolution. Source scope now includes production headers/sources plus only
  test inputs present in the Make database; dormant negative fixtures are
  excluded.
- Expanded practical direct-form coverage across TensorRT APIs/namespaces,
  forced and macro includes, linker spellings, NVIDIA math/communication
  libraries, framework headers, standard random facilities, and production CPU
  fallback identifiers. The exact `lenet_cuda` dry-run link command is checked
  independently for CPU reference objects.
- Split build evidence into explicit `build` and `dry-run` modes. Actual logs
  must be nonempty and fresh, contain a compile command, a `lenet_cuda` link
  command with both exact architecture flags, and the post-build
  `event=build status=pass target=all` marker. Dry-run output is labeled
  `commands-not-executed` and cannot satisfy actual-build mode.
- Replaced the comment line heuristic with a balanced-scope inventory covering
  namespace functions, multiline declarations, nested public types, overload
  signatures, forward/definition collisions, and attributed kernels. Manual
  IDs must resolve to exactly one real source definition. The reconciled 147
  entries remain unchecked for human review.
- Moved device, final-test, and inference record formatting into the production
  reporting module. Host C++ tests now exercise all five documented record
  types for exact field order, precision, and newline termination; Python keeps
  the README grammar checks synchronized with that production suite.
- Narrowed README error-checking claims to operational CUDA calls and explicitly
  documented best-effort nonthrowing cleanup for `cudaFree`,
  `cudaStreamDestroy`, and `cudaEventDestroy`.

### RED Evidence

The focused tests failed before each implementation slice:

```text
tests.test_check_prohibited: 45 review-case failures with MSYS Bash
tests.test_check_comments: overload, namespace/nested type, attributed kernel,
  and manual stale-ID cases failed
tests.test_documentation: 3 failures for log lifetime, evidence semantics, and
  cleanup wording
build/reporting_tests: PrintDeviceSummary, PrintFinalTestSummary, and
  PrintInferenceSummary were undeclared
```

Additional RED cases confirmed that a link-only log and a link command without
the architecture flags were incorrectly accepted before the final build-log
validation tightening.

### GREEN Evidence

- `make host-tests`: all nine host suites passed, including the production
  reporting suite.
- `make python-tests`: 15 data/converter tests passed.
- `make compliance`: 52 compliance/documentation/output tests passed; the
  147-item comment inventory and Make-scoped source policy gate also passed.
- `tests.test_check_prohibited`: 17 focused scanner tests passed.
- `make -B -n all`: emitted compile/link commands, both exact `sm_90`/PTX
  targets, and the completion-marker command without executing CUDA tools.
- `git diff --check`: passed before this report append and is rerun before the
  fix-round commit.

### Remaining Target-Only Work

Python 3.6 runtime/package execution, CUDA compilation, CUDA numerical and
workflow execution, Compute Sanitizer, `cuobjdump`, dynamic dependency
inspection, default training, timing, and the >=99% H20 accuracy gate remain
unavailable locally and are not claimed. The semantic checklist remains
deliberately unchecked.

## Fix Round 2

### Review Issues Addressed

- Replaced Make-database token scraping and unchecked process substitution with
  checked `make -Bn all` recipe analysis. The source gate now extracts every
  tests-owned translation unit from actual compiler commands, requires a
  resolvable output for each, recursively resolves project-local includes, and
  fails closed when Make execution, extraction, file reads, or required quoted
  include resolution fails. Active `tests/test_harness.h` and
  `tests/cpu_reference.h` are covered without generated dependency files;
  dormant fixtures outside the build graph remain excluded.
- Added transitive source-to-output provenance for tests-owned compilation
  recipes and reject any such source or output in the `lenet_cuda` link. This
  catches a CPU reference renamed to an unrelated object such as `oracle.o`,
  including another compiler/link output layer, while preserving the
  production-source CPU fallback identifier scan.
- Tightened actual build-log acceptance so common compiler and Make failure
  records reject the log wherever they occur, and the exact success marker must
  be the final nonempty line. Architecture, compile/link, freshness, prohibited
  dependency, and explicit dry-run non-acceptance checks remain in force.
- Updated the H20 guide with the final-line, failure-record, active-header, and
  production-link provenance rules.

### TDD Evidence

After adding the core regressions and before changing the checker, the focused
suite reported six expected failures:

```text
Ran 23 tests in 23.071s
FAILED (failures=6)
```

The failures covered a clean-tree transitive active-header violation, malformed
Make scope extraction, a renamed test-owned production object, a marker followed
by another command, a marker followed by `make ... Error 2`, and an earlier
compiler fatal error. Dormant-header exclusion, unresolved quoted includes, and
truncated-marker behavior are also retained as focused regressions. A separate
RED/GREEN regression proves provenance survives an additional renamed
relocatable-link output.

### GREEN Evidence

- `tests.test_check_prohibited`: 25 focused scanner tests passed.
- Seven-module aggregate Python run: 75 tests passed.
- `make host-tests makefile-tests python-tests compliance`: all nine host suites,
  the distinct host/CUDA basename regression, 15 data tests, 60 compliance
  tests, the 147-item comment inventory, and the source policy gate passed.
- `bash -n scripts/check_prohibited.sh` passed, and the real clean-tree source
  scan reported `scope=make-compiled-inputs`.
- Python byte compilation and Python 3.6 grammar parsing passed for the changed
  checker test module.
- Forced `all` and `cuda-tests` Make dry-runs exposed the expected commands and
  both exact `sm_90` and `compute_90` code-generation targets without executing
  CUDA tools.
- `git diff --check` is rerun immediately before the fix-round commit.

### Remaining Target-Only Work

Python 3.6 runtime/package execution, CUDA compilation, CUDA numerical and
workflow execution, Compute Sanitizer, `cuobjdump`, dynamic dependency
inspection, default training, timing, and the >=99% H20 accuracy gate remain
unavailable locally and are not claimed. The semantic checklist remains
deliberately unchecked.

## Compliance Hardening Subtask 13.1

Stabilized the recipe-analyzer foundation around Python 3.6-compatible
`shlex.split(..., posix=True)`, translation-unit suffix detection independent
of compiler/wrapper names, exactly one supported output for source-bearing
commands, and exact normalized reconciliation between Make's test-source
manifest and recipe translation units. The shell entry point preserves the
analyzer diagnostic and returns failure when analysis fails.

The three inherited parser/manifest tests were already green. A focused direct
analyzer regression for an unknown wrapper plus quoted source/output paths
failed under an intentional naive-whitespace-tokenization mutation with the
expected missing-recipe manifest error. The tightened wrapper regression failed
under intentional removal of the source-output requirement because the source
scan incorrectly returned success.

Focused final verification:

```text
python -m unittest -v \
  ...test_clang_ccache_quoted_paths_and_iquote_order_are_scanned \
  ...test_attached_include_flags_resolve_active_header \
  ...test_make_manifest_and_recipe_sources_must_match \
  ...test_analyzer_accepts_unknown_wrapper_with_quoted_source_and_output \
  ...test_source_scan_propagates_source_bearing_recipe_without_output
Ran 5 tests in 2.012s
OK

python -m py_compile scripts/analyze_build_graph.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed: scripts/analyze_build_graph.py

bash -n scripts/check_prohibited.sh
exit 0
```

Inherited active-header, artifact-provenance, build-log, and documentation work
is retained in this stabilization commit but remains assigned to independent
audit/refinement in Subtasks 13.2 through 13.5. Python 3.6 runtime execution and
all CUDA/H20 evidence remain unavailable locally and are not claimed.

## Compliance Hardening Subtask 13.1 Fix Round 1

### Review Issues Addressed

- Restricted positional archive-output inference to unambiguous `ar`,
  `llvm-ar`, or `gcc-ar` commands, including executable paths and `.exe`
  suffixes, with a recognized replace/quick-append operation, archive output
  position, and following artifact member. Every source-bearing recipe still
  requires an explicit `-o`, even when archive inputs precede other artifacts.
- Replaced convenience splitting with Python 3.6-compatible punctuation-aware
  `shlex.shlex` tokenization. Standalone and adjacent command separators,
  pipelines, redirections, command substitutions, and backticks are detected;
  quoted or escaped literal punctuation in normal compiler options and paths is
  not rejected.
- Reject response-file tokens and unsupported shell controls immediately when
  a recipe contains a normalized tests-owned translation unit, before test
  manifest reconciliation. Production-graph validation remains unchanged.
- Added direct analyzer regressions for extra recipe test sources absent from
  the manifest and checker regressions proving malformed and explicit nonzero
  Make invocations propagate failure.

### TDD Evidence

The five initial focused regressions ran before the analyzer change. Archive
ambiguity, response-file ordering, and all seven shell-control subcases failed
for the expected reasons; the existing manifest-extra and Make-failure behavior
was already correct:

```text
Ran 5 tests in 1.326s
FAILED (failures=9)
```

After the minimal analyzer change, those five tests passed. Positive focused
coverage also preserves quoted/escaped compiler punctuation and a path-qualified
`llvm-ar rcs` positional archive recipe.

### Focused Verification

```text
python -m unittest -v <13 focused Subtask 13.1 parser/checker tests>
Ran 13 tests
OK

python -m py_compile scripts/analyze_build_graph.py tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

bash -n scripts/check_prohibited.sh
exit 0

git diff --check
exit 0
```

No Subtask 13.2-or-later behavior was intentionally changed. Python 3.6 runtime
execution and all CUDA/H20 evidence remain unavailable locally and are not
claimed.
