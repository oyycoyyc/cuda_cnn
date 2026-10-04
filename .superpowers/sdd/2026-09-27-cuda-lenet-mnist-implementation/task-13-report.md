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

## Compliance Hardening Subtask 13.2

Audited the inherited active-header closure and retained its per-translation-
unit search model: the including file directory precedes all `-iquote`
directories in command order, which precede all `-I` directories in command
order. Quoted includes fail closed when unresolved, reachable project-local
angle includes are scanned, each selected header is recursively visited by its
canonical real path, escapes are rejected, and dormant duplicate fixtures are
not added to the active set.

The audit found that separated include-option operands were also being parsed
as possible translation units. A valid quoted include directory ending in
`.c` or `.cpp` therefore failed before include resolution. Include parsing now
records attached and separated option-token positions so those tokens cannot be
classified as source or artifact inputs. This is limited to active-input
classification and does not change artifact provenance or build-log behavior.

### TDD Evidence

The new complete-order regression failed before the analyzer change:

```text
test_duplicate_headers_follow_complete_quote_search_order ... FAIL
build graph analysis failed: source file does not exist: tests/include first.c
Ran 1 test
FAILED (failures=1)
```

The regression uses duplicate names at the including directory, two quoted
include directories, and two normal include directories. It mixes attached and
separated options with quoted paths, and includes a project-local angle header.
Its hand-written expected active-file set contains only each winning candidate.
After GREEN, an intentional `-I`-before-`-iquote` mutation made the same test
fail by selecting both wrong normal-include duplicates; the correct order was
then restored.

### Focused Verification

```text
python -m unittest -v \
  ...test_active_test_transitive_headers_are_scanned_without_dependency_files \
  ...test_active_test_unresolved_quoted_include_fails_closed \
  ...test_active_header_symlink_cannot_escape_repository \
  ...test_dormant_negative_fixture_is_ignored_but_compiled_test_is_scanned \
  ...test_duplicate_headers_follow_complete_quote_search_order
Ran 5 tests in 2.609s
OK

python -m py_compile \
  scripts/analyze_build_graph.py tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

The symlink-escape regression now uses a directory junction fallback when
unprivileged Windows cannot create a symbolic link, so all five named tests ran
without skips. Python 3.6 runtime execution and all CUDA/H20 evidence remain
unavailable locally and are not claimed.

## Compliance Hardening Subtask 13.2 Fix Round 1

### Review Issues Addressed

- Preserved each selected include's lexical path through recursive traversal so
  a symlinked header resolves its nested quoted includes relative to the path
  seen by the compiler, not the target file's canonical directory.
- Kept full-file canonical paths for source-root containment, file identity,
  reads, and emitted active-input paths. The visited key now includes the
  lexical route and compile search context, so a second route to the same real
  header is not suppressed when it has different quoted-include semantics.
- Strengthened the angle-include ordering regression with same-name decoys in
  the source and both `-iquote` directories plus candidates in both `-I`
  directories. Only the first `-I` candidate is active.
- Retained the external symlink/junction escape rejection regression unchanged.

### TDD Evidence

The real file-symlink regression was added before the production change, but
this Windows host denied file-symlink creation with `WinError 1314`. A
deterministic companion models two lexical routes with one real file identity;
restoring either pre-fix behavior produced the expected RED:

```text
canonical-path visited key: FAILED (alias/sibling.h not found in active set)
dirname(real_path) quoted search: FAILED (alias/sibling.h not found in active set)
```

The angle-order fixture also failed under both intentional mutations:

```text
angle searched as quote: selected tests/source dir/project_angle.h
reversed -I order: selected tests/include second/project_angle.h
```

### Focused Verification

```text
python -m unittest -v <7 focused Subtask 13.2 active-header tests>
Ran 7 tests in 2.647s
OK (skipped=1: unprivileged Windows file-symlink creation)

python -m py_compile \
  scripts/analyze_build_graph.py tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

The deterministic same-real-file regression and the junction-backed external
escape regression both ran locally. Python 3.6 runtime execution and all
CUDA/H20 evidence remain unavailable locally and are not claimed. Subtask 13.3
and later behavior was not intentionally changed.

## Compliance Hardening Subtask 13.2 Fix Round 2

### Cycle Issue Addressed

- Changed recursive active-header visit identity from the unbounded lexical file
  path to the canonical real file, canonical realpath of its lexical containing
  directory, and the translation unit's quote/include search context.
- Kept the lexical containing directory for actual quoted-include candidate
  lookup. File symlinks located in distinct real directories therefore retain
  distinct compiler lookup semantics, while directory-symlink aliases of the
  same search directory collapse and terminate.
- Kept canonical full-file paths for source-root checks, reads, and emitted
  active-file output.

### TDD Evidence

The deterministic key regression was added before the production helper and
failed with the expected missing-identity assertion:

```text
test_active_visit_key_collapses_only_equivalent_lexical_directories ... FAIL
AssertionError: analyzer must expose active_visit_key for deterministic identity tests
```

The in-root `loop -> root` recursive-include integration regression was also
added first. This Windows host denied directory-symlink creation with
`WinError 1314`, so it skipped as permitted; the deterministic regression
models the same canonical-directory collapse without symlink privilege and also
proves that distinct file-symlink directories do not collapse.

### Focused Verification

```text
python -m unittest -v <9 focused Subtask 13.2 cycle/symlink/order tests>
Ran 9 tests in 2.952s
OK (skipped=2: Windows denied file and directory symlink creation)

python -m py_compile scripts/analyze_build_graph.py tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

Python 3.6 runtime execution and all CUDA/H20 evidence remain unavailable
locally and are not claimed. Subtask 13.3 and later behavior was not changed.

## Compliance Hardening Subtask 13.2 Fix Round 3

### Context Collision Addressed

- Restored the full normalized lexical file path plus the translation unit's
  quote/include directories as the global visited context. Distinct lexical
  routes to one real header therefore retain different parent-relative include
  semantics even when both lexical directories canonicalize to one directory.
- Added a canonical-real-file set scoped to the current DFS chain. Re-entering
  the same real file stops only that recursive branch, and `finally` removal
  permits later independent lexical routes to scan normally.
- Retained canonical active-output deduplication and canonical source-root/file
  checks for every selected candidate before recursive traversal.
- Documented the safety boundary in the analyzer: an actually unguarded
  recursive include cannot compile, so terminating a repeated canonical file in
  the current chain cannot conceal executable compiled content.

### TDD Evidence

Both deterministic regressions failed before the production change:

```text
test_parent_relative_include_scans_each_same_real_header_route ... FAIL
AssertionError: AnalysisError not raised

test_modeled_directory_alias_recursion_terminates_at_canonical_file ... FAIL
AssertionError: canonical recursion did not terminate: cannot resolve quoted include ...

Ran 2 tests
FAILED (failures=2)
```

The first regression models two non-recursive lexical routes whose header and
containing directories share canonical identities. The shared header includes
`../target.h`; the first route is clean and the second resolves through a
modeled source-root escape. The second regression models directory-alias
recursion without requiring Windows symlink privilege.

### Focused Verification

```text
python -m unittest -v <10 focused Subtask 13.2 active-header tests>
Ran 10 tests in 3.260s
OK (skipped=2: Windows denied file and directory symlink creation)

python -m py_compile scripts/analyze_build_graph.py tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

The retained real directory-symlink recursion regression remains in the focused
set; its deterministic modeled companion ran locally. Python 3.6 runtime
execution and all CUDA/H20 evidence remain unavailable locally and are not
claimed. Subtask 13.3 and later behavior was not changed.

## Compliance Hardening Subtask 13.3

Audited and hardened the inherited artifact graph around exactly one normalized
`build/lenet_cuda` or `build/lenet_cuda.exe` output. Positional artifact tokens
now match analyzed producers even when intermediate names have no conventional
suffix, so provenance survives arbitrary compiler/linker executable names and
multiple renamed layers. Reachable project artifact paths that escape the
source root fail closed instead of disappearing from analysis.

Common creating forms for `ar`, path-qualified `llvm-ar[.exe]`, and `gcc-ar`
are modeled, including leading-dash operations and a valued `--plugin` option.
Archive option operands are excluded from artifact inputs, while an archive
whose reachable members come from a response file remains a known producer and
is rejected specifically for the unsupported response file. Existing rejection
of reachable shell controls, duplicate producers, missing producers, and
tests-owned sources is retained. A tests-owned object linked only into
`build/workflow_tests` remains accepted, and include-directory/output operands
are not classified as artifact inputs.

### TDD Evidence

The first eight new focused behaviors were added before the production change.
The initial seven-test RED run reported three expected failures:

```text
test_unknown_tools_trace_extensionless_multilayer_provenance ... FAIL
test_archive_tool_common_forms_trace_provenance
  (gcc-ar ... --plugin ...) ... FAIL
test_production_output_must_be_exactly_under_build ... FAIL
Ran 7 tests
FAILED (failures=3)
```

The escaping-artifact regression separately failed because `../hidden.o` was
silently ignored. During GREEN audit, the archive response-file regression also
failed first with a missing-producer diagnostic rather than response-file
rejection. Both passed after the corresponding minimal analyzer changes.

### Focused Verification

```text
python -m unittest -v <18 focused Subtask 13.3 provenance tests>
Ran 18 tests in 4.614s
OK

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

No active-header, scanner-wrapper, build-log, or public documentation behavior
was changed, and no broad, CUDA, or H20 command was run. Python 3.6 runtime
execution and all CUDA/H20 evidence remain unavailable locally and are not
claimed. Subtasks 13.4 and later remain unchanged.

## Compliance Hardening Subtask 13.3a

### Review Issues Addressed

- Added explicit per-command metadata for response/options files, encoded
  linker inputs, and unsupported shell syntax. Metadata is rejected only for a
  command in the `build/lenet_cuda` producer closure or a recipe containing a
  normalized tests-owned source; unrelated test-only link commands retain the
  existing reachability isolation.
- Rejected `-Wl,` and direct `-Xlinker`/`--linker-options` forwarding, plus
  paired separated or attached `-L` and `-l`/`-l:` forms, rather than trying to
  infer hidden artifact provenance.
- Expanded response-file recognition to bare `@file`, `-Wl,@file`, NVCC
  `--options-file` separated/equal forms, and `-optf` separated/equal/attached
  forms. Encoded option operands are excluded from ordinary source/output/input
  classification, preventing `-optf` from being mistaken for attached `-o`.
- Expanded quote-aware shell rejection to remaining `$` expansions and
  unquoted `*`, `?`, and bracket globs. Single-quoted and backslash-escaped
  literal metacharacters remain accepted, as do the project's current ordinary
  NVCC and exact gencode flags.

### TDD Evidence

The first RED run covered the hidden `-Wl,` archive, paired `-L`/`-l` forms,
all response variants, variable/glob expansion, and the positive literal case.
It reported 25 expected failing subcases while the positive case passed. Two
additional RED runs reported four direct-linker-forwarding failures and one
tests-owned linker-input failure. Failures showed the encoded inputs were
accepted, glob paths reached the wrong missing-producer diagnostic, and attached
`-optf` forms were mistaken for extra output options.

### Focused Verification

```text
$env:BASH = 'C:\personal_apps\msys64\usr\bin\bash.exe'
$env:MAKE = 'C:\personal_apps\msys64\usr\bin\make.exe'
python -m unittest -v <15 focused Subtask 13.3a analyzer/checker tests>
Ran 15 tests in 4.721s
OK

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

No extensionless-token handling, general option-operand taxonomy, canonical
tests ownership, archive grammar, broad suite, CUDA, or H20 work is included.

## Compliance Hardening Subtask 13.3a Fix Round 1

### Review Issues Addressed

- Inspected separated and equals-form NVCC `-Xcompiler`/
  `--compiler-options` and `-Xlinker`/`--linker-options` values for response
  files before making their option/value indexes opaque to normal input
  discovery. Any `@` in a forwarded scalar or comma-list value is rejected for
  production-reachable and tests-owned source commands. Existing direct-linker
  rejection remains in force when no response file is present.
- Corrected bracket-glob recognition for expressions whose optional negation is
  followed by a literal leading `]`. A real checker regression proves
  `tests/[]]hidden.h` cannot activate a dormant prohibited header without being
  rejected, while an unclosed `[` remains a literal.
- Added one quote/escape-aware shell command-prefix pass before both recipe
  tokenization and ambiguity scanning. A `#` starts a comment only at an
  unquoted, unescaped shell word boundary; quoted, escaped, and in-word hashes
  remain part of the command.

### TDD Evidence

Before the analyzer change, the six new test methods reported 16 expected
failures. Fourteen production/tests-owned forwarding subcases were accepted or
reported only as generic linker forwarding, `[]]hidden.h` passed the complete
source checker, and response/glob text in a genuine trailing comment caused
rejection. The unmatched-bracket and quoted/escaped/in-word hash boundary cases
were already green and protect against overcorrection.

### Focused Verification

```text
$env:BASH = 'C:\personal_apps\msys64\usr\bin\bash.exe'
$env:MAKE = 'C:\personal_apps\msys64\usr\bin\make.exe'
python -m unittest -v <21 focused Subtask 13.3a analyzer/checker tests>
Ran 21 tests in 6.809s
OK

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

No broad, CUDA, or H20 command was run, and 13.3b/c scope remains unchanged.

## Compliance Hardening Subtask 13.3a Fix Round 2

### Final Bracket-Glob Issue Addressed

- Replaced first-terminator bracket detection with a deterministic scanner that
  tracks shell single quotes, double quotes, and escapes inside a bracket word.
  A quoted or escaped `]` remains a member, while a later unquoted `]` closes
  the expression. Existing optional negation and initial literal `]` handling
  is retained.
- Added a bounded POSIX bracket-subexpression scanner for character classes,
  collating symbols, and equivalence classes. Complete inner `:]`, `.]`, and
  `=]` terminators are skipped so only an outer close activates the glob; a
  completed inner subexpression without an outer close remains non-expanding.
- Prevented the top-level shell scan from revisiting the inner opener of a
  malformed outer POSIX bracket candidate as an independent glob. Scanning
  resumes after the inner subexpression so later unrelated expansions are
  still detected.

### TDD Evidence

The four new test methods initially reported seven failures: double- and
single-quoted `]` members were rejected as non-globs, each inner POSIX
subexpression terminator was mistaken for the outer close in malformed input,
the partial-quote analyzer case passed open, and the combined malformed/literal
case failed closed incorrectly. The first implementation left one focused
failure because the top-level scan revisited the inner POSIX opener; the caller
skip was then added and the same four tests passed. A checker regression also
proves `[[:alpha:]]hidden.h` cannot activate a matching dormant prohibited
header without shell-ambiguity rejection.

### Focused Verification

```text
$env:BASH = 'C:\personal_apps\msys64\usr\bin\bash.exe'
$env:MAKE = 'C:\personal_apps\msys64\usr\bin\make.exe'
python -m unittest -v <25 focused Subtask 13.3a analyzer/checker tests>
Ran 25 tests in 7.454s
OK

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

No broad, CUDA, or H20 command was run, and 13.3b/c remains untouched.

## Compliance Hardening Subtask 13.3a Fix Round 3

### Bash Fallback Semantics Corrected

The Fix Round 2 statement that a completed POSIX-looking inner subexpression
without a later outer close is non-expanding was incorrect. Bash can fall back
to interpreting that inner `]` as the close of an ordinary bracket expression:
for example, `[[:alpha:]` can match `[a`.

- The bracket scanner now records each complete POSIX-looking inner terminator
  as a usable fallback close while continuing to search for a later unquoted
  outer close. Either route makes the bracket expression active.
- Removed the caller logic that skipped the same completed inner expression
  after the outer scan returned false. Truly unmatched forms with no usable
  unquoted `]` remain non-active, and partial quoting, initial literal `]`,
  negation, and escapes retain their prior behavior.
- Replaced the three incorrect helper expectations and removed the fallback
  form from the literal-positive analyzer case. Checker-level regressions now
  pair `[[:alpha:]`, `[[.t.]`, and `[[=a=]` recipes with matching dormant
  prohibited files `[aclass.h`, `[tcollating.h`, and `[aequiv.h` respectively.

### TDD Evidence

Before the analyzer correction, six subcases failed: the three helper cases
returned non-active and all three matching dormant-file checker cases passed
open. The complete POSIX-class plus outer-close checker case and quoted/escaped
literal case remained green.

### Focused Verification

```text
$env:BASH = 'C:\personal_apps\msys64\usr\bin\bash.exe'
$env:MAKE = 'C:\personal_apps\msys64\usr\bin\make.exe'
python -m unittest -v <26 focused Subtask 13.3a analyzer/checker tests>
Ran 26 tests in 9.071s
OK

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

No broad, CUDA, or H20 command was run, and 13.3b/c remains untouched.

## Compliance Hardening Subtask 13.3b

### Extensionless And Positional Inputs Hardened

- Production-reachable positional inputs now participate in the artifact graph
  regardless of filename suffix. Missing extensionless producers fail closed,
  while unrelated test-only commands remain outside the production closure.
- Both separated `-o PATH` and attached `-oPATH` outputs support extensionless
  producers. Distinct `-opt-info`, `-openmp`, and `-objc` option families are
  not interpreted as attached outputs.
- Command-prefix classification is independent of compiler basename. Arbitrary
  wrapper/tool words preceding source compile syntax are excluded, and
  `ccache`, `sccache`, and `distcc` are recognized explicitly for non-source
  commands. Linker positional token 1 remains an input, including on a command
  that also carries a source token.
- Separated operands for `-MF`, `-MT`, `-MQ`, `-isystem`, `-include`,
  `-imacros`, and `--sysroot` are excluded from provenance. Attached short
  forms and equals forms are covered without hiding source, output, forwarding,
  or ordinary positional inputs.
- Positional paths that cannot be normalized inside the source root are retained
  as command-local analysis errors and rejected only when their command is in
  the `lenet_cuda` closure.

### TDD Evidence

The inherited 13.3b WIP and seven initial focused tests were present in the
worktree. Audit added a source-bearing linker regression before changing the
prefix classifier. It failed because positional token 1 was silently discarded:

```text
test_source_bearing_link_preserves_first_positional_input ... FAIL
AssertionError: 0 == 0 : tests/cpu_reference.cpp
Ran 1 test
FAILED (failures=1)
```

Prefix inference was then limited to arbitrary pre-`-c` compile prefixes;
source-bearing link commands use the same explicit common-wrapper/executable
classification as other link commands. The wrapper-link test was strengthened
to require traversal into a tests-owned extensionless producer rather than only
asserting command acceptance.

### Focused Verification

```text
python -m unittest -v <9 focused Subtask 13.3b analyzer tests>
Ran 9 tests in 2.002s
OK

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

No canonical source-ownership or archive-grammar behavior was changed. No broad,
CUDA, or H20 command was run. Python 3.6 runtime execution and all CUDA/H20
evidence remain unavailable locally and are not claimed; Subtask 13.3c and later
remain unchanged.

## Compliance Hardening Subtask 13.3b Fix Round 1

### Prefix Classification Blocker Addressed

The prior 13.3b statement that arbitrary wrapper/tool words preceding source
compile syntax are excluded is superseded. A command prefix is now exactly zero
or more explicitly supported `ccache`, `sccache`, or `distcc` wrappers followed
by one executable/tool token. No source or `-c` position is used to infer
additional prefix words.

As a result, every later non-option positional token remains a candidate input,
including tokens before a genuine `-c`. Option and forwarding operands retain
their existing opaque indexes and cannot become syntax or artifact inputs.
Unknown wrapper chains fail closed when production-reachable while unrelated
commands retain production-closure isolation.

### TDD Evidence

Three regressions were added before the production change. The composed
`oracle_blob -MT -c src/model.cu` case and a genuine compile with a stray object
before `-c` both incorrectly returned success; the supported-wrapper positive
case already passed:

```text
test_opaque_c_operand_cannot_hide_pre_source_positional_input ... FAIL
test_compile_preserves_stray_positional_input_before_real_c ... FAIL
test_common_wrapper_and_tool_prefixes_are_excluded ... ok
Ran 3 tests
FAILED (failures=2)
```

Removing source/`-c` prefix inference made all three pass. The composed case now
traces `oracle_blob` to its tests-owned producer, and the stray object reaches
the existing missing-producer rejection.

### Focused Verification

```text
python -m unittest -v <12 focused Subtask 13.3b analyzer tests>
Ran 12 tests in 2.495s
OK

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

No 13.3c, broad-suite, CUDA, or H20 command was run. Python 3.6 runtime and
CUDA/H20 evidence remain unavailable locally and are not claimed.

## Compliance Hardening Subtask 13.3c

### Canonical Ownership And Archive Grammar

Source ownership now compares each source's canonical real path with canonical
`ROOT/tests` using platform case normalization. Manifest reconciliation uses the
same canonical source identity, while diagnostics retain the lexical recipe
path. A production lexical alias into `tests/` therefore remains tests-owned and
is rejected when reachable even when the normal test path independently
satisfies the manifest.

Recognized `ar`, `llvm-ar`, and `gcc-ar` path/`.exe` commands now parse supported
options before or after the operation, including separated and attached
`--plugin`. The parser consumes `a`/`b`/`i` relative-position operands and the
positive count required by `N`, keeps those grammar operands out of provenance,
and accepts archive creation with no members. Conflicting producer operations,
ambiguous position modifiers, missing/invalid counts, unsupported operations,
and misplaced post-archive options fail closed.

### TDD Evidence

Five focused tests were added before production changes. The deterministic
ownership helper was absent; all four valid member-bearing archive forms failed
to reach the tests-owned producer; empty archive creation had no producer; and
all three malformed forms were incorrectly accepted:

```text
test_source_ownership_uses_canonical_test_tree_metadata ... ERROR
test_archive_preoperation_options_and_position_operands_trace_provenance ...
  four subtests FAIL
test_empty_archive_creation_is_a_producer_without_members ... FAIL
test_malformed_or_ambiguous_archive_producers_are_rejected ...
  three subtests FAIL
Ran 5 tests
FAILED (failures=8, errors=1, skipped=1)
```

The POSIX source-symlink integration test was skipped on Windows by design; its
cross-platform metadata companion exercised canonical ownership and Windows
case normalization. Symlink creation errors on POSIX are skipped only for
access or platform-capability errors.

### Focused Verification

```text
$env:BASH = 'C:\personal_apps\msys64\usr\bin\bash.exe'
python -m unittest -v <16 focused Subtask 13.3c tests>
Ran 16 tests in 5.381s
OK (skipped=1: POSIX-only source-symlink integration on Windows)

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

An initial expanded focused run with `BASH` unset encountered the existing
Windows subprocess GBK decode issue in six checker-backed cases. Repeating the
identical tests with the recorded MSYS2 Bash path produced the result above.

No build-log or documentation behavior was modified. No broad, CUDA, or H20
command was run. The POSIX end-to-end symlink case and Python 3.6 runtime
execution remain unavailable locally and are not claimed.

## Compliance Hardening Subtask 13.3c Fix Round 1

### Review Findings Addressed

- Added one canonical path identity function defined as absolute real path plus
  platform case normalization. Tests ownership, manifest entries, and recipe
  sources now share that identity. Reconciliation dictionaries use it as their
  key and retain normalized lexical-relative paths only as diagnostic values.
- Case-equivalent manifest/recipe spellings now reconcile on case-insensitive
  platforms, canonical duplicate manifest entries reject, and distinct real
  files retain distinct keys. A deterministic Windows regression exercises
  recipe parsing, manifest reading, reconciliation, and production graph
  rejection with a lexical `src/` alias resolving into `tests/` under varied
  case spellings.
- Replaced the shared archive modifier set with operation-specific `q` and `r`
  sets. Placement modifiers and `u` are accepted only with `r`; `qu` rejects;
  and `N` rejects for this supported producer subset rather than consuming a
  count. Valid plugin options, `a`/`b`/`i` placement, `ru`, empty creation, and
  member provenance remain covered.

### TDD Evidence

The first focused RED run showed the case-variant alias failing during
reconciliation and both newly unsupported producer forms being accepted:

```text
test_windows_case_variant_alias_reconciles_then_taints_production ... ERROR
test_windows_case_variant_manifest_alias_is_duplicate ... ok
test_archive_preoperation_options_and_position_operands_trace_provenance ... ok
test_malformed_or_ambiguous_archive_producers_are_rejected ...
  ar rN 2 ... FAIL
  ar qu ... FAIL
Ran 4 tests
FAILED (failures=2, errors=1)
```

The duplicate fixture was then strengthened to preserve different diagnostic
case spellings for one canonical file. Before production changes it failed with
`AnalysisError not raised`, proving the shared identity was required rather
than relying on rendered path behavior.

### Focused Verification

```text
$env:BASH = 'C:\personal_apps\msys64\usr\bin\bash.exe'
python -m unittest -v <17 focused Subtask 13.3c tests>
Ran 17 tests in 5.583s
OK (skipped=1: POSIX-only source-symlink integration on Windows)

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

The first byte-compilation attempt ran concurrently with unittest and hit a
Windows `__pycache__` replacement access error. Its standalone rerun produced
the successful result above.

No build-log or documentation behavior was modified. No broad, CUDA, H20,
subagent, or reviewer command was run. The POSIX end-to-end symlink case and
Python 3.6 runtime execution remain unavailable locally and are not claimed.

## Compliance Hardening Subtask 13.3d

### Active Compiler Input Provenance

- Production artifact traversal remains rooted at exactly one
  `build/lenet_cuda[.exe]` producer and now returns the commands reached after
  all existing provenance checks. Every source-bearing command in that closure
  contributes its lexical source and compiler-search context to active-input
  discovery; manifest-reconciled test translation units retain their own
  contexts.
- Active quoted and angle includes now retain production provenance. Any
  selected input whose canonical real path is under canonical `ROOT/tests` is
  rejected for production, including file aliases and platform case variants.
  An unused `-Itests` remains accepted.
- Separated and practical attached `-isystem`, `-include`, and `-imacros`
  forms are compiler inputs rather than artifact edges. System directories are
  searched after `-I` for angle includes and after the including directory,
  `-iquote`, and `-I` for quoted includes. Forced inputs resolve first from the
  root working directory and then through quote, ordinary, and system include
  directories; unresolved, escaping, or non-file selections fail closed.
- Forced inputs are recursively scanned for both test and production commands.
  Analyzer output contains the canonical-deduplicated union of active test and
  production inputs consumed by the Bash scanner.
- The inherited WIP used one global lexical/search-context visit key for test
  and production traversals. A test traversal could therefore visit a shared
  header first and mask a later production route from that header into
  `tests/`. The visit identity now includes provenance mode, while emitted
  inputs remain canonical-deduplicated.

### TDD Evidence

The inherited Task 13.3d WIP and its focused tests were preserved and audited.
The audit added a shared-header regression before changing the implementation;
its RED run showed the production route being accepted and emitted instead of
rejected:

```text
test_test_context_cannot_mask_production_route_to_tests_header ... FAIL
AssertionError: 0 == 0 : include/shared.h
src/model.cu
tests/production_input.h
tests/smoke_tests.cpp
Ran 1 test
FAILED (failures=1)
```

Adding production/test mode to the active visit key made the same isolated test
pass before the complete focused run.

### Focused Verification

```text
$env:BASH = 'C:\personal_apps\msys64\usr\bin\bash.exe'
python -m unittest -v <20 focused Subtask 13.3d tests>
Ran 20 tests in 4.483s
OK (skipped=1: POSIX-only header-symlink integration on Windows)

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

The first checker-backed focused run used the default Windows Bash lookup and
encountered the existing subprocess GBK decoding issue. The recorded MSYS2
Bash run is the authoritative focused result. No linker-script, library,
archive, general-option, build-log, documentation, broad-suite, CUDA, H20,
subagent, or reviewer work was performed. Python 3.6 runtime execution and the
POSIX symlink integration remain unavailable locally and are not claimed.

## Compliance Hardening Subtask 13.3d Fix Round 1

### Review Findings Addressed

- GCC cross-class duplicate semantics now compare canonical real-path plus
  platform-normalized directory identities. Any ordinary `-I` occurrence also
  present in `-isystem` is removed from ordinary lookup and remains at its
  system position; other lexical directories retain their order. A
  deterministic modeled alias test covers the canonical comparison.
- NVCC `-Xcompiler` and `--compiler-options` payloads now contribute active
  compiler semantics in separated and `=` forms. Comma-separated forwarded
  `-I`, `-isystem`, `-include`, and `-imacros` options are expanded in command
  order, while original forwarding tokens remain opaque to artifact parsing.
  Missing outer payloads or active option values fail closed; unrelated host
  flags and prior response-file rejection remain intact.
- Exact manifest reconciliation still accounts only for tests-owned
  translation units. After it succeeds, every source translation unit in a
  command containing a reconciled test TU receives that command's active
  compiler context. Helpers outside `include/`, `src/`, and `tests/` and their
  local headers therefore reach the Bash scanner without becoming manifest
  entries. Production graph selection is unchanged.

### TDD Evidence

Each finding received a focused RED run before its implementation change:

```text
test_system_duplicate_removes_ordinary_include_precedence ... FAIL
test_system_duplicate_matches_canonical_directory_alias ... FAIL
Ran 2 tests
FAILED (failures=2)

test_nvcc_forwarded_active_inputs_are_enforced_for_production ... 5 FAIL
test_nvcc_forwarded_test_forced_header_is_scanned ... FAIL
test_malformed_nvcc_active_forwarding_fails_closed ... 3 FAIL
test_ordinary_nvcc_host_forwarding_values_remain_accepted ... ok
Ran 4 tests
FAILED (failures=9)

test_test_compile_scans_nonmanifest_helper_and_local_header ... FAIL
test_clean_nonmanifest_helper_context_is_emitted ... FAIL
Ran 2 tests
FAILED (failures=2)
```

The isolated green runs then passed three duplicate-order tests, five NVCC
forwarding/response tests, and four helper/manifest tests respectively.

### Focused Verification

```text
$env:BASH = 'C:\personal_apps\msys64\usr\bin\bash.exe'
python -m unittest -v <31 focused Subtask 13.3d tests>
Ran 31 tests in 7.900s
OK (skipped=1: POSIX-only header-symlink integration on Windows)

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

No linker-script, library, archive, general-option, build-log, documentation,
broad-suite, CUDA, H20, subagent, or reviewer work was performed. Python 3.6
runtime execution and the POSIX symlink integration remain unavailable locally
and are not claimed.

## Compliance Hardening Subtask 13.3d Fix Round 2

### Final Empty-Value Finding

- Include-directory values are now checked before any path join or
  normalization, so empty separated `-I`, `-iquote`, and `-isystem` operands
  cannot become `ROOT`. Exactly empty attached `-I=` and `-iquote=` spellings
  reject without changing existing nonempty attached path handling.
- Every comma-delimited NVCC host compiler forwarding field must be nonempty.
  This rejects empty values for forwarded `-I`, `-iquote`, `-isystem`,
  `-include`, and `-imacros`, and fails closed on unrelated leading, middle, or
  trailing empty fields rather than silently changing active search semantics.
- Existing forced-input checks reject empty separated, equals, and missing
  attached values for both `-include` and `-imacros`. Valid forwarding and
  forwarded response-file rejection remain unchanged.

### TDD Evidence

The four focused test matrices were added before the implementation change.
The RED run reported 12 failures: three forwarded include-directory families,
five direct include-directory spellings, and four unrelated empty forwarding
field cases were accepted. The forced-input empty-form matrix already passed,
confirming that its existing fail-closed behavior only needed regression
coverage.

```text
Ran 4 tests in 1.516s
FAILED (failures=12)
```

After adding pre-normalization and forwarding-field validation, the four empty
matrices plus valid forwarding, attached include, and response-file regressions
all passed with the configured MSYS2 Bash.

### Focused Verification

```text
$env:BASH = 'C:\personal_apps\msys64\usr\bin\bash.exe'
python -m unittest -v <35 focused Subtask 13.3d tests>
Ran 35 tests in 9.482s
OK (skipped=1: POSIX-only header-symlink integration on Windows)

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

No later task, broad-suite, CUDA, H20, subagent, or reviewer work was
performed. Python 3.6 runtime execution and the POSIX symlink integration
remain unavailable locally and are not claimed.

## Compliance Hardening Subtask 13.3e

### Production Closure Findings Addressed

- Direct linker/compiler control files are classified in separated, attached,
  and equals forms for `-T`, `--script`, GCC `-specs`, and Clang `--config`.
  Missing or empty values fail during recipe parsing. Valid control inputs on
  commands outside the `lenet_cuda` closure remain isolated.
- Every explicit attached or practical separated `-lNAME`/`-l:FILE` input on a
  production-reachable command now rejects without requiring a paired `-L`.
  Existing direct linker forwarding and paired search/library rejection remain
  fail-closed.
- Recognized `ar`, `llvm-ar`, and `gcc-ar` `q`/`r` commands are treated as
  stateful updates. If reachable from `lenet_cuda`, they reject before member
  traversal because the command cannot prove that an existing archive lacks
  unlisted stale members. Malformed grammar still rejects during parsing.
- Archive output recognition no longer depends on `.a`/`.lib`, so a valid
  extensionless archive output remains in the graph and receives the stateful
  rejection. Existing pre/post-operation options and placement parsing remain
  covered. An unrelated extensionless test-only archive remains accepted.
- The normal NVCC compile/link shape without explicit libraries, control files,
  or archives remains accepted. Direct non-archive multilayer provenance tests
  remain unchanged elsewhere in the focused module.

### TDD Evidence

The production-closure matrices and revised archive expectations were added
before implementation. The RED run demonstrated each missing behavior:

```text
python -m unittest -v <16 focused Subtask 13.3e tests>
Ran 16 tests in 3.875s
FAILED (failures=36)
```

Failures included all archive forms still reporting listed test-owned members
instead of stateful provenance, empty and stale-member updates being accepted,
extensionless archives being rejected only as invalid paths, all eight control
spellings and eight missing/empty forms lacking control diagnostics, and all
four bare/separated library forms bypassing the linker gate. The normal link
and valid test-only control isolation were already green; the extensionless
test-only archive was RED only because its output was not recognized.

The minimal implementation then made the same 16-test command pass:

```text
Ran 16 tests in 3.795s
OK
```

### Focused Verification

```text
$env:BASH = 'C:\personal_apps\msys64\usr\bin\bash.exe'
python -m unittest -v <16 focused Subtask 13.3e tests>
Ran 16 tests in 3.788s
OK

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

No general option/artifact identity work from 13.3f, build-log/documentation
work, broad suite, CUDA, H20, subagent, or reviewer work was performed. Python
3.6 runtime execution remains unavailable locally and is not claimed.

## Compliance Hardening Subtask 13.3e Fix Round 1

### Review Findings Addressed

- Exact practical NVCC options `-link`, `-lib`, `-ltoir`, and `-lineinfo` are
  classified before generic attached `-lNAME` libraries, so production-reachable
  commands using them remain valid. Explicit `-lfoo`, `-l:libfoo.a`, and
  separated `-l foo` inputs remain rejected.
- Separated NVCC `-ldir VALUE` is a non-artifact option. Its option and operand
  cannot become graph inputs, while missing or empty values fail during parsing.
  No long alias was added because the current parser supports none.
- Malformed archive regressions now pair each command with its parse-specific
  operation, modifier, option, position, or missing-path diagnostic. Every case
  explicitly asserts that a stateful-archive diagnostic is insufficient.

### TDD Evidence

The NVCC positive/malformed matrices and tightened archive assertions were
added before the parser change. The archive cases already produced the required
parse diagnostics. Six NVCC subcases failed because generic library detection
captured three exact no-value options, valid `-ldir`, and both missing/empty
`-ldir` forms:

```text
python -m unittest -v <4 focused review-finding tests>
Ran 4 tests in 1.240s
FAILED (failures=6)
```

Adding exact NVCC exclusions plus separated non-artifact `-ldir` parsing made
the same command pass:

```text
Ran 4 tests in 1.221s
OK
```

### Focused Verification

```text
$env:BASH = 'C:\personal_apps\msys64\usr\bin\bash.exe'
python -m unittest -v <19 focused Subtask 13.3e tests>
Ran 19 tests in 4.682s
OK

python -m py_compile scripts/analyze_build_graph.py \
  tests/test_check_prohibited.py
exit 0

ast.parse(..., feature_version=(3, 6))
Python 3.6 grammar check passed for 2 files

git diff --check
exit 0 (Windows LF-to-CRLF conversion notices only)
```

No later task, broad suite, CUDA, H20, subagent, or reviewer work was performed.
Python 3.6 runtime execution remains unavailable locally and is not claimed.
