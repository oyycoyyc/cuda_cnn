# Code Commenting Design

## Goal

Add accurate English comments that explain the purpose, structure, contracts,
algorithms, data layouts, ownership, and important invariants of all
project-owned code without changing program behavior or any existing
non-comment source line.

## Binding Constraint

This work is documentation-only. Existing code, declarations, expressions,
commands, recipes, tests, data, and behavior must remain unchanged.

- Do not delete, replace, reorder, or reformat an existing non-comment line.
- Add only standalone comments in the native syntax of each file.
- Do not add Python docstrings because they create executable string-expression
  nodes and can change runtime metadata.
- Do not add declarations, macros, helper functions, build rules, tests, or
  dependencies.
- Do not opportunistically fix defects or clean up style.
- If a defect is certain, record its file, line, evidence, and impact, then
  report it to the user. Do not modify the affected implementation unless the
  user approves a separate bug-fix task.

## Scope

Included:

- `include/`
- `src/`
- `tests/`
- `scripts/`
- `Makefile`
- Project-owned shell scripts

Excluded:

- `README.md` and existing files under `docs/`
- `.superpowers/` execution artifacts other than this approved design and its
  future implementation plan
- Dependency lock files
- Generated build output, caches, data, weights, and acceptance evidence
- Third-party files

The current inventory contains approximately 63 included files and 17,700
physical lines. Existing public-header and CUDA-kernel comments are generally
strong. The largest gaps are model/training implementation, host
infrastructure, build/compliance tooling, and tests.

## Comment Language And Style

All new comments use concise technical English and follow the established
project style.

### File And Module Comments

Each code file receives a short overview when its role is not already obvious.
The overview explains what the module owns, how it fits into the project, and
which contracts it enforces. It does not restate the filename or enumerate
every symbol.

### Types And Functions

Comments are added where they explain one or more of:

- inputs, outputs, and error behavior;
- object or buffer ownership and lifetime;
- tensor shapes, memory layouts, and aliasing rules;
- deterministic ordering or seed-domain behavior;
- serialization formats and validation boundaries;
- build, scanner, or test contracts;
- non-obvious reasons for an implementation choice.

Trivial wrappers, assignments, and loops do not receive comments merely to
increase comment count.

### Algorithms And Invariants

Complex blocks may receive comments that explain phases, formulas, indexing,
tail handling, synchronization, race freedom, numerical stability, and
fail-closed behavior. Comments explain why the code is structured as it is,
not a line-by-line translation of syntax.

### Existing Comments

Existing comments remain unchanged unless an inserted standalone comment is
needed nearby. Already-documented public declarations and CUDA kernel bodies
receive only a gap review. New comments must not split an existing adjacency
relationship required by `scripts/check_comments.py`.

## Language-Specific Safety Rules

### C, C++, And CUDA

- Add standalone `//` lines or standalone block comments.
- Do not add inline suffix comments to existing code lines.
- Do not insert comments inside macro continuation lines, preprocessor
  directives, launch syntax, attributes, or declaration tokens.
- Preserve every existing declaration and kernel signature byte-for-byte.

### Python

- Add standalone `#` comments only.
- Do not add module, class, or function docstrings.
- Do not insert comments inside fixture strings, generated source text,
  backslash continuations, or expression tokens.
- Avoid writing source-like definitions in comments that could interfere with
  the project's manual-definition scanner.

### Shell

- Add standalone `#` comments before logical command groups.
- Do not insert comments inside continued commands, heredocs, pipelines, or
  quoted data.

### Makefile

- Add standalone `#` comments above variable groups, target groups, and rule
  families.
- Do not insert comments into recipe bodies or continued variable definitions.
- Preserve tabs, target prerequisites, variables, and command text exactly.

## Compliance Vocabulary Constraint

The prohibited-dependency scanner reads source comments as ordinary text. New
comments in scanned files must not spell prohibited library, framework, CPU
fallback, or nondeterministic-random identifiers. Use neutral descriptions such
as "prohibited external dependency" when such context is necessary. Every batch
must run the source policy scanner before review.

## Work Batches

### Batch 1: Production Core

Files include `src/lenet.cu`, `src/train.cu`, `src/main.cu`, and the host
launcher/validation portions of `src/kernels/*.cu`.

Document the model data flow, fixed memory plan, forward and backward stages,
optimizer update, parameter ordering, finite checks, training/evaluation/
inference orchestration, device selection, timing boundary, and CUDA launch
validation. Existing per-kernel algorithm comments remain unchanged.

### Batch 2: Production Infrastructure

Files include binary I/O, dataset, checkpoint, parameter, random, CLI,
reporting, and training-data implementations.

Document binary formats, checked arithmetic, atomic replacement, canonical
parameter order, deterministic random domains, split and batch rules, CLI
validation, exact output records, and error contracts.

### Batch 3: Build And Compliance Tooling

Files include `Makefile`, project Python scripts, and shell scripts.

Document build graph structure, host/CUDA test separation, generated
dependencies, data conversion, comment inventory, source/build graph analysis,
scanner modes, temporary-file ownership, and fail-closed evidence semantics.

### Batch 4: C++ And CUDA Tests

Files include the test harness, CPU reference implementation, operator and
workflow tests, host unit tests, and Makefile regression shell tests.

Document fixture intent, oracle relationships, numerical tolerances, negative
contracts, determinism checks, and end-to-end scenarios without restating each
assertion.

### Batch 5: Python Tests And Consistency Pass

Files include Python test modules and the reproducibility-vector generator.
Document the boundary each module protects, fixture construction, failure
injection, and why important cases discriminate correct from incorrect
behavior. Finish with a gap-only review of already well-commented headers.

Each batch receives its own commit and independent read-only review. A later
batch does not begin while the prior batch has an unresolved non-comment change
or inaccurate comment.

## No-Code-Change Proof

For each batch, record its base commit and enforce all of the following:

1. The diff contains no removed source line.
2. Every added source line is a standalone comment line in that file's syntax.
3. Removing only the newly added comment lines reconstructs each base file
   byte-for-byte and in the original order.
4. No file is renamed and no executable/configuration file is added or removed.
5. A temporary verifier outside the repository performs the mechanical check;
   it is not committed as project code.

Reviewers separately inspect the diff for comment accuracy and context. Passing
the mechanical check proves preservation, not semantic accuracy, so both gates
are required.

## Verification

After each batch:

- run the external comment-only diff verifier;
- run `python scripts/check_comments.py --root . --checklist
  docs/comment-review-checklist.md`;
- run `bash scripts/check_prohibited.sh source .`;
- run language-appropriate syntax or focused test checks;
- run `git diff --check`;
- obtain an independent read-only review.

After all batches:

- run `make host-tests makefile-tests python-tests compliance`;
- run Python byte-compilation for all changed Python files;
- parse changed Python files with Python 3.6 grammar;
- run `bash -n` for changed shell scripts;
- run forced verbose Make dry-runs for `all` and `cuda-tests`;
- re-run the external comment-only diff verifier over the complete range;
- obtain a final whole-range review.

CUDA compilation and H20 execution remain target-only Task 14 work. This
commenting task must not claim those checks passed.

## Bug Reporting

Potential defects are not fixed in this task. A report is created only when the
defect is certain from direct code evidence. Each report contains:

- severity and user-visible impact;
- exact file and line references;
- the execution path or invariant that proves the defect;
- existing tests that expose or miss it;
- a proposed fix outline clearly labeled as not implemented.

The affected code remains unchanged until the user explicitly approves a
separate fix.

## Completion Criteria

The work is complete only when:

- every included code file has appropriate structure and key-logic comments;
- no comment makes a claim stronger than the code proves;
- every batch and the complete range pass the comment-only preservation check;
- all locally executable verification passes;
- independent reviews find no inaccurate comment or non-comment change;
- any certain bugs discovered during reading have been reported but not fixed;
- the worktree contains no unrelated modification.
