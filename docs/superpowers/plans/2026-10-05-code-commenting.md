# Code Commenting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add accurate English structure and key-logic comments to every project-owned code file while preserving every existing non-comment line byte-for-byte.

**Architecture:** Execute five independently committed comment batches: production core, production infrastructure, build/compliance tooling, native tests, and Python tests. A temporary verifier outside the repository rejects deleted or modified source lines and accepts only standalone comment additions; each batch also passes project scanners, focused verification, and independent review.

**Tech Stack:** C++14, CUDA C++, Python 3.6-compatible source, Bash 4.4, POSIX shell, GNU Make, Git, Python `unittest`.

**Spec:** `docs/superpowers/specs/2026-10-05-code-commenting-design.md`

## Global Constraints

- Work only in `E:\cuda_lenet_mnist\.worktrees\rebuild` on `feature/rebuild`.
- The approved code baseline is `c693370` (`docs: design code commenting pass`). The committed plan is a documentation-only descendant of that baseline before Batch 1 starts.
- Existing code, declarations, expressions, commands, recipes, tests, data, and behavior must remain unchanged.
- Do not delete, replace, reorder, or reformat an existing non-comment line.
- Add only standalone English comments in each file's native comment syntax. Use `//` line comments, not new block comments, in C++/CUDA files.
- Do not add blank lines, inline suffix comments, Python docstrings, declarations, macros, helpers, build rules, tests, dependencies, or generated files.
- Do not modify `README.md`, existing `docs/` content, `.superpowers/` history, dependency locks, generated outputs, data, weights, or acceptance evidence. The only documentation files introduced by this work are the approved design and this plan.
- Do not change existing comments. Already-documented headers and CUDA kernel bodies receive a gap review only.
- Do not mark `docs/comment-review-checklist.md` entries reviewed; semantic checklist completion remains Task 14 human work.
- Comments in scanned code must not spell prohibited dependency, fallback, or nondeterministic-random identifiers. Use neutral phrases such as `prohibited external dependency`.
- Use command-scoped commit identity `OpenCode <opencode@localhost>`; do not change Git configuration and do not amend commits.
- Commit this plan before Task 1 with subject `docs: plan code commenting pass`; Task 1 must start from that clean documentation-only descendant of `c693370`.
- Preserve unrelated changes. Never use destructive Git commands.
- Do not claim CUDA compilation or H20 execution on the Windows host.
- A certain bug is reported only. Do not change its code or test expectations without separate user approval.

## Known Bugs: Report Only, Never Modify In This Plan

1. `scripts/prepare_mnist.py:144` validates `label` but formats `row_index` in the out-of-range diagnostic. Impact is an incorrect error value only. Existing tests assert only the exception type. Do not change the format operand or its tests.
2. `scripts/check_comments.py:42-100,392-411` applies the C++ sanitizer to Python when resolving `manual:` definitions, but does not neutralize Python `#` comments. A quote or source-like `def convert_parquet(` in a comment can hide or duplicate the real definition. Do not fix the scanner. In `scripts/prepare_mnist.py` comments, use plain wording without apostrophes, quotation marks, or text resembling a Python definition.

## Temporary Comment-Only Verifier

The execution controller creates this untracked file outside the repository:
`C:\Users\26030\AppData\Local\Temp\opencode\cuda_lenet_verify_comment_only.py`.
It is a session tool, not project code.

```python
from __future__ import print_function

import os
import subprocess
import sys
import io
import tokenize


C_LIKE_SUFFIXES = (".c", ".cc", ".cpp", ".cuh", ".cu", ".h", ".hpp")
CODE_SUFFIXES = C_LIKE_SUFFIXES + (".py", ".sh")
EXPECTED_PATHS = (
    "Makefile",
    "include/checkpoint.h",
    "include/cli.h",
    "include/cuda_check.h",
    "include/dataset.h",
    "include/layers.h",
    "include/lenet.h",
    "include/parameters.h",
    "include/random.h",
    "include/reporting.h",
    "include/tensor.h",
    "include/train.h",
    "include/training_data.h",
    "scripts/analyze_build_graph.py",
    "scripts/check_comments.py",
    "scripts/check_prohibited.sh",
    "scripts/prepare_mnist.py",
    "src/binary_io.h",
    "src/checkpoint.cpp",
    "src/cli.cpp",
    "src/dataset.cpp",
    "src/kernels/activation.cu",
    "src/kernels/adam.cu",
    "src/kernels/convolution.cu",
    "src/kernels/input.cu",
    "src/kernels/linear.cu",
    "src/kernels/loss.cu",
    "src/kernels/metrics.cu",
    "src/kernels/pooling.cu",
    "src/lenet.cu",
    "src/main.cu",
    "src/parameters.cpp",
    "src/random.cpp",
    "src/reporting.cpp",
    "src/train.cu",
    "src/training_data.cpp",
    "src/workflow_test_hooks.h",
    "tests/checkpoint_tests.cpp",
    "tests/cli_tests.cpp",
    "tests/compliance_tests.py",
    "tests/cpu_reference.cpp",
    "tests/cpu_reference.h",
    "tests/cpu_reference_tests.cpp",
    "tests/dataset_probe.cpp",
    "tests/dataset_tests.cpp",
    "tests/generate_repro_vectors.py",
    "tests/makefile_tests.sh",
    "tests/operator_tests.cu",
    "tests/output_format_tests.py",
    "tests/parameters_tests.cpp",
    "tests/random_tests.cpp",
    "tests/reporting_tests.cpp",
    "tests/smoke_tests.cpp",
    "tests/test_check_comments.py",
    "tests/test_check_prohibited.py",
    "tests/test_data_interop.py",
    "tests/test_documentation.py",
    "tests/test_harness.h",
    "tests/test_prepare_mnist.py",
    "tests/training_data_tests.cpp",
    "tests/workflow_tests.cu",
)


def fail(message):
    raise SystemExit("comment-only verification failed: " + message)


def is_comment_line(path, content):
    stripped = content.lstrip()
    if not stripped:
        return False
    if path == "Makefile":
        return content.startswith("#")
    if path.endswith((".py", ".sh")):
        return stripped.startswith("#")
    if path.endswith(C_LIKE_SUFFIXES):
        return stripped.startswith("//")
    return False


def is_code_path(path):
    return path == "Makefile" or path.endswith(CODE_SUFFIXES)


def git_output(repository, arguments):
    command = ["git", "-C", repository] + arguments
    return subprocess.check_output(command, universal_newlines=True)


def verify_scope(repository):
    tracked = git_output(
        repository,
        ["ls-files", "--", "Makefile", "include", "src", "tests", "scripts"],
    ).splitlines()
    actual = set(path for path in tracked if is_code_path(path))
    expected = set(EXPECTED_PATHS)
    if actual != expected:
        fail("scope mismatch missing={0} unexpected={1}".format(
            sorted(expected - actual), sorted(actual - expected)))
    status = git_output(repository, ["status", "--porcelain", "--untracked-files=all"])
    for line in status.splitlines():
        path = line[3:].replace("\\", "/")
        if line.startswith("??") and is_code_path(path):
            fail("untracked code file: " + path)
    print("scope verification passed: files={0}".format(len(actual)))


def python_tokens(contents):
    ignored = (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING,
               tokenize.ENDMARKER)
    reader = io.StringIO(contents).readline
    return [(item.type, item.string) for item in tokenize.generate_tokens(reader)
            if item.type not in ignored]


def verify_python_lexing(repository, base, paths):
    for path in paths:
        if not path.endswith(".py"):
            continue
        baseline = git_output(repository, ["show", base + ":" + path])
        with io.open(os.path.join(repository, path), "r", encoding="utf-8") as handle:
            current = handle.read()
        if python_tokens(baseline) != python_tokens(current):
            fail("Python non-comment tokens changed in " + path)


def main():
    if len(sys.argv) == 3 and sys.argv[2] == "--scope":
        verify_scope(os.path.abspath(sys.argv[1]))
        return 0
    if len(sys.argv) < 4:
        fail("usage: verify_comment_only.py REPO BASE PATH...|--all")
    repository = os.path.abspath(sys.argv[1])
    base = sys.argv[2]
    paths = sys.argv[3:]
    if paths == ["--all"]:
        paths = list(EXPECTED_PATHS)
    unexpected = sorted(set(paths) - set(EXPECTED_PATHS))
    if unexpected:
        fail("path outside approved scope: " + repr(unexpected))
    for path in paths:
        subprocess.check_call(
            ["git", "-C", repository, "cat-file", "-e", base + ":" + path])
    command = [
        "git", "-C", repository, "diff", "--no-color", "--unified=0",
        base, "--",
    ] + paths
    output = subprocess.check_output(command, universal_newlines=True)
    current_path = None
    additions = 0
    forbidden_headers = (
        "new file mode ", "deleted file mode ", "rename from ", "rename to ",
    )
    for line in output.splitlines():
        if line.startswith("diff --git "):
            current_path = None
            continue
        if line.startswith("Binary files "):
            fail("binary diff is not permitted: " + line)
        if line.startswith(forbidden_headers):
            fail("file topology changed: " + line)
        if line.startswith("+++ b/"):
            current_path = line[len("+++ b/"):]
            continue
        if line.startswith("--- ") or line.startswith("@@") or line.startswith("index "):
            continue
        if line.startswith("-"):
            fail("existing line removed or replaced in {0}: {1}".format(
                current_path or "<unknown>", line[1:]))
        if line.startswith("+"):
            if current_path is None:
                fail("addition has no file header")
            content = line[1:]
            if not is_comment_line(current_path, content):
                fail("non-comment line added in {0}: {1}".format(
                    current_path, content))
            additions += 1
    verify_python_lexing(repository, base, paths)
    print("comment-only verification passed: additions={0}".format(additions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

The verifier's policy is intentionally stricter than the language parsers:
every changed project file must be pre-existing, every diff hunk must contain
only additions, C++/CUDA additions must be `//` lines, Make comments must start
at column zero, and Python non-comment tokens must remain identical. This
mechanically proves that removing the additions reconstructs the baseline and
catches a `#` line accidentally inserted into a Python string. Shell comments
inside quoted text or here-documents remain a manual-review concern, so shell
comments must be placed only between complete commands. C++/CUDA comments inside
raw string literals or backslash-continued directives are also a manual-review
concern and are prohibited placement points.

---

### Task 1: Comment The Production Core

**Files:**
- Modify: `src/lenet.cu`
- Modify: `src/train.cu`
- Modify: `src/main.cu`
- Modify: `src/kernels/activation.cu`
- Modify: `src/kernels/adam.cu`
- Modify: `src/kernels/convolution.cu`
- Modify: `src/kernels/input.cu`
- Modify: `src/kernels/linear.cu`
- Modify: `src/kernels/loss.cu`
- Modify: `src/kernels/metrics.cu`
- Modify: `src/kernels/pooling.cu`

**Interfaces:**
- Consumes: Approved comment style and preservation contract from the spec; baseline commit `c693370`.
- Produces: Commented model/workflow/launcher implementation with every original non-comment line intact.

- [ ] **Step 1: Create the external verifier and record the base**

Verify the approved temp parent exists:

```powershell
Test-Path -LiteralPath "C:\Users\26030\AppData\Local\Temp\opencode"
git rev-parse HEAD
git status --short
```

Expected: parent is `True` and status is clean. Create the temporary verifier
with the exact content in this plan. The committed plan means HEAD is a
documentation-only descendant of `c693370`, not `c693370` itself. Run these
gates before edits:

```powershell
git merge-base --is-ancestor c693370 HEAD
if ((& git rev-parse HEAD).Trim() -eq "c693370") { throw "plan commit missing" }
python "C:\Users\26030\AppData\Local\Temp\opencode\cuda_lenet_verify_comment_only.py" `
  "E:\cuda_lenet_mnist\.worktrees\rebuild" --scope
```

Expected: the ancestry check exits zero and the scope verifier reports exactly
61 tracked code files with no untracked code file. Run the verifier on the Task
1 paths before edits and expect `additions=0`.

- [ ] **Step 2: Add model and workflow comments**

In `src/lenet.cu`, add standalone comments for:

- the fixed parameter/activation/gradient/winner/scan budgets;
- `CheckedAdd`, `CheckedMultiply`, `RequiredBytes`, and `ValidateAdamW`;
- `ScanTensor` and the `LeNet::Impl` constructor's one-arena setup;
- the exact forward chain and zero-copy pool2-to-fc1 alias;
- the reverse gradient chain and execution-state transitions;
- AdamW bias corrections, canonical parameter iteration, and bias decay rule;
- import/export ordering, finite scans, batch validation, bump allocation, and
  the complete `PlanArena` layout;
- the purpose of major member groups and the public forwarding/test-access
  methods.

Preserve the existing pool2 flatten comment verbatim. A representative style is:

```cpp
// Owns the single device arena and partitions it into canonical parameter,
// activation, gradient, moment, winner, and diagnostic regions.
```

In `src/train.cu`, comment the stream/event/storage RAII types, device and batch
helpers, finite checks, timed evaluation boundary, learning-rate schedule, and
the complete train/evaluate/infer orchestration. Explain strict-best checkpoint
replacement and reload-before-final-test. Preserve the existing warm-up comment.

In `src/main.cu`, add a module comment and explain syntax parsing, command
dispatch, usage errors, runtime errors, and the unhandled-enum fallback.

- [ ] **Step 3: Add launcher and validation comments without touching kernels**

For each `src/kernels/*.cu` file, leave all existing comments immediately above
`__global__` kernels untouched. Add comments only for the host-side constants,
shape/range structs, overflow checks, buffer overlap checks, block-count rules,
scalar validation, and public `Launch*` wrappers.

Required file-specific content:

- `activation.cu`: 256-thread grid-stride launch cap, zero/ReLU wrappers, and
  zero-count behavior.
- `adam.cu`: disjoint buffer ranges, scalar bounds, two same-stream finite-scan
  launches, and smallest-index result.
- `convolution.cu`: derived output shape/counts and the ordered input/weight/bias
  gradient launches.
- `input.cu`: fixed 28x28 geometry, grid capacity, one-based augmentation epoch,
  and normalized translation launch.
- `linear.cu`: shape/count validation and ordered input/weight/bias gradients.
- `loss.cu`: class-count limit, one-block-per-sample softmax/loss, mean reduction,
  and inference-only softmax.
- `metrics.cu`: one-block-per-sample argmax flags followed by deterministic count.
- `pooling.cu`: even 2x2 geometry and same-stream zero-before-scatter ordering.

Place launcher comments above each function, never between adjacent kernel
launches and `CUDA_KERNEL_CHECK` calls.

- [ ] **Step 4: Prove the Task 1 diff is comments only**

Run:

```powershell
python "C:\Users\26030\AppData\Local\Temp\opencode\cuda_lenet_verify_comment_only.py" `
  "E:\cuda_lenet_mnist\.worktrees\rebuild" c693370 `
  src/lenet.cu src/train.cu src/main.cu `
  src/kernels/activation.cu src/kernels/adam.cu `
  src/kernels/convolution.cu src/kernels/input.cu `
  src/kernels/linear.cu src/kernels/loss.cu `
  src/kernels/metrics.cu src/kernels/pooling.cu
git diff --check
```

Expected: verifier passes with only comment additions; `git diff --check` exits
zero. Any removed line is a blocker: restore the exact base line rather than
accepting a semantically equivalent rewrite.

- [ ] **Step 5: Run the comment and source-policy gates**

```powershell
$env:PATH = "C:\personal_apps\msys64\usr\bin;C:\personal_apps\msys64\ucrt64\bin;" + $env:PATH
$env:BASH = "C:\personal_apps\msys64\usr\bin\bash.exe"
& "C:\personal_apps\anaconda3\python.exe" scripts/check_comments.py --root . --checklist docs/comment-review-checklist.md
& "C:\personal_apps\msys64\usr\bin\bash.exe" scripts/check_prohibited.sh source .
```

Expected: `inventory=147 checklist_items=147` and
`mode=source scope=make-compiled-inputs` pass. Do not run CUDA binaries locally.

- [ ] **Step 6: Commit Batch 1**

Stage only the eleven listed files, inspect the staged diff, then commit:

```powershell
git -c user.name=OpenCode -c user.email=opencode@localhost commit -m "docs: explain CUDA model and workflow"
```

---

### Task 2: Comment Production Infrastructure

**Files:**
- Modify: `src/binary_io.h`
- Modify: `src/dataset.cpp`
- Modify: `src/checkpoint.cpp`
- Modify: `src/parameters.cpp`
- Modify: `src/random.cpp`
- Modify: `src/cli.cpp`
- Modify: `src/reporting.cpp`
- Modify: `src/training_data.cpp`
- Inspect only: `include/checkpoint.h`
- Inspect only: `include/cli.h`
- Inspect only: `include/cuda_check.h`
- Inspect only: `include/dataset.h`
- Inspect only: `include/layers.h`
- Inspect only: `include/lenet.h`
- Inspect only: `include/parameters.h`
- Inspect only: `include/random.h`
- Inspect only: `include/reporting.h`
- Inspect only: `include/tensor.h`
- Inspect only: `include/train.h`
- Inspect only: `include/training_data.h`
- Inspect only: `src/workflow_test_hooks.h`

**Interfaces:**
- Consumes: Reviewed Task 1 commit; documented public contracts already present in `include/*.h`.
- Produces: Commented host infrastructure while preserving manual checklist definition matching.

- [ ] **Step 1: Record the Task 2 base and inspect existing contracts**

Record the batch base with
`$env:TASK2_BASE = (& git rev-parse HEAD).Trim()`. Confirm all inspect-only
files already explain public inputs, outputs, layouts, ownership, and errors. Do
not change a well-documented file merely to create a diff.

- [ ] **Step 2: Comment binary, dataset, and checkpoint formats**

Add a module overview and comments above every helper in `src/binary_io.h`.
Explain exact reads/writes, little-endian reconstruction, 32-bit float bit
transport, checked stream sizes, and path-qualified failure behavior.

In `src/dataset.cpp`, document the 24-byte header, 784-byte image, checked size
arithmetic, exact-file-size requirement, label validation, and the distinction
between loading and role-specific count enforcement.

In `src/checkpoint.cpp`, preserve all existing format/platform comments. Add
comments for metadata/schema validation, same-directory temporary naming,
replacement, exact 40-byte header and ten 60-byte metadata records, canonical
name/order validation, destination-preserving save failure, and validate-before-
allocation load behavior.

Comments for manual checklist symbols must appear above the complete signature;
never between `)` and `{`.

- [ ] **Step 3: Comment deterministic parameters, CLI, reporting, and batching**

Add comments for:

- `src/parameters.cpp`: canonical ten-tensor schema, open-unit mapping,
  seed-domain separation, He scaling, paired Box-Muller outputs, and zero biases.
- `src/random.cpp`: deterministic stream domains, unbiased bounded draws,
  in-place Fisher-Yates, canonical split, one-based epoch, and translation keys.
- `src/cli.cpp`: side-effect-free parsing, per-command allowlists, duplicate and
  range rejection, defaults, required options, and commit-on-success behavior.
- `src/reporting.cpp`: exact record grammar, numeric precision, locale isolation,
  validation, and caller stream-state preservation.
- `src/training_data.cpp`: official/nonstandard split policy, reusable batch
  storage, index preservation, partial batches, and pre-copy validation.

Use neutral vocabulary in `include/` and `src/` comments. Do not name forbidden
APIs or libraries.

- [ ] **Step 4: Prove and verify Batch 2**

Run the external verifier from `TASK2_BASE` over the eight modified files:

```powershell
python "C:\Users\26030\AppData\Local\Temp\opencode\cuda_lenet_verify_comment_only.py" `
  "E:\cuda_lenet_mnist\.worktrees\rebuild" $env:TASK2_BASE `
  src/binary_io.h src/dataset.cpp src/checkpoint.cpp src/parameters.cpp `
  src/random.cpp src/cli.cpp src/reporting.cpp src/training_data.cpp
```

Then run:

```powershell
$env:PATH = "C:\personal_apps\msys64\usr\bin;C:\personal_apps\msys64\ucrt64\bin;" + $env:PATH
$env:BASH = "C:\personal_apps\msys64\usr\bin\bash.exe"
git diff --check
& "C:\personal_apps\anaconda3\python.exe" scripts/check_comments.py --root . --checklist docs/comment-review-checklist.md
& "C:\personal_apps\msys64\usr\bin\bash.exe" scripts/check_prohibited.sh source .
& "C:\personal_apps\msys64\usr\bin\make.exe" host-tests `
  "PYTHON=C:/personal_apps/anaconda3/python.exe" "CXX=g++" `
  "BASH=C:/personal_apps/msys64/usr/bin/bash.exe"
```

Expected: only comment additions, all nine host suites pass, checklist remains
147/147, and the source policy passes.

- [ ] **Step 5: Commit Batch 2**

Commit only files that actually received comments:

```powershell
git -c user.name=OpenCode -c user.email=opencode@localhost commit -m "docs: explain data and host infrastructure"
```

---

### Task 3: Comment Build And Compliance Tooling

**Files:**
- Modify: `Makefile`
- Modify: `scripts/prepare_mnist.py`
- Modify: `scripts/check_comments.py`
- Modify: `scripts/analyze_build_graph.py`
- Modify: `scripts/check_prohibited.sh`

**Interfaces:**
- Consumes: Reviewed production comment batches and the unchanged scanner interfaces.
- Produces: Documented build graph and compliance tools without altering recipes, scanner behavior, or Python ASTs.

- [ ] **Step 1: Record the Task 3 base and guard known bugs**

Record the batch base with
`$env:TASK3_BASE = (& git rev-parse HEAD).Trim()`. Re-read the two known-bug
entries. Confirm the implementation and tests remain unchanged. In
`scripts/prepare_mnist.py` comments, do not use apostrophes, quotation marks, or
text resembling `def convert_parquet(`.

- [ ] **Step 2: Comment the Makefile only at safe boundaries**

Add standalone column-zero comments above these logical groups:

- overrideable tools/default goal, architecture derivation, compile flags, and
  output directories;
- platform suffix, test discovery, basename collision partition, object lists,
  test program/module lists, dependency files, verbosity, and phony/secondary
  declarations;
- `all`, manifest, host/CUDA/Python/compliance/aggregate targets;
- unique, colliding, explicit, alias, application, probe, and object rules;
- directory creation, clean, and optional generated dependency inclusion.

Never insert a comment inside a backslash continuation, target/recipe pair,
conditional guard, or recipe body. Preserve every tab and keep
`-include $(DEPENDENCY_FILES)` as the final logical statement.

- [ ] **Step 3: Comment data preparation and comment inventory tools**

In `scripts/prepare_mnist.py`, add `#` comments for pinned split metadata,
streaming hashes, same-directory temporary files, cache revalidation, image
decoding, exact binary layout, atomic conversion, CLI exclusivity, and main
dispatch. Preserve its existing docstrings and do not add new docstrings.

In `scripts/check_comments.py`, comment regex groups, comment/string
sanitization, adjacency, canonical signature IDs, brace/access tracking, header
and kernel discovery, checklist parsing, manual-definition resolution, and final
reconciliation. Describe behavior without fixing the Python-comment sanitizer.

- [ ] **Step 4: Comment build-graph and shell policy tools**

In `scripts/analyze_build_graph.py`, add grouped comments for constants; path
identity; executable/prefix detection; option operands; archive/output parsing;
include and forced-input contexts; shell ambiguity; recipe parsing; manifest
reconciliation; recursive active inputs; artifact graph traversal; source mode;
build-log mode; and CLI/exit behavior. Preserve the existing recursion comment.

In `scripts/check_prohibited.sh`, comment usage/error helpers, Python selection,
grep exit semantics, root discovery, pattern-family scope, source-mode temporary
ownership/data flow, and actual/dry-run build-log evidence. Do not alter quoted
regex constants or continued commands.

- [ ] **Step 5: Prove Batch 3 and run tooling verification**

Run the external verifier from `TASK3_BASE` over the five files:

```powershell
python "C:\Users\26030\AppData\Local\Temp\opencode\cuda_lenet_verify_comment_only.py" `
  "E:\cuda_lenet_mnist\.worktrees\rebuild" $env:TASK3_BASE `
  Makefile scripts/prepare_mnist.py scripts/check_comments.py `
  scripts/analyze_build_graph.py scripts/check_prohibited.sh
```

Then run:

```powershell
$env:PATH = "C:\personal_apps\msys64\usr\bin;C:\personal_apps\msys64\ucrt64\bin;" + $env:PATH
$env:BASH = "C:\personal_apps\msys64\usr\bin\bash.exe"
git diff --check
& "C:\personal_apps\anaconda3\python.exe" -m py_compile `
  scripts/prepare_mnist.py scripts/check_comments.py scripts/analyze_build_graph.py
& "C:\personal_apps\anaconda3\python.exe" -c `
  "import ast; files=['scripts/prepare_mnist.py','scripts/check_comments.py','scripts/analyze_build_graph.py']; [ast.parse(open(p,'rb').read(), filename=p, feature_version=(3,6)) for p in files]"
& "C:\personal_apps\msys64\usr\bin\bash.exe" -n scripts/check_prohibited.sh
& "C:\personal_apps\anaconda3\python.exe" -m unittest -v `
  tests.test_prepare_mnist tests.test_check_comments tests.test_check_prohibited
& "C:\personal_apps\anaconda3\python.exe" scripts/check_comments.py --root . --checklist docs/comment-review-checklist.md
& "C:\personal_apps\msys64\usr\bin\bash.exe" scripts/check_prohibited.sh source .
& "C:\personal_apps\msys64\usr\bin\make.exe" -B -n V=1 all
& "C:\personal_apps\msys64\usr\bin\make.exe" -B -n V=1 cuda-tests
```

Expected: Python syntax/3.6 grammar,  scanner modules, shell syntax, comment
inventory, source policy, and Make command generation pass. Dry-runs are command
generation only and do not prove CUDA execution.

- [ ] **Step 6: Commit Batch 3**

```powershell
git -c user.name=OpenCode -c user.email=opencode@localhost commit -m "docs: explain build and compliance tooling"
```

---

### Task 4: Comment C++ And CUDA Tests

**Files:**
- Modify: `tests/test_harness.h`
- Inspect only: `tests/cpu_reference.h`
- Modify: `tests/cpu_reference.cpp`
- Modify: `tests/cpu_reference_tests.cpp`
- Modify: `tests/operator_tests.cu`
- Modify: `tests/workflow_tests.cu`
- Modify: `tests/checkpoint_tests.cpp`
- Modify: `tests/cli_tests.cpp`
- Modify: `tests/dataset_tests.cpp`
- Modify: `tests/dataset_probe.cpp`
- Modify: `tests/parameters_tests.cpp`
- Modify: `tests/random_tests.cpp`
- Modify: `tests/reporting_tests.cpp`
- Modify: `tests/smoke_tests.cpp`
- Modify: `tests/training_data_tests.cpp`
- Modify: `tests/makefile_tests.sh`

**Interfaces:**
- Consumes: Documented production contracts and unchanged test behavior.
- Produces: Test intent and oracle comments without changing assertions, fixtures, macros, or expected values.

- [ ] **Step 1: Record the Task 4 base and comment test infrastructure**

Record the batch base with
`$env:TASK4_BASE = (& git rev-parse HEAD).Trim()`. In `tests/test_harness.h`,
explain static registration, assertion exception formatting, suite naming,
filtering, pass/fail records, and single-evaluation macro contracts. Put comments
above each `#define`; never enter its backslash-continued body.

Gap-review `tests/cpu_reference.h` without changing its strong existing
contracts. In `tests/cpu_reference.cpp`, comment checked shapes/index formulas
and the independent convolution, activation, pooling, linear, softmax/loss, and
optimizer oracle algorithms. In `tests/cpu_reference_tests.cpp`, explain the
reference-oracle contract and group the convolution, activation, pooling,
linear, softmax/loss, and optimizer correctness cases.

- [ ] **Step 2: Comment operator and workflow test architecture**

In `tests/operator_tests.cu`, comment helper groups for copies/streams,
translation oracle, vector tolerance, operator comparisons, finite-difference
objectives, independent loss/optimizer references, full-network oracle, arena
layout inspection, and source-policy checks. Add one-line group comments above
the CUDA checking, device-buffer, input, activation, pooling, linear,
convolution, loss, metrics, optimizer, finite-scan, and LeNet test groups.
Preserve its existing five comment blocks.

In `tests/workflow_tests.cu`, comment fixture ownership, binary dataset/checkpoint
writers, output parsing, reload hooks, CPU logits oracle, option builders,
case-selection infrastructure, and each named workflow scenario: schedule,
validation, batch17, global step, strict checkpoint ties, reload-before-final,
overfit, one-epoch checkpoint, evaluation/timing, inference, and executable CLI.

- [ ] **Step 3: Comment host test modules and Make regression script**

For each host test file, add a module overview, comments for fixture/helpers, and
group-level intent:

- `checkpoint_tests.cpp`: exact offsets/sizes, malformed fixture mutations,
  bitwise floats, round-trip, and destination preservation.
- `cli_tests.cpp`: parser-only boundary, defaults, bounds, required/duplicate/
  cross-command rejection, and unchanged-options-on-failure.
- `dataset_tests.cpp`: MNISTC1 fixtures, little-endian helpers, malformed/trailing
  data, labels, and count enforcement.
- `dataset_probe.cpp`: standalone diagnostic output and exit codes.
- `parameters_tests.cpp`: canonical schema, reproducibility vectors, finite
  initialization, and no-mutation validation failures.
- `random_tests.cpp`: protocol vectors, bounded-draw rejection path, split,
  permutation, epoch, and translation domains. Avoid prohibited API names.
- `reporting_tests.cpp`: exact records, precision, locale isolation, caller state,
  and validation-before-output.
- `smoke_tests.cpp`: registration/link sanity only.
- `training_data_tests.cpp`: official/nonstandard splits, partial batches,
  correspondence, capacity reuse, and validation.
- `tests/makefile_tests.sh`: collision fixtures, cleanup trap, host/CUDA dry-run
  extraction, distinct output paths, and pass record.

- [ ] **Step 4: Prove and verify Batch 4**

Run the external verifier from `TASK4_BASE` over every Task 4 modified file:

```powershell
python "C:\Users\26030\AppData\Local\Temp\opencode\cuda_lenet_verify_comment_only.py" `
  "E:\cuda_lenet_mnist\.worktrees\rebuild" $env:TASK4_BASE `
  tests/test_harness.h tests/cpu_reference.cpp `
  tests/cpu_reference_tests.cpp tests/operator_tests.cu `
  tests/workflow_tests.cu tests/checkpoint_tests.cpp tests/cli_tests.cpp `
  tests/dataset_tests.cpp tests/dataset_probe.cpp tests/parameters_tests.cpp `
  tests/random_tests.cpp tests/reporting_tests.cpp tests/smoke_tests.cpp `
  tests/training_data_tests.cpp tests/makefile_tests.sh
```

Then run:

```powershell
$env:PATH = "C:\personal_apps\msys64\usr\bin;C:\personal_apps\msys64\ucrt64\bin;" + $env:PATH
$env:BASH = "C:\personal_apps\msys64\usr\bin\bash.exe"
git diff --check
& "C:\personal_apps\msys64\usr\bin\bash.exe" -n tests/makefile_tests.sh
& "C:\personal_apps\anaconda3\python.exe" scripts/check_comments.py --root . --checklist docs/comment-review-checklist.md
& "C:\personal_apps\msys64\usr\bin\bash.exe" scripts/check_prohibited.sh source .
& "C:\personal_apps\msys64\usr\bin\make.exe" host-tests makefile-tests `
  "PYTHON=C:/personal_apps/anaconda3/python.exe" "CXX=g++" `
  "BASH=C:/personal_apps/msys64/usr/bin/bash.exe"
```

Expected: comment-only proof, shell syntax, nine host suites, Make collision
regression, checklist, and policy scan pass. Do not execute operator/workflow
binaries locally; record them as Task 14 target-only verification.

- [ ] **Step 5: Commit Batch 4**

```powershell
git -c user.name=OpenCode -c user.email=opencode@localhost commit -m "docs: explain native test coverage"
```

---

### Task 5: Comment Python Tests And Complete The Gap Pass

**Files:**
- Modify: `tests/compliance_tests.py`
- Modify: `tests/generate_repro_vectors.py`
- Modify: `tests/output_format_tests.py`
- Modify: `tests/test_check_comments.py`
- Modify: `tests/test_check_prohibited.py`
- Modify: `tests/test_data_interop.py`
- Modify: `tests/test_documentation.py`
- Modify: `tests/test_prepare_mnist.py`
- Gap-review only: all included files from Tasks 1-4 and all `include/*.h`

**Interfaces:**
- Consumes: Four reviewed comment batches and unchanged Python fixtures.
- Produces: Commented Python test boundaries plus complete repository comment coverage.

- [ ] **Step 1: Record the Task 5 base and protect fixture strings**

Record the batch base with
`$env:TASK5_BASE = (& git rev-parse HEAD).Trim()`. For every Python file, add
only standalone `#` comments above a module/class/helper/test group. Never
insert comments into adjacent string literals, generated C++/Make/recipe text,
backslash continuations, parenthesized regex construction, dict/list fixtures,
or an existing docstring.

- [ ] **Step 2: Comment data, documentation, output, and aggregate tests**

Add comments for:

- `test_data_interop.py`: Python writer to C++ probe byte agreement.
- `test_prepare_mnist.py`: in-memory images, Parquet fixtures, conversion
  rejection, verified downloads/cache, atomic preservation, dependency hashes,
  and explicit conversion CLI.
- `test_documentation.py`: README/H20 environment, data, architecture, command,
  error, ordering, evidence, scope, and placeholder groups.
- `output_format_tests.py`: production formatter execution and documented record
  regexes.
- `compliance_tests.py`: source policy, Make targets/default goal, data ordering,
  architecture flags, and build marker.

Do not modify the known wrong diagnostic assertion gap in
`test_prepare_mnist.py`.

- [ ] **Step 3: Comment checker tests and reproducibility generator**

In `test_check_comments.py`, comment temporary project ownership, checker
invocation, the documented positive fixture, and groups covering adjacency,
declaration kinds, kernels, checklist sync, overloads, nested types, role IDs,
manual definitions, review gate, and real inventory.

In `test_check_prohibited.py`, comment the temp project, checker/analyzer helpers,
failure helper, canonical build log, canonicalization mocks, and these broad
groups only: source basics; recipes/manifest; include resolution; options,
archives and linker semantics; shell/glob lexer; artifact identity; build-log
structure; fail-closed harness behavior. Do not add a comment to every test.

In `generate_repro_vectors.py`, preserve its existing module docstring and two
existing comments. Add comments for constants, mixing/derivation, stream draws,
unbiased bounds, permutation/split/epoch/translation, initialization vectors,
serialization, and CLI.

- [ ] **Step 4: Perform the all-file gap review**

Check every included file from the spec. Confirm each file now has an accurate
module/structure comment or already-contained equivalent, and complex logic has
key invariant comments. Do not change a file whose existing comments already
satisfy the design. Record the final included-file list in the Task 5 report.

- [ ] **Step 5: Prove and verify Batch 5**

Run the external verifier from `TASK5_BASE` over the eight Python files:

```powershell
python "C:\Users\26030\AppData\Local\Temp\opencode\cuda_lenet_verify_comment_only.py" `
  "E:\cuda_lenet_mnist\.worktrees\rebuild" $env:TASK5_BASE `
  tests/compliance_tests.py tests/generate_repro_vectors.py `
  tests/output_format_tests.py tests/test_check_comments.py `
  tests/test_check_prohibited.py tests/test_data_interop.py `
  tests/test_documentation.py tests/test_prepare_mnist.py
```

Then run:

```powershell
$env:PATH = "C:\personal_apps\msys64\usr\bin;C:\personal_apps\msys64\ucrt64\bin;" + $env:PATH
$env:BASH = "C:\personal_apps\msys64\usr\bin\bash.exe"
git diff --check
& "C:\personal_apps\anaconda3\python.exe" -m py_compile `
  tests/compliance_tests.py tests/generate_repro_vectors.py `
  tests/output_format_tests.py tests/test_check_comments.py `
  tests/test_check_prohibited.py tests/test_data_interop.py `
  tests/test_documentation.py tests/test_prepare_mnist.py
& "C:\personal_apps\anaconda3\python.exe" -c `
  "import ast,glob; [ast.parse(open(p,'rb').read(), filename=p, feature_version=(3,6)) for p in glob.glob('tests/*.py')]"
& "C:\personal_apps\anaconda3\python.exe" -m unittest -v `
  tests.test_prepare_mnist tests.test_data_interop `
  tests.test_check_prohibited tests.test_check_comments `
  tests.test_documentation tests.output_format_tests tests.compliance_tests
& "C:\personal_apps\anaconda3\python.exe" tests/generate_repro_vectors.py `
  --output "C:\Users\26030\AppData\Local\Temp\opencode\repro-vectors-comment-pass.txt"
& "C:\personal_apps\anaconda3\python.exe" -c `
  "from pathlib import Path; assert Path(r'C:\Users\26030\AppData\Local\Temp\opencode\repro-vectors-comment-pass.txt').read_bytes() == Path('tests/repro_vectors.txt').read_bytes()"
& "C:\personal_apps\msys64\usr\bin\make.exe" python-tests compliance `
  "PYTHON=C:/personal_apps/anaconda3/python.exe" `
  "BASH=C:/personal_apps/msys64/usr/bin/bash.exe"
```

Expected: only comments added, all Python 3.6 grammar/byte-compilation/tests pass,
the generated vectors are byte-identical, and compliance remains green.

- [ ] **Step 6: Commit Batch 5**

```powershell
git -c user.name=OpenCode -c user.email=opencode@localhost commit -m "docs: explain Python test coverage"
```

---

### Task 6: Aggregate Preservation And Review

**Files:**
- Verify only: every included code file from Tasks 1-5
- Write execution evidence only under `.superpowers/sdd/`, whose `.gitignore` ignores task artifacts

**Interfaces:**
- Consumes: Five reviewed comment-only commits based on `c693370`.
- Produces: Whole-range proof that comments are accurate and executable behavior is unchanged.

- [ ] **Step 1: Run whole-range comment-only proof**

Run the scope and all-file verifier modes with base `c693370`:

```powershell
python "C:\Users\26030\AppData\Local\Temp\opencode\cuda_lenet_verify_comment_only.py" `
  "E:\cuda_lenet_mnist\.worktrees\rebuild" --scope
python "C:\Users\26030\AppData\Local\Temp\opencode\cuda_lenet_verify_comment_only.py" `
  "E:\cuda_lenet_mnist\.worktrees\rebuild" c693370 --all
git diff --check c693370..HEAD
git diff --stat c693370..HEAD
git status --short --branch
```

Expected: the only project-code changes after the baseline are standalone
comment additions in pre-existing included files; the committed design and plan
baseline is unchanged and the plan is the only later documentation change;
worktree is clean.

- [ ] **Step 2: Run fresh aggregate local verification**

```powershell
$env:PATH = "C:\personal_apps\msys64\usr\bin;C:\personal_apps\msys64\ucrt64\bin;" + $env:PATH
$env:BASH = "C:\personal_apps\msys64\usr\bin\bash.exe"
& "C:\personal_apps\msys64\usr\bin\make.exe" `
  host-tests makefile-tests python-tests compliance `
  "PYTHON=C:/personal_apps/anaconda3/python.exe" "CXX=g++" `
  "BASH=C:/personal_apps/msys64/usr/bin/bash.exe"
& "C:\personal_apps\msys64\usr\bin\bash.exe" -n scripts/check_prohibited.sh
& "C:\personal_apps\msys64\usr\bin\bash.exe" -n tests/makefile_tests.sh
```

Expected: nine host suites, Make regression, 15 data tests, complete compliance
suite, 147/147 comment inventory, and source policy pass.

- [ ] **Step 3: Run compatibility and command-generation checks**

Byte-compile every changed Python file, parse each with Python 3.6 grammar, and
run the Make command-generation checks:

```powershell
& "C:\personal_apps\anaconda3\python.exe" -m py_compile `
  scripts/prepare_mnist.py scripts/check_comments.py scripts/analyze_build_graph.py `
  tests/compliance_tests.py tests/generate_repro_vectors.py `
  tests/output_format_tests.py tests/test_check_comments.py `
  tests/test_check_prohibited.py tests/test_data_interop.py `
  tests/test_documentation.py tests/test_prepare_mnist.py
& "C:\personal_apps\anaconda3\python.exe" -c `
  "import ast; files=['scripts/prepare_mnist.py','scripts/check_comments.py','scripts/analyze_build_graph.py','tests/compliance_tests.py','tests/generate_repro_vectors.py','tests/output_format_tests.py','tests/test_check_comments.py','tests/test_check_prohibited.py','tests/test_data_interop.py','tests/test_documentation.py','tests/test_prepare_mnist.py']; [ast.parse(open(p,'rb').read(), filename=p, feature_version=(3,6)) for p in files]"
& "C:\personal_apps\msys64\usr\bin\make.exe" -B -n V=1 all
& "C:\personal_apps\msys64\usr\bin\make.exe" -B -n V=1 cuda-tests
```

Expected: both exact `sm_90` and `compute_90` code-generation targets remain in
the dry-run output. Record that commands were not executed.

- [ ] **Step 4: Complete final independent review**

Generate a review package for `c693370..HEAD`. The reviewer must verify:

- every changed code line is a standalone comment addition;
- every comment accurately describes unchanged code;
- no prohibited vocabulary, stale claim, or misleading algorithm description;
- every included file has appropriate structure/key-logic documentation;
- both known bugs remain unmodified and are reported only;
- Task 14 target-only boundaries remain explicit.

Accept the work only with no Critical or Important finding. Fix inaccurate
comments by changing only newly added comment lines, re-run all relevant gates,
and obtain a scoped re-review.

## Completion Record

At completion, report the five batch commit SHAs, whole-range verifier result,
fresh test counts, review verdict, unchanged known-bug locations, deferred H20
verification, and remote publication status. Do not claim the semantic comment
checklist was manually reviewed.
