# Task 1 review — strict host interface, data contracts, model handoff

Reviewer scope: task-scoped new uncommitted files vs recorded base `bf6aae8`
(`course/interface.h`, `course/model_io.h`, `course/io.cpp`,
`course/model_io.cpp`, `course/tools/model_bundle.cpp`,
`course/tests/test_support.h`, `course/tests/io_tests.cpp`). I read the full
2474-line review package once in sections and inspected only a few unchanged
files (`src/checkpoint.cpp`, `src/parameters.cpp`, `include/checkpoint.h`,
`include/parameters.h`) to settle the named checkpoint-shape risk. I did not
run the suite, execute the tool, perform any git operation, or edit outside
this report. Local host is Windows; main's Linux/H20 run is treated as evidence,
not proof of the code judgments below.

## Verdicts

- **Spec compliance: NEEDS FIXES** — two explicit brief requirements are unmet:
  refusal of *unrecognized existing reserved artifacts* (brief:24) and
  validation of the *supported scalar representation* (brief:22). Both are
  code-contract gaps; the first is destructive.
- **Task quality: NEEDS FIXES** — otherwise strong implementation and genuinely
  behavioral tests, but the same destructive-overwrite defect plus an
  inconsistent/unpinned exception taxonomy and a few small test/harness gaps.

## Strengths

- Strict parsing matches the required argc 9/7, defaults unit/20/128/1337/1024,
  exact mode/epoch/batch spellings, N/dimension rules, and training-only N>=2,
  with no infer labels/input-mode path (`course/io.cpp:180-241`).
- Extents are checked and the exact file size is confirmed before any result
  vector is allocated; oversized declarations fail on size, and `bad_alloc` is
  treated as a test failure (`course/io.cpp:229-279`, `test_support.h:475-499`).
- Images are raw FP32 reads (no uint8 quantization); tests compare bit patterns
  including `-0.0`, denormals, and non-representable fractions; labels are
  validated in int32 before narrowing (`course/io.cpp:266-322`,
  `io_tests.cpp:412-495,598-646`).
- Metadata is exactly four LF lines, 256-byte capped, eight lowercase hex
  digits, with a correct reflected 0xedb88320 CRC32 (zlib vector `cbf43926`),
  bound to the actual published checkpoint bytes; extra/reordered/CFLF/short/
  uppercase metadata, mixed pairs, and missing artifacts are rejected
  (`course/model_io.cpp:167-317`, `io_tests.cpp:648-891`).
- Publication uses same-directory temporaries with metadata renamed last;
  pre-publication failures leave prior entries intact and the post-checkpoint
  failure leaves a mixed pair rejected by CRC, matching the agreed
  old-preserved-or-rejected guarantee rather than claiming a transaction
  (`course/model_io.cpp:339-373`, `model_io.h:50-59`).
- Symlink and hard-link destinations are refused before writes, unrelated
  caller files/directories are preserved, and the tool refuses a nonempty
  output, never mutates its source, is silent on stdout, and uses exit 2/1/0
  (`course/model_io.cpp:62-116,339-373`, `tools/model_bundle.cpp:73-91`).
- Test fixtures are real (actual raw files, real `CreateLenetParameters`/
  checkpoint helpers, independent CRC routine), the suite has 43 named cases,
  and scratch is an exclusively created per-run `mkdtemp` root that is the only
  tree removed (`test_support.h:351-380,502-558`).
- No CUDA calls, no CPU NN math, no scope creep; the report clearly separates
  Windows-host evidence from Linux/H20 authority.

## Critical findings

1. **Unrecognized regular reserved artifacts are silently overwritten**
   (`course/model_io.cpp:64-84,89-116,339-343`; contract stated in
   `course/model_io.h:45-47`). `InspectArtifact` checks only inode type and
   `st_nlink`; any regular single-link file at `lenet.bin`/`course.meta` is
   treated as replaceable regardless of content. The brief (line 24) and the
   user ruling require refusing *unrecognized existing reserved* artifacts
   before writes while preserving unrelated files. Repro: a caller-owned
   directory containing an unrelated regular `lenet.bin` (junk bytes, no
   `course.meta`), or a lone/stale `course.meta`, is overwritten by
   `SaveCourseModel` instead of refused. Course `/work` is explicitly not
   guaranteed clean, so this is a data-destruction path, not a theoretical
   race. The only malformed-artifact test uses a *directory* at the reserved
   name (`io_tests.cpp:832-849`), so the 43-pass result does not cover it.
   Fix: when either reserved name exists, recognize the existing pair first
   (both regular single-link files, metadata parses exactly, CRC over the
   existing `lenet.bin` matches) and allow replacement only then; otherwise
   throw `runtime_error` before creating temporaries. Add tests for junk
   regular `lenet.bin`, lone `course.meta`, valid metadata + junk checkpoint,
   and valid checkpoint + junk metadata.

## Important findings

2. **Supported scalar representation is asserted only by comment, never
   validated** (`course/io.cpp:15-16,257-279,289-307`; the "little-endian
   x86" claim is at 268-269). The brief (line 22) requires validating the
   supported scalar representation before vectors. There is no check that
   `sizeof(float)==4`, IEEE-754 binary32 (`is_iec559`, `FLT_RADIX==2`), or
   little-endian host order, and labels are read with the same host-order
   assumption. On the actual Linux/x86-64 H20 target the bytes are correct, so
   this is not an observed corruption; on a big-endian or non-IEEE host the
   size checks still pass and values/labels are silently misinterpreted.
   Fix (cheap, C++14): compile-time assertions for float size/format plus a
   one-time host byte-order check that fails with a clear runtime error, or
   explicitly declare other platforms unsupported and reject them. Document
   that the target is little-endian x86-64.

3. **Exception taxonomy contradicts itself and is not pinned by tests**
   (`course/interface.h:11` vs `:95-112`; `course/io.cpp:15-16,47-49`;
   `test_support.h:475-499`). The header preamble says file *content*
   violations throw `std::runtime_error`, but out-of-range/NaN image values
   and out-of-range labels throw `std::invalid_argument` via `ValueFail`.
   Because the eventual CLI will map exception classes to exit 2 vs 1
   (global constraints), corrupt data could be reported as a usage error.
   `EXPECT_THROWS` accepts any `std::exception`, so neither contract is
   tested. Fix: choose one taxonomy — recommended: valid argv + corrupt file
   content = `std::runtime_error` (runtime exit 1) — align the header text,
   `ValueFail`, and add typed assertions (e.g. `EXPECT_THROWS_AS`) for
   parse/usage vs file-content failures.

## Minor findings

4. `course/tests/test_support.h:502` defines `main()` inside a header. It works
   only while exactly one test translation unit includes it; Task 2 adding
   another suite that includes `test_support.h` would produce duplicate
   `main`/registry-link errors. Prefer a small runner `.cpp` or a guard.
5. Skipped POSIX-only cases return early (`io_tests.cpp:571-573,852-854,
   872-874`) but `Run` counts them as executed (`test_support.h:398-410`), so
   on non-POSIX hosts `executed=43` overstates coverage. Linux runs them, so
   this is a reporting-honesty issue only; count skips separately.
6. Metadata size boundary is untested: only a 300+ byte file is rejected
   (`io_tests.cpp:748`); exactly 256 accepted and 257 rejected are not
   asserted, although the implementation's `> 256` check is correct.
7. `model_io.h:52-58` overstates mixed-pair detection: if the new checkpoint
   bytes equal the previous ones (e.g. re-publishing the same checkpoint to
   change the mode) and the metadata rename fails, the identical CRC lets the
   loader accept the previous mode. That still satisfies the allowed
   "previous valid bundle preserved" outcome and the save call throws, but the
   comment should say the CRC detects only differing-byte mixes.

## Cannot verify

- Execution on Linux/H20: I did not run the 43 cases, the concurrency/tool
  checks, or any build; POSIX `mkdtemp`/`lstat`/symlink/hard-link branches were
  reviewed statically only. Main's recorded Linux evidence is accepted as
  evidence, not substituted for the code judgments above.
- Crash/power-loss durability (no fsync of files/directory before rename) and
  real concurrent writers/readers; the brief disclaims transactional
  isolation, and I make no stronger claim.
- Non-x86/big-endian or non-IEEE hosts: behavior is unvalidated (finding 2).
  Windows branches exist but are not target (`stat` follows symlinks in
  `model_io.cpp:66-69`, `_mkdir`/`MoveFileExA`); they are not acceptance
  evidence.
- Whether Task 2/3 CLIs will classify exceptions as intended; that depends on
  resolving finding 3.
