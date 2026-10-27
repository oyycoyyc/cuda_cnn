# Task 1 tests-only handoff — report

Status: HANDOFF READY / NEEDS_CONTEXT. No production sources were created; `course/io.cpp`, `course/model_io.cpp` and `course/tools/model_bundle.cpp` remain absent by design.

## Files created (all new)
- `course/interface.h` — declaration-only InputMode, Shape, TrainOptions, InferOptions, ParseInputMode, ParseTrainArgs, ParseInferArgs, ReadImages, ReadLabels.
- `course/model_io.h` — declaration-only CourseModel, LoadCourseModel, SaveCourseModel, plus the pinned four-line metadata/CRC32/256-byte publication contract.
- `course/tests/test_support.h` — dependency-free harness, build-owned `<exe dir>/course_io_scratch` fixtures, explicit `cases=/executed=/failures=` counts, zero-selected-case failure guard.
- `course/tests/io_tests.cpp` — 41 named cases.

## Coverage
Exact argc 9/7; defaults unit/20/128/1337/1024; empty/unknown mode+epochs and empty model dir; signed/zero/N=1 overflow (4294967296); fixed C/H/W/K; epochs 0/1001; batch 0/16385; profile exact "1" only. Real fixtures cover bitwise float preservation, unit/byte bounds, finite negative normalized values, NaN/Inf, short/long files, and oversized declarations up to UINT32_MAX N against a 1-float file (bad_alloc is a test failure). Labels -1/10/11/256/int32 extremes rejected before narrowing. Bundle tests use real `CreateLenetParameters`/SaveCheckpoint fixtures: unit/normalized/byte round-trips, exact metadata bytes + zlib-compatible CRC32 (`cbf43926`), structural/oversized/CRLF/uppercase/wrong-CRC metadata, mixed checkpoint/metadata rejection, failed-save preservation of the prior bundle, extra-artifact refusal, and symlink-alias non-mutation (POSIX).

## Local evidence (Windows/MinGW g++ 14.2.0; not a Linux/CUDA claim)
- Headers standalone `-fsyntax-only`: pass.
- `g++ -std=c++14 -O2 -Wall -Wextra -Wpedantic -Iinclude -I. -c course/tests/io_tests.cpp`: pass, no warnings.
- Link with checkpoint/parameters/random + empty stub: RED, undefined references only to the seven course functions above.

## Expected RED for main (Linux)
1. Brief's exact command fails because `course/io.cpp` and `course/model_io.cpp` do not exist yet; after empty stubs are added, the link fails on those seven symbols. Tests must compile before GREEN.
2. Scratch is `<build-course-host>/course_io_scratch`, never `/tmp` or data/output dirs; symlink cases run on POSIX and are skipped only on non-POSIX local builds.

## GREEN implementation (appended; supersedes tests-only status above)

Status: DONE host-side implementation. Main Linux/H20 host suite and original regressions remain authoritative; no CUDA success is claimed.

### Corrections applied
- Harness now exclusively creates a unique per-run root via POSIX `mkdtemp("<exe dir>/course_io_scratch.XXXXXX")` before any test; only that returned root is recursively removed. No startup deletion of a pre-existing fixed path. Windows keeps a minimal pid/sequence fallback.
- `model_io.h` now promises only: previous entries preserved on failures before publication, or new-checkpoint + old-metadata mixed pair rejected by the loader via CRC. Two-file publication is explicitly not a transaction.
- `SaveCourseModel` preserves unrelated files and directories in the caller-owned work directory. It validates only reserved `lenet.bin`/`course.meta` entries, refusing symlinks, hard-link aliases, non-regular entries, and other malformed reserved artifacts before writes. Temporary names are checked absent before use.
- `course/tools/model_bundle.cpp` still requires a fresh/empty output directory; it never writes the source checkpoint and prints diagnostics only on stderr.

### Implemented files
- `course/io.cpp`: strict parsers/readers. argc 9/7, defaults unit/20/128/1337/1024, exact mode/epoch/batch/N/dimension validation; file size checked before vectors; bit-preserving FP32 reads; unit/byte bounds and finite checks; int32 labels validated before uint8 narrowing.
- `course/model_io.cpp`: four-line metadata, 256-byte cap, lowercase 8-hex IEEE CRC32 over checkpoint bytes, unit/byte/normalized modes, same-directory checkpoint then metadata rename with metadata last, reserved-name alias/malformed refusal, unrelated-entry preservation.
- `course/tools/model_bundle.cpp`: argc4 `checkpoint.bin output_dir input_mode`, explicit ParseInputMode, original LoadCheckpoint, fresh-output enforcement; exit 2 usage, 1 runtime, 0 success.

### Test changes
- Replaced extra-artifact refusal with `CourseModel_preserves_unrelated_work_directory_entries`: saves/loads beside `notes.txt` and `keepdir/inside.txt`, verifying both remain byte-identical.
- Added `CourseModel_refuses_malformed_reserved_artifacts` (directory at `lenet.bin` and at `course.meta`), retaining symlink and hard-link alias refusal tests.
- Suite now has 43 named cases; executed count is always nonzero and printed.

### Local evidence (Windows/MinGW g++ 14.2.0; POSIX branches not exercised)
- `g++ -std=c++14 -O2 -Wall -Wextra -Wpedantic -Iinclude -I. course/tests/io_tests.cpp course/io.cpp course/model_io.cpp src/checkpoint.cpp src/parameters.cpp src/random.cpp -o <tmp>/course_io_tests.exe` then run: `cases=43 executed=43 failures=0 status=pass`, exit 0. Symlink/hard-link cases early-return only on non-POSIX; Linux executes them.
- Tool compiled with the same host deps; real generated checkpoint -> success silent, exactly `lenet.bin` (178344 B) + `course.meta` (75 B), source SHA256 unchanged, metadata CRC verified by the loader; nonempty output refused (exit 1), bad mode exit 2.
- Original RED preserved: `task-1-red-compile.txt` (empty stdout, exit 0) and `task-1-red-link.txt` (only the seven missing course symbols), from main's Linux run before this implementation.

### Concerns / limits
- Linux-only paths (`mkdtemp`, `lstat`, symlink/hard-link fixtures) were not runnable locally; main must confirm them on Linux.
- Existing `SaveCheckpoint` chooses its own inner temporary name inside its implementation; an adversarial unrelated file with that exact pid/sequence name remains a theoretical collision, outside this task's editable files.
- No CMake/CTest registration was added (Task2 owns it). No installs, SSH, staging, commits, or edits to old production sources/tests/Makefile were made.

## Fix round 1 (appended)

Status: DONE host-side. Resolved review findings 1-3; deferred minors untouched except the mixed-pair wording.

### Changes
- `course/model_io.cpp`: `SaveCourseModel` now validates an existing reserved pair before any temporary/destination write. Both files must be present, regular, single-link; `course.meta` must parse exactly; its CRC must match the existing `lenet.bin`; and the existing checkpoint must load. Lone, junk, mixed, symlink, hard-link, and non-regular reserved artifacts are refused with `std::runtime_error` and left unchanged. Unrelated `/work` entries remain untouched.
- `course/io.cpp`: corrupt image/label data now throws `std::runtime_error` (removed `ValueFail`); argv/env/shape syntax stays `std::invalid_argument`. Added a per-read scalar contract guard: 32-bit IEEE-754 binary32 (`is_iec559`, radix 2, 24 digits, max_exponent 128), 32-bit signed exact int32, and runtime little-endian verification. Unsupported hosts are rejected, not decoded by a guessed fallback.
- `course/interface.h`, `course/model_io.h`: exception taxonomy and existing-pair recognition documented; mixed-pair wording now says differing-byte mixes.
- `course/tests/test_support.h`: added typed `EXPECT_THROWS_AS`.
- `course/tests/io_tests.cpp`: added typed taxonomy tests, explicit little-endian fixture decode test, lone/junk reserved-file preservation tests, and CRC-matching-junk-checkpoint plus valid-checkpoint+junk-metadata cases. Suite is now 48 cases.

### Covering tests
`ErrorTaxonomy_usage_and_shape_errors_throw_invalid_argument`, `ErrorTaxonomy_corrupt_file_data_throws_runtime_error`, `ReadData_decodes_known_little_endian_fixtures`, `CourseModel_rejects_lone_regular_reserved_artifacts`, `CourseModel_rejects_junk_regular_reserved_pairs`, plus retained valid-pair replacement, unrelated-entry preservation, symlink/hard-link refusal, mixed-publication, and failed-save-preservation cases.

### RED (tests added before fixes; current-at-the-time implementation)
Command (Windows/MinGW g++ 14.2.0; same sources as the brief):
`g++ -std=c++14 -O2 -Wall -Wextra -Wpedantic -Iinclude -I. course/tests/io_tests.cpp course/io.cpp course/model_io.cpp src/checkpoint.cpp src/parameters.cpp src/random.cpp -o %TEMP%\course-host-build\course_io_tests.exe`
then `%TEMP%\course-host-build\course_io_tests.exe`
Output (exit 1):
```
test failure: ErrorTaxonomy_corrupt_file_data_throws_runtime_error: course/tests/io_tests.cpp:469: EXPECT_THROWS_AS(course::ReadImages(out_of_range_path, UnitShape(1), course::InputMode::kUnit), std::runtime_error) threw a different std::exception type
test failure: CourseModel_rejects_lone_regular_reserved_artifacts: course/tests/io_tests.cpp:1008: EXPECT_THROWS_AS(course::SaveCourseModel(checkpoint_directory, SampleCheckpoint(), course::InputMode::kUnit), std::runtime_error) did not throw
test failure: CourseModel_rejects_junk_regular_reserved_pairs: course/tests/io_tests.cpp:1035: EXPECT_THROWS_AS(course::SaveCourseModel(junk_pair_directory, SampleCheckpoint(), course::InputMode::kUnit), std::runtime_error) did not throw
event=test_suite name=course_io_tests cases=48 executed=48 failures=3 status=fail
```

### GREEN (after fixes)
Same compile command: exit 0, no warnings. Run output (exit 0):
```
event=test_suite name=course_io_tests cases=48 executed=48 failures=0 status=pass
```
Tool rebuilt with the same host deps and smoke-tested: silent success on a fresh dir, loader returns mode and 10 tensors, nonempty output refused with exit 1.

### Concerns
- Local host is Windows; `mkdtemp`, `lstat`, symlink/hard-link executes, and the unsupported-architecture branch are Linux-only or negative branches and remain for main verification.
- A mixed pair created by a mid-publication failure is now refused by a later `SaveCourseModel` as unrecognized; recovery requires explicitly removing the reserved names. No automatic deletion/rollback was added.
- No CMake, commits, staging, installs, SSH, or old-source edits.
