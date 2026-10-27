# Task 1 fix-round 1 review — scoped to the three prior findings

Scope: `task-1-fix-1.diff` (775 lines) versus the frozen
`task-1-review-snapshot`, applied to the current six Task1 files only
(`interface.h`, `io.cpp`, `model_io.h`, `model_io.cpp`,
`tests/test_support.h`, `tests/io_tests.cpp`; the bundle tool is untouched).
I read the updated brief, the appended fix report, the controller verification
appendix, the raw RED/GREEN/controller evidence files, the full fix diff, and
verified the changed regions in the current sources. I ran no builds, suites,
tools, SSH, or git commands. Main's Linux evidence is consistent with the test
code but is not used in place of the static judgments below.

## Verdicts on the original findings

1. **ADDRESSED — unrecognized regular reserved artifacts.**
   `SaveCourseModel` now calls `ValidateExistingBundle` before choosing or
   creating any temporary/destination path (`course/model_io.cpp:332-358`,
   call at `:382`). The helper returns both-name existence from the
   alias/type-checked directory scan (`:87-121`), refuses a lone reserved name,
   then requires exact metadata parse, a CRC match against the *existing*
   `lenet.bin`, and a successful original `LoadCheckpoint` before replacement
   is allowed. It therefore allows only a recognized valid pair and refuses
   lone, junk, mixed, symlinked, hard-linked, and non-regular artifacts without
   writes; unrelated entries remain untouched.
   Tests cover a lone junk `lenet.bin` with an unrelated note, a lone junk
   `course.meta`, a junk pair, a CRC-matching-but-unloadable junk checkpoint
   (forces the loader check), and a valid checkpoint with junk metadata, all
   asserting bytes are unchanged (`course/tests/io_tests.cpp:1002,1030`);
   retained tests still prove valid-pair replacement and unrelated-entry
   preservation (`:832,966`). Main's RED shows the lone and junk-pair cases
   failed against the frozen old implementation and pass against the fix,
   matching the code.

2. **ADDRESSED — supported scalar representation/byte order.**
   `ValidateScalarContract` (`course/io.cpp:97-124`) checks 4-byte IEEE-754
   binary32 (`is_iec559`, radix 2, 24 digits, max_exponent 128), 4-byte signed
   exact int32 with 31 value digits, and verifies little-endian storage with a
   `0x01020304` byte marker; any mismatch throws `runtime_error` instead of
   decoding with a fallback. It runs first in both readers, before shape,
   extent, file access, or allocation (`:268,311`). The contract is documented
   at `course/interface.h:100-106,110-116`, and the supported decode is pinned
   by an explicit little-endian byte fixture test
   (`course/tests/io_tests.cpp:627`). On the actual Linux/x86-64 H20 target the
   guard passes, so behavior is unchanged there. The rejection branch cannot
   be exercised on a supported host; that is an inherent verification limit,
   not an implementation gap. Nonblocking nuance: the guard cannot distinguish
   one's-complement int32 although the new doc says two's-complement; moot on
   the target (and C++20+), but a static assertion could tighten it later.

3. **ADDRESSED — exception taxonomy.**
   `ValueFail` is gone; only `ArgumentFail` → `std::invalid_argument`
   (`course/io.cpp:37-39`) and `FileFail` → `std::runtime_error` (`:41-43`)
   remain. Non-finite/out-of-mode-range image values and out-of-range labels
   now throw `runtime_error` (`:139,159,341`). The interface preamble
   (`course/interface.h:8-12`) is now consistent with the per-function
   contracts (`:95-116`): invalid argv/env/shape/path/mode stays
   `invalid_argument`; unreadable, short/long, non-finite, or out-of-range
   payloads are `runtime_error`. `EXPECT_THROWS_AS`
   (`course/tests/test_support.h:502`) pins both sides in
   `course/tests/io_tests.cpp:409` (usage/shape → invalid_argument) and `:456`
   (corrupt data, labels, model files, broken save → runtime_error). Main's RED
   shows the old implementation failed the corrupt-data case on the thrown
   type, and GREEN passes with the fix; the code matches.

## New breakage introduced by the fix diff

**None Critical or Important.** I checked the modified write path end to end:
validation happens before any temporary name is chosen/created; a fresh or
empty directory is unaffected (tool flow unchanged); a valid previous bundle is
still replaceable; invalid checkpoints and I/O failures still leave previous
entries intact; the scalar guard adds no new failure on the target. Concrete
nonblocking consequences worth recording:
- A mixed pair left by a mid-publication failure is now refused by the next
  `SaveCourseModel` as unrecognized, so retry requires manually removing the
  two reserved names. This is stricter than before but consistent with the
  brief's reject-or-preserve guarantee; the report documents it.
- Recognizing an existing pair computes the CRC of the existing `lenet.bin`
  before `LoadCheckpoint`; an unrelated full-size junk reserved file is
  therefore read once before refusal. Bounded, caller-owned name; no data loss.
- `EXPECT_THROWS_AS`'s comment says "exactly" but accepts derived types; a nit,
  not a behavioral issue.

## Out-of-scope observations (untouched, nonblocking, not re-adjudicated)

- `main()` still lives in the test header (`course/tests/test_support.h:530`);
  a second test TU including it would collide under Task 2.
- Early-return POSIX-only cases are still counted as executed
  (`course/tests/io_tests.cpp:722,1077,1097`; runner at
  `test_support.h:398-410`).
- Metadata 256-byte boundary is still untested; only a 300+ byte file is
  rejected (`course/tests/io_tests.cpp:899`).
- Windows `InspectArtifact` still uses `stat` (follows links) while POSIX uses
  `lstat`; non-target only.
- `SaveCourseModel` still requires the destination directory to exist, which
  the public comment does not state (`course/model_io.h:42-53`); callers/tool
  pre-create it.
- Temporary-name choice and pair validation remain TOCTOU-prone against
  concurrent writers; the brief disclaims transactional isolation.

## Cannot verify

- I did not execute the 48-case suite, the builds, the RED/GREEN runs, the
  Linux POSIX alias branches, or the bundle tool. The unsupported-endian /
  unsupported-scalar rejection branch is not exercised on this target.
- Crash/power-loss durability and real concurrent readers/writers are not
  verifiable here and are not claimed.

## Scoped verdict

All three original findings are **ADDRESSED**, with no new Critical or
Important breakage from the fix diff.

- **Spec compliance: APPROVED.**
- **Task quality: APPROVED.**

Remaining items are the pre-existing nonblocking minors above. Completion,
backup/checkpoint, and Task 2 dispatch remain main's decisions.
