# Main-agent Task1 verification

- Linux tests-only compilation passed, link failed only on7 missing course APIs before implementation: task-1-red-compile.txt / task-1-red-link.txt.
- Current Task1 host implementation compiled on H20/Linux GCC11.4 with C++14/O2/Wall/Wextra/Wpedantic with empty warning log: task-1-green-build.txt.
- All43 named host cases execute/pass on Linux (POSIX alias branches actually exercised), plus two concurrent13-case model subsets pass. Existing fixed-name scratch sentinel remains unchanged after tests; exclusive per-run scratch resolves ownership/concurrency concern.
- Development model_bundle generated an exact2-file unit bundle from read-only original checkpoint; independent Python zlib agrees with CRC, source/destination SHA256 equal and source unchanged. Success stdout/stderr empty; nonempty destination/unknown mode/argc errors produce expected1/2/2 exits with no stdout.
- Existing isolated legacy suite:9 host suites, full CUDA operators and evaluate/profile passed. Legacy compliance198 OK/skipped3; successful-build gate passed. Initially archived checklist had147 items; synced current reviewed checklist and verified151/151.67 existing local/remote source files match after newline normalization. No claim that old Make scanners audit new course/ sources.

Main review concerns to adjudicate against code/brief:
- model_io.cpp InspectReservedArtifacts currently validates inode type/link-count only, not the content of an existing regular reserved filename. Could Save overwrite unrelated regular garbage at lenet.bin/course.meta despite the plan requiring refusal of unrecognized existing reserved artifacts? Added test only covers a directory, so current43-pass evidence does not settle that case.
- io.cpp documents raw little-endian FP32 but lacks the plan's explicit supported-scalar-representation/endian checks; determine consequence on actual Linux/H20 target vs claimed contract.
- io.cpp ValueFail uses invalid_argument for bad file values while its introductory comment classifies file-content errors as runtime_error; later CLI must not accidentally classify malformed file contents as usage if runtime exit1 is intended.

These concerns are not pre-adjudicated review verdicts; reviewer must inspect actual diff and report severity. Main handles any fixes through original implementer and test-first evidence, never rewrites production code itself.

## Fix round1 main verification

- Reviewer confirmed the3 open issues; original implementer added5 cases then minimally fixed them. Full scoped fix diff: task-1-fix-1.diff (775lines) against the frozen7-file task-1-review-snapshot, not an unrelated HEAD diff.
- Main independently rebuilt NEW48 tests against OLD frozen io/model_io implementations on Linux: exactly3 expected failures (typed corrupt-data error, lone reserved regular artifact, junk reserved pair),48 executed. task-1-fix-1-red.txt.
- Same new tests against fixed sources:48 executed,0 failures, warning-free GCC11.4 C++14 build. task-1-fix-1-green.txt / task-1-fix-1-build.txt.
- Fixed bundle utility still silently succeeds on a fresh diagnostic destination, output CRC independently matches zlib, source SHA unchanged; pre-existing unrelated sentinel remains intact.
- Main inspected entire fix diff: reserved replacement now requires a complete recognized pair plus matching CRC and valid checkpoint; raw representation guard precedes reads; corrupt payload errors are runtime_error. Unsupported-architecture rejection cannot be executed on this actual supported host and is not claimed empirically exercised.
- Scoped re-review remains required before completion/checkpoint/Task2 dispatch.
