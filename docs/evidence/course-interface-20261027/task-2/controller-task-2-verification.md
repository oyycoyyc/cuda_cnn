# Task2 controller acceptance evidence

## Tests-first RED

H20 NVCC12.6.85, C++14/O2/sm90: test TU compiles without diagnostics. Existing reference symbols resolve; linker reports only missing new course helper/model methods. Logs task-2-red-compile.txt / task-2-red-link.txt.

## Initial real implementation acceptance

- CMake3.22.1 tests-ON explicit90 configure passes. All new GPU sources compile with real NVCC; course_io_tests and course_gpu_tests link.
- `ctest -R 'course_(io|gpu)' -V --output-on-failure`: host48 executed/failures0; GPU7 executed/failures0. Log task-2-first-ctest.txt. This exercises mode formulas/translation/finite sticky/ties, bitwise old-vs-new logits at1/17/1024, repeat/tails, allocation counts, smaller arena, rejection tests.
- Combined build requested course_model_bundle too and returns2 at its final link. `libcourse_checkpoint.a` has undefined ValidateLenetParameters/LenetParameterSpecs/CreateLenetParameters. Log task-2-cmake-build.txt. Source CMake declares no checkpoint-to-parameters edge; existing io test references parameter symbols itself and happens to conceal this dependency-order gap. No full-build success claimed.
- Main inspected new implementation and noted a concurrency concern for reviewer: ArgmaxKernel has potentially many row threads storing to the same plain `*bad_output` address without atomics. Current numerical tests poison one row, so their PASS does not establish race freedom. Determine severity independently.

Historical pending list at initial dispatch (superseded below): fresh default/OFF config, caller arch override, final link/dependency audit, complete optional tool build, protected hashes after fixes. This task is not complete/checkpointed.

## Default/override isolation acceptance

Main created an exclusive28-file minimal copy under experiment artifacts/task-2-minimal-source. It omits all course tests/tools and original training backend sources, retaining only actual default dependencies. Fresh tests-OFF default configure/build succeeds. Exported10 translation-unit commands prove O2/C++14/no fast-math and compute90/sm90 on every CUDA unit. A separate fresh caller-override86 configure exports compute86/sm86, unchanged by default logic. No old workflow/reference training TU appears. Logs task-2-default-configure.txt/default-build.txt/override-configure.txt/build-audit.txt.

Remaining gates are the known optional-tool link fix, concurrency review/fix as required, final revalidation/review/checkpoint. This minimal archive is a development acceptance fixture, not a final course submission.

## Review/fix round1

Independent reviewer agrees on2 blocking items and otherwise approves numerical/arena design. Original implementer resumed with exact findings. Many-poisoned-row coverage will exercise concurrent flag setters but is not a deterministic race detector; prior plain writes can produce the right value while still violating the memory model. Atomic store instruction review plus real tests will establish the fix without a false old-code-numerical-failure claim. Existing Compute Sanitizer self-disables on this platform, so no sanitizer result will be asserted.

Original project uses NVCC `-O2`; this is the baseline flag being preserved. Exported compile commands establish the flag and absence of fast-math, not a claim about a separate ptxas optimization level. No optimization flag tuning is included in the scoped correction.

## Fix round1 real H20 results

- Full fresh tests-ON CMake directory `build-course-task2-final`, explicit90: default all-target verbose build succeeds with zero compiler warnings, including formerly failing course_model_bundle. Link now ends `libcourse_host.a libcourse_checkpoint.a libcourse_parameters.a`.
- CTest verbose: host48/48, GPU8/8, zero failures. The added case poisons128 rows across four blocks then executes a finite launch without clearing the flag.
- cuobjdump generated PTX contains `atom.global.exch.b32 %r39, [%rd20], 1;` for the new argmax helper; no sanitizer result inferred. This verifies code generation of the source atomic, not a timing measurement.
- Fresh model-bundle tool smoke test silently succeeds; independent zlib CRC and unchanged original checkpoint SHA256 pass.
- Evidence: task-2-fix-1-configure.txt/build.txt/ctest.txt/audit.txt; fixed exported commands/link in task-2-fixed-compile-commands.json / task-2-fixed-model-bundle-link.txt. PTX full output is retained remotely in logs/task-2-fix-1-ptx.txt. Scoped diff148lines is task-2-fix-1.diff.
- Original implementer's fix report lists covering tests and local static commands honestly; main supplies actual NVCC/CMake/CTest evidence. Scoped re-review remains required.

## Final acceptance after scoped approval

Re-review approves both spec and quality after one fix round. Main post-fix fresh minimal default90 build/override86 configuration also pass; final audit logs and13-file source hashes are in task-2-final-build-audit.txt / task-2-final-verification.json. Old67 local sources, reviewed7 Task1 sources, protected8 remote artifacts/configs remain unchanged. Kernel-specific PTX extract task-2-argmax-ptx.txt confirms atomic exchange (the initial extraction script was corrected to support internal `.entry` without `.visible`; production code never failed this audit). This is functional/numerical acceptance, not startup-inclusive timing or final course submission acceptance.
