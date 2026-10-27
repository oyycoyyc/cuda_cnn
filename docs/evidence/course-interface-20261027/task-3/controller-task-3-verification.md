# Main-agent Task3 evidence

- Tests-first H20 NVCC14/O2/sm90 TU compile warning-free; linking all reviewed dependencies plus original random object fails only on new5 training API families. task-3-red-compile.txt / task-3-red-link.txt.
- New actual sources/CMake uploaded only to owned experiment. Fresh tests-ON explicit90 verbose all-target build succeeds, no compiler warnings. task-3-configure.txt / task-3-build.txt.
- CTest verbose all3 suites pass: host48, GPU8, training8 executed, no failures. task-3-ctest.txt.
- Full existing CUDA operator suite (includes original model tests) passes unchanged: task-3-legacy-operator.txt. No old workflow/data conversion suite is newly claimed green.
- Training cases run real tiny one-epoch GPU training, validate weight changes, reference/new-forward equality, unit/byte persistence, fractional-byte differences, caller-directory preservation, malformed-pair refusal, training-only validation and schedule/best-selection policies. The strengthened N15 validation fixture has distinct partition frequencies for every possible uniform predicted class; actual H20 case passes.
- Direct API batch uses positive checked-int capacity; CLI default remains128. Pure policies are approved only as actual workflow inputs. CMake production training backend moving out of tests-ON is necessary for later train; course_forward must remain independent.

## Main source review attention points

- Supplied images/labels are the only data paths; Checkpoint+metadata are the only persistent output. No old RunTrain/hidden dataset accesses are visible in the new workflow. Independent reviewer should verify source/dependencies rather than infer this from accuracy tests.
- RunCourseTrain initializes one original LeNet and reuses separately allocated TrainingStorage DeviceBuffers. The implementer report calls this one reusable arena, but only LeNet owns one arena; reusable auxiliary buffers are allowed by the brief. No runtime claim that these are one allocation.
- Full default20epoch training is NOT performed here; it remains Task6. No course-trained full-test accuracy or latency claim exists.
- Remaining main gates after review: current source/dependency/protected hashes, scoped fixes if found, checkpoint. Final official executable link audit belongs to Task4.

## Additional main acceptance while review runs

- Fresh minimal default source copy omitting course/tests and course/tools builds successfully, with default architecture90 and C++14/O2/no-fast-math commands.
- Exported source list excludes old src/train.cu/main.cu/cli.cpp/dataset.cpp/reporting.cpp. Actual course_training archive has exactly training.cu.o and random.cpp.o; test link brings reviewed original training backend separately. No hidden workflow library is inferred from an archive name.
- Integrity:67 original local source files unchanged;12 previously reviewed course sources unchanged (CMake extension is the authorized exception);8 protected remote artifacts/configs unchanged;16 current CMake/course sources match local/remote. task-3-verification.json.
- Logs/exported link/source command evidence: task-3-default-configure.txt/default-build.txt/dependency-audit.txt, task-3-compile-commands.json, task-3-training-test-link.txt. Review still pending; these checks alone do not close the task.

## Coverage fix round1

Reviewer approves source/spec but marks the missing actual epoch2 path Important. Original implementer adds a real bounded two-epoch fixture without changing training.cu/header, retains8 previous cases, and changes only the CTest count9. Main rebuild warning-free, runs all suites: host48/GPU8/training9 pass. Logs task-3-fix-1-build.txt/ctest.txt. Scoped195-line diff task-3-fix-1.diff changes only test helpers/newcase and count. New case derives strict best selection from per-epoch diagnostic values and validates saved metadata/CRC; it does not require artificial second-epoch improvement. This extends coverage of already-existing behavior, not a fabricated old-code failing RED. No complete20epoch run claimed.

Scoped re-review approves source/spec/quality and closes epoch2 gap. Final integrity proves production/header unchanged during coverage fix and16local/remote matching files, with67oldlocal/12priorreviewed/8protectedremote unchanged (task-3-final-verification.json). Main actual logs resolve reviewer runtime cannot-verify; selected best is accepted based on independent per-epoch oracle, not a fixed expected improvement. Official entrypoint/full20epoch/fulltest performance remain Tasks4/6.
