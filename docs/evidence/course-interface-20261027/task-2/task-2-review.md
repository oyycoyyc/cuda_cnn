# Task 2 independent review (read-only, spec + quality)

Reviewer scope: base reviewed checkpoint `aacad14`, six NEW uncommitted files only
(`CMakeLists.txt`, `course/gpu_ops.h/.cu`, `course/inference.h/.cu`,
`course/tests/gpu_tests.cu`), diff read once in full (1724 lines). HEAD `bf6aae8`
and index untouched by design; no git/subagent/SSH commands, no edits to sources,
no re-run of any suite. This review does not re-grade the controller's H20 run.

**Verdicts**

- **Spec Compliance: NEEDS FIXES** — one explicitly required build artifact of the
  tests-ON configuration (`course_model_bundle`) does not link; see Critical 1.
  Everything else in the Task 2 brief is present, including the four named tests
  plus three extra cases and the exact public signatures.
- **Task Quality: NEEDS FIXES** — implementation is otherwise high quality and the
  numerical design is sound (see Strengths); the blocking item is build-graph, plus
  one real-but-practically-benign data race to harden.
- Not a full-build PASS. The reported `course_io_tests` 48/48 and
  `course_gpu_tests` 7/7 do not substitute for the required tests-ON full build,
  which fails (`task-2-cmake-build.txt`).

## Strengths (verified against unchanged code, not just the tests)

1. **Input semantics genuinely match the original, and genuinely avoid uint8.**
   `course/gpu_ops.cu:229-241` computes `(x-0.1307F)/0.3081F` and
   `(x/255.0F-0.1307F)/0.3081F` in the same FP32 order as the original
   `src/kernels/input.cu:58-59`; no quantized intermediate exists anywhere in the
   new path. Padding is `NormalizeUnit(0.0F)` for every mode, which is exactly what
   the original produced for an out-of-range source pixel (uint8 zero, then
   normalized), so kUnit/kByte padding is unchanged and kNormalized padding is
   correctly the normalized black pixel, not raw `0.0F`.
2. **Translation convention and offset source are the original ones.**
   `course/gpu_ops.cu:90-96` samples `packed[sample][y-dy][x-dx]` using
   `TranslationOffsetForSample(seed, epoch, original_indices[sample])`; this matches
   the documented original convention (`include/layers.h:11-12`,
   `src/kernels/input.cu:25-27,51-52`) and the helper is `LENET_HOST_DEVICE`
   (`include/random.h:112`), so the device call needs no host fallback.
3. **Argmax contract is implemented literally.** One thread per row, `std::int32_t`
   output, strict `>` so ties keep the smallest index (`course/gpu_ops.cu:112-117`),
   and `IsFinite` (`:244-246`, `:105-111`) classifies NaN/±inf while every class of
   every row is examined — including class 0 and the last element. The flag is only
   ever written to 1; the kernel never zeroes it. Host-side dimension/null/overlap
   validation happens before launch and there is no synchronization or readback.
4. **Forward math is the unchanged original chain, not a re-implementation.**
   `course/inference.cu:151-171` calls the same launchers in the same order with the
   same shapes/counts as `src/lenet.cu` `Forward` (1x28x28->6@24x24->6@12x12->
   16@8x8->16@4x4->256->120->84->10). The only deviation from the reference is
   in-place ReLU (`course/inference.cu:153,159,166,169`), which is explicitly
   permitted and exact: `src/kernels/activation.cu:48-65` documents and implements
   read-then-write of the same index per thread, so post-ReLU values are bitwise
   identical to the out-of-place buffers `lenet.cu` uses. Reduction/arith order is
   therefore preserved, not approximated.
5. **One-arena discipline is real, and the "strictly smaller" claim is structural.**
   Validation (exact ten-tensor schema via `ValidateLenetParameters`, which pins
   exact value counts in `src/parameters.cpp`, plus a full finite scan) runs before
   any device work; exactly one `DeviceBuffer<std::uint8_t>` allocation
   (`course/inference.cu:129`), 16-byte bump alignment, `plan order ==
   RequiredBytes order` with a hard equality assertion (`:231`), direct
   `cudaMemcpyAsync` of the supplied payloads in canonical order, one stream
   synchronization, no arena zeroing, no random initialization, no
   gradient/moment/scan storage. Forward allocates nothing and returns a stable
   pointer. Per-sample storage drops from `10498+6588` float + 1120 winner bytes
   (reference) to `5814` float + 1120 winner bytes, so the arena is strictly
   smaller at *every* capacity, not only at the three capacities the test probes.
6. **Test suite is a real oracle, not a mirror of the production path.** Mode values
   come from host scalar fixtures with fractional/boundary/denormal heads; the
   no-uint8 assertions compare against both truncation and rounding; translation is
   pinned by hard-coded offsets obtained from the original helper and cross-checked
   by equal-index row equality / different-index inequality; logits are compared
   with `memcmp` against the original `LeNet` fed the same device buffer and the
   same `ParameterSet` imported through `ImportParameters`, with predictions checked
   against a host smallest-index argmax of the *reference* logits. Allocation
   counters bracket construction, Forward and argmax; destruction returns to
   baseline. No production CPU fallback is used anywhere.

## Findings

### Critical (blocks a brief requirement)

1. **`CMakeLists.txt:51-52` — `course_checkpoint` declares no dependency on
   `course_parameters`, so the required tests-ON build cannot link
   `course_model_bundle`.**
   `src/checkpoint.cpp:59,132,266-267` defines `SaveCheckpoint`/`LoadCheckpoint`
   with undefined references to `ValidateLenetParameters`, `LenetParameterSpecs`
   and `CreateLenetParameters`, all of which live in `src/parameters.cpp`. The
   archive is created with no `target_link_libraries`, and `CMakeLists.txt:59`
   lists `course_parameters` *before* `course_checkpoint` on `course_host`'s
   interface. That ordering only survives for targets that themselves reference
   parameter symbols — `course_io_tests` does, so it accidentally resolves;
   `course_model_bundle` (`CMakeLists.txt:106-107`) does not, and fails exactly as
   captured in `task-2-cmake-build.txt` (`undefined reference to
   ValidateLenetParameters / LenetParameterSpecs / CreateLenetParameters`).
   Brief line 17 requires the ON configuration to build Task1's
   `course_model_bundle` utility; the controller's "build io+gpu+bundle" step is
   therefore not satisfied. The same latent defect will hit the Task 4 `train`/
   `infer` executables as soon as they link `course_host` without directly
   touching parameter symbols, including in the default configuration where no
   executable currently exists to expose it.
   Fix: `target_link_libraries(course_checkpoint PUBLIC course_parameters)` (the
   static-library consumer must declare its own dependency; reordering
   `CMakeLists.txt:59` would only re-hide the gap).

### Important (real defect, low practical impact, cheap fix, must not ship as-is)

2. **`course/gpu_ops.cu:118-121` — many row threads perform plain stores to the
   single `*bad_output` word; a genuine data race with undefined semantics, though
   not an observable wrong value on current hardware.**
   Independently assessed: every writer stores the same value `1`, no thread ever
   writes `0`, no kernel-side reader exists, `bad_output` is an aligned `int` from
   the caller, and the host reads only after stream completion, so no torn store,
   no lost update and no ordering hazard can change the observed result. This is
   therefore not a functional blocker and the current numerical tests are
   legitimately unaffected. It is still a conflicting non-atomic same-location
   access under the CUDA/C++ memory model, i.e. formally UB, and it is the kind of
   construct that later tooling or refactoring (e.g. adding a host readback inside
   the request, or a flag reset) could turn into a real bug. One-line hardening:
   `if (nonfinite) atomicExch(bad_output, 1);` (or a single designated writer).
   Note explicitly: the suite cannot settle race freedom — `gpu_tests.cu:602-612`
   poisons one row per launch, which exercises accumulation but never a
   many-writer conflict; race freedom here is a design-review property, and it is
   currently only *practically* safe, not formally safe.

### Minor (non-blocking; options, not requirements)

3. **`course/tests/gpu_tests.cu:389-403` — the augmented expectation re-implements
   the kernel's own mapping.** The expected buffer is rebuilt with the same
   `x - offset.dx` / `y - offset.dy` rule the kernel uses, so the reconstruction
   loop cannot independently falsify a sign inversion; the independent anchors are
   the pinned offsets (asserted `2/-2` and `-2/0`), the documented convention in
   `include/layers.h:11-12`, the padding checks (`:403,419`) and the
   equal-index/different-index row comparisons. Acceptable as-is; an independent
   oracle would require a host reimplementation of the original uint8 kernel, which
   is not worth the duplication.
4. **`course/tests/gpu_tests.cu` — small uncovered boundaries.** `classes == 1`
   argmax (valid per `course/gpu_ops.h:184-186`) and the
   grid/`size_t` overflow branches of `PreparedElementCount`/`CheckedMultiply` are
   untested; arena alignment and exact `RequiredDeviceBytes() == arena bytes` are
   not observable from the public API. Optional.
5. **`CMakeLists.txt:26-27,31` — optimization flags are hard-coded and duplicated.**
   Overwriting `CMAKE_CXX_FLAGS_RELEASE`/`CMAKE_CUDA_FLAGS_RELEASE` discards
   caller/toolchain release flags, and `add_compile_options(-O2)` repeats `-O2` on
   every target. Also note NVCC documents `-O` as a host-code option, so it is not
   demonstrable from the source that device code is compiled at O2 (device
   optimization remains at NVCC's default). This cannot change results here because
   both sides of every bitwise comparison are the same archive objects and no
   fast-math is enabled anywhere, so it is a clarity/portability nit, not a
   correctness issue.
6. **`course/inference.cu:231` — `std::logic_error` from `PlanArena` is not in the
   documented exception contract** (`course/inference.h:489-499` promises
   `invalid_argument` for host misuse and `runtime_error` for CUDA/memory). It can
   only fire on an internal planning bug, so this is documentation-only.

## Brief checklist status (as reviewed)

- New sources/headers/signatures and CMake registration: met.
- Four named tests plus three extra, fixtures/reference oracle, no CPU fallback: met.
- No-uint8 preparation, original offset helper, int32/all-row/sticky argmax: met.
- Copy-validated weights, one arena, no random init/gradients/zeroing, in-place ReLU
  only under the documented alias contract: met.
- Batches 1/17/1024 bitwise + exact predictions, capacity/null/poisoned/repeat/tail/
  allocation-count/smaller-arena tests: met.
- CMake >=3.18, CXX/CUDA14 REQUIRED, arch 90 only when caller did not set it
  (before `project()`), O2, no fast-math, host/forward libraries free of old
  workflow/training dependencies, optional sources kept out of the default build:
  met (see Minor 5 for flag hygiene).
- "Optional tests ON ... builds Task1's `course_model_bundle`": **NOT met** —
  Critical 1.

## Cannot verify (out of read-only scope; main owns runtime evidence)

- The 48 host / 7 GPU pass results, the RED linker evidence, and any re-run after a
  fix: evidence is the controller's logs; suites were not re-executed here.
- A fresh default `COURSE_BUILD_TESTS=OFF` configure and a caller
  `-DCMAKE_CUDA_ARCHITECTURES=<non-90>` override: `task-2-cmake-configure.txt`
  contains no `-D` arguments and no exported compile/link commands, and
  `controller-task-2-verification.md` lists both as "not yet verified" even though
  `task-2-acceptance.md` marks the default/override item done. The source logic
  (`CMakeLists.txt:6-8` ahead of `project()`) is correct by inspection only.
- Actual device-code optimization level (see Minor 5) and a device warning review
  (`course_enable_host_warnings` deliberately applies to CXX only).
- Post-fix full tests-ON build, post-fix `course_model_bundle` link, protected
  local/remote source hashes, and Task 4 executable linkage against `course_host`.
