# Task 1 report: tests-first handoff

**Status:** GREEN implementation complete; tests unchanged since main's RED verification. The tests-first history follows, then the implementation/local static-check section.

**Created:** `course/tests/warm_benchmark_tests.cu` (7 cases, self-contained POSIX subprocess runner, existing `test_support.h` single-TU main, build-owned scratch only).

**Cases**
- Exact argc 9, strict decimal counts (`warmups`/`runs` capped at 100000), unsupported geometry.
- Exact stderr protocol: W warmup then R measure records, separate one-based indices, full N, strictly positive `wall_ms`, `predictions_identical=1`, one passing completion record; stdout exactly N canonical class lines.
- Literal finite fixture (zero/one N=2, hidden weights 1/bias 0, fc3 class-0 row 1, class-1 bias 1e8) must give `[1,0]` at B=1 and B=2 over W=2/R=3.
- Real N=17/B=4 and N=1025/B=1024 outputs must equal the official `infer` oracle (no hardcoded classes).
- Poison finite checkpoint (all weights 1, biases 0) with normalized FLT_MAX first image must fail with no stdout after a successful safe precondition.
- Missing/empty/invalid `COURSE_MODEL_DIR`/`COURSE_INFER_BATCH`, malformed/bad-CRC/corrupt bundles, short/long/non-finite/out-of-mode/missing images.
- Persisted mode drives preprocessing and completion; `COURSE_INPUT_MODE`, `COURSE_EPOCHS`, `COURSE_SEED`, `COURSE_PROFILE` are ignored and no infer/profile records appear.

**RED expectation:** Before the tool target exists, every case fails by name at the build-adjacent missing `course_warm_benchmark` entry point, before fixtures, the official oracle, or hardware.

**GREEN build note (main):** Put the tests target and tool inside `COURSE_BUILD_TESTS`. Link tests to `course_host` only (parent must not initialize CUDA before fork/exec). Suggested CTest: `PASS_REGULAR_EXPRESSION "executed=7 failures=0 status=pass"` and `TIMEOUT 300`; per-child timeout is already 240 s.

**Verification:** Static `g++ -fsyntax-only -Wall -Wextra -Wpedantic` with POSIX shims passed cleanly. No local CUDA execution or CUDA success claim. No production code, CMake, model, data, git index/HEAD, or root artifact changes.

---

## GREEN implementation

**Files**
- Created `course/tools/warm_benchmark.cu` (development-only tool).
- Modified only the `if(COURSE_BUILD_TESTS)` block of `CMakeLists.txt`; default `COURSE_BUILD_TESTS=OFF` build graph and official `train`/`infer` links are unchanged.
- `course/tests/warm_benchmark_tests.cu` and all old source/header files unchanged since RED.

**CLI and host order**
- Exact `argc == 9`: `course_warm_benchmark images.f32 N C H W K warmups runs`.
- First seven arguments parsed by `course::ParseInferArgs(7, ...)` with `COURSE_MODEL_DIR`/`COURSE_INFER_BATCH`; profile input is `nullptr`, so `COURSE_PROFILE` is ignored.
- `warmups`/`runs` parsed as strict positive ASCII decimal in `[1, 100000]`; signs, spaces, fractions, empty values, and overflow are usage errors.
- `LoadCourseModel` and `ReadImages` complete before `cudaSetDevice(0)`. Persisted model mode is authoritative; no labels, training variables, or mode override.
- Host prediction and baseline vectors are allocated before device setup. One stream, one `InferenceModel`, `packed`/`normalized` at `B*784`, all-`N` predictions, and one flag are allocated once. An untimed stream synchronization drains setup before the first warmup.

**Timed round**
- Steady-clock start precedes the per-round `cudaMemsetAsync` sticky-flag reset.
- Every round re-copies the actual whole input in batch slices with host offset `batch_index * B * 784` to device input offset zero, then runs `LaunchPrepareInput`, `InferenceModel::Forward`, and `LaunchArgmax` at the global prediction offset.
- One all-`N` prediction D2H, one flag D2H, and one `cudaStreamSynchronize` follow. Flag, every prediction range, and exact equality with the first warmup baseline are checked inside the clock; the clock stops after those checks. The first warmup copies into the preallocated baseline. No device allocation, timing event, disk I/O, or output formatting occurs in the timed region.
- Round record is formatted outside the clock with classic locale and 17 significant digits, written and flushed with an I/O check. Warmup records precede all measured records. Completion is emitted only after all rounds, stdout flush, and its own stderr flush succeed; stdout stays buffered until every round passes, so `/dev/full` cannot produce a false pass.

**CMake/CTest**
- `course_warm_benchmark` links `course_host` + `course_forward` only.
- `course_warm_benchmark_tests` links `course_host` only.
- Both targets and `add_test(NAME course_warm_benchmark ...)` are inside `COURSE_BUILD_TESTS`; CTest uses `PASS_REGULAR_EXPRESSION "executed=7 failures=0 status=pass"` and `TIMEOUT 300`.

**Local static checks (limits stated)**
- Tool: `g++ -x c++ -std=c++14 -fsyntax-only -Wall -Wextra -Wpedantic` with minimal CUDA type/function declaration stubs and project include roots; exit 0, no diagnostics.
- Tests: same warning flags with POSIX shims (`fork`/`poll`/`wait` declarations), exit 0, no diagnostics.
- Limits: these are host-syntax checks only. No NVCC, CUDA toolkit, CMake configure/build, link audit, or GPU execution was available or claimed locally. Main still owns the NVCC build, five CTests, link audit, default minimal build, independent review, and checkpoint.
