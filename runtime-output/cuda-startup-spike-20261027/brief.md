# Approved standalone CUDA startup spike

User chose option1: a temporary standalone diagnostic. This is not a production feature or optimization patch. Main agent plans/reviews/runs H20 experiments; coding subagent writes the probe. No full production implementation workflow is needed for this explicitly approved spike.

## Scope

Create exactly one source file: `diagnostics/cuda_startup_probe.cu` in the existing worktree. Do not edit any existing file, Makefile, src/, include/, tests/, scripts/, or previous evidence. No commits/staging, SSH, subagents, dependency installation, driver changes, preload changes, environment bypass, GPU clocks, cudaDeviceReset, memory allocation, kernels, datasets or checkpoints.

The same source is compiled twice:

1. g++ C++14/O2, `-x c++ -DPROBE_HOST_ONLY`: no CUDA includes or references; only `empty` mode allowed.
2. nvcc C++14/O2 with normal static CUDA Runtime linking matching the project: all modes, including `empty`. CUDA references exist in other branches so Runtime code remains linked even for empty mode. No alternative raw Driver API or driver library selection.

## Exact CLI and behavior

`PROBE MODE`, exactly one mode argument; missing/unknown mode exits2 without CUDA calls. CUDA errors/no visible devices exit1 with concise diagnostic. Successful modes exit0.

- `empty`: no explicit CUDA calls at all.
- `count-first`: cudaGetDeviceCount twice, no earlier CUDA call. Require count>0 on success.
- `set-first`: cudaSetDevice(0) twice, no earlier CUDA call.
- `properties-first`: cudaGetDeviceProperties(device0) twice, no earlier CUDA call.
- `sequence`: GetDeviceCount -> SetDevice(0) -> GetDeviceProperties(0), then repeat that same sequence; exactly6 calls. This reproduces SelectAndReportDevice's API order before the repeat.

Label these as the first/second EXPLICIT calls in main, not proof that platform preload libraries did not call CUDA before main.

Measure each call using std::chrono::steady_clock, starting immediately before and ending immediately after the CUDA API, before output/error-string formatting. Do not add synchronizations, CUDA events, version queries, warmup calls, resets, or other CUDA operations. No global/static user code that calls CUDA.

Successful API records, classic locale and six fractional decimals:

`event=api mode=MODE api=cudaGetDeviceCount call=1 elapsed_ms=VALUE code=0 count=N`

Other API names use cudaSetDevice / cudaGetDeviceProperties. Set records may include `device=0`; properties records may include numeric `major=9 minor=0`. Do not print unescaped names with spaces in key=value records. Record call=1/2 per API, not a shared global counter.

At successful completion emit exactly one:

`event=summary mode=MODE build=host|cuda api_calls=N main_ms=VALUE status=pass`

main_ms starts at main entry and ends just before summary formatting, including API execution and prior in-main work/output; it is NOT full process wall time. Controller measures process startup through exit externally. Difference between process wall and main_ms includes loader, constructors, teardown, reporting and timer wrapper overhead, not solely pre-main time.

Keep implementation small (roughly <=220 lines), C++14, clear checked errors, no proprietary/data/credential logging.

## Handoff

Write implementation only in the new diagnostics source. You may compile/run the HOST_ONLY variant locally using existing g++, test empty, missing/unknown modes, and host rejection of CUDA modes. Do not claim a CUDA pass locally. Report to `runtime-output/cuda-startup-spike-20261027/implementer-report.md`, summarize exact modes/output, local checks, files changed and concerns. Main handles remote compilation, real calls, counterbalanced repeated fresh-process measurements and interpretation. No other agent or reviewer calls.
