# CUDA startup probe — implementer report

## Files

- `diagnostics/cuda_startup_probe.cu` (new, 220 lines) — the only source file added.
- `runtime-output/cuda-startup-spike-20261027/implementer-report.md` (this report).

No existing files, build files, tests, docs, or evidence were edited. No commits or staging were performed.

## Implementation summary

Single source with `#ifndef PROBE_HOST_ONLY` around `<cuda_runtime.h>` and all CUDA
code. The HOST_ONLY translation unit contains no CUDA code references; CUDA mode
names are parsed and rejected with exit 2.

- `empty`: no explicit CUDA call, emits a host or CUDA summary with `api_calls=0`.
- `count-first`: two measured `cudaGetDeviceCount` calls; each must return
  `cudaSuccess` and `count > 0`.
- `set-first`: two measured `cudaSetDevice(0)` calls.
- `properties-first`: two measured `cudaGetDeviceProperties(&properties, 0)` calls.
- `sequence`: count -> set -> properties, repeated once; exactly 6 attempted calls.

Each call is timed with `std::chrono::steady_clock::now()` immediately before and
after the Runtime call. Formatting happens after the end sample. Records use the
classic locale and `std::fixed << std::setprecision(6)`. `call=1/2` is per API,
not a shared counter. `main_ms` starts as the first statement in `main` and ends
just before summary formatting. Successful API/summary records go to stdout;
error diagnostics go to stderr.

Successful record shapes:

```text
event=api mode=count-first api=cudaGetDeviceCount call=1 elapsed_ms=VALUE code=0 count=N
event=api mode=set-first api=cudaSetDevice call=1 elapsed_ms=VALUE code=0 device=0
event=api mode=properties-first api=cudaGetDeviceProperties call=1 elapsed_ms=VALUE code=0 major=9 minor=0
event=summary mode=MODE build=host|cuda api_calls=N main_ms=VALUE status=pass
```

`major`/`minor` are read from the returned `cudaDeviceProp`; `9`/`0` above is only
an illustrative shape. CUDA failures emit one line such as:

```text
event=error mode=count-first api=cudaGetDeviceCount call=1 code=999
event=error mode=count-first reason=no_visible_devices
```

No `cudaGetErrorString`, version query, reset, event, synchronization, allocation,
kernel, or warmup call exists. No global/static user code calls CUDA. The code
labels `call=1/2` as first/second explicit calls in `main`; it does not claim they
are the first process-wide CUDA calls, since platform preload libraries may act
before `main`.

## Local host-only verification

Compiled with the existing g++:

```powershell
& "C:/personal_apps/msys64/ucrt64/bin/g++.exe" `
  -std=c++14 -O2 -Wall -Wextra -Wpedantic -Werror `
  -DPROBE_HOST_ONLY -x c++ diagnostics/cuda_startup_probe.cu `
  -o "$env:TEMP/cuda_startup_probe_host.exe"
```

Result: exit 0, no warnings. Binary was written to `%TEMP%`, not the repository.

Executed cases (fresh process per case):

| Arguments | Exit | Output |
| --- | --- | --- |
| none | 2 | `event=error reason=usage expected=MODE` |
| `empty` | 0 | `event=summary mode=empty build=host api_calls=0 main_ms=0.001900 status=pass` |
| `empty extra` | 2 | `event=error reason=usage expected=MODE` |
| `bogus` | 2 | `event=error reason=unknown_mode` |
| `count-first` | 2 | `event=error mode=count-first reason=host_build_unsupported` |
| `set-first` | 2 | `event=error mode=set-first reason=host_build_unsupported` |
| `properties-first` | 2 | `event=error mode=properties-first reason=host_build_unsupported` |
| `sequence` | 2 | `event=error mode=sequence reason=host_build_unsupported` |

Additional check: `g++ -std=c++14 -DPROBE_HOST_ONLY -x c++ -E ...` and scanning the
preprocessed output (excluding line markers) found no CUDA code references.

The HOST_ONLY variant was also checked for line count (220) and maximum line
length (all lines <= 80 columns).

## Not verified locally

No CUDA compiler or GPU was used. There is no local CUDA-success claim, no
`nvcc` build, no device count/set/properties execution, and no `main_ms` or
process-wall measurement. The main agent owns nvcc compilation with normal static
CUDA Runtime linking (no shared-runtime override), real-device runs, and
counterbalanced repeated-process measurements.

## Concerns for the main agent

1. CUDA failures report only the numeric `code`. `cudaGetErrorString` was omitted
   deliberately to keep the error path free of additional CUDA API calls. If
   readable text is required, the controller should map the code.
2. Host-only CUDA modes are classified as unsupported and exit 2, the same status
   as usage/unknown-mode errors. That is intentional rejection, not a CUDA
   runtime failure (which exits 1).
3. For `count-first` and `sequence`, a zero count after a successful call emits
   the `event=api ... code=0 count=0` record followed by
   `reason=no_visible_devices`, then exits 1 without a summary. Consequently the
   second count call is not attempted when the first sees no devices.
4. `empty` is valid in the CUDA build and exits 0. It may still involve static
   CUDA Runtime initialization or platform preload behavior; that is part of what
   the host-vs-CUDA comparison is intended to expose.
5. `api_calls` counts attempted calls and is only emitted in a success summary.
   Partial-progress call counts are therefore not reported on failure.
6. Unknown mode text from argv is not echoed, avoiding unescaped spaces or
   arbitrary control characters in key=value output. Only the fixed
   `reason=unknown_mode` is printed.
7. The properties record always includes `major`/`minor`; the set record always
   includes `device=0`. The count record includes `count`. All other fields follow
   the pinned order.
8. The source intentionally has no `-lineinfo`, gencode, or other project build
   flags embedded. Remote compilation flags determine those details.
