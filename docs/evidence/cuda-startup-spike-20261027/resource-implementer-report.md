# Resource observation extension — implementer report

## Files

- `diagnostics/observe_cuda_startup.py` (new, 250 lines) — stdlib Linux observer.
- `runtime-output/cuda-startup-spike-20261027/resource-implementer-report.md` (this report).

No existing files were edited. `diagnostics/cuda_startup_probe.cu`, the plan, prior
evidence, build files, and report docs are unchanged. No commits, SSH, installs,
environment overrides, or CUDA execution were performed.

## Interface

CLI, exactly as requested:

```text
python3 observe_cuda_startup.py PROBE_DIRECTORY OUTPUT_JSON
```

The observer preflights all four binaries before launching anything, then performs
exactly four child launches, in this order:

| Case label | Binary under `PROBE_DIRECTORY` | Arguments |
| --- | --- | --- |
| `host-empty` | `empty_host` | `empty` |
| `count-first` | `cuda_startup_probe` | `count-first` |
| `set-first` | `cuda_startup_probe` | `set-first` |
| `sequence` | `cuda_startup_probe` | `sequence` |

Those names match the binaries recorded in `build-and-linkage.log`. A missing,
non-executable, or escaping binary aborts with exit 2 before any launch. The output
path must resolve under `PROBE_DIRECTORY`, must not be that directory, and must not
exist; JSON is opened with exclusive create (`"x"`).

The callable form for harmless pre-CUDA smoke testing is:

```python
observe(command, interval=0.02, timeout=20.0, temp_dir=None)
```

`command` is a non-empty list/tuple (for example `["/bin/true"]`). The function
rejects a missing `os.wait4`, an interval outside `(0, 1]`, or a timeout outside
`(0, 20]`. It returns the result dictionary and does not write JSON itself.

`temp_dir` is the directory used for both `TemporaryFile` handles. When omitted, it
is the resolved parent directory of `command[0]`. The CLI always passes
`PROBE_DIRECTORY`, so child stdout/stderr are created inside the supplied diagnostic
directory, not in remote `/tmp`. Harmless smoke tests should pass an explicit
build-owned writable directory because `/bin`, `/usr/bin`, and similar executable
parents may not be writable. The chosen directory is echoed as `result["temp_dir"]`.

## What is observed

For each child, the observer samples immediately after `Popen` returns and then at
approximately 20 ms intervals until exit or timeout:

- `/proc/PID/task` to enumerate only that child's threads.
- Each `/proc/PID/task/TID/stat`, parsed after the final `)` so spaces or `)` in
  `comm` do not shift fields. Raw values retained: `comm`, `state`, `ppid`,
  `utime_ticks`, `stime_ticks`, `num_threads`, `starttime_ticks`, `vsize_bytes`,
  `rss_pages`, `processor`.
- Each `/proc/PID/task/TID/wchan` raw text (including `0`/empty as observed).
- Each `/proc/PID/task/TID/schedstat` raw line plus parsed `runtime_ns`, `wait_ns`,
  `timeslices`.
- `/proc/PID/status` reduced fields: `State`, `Threads`, `VmRSS`, `VmSize`,
  `VmHWM`, `voluntary_ctxt_switches`, `nonvoluntary_ctxt_switches`.
- `/proc/PID/io` reduced fields: `rchar`, `wchar`, `syscr`, `syscw`,
  `read_bytes`, `write_bytes`.

Every sample records `timestamp_unix`, `timestamp_monotonic`, `offset_s` from the
pre-spawn reference, its own `duration_s`, thread count, per-thread records, status,
and I/O. Status and I/O values are retained as raw strings from proc; tick values
are raw clock ticks, interpretable with `metadata.clk_tck`.

The wait4 rusage is authoritative for whole-child CPU and memory and includes all
threads and teardown: `cpu_user_s`, `cpu_system_s`, `maxrss_kb` (Linux reports
KiB), `minor_faults`, `major_faults`, voluntary/involuntary context switches,
`inblock`, `oublock`, `signals`. The exact `wait_status_raw` and
`os.waitstatus_to_exitcode` result are retained.

## Reaping, timeout, and output

- `Popen` is given `stdin=DEVNULL`, `stdout`/`stderr` each a
  `tempfile.TemporaryFile(dir=temp_dir)`, `close_fds=True`, and
  `start_new_session=True`. The environment and platform preload configuration are
  inherited unchanged.
- The CLI passes `temp_dir=PROBE_DIRECTORY`; the callable form accepts any existing
  writable build-owned directory and otherwise derives the executable parent. The
  observer itself never falls back to `/tmp`.
- The observer never calls `Popen.poll`, `Popen.wait`, `Popen.communicate`, or a
  context manager on the child. It reaps only with `os.wait4(pid, WNOHANG)` and then
  assigns `Popen.returncode` manually.
- At 20 s (default), it sends `SIGKILL` to the child's process group and reaps with
  `wait4`, allowing up to 2 s. A further 2 s cleanup attempt occurs in `finally` if
  needed. Timeout and unreaped-child conditions are recorded in `timed_out` and
  `errors`; slow/failed cases are still returned.
- Child output is decoded with replacement and included in the JSON with byte counts
  and truncation flags. The cap is 1 MiB per stream; the probe outputs are expected
  to be far smaller. No child bytes are written to observer stdout/stderr.
- Metadata includes `system`, `release`, `machine`, `platform`, Python version,
  `clk_tck`, and `page_size`.

## Error and race handling

Nothing sampled is silently dropped. Each failure is an entry in the sample-level
`errors` list (or top-level `errors` for launch/observer failures) with path, errno,
message, and timestamp, classified as:

- `race` — `FileNotFoundError`/`ProcessLookupError` (for example, a thread vanished
  between task enumeration and reading), or `ChildProcessError` while reaping.
- `denied` — `PermissionError` from any proc read.
- `error` — other `OSError`.
- `parse` — a proc body existed but did not match the expected shape.
- `observer_error` — unexpected observer exception; the child is still killed and
  reaped, and the case remains in the result.
- `timeout_reap` — child did not reap after `SIGKILL`.

No broad `/proc` walk occurs: only the spawned PID's own `/proc/PID` and
`/proc/PID/task` are touched. If `task` disappears, `thread_count` is zero and the
race is recorded.

## Local validation on Windows

Executed:

```powershell
python -m py_compile diagnostics/observe_cuda_startup.py
```

Result: exit 0, no syntax errors. The file is exactly 250 lines. An import/static
check confirmed:

- four case labels in the requested order: `host-empty`, `count-first`,
  `set-first`, `sequence`;
- callable signature `(command, interval=0.02, timeout=20.0, temp_dir=None)`;
- only standard-library imports: `json`, `os`, `platform`, `signal`, `subprocess`,
  `sys`, `tempfile`, `time`;
- on Windows, `observe` refuses before spawning with
  `RuntimeError: os.wait4 is required (Linux)`.

The CLI was also executed harmlessly on Windows: no arguments returned exit 2 with
the usage line, and `PROBE_DIRECTORY OUTPUT_JSON` returned exit 2 with
`error: Linux with os.wait4 is required` without creating the output file.

`observe` was not executed locally because Windows has no `os.wait4`; no proc
sampling, wait4 rusage, timeout cleanup, or output capture was exercised here. No
CUDA binary was compiled or run. This report makes no CUDA-latency claim.

## Exact main-agent Linux smoke commands

Run from the repository root on H20. These use only harmless host processes and do
not touch the CUDA probe or its four-run series.

```bash
python3 - <<'PY'
import json
import os
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "diagnostics")
import observe_cuda_startup as obs

smoke_tmp = os.path.abspath("build/observer-smoke")
os.makedirs(smoke_tmp, exist_ok=True)

for name, command in (
    ("true", ["/bin/true"]),
    ("sleep", [sys.executable, "-c", "import time; time.sleep(0.25)"]),
    ("cpu", [sys.executable, "-c", "total = 0\nfor i in range(3000000): total += i"]),
):
    result = obs.observe(command, temp_dir=smoke_tmp)
    print(json.dumps({
        "name": name,
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
        "sample_count": result["sample_count"],
        "rusage": result["rusage"],
        "error_count": len(result["errors"]),
        "errors": result["errors"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "temp_dir": result["temp_dir"],
    }, sort_keys=True))
PY
```

`build/observer-smoke` is a build-owned directory; remove it after the smoke if
desired. The `TemporaryFile` handles are anonymous and `_drain` closes them before
the process exits. The observer contains no hardcoded data/output/pretrain paths.

Expected harmless results:

- `/bin/true`: `returncode=0`, not timed out, wait4 rusage present. Because it can
  exit almost immediately, its single immediate sample may contain expected `race`
  entries instead of usable thread data. `sample_count` is normally 1; it is 0 only
  if the spawn itself failed. That is retained, not treated as running or failed.
- Sleeping Python: `returncode=0`, roughly 0.25 s duration, several samples, low CPU
  in rusage, and mostly sleeping/unknown raw states. It should not time out.
- CPU-bound Python: `returncode=0`, user CPU > 0 in rusage, several samples, with
  some `R` proc states and growing `schedstat.runtime_ns`. `R` is only an observed
  state, not proof that this process was executing at that exact sampling instant.

Then, when the real binaries are present under the probe directory:

```bash
python3 diagnostics/observe_cuda_startup.py \
  runtime-output/cuda-startup-spike-20261027 \
  runtime-output/cuda-startup-spike-20261027/resource-observation.json
```

The main agent must place `empty_host` and `cuda_startup_probe` directly in that
directory (copy or hard link; symlinks resolving outside it are rejected). The
observer performs exactly four launches; do not point it at an unrelated `/proc`
owner or run it more than the planned once.

## Limits for interpretation

- Sampling perturbs the observed process; these are diagnostics, not a new
  acceptance benchmark.
- Wait4 CPU/max RSS cover the whole child, all threads, and teardown, not
  exclusively the first explicit CUDA API.
- `wchan`, `state`, and `schedstat` are raw point-in-time values. `wchan=0`,
  missing files, permission failures, and zero scheduler counters are unknown, not
  evidence of running or of native-vs-platform ownership.
- Snapshot state is not a measured waiting duration; use timestamped raw samples and
  rusage, and state attribution limits explicitly.
- The first sample is taken immediately after spawn, so very short children can
  produce race-only samples; zero samples means the launch itself failed. This is
  expected and recorded.
- Output is capped at 1 MiB per stream with explicit truncation flags. Exact probe
  output is expected to be much smaller.
- `temp_dir` must already exist and be writable; a failure there aborts before any
  child launch. The CLI supplies `PROBE_DIRECTORY`, and the smoke command supplies
  `build/observer-smoke`.
- This observer does not read cgroup, affinity, or CPU-quota files. The main agent
  captures cgroup `cpu.max` (reported `200000/100000`, a two-core quota) and
  affinity 240 separately; interpretation must not assume 240 usable cores.
- Python 3.9+ is needed for `os.waitstatus_to_exitcode`; the H20 environment is
  expected to satisfy this, but it was not checked by this subagent.
