#!/usr/bin/env python3
"""Read-only Linux observer for the throwaway CUDA startup probe.

Usage: python3 observe_cuda_startup.py PROBE_DIRECTORY OUTPUT_JSON

Expected children directly under PROBE_DIRECTORY: empty_host and
cuda_startup_probe. Only spawned children's /proc entries are inspected: no
ptrace, perf, driver calls, broad /proc scan, installs, or env changes.
"""
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time

CASES = (("host-empty", "empty_host", ("empty",)), ("count-first", "cuda_startup_probe", ("count-first",)),
         ("set-first", "cuda_startup_probe", ("set-first",)), ("sequence", "cuda_startup_probe", ("sequence",)))
STATUS_FIELDS = ("State", "Threads", "VmRSS", "VmSize", "VmHWM",
                 "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")
IO_FIELDS = ("rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes")
OUTPUT_LIMIT = 1024 * 1024

def _failure(path, exc, tid=None):
    kind = "race" if isinstance(exc, (FileNotFoundError, ProcessLookupError)) else "denied" if isinstance(exc, PermissionError) else "error"
    entry = {"kind": kind, "path": path, "errno": getattr(exc, "errno", None),
             "message": str(exc), "timestamp_unix": time.time()}
    if tid is not None:
        entry["tid"] = tid
    return entry


def _parse_failure(path, tid, exc):
    return {"kind": "parse", "path": path, "tid": tid, "message": str(exc), "timestamp_unix": time.time()}

def _read_text(path, errors, tid=None):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError as exc:
        errors.append(_failure(path, exc, tid))
        return None

def _parse_task_stat(text):
    end = text.rfind(")")
    if end < 0: raise ValueError("task stat is missing ')'")
    f = text[end + 1:].split()
    if len(f) < 37: raise ValueError("task stat has %d fields after comm" % len(f))
    return {"comm": text[text.find("(") + 1:end], "state": f[0], "ppid": int(f[1]),
            "utime_ticks": int(f[11]), "stime_ticks": int(f[12]), "num_threads": int(f[17]),
            "starttime_ticks": int(f[19]), "vsize_bytes": int(f[20]),
            "rss_pages": int(f[21]), "processor": int(f[36])}

def _parse_schedstat(text):
    parts = text.split()
    if len(parts) < 3: raise ValueError("schedstat has fewer than 3 values")
    return {"runtime_ns": int(parts[0]), "wait_ns": int(parts[1]),
            "timeslices": int(parts[2]), "raw": text.strip()}


def _parse_selected(text, wanted):
    values = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() in wanted: values[key.strip()] = value.strip()
    return values


def _sample(pid, t0, index):
    mono, errors = time.monotonic(), []
    sample = {"index": index, "timestamp_unix": time.time(), "timestamp_monotonic": mono,
              "offset_s": mono - t0, "duration_s": None, "errors": errors,
              "thread_count": None, "threads": [], "status": {}, "io": {}}
    task = "/proc/%d/task" % pid
    try: tids = sorted(int(name) for name in os.listdir(task))
    except (OSError, ValueError) as exc: errors.append(_failure(task, exc)); tids = []
    sample["thread_count"] = len(tids)
    for tid in tids:
        base, thread = "%s/%d" % (task, tid), {"tid": tid}
        body = _read_text(base + "/stat", errors, tid)
        if body is not None:
            try: thread["stat"] = _parse_task_stat(body)
            except (ValueError, IndexError) as exc: errors.append(_parse_failure(base + "/stat", tid, exc))
        wchan = _read_text(base + "/wchan", errors, tid)
        if wchan is not None: thread["wchan"] = wchan.strip()
        body = _read_text(base + "/schedstat", errors, tid)
        if body is not None:
            try: thread["schedstat"] = _parse_schedstat(body)
            except (ValueError, IndexError) as exc: errors.append(_parse_failure(base + "/schedstat", tid, exc))
        sample["threads"].append(thread)
    for key, path, wanted in (("status", "/proc/%d/status" % pid, STATUS_FIELDS),
                              ("io", "/proc/%d/io" % pid, IO_FIELDS)):
        body = _read_text(path, errors)
        if body is not None: sample[key] = _parse_selected(body, wanted)
    sample["duration_s"] = time.monotonic() - mono
    return sample

def _kill(pid, errors):
    try: os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError: return
    except OSError as exc:
        errors.append(_failure("/proc/%d" % pid, exc))
        try: os.kill(pid, signal.SIGKILL)
        except OSError as fallback: errors.append(_failure("/proc/%d" % pid, fallback))


def _reap_until(pid, errors, until):
    while True:
        try:
            waited, status, usage = os.wait4(pid, os.WNOHANG)
        except ChildProcessError as exc:
            errors.append(_failure("/proc/%d" % pid, exc))
            return None, None
        if waited == pid: return status, usage
        if time.monotonic() >= until:
            errors.append({"kind": "timeout_reap", "path": "/proc/%d" % pid,
                           "message": "child not reaped after SIGKILL", "timestamp_unix": time.time()})
            return None, None
        time.sleep(0.005)


def _drain(handle):
    handle.seek(0)
    raw = handle.read(OUTPUT_LIMIT + 1)
    truncated = len(raw) > OUTPUT_LIMIT
    raw = raw[:OUTPUT_LIMIT]
    handle.seek(0, os.SEEK_END)
    total = handle.tell()
    handle.close()
    return raw.decode("utf-8", "replace"), total, truncated


def observe(command, interval=0.02, timeout=20.0, temp_dir=None):
    """Run one child; return samples, wait4 rusage, exit status and output."""
    if not hasattr(os, "wait4"): raise RuntimeError("os.wait4 is required (Linux)")
    if not isinstance(command, (list, tuple)) or not command: raise ValueError("command must be a non-empty list")
    interval, timeout = float(interval), float(timeout)
    if not 0 < interval <= 1: raise ValueError("interval must be in (0, 1] seconds")
    if not 0 < timeout <= 20: raise ValueError("timeout must be in (0, 20] seconds")
    command = [os.fsdecode(os.fspath(part)) for part in command]
    temp_dir = os.path.realpath(temp_dir) if temp_dir else os.path.dirname(os.path.realpath(command[0]))
    if not os.path.isdir(temp_dir): raise ValueError("temp_dir is not a directory: %s" % temp_dir)
    start_wall, t0 = time.time(), time.monotonic()
    samples, errors = [], []
    result = {"command": command, "interval_s": interval, "timeout_s": timeout, "pid": None,
              "temp_dir": temp_dir,
              "started_at_unix": start_wall, "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_wall)),
              "finished_at_unix": None, "duration_s": None, "timed_out": False,
              "returncode": None, "wait_status_raw": None, "rusage": None,
              "sample_count": 0, "samples": samples, "errors": errors,
              "stdout": "", "stderr": "", "stdout_bytes": 0, "stderr_bytes": 0,
              "stdout_truncated": False, "stderr_truncated": False}
    out_file, err_file = tempfile.TemporaryFile(dir=temp_dir), tempfile.TemporaryFile(dir=temp_dir)
    proc, status, usage, reaped = None, None, None, False
    try:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out_file,
                                stderr=err_file, close_fds=True, start_new_session=True)
        pid, deadline, next_sample = proc.pid, t0 + timeout, t0
        result["pid"] = pid
        while True:
            if time.monotonic() >= next_sample:
                samples.append(_sample(pid, t0, len(samples)))
                next_sample = time.monotonic() + interval
            waited, status, usage = os.wait4(pid, os.WNOHANG)
            if waited == pid: reaped = True; break
            now = time.monotonic()
            if now >= deadline:
                result["timed_out"] = True
                _kill(pid, errors)
                status, usage = _reap_until(pid, errors, now + 2.0)
                reaped = status is not None
                break
            time.sleep(min(interval, deadline - now))
    except Exception as exc:
        errors.append({"kind": "observer_error", "error_type": type(exc).__name__,
                       "message": str(exc), "timestamp_unix": time.time()})
    finally:
        if proc is not None and not reaped:
            _kill(proc.pid, errors)
            status, usage = _reap_until(proc.pid, errors, time.monotonic() + 2.0)
            reaped = status is not None
        if reaped:
            result["returncode"] = os.waitstatus_to_exitcode(status)
            proc.returncode = result["returncode"]
        result["wait_status_raw"] = status
        result["rusage"] = None if usage is None else {
            "cpu_user_s": usage.ru_utime, "cpu_system_s": usage.ru_stime, "maxrss_kb": usage.ru_maxrss,
            "minor_faults": usage.ru_minflt, "major_faults": usage.ru_majflt,
            "voluntary_context_switches": usage.ru_nvcsw, "involuntary_context_switches": usage.ru_nivcsw,
            "inblock": usage.ru_inblock, "oublock": usage.ru_oublock, "signals": usage.ru_nsignals}
        result["finished_at_unix"] = time.time()
        result["duration_s"] = time.monotonic() - t0
        result["stdout"], result["stdout_bytes"], result["stdout_truncated"] = _drain(out_file)
        result["stderr"], result["stderr_bytes"], result["stderr_truncated"] = _drain(err_file)
    result["sample_count"] = len(samples)
    return result


def _error(message):
    sys.stderr.write("error: %s\n" % message)
    return 2


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: python3 observe_cuda_startup.py PROBE_DIRECTORY OUTPUT_JSON\n")
        return 2
    if not sys.platform.startswith("linux") or not hasattr(os, "wait4"):
        return _error("Linux with os.wait4 is required")
    probe_dir, output = os.path.realpath(argv[1]), os.path.realpath(argv[2])
    if not os.path.isdir(probe_dir): return _error("not a directory: %s" % probe_dir)
    if os.path.commonpath((probe_dir, output)) != probe_dir or output == probe_dir:
        return _error("OUTPUT_JSON must be under PROBE_DIRECTORY")
    if os.path.lexists(output): return _error("refusing to overwrite: %s" % output)
    commands = []
    for case, binary, args in CASES:
        path = os.path.realpath(os.path.join(probe_dir, binary))
        if os.path.commonpath((probe_dir, path)) != probe_dir:
            return _error("binary escapes PROBE_DIRECTORY: %s" % binary)
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            return _error("missing executable: %s" % path)
        commands.append((case, [path] + list(args)))
    results = []
    for case, command in commands:
        observed = observe(command, temp_dir=probe_dir)
        observed["case"] = case
        results.append(observed)
    metadata = {"system": platform.system(), "release": platform.release(),
                "machine": platform.machine(), "platform": platform.platform(),
                "python": platform.python_version()}
    for key, name in (("clk_tck", "SC_CLK_TCK"), ("page_size", "SC_PAGE_SIZE")):
        try: metadata[key] = os.sysconf(name)
        except (AttributeError, ValueError, OSError): metadata[key] = None
    document = {"schema": "cuda-startup-observation-v1", "created_at_unix": time.time(),
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "probe_directory": probe_dir, "observer_pid": os.getpid(), "metadata": metadata,
                "cases": results}
    try:
        with open(output, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(document, indent=2) + "\n")
    except OSError as exc:
        return _error("cannot write %s: %s" % (output, exc))
    sys.stdout.write("wrote=%s cases=%d\n" % (output, len(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
