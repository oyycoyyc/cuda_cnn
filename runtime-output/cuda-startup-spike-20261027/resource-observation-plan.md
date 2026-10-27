# Read-only resource observation extension

Scope: continue approved standalone diagnostic, not production changes. Main agent reviews/runs; deepseek-flash coding subagent writes a Python stdlib observer. Do not rerun the 36-run series.

- [x] Archive existing probe evidence into docs/evidence (7 artifacts; exact source/destination hashes checked).
- [x] Implement/review observer; validate with harmless host processes (6 Linux smoke cases: true, sleep, CPU, exit7/output capture, timeout/SIGKILL/reap, large output; stat comm/schedstat parsing; no temporary-file leak).
- [x] Run four observations only: host-empty, count-first, set-first, sequence (one each; all exit0).
- [x] Analyze child user/system CPU, max RSS, context switches, proc thread state/wchan/scheduler counters. Separate observed waiting from unproven attribution.
- [x] Recheck protected hashes, archive observation results, update report and recovery point (68 existing local files and6 protected remote files unchanged; probe/observer source and binary hashes match; original36 records unchanged).

Observer requirements: existing probe binaries unchanged, same platform preload configuration. No ptrace, perf, driver calls, permissions, package installs or environment changes. Sample /proc/PID/task/TID/{stat,wchan,schedstat} plus /proc/PID/status and process io approximately every 20ms; record actual timestamps and race/permission failures. Child stdout/stderr go to TemporaryFile (no deadlock), collect exact exit and resource via os.wait4 with WNOHANG (do not use Popen.poll/wait before wait4); assign Popen.returncode after manual wait. Parse stat after final ')' correctly; do not equate R state with CPU execution or poll snapshots with exact waiting durations. Whole-child CPU/RSS from wait4 are authoritative. timeout<=20s with child cleanup and wait4, all slow/failed results retained. Max 4 probe invocations and no unrelated processes inspected. Output JSON exclusive-create only in supplied diagnostic directory; no credentials in script/output.

Report limits: measurements perturb the process, not new acceptance benchmark; CPU totals include all threads and teardown, not exclusively first API. wchan lacks native-vs-platform ownership attribution; zero/permission errors are unknown, not running. No automatic production fixes/platform requests.

Existing data analysis: the four >3s samples occur at round1 positions5/6 and round2 positions2/3, spanning four different CUDA scenarios. This temporal clustering is consistent with a shared transient influence but does not prove its source. All remain in the original results; no extra benchmark series is needed to rediscover them.

Interpretation reference: https://docs.kernel.org/scheduler/sched-stats.html describes task schedstat fields as CPU execution ns, runqueue-wait ns, and timeslice count. Runqueue-wait is not sleeping in futex/poll/device I/O. Unsupported/disabled zero counters are not proof that wait time is zero. Do not subtract sums of all thread CPU from wall time to claim exact process blocking time (threads may overlap).

Read-only environment finding before observations: affinity exposes240 CPUs, but cgroup-v2 cpu.max is200000/100000 (aggregate2-CPU quota), memory.max10737418240 (10GiB). cpu.stat already has historical throttling, not attributable to the pending probe. Main will capture cpu.stat before/after the four observations without changing quota; deltas include observer and any other processes in this cgroup and cannot uniquely attribute probe delay.

Quota interpretation reference: https://docs.kernel.org/admin-guide/cgroup-v2.html#cpu-interface-files documents cpu.max as MAX/PERIOD bandwidth and cpu.stat usage as cgroup-wide (including descendants). Visible affinity count must not be presented as granted CPU bandwidth or dedicated cores.

Main review: proc stat offsets correctly account for comm parentheses; wait4 alone reaps and supplies whole-child rusage; Popen returncode assigned afterward; inherited platform configuration unchanged; CLI preflights exactly4 commands; stdout/stderr temporary files confined to diagnostic directory. Nonblocking limitations: duration includes sampling/poll-detection delay and observer setup (not exact process wall), samples can race exit, CLI successful JSON write is not proof of child success (main must inspect records), trusted diagnostic output path only. No production changes or hidden CUDA warmup.

New user steering during acceptance: goal now <100ms and >96% accuracy; major optimization allowed with Git backup. Created backup/pre-100ms-20261027 at c4c9265d4cbef1bc96a5902f5e404d053abbec8e using a temporary index; preserves44 changed files and selected scratch evidence without changing HEAD/real index/worktree. Not pushed/deployed. No new execution architecture selected yet; prior project constraints require GPU and no CPU fallback, so clarify CPU permission before building a different path.

Observation conclusion: sampled main-thread runqueue waiting<1ms for each CUDA child, cgroup nr_throttled/throttled_usec deltas0; quota alone is not established bottleneck. Significant system CPU and poll/locking waits remain; native-vs-wrapper attribution unresolved. No full inference performance gain claimed. Wait4 RSS includes fork/exec lifetime (host baseline notably exceeds sampled post-exec RSS); report this limitation.
