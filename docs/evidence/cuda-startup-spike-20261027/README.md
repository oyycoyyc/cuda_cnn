# CUDA 启动探针证据

36 次原始测量和追加 4 次只读资源观察已保存，未重复原始序列。用户最新目标为完整评估 <100 ms、准确率 >96%，并允许 Git 备份后大幅优化；已做备份，新路线尚未实施。

- `results.json`：36 次原始测量、调用记录、进程输出与二进制校验值。
- `summary.json`：6 个场景各 6 次的中位数、最小值和最大值。
- `verification.json`：源码校验以及既有本地文件、远程部署、数据、权重和预加载配置未改变的证明。
- `environment.txt`：编译器、驱动与只读的平台预加载信息。
- `build-and-linkage.txt`：两个探针构建、链接与 CUDA Runtime 符号记录。
- `smoke.txt`：首次冒烟运行的调用顺序与结果。
- `implementer-report.md`：子代理实现与本地主机检查记录。
- `resource-observation.json`：4 次资源观察的全部线程采样、输出与 wait4 用量。
- `resource-cgroup.json`：观察前后 cgroup CPU／内存限制及计数；本区间节流次数和时间均无增长。
- `resource-analysis.json`：每个场景的主线程状态、等待点、调度增量及读取错误统计。
- `resource-observer-smoke.txt`、`resource-observer-smoke.json`：6 项普通进程验证结果，含退出码、超时回收和输出捕获。
- `resource-verification.json`：恢复后再次验证生产文件、平台预加载配置及探针二进制未变。
- `resource-implementer-report.md`、`resource-observation-plan.md`：追加观察器实现、主代理审查及执行记录。
- `pre-100ms-git-backup.json`：自动化 Git 快照 `backup/pre-100ms-20261027` / `c4c9265d4cbef1bc96a5902f5e404d053abbec8e`，不改当前 HEAD 或真实暂存区，未推送。

结果：首次 CUDA 初始化仍为秒级；包含大量内核态 CPU 活动和 poll／锁等待，未证明平台封装是唯一根因，也不支持将本区间延迟归咎于 CPU 配额节流。更低的准确率门槛不会消除初始化。完整 100 ms 目标尚未达到。

恢复入口：`docs/cuda-startup-spike-results.md`。不重复已完成的测量，不修改平台配置。新实现前先确认此前 GPU/CUDA 限制是否允许放宽；不以常驻服务请求时间偷换完整进程时间。
