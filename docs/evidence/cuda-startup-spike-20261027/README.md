# CUDA 启动探针证据

36 次原始测量已保存。用户随后明确要求继续，追加只读资源侧观察，不重复该组测量。

- `results.json`：36 次原始测量、调用记录、进程输出与二进制校验值。
- `summary.json`：6 个场景各 6 次的中位数、最小值和最大值。
- `verification.json`：源码校验以及既有本地文件、远程部署、数据、权重和预加载配置未改变的证明。
- `environment.txt`：编译器、驱动与只读的平台预加载信息。
- `build-and-linkage.txt`：两个探针构建、链接与 CUDA Runtime 符号记录。
- `smoke.txt`：首次冒烟运行的调用顺序与结果。
- `implementer-report.md`：子代理实现与本地主机检查记录。

恢复入口：`docs/cuda-startup-spike-results.md`。不重新运行已完成的测量，不修改生产程序或平台配置。
