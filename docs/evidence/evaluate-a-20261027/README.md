# Evaluate A 验证证据索引

本目录保留本轮精选证据，不依赖 Git 忽略的 `.superpowers` 或 `runtime-output` 目录。未包含 SSH 密码或认证凭据，尚未提交到 Git。

- `final-alternating-benchmark.json`：第一轮 5 组交替对照的完整输出、时间和二进制校验值。
- `final-alternating-benchmark-v2.json`：完整重建后再次运行的 5 组交替对照，是当前候选版本的结果。
- `snapshot-comparison.json`：批大小 1、17、128 的全量输出比较。
- `snapshot-comparison-rebuilt.json`：完整重建后的再次比较。
- `verified-source-manifest.json`：最终复核前本地／远程 43 个文件的规范换行 SHA-256 清单。
- `final-verification.txt`：当前候选的 CUDA 算子、工作流、快照测试及策略扫描结果。
- `final-aggregate-test.txt`：完整远程聚合目标的实际失败记录（缺少 pyarrow），防止将分项通过误写为全绿。
- `final-hashes.txt`：原部署、数据与权重未改变的校验记录。
- `final-review.md`：独立整体审查及非阻塞发现。
- `controller-final-verification.md`：主代理验收说明，包括环境限制。
- `execution-ledger.md`：任务分配、审查、修复、证据和执行裁定。

源码和结果报告在验证后仅有文档收尾；清单记录的是其标注的验证时刻。所有计算代码和当前候选二进制与第二轮基准测试版本一致。

**结论：准确率保持 99.11%，全量输出逐字节一致，有性能改善，但未达到含启动的 1000 毫秒目标。** 远程聚合测试仍因缺少 pyarrow 而非全绿，Compute Sanitizer 自行禁用，不能声称完整远程测试或 sanitizer 通过。
