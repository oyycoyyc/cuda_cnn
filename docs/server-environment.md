# 服务器目录与存储注意事项

记录日期：2026-10-27。以下存储规则由用户提供，适用于 Gemini 平台的开发环境和离线训练任务。

## 目录与权限

| 用途 | 默认目录 | 环境变量 | 开发环境 | 离线训练 |
|---|---|---|---|---|
| 代码 | `/gemini/code` | `$GEMINI_CODE` | 可增、删、改；仅此目录下的内容会持久化保存 | 只读，禁止写入 |
| 数据集 1 | `/gemini/data-1` | `$GEMINI_DATA_IN1` | 挂载时只读 | 挂载时只读 |
| 数据集 2 | `/gemini/data-2` | `$GEMINI_DATA_IN2` | 挂载时只读 | 挂载时只读 |
| 数据集 3 | `/gemini/data-3` | `$GEMINI_DATA_IN3` | 挂载时只读 | 挂载时只读 |
| 预训练数据／模型 | `/gemini/pretrain` | `$GEMINI_PRETRAIN` | 挂载时只读 | 挂载时只读 |
| 训练输出 | `/gemini/output` | `$GEMINI_DATA_OUT` | 不挂载，不应手动写入 | 用于保存训练结果 |

### code：代码目录

- 上传的代码存放于 `/gemini/code`，可通过 `$GEMINI_CODE` 访问。
- 在开发环境中，仅该目录下的内容会被持久化保存，可在该目录下执行增、删、改操作。
- 不建议在该目录存放大体量数据集，以免镜像过大，影响使用体验。
- **离线训练中该目录仅有只读权限；强行写入会导致数据丢失。**

### data：数据集挂载目录

- 每个任务最多挂载三个数据集，分别对应 `/gemini/data-1`、`/gemini/data-2`、`/gemini/data-3`。
- 可分别通过 `$GEMINI_DATA_IN1`、`$GEMINI_DATA_IN2`、`$GEMINI_DATA_IN3` 访问。
- **这些目录为只读目录；对其执行写入操作会导致数据丢失。**

### pretrain：预训练数据／模型目录

- 预训练数据或模型挂载于 `/gemini/pretrain`，可通过 `$GEMINI_PRETRAIN` 访问。
- **该目录为只读目录；对其执行写入操作会导致数据丢失。**

### output：训练输出目录

- `/gemini/output` 仅在训练任务中使用，用于存储训练结果、模型检查点（checkpoint）和日志（log）等。
- 设置输出路径时使用 `$GEMINI_DATA_OUT`；训练完成后，可在项目的“结果”中查看输出内容。
- **开发环境不会挂载该目录；手动写入会导致数据丢失。**

## 本项目使用约定

- 远程项目目录：`${GEMINI_CODE:-/gemini/code}/cuda_cnn`。
- 当前数据集：`${GEMINI_DATA_IN1:-/gemini/data-1}/train.bin` 和 `test.bin`，始终按只读处理。
- 已有模型权重：项目内的 `weights/lenet.bin`；离线训练时只可读取，不可覆盖。
- 开发环境需要持久化的少量输出可放在代码目录下的 `cuda_cnn/runtime-output`；不要手动创建或写入 `/gemini/output` 来替代平台挂载。
- **离线训练的所有新增文件均应写入平台挂载的 `$GEMINI_DATA_OUT`，包括权重、日志、临时文件和编译产物。** 不要使用项目内的 `build/`、`weights/` 或 `runtime-output/` 作为离线训练写入路径。
- `scripts/gemini_env.sh` 未取得 `$GEMINI_DATA_OUT` 时会回退到项目内的 `runtime-output`。该回退仅适用于可写的开发环境，不能用于只读的离线训练代码目录。
- GPU 型号由用户确认为 **NVIDIA H20**。硬件型号不代表当前任务已获得可用 GPU；运行 CUDA 程序前应另外确认设备可访问。

本文件不保存 SSH 密码或其他认证凭据。
