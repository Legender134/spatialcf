[简体中文](README.md) | [English](README_EN.md)

# SpatialCF

SpatialCF 用于生成经过验证的空间反事实数据集。它从场景观测中冻结请求，使用最小代价
求解器规划单物体平面移动，通过 Adapter 在原生环境执行编辑，再对结果和数据集文件进行
fresh verification。

Schema、求解器和验证逻辑均为平台无关设计。Unity/AI2-THOR 是首个 Adapter，只负责把
平台事实和原生操作连接到这条公共生成链。

## 快速开始

需要 Python 3.11。以下命令创建环境、安装 AI2-THOR Adapter、生成数据，并重新打开数据集
执行验证和检查：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install "spatialcf[ai2thor]"
spatialcf generate --config configs/ai2thor-example.toml --output ./dataset
spatialcf verify ./dataset
spatialcf inspect ./dataset
```

`generate` 不会静默覆盖已经发布的数据集。`verify` 重新读取元数据、记录、资产和校验和；
`inspect` 仅在完整验证通过后输出摘要。

## 数据集内容

生成目录包含 `manifest.json`、`records.jsonl`、`report.json`、
`checksums.sha256`、内容寻址的 `assets/`，以及可恢复的 `.spatialcf/` 状态。
被拒绝的请求只进入报告计数，不会成为已接受记录。

## 文档

- [安装](docs/installation.md)
- [快速开始](docs/quickstart.md)
- [核心概念](docs/concepts.md)
- [Adapter](docs/adapters.md)
- [Python API](docs/api.md)

SpatialCF 采用 [Apache License 2.0](LICENSE) 许可。
