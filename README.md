[简体中文](README.md) | [English](README_EN.md)

# SpatialCF

SpatialCF 生成**可独立验证的空间反事实数据**：给定场景、目标和允许的修改，寻找使目标从假变真的修改，并保留可重算的证明。只有认证结果才发布为前后场景对；无解、尚不能证明和非认证可行解分别记录。

## 先运行一个 CPU 示例

需要 Python 3.11。以下使用当前 `main`，安装基础包即可，无需 Unity、AI2-THOR 或 GPU。输入由示例显式构造，输出是语义场景和证明，不是渲染图像。

```bash
git clone --branch main --depth 1 https://github.com/Legender134/spatialcf.git
cd spatialcf
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/general_dataset.py input.json --case placement
spatialcf general generate --input input.json --output dataset
spatialcf general verify dataset
spatialcf general inspect dataset
```

使用不存在的输入文件和输出目录。这个案例把地面上的盒子放到桌面，认证最终位置与最小位移。`verify` 从磁盘重新读取并验证证明；`inspect` 也会执行完整验证。复杂案例可能需要数分钟。

下一步读 [CPU 快速开始](docs/quickstart.md)，选择多物体、无解或未认证案例，并了解输出文件。

## 当前能力

| 任务 | 当前接口与边界 |
| --- | --- |
| 单物体平移、翻转左右/前后/远近关系 | v2/M2；固定高度与朝向，保留原 generation 契约 |
| 平移与绕自身/参考点转向 | M3 upright SE(2)；闭合子域可认证，其余诚实返回 UNKNOWN 或 witness |
| 生成有来源记录的语义对 | M4 semantic contrast；显式请求与预算，不生成 native 图像 |
| 放到表面、放进空腔 | M5 PLACE_ON/PLACE_IN；精确轴对齐盒体、水平支撑、显式矩形空腔 |
| 多物体三维姿态、关节与接触修改 | M6；有限精确盒体模型、关节森林、有序 edit set、每步前缀与交换性检查 |
| 批量生成、导出、独立验证 M5/M6 数据 | `spatialcf general generate/verify/inspect`；仅认证结果发布成对样本 |

当前链路：domain/core → adapter protocol → generation → fresh verification。核心负责求解与证明，Adapter 转换平台事实，生成与验证消费版本化产物。

证明只覆盖声明的模型和授权域。M6 的全域最优性/UNSAT 要求穷尽有限程序全集并通过独立 checker；连续域未封闭、资源耗尽、弱后端结果不能冒充 UNSAT。可行 witness 不等于已证最优。

当前不证明完整搬运路径、动力学、物理稳定性，也不支持任意网格和闭环关节。CPU 场景事实不是 native 观测认证。Schema、求解器与验证逻辑是平台无关设计；Unity/AI2-THOR Adapter 负责连接原生运行时。

## 输出与使用路线

CPU general 数据集包含 `input.json`、`catalog.json`、`terminals.json`、`records.json`、`report.json`、`provenance.json`、`manifest.json`。每个请求保留终态，只有 `PUBLISHED_PAIR` 进入成对记录。

原有 `spatialcf generate/verify/inspect` 面向 Adapter 数据集，文件格式与 general 不同，见 [Adapter 指南](docs/adapters.md)。两个入口不能混用验证命令。

## 版本和文档

用户仓库是 [Legender134/spatialcf](https://github.com/Legender134/spatialcf)。当前 `main` 含上述能力；`v0.1.1` 是历史 annotated tag，不含全部新接口，也不是 PyPI 发布。包元数据版本仍为 `0.1.1`，复现实验请同时记录 `git rev-parse HEAD`。本流程不创建新 tag 或 GitHub Release。

- [安装](docs/installation.md)
- [快速开始与结果解释](docs/quickstart.md)
- [核心概念与证明边界](docs/concepts.md)
- [Adapter](docs/adapters.md)
- [Python API](docs/api.md)

SpatialCF 采用 [Apache License 2.0](LICENSE) 许可。
