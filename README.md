# DLSS5 NR on Intel Arc B580：开发记录、现役内核与复现测试

**2026-10-03 更新。** 现役主线是 **720p C512＋K8／all6 快速后端**：去 FP8 激活模拟舍入、融合真实运动与历史、计算图重放，配合已验收的纹理复用／GPU 接力游戏桥。在此前《赛博朋克 2077》固定场景、帧生成关闭时，用户验收基础帧时间为 **55–60 ms**。当前尚未达到 720p 基础 30／60 帧目标。

本仓库保存从固定 RTX 4060 参考恢复计算，到 B580 精确后端、快速算术、真实游戏桥、原生矩阵与完整模块改造的开发过程。成功、失败、未采用候选和复现缺口均保留，供后续 Triton／Intel GPU 内核开发参考。

## 阅读与代码入口

| 目录／记录 | 内容 |
|---|---|
| [开发全过程](docs/history/DEVELOPMENT_HISTORY.md) | 按阶段整理尝试、判断、纠错和结果 |
| [最终验证状态](docs/history/ARCHIVE_RECONCILIATION.md) | 补齐历史报告交付后完成的复现、构建、负收益与清理 |
| [实验索引](docs/history/EXPERIMENT_INDEX.json) | 源码、脚本、输入合同、证据、采用状态和复现限制 |
| [现役源码](current/README.md) | 当前完整运行依赖与桥的源代码身份 |
| [复现测试](tests/README.md) | 无 GPU 的发布检查、隔离预编译、只读模型执行与对比 |
| [最新测试进度](docs/audits/TEST_PROGRESS_20261003.md) | 完整帧收益与负收益，分别标明离线／游戏范围 |
| [AMD 结构对照](docs/audits/amd-current720/REPORT.md) | 固定提交的完整模块、供数、布局与新增改造计划 |
| [AMD 内部执行配方](docs/audits/amd-kernel-implementation/FULL_MODULE_PORTING.md) | 线程、fragment、供数复用、生命周期与 B580 移植步骤 |
| [小通道执行配方](docs/audits/amd-execution-recipe/C32_C64_C128_RECIPE.md) | C32/C64/C128 的所有权、寄存器与同步预算，失败实现偏差 |
| [AMD 分辨率与编译](docs/audits/amd-resolution/REPORT.md) | 预编译内核与运行时几何的区别 |
| `experiments/` | INT8、融合与最新 r18 候选源码；未通过者不进入现役默认 |
| `backend/`、`reference/`、`snapshots/` | 既有精确恢复、4060 插桩、早期快速路线的历史快照 |

## 性能与画质证据

不同尺寸、版本和计时边界不能直接相加。现役 720p 离线 `FullsizeGameModes.process` 在本轮配对中约 45–46 ms；独立 GPU 主体约 39–40 ms。这些不含完整游戏、桥或 Present。55–60 ms 是此前用户实机验收值。

本轮 FDP 标准候选离线独立配对平均快约 1.80 ms，但输出改变、尚未部署游戏；其他原生 QKV、C512、Swin、ViT 候选中有完整帧负收益。源码可用、使用 XMX、减少中间张量或零 spill，均不能单独证明完整帧提速。原始记录与经过路径清理的公开摘录分别列 SHA。

快速版允许经过技术验证和用户视觉接受的数值变化。精确版的字节一致仅针对冻结 SF-v2／RTX 4060 输入与状态范围；不能扩展为 NVIDIA 官方 RTX50 的普遍一致。NR256 是历史实验，不是当前默认路线。

## 最小检查

```console
python -I -B tests/validate_current.py
python -I -B tests/cpu_tests.py
python -I -B tools/verify_release.py
```

完整模型需要用户另行提供匹配的权重、数值表、校准资产和 Torch XPU／Triton-XPU 环境。准备与 GPU 命令见 [tests/PREPARATION.md](tests/PREPARATION.md)。公共固定种子输入可验证执行和重放；复现历史画质数值需相同原输入，不能以合成样例替代私有人脸视频或游戏测试。

模型权重、第三方运行库、游戏素材、编译缓存及私有 GPU Block／DIS 实现不随源码分发。源码来源、许可及资产 SHA 见 [PROVENANCE.md](PROVENANCE.md)。本次是研究源码归档，游戏便携发行包有独立的安装与依赖边界。

2026-09-09／09-14 的成果与数值规则仍保留于既有 [恢复研究](docs/01-reconstruction.md)、[复现边界](docs/06-reproduction.md) 和 [原尺寸快速发现](docs/07-fullsize-fast-findings.md)，没有用后来的快速结果覆盖早期记录。
