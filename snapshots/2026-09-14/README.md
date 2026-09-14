# 2026-09-14研究源码快照

本目录同时保留精确优化、原尺寸快速适配和**NR256缩放实验（画面变化很大，仅作为实验路线）**。

源码导航及运行前提见[当前源码说明](../../docs/08-current-source.md)。实验结果见[本轮发现](../../docs/07-fullsize-fast-findings.md)，后续工作见[优化方向](../../docs/09-optimization-roadmap.md)。

目录维持原研究项目的相对布局。外部资产、量化profile、实验输入/索引、工具链、FFmpeg和缓存不随源码分发；脚本保留环境相关路径，不是下载后直接执行的安装包。`review_fast_fullsize_v1.py`中的gate用于检查适配是否改变旧NR256计算，full阶段才真正关闭缩放。

原始源码与发布文件的身份见[清单](../../evidence/source-manifest-2026-09-14.json)。模型资产条款及上游来源见[PROVENANCE](../../PROVENANCE.md)。旧快照与新快照不可合并为一次回归覆盖。
