# 现役 720p 快速后端（2026-10-03）

本目录提取自已验收的 Cyberpunk 游戏安装：C512＋K8／all6、去 FP8 激活模拟舍入、融合真实运动与历史、计算图重放；游戏桥使用纹理复用和 GPU 栅栏接力。用户在此前固定场景验收的基础帧时间为 **55–60 ms**，帧生成关闭。此数值不是离线模型时间，也不是本次源码提取产生的新测量。

`runtime/` 保留实际安装的 Python／Triton 源码及条件依赖；`exact/backend` 是工厂所需的基础定义，不能据目录名认定游戏正在运行精确 4060 分支。`nr256_product_stack_v1` 是沿用的工厂名称，720p 入口不经 NR256 缩小处理。

`bridge/` 包含插件、共享纹理与线程交接的 authored 源码、测试和构建配方。Python 安装字节及 C++ 源码出处分别记录；重建二进制的编译环境与外部 SDK 单列。源码与某个安装 DLL 的哈希不是同一种证明。

## 固定的运行合同

- 输入 RGB：1280×720；真实 current-to-previous 像素运动：1280×720×2；内部工作面 1280×768。
- C512 library QKV、native K8；decoder gather unround、C32 hidden native、C512 probability unround、C128 pairwise、C64 attention/project、C128 attention/project 全部开启；C128 dual QKV 关闭。
- numeric cleanup：C128 分支 FP32 累积、FP32 分数历史值、原生 front/noise；风格标准、强度／局部色调／局部结构均 1。
- 保留完整 K、真实历史与运动、重置与种子、尺寸和控制语义。首轮预编译、图捕获与稳态重放分别记录。
- 当前计算混合使用 FP16／FP32／真实 INT8。原权重的 FP8 解码与必要量化尺度仍在；“去舍入”不表示所有内部操作都是 INT8，也不表示已与 NVIDIA 官方实现等价。

## 运行与复现

从仓库根运行 `python -I -B tests/validate_current.py`。完整模型入口、外供资产、隔离环境和输入合同见 [tests](../tests/README.md)；不需要启动游戏即可复测模型，但游戏帧时间和视觉验收仍需对应场景。

开发历史、失败实验和 AMD 结构对照见 [研究导航](../README.md)。`experiments/` 中的源码保留研究用途，未采用的候选不作为此目录的默认版本。
