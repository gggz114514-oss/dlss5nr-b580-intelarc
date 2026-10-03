# Uzbekunknown/dlss-nr-on-intel 对 B580 路线的适用性（2026-09-23）

只读核对公开仓库 `3055b951bd6bfd7309fb67715b1b48998d391cee`，并对照本地此前的 `D:/Codex-NR-Experiments/nr-b580/reference/community/dlss-nr-on-intel-review-v1/REVIEW.md`（旧 `ac6d2a1` 快照）与本项目当前记录。本轮未运行 Vulkan、B580 或游戏，也未更改推理代码。

## 结论与证据边界

该仓库用 Linux/Mesa 的 Vulkan `VK_KHR_cooperative_matrix` 在 Arc 140V 核显执行完整 71 块网络，FP16 操作数、FP32 累加。它独立恢复的模型结构、16 通道输入和四通道 head 与本项目已有语义可交叉核对，但不是 B580 上更快的现成后端。作者给出的 720p 图约 488–495 ms、GEMM 约 216 ms，主要来自 140V；公开 B580 用户最初遭遇的是 Vulkan 缓冲区错误地落在系统内存，导致极慢。修订版的 device-local／staging 代码在核显强制路径上验证，作者仍明确注明未在独显上验证修订后的整网帧时间。因此不能拿两项目的 fps 直接比较或推导 Vulkan 比现有 Triton 更快。[项目说明](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/3055b951bd6bfd7309fb67715b1b48998d391cee/README.md) · [分项计时](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/3055b951bd6bfd7309fb67715b1b48998d391cee/notes/phase45-frame-profile.md) · [B580 内存问题](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/3055b951bd6bfd7309fb67715b1b48998d391cee/notes/phase63-a-discrete-gpu.md) · [修订范围](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/3055b951bd6bfd7309fb67715b1b48998d391cee/notes/phase65-the-discrete-memory-path.md)

## 优先方向

1. **原尺寸快速版：按真实设备时间定位高价内核，然后只对命中的算子改数据流。** 对方用每 pass 的 GPU 时间戳而非删算子差分识别耗时，且发现最初的手写 profiler 名称表曾错位。我们的当前 480p 最小口径前五项合计约 18.45 ms；其中 `fused_c32_projection_native_half_v1`、`fused_swin_core_native_half_v1`、`fused_c32_mlp_lut_v1` 合计约 8.20 ms/30 次，尚无相同口径的 roofline 分类。先使用已落盘 trace 核对核间空隙和名称映射，再对这三项分别测实际输入／输出字节、吞吐、寄存器与 spill，并与 B580 自测约 390 GB/s 的渐进带宽相比。若已贴带宽上限，应减少读写趟数／中间张量；若没有，才研究指令、排布和并行度。验收是完整原尺寸 480p 交错配对帧时间和快速基线字节一致，而不是核数减少。[对方方法与误标更正](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/3055b951bd6bfd7309fb67715b1b48998d391cee/notes/phase45-frame-profile.md) · [本地当前画像](reference/handover_v3/HANDOVER_v3.md)
2. **低分辨率支线：同一低分辨率网络输出下比较 head 先放大再合成。** 对方在低分辨率执行 NR 后，放大四通道 head（RGB 增量及可用时的时域门），再与未经缩放的原图和历史合成。我们的 360p 滑块当前从低分辨率最终 RGB 减基图，按原图引导重建增量，概念相近但顺序及门的处理不同。固定同一段 480p 视频、同一低分辨率网络 head 和历史，只替换显示端，比较肤色、纹理、闪烁与完整链计时；保留用户视频审核。作者半尺寸结果虽然快约三倍，但高频变化与全尺寸结果仅约 62% 同向，不能认定 head 放大可恢复原尺寸细节。我们的下游 XeSS SR 可单列实验，不能用该项目的 62% 直接预测画质。[低分辨率实验](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/3055b951bd6bfd7309fb67715b1b48998d391cee/notes/phase37-neural-upstream.md) · [其 head 合同](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/3055b951bd6bfd7309fb67715b1b48998d391cee/docs/ARCHITECTURE.md) · [本地重建](reference/optigaze_resolution_slider_v1/adapter.py)
3. **产品执行层：确认模型与中间张量常驻 B580 显存，宽度切换不重复上传权重。** 对方独显异常的根因是主机可缓存内存与设备本地内存被混淆；后来把不随尺寸变化的权重移出逐尺寸对象，使第二尺寸首帧 335→82 ms（在核显上模拟测得）。本项目的 PyTorch XPU／全 GPU 链通常不会走同一 Vulkan 分配器，故这是排查清单，不是已发现的本地 bug。只需在产品实际链路量一次设备内存、CPU↔GPU 传输和换档首帧；若已常驻则结案。[独显修订](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/3055b951bd6bfd7309fb67715b1b48998d391cee/notes/phase65-the-discrete-memory-path.md)

## 不重复投入的方向

- 完整替换为 Vulkan cooperative-matrix：同是 Xe2 XMX，不能凭接口名称推断比现有 XPU 快。我们已有 ESIMD/DPAS 局部实验，部分真实矩阵原语更快，但初值累加顺序出现 half 字节差异，且没有整帧收益证明。只有单个代表性形状连同布局准备、舍入和完整帧都胜出，才值得扩大。[本地 ESIMD 记录](../nr-b580-int8/experimental/ESIMD_DENSE_STATUS.md)
- 图捕获／重放、FP16 分块、删除重复 FP8 量化：本项目已经实做并验收过。对方 `NR_JOINT_QKV` 虽少 140 次 dispatch，却让其核显温态帧 79.05→83.21 ms；不能把少核数当收益。[对方 QKV 实验](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/3055b951bd6bfd7309fb67715b1b48998d391cee/notes/improve-joint-qkv.md) · [本地 FP8 图改写](../nr-b580-int8/experimental/FP8_GRAPH_REWRITE_STATUS.md)
- 把 Python 外围改 C：对方的 74→28 ms 是 CPU NumPy 全帧预后处理的收益；我们主要在 XPU 图内做这些工作，不能把 2.6 倍套到 B580。其输出 head 回读主机与游戏 Vulkan 层双 `vkQueueWaitIdle` 也是该项目的产品接口问题，不应带进本项目的全 GPU ComfyUI 链。[对方 C 路线](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/3055b951bd6bfd7309fb67715b1b48998d391cee/notes/phase57-native-host-passes.md) · [Vulkan 同步现状](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/3055b951bd6bfd7309fb67715b1b48998d391cee/notes/phase66-the-present-has-a-test.md)

独立模型规格可辅助查接口错误，不替代本项目冻结 4060 精确分支的逐字节门。对方关于“浮点移植无法逐元素相同”的说法属于它自己的 FP16/FP32 执行路径，不能用来否定本项目已验证范围内的精确输出。
