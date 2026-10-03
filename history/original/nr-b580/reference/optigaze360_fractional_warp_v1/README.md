# 360p 分数运动历史采样融合候选

这是**隔离实验**，没有修改 `backend/nr_backend/sampling.py`、产品、精确版或快速版。480p→360p 的准备动作会把 Block 整数运动缩小为分数运动；现有 `_sample_five_axes` 的整帧单点快路径因此不再适用。已完成的 GPU pre 分项：历史采样 480p 中位 1.633 ms，360p 中位 13.551 ms，差 11.918 ms。此差值说明这里值得优化，但不能把任何候选速度视为已通过。

## 做法与语义边界

`fused.py` 包含两个候选 Triton 核：

1. 将 `warp_history_normalized` 中的半精度运动、归一化坐标、FP32 FMA、三次权重、原生倒数表查找与中间坐标融合为一次写轴参数。
2. 从这些轴参数一次完成五点纹理采样、原整数权重与边界钳位、半精度纹理舍入、按原顺序的 FP32 FMA 和总权重倒数。输出仍保留 `numerator` 与 `reciprocal` 两个分量，避免改变后续 post 的融合顺序。

没有对分数运动取整，没有用近邻采样代替五点采样，也没有把模型历史改成展示帧。实现按 `sampling.py` 的整数纹理计数和负零处理逐项转写；**转写本身不是正确性证明**。Triton 与 PyTorch 的 FMA/半精度转换必须由 GPU probe 确认。

## Luna 的独立 GPU 入口

```powershell
powershell -NoProfile -Command "& 'E:/ComfyUI-aki-v3-IntelArc_20260722/ComfyUI-aki-v3-IntelArc/python/python.exe' 'E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/optigaze360_fractional_warp_v1/launch.py' --frame 100"
```

必要时再跑 `--frame 1`、`--frame 10`，并在第一轮编译/正确性成功后用 `--repeats 24` 复测。源码位于 E:；缓存、临时文件与 JSON 报告都在 `D:/Codex-NR-Experiments/nr-b580/optigaze360-fractional-warp-v1/`。编译环境使用已验证工具链和 14 个编译线程。入口读取冻结人脸视频的前一帧 RGB（缩到360p后转 FP16）与当前帧实际缩放 Block 运动；这是代表性采样输入，**不是**运行 NR 后的私有历史。完整链路的最终速度仍须另测。

`probe.py` 逐阶段比较：轴参数 → 五个纹理 tap → 五 tap 累加的 numerator → 原生 reciprocal → 归一化值 → 两核串联结果 → 公开参考接口。它另外生成包含负零、FP16 次正规数、图像边缘外运动的 4×5 合成 GPU 输入；这个输入只作正确性检查。每阶段统计 FP32 位模式不相等数、最大/p99/平均绝对误差；另有候选错误索引计数。**只有所有阶段字节一致且索引有效，才有无损接入资格。** 数值门（最大绝对误差≤2e-4、p99≤1e-5）仅标记是否值得继续研究有损快速版，不等于精确通过。计时分别列参考整段、参考坐标、参考五点、候选坐标、候选五点和候选整段，GPU event 与同步墙钟都记录；不要把此微基准与整帧 FPS 混同。

本地仅运行了 `cpu_check.py`：220 个坐标、660 个 FP32 分量与后端 CPU 纹理整数公式逐位一致；同时通过 AST 解析与 probe 命令行加载检查。这些检查**不能替代**上面的 B580 GPU 门。

待验证：Triton 编译、每阶段字节一致、各种纹理值和边缘情况、真实私有历史、长视频画面/帧数/音频及 243 帧稳定速度。若 GPU 报错或逐字节失败，保留报告与候选，不接入任何现有运行链路。
