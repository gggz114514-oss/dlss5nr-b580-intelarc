# 连续帧后端验证

2026-09-08：`ZeroMotionNR` 已独立完成 256×256 四帧 RGB→RGB 连续运行。
重置模式 `[1,0,0,1]`，随机种子 `[0,1,2,0]`；四帧最终 RGB32F 各
786,432 字节均与固定 SF-v2 RTX4060 Laptop 参考逐字节一致。
验证报告：`reference/results/temporal-rgb-b580-v2/validation.json`。

| 帧 | 重置 | 不同字节 | B580 秒/帧 |
|---|---|---|---|
| 0 | 是 | 0 | 20.58 |
| 1 | 否 | 0 | 19.62 |
| 2 | 否 | 0 | 19.94 |
| 3 | 是 | 0 | 19.89 |

第四帧与第一帧完全一致。测试还在每帧返回后清零调用方拿到的张量，验证下一帧
使用的是后端私有历史副本；显式 `reset()` 清空历史和种子。运行时只读原始 RGB、
固定权重和标量函数表，历史来自本次 B580 自己算出的最终 RGB16F。
4060 输出只供测试结束后的比较，不参与推理。按用户最新规则，完全一致的结果免肉眼复审。

## 已验证计算

- 前端通道 7–9 使用上一帧最终 NR 输出，零运动时与当前 RGB 使用相同反射坐标。
  `reference/results/temporal-pre-v1/validation.json` 另验证原生 pre 独立重放两帧的
  skip/down 字节完全一致；使用上一帧原图的反例明显不一致。
- 后处理保留八个 head 通道，以通道 3 的 sigmoid 和原权重中
  `block70.layer0.blend_scale = 0.73974609375` 控制历史融合。当前基色截断到 [0,1]，
  历史融合后按原生表面存储规则向零舍入至 FP16。上述零运动样例最终值仍在 [0,1]；
  后续非零运动测量证明最终融合值可超界，因此不能将最终截断解释为通用规则。
- sigmoid 表穷举全部 65,536 种 half 输入，256 KiB，不含图像或模型中间特征。
  加载时验证固定 SHA-256：
  `394394a5258bad437495d68076be75d8413fa0ed5b752400e3947c332437b850`。
- 原生时序 post 独立重放、B580 隔离 post 两帧均已通过，分别见
  `reference/results/temporal-post-v2/native-validation.json` 和
  `reference/results/temporal-post-b580-v1/validation.json`。

## 调用

```python
from nr_backend import ZeroMotionNR
model = ZeroMotionNR.from_assets(
    'model-assets/sf-v2/WEIGHTS_HT.bin',
    'model-assets/noise-sm89-v2',
    'model-assets/sigmoid-sm89-v1',
).to('xpu').eval()
first = model(rgb0, reset=True)  # HWC3 floating point, 256×256
second = model(rgb1)
model.reset()
```

## 边界与后续工作

本报告只证明上述 256×256、SDR、默认控制、零运动/零深度序列。非零运动已有独立
后续报告。深度/控制参数、任意尺寸、真实视频长序列、XMX/INT8 性能与 OptiScaler/
现有工具集成仍待完成。不能据此声称完整迁移完成或通用逐位等价。

报告 v2 保存当时源码哈希。全尺寸显存分批改动已在
`reference/results/temporal-rgb-b580-v3/validation.json` 重新通过四帧零差异验证。
非零运动采样、带符号历史及post融合的后续证据见 [MOTION_SAMPLING_STATUS.md](MOTION_SAMPLING_STATUS.md)。
