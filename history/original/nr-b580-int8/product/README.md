# NR B580 产品组件 v1

本轮停止新增算子优化，将用户已经接受的快速画质路线收成一个固定入口。
状态：整合候选，完整产品 API 连续视频验收通过前不视为可发行版本。

入口为 `nr_runtime_v1.Session`，构造一次、每个真实帧调用 `process`、片段切换
调用 `reset`，结束调用 `close`。权重及已校准范围在构造时装载，正常帧不重新
校准或编译已有图。不同视频使用不同 Session；本 API 在进程内串行执行。

## 调用契约

- 颜色：XPU 上 float32、HWC RGB、SDR [0,1]；不能直接传 BGRA/NV12。
- 运动：同设备 float32 HWC2，当前帧指向上一帧，单位为源尺寸像素。
- 首版已选尺寸：1920×1080、864×480，以及直接调用 NR 的 256×256。
  改变源尺寸、跳剪或时间轴跳转必须 reset；其他尺寸明确拒绝。
- 返回源尺寸 float32 RGB、256×256 half NR 输出、序号和 reset 信息。
  调用者可持有或修改返回值，不能污染会话私有历史。
- 真实帧顺序和时间戳由调用方保持；XeFG 生成帧不回灌 NR 历史。
- 运行异常后关闭并重建会话；不静默改用另一套算术继续处理。

```python
from nr_runtime_v1 import Session

with Session.create(exact_root, profile_path, profile_sha256) as nr:
    result = nr.process(rgb_xpu, motion_xpu, reset=scene_cut)
    enhanced_rgb_xpu = result.color
```

当前运行环境仍使用本机已固定的 PyTorch XPU / Triton 3.8 / Intel 编译器，
导入前将 `product`、`experimental`、`backend` 和固定 Triton site 放入模块路径。
产品标定包保存在 D:/Codex-NR-Experiments/nr-b580/product-v1/；不读取巨大实验
报告来逐帧运行，也不在 E 盘复制视频、模型或编译缓存。

## 收拢范围

采用已有矩阵分块、XMX 混合精度、MLP/注意力融合、短 FP8 舍入、图执行和历史
采样、C512/ViT 布局、post 依赖裁剪、已修复范围的 ViT INT8、已目视接受的 C512
floor16、已验证字节一致的原生 query 布局，以及本轮 decoder gather。
详细取舍与数值边界见 ../BODY_OPTIMIZATION_SUMMARY.md。

这套快速输出不声称与 4060 原生逐字节相同；独立 nr/exact 后端仍保留。
“输出不变”指整合前后等于已获用户接受的快速输出。旧实验文件与默认工厂冻结，
通过新增固定入口收拢，避免破坏历史验证记录。

## 产品接入顺序

1. 完成该入口的独立历史、异常拒绝、尺寸切换、13 帧配对及完整 243 帧人脸验收。
2. 补 NR XPU 张量与隔壁 D3D12 帧纹理的导入/导出和 fence；复用既有运动/深度。
3. 在独立产品入口接 NR、XeSS-SR、XeFG，生成帧最后处理；实际比较 NR/SR 顺序。
4. 测完整处理、帧节奏、退出与显存回收，再做安装包和用户最终视频验收。

尚未提供 D3D12 互操作或完整实时窗口，不能把该组件当成已完成的实时产品。
本轮不自动发布 GitHub，不改隔壁已经发布的默认链路。
