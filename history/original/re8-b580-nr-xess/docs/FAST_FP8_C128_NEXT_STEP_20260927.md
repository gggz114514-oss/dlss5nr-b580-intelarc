# C32 去舍入之后的下一段：C128 独立激活舍入

状态：Astra 已按现役《村庄》G 盘快速后端静态审读；C128 D 和 U1 臂已完成 540p 首验，均未推广到游戏。C32 七块在 540p 离线快约 1.16 ms、480p 快约 0.58 ms；243 帧人脸对照已由用户肉眼审核为“看不出区别”，解码视频全幅平均 SSIM 0.996149。此结论限于该素材，游戏灯光场景仍待验。基础记录见 [快速版去舍入实验](FAST_FP8_REMOVAL_ABLATION_20260927.md)。

选择 C128 先试：现役 480/540 共有 6 个编码块和 6 个解码块；代码静态账对应 64 个独立激活 `q`，须在捕获图重新核对。C64 的八块已装 direct-pack，作用域更复杂；C256 有十六块、尺寸更小。首轮应与当前 **G 盘三结构组合**比较，暂不叠加仍待画质审核的 C32 有损候选。

## 必须区分的边界

| 边界 | 当前路径 | 实施约束 |
| --- | --- | --- |
| ① 块入口 | `WindowBlocks.apply` 先 `q(features)` 再 pad | 与 ② 成对看；保持原 pad/shift/布局 |
| ② MLP 入口 | `batched_branched_mlp_v1.forward` 再 `q(features)` | 先可只去②，验证重复转换核；再与①一起去，才让 MLP 吃 FP16 |
| ③ MLP 输出 | `BranchedMLP.forward` 发布时 `q` | 与 ④ 成对看；保持融合 cubic LUT 与 reduce 舍入 |
| ④ QKV 入口 | `WindowBlocks.windows` 再 `q(features)` | C128/C256 可实例覆盖；C64 的 direct-pack 需另做 |
| ⑤ 块发布/池化/下采样 | `MultiHeadSwinBlock.forward/forward_outputs` | 保持从未舍入的 `full` 池化、半精度运算次序和 down 四对齐 pad |
| ⑥ decoder 输入与 skip | `DecoderGather.upsample/merge` | 仅目标 C128 的 decoder 转接点；融合合并核里的舍入留给后轮 |

建议在 E 盘新建实例级作用域，使用对象身份锁定 12 个 C128 块及其解码转接点，保留 G 盘原对象及源码哈希。不要全局替换 `nr_backend.multihead_block.quantize_fp8`：其他 head layout 会经此调用，形状守卫也不足以证明没有旁路误伤。`stack.window_blocks.apply/windows`、`FusedBatched.apply`、目标 `block.mlp.forward` 和 `block.forward/forward_outputs` 分别隔离覆盖；不改已由 WindowBlocks 类作用域持有的 `forward_unquantized`。decoder 仅改目标实例的 `upsample/merge`，其安装 `forward` 会动态调用这两个方法。退出顺序先还原候选覆盖，再退出原组合，保留原身份校验。

测试先做两个臂：**D** 只去②和④，保留①和③，核对是否仅消掉重复舍入核、RGB 是否仍一致；**U** 再去①、③、⑤、⑥，接受有损输出并看整帧收益。每个臂都要在新会话捕获，D 盘隔离缓存准备后用新进程只读缓存做 B-D-D-B 与 B-U-U-B；记录 480/540 的中位数、P95、有限值、实际 capture/重放、消失的转换核和新增拷贝。目标 C128 的 12 块均须命中，非目标族的调用必须保持原路径。用户审核 C32 视频后，再决定是否制作 C128 连续视频与组合候选。

融合核内的 QKV、attention、MLP cubic/LUT 舍入，权重 E4M3 解码和 INT8 scale **均不在这一轮**。C64 后续须单独处理八块 direct-pack 已占用的 `windows` 实例绑定；不能把 C128 覆盖直接套到 C64。

## D 臂 540p 首验

[D 臂作用域](../game/c128_activation_unrounded_v1.py)仅跳过 12 块的 MLP 输入和 QKV 输入重复转换，现役 C64 direct-pack 仍拥有非目标实例。Luna 用隔离 D 缓存先准备图，再以新进程只读缓存做 B-D-D-B；每段 120 计时帧均为图重放，输出有限、12 块的 MLP/QKV 捕获命中。prepare 与四段末帧的 RGB 字节哈希一致。两张准备图合计少 48 次独立 FP8 转换核，即每次捕获少 24 次，其他物理 Triton 核数不变。原始记录：[准备](D:/Codex-NR-Experiments/nr-b580/c128-activation-unrounded-v1-20260927/prepare-540.json)、[计时](D:/Codex-NR-Experiments/nr-b580/c128-activation-unrounded-v1-20260927/benchmark-D-540.json)。

| 540p | 基线三结构组合 | D 臂 | 差异 |
| --- | ---: | ---: | ---: |
| 240 帧合并中位数 | 50.1651 ms | 49.9972 ms | −0.1680 ms（−0.33%） |
| P95 | 50.7337 ms | 50.8392 ms | +0.1055 ms |

两组配对的中位收益分别为 0.1373 和 0.2424 ms，但 P95 都略退。此处减少 24 次发射仅带来较小整帧收益，不能把核调用数直接换算成毫秒。还没有逐帧读回全部计时输出、480p 验证或游戏实机验收；**D 臂未安装为默认**。

## U1 候选边界

[U1 作用域](../game/c128_activation_unrounded_core_v1.py)在 D 上继续去掉块入口、MLP 发布、块发布/池化/下采样的独立激活舍入；C128 解码合并仍使用原有融合内核，故 U1 不是完整去 q⑥。Astra 静态核对了当前 G 盘调用、pad、half 顺序和目标对象隔离，未发现必然错误。它会改变后续 C256 的数值，整帧和完整视频仍须单独验证。U1 的计时控制臂是已通过的 D，不能把 D＋U1 的总收益都归因于新增改动；建议先测 540p 最小判别，有效后再做 480p、视频与用户肉眼审核。

Luna 随后完成 540p D-U1-U1-D 配对：每段 120 帧均为图重放，12 块命中、有限值、只读缓存均通过；两臂输出哈希不同，各自重复稳定。两张准备图合计少 76 次独立 FP8 转换核，捕获期 `aten::copy_` 两臂均为 2 次、`aten::contiguous` 均为 169 次；这些调用数不是搬运字节量。原始记录：[准备](D:/Codex-NR-Experiments/nr-b580/c128-activation-unrounded-v1-20260927/prepare-U1-540.json)、[计时](D:/Codex-NR-Experiments/nr-b580/c128-activation-unrounded-v1-20260927/benchmark-U1-540.json)、[拷贝审计](D:/Codex-NR-Experiments/nr-b580/c128-activation-unrounded-v1-20260927/prepare-U1-540-copy-audit.json)。

| 540p | D 控制臂 | U1 有损臂 | 差异 |
| --- | ---: | ---: | ---: |
| 240 帧合并中位数 | 49.8701 ms | 49.7207 ms | −0.1495 ms |
| P95 | 50.6798 ms | 51.0677 ms | +0.3879 ms |

两组配对中位差分别为 +0.2158 和 −0.2846 ms，方向相反。因此这轮没有稳定整帧收益，**不做 480p/视频推广，不安装游戏默认**；保留隔离代码和测量结果。此结论仅限当前 U1 边界，不能外推为 C128 所有融合核内舍入或其他家族均无价值。
