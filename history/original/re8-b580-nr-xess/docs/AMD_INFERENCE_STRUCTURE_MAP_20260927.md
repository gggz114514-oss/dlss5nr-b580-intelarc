# AMD 推理执行结构与 RE8/B580 原尺寸快速线对照（2026-09-27）

范围：只对照 960×540 原尺寸快速 NR 的模型执行结构；不以 NR256 为目标，不修改精确分支、游戏安装或用户环境。本轮为源码和既有测量审计，**没有新的 GPU 提速实测**。AMD 固定源码为 [`0bf535c`](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/0bf535c64c35bc41d81fb32191bb40984887ca70)。本地运行路径要看 `game/fullsize_session_v1.py` 安装的 `stack.components` 和 `game/re4_session_v1.py`／`game/nr_game_fullsize.py` 的尺寸适配，不能从基础 `nr_backend/triton_math.py` 推断最终分派。

## 比较口径

RE8 的 960×540 输入走 544×960 模型面、640×1024 内部画布；240 帧图重放整链中位数 53.93475 ms，另轮模型段 54.429 ms。AMD 900/1080 网络档的计算画布为 1600×960／1920×1152；旧 900/1080 回放约 12.1／17.1 ms，新组合候选生产 host 长测约 11.26／15.86 ms。这些结果使用不同设备、精度、时序、块配置和计时合同。按画布像素粗算的约 10 倍差距只是寻找结构低效的信号，**不是同工作量的加速目标**。[本地基线](PERFORMANCE_ARCHITECTURE_20260925.md) · [AMD 几何](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/src/native_network_geometry.h) · [AMD 组合长测](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/Development/results/wave-owned-combined-20260926/README.md)

## 逐段结构映射

| 阶段 | AMD 当前快速链 | RE8/B580 当前快速链 | 借鉴点与限制 |
|---|---|---|---|
| 输入／时序 | D3D12 编码到固定网络档并反射填充；超分前路线每帧重置网络自身历史，时域效果主要交给宿主 FSR。 | XeSS 前取游戏渲染帧，360/480/540 可选；540 档保留 NR 历史和运动矢量，并已用图重放。 | 资源常驻、预分配、编码合并可比；**不能直接复制每帧重置历史**，那会更改已验收的运动/灯光行为。此段桥接已测为小头，不是第一优先级。 |
| pre／C32 | `prefix_fast` 与 C32 的前馈、QKV、64-token 注意力、投影朝同一窗口／wave 组织，隐层尽量在寄存器中；0.31 对 C32 用单 wave 负责一窗口。 | `pre.forward_features_outputs` 后按块调用；GraphFront、FusedSwin、FP16 XMX 等优化已在运行栈中，仍有独立核和中间张量。 | 借鉴**完整窗口拥有权与生产者直接交给消费者的布局**，按 B580 subgroup/XMX 和寄存器限制重新设计。NR256 的全 C32 融合曾在 Triton 3.8 下保持字节一致但首模块略慢；尾段融合局部较快而整网收益不稳，不能当作 540p 成功或失败结论。 |
| C64/C128/C256 | 多头注意力按一头一 wave 组织，softmax 中间值保留在片上；生产配置针对各通道不同，不强行套一个核。 | FusedSwin、QKV pack、窗口注意力等已有专项优化，但仍保留多个 FFN／QKV／注意力／投影边界。 | 测完整多头段的物理读写、XMX 活跃与下游格式。优先尝试输出布局直通或 attention+projection 的有界融合；保留 shift、归一化和舍入顺序。 |
| C512／ViT | 静态权重预排 WMMA 片段；残差可跨块保持 E4M3 字节；FFN 以 FP8 为主，QKV 使用 FP16；C512 将一 wave 的 token 宽度从 16 提到 32，ViT 投影扩宽列片段。 | C512 与 ViT FFN 已由行级 INT8 XMX 核接管，QKV／注意力仍有 FP16/FP8 边界；已有紧凑 C512 QKV、ViT 布局/投影优化，不能重复计算旧收益。 | 借鉴权重片段复用和**连续边界**，先查生产输出是否能被下一核直接消费。B580 没有同样的原生 FP8 矩阵路径；纯 INT8 的早期候选整段未获益且画质可能变化，不能照搬 AMD E4M3 字节流。AMD 的 C512 大融合与部分 ViT 候选也曾无益或溢出。 |
| 解码／post／RGB | 保留不同通道的块特化；输出与原画面经专用 D3D12 解码合成，再交给超分。 | decoder 重用独立块与 skip；post 包含注意力、RGB 头点积、投影、历史混合和显示端合成。 | 当前 540p 图内标记把 post 列为需细查候选；优先查 attention→RGB head 连续段和 `_tiled` 回退，再查尾端融合。AMD 的显示合成与 B580 NR 输出语义不同，不能直接替换。 |
| 整图提交与缓存 | HIP 生产核按网络阶段组织，模块预编译、权重预打包和常驻；可选依赖式提交。29 个 HIP 模块是二进制文件数，**不是每帧 29 次启动**。 | PyTorch/Triton 静态图捕获重放、共享图池、预编译缓存已存在；图内仍有大量计算任务。 | 重写 Python 调用入口、增加一层图包装不会减少 GPU 图内核数。优化点是算子边界、片上复用、输入/输出布局，而不是仅换语言或换提交 API。 |

AMD 推理事实来源：[完整工程导读](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/Development/docs/Project-Intro.md)、[模型和数值说明](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/README.md)、[当前配置](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/scripts/hip-game-flags.txt)。B580 运行路径与负结果：[RE8 会话](../game/fullsize_session_v1.py)、[模型主体](../../nr-b580-int8/experimental/capture_body_v1.py)、[C32 融合记录](../../nr-b580-int8/experimental/C32_WINDOW_FUSION_STATUS.md)、[C32 整网结果](../../nr-b580-int8/experimental/C32_TAIL_AND_COMPILER_STATUS.md)、[图内画像](GRAPH_STAGE_PROFILING_20260925.md)。

## 决策：模仿执行组织，逐段验证，不照搬内核

1. **先列真实执行图。** 对 540p 实装栈记录每个阶段的矩阵形状、实际 provider、FP16/INT8/残余 `_tiled` 分派、FP8 舍入边界、临时张量生产/消费、编译资源与 GPU 时间。已有 VTune 标记轮有桌面活动和分析器干扰，`valid_for_performance=false`；只能提供查找位置，不可用其毫秒数承诺收益。
2. **第一个原型选可控的连续段。** 在 pre/C32 或 post 中取实测有分量的一个完整生产者→消费者链；先维持现有 FP16/FP8 数值边界和输出字节，再把共享窗口数据留在片上或让下游直接读上游布局。让 B580 决定 tile、subgroup、工作组数量；如果 Triton 无法有效表达布局或资源分配，再对这一段试 SYCL/ESIMD/原生 XMX，不把整网一次性改写。
3. **以整帧验收。** 同尺寸、同图状态下对照模块输出、RGB、连续历史和完整游戏链时延；仅在局部与整帧均稳定获益后推广到同类块。若数值边界变化，则作为快速线有损候选并按视频画质门验收；精确线继续单独守冻结 4060 字节合同。AMD 0.31 的 46 块窗口/头改造在其宿主长测约改善 6%～7%，是结构方向的正证据，不是 B580 的可套用收益。[AMD 组合回归](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/Development/results/wave-owned-combined-20260926/README.md)

这份映射的结论是：**应当以 AMD 整网结构为设计参照，但不应把 HIP/Wave32 的具体核直接搬到 Xe2。** 最大待证假设是 B580 的小形状 XMX 有效吞吐和连续层数据驻留不足；先用现有 540p 路径的真实执行清单把二者分开，再动第一段。

## 540p 实装分派核对（2026-09-27）

隔离脚本 [`tools/audit_540_execution.py`](../tools/audit_540_execution.py) 在 RE8 便携运行时上仅使用已有编译缓存，执行三帧 960×540、带历史的模型图；三帧均确认为图重放，输出有限，报告为 `D:/Codex-NR-Experiments/nr-b580/re8-amd-architecture-inventory-20260927/dispatch-540.json`。六次捕获期模型主体调用各记录 **958 次 Triton 内核发射请求**；其中独立 E4M3 舍入核 `nr_backend.triton_fp8._kernel` **383 次**、通用 `fast_matrices_v3._matmul` **115 次**。矩阵 provider 的 125 次调用里，107 次是通用 `fp16_dense`、16 次是 FP16 batched、2 次是原精确 K8；`fp16_tiled` 为零。`CurrentDenseTiledMatrices.POLICY` 来自早期 NR256 形状，540p 未命中，但通用路径仍使用 `tl.dot`/XMX，不能称为标量回退。计数只说明执行图结构，**不是单核耗时或可节省毫秒**；图重放不重新调用 Python，故此清单取自捕获期。

按阶段，独立 E4M3 舍入核出现在 ViT 64 次、编码和解码 C512 合计 82 次、C256 合计 84 次、C128 合计 64 次、C64 合计 44 次、C32 合计 35 次，pre/post 各 4 次。是否可与生产者输出融合，要逐个确认下一消费者与该 FP8 边界的读者，不能据次数直接删去或重排。配合 [`GRAPH_STAGE_PROFILING_20260925.md`](GRAPH_STAGE_PROFILING_20260925.md) 的**受背景 GPU 活动干扰**标记结果，pre、post、C32 比单纯按 ViT 调用次数排序更值得先查；正式收益仍以无分析器同尺寸整链配对为准。

`post` 本身在这个计数口径里是 11 次 Triton 发射请求，包含 4 次独立 FP8、MLP/QKV/注意力/投影/头等；即便把这一段的小核合并，也不等于清除全图 958 次请求的大部分。选择它是为了验证**连续窗口数据流**与末端数值边界，而不是以减少全图启动次数宣称质变。

后续已按**当前游戏实装**的 360p／480p／540p 三档对剩余两处 K8 `_tiled` 做同帧 VTune 任务计时，而非将上面的分派次数推成毫秒。三档 pre＋post 合计约 1.09／1.70／2.42 ms，带分析器整帧占比约 2.91%／3.40%／3.73%；无分析器完整模式墙钟分别约 27.04／38.68／53.75 ms。两套计时不能混除，post 前的非连续张量复制也未计入这两个 `_tiled` 核时长。见 [三档实测与边界](GAME_FASTLINE_K8_MATRIX_PROFILE_20260927.md)。

## Intel Vulkan 大模块路线的交叉检查

[`Uzbekunknown/dlss-nr-on-intel` 固定提交](https://github.com/Uzbekunknown/dlss-nr-on-intel/tree/f478901918340ece63cc9e5dadd234e74d04e890) 同样在 Xe2 XMX 上运行 71 块 U-Net，但其游戏接入是在 Vulkan **呈现阶段**通过 layer/socket 把画面送到 daemon；它不使用我们游戏给出的真实运动矢量，也不在 XeSS 超分前处理。其公开成绩主要来自 Arc 140V 集显、Linux/Mesa 和缩小的网络输入，不能与 B580/RE8 的 540p 模型段直接比快慢。[架构与计时口径](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/f478901918340ece63cc9e5dadd234e74d04e890/README.md)

它的大模块有清楚的适用边界：**C32 单头窗口**可让一个工作组直接持有 Q/K/V、注意力、投影和残差，避免中间值出片；但 C512 窗口的 Q/K/V 达约 192 KB，因此没有照样融合。第一次 C32 大核还比原三 pass 慢；修改归一化的 lane 分配及权重装载后，三 pass 的局部速度才达到约 1.44–1.50 倍，320×320 整图从 25.2 到 23.7 ms。C32 两层 FFN 融合把隐层留在片上；更宽的分支 FFN 融合在小尺寸反而变慢。共同约束是寄存器、共享内存与工作组驻留，而不只是提交次数。[窗口实测](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/f478901918340ece63cc9e5dadd234e74d04e890/notes/HANDOFF.md)、[融合与负结果](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/f478901918340ece63cc9e5dadd234e74d04e890/notes/improve-fusions.md)

反例也很有价值：把 Q/K/V 准备合为一核，记录的 pass 从 1128 减到 988，448×320 网络档的温态中位却从 79.05 **升至** 83.21 ms；仅把 FFN 分组调用合批，pass 从 1664 减到 1128，整图只从 289.52 到 287.83 ms，区间重叠。两组都是 Arc 140V 的记录，尚未测 B580。我们的原型应以**消除大中间张量写回、减少重复装载并保持 XMX 占用**为目的，逐段配对；合核数量不作为验收指标。[QKV 负结果](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/f478901918340ece63cc9e5dadd234e74d04e890/notes/improve-joint-qkv.md)、[FFN 合批结果](https://github.com/Uzbekunknown/dlss-nr-on-intel/blob/f478901918340ece63cc9e5dadd234e74d04e890/notes/improve-ffn-batching.md)

## 第一段隔离原型：post C32 注意力接输出投影

按 Astra 的结构审阅建议，第一段选择 540p `post.forward_head` 中的 C32 窗口注意力到输出投影。实验代码为 [`game/post_attention_fusion_v1.py`](../game/post_attention_fusion_v1.py)：现有 MLP、QKV 归一化与打包核、最终 8 通道 RGB 头仍走已验证路径；候选只把窗口注意力、attended 的 FP8 边界、投影、半精度残差和裁边写回放到同一个 Triton 工作组。它复用早期验证过的 C32 尾段算法，**并非**把整块 QKV/FFN 都融合，也没有改变默认游戏版或改写权重。

[`tools/check_post_attention_540.py`](../tools/check_post_attention_540.py) 用两组不同特征/skip 输入比对 post 的全部 8 通道，FP16 位模式差异为 0；[`tools/compare_post_attention_fullchain_540.py`](../tools/compare_post_attention_fullchain_540.py) 用同一 960×540 输入和非零运动，分别建立基线/候选图，重置帧及两帧连续历史的最终 RGB SHA-256 全部相同，三个输出都确认为图重放。首次候选图另离线编译了 1 个专用内核，记录在 `D:/Codex-NR-Experiments/nr-b580/post-attention-fusion-20260927/`；游戏和 G: 便携运行时未改。这是**正确性门**，性能结论须看下方独立配对计时。

候选的编译 IR 中有 Xe2 `#ttig.dpas`，元数据为约 2 KiB shared、`global_scratch_size=0`。这表明没有显而易见的 global scratch 溢出；并不说明 GRF 占用、实际 DRAM 流量或有效吞吐理想，必须等下一轮实测。

**配对计时已完成。** 详情见 [`POST_ATTENTION_FUSION_RESULT_20260927.md`](POST_ATTENTION_FUSION_RESULT_20260927.md)：960×540、融合历史、预捕获图重放的 B-C-C-B 四段（各 16 帧预热＋120 帧计时）中，完整 `modes.process` 中位数由 53.420 降到 52.498 ms（−0.921 ms／−1.72%）；局部 post 的独立同步计时由 7.426 降到 6.147 ms（−1.280 ms／−17.23%）。四段输出末帧哈希一致，候选命中 D: 缓存。完整链 P95 却由 54.154 升到 54.368 ms，波动增加，且输入是合成帧而非游戏。因此该原型**保留为隔离候选，暂不接入默认 RE8 游戏版**。下一轮按 Astra 路线复用尾段思路到 pre/C32 时，应先核对额外的未量化 full 池化输出，再重新做完整链与实机验收。

## Astra 审阅后的下一段路线

每一步独立成候选；沿用 540p 原尺寸输入、当前 FP16／FP8／已有 FFN INT8 的舍入位置、完整 71 块与时序，不把 AMD 的跳块／FP8 WMMA／每帧重置历史当成结构收益。实现手段先用现有 Triton；只有局部布局或片上存储难以表达时，才对同一段比较 SYCL/ESIMD。

| 顺序 | 对应的现有入口及修改边界 | 要验证的结构收益 | 明确不重做的部分 |
| --- | --- | --- | --- |
| 1. post 结果决定 C32 路线 | 若尾段整帧获益，依次适配 [`PreBlock.forward_features_outputs`](../../nr-b580-int8/backend/nr_backend/pre_block.py) 和 [`C32SwinBlock.forward_outputs`](../../nr-b580-int8/backend/nr_backend/c32_block.py)；先保留 MLP/QKV。再单独试每工作组拥有一个 64-token 窗口，使 QKV 归一化到注意力之间少写读一次。 | 计 Q/K/V、attended、full 写读，检查寄存器/SLM/溢出及 XMX/DPAS 占用。pre 的池化必须取**未量化** full 并保持半精度加法次序。若 post 尾段不获益，先拆解局部核退化与整帧淹没，只允许利用 pre 双输出有不同存储机会做一次有界复核。 | 不把现有 C32 tail 的 NR256 局部收益推广为 540p 收益；不立刻融入 FFN。 |
| 2. C64–C256 | 在 [`window_blocks_v3.py`](../../nr-b580-int8/experimental/window_blocks_v3.py) 的 QKV 生产端，把 GEMM 结果直接做每个完整 head 的归一化/FP8，写当前 head-window-token 格式，先消除 `z` 中间张量。单独比较 BM16/32、BN32/64，后续只从 C64 开始试“一个窗口、各头独立、片上汇合”。 | 确认 `z` 的真实写读是否消失、下一注意力能直接消费、权重装载是否减少；大通道若工作组过重即保持分段。 | 现有 attention→projection 已采用窗口布局，不能重复申报 unpack 收益；C256 不一次融合全部头和 FFN。 |
| 3. C512 | [`fullsize_session_v1.py`](../game/fullsize_session_v1.py) 的 `c512()` 和 [`c512_int8_ffn_rows_v1.py`](../game/c512_int8_ffn_rows_v1.py) 的分组 FFN：先对当前 768/960 行 QKV 比较 16/32 行权重片段复用；再只在分组 `_expand`→`_reduce` 试让量化 hidden 片上直达 contraction。 | 分离权重片段复用与 `QH` 全局写读消除的效果；保持行数、分组、现有 INT8 缩放/饱和及 FP8 输出。 | 当前 C512/ViT FFN 已有 XMX INT8 和部分直通，不再提出整网重新量化；大 tile 溢出即停。 |
| 4. ViT | [`fullsize_session_v1.py`](../game/fullsize_session_v1.py) 的 `vforward()`：优先对 192-key 注意力试一个工作组持有一 head 的 16/32 query，融合 score、指数、原顺序归约和 AV，输出给现有 projection；QKV 与投影先保持独立。 | 测 score/指数物化、KV 重读、XMX 活跃、工作组驻留与最终整帧收益；维持三个 64-key 块的 half 分母归约及 numerator 的 FP8 边界。 | 现有投影已试过 BM32/BN64，不能把 AMD 的宽列 tile 当未做过的新优化；不先融合整个 4096 hidden FFN。 |

对所有候选先列**逻辑字节账**，物理 DRAM/L3/SLM 流量只在有硬件计数时下结论；编译产物记录 grid、GRF、SLM、scratch/spill 和 DPAS 指令。一个调用点至多试两三个有依据的配置；正确性差异立即停，局部核或整帧 ABBA 无稳定收益就不扩散。GPU busy、少几个 launch、编译出 DPAS 均不足以单独证明性能收益。
