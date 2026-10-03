# B580 DLSS5 NR 原尺寸后端：借鉴 AMD 移植的结构优化路线

日期：2026-09-26。本文是下一轮实施与验收依据，**不是已经实现的提速声明**。对象是《生化危机 8》集成版的 B580 快速后端：游戏渲染帧 → 同尺寸 NR → XeSS 超分；以 960×540 输入为首个实验形状，后续验证 720p、1080p 等游戏渲染输入。网页选择的降分辨率档和 NR256 属于另一路画质／速度取舍，不能替代这里的原尺寸优化。冻结的 4060 字节一致后端保持独立。

## 现在的瓶颈到底在哪里

当前可复核的生产图离线基线是 2026-09-25 的 960×540、16 帧预热＋240 帧计时，实际图重放 240/240 帧，桥接加 NR 与导出调用墙钟中位数 **53.935 ms**、P95 **54.567 ms**。另一轮同条件分段测得输入桥约 **0.254 ms**、输出桥约 **0.289 ms**，模型段约 **54.429 ms**，尺寸准备 **0.305 ms**、结果合成 **0.094 ms**。两轮绝对值不能相减；它们共同支持当前应优化**模型图内部**，而不是先重写游戏桥。图重放消掉的是主机逐核调度成本，并没有消掉图中大量小 GPU 核及核间依赖。[基线与分段](PERFORMANCE_ARCHITECTURE_20260925.md) · [计时边界](COMPUTE_VS_TRANSFER_20260925.md)

同一 VTune 捕获中，启动／预热期的显式拷贝占了绝大部分；取后段时间窗，显式 Level Zero 拷贝只占 GPU 任务**累计时长**约 **0.96%**。这不表示内部数据搬运免费：布局变换、量化、填充和张量物化会被记作 Compute，且累计任务时间不是整帧墙钟。不能据此宣称“全是矩阵算力问题”，也不能用早期的 12.47% 拷贝占比宣称“显存拷贝是主因”。[稳态时间窗](STEADY_GRAPH_COPY_20260925.md)

带标记的 540p 图给出了候选位置，但采集时有桌面 GPU 活动，**性能数字无效**，只用于排列下一轮核查顺序：post、pre、C32 编解码值得先看；ViT 在该次图内排序低于旧 eager 同步画像，不能继续按 eager 的 9.62 ms 把 ViT 当首选。post 二层标记提示注意力与 RGB 头点积是大项。必须由无标记、同场配对的整帧测试确认收益。[图内标记限制](GRAPH_STAGE_PROFILING_20260925.md)

## 从 AMD 项目学什么

审读对象固定为 [`lmxxf/dlss5-on-amd-9070xt-porting` 的 `7ef24e7`](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/7ef24e7c1498bce59738277e174249866608c4ed)。它的价值在于**围绕真实硬件执行片段设计整段生产／消费链**：FP8/FP16 WMMA、静态权重预排布、上游输出下游所需字节布局、注意力寄存器留值、成组加载与提交，以及每项都回到整网 ABBA 检验。比如 C32 向量化 staging 的作者实测整网约省 0.10–0.16 ms；注意力暂存改为寄存器后，三个通道档合计约省 0.11–0.18 ms。它说明可累积的小收益有价值，同时说明**单一局部技巧不会产生数倍突破**。[C32 实验](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/7ef24e7c1498bce59738277e174249866608c4ed/Development/results/c32-vec-stage-20260923/README.md) · [注意力实验](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/7ef24e7c1498bce59738277e174249866608c4ed/Development/results/mh-register-attention-20260923/README.md)

这不是把 AMD 二进制、HIP 指令或网络配置搬到 B580。AMD 快速线用的是 RDNA4 FP8/FP16、FP32 累加；B580 要按 XMX/DPAS 可用精度和实际矩阵形状重做。AMD 的 900／1080 网络档离线约 12.11／17.06 ms 与本机 540p 基线的输入、档位、硬件、执行块、历史和计时边界都不同，不可计算“差了几倍”。其默认配置跳过块 42/43/46，部分路径每帧重置网络历史；这些是语义／画质选择，不能并入我们的原尺寸基线，更不能进入精确分支。[AMD 审读与口径](../../nr-b580/reference/amd9070xt_audit_v1/RESULT.md) · [默认配置](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/7ef24e7c1498bce59738277e174249866608c4ed/scripts/hip-game-flags.txt)

## 实施顺序：先沿现有 FP16 快速线动刀

| 顺序 | 要改的完整链段 | AMD 启发与 B580 的具体判别 | 验收要求 |
| --- | --- | --- | --- |
| 1 | **post 注意力 → RGB 头 → 历史写回** | 沿最终 RGB 与下一帧历史反推依赖。当前头点积在 640×1024 上算 8 通道，调用者只消费 RGB 3 通道及历史混合 1 通道，最终显示区域又更小。先保留注意力所需完整窗口、边缘上下文和历史，再尝试只为真正消费者生成头输出；让头和下游以相同布局直连。 | 逐字节检查 post 输出与历史；无标记整帧 ABBA。不能按“8 变 4 通道”直接推算速度，因为矩阵 tile、访存和核数会变。 |
| 2 | **pre 与 C32 编解码边界** | 对照 AMD C32 成组加载与生产者预排布，画出本机实际生产者→消费者图：哪些核把结果写成 HWC、下一个核又 pack/window 化，哪些量化结果被重复生成。一次只改一个连续边界，让生产者直接写消费者读取的 XMX 友好布局；有需要时融合归一化／舍入／打包，避免新增中间大张量。 | 先证明原版确有物理读写或多次转换；比较该层组核数、字节量与耗时，随后比完整 540p 帧及实机稳定性。 |
| 3 | **C512／ViT 中真正剩余的量化和矩阵链** | AMD 的静态权重预打包和 FP8 字节直通提示：对尚未预排布的权重，启动时一次转为 XMX 所需格式；动态激活在确定舍入点之后尽量让多个下游消费同一表示。先检查现有 C512、ViT FFN 的直通 INT8 路径和已有权重缓存，不能把已完成的工作再当新机会。`_round_fp8_half` 的全 FP16 位型幂等探针只提供“重复舍入可能可删”的数值线索，未证明图内历史／布局等价。 | 固定一个连续层组和真实 540p 形状，模块输出、完整 RGB、连续帧历史分别对拍。静态准备时间与热帧成本分开报；只在整帧稳定获益时扩散到其他块。 |
| 4 | **注意力片段在 B580 的驻留与矩阵吞吐** | AMD 在 C64/128/256 把概率留在寄存器，减少共享内存往返与 barrier；B580 应对同一个实际热点核检查 SLM、寄存器、spill、占用率、XMX 活跃率、tile 和 K 对齐。试成组加载、复用输入 tile、保存所需中间结果，避免融合后寄存器爆炸或并行度下降。 | 保持相同归约／舍入次序的先逐字节对拍。局部核更快而连续层组或整帧更慢即淘汰；性能证据需要无标记配对帧。 |

优先级 1 有最明确的**待算而未被消费的输出**，但仍只是候选，不是保证收益。优先级 2 适合压缩小核、临时张量与依赖链；优先级 3、4 决定矩阵指令是否真正被用好。若新一轮无干扰画像改变热点顺序，就按有效画像调整顺序。`post` 中单独的特征扩展曾只占标记窗口约 0.5 ms，不能因逻辑写入量大就优先重写它。[post 细分](GRAPH_STAGE_PROFILING_20260925.md) · [生产图读写](STEADY_GRAPH_COPY_20260925.md)

## INT8 放在什么位置

**不把“全换 INT8”作为前置任务。** AMD 的高吞吐主要来自原生 FP8 WMMA 和连续字节链；B580 INT8 峰值高，只有在实际矩阵上也能保持高利用率、且没有每层反复量化／恢复／换布局时，才可能转化为整帧收益。已有 B580 局部混合 INT8 在真实小形状上比 FP16 慢，纯 INT8 候选也曾在下游边界吃掉局部收益；这证明当前实现尚不能靠理论 TOPS 直接获益，不证明整数路线永远不可行。[已有筛选](../../nr-b580-int8/experimental/FP8_PARTIAL_INT8_RESULT_STATUS.md) · [AMD 固定提交的矩阵实现](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/7ef24e7c1498bce59738277e174249866608c4ed/hip/deep_fast.hip)

只有当某个**连续算子段**的 FP16 瓶颈、可共用的整数尺度、下游直接消费的整数布局都已明确，再做 W8A8 候选；计时范围必须包括量化、矩阵、重缩放、下游消费与历史状态。若纯 INT8 改变面部纹理，就作为快速线有损档单独提供完整视频给用户审核。FP16 与 INT8 整链速度近似时，选择更接近 4060 原生效果的 FP16。精确分支可复用无损布局、权重预排和依赖裁剪，但必须通过冻结 4060 字节门；不能先把有损整数路线混进去。

## 一轮候选怎样决定留不留

1. 固定同一版 RE8 运行时、游戏渲染输入 960×540、画质参数、历史模式和图重放配置；保留冻结输入、RGB 与历史哈希。先确认本次改动确实进入**游戏使用的全尺寸图**，而不是只改了旧 NR256 栈。
2. 只在隔离代码与缓存中改一个连续链段。先证明算术、布局与帧间状态符合目标分支；语义不变候选检查模块→整帧→连续帧的字节一致性，画质变化候选输出完整视频供用户肉眼审核。
3. 用无标记图重放做 A–B–B–A 配对；报告中位数、P95、离群帧、实际重放率、冷启动／预编译代价，并保持 B580 独占或记录竞争。已插标记／VTune 采集只用来找位置，不作节省毫秒的结论。小收益可累计，但合并后必须重新测全链。
4. 通过离线门后在 RE8 的同一实机场景测 NR→XeSS、快速转视角、灯光、手柄／鼠标焦点切换及帧生成；先前这些是实机稳定性敏感点。最终区分“NR 推理吞吐”和“游戏实际基础帧率／生成帧率”。

如果一轮候选只让某个核更快、完整链无稳定收益，就把它作为研究记录，不并入默认版。下一轮首先实施 **post 所需输出与连续布局** 的最小候选；若正确性或整帧计时失败，转查 **pre／C32 的真实生产消费边界**。目标是在保留现有画质的基础上持续压低原尺寸帧墙，而不是用更激进的降采样或 AMD 的跳块配置换一个看似好看的数字。

## 补充：Blackwood416 与 allanmeng 在其他模型上的可迁移经验

2026-09-26 复查两位作者的公开仓库与本项目 9 月上旬的已做实验。**新信息主要帮助选实验和避免负收益，并没有现成的 NR 加速插件。** [原 Blackwood 审读](../../nr-b580-int8/experimental/BLACKWOOD_XPU_REVIEW.md) · [原 allanmeng 审读](../../nr-b580/ALLANMENG_XPU_REVIEW.md)

1. **按实际形状选择算子，比“统一换成 ESIMD/INT8”更值得学。** Blackwood 的 A770/DG2 注意力在 `D=128` 的长序列可赢过 PyTorch SDPA，但其 `D=64` 实测只有对方 0.57–0.92 倍，故保留库回退。他的 [A770 调优指南](https://github.com/Blackwood416/A770-Kernel-Optimization/blob/main/SKILL.md)也要求记录形状、布局、精度和完整流水线时间。我们的窗口注意力、ViT 和 post 头形状各异；只能在 B580 的真实形状上分别比较当前实现、库实现和新内核，不能移植 A770 的分派阈值。[omni-xpu-kernel 的实测范围](https://github.com/Blackwood416/omni-xpu-kernel)
2. **融合的收益取决于消掉了什么边界。** Blackwood 在 MiniMax H3 把 ConvRot 旋转和行量化融合，报告该特定大张量 7.3→4.3 ms；SeedVR2 的 cat-pad、GroupNorm 则分别有其特定大形状收益。这支持我们在 pre/C32 中寻找“上一核输出立即被下一核重排／量化”的边界，**不支持把 LLM 旋转核或视频模型的尺寸直接拿来套 NR**。本机此前已试过他的 DPAS 预排思路：部分单矩阵含动态打包快了 26–44%，但旧完整链 NR256 仅约快 0.84%，另一条 1080p 残差链约慢 0.55%，未晋升运行时。[作者的融合记录](https://github.com/Blackwood416/omni-xpu-kernel/blob/main/CHANGELOG.md) · [本机已测结果](../../nr-b580-int8/experimental/BLACKWOOD_XPU_FOLLOWUP.md)
3. **allanmeng 的经验更偏向集成、依赖和正确判断“量化是否真在计算”。** 其 Qwen3TTS 旧版 INT8 路线在前向前把权重转回浮点再调用 `F.linear`，主要是权重存储／显存方案，不能作为 B580 NR 原生 INT8 点积的证据；其 B580 llama.cpp/SYCL 打包记录说明 oneDNN Flash Attention 可在匹配的语言模型路径上启用，但仍要检查我们 NR 的窗口 bias、FP8 舍入和历史语义，不能直接替换。运行库与目标架构匹配、预编译缓存和按设备能力选择路径，对便携包可靠性有帮助，不是当前约 54 ms 模型段的直接解法。[Qwen3TTS 固定源码](https://github.com/allanmeng/ComfyUI-Qwen3TTS-XPU/blob/26b7caef49e9fb0f0d82eec232e8280c02bcbad1/nodes.py#L418-L445) · [B580 SYCL 记录](https://github.com/allanmeng/llama-cpp-python-sycl-windows/blob/main/README_EN.md)

因此这两位作者带来的**新增优先级**只有一个：在 pre/C32 的下一个真实生产消费边界上，用 Blackwood 的形状分派与融合方法比较三种完整路径——现状、单独重排＋新矩阵核、上游直接产出目标布局＋新矩阵核。先计动态打包和下游消费，再以整帧决定留用。A770 注意力、LLM 权重量化和 ComfyUI 全局运行时优化暂不插队到 post 首候选之前。

## 补充：2026-09-21 IntelGPU-ComfyUI 指南包

来源是用户提供的 `E:/下载/IntelGPU-ComfyUI-系统优化指南-20260921.zip`。本轮只读其 1111 行指南、安装包目录、BMG wheel 的元数据与部分源码，**没有执行包内脚本、安装 wheel 或替换当前 Python**。它比 8 月旧指南有新内容：将原生 `omni_xpu_kernel`、`comfy-kitchen` 的量化张量／算子分发、`ComfyUI-OmniXPU` 接入层分开；BMG wheel 内有按 B580 设备 ID `0xE20B` 选择的独立策略表。我们可以借鉴这种“按物理 SKU＋算子形状分派、未命中显式回退”的组织方式，但策略表里的 INT8 解量化、注意力和其他模型算子的 tile 不是 NR 各层的已验证参数。[上游 Kitchen XPU 的能力协商说明](https://github.com/xiangyuT/comfy-kitchen-xpu)

指南称 GGUF 图像工作流 4.00→2.50 秒／步（37.5%）；这是针对其量化模型与加载器的整步结果，**不能作为 NR 速度预期**。其 `Model Sparse Attention` 配方面向 BF16、D128、长序列（例如 12,288 token 门槛），与本项目大量短窗口和固定 FP8 舍入边界不合；直接启用还会改变注意力语义。DynamicVRAM 是显存紧张时的稳定性手段，模型已有常驻路径且当前 54 ms 主要在模型图内，不能把避免 OOM 当成本轮帧耗时优化。

更有实际产品价值的是 **provider 版本锚与启动自检**：官方 `comfy-kitchen`／`comfy-aimdo`、XPU provider、Torch ABI、目标架构和真正命中的算子都应在节点启动日志中给出；`available=True` 或节点加载成功仍不足以证明 NR 算子真的在跑。当前本机 ComfyUI 与 RE8 隔离运行时均为 `torch 2.13.0+xpu`；ComfyUI 还装有 `comfy-kitchen 0.2.31`、`comfy-aimdo 0.4.13`。该压缩包 BMG 原生 wheel 固定 `torch==2.14.0`，指南 provider 锚为 Kitchen `0.2.35`、AIMDO `0.5.5`，所以**不能直接覆盖当前环境试装**。未来若为 ComfyUI 节点另建 2.14 隔离环境，可利用其静态自检与运行时能力核对；后端提速仍要按上文对 NR 实际形状做整链配对。

## 2026-09-27 更新：AMD 0.31／0.32 新内核的正反证据

复查仓库 `main` 的 [`0bf535c`](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/commit/0bf535c64c35bc41d81fb32191bb40984887ca70)。README 的发布行仍写 0.31，而 [WorkingPlan](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/Development/WorkingPlan.md) 与新 [Project-Intro](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/Development/docs/Project-Intro.md) 记录 0.32 已发布；以下结构结论按固定提交的实验记录，不以 README 的版本行推测实际安装包。

1. **完整窗口归属值得在 B580 的 C32 连续链上试，而非只换一颗注意力核。** AMD 将 C32 的 FFN、QKV、64-token 注意力、投影及 pre/post 两端组织为单 wave 的窗口链，隐层与注意力中间值尽量留在寄存器，减少跨核写读和同步；C64/C128 用一 head 一 wave，C256 仅替换注意力。两档生产 host 的无插桩千帧 ABBA 在 900 为 11.983→11.260 ms（−6.03%）、1080 为 16.993→15.873 ms（−6.59%）；单独 72 帧 NativeGameFrame RGB 哈希逐帧一致。这里的“逐位”指 AMD 快速链新旧对照，不能当成 NVIDIA 原版验收。[组合实验](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/Development/results/wave-owned-combined-20260926/README.md) · [生产回归](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/Development/results/wave-owned-production-20260926/README.md)
2. **不能一律融合。** AMD 的 C512 注意力＋投影合并同样逐位，但只有约 −0.02 ms、处于噪声内；900/1080 档仅 104/135 个窗口，一窗口一大工作组使并行度不足，双 wave 版本还出现寄存器溢出。ViT 注意力扩大 query、转置 V、去 LDS/barrier 也未稳定获益；C256 FFN 扩大 token 档反而变慢。B580 若在 C512/ViT 盲目复制“一个工作组全包”可能重演这些负结果。[C512 反例](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/Development/results/c512-wave-20260926/README.md) · [ViT/输入反例](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/Development/results/vit-attn-c32-input-20260926/README.md)
3. **C512／ViT 要按 A、B 片段复用测。** AMD C512 QKV 与 mix 从每 wave 16 token 改为 32，复用权重片段，完整 host 900/1080 分别约 −0.99%/−1.29%；64 token 导致溢出。ViT 输出投影改 16 token×64 列，使同一输入片段供四个矩阵片段复用，固定网络 ABBA 1080 约 −0.28 ms；ViT QKV 同样加宽则变慢。两项与波归属的百分比来自不同基线，不可直接相加推算 B580 收益。[C512 FFN 实验](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/Development/results/c512-ffn-20260926/README.md) · [ViT 宽投影实验](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/Development/results/m32-sweep-20260926/README.md)

**对我们下一轮的具体影响：** 保持 post 所需输出候选的原有优先级，同时增加一个隔离的 540p C32 完整窗口候选：先量出现版 C32 MLP→QKV→注意力→投影之间实际核数、写读、XMX tile 与寄存器／SLM，再让一个工作组或多个子组协作消费同一窗口，保留现有 FP8／FP16 舍入边界。不要把 AMD wave32、WMMA 和 VGPR 数直接当作 B580 的子组或资源阈值。若局部并行度下降、spill 或整帧无益即停止该形状；C512 先对真实 540p QKV/mix 测 16/32 token 的权重复用，ViT 只对输出投影试 N 宽度。原尺寸快速线先逐字节对自身基线，再用无标记同场 ABBA 看完整 NR→XeSS 帧；精确线若借用布局思路，仍单独以冻结 4060 字节输出为准。当前只完成源码与记录审读，**没有在 B580 上实测这些新候选**。

附注：仓库另有 [AMDNR 0.3.3.2 实机对照](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/Development/results/amdnr-0332-ingame-20260926/README.md)，但其 C32w 核仍在加密包中、未解密比对；该记录不是另一份可直接移植的内核源码。

## 2026-09-27 纠错：基础算子源码不等于 RE8 实机执行栈

AMD [版本年表](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/0bf535c64c35bc41d81fb32191bb40984887ca70/README.md#changelog)记载其 0.01 精确链已用硬件 wave-matrix，随后 0.02/0.03 建立快速链，0.08 扩展紧凑块间流。这是算法结构的对照，不能只按版本号判断本项目的进度。

本报告早先曾误将 `nr_backend/triton_math.py` 的基础 `_tiled`／`_dot` 视为 RE8 当前大量矩阵调用的最终分派，并据此写出“硬件矩阵覆盖甚至低于 AMD 0.01”的推断；**该推断已撤回**。实际 `game/fullsize_session_v1.py::_installed()` 会安装 `stack.components`；`nr-b580-int8/experimental/nr256_selected_stack_v1.py` 创建 `CurrentDenseTiledMatrices` 并选择 `fp16_xmx`。其父 provider `fast_matrices_v3.py` 将符合条件的 K16 dense 和 batched 调用送到含 `tl.dot` 的 FP16 原生矩阵核；`k8_tiled_provider_v1.py` 默认把两处已验收的 K8 点送到 FP16 矩阵核；`current_dense_tiled_provider_v1.py` 再为已测权重和形状选 tile。RE8 全尺寸会话另把 C512／ViT FFN 换成含 `tl.dot(..., out_dtype=tl.int32)` 的 INT8 核。先前只看基础 `triton_math.py`、没有追踪运行时作用域，是错误归因。更早的 [HANDOVER_v3](../../nr-b580/reference/handover_v3/HANDOVER_v3.md) 已记载 K16 整数模拟切除及 K8 FP16 XMX 画质验收，不能说旧审计没有识别该问题。

仍有回退到基础 `_tiled` 的算子；540p 的 VTune 审计识别到此类核，但采样和通用名称不足以算出当前整帧的准确覆盖率或收益空间。下一轮应记录 **RE8 当前 provider 的实际分派计数、具体矩阵形状、编译出的 DPAS/XMX 指令与有效时间**，同时区别 FFN INT8、FP16 native、残余基础模拟和非矩阵工作。只有这张执行清单出来后，才能把本项目与 AMD 某个推理阶段作可靠类比并重新排优化优先级。

2026-09-27 追加：当前游戏三档的后续源码＋设备审计已确认残余 `_tiled` 只在 pre 16×32、post 32×8 两处。360／480／540p 合计 GPU 任务时长约 1.09／1.70／2.42 ms，在各自带 VTune 的完整帧中约占 2.91%／3.40%／3.73%；这不是无分析器实机收益。旧文此段“下一轮应记录”已由 [实装三档测量](GAME_FASTLINE_K8_MATRIX_PROFILE_20260927.md) 部分完成；后续仍需区分 post 连续化复制及评估其他矩阵核的有效吞吐。

## 2026-09-27 差距归因：重点是模型图的有效吞吐

RE8 960×540 原尺寸生产配置的离线图重放整链中位数为 53.93475 ms；另轮分段中模型段为 54.429 ms、输入／输出桥各约 0.25／0.29 ms，不能跨轮相减求精确占比，但足以确定主要时间在模型图内。AMD 公布的 900 网络档约 12 ms 是不同设备、网络档、块选择、历史与计时合同，不能直接定义 B580 的“效率落后 4.5 倍”；它仍是需要解释的强烈性能信号。AMD 当前快速链使用原生 FP8／FP16 WMMA、跨块 E4M3 字节流及完整窗口算子融合，常规配置还跳过 42、43、46 块并启用自适应 ViT 复用。B580 当前已有 FP16 XMX 和部分 INT8 XMX，不能把差距简单归咎于 Triton 未编译或全网整数模拟。

**尺寸修正：只报 54／12≈4.5 倍会低估待解释的问题。** B580 540p 的内部计算画布为 1024×640＝655,360 像素；AMD 900 档为 1600×960＝1,536,000（2.34 倍），1080 档为 1920×1152＝2,211,840（3.375 倍）。拿旧记录约 12.1／17.1 ms 作仅供直观比较的“每画布像素耗时”，AMD 分别约为 B580 的 1/10.5、1/10.7；换言之，这个**粗指标**显示约 10 倍差距，不是已测出的同工作量推理效率比。ViT token 数、各层 padding 与算力密度并不按画布像素线性缩放，且 AMD 的 FP8 精度、跳块、历史和计时合同均不同；不能把 10 倍归因于 Triton 或某一种核。正确结论是：考虑更大的 AMD 网络档后，执行结构和真实硬件利用率更值得优先调查，而不是把 4.5 倍当成差距上限。

初轮 540p 图内标记显示 pre、两侧 C32 与 post 都值得查，其记录的每帧 GPU 任务合计约 1,358；这只是该次 VTune 的任务数，不等于可直接对比 AMD HIP 的 kernel launch 数，更不能由任务数推导收益。该次采样存在桌面 GPU 活动及分析器扰动，已标记 `valid_for_performance=false`，其阶段毫秒值仅用于选诊断位置。显式 Level Zero 拷贝在另一稳态窗口的累计 GPU 任务时长约 0.96%，但通用 `_kernel` 超过三分之一，既可能是有效数值计算也可能是布局／量化；现有画像尚不能在“矩阵利用率低”和“非矩阵数值核／依赖链太碎”之间分配责任。

下一轮先取得当前实装路径的逐类清单：每个模块的矩阵形状、实际分派 provider、DPAS/XMX 指令、寄存器溢出、占用及下游可直接消费的张量格式。再以无分析器、同图状态的配对整帧作为性能门，优先在 pre→C32 或 post 内选一个连续段做保持当前 FP16/FP8 边界的原型，测“整个连续段＋下游”，而不是单个微核；出现收益后扩到相同形状。若硬件矩阵有效吞吐已经接近该形状上限，则转向融合非矩阵数值核与连续层的数据驻留；若实际大段回退到基础 `_tiled`，先修对应分派或 tile。此处是可证伪的归因路线，尚无新的 B580 候选实测。

## 2026-09-27 首个结构候选的结果

上段“尚无新的候选实测”是撰写时状态；现在已完成 540p post C32 注意力→投影融合的隔离实现与配对测试。两组局部输入的全部 8 通道、三帧含运动历史的整链输出均逐字节一致；B-C-C-B 完整图重放中位数 53.420→52.498 ms，局部 post 同步调用 7.426→6.147 ms，但整链 P95 54.154→54.368 ms、波动上升。候选暂不进入默认游戏版。源码、完整口径与下一段 Astra 技术路线统一见 [`AMD_INFERENCE_STRUCTURE_MAP_20260927.md`](AMD_INFERENCE_STRUCTURE_MAP_20260927.md) 和 [`POST_ATTENTION_FUSION_RESULT_20260927.md`](POST_ATTENTION_FUSION_RESULT_20260927.md)。
