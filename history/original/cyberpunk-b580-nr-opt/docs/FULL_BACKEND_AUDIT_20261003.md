# 现役720p NR全面审计：结论与改造清单

更新：2026-10-03。五组源码审计已完成，主任务完成执行范围、来源、共享归属和遗漏核验。67条原始问题及改造记录归并为18组工作包；记录包含热路径、条件路径、稳定性及冷资产，不能解释为67处已测瓶颈，不能相加估算提速。

本轮唯一对象为《赛博朋克》现役固定1280×720的优化集成后端：C512＋K8、当前已启用的数值优化、融合采样＋真实运动＋计算图重放，以及已验收的纹理复用＋GPU接力桥。先优化720p，再兼容输入分辨率缩放。精确4060实现只在当前依赖或语义来源需要时引用。

**完成的是审计和执行计划。本轮未修改生产数学代码、替换游戏文件或运行新GPU测试；新增改造的收益及画质尚未验证。** 用户最近验收的完整基础帧时间仍为55–60ms。30/60基础帧的完整帧预算分别为33.33/16.67ms，不能用NR网页吞吐或帧生成显示帧率代替。

## 覆盖与核验

五名GPT‑6.1 Sol max负责独立范围，主任务负责有效配置、动态选择条件、跨组遗漏、来源核验及去重。

| 组 | 覆盖节点 | 原始记录 | 范围 |
| --- | ---: | ---: | --- |
| Swin/MLP | 237 | 11 | C32/C64/C128/C256实际site、MLP/QKV/注意力/投影/窗口 |
| C512/ViT | 740 | 10 | 16个局部C512、全部全局ViT层、连续INT8 FFN及矩阵库接口 |
| front/Decoder/post | 258 | 14 | 控制/noise、pool/down/up、Decoder、post及输出缓冲 |
| history/host/graph | 655 | 13 | 历史与运动、检查、图、缓存、切档、恢复、寿命及CPU外圈 |
| native桥 | 644 | 19 | 实装ASI/helper/HLSL、adapter/prehost、线程/队列/fence/回送 |
| 合计 | 2534 | 67 | 固定现役720p源码、选择条件与接口 |

每个节点给出12维结论：调用、数值语义、矩阵、非矩阵、生产/消费、全模型工作量、内存、CPU、计算图、桥时序、路由异常、改造与验证。节点数包含接口、分支及编译变体，不能当作每帧kernel启动次数。

- 835份物理来源、8187个Python函数/类声明已建立清单；包含同源副本、旧参考与工具链文件，不代表全部在现役逐帧执行。
- 保存的137个实际编译键全部关联到具体节点、源码及有效条件；这是已保存编译变体集合，不是完整硬件任务数。
- 223份复制的数学来源中197份有分组引用；剩余26份由主任务核对选择或冷接口，明确未选中实现和冷路径，不把旧候选算成现役优化。
- 来源SHA、引用行号、报告字段、v1记录映射及共享owner核验通过。结构检查包含27456条引用记录，错误为0；格式通过不单独作为效率或动态触发证明。
- 生命周期检查缓存、批量校验、桥缓存、纹理池及GPU接力当前确实开启。旧安装记录的开关状态不覆盖后续已验收配置。

## 数值遗留与实装边界

ViT归一化/指数仍有复刻half融合乘加的补偿计算；QKV、投影及部分Decoder输入仍有分段half结果和有序合并。四级非C32 Decoder合并输出的有效参数仍为`ROUND_OUTPUT=true`，前一轮去舍入开关尚未覆盖全部现役输出边界。C64/C256分支可借鉴已实装C128的FP32连续累加方式。

这些操作可作为快速版原生数值候选，分别记录误差和完整模块收益。近似指数、有效key、五tap采样、控制、运动及历史具有算法功能，数值替代须保留这些功能，不能仅因来自精确版就删除。

当前Decoder C512窗口投影已经按16个K32 head连续FP32累加；native K8前后矩阵也已是完整K的FP32累加。C128原生分支累加、C32输入共享和部分attention融合已实装，不能再计为新收益。现役矩阵包含原生FP16/INT8路径，其tile、布局、非矩阵和下游效率仍需测量。

## 18组改造任务

全部原始记录保留在台账，每条都有归属。以下各组均为“本轮已审计，尚未实施和测量新增收益”；部分已有关闭的实验代码可复用，须适配当前生产者、消费者及图。

| 组 | 实施范围 | 验证重点 |
| --- | --- | --- |
| P01 ViT原生数值 | norm/exp补偿FMA、分母布局、exp(0) | 保留归一化和近似指数算法，分开记录误差 |
| P02 Decoder合并/post入口 | 四级输出去E4M3舍入、native FMA | 舍入与FMA分别对照，连消费者测完整尾段 |
| P03 完整矩阵段 | DecoderInput、ViT QKV/投影full-K | 一次FP32累加，协调真实stride及消费者布局 |
| P04 C64/C256分支 | 扩展现役C128原生累加方式 | 不重复统计已启用C128，测整块及整帧 |
| P05 历史原生数值 | 倒数、维度坐标、near-integer路径 | 分数采样与条件near路径分开，保留历史与运动 |
| P06 post完整尾段 | sigmoid、颜色/alpha、存储、条件风格 | 默认style0与style1/2区分，保留风格效果 |
| P07 MLP/QKV调度 | 输入复用、输出tile、成对分支 | C32已共享X，优化串行部分及真实几何 |
| P08 完整局部注意力 | Q/K/V至投影、head调度、有效query | 保留padded K/V上下文；store mask不证明少算 |
| P09 首个Decoder C32 | attention/project及下一消费者 | Swin与Decoder同一发现只实现一次 |
| P10 post连续消费 | entry/native MLP、投影/native head | 减少HWC32全图写读，保持有效输出及内部画布 |
| P11 Encoder C512 | 窗口投影至pool/down | 复用已实装Decoder packed投影，适配Encoder消费者 |
| P12 连续INT8 FFN | 现役C512/ViT的scale、requant、partial/merge、scratch、tile | 分离矩阵/非矩阵/布局/下游；旧C128探针不能代替 |
| P13 pool/down/up | 全部九个矩阵几何及相邻pool | loader与平均数值分开，保留内部padding语义 |
| P14 历史/front/publication | 无消费者normalized store、坐标共享、控制与缓冲 | 初步保留private/public寿命，桥和数学不重复算收益 |
| P15 CPU外圈 | 剩余JIT查询、重复认证、receipt、JSON、V2观测 | 缓存/批量检查已开，测CPU自身及实际关键路径 |
| P16 桥调度 | 命令资源复用、成功路径原色copy、锁范围、退休 | 保留失败回退与真实消费者同步，terminal候选未算实装 |
| P17 稳定性与恢复 | 尝试限额、锁顺序、generation、Signal/retirement、关闭锚点 | 源码问题与未观测触发区分，保留正确安全状态 |
| P18 冷资产/驻留 | noise、风格资产、固定720无用索引及释放 | 逻辑请求量不等于每帧搬运或实际显存峰值 |

按用户主线先处理P01–P06的精确版数值遗留，再推进P07–P14的完整计算段及布局。桥/CPU可独立开发，GPU测试由Luna串行完成。影响持续运行或恢复的P17修复与性能实验分开验收；P18单独记录显存和冷启动。

## 桥与长期运行

纹理复用和GPU接力已实装，但桥仍有命令资源创建、打包/复制/解包边界、最终原批次CPU等待、锁范围和诊断处理。逻辑字节统计用于定位边界；缺少硬件计数器时不能直接换算毫秒或带宽。

源码确认live路径有100000次尝试限额，普通metrics reset不清该计数；达到限额后持续拒绝NR，直到对应状态重新初始化或计数后续变化。关闭、失败及不匹配输入也影响计数，不能按当前FPS推导多久必现。本轮未在实机触发，已单列持续运行修复项。

锁顺序、generation不匹配仍提交、Signal失败处理、异常retirement和强引用寿命等条件发现也已保留。19条桥记录不代表19种已复现游戏故障；当前用户验收的快速转镜头和Alt+Tab仍正常。

## 测量边界与验收

应用源码和接口的覆盖已闭合，以下仍需测量或条件用例，未填成“无问题”：真实ISA/spill/cache及物理内存流量；16个C512 QKV `torch.mm`的库内tile、任务及复制；同帧Present关键路径和CPU/GPU重叠；near/风格/强度/异常/恢复命中频率；实际显存峰值；新候选净收益及画质。

之后分别测矩阵本体、非矩阵、格式/布局、消费者及CPU/等待，再比较完整模块、整图和同场景《赛博朋克》完整帧。有可重复净收益即可采纳，没有1ms门；数值及画质损失单列。先前3–8ms/8–15ms只属工程估计，本轮没有实测总收益或60帧承诺。

## 可复查记录

- [审计合同](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-full-coverage-audit-v2-20261003/AUDIT_CONTRACT.md)、[固定配置/来源](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-full-coverage-audit-v2-20261003/RUNTIME_SCOPE_720.json)、[源码入口索引](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-full-coverage-audit-v2-20261003/INVENTORY_INDEX.json)。
- 分组报告：[Swin/MLP](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-full-coverage-audit-v2-20261003/swin-mlp/REPORT.md)、[C512/ViT](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-full-coverage-audit-v2-20261003/c512-vit/REPORT.md)、[front/Decoder/post](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-full-coverage-audit-v2-20261003/front-decoder-post/REPORT.md)、[history/host/graph](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-full-coverage-audit-v2-20261003/history-host-graph/REPORT.md)、[native桥](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-full-coverage-audit-v2-20261003/native-bridge/REPORT.md)。
- [合并索引](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-full-coverage-audit-v2-20261003/INTEGRATED_INDEX.json)、[67条记录/18组台账](D:/Codex-NR-Experiments/cyberpunk-opt/b580-full-coverage-audit-v2-20261003/INTEGRATED_FINDINGS.json)、[主任务遗漏核对](D:/Codex-NR-Experiments/cyberpunk-opt/b580-full-coverage-audit-v2-20261003/COVERAGE_CHALLENGE.json)。
- [最终来源与结构检查](D:/Codex-NR-Experiments/cyberpunk-opt/b580-full-coverage-audit-v2-20261003/MAIN_FULL_COVERAGE_REVIEW_FINAL.json)、[主任务范围验收](D:/Codex-NR-Experiments/cyberpunk-opt/b580-full-coverage-audit-v2-20261003/MAIN_AUDIT_ACCEPTANCE.json)。验收限定于应用源码、选择条件、接口及保存编译产物；外部二进制内部与硬件未知保留。

v1四组38条记录作为历史来源保留。本轮五组补审及去重结果以这里的合并索引为准。
