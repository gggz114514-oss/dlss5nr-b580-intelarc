# 精确后端复用快速版优化：语义审计

日期：2026-09-13。范围：审计当前源码和已有实验，确定可迁移的执行优化；本次不修改后端、不启动 GPU 编译/推理，不改变产品默认。

**2026-09-14最新授权覆盖下文暂停记录：用户已要求开始无损迁移。** 当前已落地K8独立候选并交Luna验证，见 [EXACT_OPTIMIZATION_MIGRATION_STATUS.md](EXACT_OPTIMIZATION_MIGRATION_STATUS.md)。本审计的历史证据与限制不变，不能将候选派发写成验收通过。

**后续优先级变更：用户要求先确认社区可调功能，性能迁移暂停。** 本文中的K8及其他迁移顺序仅供将来恢复优化时参考，不是当前派单。当前安排见 [NR_CONTROLS_FUNCTIONAL_PLAN.md](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/NR_CONTROLS_FUNCTIONAL_PLAN.md)。后续功能盘点还确认已有 `graph_front_v7.py` 的精确控制研究验收，可复用其保留派生控制入口的设计；当前产品v6/INT8并未因此自动获得相同覆盖。

## 1. 结论与基线

**可以复用不少优化，优先级最高的是 pre/post 的精确 K8 分块、ShortFP8、特定 cubic 激活、Decoder gather。连续 INT8 段与普通 FP16 XMX 矩阵不能直接移入精确分支。** 跨算子布局、有效输出裁剪和图执行也能借鉴，但须拆掉快速算术和固定尺寸假设。

本次逐文件 SHA256 比较，两个研究树的 `backend/nr_backend/*.py` **35/35 完全相同**。主要差异在快速树的 `experimental` 执行适配器与产品 Stack，而不是另一个独立重写的基础后端。

| 项目 | 本次审计身份 |
| --- | --- |
| 精确源码 | `nr-b580`，分支 `nr/exact`，HEAD `7355848c4b8952fb5e383fe29d3e2fb3cdd00b1d` |
| 快速源码 | `nr-b580-int8`，分支 `nr/int8`，HEAD `c876d4c43597a454cc9fddba7310fec53fd9d540` |
| 当前快速入口 | `experimental/nr256_product_stack_v1.py`，继承 compact query、连续 INT8、layout/crop、ShortFP8 等层，并添加 DecoderGather |
| 当前精确产品 | `nr-b580-int8/product/nr_exact_runtime_v1.py`，独立 worker 加载精确 `MotionNR`，显式 `triton`，没有安装上述快速 Stack |
| 原生目标 | 已冻结 RTX 4060 SF-v2 行为；运行库 SHA256 `6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927` |

精确迁移必须保留模型/权重、K8/K16 分组与每步舍入、归一化顺序、FP8 边界、控制参数、历史/seed/reset、输出所有权及失败时的状态语义。误差小或肉眼相同不能代替精确分支的字节验收。

证据分三层：**原语等价**、**优化前后快速链字节一致**、**精确链对冻结 4060 参考一致**。第二层不会自动成为第三层。本审计尚未产生任何迁移后的精确整帧验收。

## 2. 优先迁移清单

| 顺序 | 项目 | 可复用的部分 | 适配条件与证据边界 |
| --- | --- | --- | --- |
| 1 | pre/post 精确 K8 分块 | 复用同一 `_tiled_dot(chunk_k=8)`；pre BM2/BN32/1 warp，post BM4/BN8/1 warp | 只迁移这两个选择规则，不继承快速 provider 的 K16/XMX 分派。保持原权重、初始累加值和不匹配情况回退。已有真实权重、边界及快速整链记录；精确整链未验收。 |
| 2 | ShortFP8 | 保留 SATFINITE E4M3 编解码，简化 subnormal 舍入指令 | 全部 65,536 种 half 编码旧/新结果相同。移植 helper 及受影响调用，不导入整个 `ShortFP8Graph`；新编译语境仍需局部核验。 |
| 3 | 特定 cubic 原生 half FMA | 只替换 cubic 内两次 FMA，保留 clamp、两次 half 舍入、最终乘法和 FP8 | 已有全部 half 编码、6 组/18 输出记录。不能全局替换通用 half FMA，也不能复制夹带 `tl.dot` 的快速 MLP。认证 LUT 可作为另一个等价实现，收益须单独判断。 |
| 4 | Decoder gather | 五次解码过渡直接按 `y//2,x//2` 读取小图，与 skip 合并后写出，省去物化放大图 | 现成 scope 仍调用原 `decoder.dot/split_k_projection`，很适合单独提取。保留 compensated half FMA；C32 写未量化 residual 和原正零 padding。内核没有锁死 NR256，完整模型证据目前仍是快速 NR256。 |
| 5 | 冗余 FP8 消除与输出写入融合 | 已在同一 FP8 域的值不重复量化；仅在原有舍入位置把 q 合入 producer store | `q(q(x))=q(x)` 全编码证据可复用。须证明具体张量仍在该数值域及借用后无写入，重建精确 producer 契约；不能只看 dtype/data_ptr。 |

### K8 为什么是直接机会

当前精确 `triton_math.fused_dot` 对 K16 已走 `_tiled_dot`，对 K8 仍走扁平 `_dot`。快速版 `k8_tiled_provider_v1.py` 针对 pre 16→32 和 post 32→8 选择已有精确分块内核，保留共享指数、逐乘积截断、24 位对齐精度和每 K8 组 half 舍入。它并没有把这两处改成近似 XMX。

旧快速实验中，局部 pre 1.636670→0.162490 ms、post 0.743120→0.250030 ms，完整 NR256 调用 20.170368→18.181237 ms。这证明该执行排布值得复用，**不证明当前精确版也有同样的百分比收益**。新尺寸的 tile 先核对资源和边界，不照抄快速 K16 的大 tile 表。

### ShortFP8、cubic 和 Decoder 的实施边界

- ShortFP8 保留符号零、饱和和 NaN 规范化规则；输入先转 half 的位置也不变。它只是实现同一个 FP8 函数，不是新量化。
- cubic 只适用于现有固定系数的 unary 函数。原通用 FMA 实现处理半精度中点误差；普通 FP32 乘加后转 half 不能替代它。原生 half FMA 在通用三元组实验中出现过 NaN payload 差异，特定 cubic 的全域记录不能推广到所有 FMA。
- DecoderGather 需要从实验作用域中拆出，避免强制依赖整套 `quantization_dataflow_v1.CONTRACTS`。仅对契约匹配的 XPU half 路径使用，CPU、其他合法输入及公共边界 API 保持原行为。
- Decoder 旧配对实验：本体 8.032348→7.970392 ms（0.77%），五处局部合并 0.064745→0.035038 ms（45.88%）。局部数不能当作整网收益，二者也不能相加。

## 3. 值得移植思路、需要改造实现的部分

### C512 与 ViT 跨算子布局

可复用窗口/head 的地址映射、直接消费 producer 布局、少做 unpack/transpose/contiguous 的思路。可以先让精确投影读取已有布局，矩阵仍使用原精确整数累加规则，减少同时变化的部分。

**现成投影内核不能原封不动使用**：

- `c512_window_projection_v1._project` 用普通 `tl.dot` 按 K32 累积 FP32，末尾加入 half residual；精确版要求按 K16 共享指数逐乘积截断并逐组 half 舍入。
- `vit_head_major_projection_v1._parts` 虽保留四路 split-K 和有序 half 合并，内部也换成 K32 `tl.dot`。分片数相同不代表精确累加相同。
- `compact_c512_qkv_stack_v1.py`、`c512_quad_queries_v2.py` 的 QKV/attention 也包含快速矩阵规则。有效 query 的几何可复用，分数与 value 累加仍须原精确数学。
- 当前 owned scopes 锁定 C512 `(12,12,512)` 和 ViT `(64,1024)`。原生 480p/1080p 需要参数化形状；ViT 跨多个 64-key 块的求和次序、填充扣除不能简化。

两条不可破坏的依赖：

1. 最后一个 encoder C512 的池化消费 **未量化 `full`**：先 top 两项 half 相加、bottom 两项 half 相加，再 half 合并、乘 0.25、转 half，最后才 q。不能直接让它消费量化后的 output，也不能任意改变归约顺序。
2. 可以省略确实无人消费的 query 输出，必须保留这些有效 query 依赖的完整 K/V、对应位置 bias、窗口像素顺序及原填充。不能把 padding token 当作“无贡献”删掉。

### post 先裁依赖范围，再计算

当前 `post_region_scope_v1.py` 对 256 RGB 保留 `[0,264)` 的完整 padded attention 窗口，读取真实 merged `[0,260)` 与 coarse `[0,130)`，最后仅对 256 RGB 区域做 projection/head。这个因果裁剪可用于精确版；现成 wrapper 却要求 `fp16_xmx` 和固定形状，不能直接推广。

在当前 shift=4、window=8 的 post 上，RGB 查询区间为 `[4,H+4) × [4,W+4)`，所需 padded 终点为 `ceil((H+4)/8)*8`、`ceil((W+4)/8)*8`。读取真实输入与原有零填充的边界分别处理；到达原图末端时不能把真实上下文换成新零，也不能越界读取不存在的 coarse 值。

| RGB 尺寸（H×W） | 原 post attention 画布 | 依赖窗口画布 | 少计算的窗口像素 |
| --- | --- | --- | --- |
| 256×256 | 328×328 | 264×264 | 35.22% |
| 480×864 | 520×904 | 488×872 | 9.48% |
| 512×512 | 520×584 | 520×520 | 10.96% |
| 1080×1920 | 1160×1928 | 1088×1928 | 6.21% |

表中只是根据当前 `PADDED_SIZES` 推导的几何比例，不是 GPU 性能或已通过的大尺寸结果。1080p 收益空间比 NR256 小。head 要保留 post 的所有实际依赖，尤其 channel 3 的 temporal blend logit；不能只留 RGB 三个输出。本次不提出改变公开 `forward_head` 的八通道契约。

### 历史采样与动态 front 融合

`fused_square_history_v2.py` 与 `fused_graph_history_warp_v2.py` 不依赖 INT8 模型，适合独立候选。已有 19 组原语完整输出/错误路径记录。保留 FP32 FMA 次序、纹理 1/256 系数、14 位定点对齐、INT64 求和、ties-away half 舍入、负零及认证倒数表。输出仍分别返回 numerator/reciprocal；不能提前把历史归一化后交 post，后者的融合减法会出现舍入差异。

**现成核仅支持 256/512 方形路径。** 原生 480p/1080p 走 `warp_history_normalized`，有 dimension reciprocal、归一化坐标 FMA 和纹理坐标定点规则，不能把现成方形核的 H/W 限制删掉就称为支持。它是 NR 历史重采样，不是 Block/DIS 运动估计；本次不改变运动估计链。

`fused_dynamic_front_v1.py` 可融合噪声查表、反射和 HWC16 构建，seed 必须每帧有效，不能缓存上帧激活。它写的是默认 front 控制通道；用于 `ControlledMotionNR` 时，后续参数与 mask 写入必须照常发生。

### 静态图、同步与常量复用

图执行的所有权经验可复用：reset/temporal 分图、持久输入/输出放共享临时图池之外、返回值独立、历史不受调用方改写、关闭释放引用。当前精确 `executor.mark()` 在没有 progress 时仍逐阶段同步，C32/attention 还有分段同步；可以按依赖关系调整调度，而不改数学。

不能直接安装快速 `GraphFront`：其 provider 签名绑定快速实现，且会在实例上覆盖 `_forward_front`。若直接套在 `ControlledMotionNR` 上，非 fallback 路径将绕过其 control/mask 通道写入及 `return_float32`/`_raw_private` 逻辑，破坏强度混合和私有历史语义。应把图边界放在控制注入后的公共数学主体内，保留外层验证、style/UI 后处理、失败回滚及 progress 行为。

减少同步不是删去验收或让 CPU 停止纠错；CPU 本来就没有替 GPU 改正像素。保留返回/提交前必需的完成与错误检查。大尺寸图还要评估显存峰值和图池寿命，不能用 NR256 可捕获推断原生1080p同样经济。

固定权重的无损布局打包、认证表与不可变模型缓冲可构造一次复用。跨帧缓存动态量化激活不成立；同帧共享必须是相同数值、相同 scale/轴/布局、且无写入冲突。

## 4. 不直接迁移、避免重复计算的部分

| 部分 | 判断 |
| --- | --- |
| ViT 连续 INT8 FFN、C512 floor16 与范围修复 | 重新量化改变中间值。快速版可接受，不满足精确标准；预先缓存 scales/weights 不能恢复丢掉的精度。 |
| 普通 FP16 XMX dense、融合 MLP/QKV/attention | 只保留表面 half/FP8 边界不够。精确 K16/QMMA 还依赖共享指数、逐乘积截断和每组舍入；必须重写内部算法才能复用调度/布局。 |
| 通用 half FMA 全局替换、求和重排 | 特定 unary 全域证据不能覆盖多输入运算与 NaN payload；默认保留现规则。 |
| “保留 FP8，用 INT8 执行部分计算”旧混合原型 | 只验证有限输入，旧局部投影还慢约96%/127%，没有理由当成现成精确加速。须另立数学/性能假设才重启。 |
| NR256 缩放、增强残差回贴、限定画面 ROI | 改变模型输入或增强范围，属于另一输出模式，不能用于宣称原生精确模型提速。 |
| 已有 exact K16 tiling、FP8 单核、cubic+FP8 融合、C32 normalize/指数、attention 权重/row64 融合、compensated half FMA | 已在两个树相同的基础后端里，不再作为“从 INT 版新移植”的收益。ShortFP8/native cubic 是这些旧融合内部的进一步简化。 |

当前精确旧剖析仍指向 dense/batched 矩阵是重要成本：1080p 插桩记录 dense 1918 次、包含式事件区间约1442.9ms，batched1258次约377.9ms，FP8约204.6ms。类别含主机调度间隙且有嵌套，**不能相加算占比或当正常帧延迟**。这解释了为何这些无损执行优化值得做，却不能承诺靠点算子/搬运优化就追平4060。

## 5. 最小迁移与验收顺序

本节是下一轮实现建议，不表示已启动 zcode/Luna 或已经合入。

1. **第一张实现任务只做精确 K8 分块。** 在精确实验入口选择已存在 `_tiled_dot` 的 pre/post 两处；保留其余分派。新实验源与缓存独立保存，不改冻结基线。
2. 顺序提取 ShortFP8、unary cubic、DecoderGather；各自先与同次原精确算子比较受影响真实边界，再合并。复用已有原语证据，针对新编译入口、形状和组合补缺口，不反复重跑无关矩阵/视频。
3. 再做精确 FP8 数据流消冗、C512/ViT 布局直读；有效 query 与 post 裁剪作为独立候选。两条用户关心的工作线都保留：跨算子布局，以及输出依赖范围，不能只做后者。
4. 每项起步采用现有 256 重置帧+相邻时序帧，并检查受影响边界、完整 RGB、私有历史和 seed。涉及 front/图边界时加非默认控制或显式 mask 的最小案例；没有覆盖的参数/尺寸继续旧路径。目标原生大尺寸时，在推广前补一组480p或1080p的匹配参考及边缘条件。
5. 候选通过局部对照后，才做同输入、同精度、同尺寸、同计时边界、充分预热的交替配对整帧计时。短测或单一场景不能声称所有输入数学证明；持续历史/产品视频在准备推广且相关行为变化时验收。
6. 两份独立精确模型维护各自连续历史；输出比较用原始字节/可信完整哈希，不靠 allclose。已具备匹配原生归档时复用，缺的语义再采4060。误差一出现就定位，不能自动放宽为画面相似。

长编译/测试按用户约定交 Luna max 监控，失败及时回报主任务诊断，不自行改源码或重试；本次没有启动任务。新 kernel 要有对应来源/目标/缓存记录，保留 CacheOnly 运行约束，不改 expected hash 来放过缺失缓存。只编译受影响规格，产品集成前再补其声明尺寸。

## 6. 本次实际核验与证据入口

本次运行仅使用 Python/NumPy 在 CPU 重读小型归档和哈希：

- 两树35个后端源文件逐一相同；28个候选源文件另记SHA256。
- ShortFP8 65,536编码、FP8幂等性65,536编码均检查输入覆盖与完整保存输出字节。
- unary cubic 6组/18份保存输出重读一致，包含原实现、native half和认证表路径；5个唯一压缩数组合计318,017存储字节。
- 读取10份相关原报告及其范围/哈希，未重新执行旧完整模型、未遍历重验旧报告所有递归 source gate。
- 新GPU推理0帧，新基准0项。CPU重读确认保存证据，不是重新执行GPU数学或独立CPU模型复现。

机器记录：[audit.json](D:/Codex-NR-Experiments/nr-b580/exact-fast-reuse-audit-20260913/audit.json)，包含源文件与报告SHA256、重读数组、原语核验和post几何推导。

| 证据 | 路径（相对 `D:/Codex-NR-Experiments/nr-b580/reference/`） |
| --- | --- |
| K8 边界 / 完整快速链 | `experimental/k8-tiled-boundaries-v1/validation.json` / `results/k8-tiled-native-parity-v1/validation.json` |
| ShortFP8 全域 | `experimental/short-fp8-body-v2/validation.json` |
| cubic 全域 | `experimental/native-half-cubic-v1/validation.json` |
| FP8 幂等及数据流 | `experimental/fp8-graph-rewrite-v1/validation.json` |
| Decoder gather | `results/decoder-gather-v1/validation.json` |
| 历史融合 | `experimental/fused-history-complete-v2/validation.json` |
| C512/ViT/post 快速整链 | `results/layout-crop-native-parity-v1/validation.json` |
| C512 compact query 快速整链 | `results/c512-quad-full-v1/validation.json` |
| 精确后端旧剖析 | `profiles/weights-pointwise-1080-v1/results.json` |

### 关键源码入口

- 精确计算：[triton_math.py](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/backend/nr_backend/triton_math.py)、[controlled_temporal.py](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/backend/nr_backend/controlled_temporal.py)、[post.py](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/backend/nr_backend/post.py)。
- 首批候选：[k8_tiled_provider_v1.py](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-int8/experimental/k8_tiled_provider_v1.py)、[short_fp8_v2.py](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-int8/experimental/short_fp8_v2.py)、[native_half_cubic_v1.py](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-int8/experimental/native_half_cubic_v1.py)、[decoder_gather_scope_v1.py](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-int8/experimental/decoder_gather_scope_v1.py)。
- 架构候选：[quantization_dataflow_v1.py](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-int8/experimental/quantization_dataflow_v1.py)、[c512_window_layout_scope_v1.py](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-int8/experimental/c512_window_layout_scope_v1.py)、[vit_head_layout_scope_v1.py](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-int8/experimental/vit_head_layout_scope_v1.py)、[post_region_scope_v1.py](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-int8/experimental/post_region_scope_v1.py)。
- 产品边界与背景：[PARAMETER_SCOPE_CHECK.md](E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/PARAMETER_SCOPE_CHECK.md)、[ZCODE_NR_B580_HANDOFF.md](E:/ComfyUI-aki-v3-IntelArc_20260722/ZCODE_NR_B580_HANDOFF.md)。
