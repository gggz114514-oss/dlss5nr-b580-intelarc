# 精确版无损优化迁移

2026-09-14，用户在六项原生控制及新进程缓存验证后授权：“接下来把快速版通用的优化搬到精确版”。此前暂停优化指令被本次授权更新；原生控制产品接入仍需完成，不扩展社区功能。

## 实现与当前验收状态

### 分支大尺寸通过；QKV融合256²本批降低1.24%

主任务复核Luna报告与原始JSON。branched-dimensions-v1：14worker编译861规格，512²、480×864、1080×1920、1439×2559各45/45差分/生命周期检查通过；不等于标准1440p/2160p或所有控制的大尺寸验收。

exact-qkv-v1四阶段通过：14worker编译197规格；3组pack primitive（含全部half编码）字节一致、pack无spill；13参数26帧显示/历史/seed一致，冷进程191/191group与4/4helper命中。相对同批branched-v1，95.99755→94.803725ms，降低1.243599%（1.193825ms），6个正式配对均更快。不能跨批把95.2815与94.8037相减解释本次收益。

当前QKV叠加分支候选约10.55fps，仅256² Session计时范围。尚需QKV自身多尺寸及连续视频验收；不能继承branched大尺寸结果充当QKV覆盖。产品默认尚未替换。

### 分支候选多尺寸与QKV融合已派发

Luna先执行 `exact_branched_dimensions_manifest_v1.json`：14worker编译、branched候选4个既有尺寸契约的45项差分状态检查。新validator独立于原all-v1测试，不沿用旧候选验收替代新结果。

新增 `product/qkv_exact_v1/`：叠加于branched-v1，仅C128/C256注意力，矩阵投影不变，将原C32 half平方/FMA/XOR归约、rsqrt、Q缩放、FP8转换直接写入head/window顺序，省去中间切片contiguous与window重排物化。所有原half舍入和NaN处理保留。尚未GPU验证，不宣称提速。

前置多尺寸通过后，Luna执行 `exact_qkv_manifest_v1.json`：独立14worker预编译（不修改当前正运行collector）、三组pack原语含全部half编码、新内核资源、原生26帧、缓存及相对branched-v1公平配对。首错停止，主任务修复。结果根exact-branched-dimensions-v1与exact-qkv-v1；新候选源码已过AST检查，GPU结果待回报。

### C128/C256分支融合通过：256²本批耗时降低3.08%

主任务复核exact-details-v4及exact-branched-v1完成报告与原始结果。细分76段序列/字节通过：代表C256块MLP约0.63ms、attention约0.51–0.54ms（含QKV约0.27ms）；为嵌套独立诊断，不加和。

branched-v1四阶段通过，14实际worker预编译212规格；C128/C256两组primitive及13参数26帧显示/历史/seed一致，新增分支内核资源记录spill=0；新进程190/190 group、4/4 helper命中，无blocked。公平配对all-v1中位98.308025ms→95.281525ms，降低3.078589%（3.0265ms）；6个正式配对候选均更快。诊断统计仅热身，计时排除加载和读回。

这是256²默认控制reset+temporal Session的单批证据，约10.50fps，不外推标准1440p/2160p。该候选尚需自身多尺寸/连续视频验证，不能用旧all-v1多尺寸结果替代；尚未替换产品默认。后续优先验证该收益候选的大尺寸与状态，再针对同样较重的QKV/注意力继续独立优化。

### C128/C256细分与分支融合候选

新增details-v4，沿用已通过独立图诊断，细分encoder128/256首块及decoder128/256首块的MLP、attention、QKV和输出投影，并细分encoder256首块8分支expand/reduce/project。Triton序列与字节检查保留；嵌套段不可加。Luna单次执行，首错停止。

独立 `product/branched_exact_v1/` 已实现：共享输入的多分支expand批处理+原精确cubic，reduce批处理+原FP8写出，以及单kernel按分支顺序、每K16舍入累加projection。仅替换所属C128/C256模块forward_unquantized，保留输入q、外部输出q；没有XMX近似，也没有并行重排分支累加。尚未GPU验收。

细分诊断通过后，Luna执行 `exact_branched_manifest_v1.json`：14worker编译→C128/C256 primitive与三新kernel资源检查→原生26帧→新进程cache-only→相对all-v1公平配对。结果根exact-details-v4、exact-branched-v1。当前只有源码AST检查，不宣称提速或合入默认。

### 分段图诊断完成：优先细分C256/C128

主任务已读取 `exact-stages-v3/luna-report.md` 与原始profile.json：进程正常退出，36项（含whole）各5样本、字节全通过；分段/整段Triton序列相同（3727调用），历史与池外持久I/O检查通过。全body独立图100.33712ms，主要单段为decoder256 10.34830ms、encoder256 10.20320ms、encoder128 6.10884ms、decoder128 5.97354ms；pre及encoder/decoder32各约5.1ms。这些独立阶段时间不相加或当完整图占比。

据此下一优先级应细分C256/C128段中的branched MLP、QKV/窗口注意力及输出投影，在实际输入/原精确K16数学下比较融合/调度候选。不能仅凭每个C512/ViT块排名较低认定整个C512/ViT族总成本低，因为划分粒度不同。尚未完成新内核优化，100.33712ms是诊断body区间，不是对98.5844ms Session基线的退化测量。

### 精确分段图诊断已派发

新增 `reference/profile_exact_stages_v3.py` 与固定manifest。复用此前快速分支的独立阶段图诊断思路：当前all-v1真实temporal图输入，完整body与分段body先核对Triton调用序列和输出字节，再分别捕获整段、encoder组、各C512/ViT块、decoder组及post。5轮轮换顺序，每项5次重放后等待GPU完成，计时外核对每段输出、输入、历史/seed、池外持久I/O和源码不变。阶段时间非可加占比，不冒称完整图内kernel事件。

不再用已崩溃的Kineto/XPU profiler；旧图内Event探针亦为不可用证据，不重复测试。脚本禁止Triton缓存写入，缺少规格即停止，按14worker另行补编，不默默串行编译。Luna只执行监控，第一失败由主任务诊断。结果根 `D:/Codex-NR-Experiments/nr-b580/exact-stages-v3`；目前仅AST检查完成，不能声称已取得热点。

### reuse_v2完成：数值通过，无明确性能收益

主任务复核Luna完成报告、batch及各原始benchmark/build/cache：13/13阶段通过，14worker编译241/241规格，四组均7个primitive及原生13参数26帧显示/历史/seed通过，新进程缓存通过。范围仍为256²；此轮未建立新候选大尺寸覆盖。

| 候选 | 同批all-v1基线→候选 ms/帧 | 耗时降低 |
| --- | --- | --- |
| dataflow量化消冗/融合写出 | 102.873275→103.026375 | -0.1488% |
| compact紧凑query直读 | 105.1968→105.6296 | -0.4114% |
| 两者组合 | 98.787825→98.4595 | 0.3324% |
| 组合+BM8投影 | 102.753275→111.988775 | -8.9880% |

组合差0.328325ms，6轮配对有正有负，不能宣称稳定收益；BM8在本批明显退化。保留独立实验，不默认启用，继续以all-v1为已验证性能基线。不能跨批拿98.46与105ms相减宣称加速。下一步应解决图内剖析，定位精确矩阵实际成本；本轮结果不足以证明寄存器溢出或某一种算术是唯一瓶颈。Luna的资源记录来自现有hook，不能仅凭其spill=0就宣称覆盖所有新增kernel。

### reuse_v2数值与性能批次已交Luna

`reference/exact_reuse_manifest_v2.json`共13阶段：14worker预编译（包含primitive规格），随后dataflow、compact、combined、tile四组各原生26帧/新进程缓存/配对计时。比较基线为已验证all-v1约98.6ms组合；不是旧250ms基线。长任务由Luna监控，首错停止，主任务修复。源文件已过AST检查，结果根 `D:/Codex-NR-Experiments/nr-b580/exact-reuse-v2`；未收到GPU结论。

尺寸说明纠正：2559×1439是已有非整齐边界契约，不是标准2560×1440；此前测试不能宣称覆盖标准1440p或3840×2160。此次继续的是已准备的256²优化候选数值/收益验证，不静默将常用尺寸加入未经认证的PADDED_SIZES。

### 多尺寸复测通过；profiler失败单独处理

主任务已复核 `exact-dimensions-v2` 完成报告与四份原始JSON：512×512、480×864、1080×1920、1439×2559各45/45检查通过，包含三次调用的显示/low/history/seed及reset、非法motion、输出所有权、close。此为默认控制的冻结精确Session差分，不是新的4060逐尺寸原生采集，也没有大尺寸性能/峰值显存数据。

预编译1226/1226完成，配置workers=14，完成记录包含14个不同worker PID。旧测试InferenceMode错误已修复并复测通过。

随后 `exact-profile-v2` 失败：lease退出3221226505（0xC0000409），无Python traceback；虽导出profile/trace并局部写passed=true，但外层批次失败，不能算成功。导出仅14组GPU事件，缺少已知主体矩阵，无法覆盖约100ms图重放；不能据这些外围事件给出本体占比或瓶颈排序。需要替代/修复图内剖析，不能绕过异常退出把报告当作验收。reuse_v2新候选仍未完成GPU测试，98.5844ms仍是当前有效256组合计时。

### 下一轮与多尺寸错误修复

用户要求继续完成数据流消冗、紧凑query直读、剖析后精确矩阵优化及测试，并要求编译并行14。新增独立 `product/reuse_v2/`：C512已知FP8生产者消冗、原精确dot写出融合q、紧凑query连续缓冲及索引投影，以及BM8投影候选；目前仅语法检查，未宣称通过或提速，v1原候选未修改。

首次多尺寸批次在512²的baseline第一帧因测试脚本在InferenceMode外执行 `value.color.zero_()` 抛错；baseline状态/非法motion检查已通过，尚未进入候选比较。修复仅把原输出的原地修改置于 `torch.inference_mode()`，不通过clone绕过别名检查。旧失败证据保留于 `exact-dimensions-v1`。

新增 `reference/precompile_reuse_v2.py`：独立FakeTensor规格采集（不是数值测试），复用既有precompile包的ProcessPool编译，明确14个worker，每worker单BLAS线程。不能把OMP=14当作14路Triton编译。Luna获得新 `exact_dimensions_manifest_v2.json` 单次执行授权：先并行预编译baseline/v1四尺寸，再数值/生命周期验收；成功后跑 `profile_exact_graph_v2.py` 的稳定图GPU事件剖析。第一处失败停止由主任务诊断。当前新批结果未收到。

### 组合批次已完成：256²精确链明显提速

主任务已读取 `exact-batch-v1/luna-report.md`、batch及各组原始benchmark/validation/cache JSON：15/15阶段通过，五组build/cache各26帧显示、历史、seed一致，新进程缓存通过。下文“待验收”为本批派发时记录，由本节更新。

| 配对候选 | 原版→候选 ms/帧 | 耗时降低 |
| --- | --- | --- |
| 执行链融合及裁减 | 260.7392→248.92665 | 4.53% |
| 图重放 | 261.27375→110.679475 | 57.64% |
| C512/ViT布局直连 | 250.416675→246.206475 | 1.68% |
| 有效query裁减及布局直连 | 251.4687→201.9177 | 19.70% |
| 全部组合 | 251.670325→98.5844 | 60.83% |

范围：256×256 SDR、默认控制的reset+temporal Session配对计时，2轮热身+6轮正式，包含Session同步，不含加载/读回/视频外围。全组合约2.55倍吞吐、10.14帧/秒；不能外推1080p或实时整条链。13组控制的26帧字节一致另行验证通过。全组合新进程195/195 kernel group与4/4 helper命中，但仍需进程内图捕获，且未排除驱动内部JIT。主要收益来自图重放及其调度调整，不能归因于INT8算术变化。多尺寸/生命周期尚未完成，产品默认仍未替换。

### 2026-09-14 组合迁移批次（本节优先于下方旧派发记录）

用户要求所有可复用的无损优化实现并测试。已建立独立候选，不覆盖产品默认；没有引入近似 INT8 数学。

| 项目 | 当前证据 |
| --- | --- |
| K8 分块、ShortFP8、固定 cubic | 原生26帧和新进程缓存通过；配对收益分别约0.447%、-0.150%、0.325%，不足以宣称稳定加速 |
| Decoder gather | 五组几何 primitive、原生26帧、冷缓存通过；251.0862→249.381375 ms，约0.679%，尚不能外推 |
| 融合 front、历史采样、body 调度、post 依赖区域裁减 | 四项分别原生26帧 display/private/seed 通过；组合尚待完成 |
| XPU 图重放 | 已实现常量校验、外部持久输入输出、输出独立所有权；捕获中 Swin 改用同1024窗口计算但不做设备 fence，最终同步保留；待GPU验收 |
| C512／ViT 布局直连 | 已实现 packed/head-major 直接投影，保留K16舍入和split-K合并；待 primitive、原生和计时验收 |
| C512 无用 query 裁减 | 已实现只计算被下游读取的query，完整64个K/V上下文保留；待验收 |
| 全部组合 | 已实现显式开关与作用域恢复；待独立原生验证、冷进程缓存及配对计时 |

上批次停止原因是启动命令把 python.exe 作为脚本，不是数值失败。已用 `reference/run_exact_manifest_v1.py` 从固定 JSON 生成参数数组，CPU dry-run 检查15阶段；每阶段独立GPU lease，首错停止，无自动重试。任务清单 `reference/exact_batch_manifest_v1.json` 已交 Luna max 单次执行，结果根 `D:/Codex-NR-Experiments/nr-b580/exact-batch-v1`。主任务负责修复，Luna仅执行和监控。尚未收到本批完成报告，不宣称组合通过或有加速。

新增 `reference/validate_exact_dimensions_v1.py` 准备在组合通过后逐尺寸检验：相同输入对照冻结精确 Session，reset→temporal→显式reset，显示/low/private原始字节、seed、拒绝非法motion后状态、调用者输出所有权、关闭后拒绝使用。每进程一个尺寸和控制组合，避免同时驻留两份模型；待覆盖512²、480×864、1080×1920、1439×2559。该扩展属于精确后端差分，不能冒称新4060采集。当前未运行，不宣称全尺寸可用。非方形历史仍走原始实现。

验收后按完整链路实测选择组合；各项百分比不能相加，无收益候选不默认启用。仍需完成多尺寸/生命周期验证后才能宣布迁移落地到产品。

### Decoder gather：独立候选已派发

`product/exact_decoder_v1/{kernels,scope,primitive}.py`及`nr_exact_decoder_candidate_v1.py`已从快速版gather提取；移除快速数据流CONTRACTS依赖，保留原精确dot/split-K/q/compensated half FMA。五处过渡直接读取小投影对应像素与skip合并；C32单独存未量化结果及正零padding，后续body不变。只对256源尺寸启用，其他尺寸回退。没有叠加先前候选；来源哈希在provenance.json。

验证计划已交Luna：5组实际通道宽度的奇数尺寸/非连续输入及C32padding边界对照，接13组26帧原生显示/私有历史/seed验证（五处各26次命中）；再独立新进程cache-only，最后2轮热身+6轮交替配对公平计时。正式计时不做资源记录/调用统计，warmup检查spill，字节比较在计时外。任一步失败即停并报主任务；Luna不修复/重跑。结果根`D:/Codex-NR-Experiments/nr-b580/exact-decoder-controls-v1`，报告luna-report.md。当前只有源码与CPU语法检查完成，GPU结果待回报。

### 固定cubic候选：已实现并派发

**最新结果：正确性与缓存通过，单次计时收益很小。** 主任务已核对Luna报告与primitive/cache/frames/benchmark原始JSON：65,536编码零差异，build/cache各26帧显示/历史/seed通过；15个候选kernel无spill；新进程175/175 group、4/4 helper命中。公平计时250.323575→249.509625ms，差0.81395ms（0.3252%），正式计时诊断统计未变。尚不足以认定稳定整帧收益，不替换默认，不与K8百分比相加。下文“尚未收到结果”为旧派发记录。下一项按既定顺序提取Decoder gather，减少中间张量物化，仍保留原精确投影和合并次序。

ShortFP8公平计时benchmark-02已由主任务复核：252.17630→252.555375ms，候选慢0.1503%，正式计时诊断统计不变；未发现收益，不合入默认，也不继续择优重测。

接续独立`product/nr_exact_cubic_candidate_v1.py`及`product/exact_cubic_v1`，仅迁移快速版`native_half_cubic_v1.py`固定unary cubic两次half FMA，保留clamp、half乘法和原FP8，不叠加K8/ShortFP8，不改通用FMA。来源hash保存在provenance.json。256以外回退。复用诊断仅热身的公平计时方式。

已交Luna三阶段：新入口全65536编码对比+26帧六项控制 → 新进程cache-only → 公平配对计时，任一失败停止上报，不自行修复。根目录`D:/Codex-NR-Experiments/nr-b580/exact-cubic-controls-v1`，完成报告luna-report.md。此记录尚未收到cubic结果，不宣称通过或提速。

### ShortFP8：已实现独立候选并交Luna验证

**计时修正已派发（结果待回报）**：Session新增`diagnostics`，默认true保留原验收。`benchmark_exact_shortfp8_candidate_v2.py`只在2轮热身启用资源hook与调用统计；6轮正式配对直接调用相同候选函数，检查统计记录不变、全局hook恢复，并在计时区间外继续核对显示和历史字节。内核与量化数学未改，故不重复此前全域/26帧测试。已过CPU语法检查，Luna单次执行benchmark-02，输出根同上，报告`luna-benchmark-02-report.md`。不覆盖第一次含检查耗时，也不择优重测。

**最新结果：数值和缓存通过，当前带检查候选未体现收益。** 主任务已读Luna完成报告并核对primitive/resources/cache/benchmark JSON：65,536种half模式零差异；build/cache各26帧显示、历史、seed通过；54个新kernel无spill；新进程175/175 group和4/4 helper命中。当前配对为253.09955→259.68785 ms（候选慢2.603%，6.5883ms），不合入产品默认。

计时解释有一项明确限制：候选process仍安装每kernel的资源检查hook、构造记录并累计调用，baseline没有相同开销。故本结果是带验证检查的候选Session耗时，不能证明ShortFP8算术本身更慢；下一步若评估该候选收益，应先把已验证资源检查移出计时区间或做对称控制，保留数值核对，不能直接反复跑同一脚本挑最好成绩。报告 `D:/Codex-NR-Experiments/nr-b580/exact-shortfp8-controls-v1/luna-report.md`；下文尚未收到结论的派发记录已被本段取代。

候选 `../nr-b580-int8/product/nr_exact_shortfp8_candidate_v1.py` 和 `product/exact_shortfp8_v1/`：提取原精确fp8/cubic/attention-weight模块，仅替换FP8 half编码helper；独立模块避免污染baseline的JIT全局依赖，Python入口在序列化process作用域替换并finally恢复。暂仅256源尺寸启用，不叠加K8。原始来源哈希见候选包provenance.json。新增kernel记录寄存器/shared/spill，非零或未知spill拒绝。

新编译入口先比较所有65,536种half编码，之后复用13组26帧原生显示/历史/seed验收，并确认三个优化入口均有调用；再独立新进程cache-only，最后同输入配对计时。脚本 `validate_exact_shortfp8_candidate_v1.py`、`benchmark_exact_shortfp8_candidate_v1.py` 位于reference。已交原Luna max顺序执行，前步失败即停并由主任务修复，结果根 `D:/Codex-NR-Experiments/nr-b580/exact-shortfp8-controls-v1`，报告luna-report.md。此处尚未收到GPU结论，不宣称通过或提速。

### K8结果：正确性与缓存通过，收益很小

Luna三阶段报告已完成，主任务已复核cache-audit、原生验证与benchmark原始JSON。build/cache两轮各26帧显示/私有历史/seed全部通过，pre/post各26次命中；新进程175/175 kernel group及4/4 helper命中，无blocked。配对计时为249.8422→248.72525 ms/帧，中位数差1.11695 ms（0.447%）。这是256默认控制下的单次短测，收益很小，不能视为稳定的普遍加速或推广到1080p；保留独立候选，不替换产品默认。

报告 `D:/Codex-NR-Experiments/nr-b580/exact-k8-controls-v1/luna-report.md`。下文派发时“尚未收到GPU结果”已由本段取代。下一项仍是ShortFP8等价编码迁移，采用独立差分验收，不将近似INT8计算引入精确链。

首批只迁移pre/post K8分块，未改冻结基础数学或已安装产品。候选 `../nr-b580-int8/product/nr_exact_k8_candidate_v1.py` 继承精确控制Session；仅自身pre16→32、post32→8权重、无initial的K8调用改为原 `_tiled_dot`，排布分别BM2/BN32/1warp、BM4/BN8/1warp。来自快速版 `experimental/k8_tiled_provider_v1.py`，没有导入其快速K16/XMX provider。其余调用回退，当前只对256×256源尺寸启用。

这项优化复用权重与每K8组的精确共享指数/整数累加/舍入；改变的是数据分块方式。全局入口替换只在隔离Session锁内的process作用域存在，finally恢复；固定控制和私有历史仍走已验证入口。

已准备并通过CPU语法检查：

- `reference/validate_exact_k8_candidate_v1.py`：复用13组26帧原生控制验证，额外要求pre/post各命中26次。支持独立新进程 `--cache-only`，复用禁止编译守卫。
- `reference/benchmark_exact_k8_candidate_v1.py`：默认控制、相同GPU输入、两份独立Session、reset+相邻时序帧；2轮热身及6轮交替先后顺序，核对显示与历史字节。计时为Session调用，不含加载和读回，不称纯kernel耗时。

已交Luna max三阶段顺序执行，任一失败即停并上报主任务：build-01原生一致 → cache-01新进程缓存一致 → benchmark-01交替计时。每阶段lease900秒、wait0，不自主修复/重试，不修改安装环境。结果根 `D:/Codex-NR-Experiments/nr-b580/exact-k8-controls-v1`，完成报告 `luna-report.md`。此记录尚未收到GPU结果，不能宣称已验收或已有速度提升。

## 后续顺序及限制

K8通过且有收益后，补声明尺寸的几何/边界对照再启用大尺寸。依次提取ShortFP8等价编码、固定cubic等价计算、Decoder gather，逐项留差分记录，再组合；不一次叠加未验收候选。ShortFP8是同一FP8映射的较短实现，不是把精确模型重新量化为INT8。

跨算子布局与输出依赖窗口裁剪保留为后续独立项；它们不能直接复制快速版的普通tl.dot、固定NR256或残差缩放假设。每步保留六项控制、精确字节/历史/seed、输出所有权和失败语义。无收益或数值变化的候选不替换默认。产品cache-only清单及可分发包另行验收；研究缓存通过不代表安装包已更新。
