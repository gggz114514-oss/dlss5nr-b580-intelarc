# DLSS5 NR／Intel Arc B580 开发历史与证据（截至2026-10-03）

**截至截点，唯一现役性能主线为 Cyberpunk 1280×720 C512＋K8／all6，去 FP8 激活模拟舍入、融合真实运动／历史、图重放、纹理复用与 GPU 接力。用户实机验收为 55–60 ms，按该轮约定关闭帧生成。** 这项成绩是用户同场景 RTSS 读数，不是本 worker 新测的 Present trace，也不能把此前所有改善归因于某一个内核。

本档从已找到的立项与首轮诊断材料开始，按阶段整理主要实现、失败与更正，覆盖精确复刻、快速量化、全尺寸执行、产品集成、游戏桥和当前候选。最早具有明确日期的已读诊断为2026-09-07；工作区名称中的20260722不作为本项目立项日期。更早的具体立项时刻没有独立证据，不补写推测日期。

配套 `EXPERIMENT_INDEX.json` 包含41条主要路线与141份补充报告记录，每条有脚本定位、报告／可取得原结果 SHA、参数摘录、输入 hash 来源、计时口径、画面门和采用边界。它不是每个瞬时参数 arm 的穷举目录。`HISTORY_SOURCE_SELECTION.json` 是本地最小必留文件清单，公开伴随版使用别名和拟发布相对路径。没有执行历史实验、GPU、Torch/Triton 导入、安装、清理或发布。

证据路径统一使用 `@project`（原项目根）、`@experiments-D`（D盘任务实验根）、`@experiments-E`（E盘任务实验根）。公开索引不包含个人目录、局域网地址或 SSH 命令。这些路径是原证据定位符；拟发布副本的最终路径、许可和内容脱敏由主代理核验。

## 证据和计时的解释规则

“成功”拆成数学／字节、局部速度、完整调用速度、冷启动／缓存／退出、游戏功能及用户画面六类。通过一种门不自动通过其他门。精确分支在文档限定的输入、尺寸、控制与历史范围内保持冻结4060字节合同；fast 可在单独声明的质量门内改变输出。

| 计时名称 | 实际范围 | 不可替代的结论 |
| --- | --- | --- |
| CPU穷举、IR／compile | 标量／tile数学、源码lowering或CPU预编译 | 不能证明GPU性能、物理ISA成本或游戏画面 |
| local kernel／core | 可能已经预量化／prepack，单模块device时间 | 不含row quant、launch、padding、layout或整帧时不能称全网收益 |
| raw graph | 已捕获body replay及规定等待 | 不等于FullsizeGameModes.process，也不含桥和Present |
| body／NR256 residual pipeline | 当时网络body，或低分辨率NR再残差放大 | 不是现役720完整游戏成绩 |
| 完整离线process | 当轮FullsizeGameModes.process及规定sync，常排除decode/upload/build/capture | 不含游戏render、完整桥和Present；process−graph不自动等于纯CPU成本 |
| bridge CPU record／record→retire／web | 桥录制、录制到真实消费退休跨度、网页处理各自范围 | inclusive父子不能相加；同帧配对才能比较；不等于FPS |
| 人类RTSS 55–60ms | 已验收场景当前完整游戏链基础帧时间 | 没有同帧API/Present追踪，不外推所有场景或独立机器 |

## 2026-09-07～09-08：恢复真实网络，先建立冻结参考（H01–H02）

立项目标是完整迁移NR，先对齐原单帧、运动历史和控制，再研究INT8；不是以“能输出一张图”代替恢复正确语义。固定参考为 RTX4060 **Laptop 8GB**，原运行库 `310.8.SF-v2`，DLL SHA256 `6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927`。`WEIGHTS_HT` 长147,695,410字节、153 records／71 blocks，SHA256 `836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4`。它们是冻结身份，不是本档授权分发的资产。参考机器原驱动不足，后续版本满足原库要求；版本差异必须随每份回执保留。

初始恢复的Torch图映射489权重、145,754,963参数，却在B580和4060 CUDA上分别出现约1.64796和1.74049的gradient MAE。跨设备同类失败排除了“只是B580算错”的简单解释；16处 `[1,8,8,512]` 中间结果非有限，后续饱和掩盖了问题。仅检查最终finite会误判成功。

原生运行最初在非交互Session0返回 `FeatureNotSupported / 0xBAD00001`，相同程序与输入在已登录Session1成功。这是会话条件证据，不能写成4060不支持。后续捕获固定RGB／depth／motion、原输出hash与实际调用trace：9个模块、156调用，pre/post arena共15,711,232字节。5种直接layout解释都失败，说明arena的物理排布不能当逻辑Tensor。65536种half bit-pattern的FP8标量一致性验证解决NaN→0与±448饱和等边界，但没有单独证明全图。

恢复出的结构包括C32完整MLP而非残缺投影、未量化full输出参与pool、C32×4／C64×4／C128×6／C256×8／C512×8及8个ViT，C256 10→12 padding、C512分组FFWD和6→8 pool。9月8日形成独立 RGB→RGB `ResetNR256`，不读取参考中间层作为计算答案；256六输入／四seed逐字节一致，用户人物／车辆／Apex画面接受。

后续对齐扩展至512、864×480、1920×1080与Apex2559×1439；Apex最终44,188,812个RGB字节相同。时序从256零运动四帧扩展到非零归一化运动、重置／历史、depth与控制。原1080早期调用约181–183秒属于恢复阶段，不与后续优化秒／毫秒结果拼接。素材原始帧数、尺寸、帧率及音频作为合同保留：480素材243帧／24fps／10.125s；1080素材390帧／60000÷1001fps／约6.5065s。

证据：`@project/nr-b580/README.md`、`RESET_BACKEND_STATUS.md`、`DIMENSION_BACKEND_STATUS.md`、`TEMPORAL_BACKEND_STATUS.md`、`NORMALIZED_MOTION_STATUS.md`、`DEPTH_REFERENCE_STATUS.md`、`NR_CONTROLS_FUNCTIONAL_STATUS.md`、`VIDEO_FIXTURE_CORRECTION.md`。初始失败图、layout否决和Session0失败保留；冻结原输入／输出／trace均保护。

## 2026-09-08～09-09：精确语义和执行结构优化（H03–H04）

精确内核要复现原K8/K16的逐步half截断、FMA／量化边界，不能直接把整个MLP换成一次大矩阵乘就保持字节。优化依次涉及二维tile、cubic＋FP8融合、4×32单warp、attention exponent／normalize／weights融合，以及运动／控制相关融合。35源文件、736完整RGB／372history／26graph比较是报告明确的样本范围，不表示所有未来输入都已证明。

在1080配对中，attention weights完整模型约2.297755→2.189442秒、normalize约2.4620717→2.3089371秒；exp与tile路线也有受测收益。否决路线同样重要：简化K16舍入破坏字节；C32增batch约慢0.9%；为exact使用XMX需要eligibility与纠正，整数products展开出现约4倍成本／spill；继续展开integer loop既可能变慢也可能不符边界。

这些经验可迁移为“保留完整算术合同后再优化layout／融合”。不能迁移为“XMX硬件没有收益”或“当前fast必须保留所有模拟FP8舍入”。fast在9月27日得到不同质量合同后，保留的算术边界另判。

证据：`SPEED_OPTIMIZATION_STATUS.md`、`HALF_FMA_STATUS.md`、`HALF_FMA_FUSION_STATUS.md`、`FULL_INT32_ARITHMETIC_STATUS.md`、`INTEGER_SUBGROUP_SPEED_STATUS.md`、`XMX_ABLATION_AND_INTEGER_PRODUCTS_STATUS.md`、`ATTENTION_WEIGHTS_SPEED_STATUS.md`、`C32_NORMALIZATION_SPEED_STATUS.md`、`INT8_XMX_DIAGNOSIS.md`（均位于`@project/nr-b580`）。通过byte门的结构保留；错误／低效展开保留为反例。

## 2026-09-09～09-11：fast分支、图重放与NR256快速栈（H05–H18）

这阶段建立exact与fast两支：exact冻结4060字节合同；fast采用FP16 XMX及INT8 dense、FP16 attention等混合精度，并以误差、视频与性能独立验收。“INT8”标签从来不应自动解释为所有输入、所有层、整个网络均纯INT8。

| 主要尝试 | 意图与实施 | 原结果与边界 | 保留／否决与画面门 |
| --- | --- | --- | --- |
| 初始FP16／INT8 | 将dense投影交给XMX | 480 resident 13帧，exact607.501／FP16 404.271／INT8 450.526ms；非充分warmup稳态；PSNR55.970／46.117 | 短片用户接受，随后长片再验 |
| weight views／Kparts | 固定1262 dense调用的prepack、缓存和参数 | INT8 fused392.851→371.269；scheduled371.124→328.239ms | 243帧保持当时fast输出 |
| graph front v3 | 捕获固定body，noise/history动态、owner/reset/commit明确 | FP16 314.373→137.230；INT8 330.580→142.082ms | 采用；失败重试／退出／历史并非可省条件 |
| shared graph v6 | reset与temporal共享池，降低1080内存压力 | pool6,421,479,424→3,210,739,712B；后复测FP16 780.890→653.596、INT8 793.988→640.203ms | Windows shared memory2.37GB→97.5MB为抽样；旧3.88×叙述被修正 |
| window/layout融合 | strided attention、减少拆装与padding | 480FP16 133.610→112.819／INT8 137.207→117.142ms | 有效实现采用；INT8v1约1.7s压力异常否决 |
| low-NR residual | 256／512低NR signed residual放大到1080 | 完整低256约44.5ms／低512約78.9ms | 历史路线；不是60fps或当前720默认 |
| DIS预算／GPUBlock | 运动成本与私有GPU接口复用 | 1080 CPU DIS2线程131.23、8线程58.10ms | 后续产品仅GPUBlock，DIS取消；私有实现不发布 |
| MLP/Swin/front/history融合 | 减少launch、中间张量与row reshape | residual44.455→34.725／temporal45.796→35.357ms | 组合门通过；不能累加不同轮次局部成绩 |
| split heads／QKV／ViT | 8组head batching、window normalize+quant融合 | 35.503→30.087、30.073→29.152、29.250→28.142ms | 当时栈采用、243byte門限定 |
| LUT／branch batch | cubic索引、MLP查表、分支批处理 | 大尺寸与NR256的收益不同；wide LUT低NR可能慢 | 只保留净收益组合 |
| nativehalf／Triton3.8 | native FMA并隔离编译器ABBA | nativehalf22.695921→22.143602；Triton full22.322634→20.252519ms | 没改用户Comfy Python；精度与版本均须固定 |
| K8 tile／fused history | 二维tile、真实warp/noise融合 | 20.170368→18.181237、18.102812→14.372799ms | 有效栈采用，历史语义继续保持 |
| FP8等价rewrite | 删除重复quant、短rounder、layout crop | quant986→755；12.850255→12.345622；Stackv4 11.745086→11.269206ms | 当时保留FP8语义；short v1生命周期错、v2/v3修复；spill/epilogue無净利不取 |
| ESIMD DPAS | oneAPI2026.1 AOT与Triton互操作 | v1half parity失败；v2fullNR快2.843%，residual慢0.546% | 不部署；局部primitive快不代表pipeline快 |

以上ms来自不同版本、尺寸与轮次，按原对照成对阅读。不能把表的首尾除法叫“整个项目提速倍数”。4060后续测得256 host3.566807ms／GPU NR2.4416ms、1080 host14.1217／GPU13.196ms，设备／host口径和机器不同，不能直接除B580另一路结果。

INT8数学与质量审查经历几层失败：254 finite E4M3值、96真实operands的CPU可表示性，不等于实际tile覆盖。C512 whole block有效覆盖仅0.27047%，ViT约3.35016%；two-DPAS split增加工作。partial projection先compile失败，再因NumPy bool序列化退出，修复后的GPU局部约0.010820→0.024571ms（+127%）和另一档+96%，虽字节相同／零spill仍否决。

连续INT8 FFN完整body约8.995105→8.391660ms，NR256 temporal11.304696→10.841529ms，但脸部颜色出现异常。hidden clipping约1.53–3.74%，CPU ROI消融显示去clip可大幅降低该ROI误差，history影响更大；因素有交互、ROI含背景，百分比不能相加。只有随后真正GPU量化range修复及完整视频用户接受才构成采用证据，48个相关calibration样本仍不能代表所有游戏内容。

C512 compact query把有效144行以BM32覆盖160，保留全部64KV；body8.366100→8.263140、pipeline11.196961→11.089721ms，全243输出保持。floor16的body虽8.266830→8.076365，完整pipeline却11.446748→11.613342ms；用户接受其画面，不可写成净提速。constant/signature由819增至1011、guard1.442714→1.923720ms是host验证成本，不是CPU修改像素。

C512 directQKV约+12.46%慢，QKV新INT8虽局部快但质量代价被用户拒绝。native query完整body小退化，不能写作性能成功。decoder五gather融合643→639 launches、local0.064745→0.035038ms、body8.032348→7.970392ms，则通过限定字节门后进入产品。

证据组：`@project/nr-b580-int8/experimental/*_STATUS.md`及同目录原`audit_*.py`、`screen_*.py`／实现文件；`INT8_FACE_COLOR_CAUSE_STATUS.md`、`INT8_RANGE_REPAIR_STATUS.md`、`INT8_FACE_REVIEW_STATUS.md`、`QUANTIZED_PROJECTION_STATUS.md`。精细文件映射见H05–H18索引。当前只读确认该experimental目录顶层原`.py/.md`共684项、5,276,042字节；它不是生成缓存全集。

## 2026-09-11～09-24：产品、视频与Comfy环境（H19）

历史 `nr256stackv1` 产品session冻结当时接受的floor16／query／gather等路径，exact另保留。用户把原四节点／DIS方案收敛为两个节点：NR，以及NR＋XeSS SR＋FG，分别支持exact／fast；运动选择遵照用户GPUBlock，不重新默认DIS。

视频验收包含帧数、分辨率、音频和FG的实际语义。原243帧视频，SR保持243帧，FG应为485帧／48fps；一次技术输出484帧被最初误报，后续明确修正。仅CPU remux重建1/48 timebase与index0..484可修封装，不重跑GPU，不能隐瞒这次失败。AA1× SDK QUALITY106也实际执行SR，1×／2×／FG路线分别验收。

实际Comfy服务环境出现 `ONEAPI_DEVICE_SELECTOR` 和系统icpx PATH污染；子进程隔离成功并不自动说明实际启动成功，后续实际失败和运行时预载、路径修复单独记录。跨盘只读DiskOnly、中文安装路径、独立Python／oneAPI／缓存包以及分片安装测试是可移植性验证的一部分。历史v0.1／v0.2发布记录只说明当时存在发布流程；本 worker没有根据旧授权上传任何东西。

证据：`@project/nr-b580-int8/product/STATUS.md`、`README.md`、`FULL_GPU_COMFY_PLAN.md`、`experimental/PUBLICATION_STATUS.md`、`@project/nr-b580/GPU_RESOURCE_INTEROP_STATUS.md`、`OPTISCALER_INTEGRATION_STATUS.md`。旧产品仍是独立历史路线，不能以NR256的约10–12ms描述当前720p实机。

## 2026-09-13～09-15：向exact回迁，与真INT8无损算术的否决（H20–H21）

fast中可保持语义的graph／shortFP8／cubic／K8／decoder／layout被分阶段回迁exact。branched阶段完整调用98.308025→95.281525ms（约3.08%）、26帧／13参数；尺寸门另列256、480、1080、Apex。QKV另一轮95.99755→94.803725ms（約1.24%），后续默认推广／多尺寸证明以对应报告为准。reuse_v2的dataflow和compact各自变慢，组合约0.332%的微利不稳定，BM8更慢约8.99%。

`PROJECT_TIMING_AND_OPTIMIZATION.md` 的“精确线收益0”属于该账表的快线／行分批窗口，不能覆盖这阶段已经有报告的精确结构收益；其跨轮合成毫秒也不是全项目倍数证据。

无损INT8研究先问“哪些half累加块可用整数MMA精确表示”。CPU局部eligibility可得到0／16／92%等上界，但真实GPU输入上的快速分支为0；C128、C256、C512分别约0.137675→0.829025、0.23115→1.187735、0.0711→0.38444ms，动态检查／补偿／spill抵消收益。人工100% positive control只证明路径可用，不证明现实覆盖。

remainder／base128 digits在CPU完成39,401组合、143,360 fulltile groups及3,072 controls等数学门；真实补偿相当密集。GPU digits C128约25.63×慢，v2仍10.78×慢并有spill。它们是真整数MMA，却未采用；“真INT8”必须与量化边界、激活重复量化、packing和完整成本一起说。

证据：`@project/nr-b580/EXACT_OPTIMIZATION_MIGRATION_STATUS.md`、`EXACT_FAST_OPTIMIZATION_REUSE_AUDIT.md`、`EXACT_SEMANTIC_COST_AUDIT.md`、`EXACT_INT8_EXECUTION_PROBE.md`、`COMMUNITY_INT8_AND_REUSE.md`。CPU证明、GPU否决及compile v1→v2修复均保留。

## 2026-09-15～09-23：原尺寸eager、W8A8和launch/同步审计（H22–H27）

交接v3覆盖约110轮，包括当时原尺寸eager快线以及不成立的上界／归因。host prep／select fastpath、175重复quant删除、28而非40 padguard、156workspace reuse和合并readback在各自字节门下产生收益；matmul cache v1/v2没有可重复净收益而撤回。

| 路线 | 记录结果 | 本次解释 |
| --- | --- | --- |
| prep/select/quanttrim | prep約5.911ms；select配对0.647；quanttrim約3.8558 | eager特定版本收益；graph shipping绕过部分host路径 |
| padguard/qkvcache/FMA scalar cache | 约0.4255／0.7303／0.561ms | 每轮原对照与字节门有效，不能相加成当前720收益 |
| syncfast | 約6.94ms | 跨轮结果，不提升成同轮严谨因果值 |
| cut／tile | local看好，完整+5.5396ms，243字节同 | 完整链否决 |
| dense merge上界 | 初估6.73ms改为0.911ms | 411项串行链依赖使简单求和无效 |
| tail／sync | 7.6ms tail約96%排队排空；perfect-overlap估计0.2399ms | tail不是额外独立工作；不能全部省掉 |
| TTIR/roofline | IR operation／233TOPS等静态预算 | 不是ISA cycles或B580实测吞吐 |
| a_insitu/recheck W8A8 | GPUbusy+4.9636ms／+13.3%；另一整链约+11.4%、PSNR−9.62dB | 否决已实现未融合版本，不能否决所有潜在融合INT8 |
| corr strip原生FP8 | 12/12RGB相同，但GPUbusy+1.7835ms | LLIR少代码不等于硬件快；FP8软件lowering仍有成本 |

W8A8 producer-consumer chain捕获真实encoder.0.0输出：256×448×32、W32×32投影、FP8 byte边界、C32 pad264×456，验证runtime pointer identity。新两kernel device0.407656ms，比原融合三kernel0.387656ms慢5.16%；wall0.7129比0.6897也慢，虽然比旧split0.571771 device好。15 samples／5warmup、compile外置；不是243视频／当前整帧。trace的teardown异常在完成结果后出现，保留其失败边界，不扩展成“所有退出崩溃都允许”。

运动5tap short-scope约省1.6995ms，但1/243帧不同、max abs0.0557；K8两点FP16 XMX约省1.3285 frame／1.6075 GPU ms，峰差14.19/255；用户接受这些特定历史画面。ViT INT8→FP16提高約8.331dB，净性能CI跨0，因此属于质量支线。

**42.9709ms必须纠正。** 旧交接把它称为“产品默认态”，9月23日专门复核明确：这是原尺寸480×864、INT8 C512＋INT8 ViT的eager模型调用，不是Comfy产品节点，也不是独立ViT FP16质量支线。旧产品480输入实际走Face480ScaleNR256；旧G输出帧已遗失，不能宣称还能做原42.9709输出的逐像素重现。

新的同输入ABBA OFF41.0723／41.1688、ON36.2291／36.3562ms，取frames4..242共239稳态帧，约省4.8279ms／11.7408%。图使用front v6的reset／temporal两entry，排除decode、upload、readback、build/capture/warmup，并固定owned capture／输入manifest／profile hash。第一次GBK脚本解码与错误cache路径属于测量前失败，不写成GPU性能失败。

OptiGaze240／360、fractional motion、graph视频和分辨率滑块也在这阶段发展；shape/source/model/internal geometry必须分别记录。仅有`optigaze_resolution_slider_fused_v1`源码而未取得RESULT时，不补写成功或当前采用。

交接材料另记录一次目录／Git损坏与部分恢复事件，导致部分历史来源缺失；这是原记录的环境事故，不是本 worker验证了标准Python/Windows删除语义。缺失Git对象／旧帧是历史审计和许可追溯的缺口，不能用后来快照补称原history完整。

证据：`@project/nr-b580/reference/handover_v3/HANDOVER_v3.md`及H22–H27逐目录`RESULT.md`／`PLAN.md`／原脚本，尤其`w8a8_chain_v1/RESULT.md`、`corr_strip_v1/RESULT.md`、`graph_replay_480x864_abba_v1_20260923/RESULT.md`。部分原数据位于早期独立数据根，未纳入这次七根库存；它们在报告中有路径不代表本次重新hash或资产可公开。

## 2026-09-24～09-28：RE8、AMD结构与去模拟舍入（H28–H32）

RE8工程以OptiScaler固定基底、REFramework插件和原生XeSS组合。安装需要完整runtime、overlay与用户本地提取的SF模型资产；一个DLL不足。第三方版本、GPL边界、许可证与专有模型均须在公开工程中单列，历史开发成功不构成资产发布许可。

焦点问题表现为NR关闭时AltTab／手柄正常，NR开启的默认窗口消息／输入hook下崩溃；`ManualInputPolling=true`得到用户稳定性确认，但付出菜单鼠标点击透传副作用。不能把它称为所有焦点配置问题完全解决。

FSR桥命中FFX hook，但OptiScaler D3D12 device为空，未进入nested NGX，按合同fail-closed；用户在9月26日暂停。DLSS路线为NGX前NR再SR，不是Present末端覆盖；XeSS原生外层API嵌套NGX，需要正确color格式、motion尺寸和resource state。generic DLSS的12个mock调用证明接口代码，不等于第二个真实游戏已合格。

9月25日的copy/compute审计发现startup copy约12.47%不能代表稳态；14–18秒稳态显式copy约0.96%，但Compute标签仍含layout／数据移动，不能宣称其余99%全是乘加。XPU graph capture中记录profiling event不支持，Kineto/XPU崩溃留为工具限制；没有把失败重跑当正常性能数据。

AMD结构参考固定到`0bf535c64c35bc41d81fb32191bb40984887ca70`。作者900／1080结果、GPU和history/geometry不同，只借鉴共享QKV、token/branch布局和依赖表达，不把12.1／17.1或11.26／15.86ms迁移成B580实测。

| 结构筛选 | 完整受测结果 | 采用和限制 |
| --- | --- | --- |
| C64 QKV八块＋C32 tail七块＋post可见4channel K8 | 48039.249→37.199；54054.124→50.282；36027.625→26.849ms | 480／540字节门有效；360 P95+2.656，不默认 |
| post attention＋K8 joint | 540约省0.9655／480約0.7646ms | 非零motion BCCB、120/臂16warmup，各轮配对 |
| C128 direct QKV十二块 | 480约省0.48／540約1.03ms | 完整243byte门与cache门，不外推720所有路径 |
| C256 direct BM32十六块 | 约省0.93／1.25ms | 有效组合采用 |
| C512 row reuse／BM32 | 局部和全调用三模式均慢 | 保BM16，否决“减少读weight必快”的猜测 |
| ViT192／convrot／C32重复tail | 有微利／退化或仅candidate | 以各报告具体selected为准，不默认全用 |
| pre-att未量化full pool＋decoder C512window | 最终combo540两轮约1.091／0.849，4800.653／0.600ms | 全243byte及168param缓存门；切模式约4秒冷capture仍存在 |

几何是核心约束：540源960×540，模型544×960、内部640×1024；480模型480×864、内部512×896；motion以真实pixel尺度resize并缩放vector。padding可减少无效工作，但source、model、internal尺寸不允许混写。

9月27日用户允许fast尝试去掉FP8激活模拟舍入。先pre-att一处约省0.2446ms，再C32七块54050.5119→49.3495、48037.2496→36.6735ms；视频用户“肉眼看不出区别”，编码视频MAE约0.571/255、PSNR47.63、SSIM0.996149。它们是lossy视频指标，不是raw tensor误差。后续overlay覆盖C32/C128/C512/decoder/ViT/post等，权重E4M3解码、必要INT8量化scale和其他未删边界继续保留。

这条授权改变fast合同，不能倒过来声称9月15日corr-strip的严格字节失败经验无价值；同样不能要求当前fast重复实现已授权删除的激活舍入。exact独立保留。

证据：`@project/re8-b580-nr-xess/README.md`、`BUILD.md`、`THIRD_PARTY.md`、`VALIDATION.md`、`docs/*_20260925.md`、`AMD_INFERENCE_STRUCTURE_MAP_20260927.md`、`GAME_PADDING_GEOMETRY_AUDIT_20260927.md`、各`*_RESULT_20260927/28.md`、`FAST_FP8_REMOVAL_ABLATION_20260927.md`；接口路线见`@project/cyberpunk-b580-nr-opt/THREE_ROUTE_BRIDGES.md`、`GENERIC_DLSS_PORT.md`。

## 2026-09-28～09-30：Cyberpunk720、C512＋K8与FG功能门（H33）

Cyberpunk逐步从360／540诊断移到720p C512 library＋真实K8，再结合native C32和all6等已接受路径。早期540约89ms、360約35–37ms的静态／零运动baseline不是当前55–60ms实机。

原生XeSS链捕获1280×720输入、2560×1440输出，color为R11G11B10、motion为RG16F；与DLSS尾端state／额外original-color copy差异需要独立8帧证明，不能看到菜单选项就断言NR执行。

DLSSG→XeFG通过DebugView由用户确认实际生成。HUDless反复切换导致中断，`DisableHudless=true`清启动超过4100 NR帧后用户HUD正常；`DxgiSpoof=true`隐藏原生XeSS FG菜单，改false恢复，Streamline保留。NR＋native XeSS FG连续功能通过，但没有独立display-frame计数，不补称固定倍数／延迟。FSR仍为暂停路线。

证据：`@project/cyberpunk-b580-nr-opt/STATUS.md`、`DEFAULT_FAST_MODE_20260929.md`、`LIVE_DLSS_IDENTITY_RELAY.md`、`LIVE_XESS_DIAGNOSTIC.md`、`THREE_ROUTE_BRIDGES.md`、`BRIDGE_TIMER_20260929.md`。这部分保留接口和故障经验，不把FG成绩混入基础帧时间。

## 2026-10-01：numeric、CPU生命周期、闪烁与缓存（H34–H36）

30项numeric flags筛选中，去decoder舍入、native C32 cubic、fp32 fractional history、native front是当天阶段选项。三项54.569→51.437ms与随后加front51.476→50.938ms来自不同轮次，不能把差值相加。旧“六项”短轮51.575→51.339ms，长轮仅约0.142ms差且0.470ms漂移、候选未同时低于两baseline，曾恢复三／四项。这一阶段文档不能覆盖10月3日实际accepted all6。

数值／性能归因发现C64 repeat的完整adapter约+3.014ms，raw graph仅+0.118；validate+1.391、handoff+1.174等要区分inclusive。ViT denom完整adapter+0.640，而graph约−0.078，说明局部GPU小收益容易被完整调用成本覆盖。input copy57,262,080B和output clone5,529,600B是逻辑字节数，不是相应阶段耗时，也不是CPU修像素。生命周期优化收紧constant cache与owner/resource generation，不能为性能绕开真实退休或身份验证。

周期闪烁的旧90秒静止日志中1353 SR帧、1337 NR、16 raw fallback，source age66–247超过≤64门；下一NR恢复但没有reset。v1过度失效全部alias导致启动零NR；v2/v3改善但跨list仍出现age102等fallback。v4只依据当前CPU commandlist局部transition声明，保留真实state合同，不用不同CPU录制顺序推GPU执行顺序。

v4固定720连续90秒1309 NR/replay/compose/retire、无新fallback/失败/reset，用户“好像不闪了”；样本maxage57没有实机越界正样本，CPU门覆盖另列。v5默认精确识别目标进程、不需env开关，约30秒397新NR、0fallback，C512/K8资格真实执行。静止片段分数history的conditional路径未触发，不能称所有运动场景完全覆盖。

immutable shader cache的CPU双文件平均4.8551ms不是游戏关键预算。GPU44 OFF/ON对、88臂、973,209,600字节相同；32 ON仅两shader compile／一个root／两个PSO，warm命中无再hash。旧实机shadercache ON/OFF对为540p，不能冒作现役720对照。规范化cache-export metadata解决路径和group identity，不以“metadata完整”替代冷进程DiskOnly零compile/miss/write证明。

证据：`@project/cyberpunk-b580-nr-opt/docs/LIVE_WEB_720_OPTIONS_20261001.md`、`NUMERIC_COST_ATTRIBUTION_20261001.md`、`CPU_LIFECYCLE_20261001.md`、`PERIODIC_FLASH_20261001.md`、`BRIDGE_CPU_GPU_AND_OUTER_TIMING_20261001.md`。错误alias规则、字段／脚本失败、cache首轮失败均保留。

## 2026-10-02：纹理池、GPU接力与55–60ms用户验收（H37–H38）

原桥pack／device USM memcpy／unpack已经在GPU，优化对象是资源重复创建、producer宽CPU等待和串行worker资格，不是把一个原来CPU逐像素处理的桥突然改成GPU。真实consumer fence退休、owner／queue/context/device/epoch资格和OFF恢复门一直需要保留。

共享external fence接口存在并不证明当前UR实现可用。v3 wait10/11、signal20/21可行，但重复同import-object的新值失败；v4用两个forward fences、v5同fence两个import objects都512/512words通过，定位到import-object复用问题。每帧新forward import及完成退休的serial owner合同才形成后续路线。

CPU双持久线程、synthetic GPU三ON三OFF六帧、九类C++及48 Python检查是不同层证据。真实game run05 pool/handoff各30 snapshots、90.39秒1559完成帧／177 samples、无fail/mismatch/quarantine；后做同帧组合OFF/ON/ON/OFF，各30唯一完成帧，不能简单相加单开关局部数值。

| 同场景组合读数 | OFF→ON | 解释 |
| --- | --- | --- |
| CPU录制 | 3.38934→2.26551ms，省1.12383 | CPU record |
| 录制至真实退休跨度 | 54.94664→51.97559ms，省2.97105 | 含真实消费等待；两个ON均低于两个OFF |
| 同完成帧网页process | 50.40297→48.46948ms，省1.93349 | 网页与同帧桥数据匹配；不与前两行相加 |
| 人类RTSS／运动焦点 | 55–60ms；“画面和稳定性都正常” | FG关闭的基础帧时间；没有自动Present trace |

手动验收监看最初误读字段、第二次str/Path类型错，虽然实际运行健康，错误监看回执不能写成产品／GPU失败。修监看后实际10新增hits/bypass、native/host serial true、pending false，并由用户快转镜头／AltTab接受。纹理池＋GPU接力成为当前保留组合。

25项canonical源按gpu-hooks-only定点合入，pre-XeSS host只投影四处已测hook；9/9 C++和48 Python CPU门通过，旧ViT freeze未改。已验收trial ASI SHA为`12fe0f89f183e614c3e989a15de05c1f3b85601b9a4d7d0cf26096508a149309`，canonical重编ASI SHA为`368d973007ce84e888b346fce43dc32c9527d83c90745039e5e188a3f29611a9`，ABI一致不等于两二进制byte相同。实际G源码提取由main核验，本worker只读合入／本地提取回执。

外层terminal-sync v1只是CPU 19tests/44cases通过，真实资格错写成baseline而非c512_k8，现役无法取得；v2修到正确路由并加四factory/冷热点case，共23 CPU tests，仍无GPU净收益／部署证据，不能算已省同步。

真INT8重审确认C128 `_pairs`／`_project_dual`／`_qkv_direct`为实际整数MMA。owned facefixture局部16graph，FP16 complete0.099423ms、INT8 BM32 complete0.239710ms；bare INT8 core0.017543比FP16 0.020162快，却全模块2.41倍慢。CPU math oracle与中间MAE0.1021285、RMS0.1581487、max1.1953125不构成最终RGB画面门；IR真MMA也未测final ISA物理成本。旧random720输入不是实际游戏activation，历史标签须纠正。

证据：`BRIDGE_CPU_GPU_AND_OUTER_TIMING_20261001.md`、`SERIAL_GPU_BRIDGE_SOURCE_INTEGRATION_20261002.md`、`INT8_NATIVE_AND_DATAFLOW_AUDIT_20261002.md`，原`MAIN_CANONICAL_SOURCE_RECEIPT.json`／promoted／CPU_BUILD／MAIN_COMBINED_RAW_REVIEW／HUMAN_ACCEPTANCE位于`@experiments-D/cyberpunk-opt`相应目录，索引已pin可读取回执。

## 2026-10-03：现役全网审计与未采用候选（H39–H41）

全后端静态库存为2534 nodes／137 compiler keys／67 issues；18 workpackages、114 arms是覆盖／计划清单，不是已取得114组独立性能或67项实测瓶颈。CPU预编译用户16-core／48GB目标，设默认16 workers、内存限32／reserve8，实际一轮8workers峰4.24GiB、free33.60；GPU仍由Luna串行。

AMD再分析固定`9ec741522d267c4d2365af53081716fb2943068a`。六项计划涉及C512有效query rows、ViT stream-tail、C32 owned window、C64/C128共享QKV、跨token共享weight、C256依赖队列。C512 19712→15360行、静态减少4352行／3,422,552,064 MAC是预算；不是实测3GB带宽或已节省毫秒。作者1080 network9.5–10ms不能写成90fps完整Present。六方案截至该报告未测。

| 完整模块尝试 | 原始／重复结果 | 是否保留为现役 |
| --- | --- | --- |
| FDP标准档 | independent BCCB45.410→43.610ms，约省1.7997863／3.96%，输出改变 | 未采用；待游戏画面／实机 |
| FDP自然／电影 | 单轮52.820→44.694／53.176→45.182ms，MAE约6.77e-5／6.93e-5 | 未采用；需独立复测，收益不相加 |
| complete C512＋ViT | 46.005→54.067，慢8.062ms | 未采用 |
| r13真实QKV16 sites／FFN | 45.916→47.540／46.239→47.357ms；13output/history byte同 | 慢，未采用 |
| encoder | 46.325→47.750ms，小数值差max0.003173828 | 未采用 |
| history ScreenedLaunch | 46.269→46.418ms，cycles混合；此前热冷资格失败根因未闭合 | 未采用／未证收益 |
| r15 Swin | 44 modules／132 entries，实际block66；46.007→54.631ms，raw39.500→47.517 | 完整慢8.624；数值资格部分失败，未采用 |
| 原ViT | 45.937→55.122ms，真INT8慢9.185 | 未采用 |
| r16 hot-admission | 同未采用FFN/encoder约省0.427／0.381ms，部分cycle逆转 | 不是现役净收益 |
| r17→r18 rowquant-once | r17 AST membership static_assert compile失败；r18修同义三assert，45.649→46.542／45.453→46.456ms | 数学/冷热点/零spill通过，但完整慢0.893／1.003；未采用 |
| r18 raw graph | 約39.237→39.166／39.230→39.176ms | 0.071／0.054ms局部差不能覆盖完整退化 |
| native HDR raw2 | prepare/verify两帧raw pack/export字节及consumer退休通过 | 未跑模型／game／Present，不是新整桥认证 |
| Luna37–39修复 | 原截点报告仍precompile／串行待测 | 本worker未读当前active attempts，不补写结果 |

若模型资格失败、readonly cache miss、冷hot不一致、图消费者退休／teardown失败发生在测量前或退出阶段，要保留准确失败位置；不能全部称“内核慢”，也不能为了有timing把资格失败跳过。

r18 `accepted_baseline` constructor、空selectors/packages和baseline=true仅定义实验对照。`audit_impl`下优化arms没有部署；把r18全部目录叫“当前内核”会混入已否决候选。真实现役源由main从游戏runtime逐文件提取，初始回执current449／r18候选260；随后current-source manifest增加authoring来源，不能把扩展总数都叫installed runtime。worker已对本地stage按该manifest重hash，具体快照count／SHA在`CURRENT_SOURCE_RECEIPTS.json`。

用户本轮明确授权公开`gggz114514-oss/dlss5nr-b580`，main已告知clone/publish及源码／测量摘录落盘。历史worker没有执行GitHub写入；公开工程的资产许可、实际构建与干净安装门以main的发布review为准。源提取byte identity不意味着模型weights、frozen资产、私有运动接口和全部旧输入齐备。

证据：`@project/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/TEST_PROGRESS_20261003.md`、`WORK.json`、`phase2-complete/PHASE2-r18.json`、`BASELINE.json`、`reviews/amd-20261003-current720-reanalysis/REPORT.md`、`docs/FULL_BACKEND_AUDIT_20261003.md`、`CURRENT_BACKEND_REAUDIT_20261003.md`及main extraction/current-source回执。阅读入口和accepted参数另见`CURRENT_BASELINE_EVIDENCE.md`。

## 可迁移结论、受测经验与未证猜测

| 性质 | 结论 | 适用边界 |
| --- | --- | --- |
| 多阶段复证 | 真实producer/consumer全链、量化／padding／layout／launch成本决定INT8净收益 | 可用于新实现验收设计；不能断言INT8永远无收益 |
| 多阶段复证 | 字节一致、zero spill、短IR、局部快均不足证明完整快 | 以同版本完整process、重复cycle与游戏门闭合 |
| 多阶段复证 | graph让部分host eager fastpath离开热路径；shape/history/owner/retirement不能省 | 新shape/control/resource generation须重新资格 |
| 多阶段复证 | 原metadata、失败报告、冷cache/退出失败和撤回数字是不可替代证据 | 只清已证副本／可再生中间数据，保留源和manifest |
| 已受测 | 去激活模拟FP8舍入、C512+K8、all6及纹理池/接力可在当前游戏配置接受 | fast质量合同、特定场景55–60ms；exact独立 |
| 未证预算 | AMD六结构、MAC减少、全部共享weight/copy等可能收益 | 静态预算不可相加，不作部署／帧率声明 |
| 未证预算 | terminal-sync v2可以净省完整游戏时间 | 仅CPU ready，真实GPU等待可能转移，须Luna串行复测 |
| 已撤回／修正 | 42.9709产品默认、所有精确优化收益0、旧3.88×、tail全可省、TTIR=cycles | 原报告保留，但本档使用后续限定与纠正 |

历史可重读、源码和参数可重新索引，不等于全部实验可以完整独立重跑。缺口和分层复现操作见`REPRODUCIBILITY_GAPS.md`；本地库存27MB不要求整体发布。公开使用`PUBLIC_HISTORY_INDEX.json`、`HISTORY_SOURCE_SELECTION.public.json`及最终脱敏报告，最小文件选择不复制全部生成项目。
