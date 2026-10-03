# accepted720 与 AMD：逐核及纯数据操作补审 v1

这份补审把现役主调用链拆到实际 callable、库 GEMM、纯数据/标量操作和模型外桥。AMD 作者归档约9.5～10ms网络成绩及结构分析沿用原报告；此处集中交付可实施的逐核差异。

主对照：r18 accepted_baseline，manifest `699ee05cf1008610a4772dfeebe6398d7099c7c1ed141cbbaa91967da0a05f7f`；r15/r18 constructor逐字段相同。固定1280×720，内部1280×768，C512有效24×40，ViT240 tokens。游戏现役用户RTSS55–60ms；未采用完整模块回退及r18 audit候选均排除。

AMD固定HEAD：`9ec741522d267c4d2365af53081716fb2943068a`。本次重用已联网固定的源码；所有下列链接固定SHA，无新GPU运行。

库存共 **119条角色行**，涉及 **43个当前静态Triton callable定义**、**71个模型块位点**。同核不同producer/consumer会重复列角色；不等于这个数量的物理launch。另存139个编译绑定，132个与角色关联，7个单列未选/备选绑定。261个源码SHA只用于来源校验。

13帧只有两条捕获签名（首帧reset、随后history）、11次后续replay；front reset1/history12，fractional warp12/near0。图计数13包括捕获帧本身；whole-process78及rawbody50属于不同范围。captured_dispatch是逻辑旧算子计数，不是物理kernel trace。库内核、逐次物理派发与ISA覆盖率未被记录，不能写100%。

读取说明：每个编号下第一表给实际接口/算量，第二表给布局/AMD对应/改造与证据。JSON保留每个编译key的完整signature/常量、源SHA、capture/runtime计数和AST调用点。无AMD同配方对应的时序/控制照样列出。

快速版允许数值变化。N2先保持两个K512 QKV及P4 half投影仅为隔离结构收益；结构通过后可测full-K单累加、native norm/exponent/denominator。当前accepted这些开关仍false/reference；C128跨branch FP32已采纳。C512/ViT FFN已是i8矩阵→i32，不能重做“从FP16模拟改INT8”。不追加FP8模拟规则；四级Decoder merge已有输出lattice属于当前代码事实，去掉它是另一个未选fast候选。

## 选择链与块位点

| 函数 | 实际选择 |
|---|---|
| [FullsizeGameModes._select](../../../experiments/2026-10-03/r18/source/game/nr_game_fullsize.py#L465) | constructor决定现役all6，height720，variant unrounded |
| [open_session](../../../experiments/2026-10-03/r18/source/game/re4_session_v1.py#L211) | 安装geometry720/internal768 contract |
| [FullsizeSession.__init__](../../../experiments/2026-10-03/r18/source/game/fullsize_session_v1.py#L60) | rows_scopes全行FFN；覆盖SplitSwin/Vit.forward；排除graph/rewrite/call_guard/compact_queries旧小图 |
| [install](../../../experiments/2026-10-03/r18/source/game/rows_scopes_v1.py#L117) | 实际FFN5/4核；不安装audit_impl |
| [installed](../../../experiments/2026-10-03/r18/source/game/three_structure_combo_v1.py#L23) | C64/C128/C256 directpack，普通C32七tail，pre尾，C512 decoder projection |
| [installed](../../../experiments/2026-10-03/r18/source/game/post_attention_k8_combined_v1.py#L52) | post entry→MLP→QKV→tail→croppedhead |
| [installed](../../../experiments/2026-10-03/r18/source/game/c512_k8_joint_scope_720_v1.py#L20) | torch.mm all16 C512 +native pre/post K |
| [installed](../../../experiments/2026-10-03/r18/source/game/c32_hidden_native_720_v1.py#L122) | pre/8C32/post native hidden |
| [installed_720](../../../experiments/2026-10-03/r18/source/game/c512_probability_unround_scope_v1.py#L86) | all16 no probability FP8 simulation |
| [installed](../../../experiments/2026-10-03/r18/source/game/numeric_cleanup_suite_720_v1.py#L259) | C128 native branch /fp32_fractional history/native_both front |
| [installed](../../../experiments/2026-10-03/r18/source/game/audit_history_host_720_v1.py#L160) | before_numeric allfalse：准入/归属，未替换数学计算候选 |
| [forward_front](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/capture_body_v1.py#L50) | 完整71block源调用链；capture期间保替换作用域 |
| [GraphFront.forward](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/graph_front_v1.py#L62) | 2capture签名+11固定序列replay；Python计数不随图内物理派发重新递增 |

块0 pre；1–4 C32、5–8 C64、9–14 C128、15–22 C256；23–30编码C512；31–38 ViT；39 decoder输入；40–47解码C512；48–55 C256、56–61 C128、62–65 C64、66–69 C32；70 post。每个具体形状/shift/up或down role保存在JSON `main_block_sites`。没有跳任何块。

## 逐核差异


### front

#### F-reset — `front_noise_native_720_kernel_v1.front`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/front_noise_native_720_kernel_v1.py:29](../../../experiments/2026-10-03/r18/source/game/front_noise_native_720_kernel_v1.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | RGB f32[720,1280,3]→f16[768,1280,16]；B128，4warps；TEMPORAL=false，seed 0；HISTORY_COMPONENTS=false |
| 运算/必要half边界 | 标量 native radius+trig；反射边界/控制10…14；half 发布，无新增FP8 |
| producer → consumer | RGB / H-normalized → pre K16 |
| 读写、布局和临时 | 读RGB及normalized history/reciprocal；写16-lane front 31,457,280B；融合noise，无逐lane临时 |
| AMD 固定函数 | [hip/prefix_fast.hip:57 dlss5_prefix_fast_features](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/prefix_fast.hip#L57)（analogous_compute） |
| 差异/已实现 | AMD prefix可连K16投影/整窗；regular重置输入不等价真实时序 |
| 改造优先级、收益证据 | gap-front-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | runtime_hash_and_calls；关联2个绑定（不自动当执行）；无逐物理dispatch轨迹 |

fixed13实际调用：1；callee hash及binary见JSON runtime_evidence。

#### F-history — `front_noise_native_720_kernel_v1.front`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/front_noise_native_720_kernel_v1.py:29](../../../experiments/2026-10-03/r18/source/game/front_noise_native_720_kernel_v1.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | RGB f32[720,1280,3]→f16[768,1280,16]；B128，4warps；TEMPORAL=true，seed 1…12；HISTORY_COMPONENTS=false |
| 运算/必要half边界 | 标量 native radius+trig；反射边界/控制10…14；half 发布，无新增FP8 |
| producer → consumer | RGB / H-normalized → pre K16 |
| 读写、布局和临时 | 读RGB及normalized history/reciprocal；写16-lane front 31,457,280B；融合noise，无逐lane临时 |
| AMD 固定函数 | [hip/prefix_fast.hip:57 dlss5_prefix_fast_features](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/prefix_fast.hip#L57)（analogous_compute） |
| 差异/已实现 | AMD prefix可连K16投影/整窗；regular重置输入不等价真实时序 |
| 改造优先级、收益证据 | gap-front-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | runtime_hash_and_calls；关联2个绑定（不自动当执行）；无逐物理dispatch轨迹 |

fixed13实际调用：12；callee hash及binary见JSON runtime_evidence。


### history

#### H-axes — `history_numeric_suite_720_v1_kernel._prepare_axes`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/history_numeric_suite_720_v1_kernel.py:54](../../../experiments/2026-10-03/r18/source/game/history_numeric_suite_720_v1_kernel.py#L54)；triton_kernel |
| 配置、shape、dtype、full K | motion f32[720,1280,2]；BLOCK128；10-plane f32 axes，NATIVE=false/DIRECT=false |
| 运算/必要half边界 | 标量坐标/权重+reciprocal table，full five-tap轴系数 |
| producer → consumer | 实际运动及private previous → H-sample |
| 读写、布局和临时 | 读motion/table/维度倒数；写axes 36,864,000B +invalid 3,686,400B（每历史帧） |
| AMD 固定函数 | [src/native_pre_upscale.h:220 reset](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/src/native_pre_upscale.h#L220)（no_equivalent_real_motion_history_on_regular_route） |
| 差异/已实现 | AMD regular无相同运动warp；不得以去掉history换收益 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | runtime_hash_and_calls；关联2个绑定（不自动当执行）；无逐物理dispatch轨迹 |

fixed13实际调用：12；callee hash及binary见JSON runtime_evidence。

#### H-sample — `history_numeric_suite_720_v1_kernel._five_tap`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/history_numeric_suite_720_v1_kernel.py:176](../../../experiments/2026-10-03/r18/source/game/history_numeric_suite_720_v1_kernel.py#L176)；triton_kernel |
| 配置、shape、dtype、full K | previous f16 HWC3；VALUE_FP32=true；BLOCK64，4warps；DEBUG=false/WRITE_NORMALIZED=true |
| 运算/必要half边界 | 5个bilinear tap每tap4texel；FP32 fractional值混合，原half坐标/标量table边界 |
| producer → consumer | H-axes → F-history及post temporal blend |
| 读写、布局和临时 | 读axes/previous/table；写numerator11,059,200B、reciprocal3,686,400B、normalized11,059,200B、invalid3,686,400B；非DEBUG无TAPS存储 |
| AMD 固定函数 | [src/native_pre_upscale.h:220 reset](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/src/native_pre_upscale.h#L220)（no_equivalent_real_motion_history_on_regular_route） |
| 差异/已实现 | 已融合5tap；normalized虽然不返回仍写；tap逻辑计数60不是60个核 |
| 改造优先级、收益证据 | gap-history-output；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | runtime_hash_and_calls；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

fixed13实际调用：12；callee hash及binary见JSON runtime_evidence。

#### H-route — `motion.half().float() / round / sub / abs / le / all / bool`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/history_numeric_suite_720_v1.py:984](../../../experiments/2026-10-03/r18/source/game/history_numeric_suite_720_v1.py#L984)；torch_data_or_ALU |
| 配置、shape、dtype、full K | HWC2 1,843,200元素→host bool；fixed13 12 fractional /0 near |
| 运算/必要half边界 | 标量/通用reduction，无DPAS |
| producer → consumer | motion → H-axes或H-near |
| 读写、布局和临时 | 数次全motion遍历和bool同步；具体ATen物理核未知 |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | AMD没有对应路线；不能因bool同步忽略真实motion predicate |
| 改造优先级、收益证据 | gap-summary；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### H-near — `owned near_warp→_axis_weights→_sample_five_axes→_half_texture_counts / NativeReciprocalTable.forward`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/sampling.py:113](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/sampling.py#L113)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 720×1280，near-integer predicate；5tap/full image |
| 运算/必要half边界 | half纹理计数及FP32 table；fp32_fractional不把near改为fp32_all |
| producer → consumer | H-route → F-history / post |
| 读写、布局和临时 | torch meshgrid/half/floor/索引/权重/5sample/表查/分组及total；详见附录AST逐调用点 |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | 仍为可达accepted路径，但fixed13近整数0；非新pixel_rgb/fused_near |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | conditional_unobserved；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### H-tables — `dimension table indexed view / cached scalar arguments`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/reciprocal.py:47](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/reciprocal.py#L47)；torch_data_or_ALU |
| 配置、shape、dtype、full K | W1280/H720 f32标量 |
| 运算/必要half边界 | 仅查表/metadata，非独立矩阵 |
| producer → consumer | 常量 → H-axes/H-sample |
| 读写、布局和临时 | 读常量，返回标量view；dimension.width/height各12为逻辑调用，非24物理核 |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | AMD可原生倒数；保留table不等于需要另起矩阵核 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### pre

#### P-k16 — `native_k8_fp16_v1._native_k8_fp16`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/native_k8_fp16_v1.py:26](../../../experiments/2026-10-03/r18/source/game/native_k8_fp16_v1.py#L26)；triton_kernel |
| 配置、shape、dtype、full K | M983040 K16 N32；BM/BN/BK32，4warps；f16 |
| 运算/必要half边界 | FP16 tl.dot→f32 accumulator→half；K16完整（文件名K8不是有效K） |
| producer → consumer | F-reset/history → P-mlp |
| 读写、布局和临时 | front HWC16读一次到tile；写HWC32 62,914,560B |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | native前投影已实现；AMD inline prefix消除该HWC32中间 |
| 改造优先级、收益证据 | gap-front-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

捕获provider门：`native_k8_pre_post`=True。该门证明对应provider参与捕获；具体kernel为源码追溯，未声称library/ATen底层全部观察。

#### P-mlp — `native_cubic_c32_v1._kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/native_cubic_c32_v1.py:24](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/native_cubic_c32_v1.py#L24)；triton_kernel |
| 配置、shape、dtype、full K | M983040 C32→128→32；BM32，4warps，ROUND_ACTIVATION=false |
| 运算/必要half边界 | FP16 DPAS候选；4×K32 hidden片顺序累计，cubic native half，half residual/store |
| producer → consumer | P-k16 / padded C32 / O-entry → P-qkv |
| 读写、布局和临时 | 读X/Wexp/Wcontract/skip；hidden保留寄存器，写MLP HWC32；无需重做hidden fusion |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 已融合exp+cubic+contract；AMD继续在同窗做QKV/tail，避免MLP落地 |
| 改造优先级、收益证据 | gap-front-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联6个绑定（不自动当执行）；无逐物理dispatch轨迹 |

捕获provider门：`c32_hidden_native_ten`=True。该门证明对应provider参与捕获；具体kernel为源码追溯，未声称library/ATen底层全部观察。


### C32

#### S32-mlp — `native_cubic_c32_v1._kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/native_cubic_c32_v1.py:24](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/native_cubic_c32_v1.py#L24)；triton_kernel |
| 配置、shape、dtype、full K | M245760/254016/248832/250880 C32→128→32；BM32，4warps，ROUND_ACTIVATION=false |
| 运算/必要half边界 | FP16 DPAS候选；4×K32 hidden片顺序累计，cubic native half，half residual/store |
| producer → consumer | P-k16 / padded C32 / O-entry → S32-qkv或D32-qkv |
| 读写、布局和临时 | 读X/Wexp/Wcontract/skip；hidden保留寄存器，写MLP HWC32；无需重做hidden fusion |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 已融合exp+cubic+contract；AMD继续在同窗做QKV/tail，避免MLP落地 |
| 改造优先级、收益证据 | N3；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联6个绑定（不自动当执行）；无逐物理dispatch轨迹 |

捕获provider门：`c32_hidden_native_ten`=True。该门证明对应provider参与捕获；具体kernel为源码追溯，未声称library/ATen底层全部观察。

N3整个方案沿用[0, 2.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### post

#### O-mlp — `native_cubic_c32_v1._kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/native_cubic_c32_v1.py:24](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/native_cubic_c32_v1.py#L24)；triton_kernel |
| 配置、shape、dtype、full K | M999488=776×1288 C32→128→32；BM32，4warps，ROUND_ACTIVATION=false |
| 运算/必要half边界 | FP16 DPAS候选；4×K32 hidden片顺序累计，cubic native half，half residual/store |
| producer → consumer | P-k16 / padded C32 / O-entry → O-qkv |
| 读写、布局和临时 | 读X/Wexp/Wcontract/skip；hidden保留寄存器，写MLP HWC32；无需重做hidden fusion |
| AMD 固定函数 | [hip/wave_owned_c32.inc:537 c32_wave1_post](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L537)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 已融合exp+cubic+contract；AMD继续在同窗做QKV/tail，避免MLP落地 |
| 改造优先级、收益证据 | gap-front-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联6个绑定（不自动当执行）；无逐物理dispatch轨迹 |

捕获provider门：`c32_hidden_native_ten`=True。该门证明对应provider参与捕获；具体kernel为源码追溯，未声称library/ATen底层全部观察。


### pre

#### P-qkv — `fused_c32_projection_native_half_v1._kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fused_c32_projection_native_half_v1.py:25](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fused_c32_projection_native_half_v1.py#L25)；triton_kernel |
| 配置、shape、dtype、full K | 768×1280 C32，BM32/4warps；grid(rowtile,3)，fullK32 |
| 运算/必要half边界 | 每Q/K/V独立tl.dot FP16→f32→half；Q/K native half norm/scale，unround |
| producer → consumer | P-mlp → P-tail |
| 读写、布局和临时 | 读MLP/W/order/scale；3个packed[windows,64,32]half，写3×M×32×2B |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 已直接pack，无raster QKV中间；AMD一窗保有MLP与QKV |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### C32

#### S32-qkv — `fused_c32_projection_native_half_v1._kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fused_c32_projection_native_half_v1.py:25](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fused_c32_projection_native_half_v1.py#L25)；triton_kernel |
| 配置、shape、dtype、full K | 384/392×640/648 C32，BM32/4warps；grid(rowtile,3)，fullK32 |
| 运算/必要half边界 | 每Q/K/V独立tl.dot FP16→f32→half；Q/K native half norm/scale，unround |
| producer → consumer | S32-mlp → S32-tail |
| 读写、布局和临时 | 读MLP/W/order/scale；3个packed[windows,64,32]half，写3×M×32×2B |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 已直接pack，无raster QKV中间；AMD一窗保有MLP与QKV |
| 改造优先级、收益证据 | N3；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N3整个方案沿用[0, 2.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### decoder32

#### D32-qkv — `fused_c32_projection_native_half_v1._kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fused_c32_projection_native_half_v1.py:25](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fused_c32_projection_native_half_v1.py#L25)；triton_kernel |
| 配置、shape、dtype、full K | 384×640 C32，BM32/4warps；grid(rowtile,3)，fullK32 |
| 运算/必要half边界 | 每Q/K/V独立tl.dot FP16→f32→half；Q/K native half norm/scale，unround |
| producer → consumer | D32-mlp → D32-attn |
| 读写、布局和临时 | 读MLP/W/order/scale；3个packed[windows,64,32]half，写3×M×32×2B |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 已直接pack，无raster QKV中间；AMD一窗保有MLP与QKV |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### post

#### O-qkv — `fused_c32_projection_native_half_v1._kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fused_c32_projection_native_half_v1.py:25](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fused_c32_projection_native_half_v1.py#L25)；triton_kernel |
| 配置、shape、dtype、full K | 776×1288 C32，BM32/4warps；grid(rowtile,3)，fullK32 |
| 运算/必要half边界 | 每Q/K/V独立tl.dot FP16→f32→half；Q/K native half norm/scale，unround |
| producer → consumer | O-mlp → O-tail |
| 读写、布局和临时 | 读MLP/W/order/scale；3个packed[windows,64,32]half，写3×M×32×2B |
| AMD 固定函数 | [hip/wave_owned_c32.inc:537 c32_wave1_post](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L537)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 已直接pack，无raster QKV中间；AMD一窗保有MLP与QKV |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### pre

#### P-tail — `post_attention_fusion_v1._attention_project`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/post_attention_fusion_v1.py:24](../../../experiments/2026-10-03/r18/source/game/post_attention_fusion_v1.py#L24)；triton_kernel |
| 配置、shape、dtype、full K | 768×1280；2query tiles/window×32q，full64 keys，C32 |
| 运算/必要half边界 | QK/AV/projection FP16 tl.dot→f32；exp/denominator half，unrounded probabilities；half残差 |
| producer → consumer | P-qkv → 下一块/pool/post head |
| 读写、布局和临时 | 读QKV/bias/Wproj/MLP；score/exp/AV寄存器；写有效HWC32，已融合crop/residual |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | attention+projection已经融合；AMD优势在继续闭合前段 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联6个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### C32

#### S32-tail — `post_attention_fusion_v1._attention_project`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/post_attention_fusion_v1.py:24](../../../experiments/2026-10-03/r18/source/game/post_attention_fusion_v1.py#L24)；triton_kernel |
| 配置、shape、dtype、full K | 384×640；4shift；2query tiles/window×32q，full64 keys，C32 |
| 运算/必要half边界 | QK/AV/projection FP16 tl.dot→f32；exp/denominator half，unrounded probabilities；half残差 |
| producer → consumer | S32-qkv → 下一块/pool/post head |
| 读写、布局和临时 | 读QKV/bias/Wproj/MLP；score/exp/AV寄存器；写有效HWC32，已融合crop/residual |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | attention+projection已经融合；AMD优势在继续闭合前段 |
| 改造优先级、收益证据 | N3；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联6个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N3整个方案沿用[0, 2.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### post

#### O-tail — `post_attention_fusion_v1._attention_project`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/post_attention_fusion_v1.py:24](../../../experiments/2026-10-03/r18/source/game/post_attention_fusion_v1.py#L24)；triton_kernel |
| 配置、shape、dtype、full K | 768×1280；shift4,4；2query tiles/window×32q，full64 keys，C32 |
| 运算/必要half边界 | QK/AV/projection FP16 tl.dot→f32；exp/denominator half，unrounded probabilities；half残差 |
| producer → consumer | O-qkv → 下一块/pool/post head |
| 读写、布局和临时 | 读QKV/bias/Wproj/MLP；score/exp/AV寄存器；写有效HWC32，已融合crop/residual |
| AMD 固定函数 | [hip/wave_owned_c32.inc:537 c32_wave1_post](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L537)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | attention+projection已经融合；AMD优势在继续闭合前段 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联6个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### decoder32

#### D32-attn — `fused_swin_core_native_half_v1._kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fused_swin_core_native_half_v1.py:59](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fused_swin_core_native_half_v1.py#L59)；triton_kernel |
| 配置、shape、dtype、full K | windows3840×64×32；BM32，4warps，ROUND_WEIGHTS=false；full64key |
| 运算/必要half边界 | FP16 QK/AV +half exp/denominator；无FP8 probability |
| producer → consumer | D32-qkv → D32-unpack |
| 读写、布局和临时 | QKV→packed attended half；score局部，无global score |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | block66经DecoderGather专路，不是ordinary7融合tail |
| 改造优先级、收益证据 | N3；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N3整个方案沿用[0, 2.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### D32-unpack — `window_layout_v1._unpack`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/window_layout_v1.py:28](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/window_layout_v1.py#L28)；triton_kernel |
| 配置、shape、dtype、full K | 384×640 C32；BLOCK256，inverse order |
| 运算/必要half边界 | 纯数据搬运/布局，非DPAS |
| producer → consumer | D32-attn → D32-project |
| 读写、布局和临时 | packed attended读一写一→HWC32 15,728,640B |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | block66还有显式unpack；AMD up-body闭合 |
| 改造优先级、收益证据 | N3；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联5个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N3整个方案沿用[0, 2.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### D32-project — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | M245760 K32 N32 INITIALIZED=true；BM16 BN32 BK32，4warps |
| 运算/必要half边界 | FP16 tl.dot→f32+half initial→half |
| producer → consumer | D32-unpack / D32-skip → post input |
| 读写、布局和临时 | 读attended/W/half initial；写HWC32；skip multiply另列 |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 此支有独立projection；不要把post融合误贴block66 |
| 改造优先级、收益证据 | N3；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N3整个方案沿用[0, 2.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### D32-skip — `(mlp*skip_scale).half()`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/decoder_gather_scope_v1.py:79](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/decoder_gather_scope_v1.py#L79)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 384×640×32 f16 |
| 运算/必要half边界 | 标量乘+half初值 |
| producer → consumer | D32-mlp → D32-project |
| 读写、布局和临时 | 写initial15,728,640B→D32-project |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | AMD窗内保有残差 |
| 改造优先级、收益证据 | N3；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N3整个方案沿用[0, 2.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### C32

#### S32-pad — `torch.nn.functional.pad / crop views`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c32_repeated_tail_fusion_v1.py:34](../../../experiments/2026-10-03/r18/source/game/c32_repeated_tail_fusion_v1.py#L34)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 384×640→392×648/384×648/392×640；4shift |
| 运算/必要half边界 | 零扩展非cyclic roll；pure data |
| producer → consumer | 前块 → S32-mlp |
| 读写、布局和临时 | 非零shift读有效X写完整padded X；crop为view/或tail地址处理 |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | AMD映射输入避免padded global surface |
| 改造优先级、收益证据 | N3；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N3整个方案沿用[0, 2.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### C64

#### C64-pairs — `batched_branched_mlp_v1._pairs`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/batched_branched_mlp_v1.py:22](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/batched_branched_mlp_v1.py#L22)；triton_kernel |
| 配置、shape、dtype、full K | 192/200×320/328 C64；2 branches C→128→32；BM32/4warps；fullK64/K128 |
| 运算/必要half边界 | FP16 tl.dot，K32片；native cubic half（表/实现按callee），reduce half，ROUND_REDUCED=false |
| producer → consumer | C64-pad → C64-mlpproject |
| 读写、布局和临时 | 读padded X与branch W；hidden寄存器；写half latent[2,M,32]（总M×C×2B） |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | C128 pairwise已开；AMD QT4共享W并接QKV/tail |
| 改造优先级、收益证据 | N5；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

捕获provider门：`c64_all_eight`=True。该门证明对应provider参与捕获；具体kernel为源码追溯，未声称library/ATen底层全部观察。

N5整个方案沿用[0, 0.6]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C64-mlpproject — `batched_branched_mlp_v1._project`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/batched_branched_mlp_v1.py:48](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/batched_branched_mlp_v1.py#L48)；triton_kernel |
| 配置、shape、dtype、full K | 192/200×320/328 C64；BM16 BN64，4warps；2 full K32 branches |
| 运算/必要half边界 | 每branch FP16 dot→FP32加上一half结果→half；顺序边界 |
| producer → consumer | C64-pairs → C64-qkv |
| 读写、布局和临时 | 读latent/Wproj/X/skip；写MLP HWC64；BN64分列重复读latent，次数C/64 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | C128 native branch累积已实现；C64/256旧half边界可fast数学后续 |
| 改造优先级、收益证据 | N5；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N5整个方案沿用[0, 0.6]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C64-qkv — `c64_qkv_direct_pack_v2._qkv_direct_pack`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c64_qkv_direct_pack_v2.py:31](../../../experiments/2026-10-03/r18/source/game/c64_qkv_direct_pack_v2.py#L31)；triton_kernel |
| 配置、shape、dtype、full K | 192/200×320/328 C64；head2，32channels/head；BM16 BN32 BK32，4warps；fullK64 |
| 运算/必要half边界 | FP16 DPAS目标；Q/K half norm+scale，Vhalf；ROUND_QKV=false |
| producer → consumer | C64-mlpproject → C64-tail |
| 读写、布局和临时 | head×Q/K/V grid独立重复读A；直接写3×head×windows×64×32 half，没有raster QKV中间 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:563 QKV_FUSE](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L563)（analogous_compute） |
| 差异/已实现 | 直接pack已开；AMD C64/128 QKV_FUSE共读A，C256结构不同 |
| 改造优先级、收益证据 | N4；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N4整个方案沿用[0, 0.6]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C64-tail — `c64_attention_project_fused_720_v1._fused`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c64_attention_project_fused_720_v1.py:19](../../../experiments/2026-10-03/r18/source/game/c64_attention_project_fused_720_v1.py#L19)；triton_kernel |
| 配置、shape、dtype、full K | 192/200×320/328 C64；BM32，4warps；2heads，全64keys，全K64 projection |
| 运算/必要half边界 | QK/AV/跨head projection FP16 DPAS→f32；half exp/denom/attended/residual |
| producer → consumer | C64-qkv → 下一块/pool |
| 读写、布局和临时 | 读packed QKV/BIAS/W/MLP；score/AV局部；直接写valid HWC，已免unpack及attended落地 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 现役已经融合attention/projection/crop；不能重复计此收益 |
| 改造优先级、收益证据 | N4；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

捕获provider门：`c64_attention_project_eight`=True。该门证明对应provider参与捕获；具体kernel为源码追溯，未声称library/ATen底层全部观察。

N4整个方案沿用[0, 0.6]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C64-pad — `torch.nn.functional.pad`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/multihead_block.py:89](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/multihead_block.py#L89)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 192/200×320/328 C64；shift0/4，每边对8齐 |
| 运算/必要half边界 | pure data，unround family边界返回原half |
| producer → consumer | 前块/gather → C64-pairs |
| 读写、布局和临时 | 写padded HWC；后续全部MLP会处理padding；AMD mapped input与pad-attention需分开 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 替换padding只省搬运不能自动删attention full64keys；编码tail仍需pool |
| 改造优先级、收益证据 | N4；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N4整个方案沿用[0, 0.6]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### C128

#### C128-pairs — `branched_mlp_pairwise_720_v1._pairs_pairwise`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/branched_mlp_pairwise_720_v1.py:35](../../../experiments/2026-10-03/r18/source/game/branched_mlp_pairwise_720_v1.py#L35)；triton_kernel |
| 配置、shape、dtype、full K | 96/104×160/168 C128；4 branches C→128→32；BM32/4warps；fullK128/K128 |
| 运算/必要half边界 | FP16 tl.dot，K32片；native cubic half（表/实现按callee），reduce half，ROUND_REDUCED=false |
| producer → consumer | C128-pad → C128-mlpproject |
| 读写、布局和临时 | 读padded X与branch W；hidden寄存器；写half latent[4,M,32]（总M×C×2B） |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | C128 pairwise已开；AMD QT4共享W并接QKV/tail |
| 改造优先级、收益证据 | N5；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

捕获provider门：`c128_pairwise_twelve`=True。该门证明对应provider参与捕获；具体kernel为源码追溯，未声称library/ATen底层全部观察。

N5整个方案沿用[0, 0.6]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C128-mlpproject — `branch_accum_native_720_kernel_v1._project_fp32`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/branch_accum_native_720_kernel_v1.py:11](../../../experiments/2026-10-03/r18/source/game/branch_accum_native_720_kernel_v1.py#L11)；triton_kernel |
| 配置、shape、dtype、full K | 96/104×160/168 C128；BM16 BN64，4warps；4 full K32 branches |
| 运算/必要half边界 | C128已FP32跨branch累积→half |
| producer → consumer | C128-pairs → C128-qkv |
| 读写、布局和临时 | 读latent/Wproj/X/skip；写MLP HWC128；BN64分列重复读latent，次数C/64 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | C128 native branch累积已实现；C64/256旧half边界可fast数学后续 |
| 改造优先级、收益证据 | N5；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | runtime_hash_and_calls；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N5整个方案沿用[0, 0.6]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C128-qkv — `c128_qkv_direct_pack_one_v1._direct_pack`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c128_qkv_direct_pack_one_v1.py:30](../../../experiments/2026-10-03/r18/source/game/c128_qkv_direct_pack_one_v1.py#L30)；triton_kernel |
| 配置、shape、dtype、full K | 96/104×160/168 C128；head4，32channels/head；BM16 BN32 BK32，4warps；fullK128 |
| 运算/必要half边界 | FP16 DPAS目标；Q/K half norm+scale，Vhalf；ROUND_QKV=false |
| producer → consumer | C128-mlpproject → C128-tail |
| 读写、布局和临时 | head×Q/K/V grid独立重复读A；直接写3×head×windows×64×32 half，没有raster QKV中间 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:563 QKV_FUSE](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L563)（analogous_compute） |
| 差异/已实现 | 直接pack已开；AMD C64/128 QKV_FUSE共读A，C256结构不同 |
| 改造优先级、收益证据 | N4；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N4整个方案沿用[0, 0.6]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C128-tail — `c128_attention_project_fused_720_v1._fused`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c128_attention_project_fused_720_v1.py:19](../../../experiments/2026-10-03/r18/source/game/c128_attention_project_fused_720_v1.py#L19)；triton_kernel |
| 配置、shape、dtype、full K | 96/104×160/168 C128；BM16，4warps；4heads，全64keys，全K128 projection |
| 运算/必要half边界 | QK/AV/跨head projection FP16 DPAS→f32；half exp/denom/attended/residual |
| producer → consumer | C128-qkv → 下一块/pool |
| 读写、布局和临时 | 读packed QKV/BIAS/W/MLP；score/AV局部；直接写valid HWC，已免unpack及attended落地 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 现役已经融合attention/projection/crop；不能重复计此收益 |
| 改造优先级、收益证据 | N4；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

捕获provider门：`c128_attention_project_twelve`=True。该门证明对应provider参与捕获；具体kernel为源码追溯，未声称library/ATen底层全部观察。

N4整个方案沿用[0, 0.6]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C128-pad — `torch.nn.functional.pad`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/multihead_block.py:89](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/multihead_block.py#L89)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 96/104×160/168 C128；shift0/4，每边对8齐 |
| 运算/必要half边界 | pure data，unround family边界返回原half |
| producer → consumer | 前块/gather → C128-pairs |
| 读写、布局和临时 | 写padded HWC；后续全部MLP会处理padding；AMD mapped input与pad-attention需分开 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 替换padding只省搬运不能自动删attention full64keys；编码tail仍需pool |
| 改造优先级、收益证据 | N4；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N4整个方案沿用[0, 0.6]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### C256

#### C256-pairs — `batched_branched_mlp_v1._pairs`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/batched_branched_mlp_v1.py:22](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/batched_branched_mlp_v1.py#L22)；triton_kernel |
| 配置、shape、dtype、full K | 48/56×80/88 C256；8 branches C→128→32；BM16/4warps；fullK256/K128 |
| 运算/必要half边界 | FP16 tl.dot，K32片；native cubic half（表/实现按callee），reduce half，ROUND_REDUCED=false |
| producer → consumer | C256-pad → C256-mlpproject |
| 读写、布局和临时 | 读padded X与branch W；hidden寄存器；写half latent[8,M,32]（总M×C×2B） |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | C128 pairwise已开；AMD QT4共享W并接QKV/tail |
| 改造优先级、收益证据 | N5；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

捕获provider门：`c256_all_sixteen`=True。该门证明对应provider参与捕获；具体kernel为源码追溯，未声称library/ATen底层全部观察。

N5整个方案沿用[0, 0.6]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C256-mlpproject — `batched_branched_mlp_v1._project`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/batched_branched_mlp_v1.py:48](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/batched_branched_mlp_v1.py#L48)；triton_kernel |
| 配置、shape、dtype、full K | 48/56×80/88 C256；BM16 BN64，4warps；8 full K32 branches |
| 运算/必要half边界 | 每branch FP16 dot→FP32加上一half结果→half；顺序边界 |
| producer → consumer | C256-pairs → C256-qkv |
| 读写、布局和临时 | 读latent/Wproj/X/skip；写MLP HWC256；BN64分列重复读latent，次数C/64 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | C128 native branch累积已实现；C64/256旧half边界可fast数学后续 |
| 改造优先级、收益证据 | N5；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N5整个方案沿用[0, 0.6]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C256-qkv — `c256_qkv_direct_pack_one_v1._direct_pack`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c256_qkv_direct_pack_one_v1.py:30](../../../experiments/2026-10-03/r18/source/game/c256_qkv_direct_pack_one_v1.py#L30)；triton_kernel |
| 配置、shape、dtype、full K | 48/56×80/88 C256；head8，32channels/head；BM32 BN32 BK32，4warps；fullK256 |
| 运算/必要half边界 | FP16 DPAS目标；Q/K half norm+scale，Vhalf；ROUND_QKV=false |
| producer → consumer | C256-mlpproject → C256-attn |
| 读写、布局和临时 | head×Q/K/V grid独立重复读A；直接写3×head×windows×64×32 half，没有raster QKV中间 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:563 QKV_FUSE](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L563)（analogous_compute） |
| 差异/已实现 | 直接pack已开；AMD C64/128 QKV_FUSE共读A，C256结构不同 |
| 改造优先级、收益证据 | N6；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N6整个方案沿用[0, 0.25]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C256-pad — `torch.nn.functional.pad`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/multihead_block.py:89](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/multihead_block.py#L89)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 48/56×80/88 C256；shift0/4，每边对8齐 |
| 运算/必要half边界 | pure data，unround family边界返回原half |
| producer → consumer | 前块/gather → C256-pairs |
| 读写、布局和临时 | 写padded HWC；后续全部MLP会处理padding；AMD mapped input与pad-attention需分开 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 替换padding只省搬运不能自动删attention full64keys；编码tail仍需pool |
| 改造优先级、收益证据 | N6；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N6整个方案沿用[0, 0.25]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C256-attn — `window_block_attention_v3._kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/window_block_attention_v3.py:16](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/window_block_attention_v3.py#L16)；triton_kernel |
| 配置、shape、dtype、full K | 8heads；60/77/66/70 windows/head；BM32 4warps；full64 keys，ROUND_WEIGHTS=false |
| 运算/必要half边界 | FP16 QK/AV；half exp/denom/attended |
| producer → consumer | C256-qkv → C256-project |
| 读写、布局和临时 | 读QKV/bias；写packed AV[M,256]half；score不落地 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 此家族尚分attention/projection；AMD wave tail或persistent queue |
| 改造优先级、收益证据 | N6；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N6整个方案沿用[0, 0.25]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C256-project — `window_block_projection_v3._project`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/window_block_projection_v3.py:14](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/window_block_projection_v3.py#L14)；triton_kernel |
| 配置、shape、dtype、full K | 48×80输出；HP48/56 WP80/88；BM16 BN32 4warps；fullK256 |
| 运算/必要half边界 | FP16 dot→f32+half residual→half |
| producer → consumer | C256-attn → 后续块/pool/gather |
| 读写、布局和临时 | 读packed AV/MLP/W；每列tile8份输入请求，写有效HWC256；crop在地址中 |
| AMD 固定函数 | [hip/wave_owned_mh.inc:258 swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | packed projection/crop已开；队列/共享producer收益待测 |
| 改造优先级、收益证据 | N6；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N6整个方案沿用[0, 0.25]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### pool

#### P-pool — `slice views + top add.half + bottom add.half + sum.half + mul(.25).half + alignment pad`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/pre_block.py:45](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/pre_block.py#L45)；torch_data_or_ALU |
| 配置、shape、dtype、full K | unquantized 768×1280 C32→half spatial/2；C32下一投影 |
| 运算/必要half边界 | 标量half边界；不得从量化skip重新pool |
| producer → consumer | 对应tail → down projection / encoder next stage |
| 读写、布局和临时 | slice本身view；top/bottom/sum/scaled为数据输出，具体ATen合并未知；最终pool落地 |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | AMD finish或pool_project就地消费；现役尾部fusion没有吞掉这些ATen操作 |
| 改造优先级、收益证据 | gap-pool-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C32-pool — `slice views + top add.half + bottom add.half + sum.half + mul(.25).half + alignment pad`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/c32_block.py:46](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/c32_block.py#L46)；torch_data_or_ALU |
| 配置、shape、dtype、full K | unquantized 384×640 C32→half spatial/2；C64下一投影 |
| 运算/必要half边界 | 标量half边界；不得从量化skip重新pool |
| producer → consumer | 对应tail → down projection / encoder next stage |
| 读写、布局和临时 | slice本身view；top/bottom/sum/scaled为数据输出，具体ATen合并未知；最终pool落地 |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | AMD finish或pool_project就地消费；现役尾部fusion没有吞掉这些ATen操作 |
| 改造优先级、收益证据 | N3；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N3整个方案沿用[0, 2.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C32-pool-dot — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | M61440 K32 N64；BM16 BN32 BK32，4warps |
| 运算/必要half边界 | FP16 tl.dot→f32→half；完整K |
| producer → consumer | C32-pool → 下一编码层或ViT |
| 读写、布局和临时 | 读pooled half/W；写down HWC；pad对4齐均能是0但源调用仍列明 |
| AMD 固定函数 | [hip/multihead_fast.hip:200 mh_pool_project_production](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/multihead_fast.hip#L200)（analogous_compute） |
| 差异/已实现 | AMD pool+project融合；不是现役已有全模块融合 |
| 改造优先级、收益证据 | gap-pool-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C64-pool — `slice views + top add.half + bottom add.half + sum.half + mul(.25).half + alignment pad`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/multihead_block.py:102](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/multihead_block.py#L102)；torch_data_or_ALU |
| 配置、shape、dtype、full K | unquantized 192×320 C64→half spatial/2；C128下一投影 |
| 运算/必要half边界 | 标量half边界；不得从量化skip重新pool |
| producer → consumer | 对应tail → down projection / encoder next stage |
| 读写、布局和临时 | slice本身view；top/bottom/sum/scaled为数据输出，具体ATen合并未知；最终pool落地 |
| AMD 固定函数 | [hip/multihead_fast.hip:200 mh_pool_project_production](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/multihead_fast.hip#L200)（analogous_compute） |
| 差异/已实现 | AMD finish或pool_project就地消费；现役尾部fusion没有吞掉这些ATen操作 |
| 改造优先级、收益证据 | gap-pool-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C64-pool-dot — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | M15360 K64 N128；BM16 BN32 BK32，4warps |
| 运算/必要half边界 | FP16 tl.dot→f32→half；完整K |
| producer → consumer | C64-pool → 下一编码层或ViT |
| 读写、布局和临时 | 读pooled half/W；写down HWC；pad对4齐均能是0但源调用仍列明 |
| AMD 固定函数 | [hip/multihead_fast.hip:200 mh_pool_project_production](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/multihead_fast.hip#L200)（analogous_compute） |
| 差异/已实现 | AMD pool+project融合；不是现役已有全模块融合 |
| 改造优先级、收益证据 | gap-pool-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C128-pool — `slice views + top add.half + bottom add.half + sum.half + mul(.25).half + alignment pad`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/multihead_block.py:102](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/multihead_block.py#L102)；torch_data_or_ALU |
| 配置、shape、dtype、full K | unquantized 96×160 C128→half spatial/2；C256下一投影 |
| 运算/必要half边界 | 标量half边界；不得从量化skip重新pool |
| producer → consumer | 对应tail → down projection / encoder next stage |
| 读写、布局和临时 | slice本身view；top/bottom/sum/scaled为数据输出，具体ATen合并未知；最终pool落地 |
| AMD 固定函数 | [hip/multihead_fast.hip:200 mh_pool_project_production](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/multihead_fast.hip#L200)（analogous_compute） |
| 差异/已实现 | AMD finish或pool_project就地消费；现役尾部fusion没有吞掉这些ATen操作 |
| 改造优先级、收益证据 | gap-pool-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C128-pool-dot — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | M3840 K128 N256；BM16 BN32 BK32，4warps |
| 运算/必要half边界 | FP16 tl.dot→f32→half；完整K |
| producer → consumer | C128-pool → 下一编码层或ViT |
| 读写、布局和临时 | 读pooled half/W；写down HWC；pad对4齐均能是0但源调用仍列明 |
| AMD 固定函数 | [hip/multihead_fast.hip:200 mh_pool_project_production](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/multihead_fast.hip#L200)（analogous_compute） |
| 差异/已实现 | AMD pool+project融合；不是现役已有全模块融合 |
| 改造优先级、收益证据 | gap-pool-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C256-pool — `slice views + top add.half + bottom add.half + sum.half + mul(.25).half + alignment pad`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/multihead_block.py:102](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/multihead_block.py#L102)；torch_data_or_ALU |
| 配置、shape、dtype、full K | unquantized 48×80 C256→half spatial/2；C512下一投影 |
| 运算/必要half边界 | 标量half边界；不得从量化skip重新pool |
| producer → consumer | 对应tail → down projection / encoder next stage |
| 读写、布局和临时 | slice本身view；top/bottom/sum/scaled为数据输出，具体ATen合并未知；最终pool落地 |
| AMD 固定函数 | [hip/multihead_fast.hip:200 mh_pool_project_production](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/multihead_fast.hip#L200)（analogous_compute） |
| 差异/已实现 | AMD finish或pool_project就地消费；现役尾部fusion没有吞掉这些ATen操作 |
| 改造优先级、收益证据 | gap-pool-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C256-pool-dot — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | M960 K256 N512；BM16 BN32 BK32，4warps |
| 运算/必要half边界 | FP16 tl.dot→f32→half；完整K |
| producer → consumer | C256-pool → 下一编码层或ViT |
| 读写、布局和临时 | 读pooled half/W；写down HWC；pad对4齐均能是0但源调用仍列明 |
| AMD 固定函数 | [hip/multihead_fast.hip:200 mh_pool_project_production](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/multihead_fast.hip#L200)（analogous_compute） |
| 差异/已实现 | AMD pool+project融合；不是现役已有全模块融合 |
| 改造优先级、收益证据 | gap-pool-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C512-pool — `slice views + top add.half + bottom add.half + sum.half + mul(.25).half + alignment pad`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/fullsize_session_v1.py:111](../../../experiments/2026-10-03/r18/source/game/fullsize_session_v1.py#L111)；torch_data_or_ALU |
| 配置、shape、dtype、full K | unquantized 24×40 C512→half spatial/2；C1024下一投影 |
| 运算/必要half边界 | 标量half边界；不得从量化skip重新pool |
| producer → consumer | 对应tail → down projection / encoder next stage |
| 读写、布局和临时 | slice本身view；top/bottom/sum/scaled为数据输出，具体ATen合并未知；最终pool落地 |
| AMD 固定函数 | [hip/multihead_fast.hip:200 mh_pool_project_production](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/multihead_fast.hip#L200)（analogous_compute） |
| 差异/已实现 | AMD finish或pool_project就地消费；现役尾部fusion没有吞掉这些ATen操作 |
| 改造优先级、收益证据 | gap-pool-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C512-pool-dot — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | M240 K512 N1024；BM16 BN32 BK32，4warps |
| 运算/必要half边界 | FP16 tl.dot→f32→half；完整K |
| producer → consumer | C512-pool → 下一编码层或ViT |
| 读写、布局和临时 | 读pooled half/W；写down HWC；pad对4齐均能是0但源调用仍列明 |
| AMD 固定函数 | [hip/multihead_fast.hip:200 mh_pool_project_production](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/multihead_fast.hip#L200)（analogous_compute） |
| 差异/已实现 | AMD pool+project融合；不是现役已有全模块融合 |
| 改造优先级、收益证据 | gap-pool-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### C512

#### C512-entry — `c512_int8_ffn_rows_v1._entry`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c512_int8_ffn_rows_v1.py:32](../../../experiments/2026-10-03/r18/source/game/c512_int8_ffn_rows_v1.py#L32)；triton_kernel |
| 配置、shape、dtype、full K | M960，C512，一program/row，4warps |
| 运算/必要half边界 | 标量quantizer |
| producer → consumer | C512 block input → C512-linear |
| 读写、布局和临时 | Xhalf→QX i8[960,512] +SX f32[960]；DEBUG=false；OUT/ENTRY ROUND_FP8=false |
| AMD 固定函数 | [hip/c512_m32_deep.inc:151 FFN_ONE](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_m32_deep.inc#L151)（one_AMD_body_to_five_local_kernels） |
| 差异/已实现 | 逐行absmax/scale/quant已为完整INT8矩阵，非FP16模拟；AMD FP8 WMMA分组融合省QH中间 |
| 改造优先级、收益证据 | gap-c512-groups；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C512-linear — `c512_int8_ffn_rows_v1._linear`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c512_int8_ffn_rows_v1.py:46](../../../experiments/2026-10-03/r18/source/game/c512_int8_ffn_rows_v1.py#L46)；triton_kernel |
| 配置、shape、dtype、full K | M960 valid24×40；fullK512；BM32 BN64，4warps/1stage |
| 运算/必要half边界 | tl.dot i8*i8→i32，DPAS目标；f32 scale，half activation/residual边界 |
| producer → consumer | QX/SX → C512-expand |
| 读写、布局和临时 | QX+W[N,K] i8→QZ i8[960,512]；scale/half再量化；DEBUG=false；OUT/ENTRY ROUND_FP8=false |
| AMD 固定函数 | [hip/c512_m32_deep.inc:151 FFN_ONE](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_m32_deep.inc#L151)（one_AMD_body_to_five_local_kernels） |
| 差异/已实现 | INT8 linear已为完整INT8矩阵，非FP16模拟；AMD FP8 WMMA分组融合省QH中间 |
| 改造优先级、收益证据 | gap-c512-groups；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C512-expand — `c512_int8_ffn_rows_v1._expand`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c512_int8_ffn_rows_v1.py:66](../../../experiments/2026-10-03/r18/source/game/c512_int8_ffn_rows_v1.py#L66)；triton_kernel |
| 配置、shape、dtype、full K | M960 valid24×40；fullK64；BM32 BN64，4warps/1stage |
| 运算/必要half边界 | tl.dot i8*i8→i32，DPAS目标；f32 scale，half activation/residual边界 |
| producer → consumer | QZ → C512-reduce |
| 读写、布局和临时 | QZ→QH i8[960,2048]；8group×256hidden；native half cubic；DEBUG=false；OUT/ENTRY ROUND_FP8=false |
| AMD 固定函数 | [hip/c512_m32_deep.inc:151 FFN_ONE](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_m32_deep.inc#L151)（one_AMD_body_to_five_local_kernels） |
| 差异/已实现 | 8group expand+cubic已为完整INT8矩阵，非FP16模拟；AMD FP8 WMMA分组融合省QH中间 |
| 改造优先级、收益证据 | gap-c512-groups；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C512-reduce — `c512_int8_ffn_rows_v1._reduce`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c512_int8_ffn_rows_v1.py:86](../../../experiments/2026-10-03/r18/source/game/c512_int8_ffn_rows_v1.py#L86)；triton_kernel |
| 配置、shape、dtype、full K | M960 valid24×40；fullK256；BM32 BN64，4warps/1stage |
| 运算/必要half边界 | tl.dot i8*i8→i32，DPAS目标；f32 scale，half activation/residual边界 |
| producer → consumer | QH → C512-project |
| 读写、布局和临时 | QH→QG i8[960,512]；8group×64output；DEBUG=false；OUT/ENTRY ROUND_FP8=false |
| AMD 固定函数 | [hip/c512_m32_deep.inc:151 FFN_ONE](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_m32_deep.inc#L151)（one_AMD_body_to_five_local_kernels） |
| 差异/已实现 | 8group reduce已为完整INT8矩阵，非FP16模拟；AMD FP8 WMMA分组融合省QH中间 |
| 改造优先级、收益证据 | gap-c512-groups；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C512-project — `c512_int8_ffn_rows_v1._project`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c512_int8_ffn_rows_v1.py:134](../../../experiments/2026-10-03/r18/source/game/c512_int8_ffn_rows_v1.py#L134)；triton_kernel |
| 配置、shape、dtype、full K | M960 valid24×40；fullK512；BM32 BN64，4warps/1stage |
| 运算/必要half边界 | tl.dot i8*i8→i32，DPAS目标；f32 scale，half activation/residual边界 |
| producer → consumer | QG/X → C512-pad |
| 读写、布局和临时 | QG/W/skip→MLP half[960,512]；半精度initial再加；DEBUG=false；OUT/ENTRY ROUND_FP8=false |
| AMD 固定函数 | [hip/c512_m32_deep.inc:151 FFN_ONE](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_m32_deep.inc#L151)（one_AMD_body_to_five_local_kernels） |
| 差异/已实现 | INT8 final+residual已为完整INT8矩阵，非FP16模拟；AMD FP8 WMMA分组融合省QH中间 |
| 改造优先级、收益证据 | gap-c512-groups；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C512-pad — `torch.nn.functional.pad`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/fullsize_session_v1.py:111](../../../experiments/2026-10-03/r18/source/game/fullsize_session_v1.py#L111)；torch_data_or_ALU |
| 配置、shape、dtype、full K | MLP24×40×512→24×40/32×48/24×48/32×40 |
| 运算/必要half边界 | pure zero extension；full64key不变 |
| producer → consumer | C512-project → C512-QKV |
| 读写、布局和临时 | 有效960行变960/1536/1152/1280；16站共19712 padded rows |
| AMD 固定函数 | [hip/c512_qkv_attention_compact.inc:96 c5c_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_qkv_attention_compact.inc#L96)（one_AMD_body_to_GEMM_pack_attention） |
| 差异/已实现 | AMD validraster访问QKV；删无效QKV行需要attention保持补0/fullK |
| 改造优先级、收益证据 | N1；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N1整个方案沿用[0, 0.8]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C512-QKV — `torch.mm(features.reshape(-1,512), weight[512,1536])`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c512_qkv_library_16_v1.py:37](../../../experiments/2026-10-03/r18/source/game/c512_qkv_library_16_v1.py#L37)；library_GEMM |
| 配置、shape、dtype、full K | M960/1536/1152/1280 K512 N1536；f16，fullK；16sites |
| 运算/必要half边界 | XPU library GEMM；API f16→f16，内部accum/tile/ISA未知 |
| producer → consumer | C512-pad → C512-pack |
| 读写、布局和临时 | 读padded MLP/W，写z[M,16,3,32]half；约60,555,264B/整帧 QKV output |
| AMD 固定函数 | [hip/c512_qkv_attention_compact.inc:96 c5c_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_qkv_attention_compact.inc#L96)（one_AMD_body_to_GEMM_pack_attention） |
| 差异/已实现 | 库QKV拼3已经实现；AMD紧凑QKV+attention不落z |
| 改造优先级、收益证据 | N1；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_provider_calls_library_internals_unknown；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N1整个方案沿用[0, 0.8]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C512-pack — `fused_qkv_pack_native_half_v1._pack`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fused_qkv_pack_native_half_v1.py:17](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fused_qkv_pack_native_half_v1.py#L17)；triton_kernel |
| 配置、shape、dtype、full K | 16heads×15/24/18/20windows；BR16，4warps；fullhead32 |
| 运算/必要half边界 | half norm/scale；ROUND_QKV=false；pure vector math+pack非矩阵 |
| producer → consumer | C512-QKV → C512-attn |
| 读写、布局和临时 | 读raster z；写Q/K/V head/window/order packed half，总3×M×512×2B |
| AMD 固定函数 | [hip/c512_qkv_attention_compact.inc:96 c5c_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_qkv_attention_compact.inc#L96)（one_AMD_body_to_GEMM_pack_attention） |
| 差异/已实现 | 现役已norm+pack一核；AMD该边界与QKV/attention闭合 |
| 改造优先级、收益证据 | N1；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N1整个方案沿用[0, 0.8]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C512-attn — `c512_probability_unround_kernel_v1._kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/c512_probability_unround_kernel_v1.py:14](../../../experiments/2026-10-03/r18/source/game/c512_probability_unround_kernel_v1.py#L14)；triton_kernel |
| 配置、shape、dtype、full K | 16heads，15/24/18/20windows/head；BM32，4warps；full64 keys |
| 运算/必要half边界 | FP16 DPAS QK/AV→f32；half exp/denom/AV，ROUND_WEIGHTS=false |
| producer → consumer | C512-pack → C512-E-unpack / C512-D-proj |
| 读写、布局和临时 | score在寄存器；写packed attended[M,512]half；producer for encoder unpack/decoder packed proj |
| AMD 固定函数 | [hip/c512_qkv_attention_compact.inc:96 c5c_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_qkv_attention_compact.inc#L96)（one_AMD_body_to_GEMM_pack_attention） |
| 差异/已实现 | probability unround及full64key已在现役；AMDQKV与deep4ring就地feed |
| 改造优先级、收益证据 | N1；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

捕获provider门：`c512_probability_all_sixteen`=True。该门证明对应provider参与捕获；具体kernel为源码追溯，未声称library/ATen底层全部观察。

N1整个方案沿用[0, 0.8]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### C512-encoder

#### C512-E-unpack — `window_layout_v1._unpack`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/window_layout_v1.py:28](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/window_layout_v1.py#L28)；triton_kernel |
| 配置、shape、dtype、full K | valid24×40及3种padding，heads16；TOTAL491520/786432/589824/655360，BLOCK256 |
| 运算/必要half边界 | 纯数据/逆pixel_order |
| producer → consumer | C512-attn → C512-E-proj |
| 读写、布局和临时 | 读packed AV写padded raster HWC512→crop view，crop后contiguous可能复制 |
| AMD 固定函数 | [hip/c512_qkv_attention_compact.inc:96 c5c_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_qkv_attention_compact.inc#L96)（one_AMD_body_to_GEMM_pack_attention） |
| 差异/已实现 | encoder仍raster projection，decoder已直接packed消费；两路不能混同 |
| 改造优先级、收益证据 | N1；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N1整个方案沿用[0, 0.8]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C512-E-initial — `(q(residual)*skip_scale).half() / crop.contiguous in dot`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/split_block.py:37](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/split_block.py#L37)；torch_data_or_ALU |
| 配置、shape、dtype、full K | valid24×40×512 half |
| 运算/必要half边界 | 标量half残差；q在unround下identity |
| producer → consumer | C512-project / C512-E-unpack → C512-E-proj |
| 读写、布局和临时 | 写half initial983040B；shift crop noncontiguous可能读写attended983040B |
| AMD 固定函数 | [hip/c512_qkv_attention_compact.inc:96 c5c_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_qkv_attention_compact.inc#L96)（one_AMD_body_to_GEMM_pack_attention） |
| 差异/已实现 | AMD窗内保有残差并mapped projection |
| 改造优先级、收益证据 | N1；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N1整个方案沿用[0, 0.8]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### C512-E-proj — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | M960 K512 N512 initialized=true；BM16 BN32 BK32，4warps |
| 运算/必要half边界 | FP16 dot→f32+half initial→half；fullK512 |
| producer → consumer | C512-E-unpack / C512-E-initial → 下一block或C512-pool |
| 读写、布局和临时 | 读validraster AV/W/initial，写full[24,40,512]half |
| AMD 固定函数 | [hip/c512_qkv_attention_compact.inc:96 c5c_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_qkv_attention_compact.inc#L96)（one_AMD_body_to_GEMM_pack_attention） |
| 差异/已实现 | 编码投影与decoder packed consumer不同 |
| 改造优先级、收益证据 | N1；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N1整个方案沿用[0, 0.8]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### C512-decoder

#### C512-D-proj — `c512_window_projection_v1._project`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/c512_window_projection_v1.py:15](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/c512_window_projection_v1.py#L15)；triton_kernel |
| 配置、shape、dtype、full K | 24×40 output；HP24/32 WP40/48；BM16 BN32，4warps；fullK512 |
| 运算/必要half边界 | FP16 dot→f32+half residual→half |
| producer → consumer | C512-attn / C512-project → 下一decoder或D256 input |
| 读写、布局和临时 | 读packed attended及validMLP，写validHWC512；已经免unpack/crop临时 |
| AMD 固定函数 | [hip/c512_qkv_attention_compact.inc:96 c5c_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_qkv_attention_compact.inc#L96)（one_AMD_body_to_GEMM_pack_attention） |
| 差异/已实现 | decoder已有packed→raster投影；不能再声称同边界新增收益 |
| 改造优先级、收益证据 | N1；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联4个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N1整个方案沿用[0, 0.8]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### ViT-FFN

#### V-entry — `int8_ffn_segment_rows_v1._entry`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/int8_ffn_segment_rows_v1.py:22](../../../experiments/2026-10-03/r18/source/game/int8_ffn_segment_rows_v1.py#L22)；triton_kernel |
| 配置、shape、dtype、full K | 240×1024，一program/row，4warps/1stage；DEBUG=false |
| 运算/必要half边界 | absmax scale+标量INT8 quant |
| producer → consumer | ViT input → V-expand |
| 读写、布局和临时 | 写QX245760B/SX960B |
| AMD 固定函数 | [hip/deep_fast.hip:139 void vit_expand](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L139)（analogous_compute） |
| 差异/已实现 | 现役已INT8矩阵+行量化一次+完整tokens；r18 audit候选K64变化不属于此baseline |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### V-expand — `int8_ffn_segment_rows_v1._expand`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/int8_ffn_segment_rows_v1.py:31](../../../experiments/2026-10-03/r18/source/game/int8_ffn_segment_rows_v1.py#L31)；triton_kernel |
| 配置、shape、dtype、full K | M240 K1024 N4096，BM32 BN32，4warps/1stage；DEBUG=false |
| 运算/必要half边界 | i8*i8→i32 DPAS；f32 scale→half→native cubic→quant |
| producer → consumer | V-entry → V-contract |
| 读写、布局和临时 | 读QX/SX及Wout-major；写QH983040B |
| AMD 固定函数 | [hip/deep_fast.hip:139 void vit_expand](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L139)（analogous_compute） |
| 差异/已实现 | 现役已INT8矩阵+行量化一次+完整tokens；r18 audit候选K64变化不属于此baseline |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### V-contract — `int8_ffn_segment_rows_v1._contract`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/int8_ffn_segment_rows_v1.py:64](../../../experiments/2026-10-03/r18/source/game/int8_ffn_segment_rows_v1.py#L64)；triton_kernel |
| 配置、shape、dtype、full K | M240 fullK4096 N1024 PARTS4，BM32 BN32，4warps/1stage；DEBUG=false |
| 运算/必要half边界 | 4个完整K1024 i32 partition；总fullK4096 |
| producer → consumer | V-expand → V-merge |
| 读写、布局和临时 | 写partial[4,240,1024]i32 3,932,160B，无half partition |
| AMD 固定函数 | [hip/vit_stream.inc:191 vit_stream_contract](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/vit_stream.inc#L191)（analogous_compute） |
| 差异/已实现 | 现役已INT8矩阵+行量化一次+完整tokens；r18 audit候选K64变化不属于此baseline |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### V-merge — `int8_ffn_segment_rows_v1._merge`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/int8_ffn_segment_rows_v1.py:89](../../../experiments/2026-10-03/r18/source/game/int8_ffn_segment_rows_v1.py#L89)；triton_kernel |
| 配置、shape、dtype、full K | M240 N1024，B512，4warps/1stage；DEBUG=false |
| 运算/必要half边界 | 四i32相加→scale+half initial→half |
| producer → consumer | V-contract → V-qkv |
| 读写、布局和临时 | 读partial4份、X/scale/skip；写MLP491520B |
| AMD 固定函数 | [hip/vit_stream.inc:191 vit_stream_contract](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/vit_stream.inc#L191)（analogous_compute） |
| 差异/已实现 | 现役已INT8矩阵+行量化一次+完整tokens；r18 audit候选K64变化不属于此baseline |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### ViT-attention

#### V-qkv — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | 两次 M240 K512 N3072（总fullK1024），BM16 BN32 BK32，4warps，FP16 |
| 运算/必要half边界 | 每K512 GEMM f32累积→half；两half输出add→half另列 |
| producer → consumer | V-merge → V-qkv-add |
| 读写、布局和临时 | 两个MLP半列view→contiguous copy；两个QKVhalf临时各1,474,560B；写合并z同尺寸 |
| AMD 固定函数 | [hip/vit_stream.inc:152 vit_stream_qkv_frag_hin_w5f8](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/vit_stream.inc#L152)（analogous_compute） |
| 差异/已实现 | AMD全K融合QKV+norm发布head-major；保两K512仅用于结构收益隔离 |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### V-qkv-add — `(GEMM0+GEMM1).half().reshape(240,32,3,32)`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/fullsize_session_v1.py:131](../../../experiments/2026-10-03/r18/source/game/fullsize_session_v1.py#L131)；torch_data_or_ALU |
| 配置、shape、dtype、full K | z FP16[240,32,3,32]；fullK1024总和 |
| 运算/必要half边界 | 标量half add边界，reshape通常view |
| producer → consumer | V-qkv → V-norm-Q/K/V |
| 读写、布局和临时 | 读2份half，写z1,474,560B；不可把reshape算必有核 |
| AMD 固定函数 | [hip/vit_stream.inc:152 vit_stream_qkv_frag_hin_w5f8](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/vit_stream.inc#L152)（analogous_compute） |
| 差异/已实现 | AMD不需两个输出与add；fast版允许随后fullK单累积 |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### V-norm-Q — `nr_backend.triton_attention_normalize._normalize`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/triton_attention_normalize.py:21](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/triton_attention_normalize.py#L21)；triton_kernel |
| 配置、shape、dtype、full K | 7680 rows×32 f16，BR16/4warps |
| 运算/必要half边界 | native顺序half norm/FMA补偿 helper，最终half |
| producer → consumer | V-qkv-add → Qscale / V-QK |
| 读写、布局和临时 | 读z strided Q/K（wrapper contiguous）；写normalized491520B/种 |
| AMD 固定函数 | [hip/vit_stream.inc:152 vit_stream_qkv_frag_hin_w5f8](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/vit_stream.inc#L152)（analogous_compute） |
| 差异/已实现 | AMDQKV末端寄存器norm；现役独立norm，原生fast norm可后续 |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### V-norm-K — `nr_backend.triton_attention_normalize._normalize`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/triton_attention_normalize.py:21](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/triton_attention_normalize.py#L21)；triton_kernel |
| 配置、shape、dtype、full K | 7680 rows×32 f16，BR16/4warps |
| 运算/必要half边界 | native顺序half norm/FMA补偿 helper，最终half |
| producer → consumer | V-qkv-add → Qscale / V-QK |
| 读写、布局和临时 | 读z strided Q/K（wrapper contiguous）；写normalized491520B/种 |
| AMD 固定函数 | [hip/vit_stream.inc:152 vit_stream_qkv_frag_hin_w5f8](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/vit_stream.inc#L152)（analogous_compute） |
| 差异/已实现 | AMDQKV末端寄存器norm；现役独立norm，原生fast norm可后续 |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### V-Qscale — `(Qnorm*5.65625).half()*query_scale / half q / head transpose`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/fullsize_session_v1.py:131](../../../experiments/2026-10-03/r18/source/game/fullsize_session_v1.py#L131)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 240×32×32 half；scale32heads |
| 运算/必要half边界 | 两次half边界；transpose为view，值转换按dtype才有搬运 |
| producer → consumer | V-norm-Q/K → V-QK |
| 读写、布局和临时 | 读Qnorm/scale，写Q；Knorm/V切片及头布局是否拷贝见wrapper AST |
| AMD 固定函数 | [hip/vit_stream.inc:152 vit_stream_qkv_frag_hin_w5f8](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/vit_stream.inc#L152)（analogous_compute） |
| 差异/已实现 | head-major producer可直接供stream attention；不需新FP8舍入 |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### V-pad-KV — `pad(key/value, 240→256 keys)`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/vit_block.py:32](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/vit_block.py#L32)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 32heads×256×32 half，16zero keys |
| 运算/必要half边界 | 纯数据；完整240真实key+16pad |
| producer → consumer | V-norm-K / zV → V-QK / V-AV |
| 读写、布局和临时 | 读strided K/V写padded各524288B；pad贡献exp(0)用于旧denom补偿 |
| AMD 固定函数 | [hip/deep_fast.hip:363 vit_attention_fused_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L363)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 流式核可mask padding但保初始隔离half denom；随后可原生denom |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### V-QK — `strided_batched_v1._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/strided_batched_v1.py:9](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/strided_batched_v1.py#L9)；triton_kernel |
| 配置、shape、dtype、full K | batch32 M240 N256 fullK32，BM16 BN32 BK32 4warps |
| 运算/必要half边界 | FP16 DPAS dot→f32→half scores |
| producer → consumer | Qscale / V-pad-KV → V-exp |
| 读写、布局和临时 | 按Q/K stride读，不强制Qcopy；写scores[32,240,256]half3,932,160B |
| AMD 固定函数 | [hip/deep_fast.hip:363 vit_attention_fused_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L363)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | AMD按keytile就地消费score；strided输入优化已有 |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### V-exp — `nr_backend.triton_attention_exp._kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/triton_attention_exp.py:9](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/triton_attention_exp.py#L9)；triton_kernel |
| 配置、shape、dtype、full K | N1966080 half scores；VIT=true B512；另N1 exp(0)每block |
| 运算/必要half边界 | 标量特殊half exponent，非softmax通用exp |
| producer → consumer | V-QK → V-AV/V-denom |
| 读写、布局和临时 | 读scores，写e3,932,160B；N1scalar用于padding correction |
| AMD 固定函数 | [hip/deep_fast.hip:363 vit_attention_fused_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L363)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | AMD融合exp/AV/denom，原生exponent后续仍需质量验证 |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联2个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### V-AV — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | batch32 M240 N32 fullK256，BM16 BN32 BK32，4warps |
| 运算/必要half边界 | FP16 probability (unround e)×V→f32→half numerator |
| producer → consumer | V-exp/V-pad-KV → V-final-div |
| 读写、布局和临时 | 读e/padded V（contiguous wrapper可能copy），写[32,240,32]half491520B |
| AMD 固定函数 | [hip/deep_fast.hip:363 vit_attention_fused_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L363)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | AMDstream按64keytile累积分子，无e全局store |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### V-denom — `nr_backend.triton_attention_weights._weights`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/triton_attention_weights.py:25](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/triton_attention_weights.py#L25)；triton_kernel |
| 配置、shape、dtype、full K | ROWS7680 BR32，NORMALIZE=false；4个64keychunk/ViT block |
| 运算/必要half边界 | 有序half row_sum64；非FP32softmax sum |
| producer → consumer | V-exp → V-denom-merge |
| 读写、布局和临时 | wrapper将4个e[...,start:start+64]非连续view contiguous；每chunk读491520×2B，写7680half15360B |
| AMD 固定函数 | [hip/deep_fast.hip:363 vit_attention_fused_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L363)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 4chunk是4次调用；逻辑captured计数32不等同4×8物理全部eventtrace |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### V-denom-merge — `three add.half / exp(0)*padding.float().half / sub.half / clamp.float.reciprocal.half`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/vit_block.py:32](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/vit_block.py#L32)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 32×240×1；pad16 |
| 运算/必要half边界 | 原ordered half denominator+padding correction；fullK240 |
| producer → consumer | V-denom / V-exp N1 → V-final-div |
| 读写、布局和临时 | row sum小临时多次写读；N1 exp-zero也是实际原路径，非候选constant shortcut |
| AMD 固定函数 | [hip/deep_fast.hip:363 vit_attention_fused_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L363)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | AMD在线denominator；fullK新数学允许通过后替换 |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### V-final-div — `(numerator*reciprocal).half / transpose(0,1).reshape(240,1024) / (MLP*attn_skip).half`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/vit_block.py:32](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/vit_block.py#L32)；torch_data_or_ALU |
| 配置、shape、dtype、full K | attended240×1024 f16；skip initial同shape |
| 运算/必要half边界 | 乘half；transpose为view但reshape可能copy；half residual initial |
| producer → consumer | V-AV / V-denom-merge → V-proj-parts |
| 读写、布局和临时 | 写attended/initial各491520B；未由事件确认各copy的内部核数 |
| AMD 固定函数 | [hip/deep_fast.hip:363 vit_attention_fused_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L363)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | AMDhead-major尾部至projection；native fullK projection为数值候选，非必须永守P4 |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### ViT-projection

#### V-proj-parts — `fused_vit_projection_v2._parts`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fused_vit_projection_v2.py:15](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fused_vit_projection_v2.py#L15)；triton_kernel |
| 配置、shape、dtype、full K | M240 fullK1024 N1024 PARTS4，BM32 BN64（parts）；merge B512，4warps |
| 运算/必要half边界 | FP16 DPAS→f32，K256 partition各half；part0 initial，merge sequential half |
| producer → consumer | V-final-div → V-proj-merge |
| 读写、布局和临时 | 写partial[4,240,1024]half1,966,080B；part0含halfinitial |
| AMD 固定函数 | [hip/vit_stream.inc:191 vit_stream_contract](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/vit_stream.inc#L191)（analogous_compute） |
| 差异/已实现 | P4分区是一个grid带part维，不是4个launch；现役已两核projection |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。

#### V-proj-merge — `fused_vit_projection_v2._merge`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fused_vit_projection_v2.py:35](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fused_vit_projection_v2.py#L35)；triton_kernel |
| 配置、shape、dtype、full K | M240 fullK1024 N1024 PARTS4，BM32 BN64（parts）；merge B512，4warps |
| 运算/必要half边界 | 标量有序half add，非DPAS |
| producer → consumer | V-proj-parts → 下一ViT/decoder |
| 读写、布局和临时 | 读4个halfpartition，有序half归并，写output491520B |
| AMD 固定函数 | [hip/vit_stream.inc:191 vit_stream_contract](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/vit_stream.inc#L191)（analogous_compute） |
| 差异/已实现 | P4分区是一个grid带part维，不是4个launch；现役已两核projection |
| 改造优先级、收益证据 | N2；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N2整个方案沿用[0, 1.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### decoder

#### D512-zero — `torch.zeros([12,20,512])`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/decoder_gather_scope_v1.py:69](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/decoder_gather_scope_v1.py#L69)；torch_data_or_ALU |
| 配置、shape、dtype、full K | initial half[240,512] |
| 运算/必要half边界 | 填0纯数据/库allocation |
| producer → consumer | ViT output → D512-proj |
| 读写、布局和临时 | 写245760B；用于第一K256partition initial |
| AMD 固定函数 | [hip/deep_fast.hip:462 void decoder_project2x_h16w](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L462)（one_AMD_kernel_to_projection_and_gather） |
| 差异/已实现 | AMDprojection通常寄存器0初值 |
| 改造优先级、收益证据 | gap-decoder-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### D512-proj — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | 4次M240 K256 N512（总fullK1024）；第一initialized=true其他false；BM16 BN32 BK32，4warps |
| 运算/必要half边界 | 每K256独立FP32→half，随后3次有序half add；decoder模块持原split_k_projection别名 |
| producer → consumer | ViT output/D512-zero → D512-half-merge |
| 读写、布局和临时 | 写4个projected half[240,512]，3个merged half；ViT _parts/_merge不拥有decoder weight |
| AMD 固定函数 | [hip/deep_fast.hip:462 void decoder_project2x_h16w](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L462)（one_AMD_kernel_to_projection_and_gather） |
| 差异/已实现 | AMD decoder_project2x融合fullprojection+upmerge；不能把D512说成ViT两核projection |
| 改造优先级、收益证据 | gap-decoder-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联2个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### D512-half-merge — `3×(result+partition).half()`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/vit_block.py:12](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/vit_block.py#L12)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 240×512 half；fullK1024总和 |
| 运算/必要half边界 | 有序half，数值改变另验收 |
| producer → consumer | D512-proj → D512-gather |
| 读写、布局和临时 | 读previous +partition→half[240,512]，3次 |
| AMD 固定函数 | [hip/deep_fast.hip:462 void decoder_project2x_h16w](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L462)（one_AMD_kernel_to_projection_and_gather） |
| 差异/已实现 | fast版已存在decoder_input_full_k候选但accepted=false |
| 改造优先级、收益证据 | gap-decoder-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### decoder512

#### D512-gather — `decoder_gather_merge_v1._quantized`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/decoder_gather_merge_v1.py:25](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/decoder_gather_merge_v1.py#L25)；triton_kernel |
| 配置、shape、dtype、full K | target 24×40×512 half；B256/4warps；stridedprojected/skip；ROUND_OUTPUT=true |
| 运算/必要half边界 | nearest gather +补偿half FMA；C512/256/128/64保留现役merge E4M3 lattice，未追加新规则 |
| producer → consumer | D512-proj → C512-entry |
| 读写、布局和临时 | 读低projected按2×gather/skip/scale，写merged有效HWC512；没有repeat_interleave中间 |
| AMD 固定函数 | [hip/deep_fast.hip:462 void decoder_project2x_h16w](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L462)（one_AMD_kernel_to_projection_and_gather） |
| 差异/已实现 | all6 decoder_gather_unround去入口舍入，merge输出舍入另开关默认false；不得把未采用merge_unround当现役 |
| 改造优先级、收益证据 | gap-decoder-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### decoder256

#### D256-proj — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | M960 fullK512 N256；BM16 BN32 BK32，4warps |
| 运算/必要half边界 | FP16 DPAS→f32→half |
| producer → consumer | 前decoder family → D256-gather |
| 读写、布局和临时 | 读low half/W；写低分辨率projected[24,40,256]half |
| AMD 固定函数 | [hip/deep_fast.hip:462 void decoder_project2x_h16w](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L462)（one_AMD_kernel_to_projection_and_gather） |
| 差异/已实现 | projection仍独立；gather融合已经存在 |
| 改造优先级、收益证据 | gap-decoder-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### D256-gather — `decoder_gather_merge_v1._quantized`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/decoder_gather_merge_v1.py:25](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/decoder_gather_merge_v1.py#L25)；triton_kernel |
| 配置、shape、dtype、full K | target 48×80×256 half；B256/4warps；stridedprojected/skip；ROUND_OUTPUT=true |
| 运算/必要half边界 | nearest gather +补偿half FMA；C512/256/128/64保留现役merge E4M3 lattice，未追加新规则 |
| producer → consumer | D256-proj → C256-pad/pairs |
| 读写、布局和临时 | 读低projected按2×gather/skip/scale，写merged有效HWC256；没有repeat_interleave中间 |
| AMD 固定函数 | [hip/deep_fast.hip:462 void decoder_project2x_h16w](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L462)（one_AMD_kernel_to_projection_and_gather） |
| 差异/已实现 | all6 decoder_gather_unround去入口舍入，merge输出舍入另开关默认false；不得把未采用merge_unround当现役 |
| 改造优先级、收益证据 | gap-decoder-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### decoder128

#### D128-proj — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | M3840 fullK256 N128；BM16 BN32 BK32，4warps |
| 运算/必要half边界 | FP16 DPAS→f32→half |
| producer → consumer | 前decoder family → D128-gather |
| 读写、布局和临时 | 读low half/W；写低分辨率projected[48,80,128]half |
| AMD 固定函数 | [hip/deep_fast.hip:462 void decoder_project2x_h16w](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L462)（one_AMD_kernel_to_projection_and_gather） |
| 差异/已实现 | projection仍独立；gather融合已经存在 |
| 改造优先级、收益证据 | gap-decoder-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### D128-gather — `decoder_gather_merge_v1._quantized`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/decoder_gather_merge_v1.py:25](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/decoder_gather_merge_v1.py#L25)；triton_kernel |
| 配置、shape、dtype、full K | target 96×160×128 half；B256/4warps；stridedprojected/skip；ROUND_OUTPUT=true |
| 运算/必要half边界 | nearest gather +补偿half FMA；C512/256/128/64保留现役merge E4M3 lattice，未追加新规则 |
| producer → consumer | D128-proj → C128-pad/pairs |
| 读写、布局和临时 | 读低projected按2×gather/skip/scale，写merged有效HWC128；没有repeat_interleave中间 |
| AMD 固定函数 | [hip/deep_fast.hip:462 void decoder_project2x_h16w](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L462)（one_AMD_kernel_to_projection_and_gather） |
| 差异/已实现 | all6 decoder_gather_unround去入口舍入，merge输出舍入另开关默认false；不得把未采用merge_unround当现役 |
| 改造优先级、收益证据 | gap-decoder-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### decoder64

#### D64-proj — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | M15360 fullK128 N64；BM16 BN32 BK32，4warps |
| 运算/必要half边界 | FP16 DPAS→f32→half |
| producer → consumer | 前decoder family → D64-gather |
| 读写、布局和临时 | 读low half/W；写低分辨率projected[96,160,64]half |
| AMD 固定函数 | [hip/deep_fast.hip:462 void decoder_project2x_h16w](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L462)（one_AMD_kernel_to_projection_and_gather） |
| 差异/已实现 | projection仍独立；gather融合已经存在 |
| 改造优先级、收益证据 | gap-decoder-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### D64-gather — `decoder_gather_merge_v1._quantized`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/decoder_gather_merge_v1.py:25](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/decoder_gather_merge_v1.py#L25)；triton_kernel |
| 配置、shape、dtype、full K | target 192×320×64 half；B256/4warps；stridedprojected/skip；ROUND_OUTPUT=true |
| 运算/必要half边界 | nearest gather +补偿half FMA；C512/256/128/64保留现役merge E4M3 lattice，未追加新规则 |
| producer → consumer | D64-proj → C64-pad/pairs |
| 读写、布局和临时 | 读低projected按2×gather/skip/scale，写merged有效HWC64；没有repeat_interleave中间 |
| AMD 固定函数 | [hip/deep_fast.hip:462 void decoder_project2x_h16w](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L462)（one_AMD_kernel_to_projection_and_gather） |
| 差异/已实现 | all6 decoder_gather_unround去入口舍入，merge输出舍入另开关默认false；不得把未采用merge_unround当现役 |
| 改造优先级、收益证据 | gap-decoder-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### decoder32

#### D32-proj — `fast_matrices_v3._matmul`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/fast_matrices_v3.py:29](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/fast_matrices_v3.py#L29)；triton_kernel |
| 配置、shape、dtype、full K | M61440 fullK64 N32；BM16 BN32 BK32，4warps |
| 运算/必要half边界 | FP16 DPAS→f32→half |
| producer → consumer | 前decoder family → D32-gather |
| 读写、布局和临时 | 读low half/W；写低分辨率projected[192,320,32]half |
| AMD 固定函数 | [hip/deep_fast.hip:462 void decoder_project2x_h16w](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L462)（one_AMD_kernel_to_projection_and_gather） |
| 差异/已实现 | projection仍独立；gather融合已经存在 |
| 改造优先级、收益证据 | gap-decoder-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### D32-gather — `decoder_gather_merge_v1._raw_padded`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/decoder_gather_merge_v1.py:37](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/decoder_gather_merge_v1.py#L37)；triton_kernel |
| 配置、shape、dtype、full K | target 384×640×32 half；B256/4warps；stridedprojected/skip；rawC32 |
| 运算/必要half边界 | nearest gather+补偿half FMA，保留rawC32 residual |
| producer → consumer | D32-proj → D32-mlp |
| 读写、布局和临时 | 读低projected按2×gather/skip/scale，写merged有效HWC32；没有repeat_interleave中间 |
| AMD 固定函数 | [hip/deep_fast.hip:462 void decoder_project2x_h16w](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L462)（one_AMD_kernel_to_projection_and_gather） |
| 差异/已实现 | all6 decoder_gather_unround去入口舍入，merge输出舍入另开关默认false；不得把未采用merge_unround当现役 |
| 改造优先级、收益证据 | gap-decoder-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### D32-mlp — `native_cubic_c32_v1._kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/native_cubic_c32_v1.py:24](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/native_cubic_c32_v1.py#L24)；triton_kernel |
| 配置、shape、dtype、full K | M245760 C32→128→32，BM32/4warps，unrounded hidden |
| 运算/必要half边界 | FP16 dot+cubic/residual half |
| producer → consumer | D32-gather → D32-qkv |
| 读写、布局和临时 | merged rawC32读→MLP half写；与ordinary7入口不同 |
| AMD 固定函数 | [hip/wave_owned_c32.inc:206 cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 已包含S32-mlp同callable；单列block66链防遗漏 |
| 改造优先级、收益证据 | N3；unmeasured estimate inherited from original PLAN; shared by all rows of module, not additive per row |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联6个绑定（不自动当执行）；无逐物理dispatch轨迹 |

N3整个方案沿用[0, 2.5]ms待证伪预算，不能逐行相加。修改入口即上方现役文件；最小Luna实验是只换此完整模块，在相同13帧上配平A/B/B/A，保留full K及真实时序控制。


### post

#### O-entry — `post_entry_mlp_fusion_v1._entry_kernel`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/post_entry_mlp_fusion_v1.py:70](../../../experiments/2026-10-03/r18/source/game/post_entry_mlp_fusion_v1.py#L70)；triton_kernel |
| 配置、shape、dtype、full K | low384×640×32，skip768×1280×32→padded776×1288×32；BM32/4warps，ROUND_INPUT=false |
| 运算/必要half边界 | 标量low scale half +skip补偿halfFMA，zero extension |
| producer → consumer | decoder32 output / pre_skip → O-mlp |
| 读写、布局和临时 | 写raw merged63,967,232B；nearest无需expanded中间 |
| AMD 固定函数 | [hip/wave_owned_c32.inc:537 c32_wave1_post](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L537)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | 已融合up/merge/pad；AMDpost还含MLP/QKV/tail/head；FDP candidate不是本baseline |
| 改造优先级、收益证据 | gap-post-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

捕获provider门：`post_entry`=True。该门证明对应provider参与捕获；具体kernel为源码追溯，未声称library/ATen底层全部观察。

#### O-head — `native_k8_active_720_v1._post_half_dot`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/native_k8_active_720_v1.py:20](../../../experiments/2026-10-03/r18/source/game/native_k8_active_720_v1.py#L20)；triton_kernel |
| 配置、shape、dtype、full K | M921600，W[32,8]，只写4channels RGB+logit；BM32/4warps；分K8四段累加实现见源码 |
| 运算/必要half边界 | 原生half dot目标；half累加/最终half，非模拟4060指令 |
| producer → consumer | O-tail → O-base / O-sigmoid |
| 读写、布局和临时 | 读HWC32与headW，写720×1280×4 half7,372,800B |
| AMD 固定函数 | [hip/wave_owned_c32.inc:537 c32_wave1_post](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L537)（one_AMD_body_to_many_local_ops） |
| 差异/已实现 | native post K8已开；AMD头与post合一但只3RGB，不能删logit历史blend |
| 改造优先级、收益证据 | gap-post-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | capture_route_static_kernel_resolution；关联1个绑定（不自动当执行）；无逐物理dispatch轨迹 |

捕获provider门：`native_k8_pre_post`=True。该门证明对应provider参与捕获；具体kernel为源码追溯，未声称library/ATen底层全部观察。

#### O-base — `head[:3].float / rgb.half.float / scalar gain-center / clamp`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/post.py:58](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/post.py#L58)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 720×1280×3 f32；gain1/32及baseRGB |
| 运算/必要half边界 | 标量FP32，纹理RGB半精度边界 |
| producer → consumer | O-head / RGB → O-blend/store |
| 读写、布局和临时 | 读head/RGB，多次临时value；reset可clamp输出，history分支还blend |
| AMD 固定函数 | [hip/boundary_fast.hip:13 hip_post_head_fast_half](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/boundary_fast.hip#L13)（analogous_compute） |
| 差异/已实现 | AMDhead做RGB就地epilogue；FDP的1.80ms整段证据只可作为已有候选证据，不分摊本行 |
| 改造优先级、收益证据 | gap-post-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### O-sigmoid — `logit.half.contiguous.view(i16).int &65535 / table[bits.long()]`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/sigmoid.py:26](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/sigmoid.py#L26)；torch_data_or_ALU |
| 配置、shape、dtype、full K | H×W logit，65536 f32 table；alpha H×W f32 |
| 运算/必要half边界 | 标量索引/查表，非DPAS |
| producer → consumer | O-head logit → O-blend |
| 读写、布局和临时 | logit contiguous、i32/i64 bits数组、table gather及scale/clamp；内部ATen核数未知 |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | AMDregular reset缺logit时序等价；native sigmoid为现有未选数值候选 |
| 改造优先级、收益证据 | gap-post-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### O-blend — `fma32(previous,reciprocal,-value) / fma32(delta,alpha,value)`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/post.py:58](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/post.py#L58)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 720×1280×3 f32 numerator +H×W rcp；真实历史 |
| 运算/必要half边界 | 补偿FP32 FMA；完整motion/history，不从上一输出替代当前网络 |
| producer → consumer | H-sample/O-base/O-sigmoid → O-store |
| 读写、布局和临时 | 读numerator/rcp/alpha/value；delta/value临时f32；不得只用normalized重新round替换 |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | AMDregular无同配方；不可照搬只RGB epilogue |
| 改造优先级、收益证据 | gap-post-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### O-store — `float.half / bits - overshoot correction / bitcast half`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/post.py:17](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/post.py#L17)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 720×1280×3→f16；reset先unitclamp，history保overshoot |
| 运算/必要half边界 |  toward-zero half store，数值accepted reference |
| producer → consumer | O-base或O-blend → graph output/history |
| 读写、布局和临时 | 多个half/f32/bit整数临时；视图不是物理kernel；FDP/native store候选尚未成为accepted |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | 原生rtz store可另验收；不得等同4060逐位刚需 |
| 改造优先级、收益证据 | gap-post-fusion；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### graph

#### G-input — `entry.inputs[name].copy_(t) / graph.replay`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/graph_front_v1.py:62](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/graph_front_v1.py#L62)；torch_data_or_ALU |
| 配置、shape、dtype、full K | reset input42,516,480B；history input57,262,080B（metadata记录） |
| 运算/必要half边界 | 纯copy+graph submission，非DPAS |
| producer → consumer | front/history → capture_body mainchain |
| 读写、布局和临时 | rgb/front/previous/rcp各copy到静态输入；图重放已开 |
| AMD 固定函数 | [Development/HIP/hip_d3d12_bridge.h:215 ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)（interop_protocol_analogy_only） |
| 差异/已实现 | AMDgraph/reuse不等于本地尚未有graph；静态目的地可避免部分copy但新hooks全false |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### G-output-pool — `output.copy_(temporary_result) inside captured graph`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/graph_front_v5.py:18](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/graph_front_v5.py#L18)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 720×1280×3 half5,529,600B |
| 运算/必要half边界 | 纯copy，图输出pool-external存活 |
| producer → consumer | O-store → G-output-public |
| 读写、布局和临时 | 临时body output→每图独立输出一读一写 |
| AMD 固定函数 | [Development/HIP/hip_d3d12_bridge.h:215 ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)（interop_protocol_analogy_only） |
| 差异/已实现 | 共享pool生命周期已处理；删copy须证明输出ownership |
| 改造优先级、收益证据 | gap-copy；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### G-output-public — `entry.output.clone()`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/graph_front_v1.py:62](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/graph_front_v1.py#L62)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 720×1280×3 half5,529,600B |
| 运算/必要half边界 | copy，不是view |
| producer → consumer | G-output-pool → H-commit / geometry composite |
| 读写、布局和临时 | 图持有output→caller-owned clone；避免下一replay覆写 |
| AMD 固定函数 | [Development/HIP/hip_d3d12_bridge.h:215 ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)（interop_protocol_analogy_only） |
| 差异/已实现 | AMDpool内复用与caller生命周期不同，不能无证明返回旧buffer |
| 改造优先级、收益证据 | gap-copy；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### history

#### H-commit — `result.detach().clone(); seed=(seed+1)&MASK32`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/temporal.py:102](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/temporal.py#L102)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 720×1280×3 half5,529,600B；13frame seed0…12 |
| 运算/必要half边界 | clone纯copy；detach metadata；seedCPU |
| producer → consumer | G-output-public → 下一frame H-route / axes |
| 读写、布局和临时 | 再复制private previous，真实raw历史；没有host下载 |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | caller-output与history所有权不同；可比较双buffer，但不是采上帧少算 |
| 改造优先级、收益证据 | gap-copy；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### geometry

#### E-composite — `processed.float().clamp(0,1)`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/nr_game_fullsize.py:124](../../../experiments/2026-10-03/r18/source/game/nr_game_fullsize.py#L124)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 固定source720×1280=inputmodel，输出f32 HWC3 |
| 运算/必要half边界 | convert +clamp标量 |
| producer → consumer | G-output-public → bridge.export |
| 读写、布局和临时 | 同尺寸不Lanczos、不residual-upscale；half→float11,059,200B |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | 固定720无降低画布收益；其他尺寸分支另列 |
| 改造优先级、收益证据 | gap-copy；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### E-prepare — `color/motion.float().contiguous() identity path`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/nr_game_fullsize.py:94](../../../experiments/2026-10-03/r18/source/game/nr_game_fullsize.py#L94)；torch_data_or_ALU |
| 配置、shape、dtype、full K | source720×1280 HWC3/HWC2 f32，active=model720无inset |
| 运算/必要half边界 | metadata/no-op当输入dtype与layout满足 |
| producer → consumer | bridge.prepare → MotionNR/front |
| 读写、布局和临时 | 不会宣称每个float/contiguous必起核；同shape直接传canvas/flow |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | 该path非resizer，不能据源码_axis即算GPU resizer已跑 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### control

#### C-lanes — `for lane10…14: front[...,lane]=control_lane`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/nr_game_controlled_model.py:96](../../../experiments/2026-10-03/r18/source/game/nr_game_controlled_model.py#L96)；torch_data_or_ALU |
| 配置、shape、dtype、full K | front padded768×1280×16 half；controls style0/intensity1/local1 |
| 运算/必要half边界 | 标量fill；若native_controls_present有receipt则省写 |
| producer → consumer | F-front +NRControls → G-input |
| 读写、布局和临时 | 默认source能走五个lane fill；front_native kernel已写控制但producer receipt能力须看实际hook，不能假设全部消失 |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | 保留控制；具体fill物理执行无receipt，列静态上界 |
| 改造优先级、收益证据 | gap-control；new gap unbudgeted; no measured gain assigned |
| 覆盖证据/未知 | static_path_no_individual_dispatch_receipt；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C-default — `default controls→MotionNR.forward`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [game/nr_game_controlled_model.py:30](../../../experiments/2026-10-03/r18/source/game/nr_game_controlled_model.py#L30)；host_control |
| 配置、shape、dtype、full K | style0/intensity1/local1/auto_maskfalse，fixed13 |
| 运算/必要half边界 | CPU分支/metadata；默认绕过style post |
| producer → consumer | NRControls → H-route/front/body |
| 读写、布局和临时 | 不把loaded NativeStylePost权重当style shader已执行 |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | 当前13帧仅默认控制；风格/lowstrength/aux全部未观测 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | static_main_chain；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### C-nondefault — `Controlled/local/style/intensity/auto-mask/skin auxiliary routes`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/nr_backend/live_temporal.py:31](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/nr_backend/live_temporal.py#L31)；torch_data_or_ALU |
| 配置、shape、dtype、full K | 720×1280；控制可变，强度<1 rawFP32图或eager |
| 运算/必要half边界 | 条件标量/表/采样/FP32mix，完整帧 |
| producer → consumer | NRControls → controlled front/post/history |
| 读写、布局和临时 | 非默认需clamp/skin/luma/style LUT及私人rawhistory；附录给每处AST调用点；不假定库物理核已审 |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | fixed13 style1/2未执行，lowstrength须adapter harness；不用新GPU求100% |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | conditional_unobserved；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### geometry

#### E-resize — `residual_scale_v1._axis`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [experimental/fp8_unround_overlay/modules/residual_scale_v1.py:56](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/residual_scale_v1.py#L56)；conditional_triton_kernel |
| 配置、shape、dtype、full K | 仅source≠model：Lanczos2 RGB、area motion×比例、Catmull residual；BLOCK256 |
| 运算/必要half边界 | 标量filter，无DPAS；shape动态 |
| producer → consumer | source/NR residual → model/bridge |
| 读写、布局和临时 | 每轴读写完整tensor；本次1280×720→1280×720没有此dispatch |
| AMD 固定函数 | 无现役同配方AMD对应，明确未映射 |
| 差异/已实现 | 不得把输入下降/减少画布列为内核收益 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | conditional_unobserved；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |


### model-external bridge

#### B-HDR-prepare — `prepare [numthreads(8,8,1)]`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [shaders/nr_hdr_proxy.hlsl:16](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/baseline/native/shaders/nr_hdr_proxy.hlsl:16)；native_GPU_command_or_shader |
| 配置、shape、dtype、full K | R11G11B10 color / RG16F motion，1280×720 |
| 运算/必要half边界 | D3D12 shader标量 / SYCL memcpy / GPU fence；无矩阵K |
| producer → consumer | 本帧游戏color/motion producer → RE8 pack |
| 读写、布局和临时 | 写RGBA16/32 SDR代理及RG16 pixelmotion；motion scales/finite/clamp保留 |
| AMD 固定函数 | [Development/HIP/hip_d3d12_bridge.h:215 ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)（interop_protocol_analogy_only） |
| 差异/已实现 | 纹理复用/真实motion转换已存在，r18 audit native fastpack不计现役 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | accepted_bridge_source_only_not_offline_model_capture；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### B-HDR-composite — `composite [8,8,1]`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [shaders/nr_hdr_proxy.hlsl:34](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/baseline/native/shaders/nr_hdr_proxy.hlsl:34)；native_GPU_command_or_shader |
| 配置、shape、dtype、full K | 720×1280 scene/NR→R11G11B10 XeSSInput |
| 运算/必要half边界 | D3D12 shader标量 / SYCL memcpy / GPU fence；无矩阵K |
| producer → consumer | NR完成输出+原scene → XeSS input consumer |
| 读写、布局和临时 | scene+highlights weight×(NR−SDR)，保原scene high values；逐pixel读写 |
| AMD 固定函数 | [Development/HIP/hip_d3d12_bridge.h:215 ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)（interop_protocol_analogy_only） |
| 差异/已实现 | AMDregular pre-tonemap/HDR配方不同；不能删highlights保护 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | accepted_bridge_source_only_not_offline_model_capture；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### B-pack — `embedded HLSL pack [8,8,1]`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [nr_texture_bridge_re8_v1.cpp:50](../../../current/native/nr_texture_bridge_re8_v1.cpp#L50)；native_GPU_command_or_shader |
| 配置、shape、dtype、full K | textures→USM-shared staging HWC3/HWC2 f32 |
| 运算/必要half边界 | D3D12 shader标量 / SYCL memcpy / GPU fence；无矩阵K |
| producer → consumer | HDR prepare+producer fence → B-copy-in |
| 读写、布局和临时 | 各pixel颜色/运动→linear RGB/MV；中间stagebuffer |
| AMD 固定函数 | [Development/HIP/hip_d3d12_bridge.h:215 ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)（interop_protocol_analogy_only） |
| 差异/已实现 | 已验收pack；不是r18 new audit rawHDR候选 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | accepted_bridge_source_only_not_offline_model_capture；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### B-unpack — `embedded HLSL unpack [8,8,1]`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [nr_texture_bridge_re8_v1.cpp:58](../../../current/native/nr_texture_bridge_re8_v1.cpp#L58)；native_GPU_command_or_shader |
| 配置、shape、dtype、full K | linear result f32 HWC3→RGBA texture |
| 运算/必要half边界 | D3D12 shader标量 / SYCL memcpy / GPU fence；无矩阵K |
| producer → consumer | B-copy-out / XPU completion → HDR composite |
| 读写、布局和临时 | 读linear result写output texture |
| AMD 固定函数 | [Development/HIP/hip_d3d12_bridge.h:215 ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)（interop_protocol_analogy_only） |
| 差异/已实现 | 底层shader ISA未知，只有accepted source |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | accepted_bridge_source_only_not_offline_model_capture；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### B-copy-in — `prepare_handoff xpu->memcpy(color_ptr,rgb.ptr) + memcpy(motion_ptr,mv.ptr)`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [nr_texture_bridge_re8_v1.cpp:365](../../../current/native/nr_texture_bridge_re8_v1.cpp#L365)；native_GPU_command_or_shader |
| 配置、shape、dtype、full K | 720×1280×(3+2)f32=18,432,000B |
| 运算/必要half边界 | D3D12 shader标量 / SYCL memcpy / GPU fence；无矩阵K |
| producer → consumer | B-pack → E-prepare |
| 读写、布局和临时 | 共享staging到模型USM，两copies依赖forward import event |
| AMD 固定函数 | [Development/HIP/hip_d3d12_bridge.h:215 ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)（interop_protocol_analogy_only） |
| 差异/已实现 | GPU栅栏接力已开，不能计作新host同步消除收益 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | accepted_bridge_source_only_not_offline_model_capture；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### B-copy-out — `export_handoff xpu->memcpy(result.ptr,color)`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [nr_texture_bridge_re8_v1.cpp:400](../../../current/native/nr_texture_bridge_re8_v1.cpp#L400)；native_GPU_command_or_shader |
| 配置、shape、dtype、full K | 720×1280×3 f32=11,059,200B |
| 运算/必要half边界 | D3D12 shader标量 / SYCL memcpy / GPU fence；无矩阵K |
| producer → consumer | E-composite → B-unpack |
| 读写、布局和临时 | 模型USM→shared result，backward completion signal后D3D12 unpack |
| AMD 固定函数 | [Development/HIP/hip_d3d12_bridge.h:215 ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)（interop_protocol_analogy_only） |
| 差异/已实现 | 异步export存活到consumer retirement，不能无证明消除copy |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | accepted_bridge_source_only_not_offline_model_capture；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### B-fences — `queue Wait producer →pack Signal →XPU external semaphore / backward Wait →unpack Signal`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [nr_texture_bridge_re8_v1.cpp:365](../../../current/native/nr_texture_bridge_re8_v1.cpp#L365)；native_GPU_command_or_shader |
| 配置、shape、dtype、full K | 每帧fence/value和retained imports |
| 运算/必要half边界 | D3D12 shader标量 / SYCL memcpy / GPU fence；无矩阵K |
| producer → consumer | 游戏本帧producer → XPU model / game consumer |
| 读写、布局和临时 | GPU queue依赖+ownership lease，serial model owner |
| AMD 固定函数 | [Development/HIP/hip_d3d12_bridge.h:215 ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)（interop_protocol_analogy_only） |
| 差异/已实现 | 与AMD hip_d3d12_bridge接口对应；现役已有relay，不用CPU_wait旧版作主对照 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | accepted_bridge_source_only_not_offline_model_capture；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### B-retire — `consumer fence completion / lease retirement / command allocator reuse`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [nr_texture_bridge_re8_v1.cpp:260](../../../current/native/nr_texture_bridge_re8_v1.cpp#L260)；native_GPU_command_or_shader |
| 配置、shape、dtype、full K | 上一borrowed output退休 |
| 运算/必要half边界 | D3D12 shader标量 / SYCL memcpy / GPU fence；无矩阵K |
| producer → consumer | XeSS consumer → next prepare reuse |
| 读写、布局和临时 | 保texture/stagebuffer/command pool至真实consumer signal |
| AMD 固定函数 | [Development/HIP/hip_d3d12_bridge.h:215 ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)（interop_protocol_analogy_only） |
| 差异/已实现 | 不在offline capture；不能把source存在等同same-frame Present计时 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | accepted_bridge_source_only_not_offline_model_capture；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

#### B-pool — `HdrProxy::bind/record_prepare/record_composite +resource pool`

| 项目 | 核查结果 |
|---|---|
| 现役入口 | [src/nr_hdr_proxy.cpp:480](${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/baseline/native/src/nr_hdr_proxy.cpp:480)；native_GPU_command_or_shader |
| 配置、shape、dtype、full K | width/height=format-scoped resources/PSO/root/descriptors |
| 运算/必要half边界 | D3D12 shader标量 / SYCL memcpy / GPU fence；无矩阵K |
| producer → consumer | game resources → HDR shaders |
| 读写、布局和临时 | Dispatch ceil(W/8)×ceil(H/8)，Barrier/ResourceState；cache复用 |
| AMD 固定函数 | [Development/HIP/hip_d3d12_bridge.h:215 ](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_d3d12_bridge.h#L215)（interop_protocol_analogy_only） |
| 差异/已实现 | texture/resource复用已验收；不与r18 native protocol候选混算 |
| 改造优先级、收益证据 | already；already implemented or no current new gain |
| 覆盖证据/未知 | accepted_bridge_source_only_not_offline_model_capture；关联0个绑定（不自动当执行）；无逐物理dispatch轨迹 |

## 一对多结构、实施接口与新缺口

| AMD闭合单元 | 本地对应角色集合 | 实施边界及真实减少量 |
|---|---|
| c5c_body 紧凑QKV→attention | C512-pad/QKV/pack/attn，以及encoder/decoder不同projection | N1先保持库GEMM在valid raster；19712→15360 QKV行，少4352行、3,422,552,064 MAC，约22.08% QKV算量；attention仍每窗64 keys。packed consumer保持fullkeys和bias/order。 |
| vit_stream_qkv + fused attention | V-qkv/add/norm/Qscale/pad/QK/exp/AV/denom/final-div | N2 producer改head-major `[head,tile,64,32]`，consumer64key tile DPAS；初期保两个K512 half边界，score与exp各3,932,160B/块免落地，8块仅这两项写读少125,829,120B；padding和full240不削。该字节数是逻辑global张量容量，不是DRAM实测。 |
| c32 wave整窗 cw_body | S32-pad/mlp/qkv/tail，block66还unpack/projection | N3一个window由工作组拥有，MLP/QKV/attention/tail依次局部消费；预留half residual/完整keys，滚动hidden16/32tile控SLM/GRF，先只ordinary7，再独立block66资格。现役tail已有fusion，只把MLP/QKV额外边界作新增。 |
| mh QKV_FUSE | C64/C128-qkv | N4每head同tile共读A→Q/K/V三accumulator；输入请求3→1，布局仍packed，不删norm/keys。DPAS映射由Intel编译IR验证，不能照搬AMD warp寄存器排列。 |
| mh QT4 FFN | C64/C128/C256-pairs/mlpproject | N5四token tile同组SLM共享W，逻辑W请求4→1；C128pairwise和FP32branch已开，改的是跨M复用。SLM/barrier与占用可能抵消收益。 |
| sp_run_body C256依赖队列 | C256-pad/pairs/project/qkv/attn/project | N6完整多块逐window就绪依赖，图已重放；block-owned数据ready后消费，acquire/release fence及border窗口处理必须明确。不是单纯删Python launch。 |
| decoder_project2x / wave up-body / post whole block | 五transition projection+gather、pool、post entry/body/head/temporal/store | 新缺口不并入N1–N6预算：投影与nearest/skip融合、pool就地project、post完整真历史epilogue。FDP已重复1.80ms是另候选完整帧结果，未装游戏；不能拆给各行或加到六方案。 |

后续fast数学候选：现有decoder_input_full_k、vit_qkv_full_k、vit_projection_full_k、vit_norm_fma、vit_exp_fma、vit_denominator与post_sigmoid/store等已在numeric_cleanup中有实现/选择器，但本accepted未收；本次不重新开发、不重新追加舍入。跨完整K的一次FP32/i32累积仍是full K，允许按误差/视觉/可重复完整帧收益验收。

## 绑定备选、排除项与盲区

139个cache key只证明编译/绑定：例如C128旧project、history debug、front components/high-seed等可以被预编译却没在固定13帧执行。JSON把未关联项逐条保留；loaded_modules不作为运行门。当前audit_history_host全false只作准入；所有audit_impl新核均在excluded_not_current列出。

源代码的零spill选择门仍在：rows_scopes.launch调用spill_preflight.select，C32 native preflight调用筛选。它约束可选tile，却不证明保留tile最快，也不把spill等同慢根因。逐位/ordered-half来自原accepted实现；快速新方案允许数值变化，不是4060永久刚需。本补审没有新单核计时或删门收益可宣称。

模型外桥以PHASE1指向RUNTIME_SCOPE和已验收canonical收据为来源，shared native C++ SHA匹配e45f5af…；离线RESULT没有game/API调用，不能拿它证明桥派发/渲染重叠/Present。ATen/torch.mm内部库kernel和ISA、逐次copy的kernel合并/调度、硬件内存流量仍未知。

近整数motion、非默认style/lowstrength/auto-mask/skin、其他source尺寸只完成静态条件库存。附录JSON `static_wrapper_and_conditional_calls`给逐调用点/表达式和行号，但它包括未选分支；不能伪装动态执行清单。当前主链所有静态role均列入，上述边界明确保留，无另跑GPU。

## 复算

`extract_inventory.py`只用stdlib AST/JSON/hash，读取固定r18/RESULT/AMD及accepted桥收据，验证SHA与constructor，重建三份输出。用bundled Python加`-B`运行，禁止导入项目torch/Triton；所有输出限定本目录。手写结构映射是review annotation，AST不能自行推导monkey patch的最终选择。

验证：261个r18 source pins匹配，manifest/RESULT SHA正确，r15/r18 constructor同，AMD HEAD固定，所有函数/行链接解析通过。物理event覆盖数0，库内部已核数0；这两项明确留空。

