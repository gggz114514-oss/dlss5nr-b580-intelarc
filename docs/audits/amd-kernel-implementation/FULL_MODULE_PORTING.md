# AMD 成功实现 → accepted720：N1–N6 完整模块移植配方

2026-10-03 收尾版；仅 CPU 静态研究。AMD 固定 SHA **9ec741522d267c4d2365af53081716fb2943068a**。作者归档的1080档网络约 **9.5～10ms**，10-02整网回放汇总约9.84ms，是有日志支持的成绩。值得移植的不仅是融合边界，还包括线程所有权、consumer 所需的 fragment 方向、权重装载复用、短生命周期和明确的软件流水。网络回放不等于完整游戏 Present 90fps；配方与计时口径沿用既有 REPORT，不在此重做演进史。[归档汇总][amd-log]

唯一主对照为 **accepted720 C512+K8/all6**：1280×720输入、1280×768内部画布、ViT240 tokens，真实 motion/history/control/full K和13帧。游戏现役用户RTSS55–60ms。r18 manifest SHA为 **699ee05cf1008610a4772dfeebe6398d7099c7c1ed141cbbaa91967da0a05f7f**；r15/r18 accepted constructor已逐字段相同。本文本地引用均落到r18冻结树中被 accepted 安装链选中的函数；其 audit_impl 候选排除。119行逐核库存及覆盖边界见同目录 KERNEL_DIFF、KERNEL_INVENTORY、COVERAGE。

文中的“已证”指已读固定源码或已有归档，“配方”指据此设计的 Intel 实现步骤，“待测”指本次没有GPU证据。后续快速版允许数值变化；保留旧half边界只用于先隔离组织收益。没有4060逐位刚需，不重新加入FP8模拟舍入。原N1–N6预算保持不变：分别 **0～0.8、0～1.5、0～2.5、0～0.6、0～0.6、0～0.25ms**，均为待证伪预算，不相加，不承诺兑现。

## 公共执行约定：复制组织，重写硬件 fragment

**已证。** AMD wave32的lane拆为 `rc=lane&15, gr=lane>>4`。其16×16 FP8 fragment每lane是8字节，`w2_load<C>` 地址为 `((qt*(C/16)+ct)*32+lane)*8`，一次int2 load/store。普通权重片段地址为 `((n/16)*(K/32)+k/32)*512+(((k%32)/16*2+gr)*16+n%16)*8`。**当前W16仅用于C256的对应导出及宿主选路**：将同lane两个K16片段相邻，改为一次16字节load，再拆两个int2供两条WMMA。C64/C128普通当前路径仍用旧fragment布局、8字节权重读取；W16_SMALL未定义，小通道W16试验未采用。CPU packer明确是同一批字节的排列，且以新导出名/缓存key区分ABI；HasFn保持旧宿主/旧导出的旧布局，不能仅由模块编译宏推断family已启用。[fragment读写][amd-mh-load]、[权重地址][amd-mh-weight]、[C256导出限定][amd-mh-w16-scope]、[CPU排列][amd-pack]

**Intel执行约定。** 本次已只读检查现役绑定对应的TTIR/TTGIR，SHA与RESULT记录一致：ViT `_expand` 是i8×i8→i32、executionSize/threadsPerWarp=16；C64 QKV与post是FP16输入、FP32 DPAS累加。它证明编译产物的类型和布局，不提供新方案机器ISA、执行耗时或occupancy。库 `torch.mm` 内部ISA仍未知。下面的DPAS是逻辑伪代码，实际A/B寄存器装配沿用现有Intel backend的dot-layout转换，不能把AMD `gr/rc` 地址原样交给DPAS。

统一用SG16，先以FP16逻辑微块 M8×N16×K16、INT8 M8×N16×K32组织；较大M由重复微块组成。输出lane `l` 持有 `C[m0+r,n0+l]，r=0…7`。可复刻的是索引、每次消费后的释放点和复用次数；需重写的是A/B交错、广播、cross-SG通信和barrier语义。供数草案如下：

```text
FP16 A源块：lane l读 X[m0+l/2, k0+(l%2)*8 : +8]，一次16B
FP16 B离线pack：Bpack[kt,nt,l,p] = pack_half(W[k0+2p,n0+l], W[k0+2p+1,n0+l])
                p=0…7；每lane两次16B读，得到K16×N16的B块
INT8 B离线pack：相同槽位改为pack_i8(W[k0+4p : +4,n0+l])，覆盖K32
A源块 → backend既有DPAS A布局；Bpack → DPAS B布局；不假设转换免费
acc = dpas(Afrag, Bfrag, acc)；只保consumer下一次需要的片段
store_half/output[m0+r,n0+l]；边缘mask及原window ORDER由现役接口提供
```

这个源块合作分工是迁移设计，不冒充当前Intel物理lane分配。A转换可能用shuffle或SLM；必须查看新编译IR确认，不能因“用了DPAS”就宣称供数高效。FP16/INT8 pack必须分缓存key、dtype、K、head和variant；载入时一次完成，不能每帧重排权重。

AMD编译recipe也是实现的一部分。C32生产行使用 `CW_ROLL_HIDDEN=1/CW_ROLL_WINDOW=1/CW_VEC_INPUT=1/CW_PACK8=1`，LLVM23预编译路线加 `-enable-post-misched=0` 和 `-amdgpu-sched-strategy=max-ilp`；名为c64-wave2的模块行使用QT_BATCH4、HIDDEN_TILES2、ROLL_QUERY、LAUNDER_QKV、SCHED_FENCE、QKV_FUSE3，并编入 `W2_FFN_W16=1`。**这个模块同时承载多个family；W16宏只让C256对应导出可用，不表示C64/C128当前走W16。** build＋host限定已由Main/Hypatia核实，宿主HasFn及导出模板实参决定具体路径。C512 compact使用DEEP4；ViT使用W5/F8W。**行opts及LLVM23选择只在 `-RowOpts` 路线生效**，普通COMGR构建不能自动视为同配方。LLVM22+ gfx12 split barrier还需要 `HIP_BARRIER_FENCE=1`，裸barrier可能漏LDS等待。[recipe及编译条件][amd-build]、[wave2模块行][amd-build-wave2]、[C256导出限定][amd-mh-w16-scope]

Intel不照抄AMD `max-ilp`、MODE、sched_barrier或空asm；先固定SG16和tile/loop展开，给各阶段独立词法scope、限制展开宽度，核查load是否提前、旧fragment是否退出live range。跨SG共享必须使用Intel backend支持的SLM发布/工作组barrier；subgroup同步只覆盖本SG。

## N1：C512有效栅格QKV，随后迁移compact供数

**现役与已证差异。** accepted的[库QKV入口][our-c512-qkv]将三套权重合成一次 `torch.mm`，但处理shift后的padded HWC；[pack][our-c512-pack]再norm并发布head/window布局，[attention][our-c512-attn]保留全64 keys且probability unround已开。16块有效行共15360，当前QKV共19712，差4352行；按512×1536，少算约 **3.423G MAC，22.08%**。这仅删无效QKV行，不缩网络画布、不删attention keys。

AMD [c5c_body][amd-c512]每workgroup负责一个window/head，body支持两/四wave；两wave配置各自两个16-token tile；六个输出fragment对应Q/K/V各两列块。A按有效raster地址取数，边界从合法p=0取数、到消费时select成0。DEEP4将32个K16步骤完全展开，维持 `a[4][QT] / b[4][6]` 环；先复制当前槽到 `ca/cb`，再填同槽未来step，最后WMMA，故旧load不会在读完前被覆盖。QKV写6144B LDS，发布屏障后各wave只处理自己的query；ex/prob留寄存器，后面不再做第二个工作组屏障。[环与屏障][amd-c512-ring]

**移植执行顺序。** 第一版保留高效库GEMM：接口改为 `valid_hwc[24,40,512] + padded_shape + sx/sy + ORDER`；库只做960×512×1536。pack以window/token求有效坐标，有效从紧凑z读，无效直接产生零Q/K/V，仍发布旧packed布局和全64 keys。producer仍是现役完整INT8 FFN，consumer仍是现役attention；norm/概率数学先保持，避免把两项收益混为一项。修改位置为库QKVscope与pack接口，另建fast variant；不改当前冻结树。

第二版才替换库QKV+pack：每head用4个SG16覆盖64 tokens，K32源块内拆两个FP16 K16 DPAS；先试D2，再试D4。Bpack按 `part/head/kt/n16/lane` 索引；同一A依次供六个输出半head fragment。QKV accumulator只活到当前token tile的norm/发布，不能将整个window的六套float矩阵常驻。K/V共享SLM供全部query，Q仅保本SG query。float数据比AMD FP8 fragment宽，D4不能因为AMD快就成为Intel默认。

```text
load D个(A,B)槽
for kt递增到fullK512:
    ca,cb = ring[kt%D]       # 消费者私有副本
    prefetch(kt+D, same_slot)
    Q/K/V += DPAS(ca, cb)    # 每个acc保持K递增
norm当前tile → 发布packed half；释放ca/cb/float QKV
barrier；全64-key attention → 原packed projection/crop consumer
```

**待测。** 先验证valid-only版完整13帧、padding/shift及历史，再比较库版与D2/D4版。前者少算MAC明确，后者MAC不变、只减少供数/中间写读；库已有良好调度，融合也可能变慢。C512后续INT8分组融合是另一个缺口：AMD `split_ffn_one_w2f8`在同wave完成mix→expand→contract，expanded逐隐藏块消费、无LDS/global mix；本地 `_groups` 已有但现役未选，不能报成新实现，更不能把AMD FP8复制成我们FP8模拟。[AMD分组实现][amd-c512-ffn]、[现役五核][our-c512-rows]

## N2：ViT尾部按consumer方向流式执行

**现役与已证差异。** [完整ViT forward][our-vit-forward]是两次K512 QKV→half合并→Q/K norm→QK→exp→AV→四个64-key row_sum→分母修正→投影P4及merge。240是真实tokens，补到256只为当前attention路径；QK/e各写约3.93MB/块。strided QK、两核投影已经实现。[现役投影parts][our-vit-parts]、[merge][our-vit-merge]

AMD当前400/640导出选择 [vit_attention_transposed_score_body][amd-vit-attn]：交换Q/K WMMA操作数，使score直接处于分母/AV需要的方向；每次16 keys，寄存器内形成exponent及PACK8，累积分母与AV。再交换AV方向，让一个query的重复分母只求一次倒数，最后输出对应head/channel连续地址。这个当前路径没有老 `vit_attention_fused_body` 的score转置LDS/barrier；256导出仍是另一个fallback，不能把363行旧body当1080当前最佳实现。完整K640没有因流式而删keys。

QKV [W5实现][amd-vit-w5]是一个head/part、五wave共用权重：前四wave合作搬运，两槽双缓冲，每槽2个N16×8个K16权重tile；先stash并barrier，再预取下一chunk，五wave各自消费不同16-token tile。FP16版权重staging共16KB；F8W只用8KB字节staging，但保留16KB LDS供后续五个16×33 float norm tile复用。**先释放权重staging再把同片LDS交给norm**，不是同时保留两份scratch。此组织来自已运行导出；转换FP8成立依赖作者E4M3值域证明，我们不具备相同数值契约。

**移植执行顺序。** 第一版接口保持 `qkv_two_K512 → normalized_Q/K/V → stream_attention → projection_P4`；两K512 half和P4 half仅用于隔离结构收益。attention每SG16负责8 query×一个head32，每32 keys算一次QK、调用现役exponent规则、立刻DPAS乘V并累积分子。consumer方向由DPAS A/B布局决定；若交换操作数后仍需convert_layout，明确计入成本。fullK240读满，末16虚拟padding按旧规则参与分母再修正；不能直接用mask后分母冒充保边界版。

```text
Qfrag常驻；numerator_f32[8,32]=0；denom_chunk[4]初始化
for k0=0,32,…,224:                     # 最后一块仅16个真实key
    K/V取完整真实key，虚拟padding按旧路径构造
    score_half = DPAS(Q,K).half
    e_half = 现役exponent(score_half)
    numerator_f32 += DPAS(e_half,V)
    更新当前64-key chunk所需的有序half归约槽；释放score/e/K/V
旧四chunk half merge + exp(0)padding correction → reciprocal
numerator.half × reciprocal → attended.half → 原P4 projection
```

这里不能用普通softmax替换现役特殊exponent，也不能只在FP32中求和再声称“结构完全一样”。结构通过后，快速数学版可以保fullK改为原生norm/exponent、FP32分母、fullK QKV单accumulator、投影单accumulator；分别验收数值变化，无需永守旧拆分half。

QKV第二步借W5合作供数，但Intel先用4个SG16而非照搬160 threads；每SG只负责一个M8微块，共读同head权重。FP16权重chunk按2×N16×8×K16设计，双槽16KB；SG数与chunk宽分别编译。ViT FFN现役已[entry一次+INT8 expand/contract][our-vit-rows]，不是待从FP16改INT8。AMD expand M2/M4复用权重、contract按块累加并用4096B dummy LDS限制occupancy，[相应body][amd-vit-ffn]可学习，但完整FFN合成一核并非AMD当前stream的默认事实。Intel完整K4096 contract会减少P4并行，不能预设更快。

**待测。** N2先只替换尾部，固定240/fullK/8块/两图签名；确认无score/e全局surface、无新增hidden重复量化，再比较完整帧。逐段减少的写读量不是DRAM实测，更不能用9070 XT的每核时间乘比例推算B580收益。

## N3：C32整窗resident，细粒度释放而非大张量堆叠

**现役与已证差异。** ordinary七块是[native cubic MLP][our-c32-mlp]→[直接packed QKV][our-c32-qkv]→[已融合attention+project tail][our-c32-tail]。decoder首个C32另有unpack/投影，pre/post另有输入/输出契约；都在库存列出，不能把七块收益泛化到十块。C32 MLP已融合expand/cubic/contract，hidden没有全局128-channel落地。

AMD [cw_body][amd-c32]一wave32拥有64 tokens×唯一head32。qt逐16-token推进，hidden逐16通道展开后立即contract；`CW_ROLL_WINDOW/HIDDEN`禁止整窗/全hidden展开。Q/K/V存紧凑整数fragment，half残差4096B LDS；QKV操作数方向分别适配QK和AV。输入用8B字节或32B float向量读，half结果直接保留位型，不做half→float→half往返。finish按有效tile与边缘tile分支，尾部改lane所有权为pixel slot/channel group，一次16B LDS读、8B字节store；不是每通道u8散写。[滚动及消费][amd-c32-core]、[尾部向量地址][amd-c32-tail]

**移植执行顺序。** 接口为 `window_resident(X,Wmlp,Wqkv,Wproj,scale,bias,ORDER,shift,valid_hw,OUT)`。第一版4个SG16/64 threads负责一窗，每SG负责16 queries、两个M8微块；全窗K/V共享。输入地址保本地padding、shift4、ORDER，绝不照搬AMD post shift3。使用half数据，而非AMD字节流：残差64×32 half=4096B，K/V各4096B，AV槽4096B；Q只保当前query。按阶段复用SLM，必须列清旧consumer完成点，不能靠同地址假定安全。

```text
1 mapped load完整逻辑窗 → 当前MLP各part32的expand/half cubic/contract
2 feature.half → residual槽；释放expanded/hidden，只保当前part
3 QKV(fullK32)；Q/K按现役norm；K/V发布共享槽，Q留本SG
4 工作组barrier；每SG query遍历全部64 keys，score/e逐tile消费
5 AV.half发布复用槽；工作组barrier，确认所有heads/queries写齐
6 projection(fullK32)+原half残差 → mapped输出；释放K/V/AV
7 finish若需要，写真实skip与2×2 pool；新窗口依赖仍以global输出交接
```

阶段3前可释放MLP输入，阶段5后可覆盖K/V，但不能在另SG仍读V时覆写。C32唯一head也有cross-SG数据依赖，所以AMD单wave无工作组barrier的优势不能原样复制。也可编译2个SG16版，串行较多query，比较barrier、寄存器与并行之间的实际结果。

每消除一个有效HWC32 half surface，避免一次写与一次consumer读的逻辑量为 `4*M*32 B`；QKV省的是3个surface。MAC不变，SLM与布局转换会抵消部分收益。pre的K16→C32和post的上采样/merge已有融合入口，后续闭合必须继续保真实history/control以及RGB+logit，不能复制AMD只RGB/reset epilogue。[现役pre][our-pre]、[post入口][our-post-entry]、[post head][our-post-head]

**待测。** 先替换普通七块，保持same full windows/key set。r18 split-tail/gated-C64是在修候选压力，不能作为N3已经实现的证据。结构通过后可分别放宽half norm/激活/残差累加；不需要给resident边界新增FP8 pack。

## N4：C64/C128 QKV共读A，按tile限住live range

**已证偏差。** [C64 `_qkv_direct_pack`][our-c64-qkv]的program_id(1)是 `segment=head*3+family`，每family独立完整K循环；[C128 `_direct_pack`][our-c128-qkv]相同。已经norm+直接packed store，但同一head/tile的A请求三次。AMD [QKV_FUSE][amd-mh-qkv]在一个kt循环load A一次，更新Q/K/V各两个fragment；V的WMMA方向反过来，直接适配P×V。仅处理一个qt后norm/pack，释放六个float accumulator。空asm刷新qlane、sched_barrier和ROLL_QUERY用于阻止编译器将所有qt的A/B提前并延长生命。

**移植执行顺序。** 保原接口和packed输出，只将grid第二维从 `heads*3` 改为 `heads`。第一版BM16、BN32、BK32、4个SG16，与现役tile相同；C64/fullK64、C128/fullK128。每步load X一次，三个B块分别dot，保原half z及norm，先发布当前tile再推进下一tile。基于现役DPAS布局，4个SG按2个M8×2个N16分工；每SG需约三组8-row输出acc，而不是将4个qt全部摊开。

```text
for one rowtile/head:
    q,k,v = zero_f32(M16,N32)
    for k0到fullK递增:
        a = load_A_once(k0)
        q += DPAS(a,Wq[k0]); k += DPAS(a,Wk[k0]); v += DPAS(a,Wv[k0])
        release a/Wq/Wk/Wv
    原half z/norm/scale → 原packed Q/K/V store；release q/k/v
```

FP16 B pack见公共约定；head内跨N16的norm归约继续用现役布局转换/half树，不能把lane16内归约当完整head32。consumer仍是已采纳的[C64][our-c64-tail]/[C128 attention+project][our-c128-tail]，无需重做tail融合。A逻辑请求3→1，B和MAC不变；不是DRAM必减三分之二。C256 current recipe的QKV_FUSE3没有启用bit4；其bit8“Q+K合读、V另读”是减压力选项，不能宣称AMD已采用三路融合。

**待测。** 先单改C64，再C128，整帧重复比较；IR确认A只加载一次且不跨qt常驻。若triplet更慢，测Q+K/V拆分并记录多一次A的代价；这定位的是实现与寄存器压力，不是“融合原理无效”。原生norm候选可后续与同fullK快版一起验收。

## N5：FFN按token tile复用B，W16装载与短hidden生命周期

**现役与已证差异。** C64/C256 [branched `_pairs`][our-pairs]逐hidden part，C128 [pairwise][our-c128-pairs]已一次读X供两个hidden part；C128 [project_fp32][our-c128-project]跨branch FP32累加已采纳。AMD [QT_BATCH][amd-mh-ffn]以window为workgroup、tid/32确定head，一wave拥有该head全部64 tokens；每head只contract自有128 hidden，四个16-token tile共用一份B；HIDDEN_TILES2一次保两个hidden16；expand完成即cubic/PACK8→contract，然后释放expanded。**C64/C128当前用旧fragment的8B读取共享B；C256的W16路径**一次16B读供两个K16 WMMA，再用于四个qi。两种路径都复用权重，但读取宽度和ABI不同，不能合称小通道W16默认开启。输出contract才交给plane1；plane0=input、plane1=contract，随后plane0=feature、plane1=AV，阶段之间 `WG_FENCE; barrier; WG_FENCE`。[两plane及归属][amd-mh-planes]、[C256 W16限定][amd-mh-w16-scope]

**移植执行顺序。** 本地仍half激活与当前fullK，不用PACK8。接口保持branch latent与project consumer，先改B合作装载：一个workgroup负责同branch/窗口的64 tokens，编译QT2/QT4两版。每个SG负责M8/16 tile；B按K16×N16 FP16片段一次合作load，供2/4个tile消费。hidden一次只保32列，原half cubic后立即reduce到32列；跨branch projection保持C64/C256现役half顺序、C128现役FP32，仅做组织隔离。随后可按快速数学契约另测single accumulator。

```text
for hidden32递增到128:
    expanded[QT] = zero
    for fullK的K块递增:
        B = load_once(当前branch,hidden32,K块)
        for qi: expanded[qi] += DPAS(A[qi],B)
    H[qi] = 当前half cubic(expanded[qi].half)；release expanded
    R = load_reduce_weight_once(hidden32)
    for qi: contracted[qi] += DPAS(H[qi],R)
    release H/R/B；只保contracted[QT]
发布latent.half → 原project consumer
```

相对accepted算清复用基数：C64/C128当前BM32已在一个program覆盖两个M16，64-token组织理论B请求 **2→1**；C256 BM16理论 **4→1**。原N5“最多四tile共享”不应理解成三个family都再减四倍，也不代表cache后的DRAM量。扩到QT4增加accumulator与SLM，按QT2→QT4渐进，不能再采用“所有展开张量常驻”方案。

AMD扩展hidden不进LDS，因为FP8 byte fragment可直接喂下一WMMA；Intel half激活的DPAS A布局可能不同，必须在现有convert_layout中定位一次转置/广播，必要时只放当前hidden32的小SLM槽，不能把全部128 hidden落global。Intel上述FP16 B pack是迁移候选，不称作AMD小通道已采用的W16；每lane两次16B读取也不等于C256 W16的一个16B FP8读取。C64/C128先学习其当前8B fragment供数与QT复用，C256另测相邻K片段宽读，不能复制吞吐预算。

**待测。** C64、C128、C256逐family单开后做完整帧，打印actual tile与编译资源；现役C128 pairwise收益不能再计一次。r18候选的行量化一次是修重复工作，accepted本来已有独立entry一次；N5新增是B/fragment组织，不是重新声称INT8或pairwise。

## N6：C256依赖窗口调度，保留producer发布顺序

**现役与已证差异。** C256目前[packed attention][our-c256-attn]→[packed projection/crop][our-c256-proj]仍分开，encoder/decoder各8块按整层图依赖。AMD [sp_run_body][amd-queue]由leader原子领取ticket，取得window/layer后调用同一 `swin_wave2_body`；group完成输出发布，再对最多四个child计数，最后一个依赖到达才release写queue。空队列轮询是relaxed，成功后acquire；失败有timeout/abort。该body每WG取一个job，不能把名字“persistent”自动翻译成无限worker循环。

**移植执行顺序。** 先把C256 attention+projection+输出发布闭合到一个明确window task，再做队列。每个任务4/8个SG16分query/head，plane half按64×256=32768B，若双plane需65536B；这已远大于AMD FP8双plane32768B。因此第一版不能机械双plane常驻：保当前packed feature input，只为当前AV head/列组分SLM槽，读齐后由projection consumer消费并发布global。若SLM/GRF降低驻留太多，保split task，先不启队列。

CPU根据现役shift/ORDER生成两段C256 DAG，每node列明前层所需窗口及need数；不能删边界依赖。Intel以device-scope release/acquire替代HIP agent fence。禁止无条件全GPU counter barrier；Intel调度顺序与AMD未证等价，任何worker循环属于需单独验证的改写。

```text
leader领取已经ready的node；broadcast到整个WG；barrier
全部SG：完整FFN/QKV/64-key attention/fullK256 projection
global输出写齐 → WG barrier + device release
leader对child做atomic依赖计数；last predecessor release-publish(child)
consumer成功读queue后acquire，再读producer输出
frame末保原输出/历史consumer；abort则回原graph路径，不能发布半成品
```

这是Intel安全执行草案，不能称已复现AMD实际启动网格。AMD持久化variant使用QT_BATCH2而非普通wave2的4，说明调度与kernel资源需共同配置；C64/C128无相同成功收益证据，不推广队列。[persistent编译行][amd-build-persistent]

**待测。** N6只在前五项已稳定后；先C256单段及完整帧证伪，记录consumer等待、queue完整性和fallback。0～0.25ms预算优先级低。现役graph replay、纹理池、GPU fence relay已存在，这些不计新增；离线model RESULT没有模型外桥/Present dispatch，桥只能按accepted source核对。[现役handoff发布][our-bridge]

## 已有失败如何使用，以及本批交付边界

已有Swin完整29、ViT FFN完整30的完整帧回退是真实证据，不能覆盖现役55–60ms，也不能据此将原因写成CPU问题。ViT已证INT8 DPAS，但融合entry曾按N列重复扫描/量化1024-row，完整contract又改变P4并行；AMD是预打包、短fragment生命周期及明确供数次序。r18行量化一次/K64/P4、Swin split-tail/gated-C64是Main正在修整的候选，尚不能计为accepted新增收益。33的unsupported JIT `In`→`or`已由Main修复，当前未提供本补审可用的新测速。我们需要对照新IR中的实际load/live range，而不是用“DPAS已出现”结束诊断。

零spill门确有代码：accepted C128 pairwise preflight在155–156行拒绝非零spill；candidate Swin admission在739–743行只给显式diagnostic latent例外。[现役门][our-pair-gate]、[候选门][our-audit-gate]。它们是准入政策，不能证明有spill一定更慢，亦不能证明审计包装是帧回退主因。未来fast实验可以记录少量spill的完整帧结果再选择；本批不更改门、不绕过真实buffer/fence归属。所有旧拆分half边界也不是永久限制：AMD `CW_FAST_NUM/W2_FAST_NUM`已另有可选fast recipe，本地fullK单累加、原生norm/denominator/exponent仍须作为独立数值候选验证。

落地建议先 **N4→N1→N3→N2→N5→N6**，每次仅替换明确producer/consumer；先少变量定位组织收益，再放宽原生fast数学。最小证伪仍交由Luna串行跑同完整13帧、两graph签名、真实历史/运动/控制及全块/fullK；对照完整process/rawGPU和输出/历史变化，任何可重复收益都可接受，无1ms门。没有按模块计时的地方继续标未知，不把逻辑请求量当DRAM实测。

本批只交付此实现指导；没有GPU运行、内核开发、游戏安装或发布。pure data操作、前端/真实历史、五级decoder、post、模型外HDR/桥的逐核行继续保留在119行库存。AMD成功的decoder投影→2×上采样/skip、pool group供数、post尾部向量store值得后续学习，但本批不再追加预算或方案。真实history/控制在AMD regular reset中无同配方实现，不能编造对应成功kernel，也不能靠去掉它们取得所谓移植收益。

[amd-log]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/results/next-candidate-20261002/full-N.txt
[amd-mh-load]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L18
[amd-mh-weight]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L118
[amd-mh-w16-scope]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L126
[amd-pack]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/packed_weights.h#L61
[amd-build]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/build-modules.ps1#L24
[amd-build-wave2]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/build-modules.ps1#L66
[amd-build-persistent]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/build-modules.ps1#L74
[amd-c512]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_qkv_attention_compact.inc#L96
[amd-c512-ring]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_qkv_attention_compact.inc#L130
[amd-c512-ffn]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/c512_m32_deep.inc#L242
[amd-vit-attn]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L257
[amd-vit-w5]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/vit_stream.inc#L90
[amd-vit-ffn]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/deep_fast.hip#L650
[amd-c32]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206
[amd-c32-core]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L331
[amd-c32-tail]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L506
[amd-mh-qkv]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L563
[amd-mh-ffn]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L417
[amd-mh-planes]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L258
[amd-queue]: https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/swin_persistent.inc#L38
[our-c512-qkv]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/c512_qkv_library_16_v1.py:37
[our-c512-pack]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/experimental/fp8_unround_overlay/modules/fused_qkv_pack_native_half_v1.py:17
[our-c512-attn]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/c512_probability_unround_kernel_v1.py:14
[our-c512-rows]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/c512_int8_ffn_rows_v1.py:32
[our-vit-forward]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/fullsize_session_v1.py:131
[our-vit-parts]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/experimental/fp8_unround_overlay/modules/fused_vit_projection_v2.py:15
[our-vit-merge]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/experimental/fp8_unround_overlay/modules/fused_vit_projection_v2.py:35
[our-vit-rows]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/int8_ffn_segment_rows_v1.py:22
[our-c32-mlp]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/experimental/fp8_unround_overlay/modules/native_cubic_c32_v1.py:24
[our-c32-qkv]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/experimental/fp8_unround_overlay/modules/fused_c32_projection_native_half_v1.py:25
[our-c32-tail]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/post_attention_fusion_v1.py:24
[our-pre]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/native_k8_fp16_v1.py:26
[our-post-entry]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/post_entry_mlp_fusion_v1.py:70
[our-post-head]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/native_k8_active_720_v1.py:20
[our-c64-qkv]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/c64_qkv_direct_pack_v2.py:31
[our-c128-qkv]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/c128_qkv_direct_pack_one_v1.py:30
[our-c64-tail]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/c64_attention_project_fused_720_v1.py:19
[our-c128-tail]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/c128_attention_project_fused_720_v1.py:19
[our-pairs]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/experimental/fp8_unround_overlay/modules/batched_branched_mlp_v1.py:22
[our-c128-pairs]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/branched_mlp_pairwise_720_v1.py:35
[our-c128-project]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/branch_accum_native_720_kernel_v1.py:11
[our-c256-attn]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/experimental/fp8_unround_overlay/modules/window_block_attention_v3.py:16
[our-c256-proj]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/experimental/fp8_unround_overlay/modules/window_block_projection_v3.py:14
[our-bridge]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/gpu-handoff-owner-transfer-v1-20261002/payload/native/nr_texture_bridge_re8_v1.cpp:365
[our-pair-gate]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/branched_mlp_pairwise_720_v1.py:155
[our-audit-gate]: ${WORKSPACE}/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/phase2-complete/revisions/r18/source/game/audit_swin_720_v1.py:739
