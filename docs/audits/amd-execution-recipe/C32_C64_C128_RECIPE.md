# C32 / C64 / C128：AMD 执行配方与 Intel 转换记录

状态：READY，CPU 分析交付。metadata / AST / 地址布局检查不构成 GPU 资格；actual only Luna。本次没有运行 GPU、导入 Torch/Triton、编译或创建新 kernel。

比较对象是用户确认的现役 720p C512+K8/all6。Swin 的选择以 PHASE2-r18.json 的 arms.accepted_baseline 为准：C32 native hidden、C128 pairwise 与 C64/C128 attention-project 已启用，BranchAccum 仅 c128；三个完整 Swin opt-in 都未采用。r18 source 是共同库，文件存在不能证明启用。本次不访问 G，也不把 “all6” 当作 Swin 全部选项开启。固定 AMD HEAD 为 9ec741522d267c4d2365af53081716fb2943068a。精确路径、SHA、原始 RESULT 角色与资源摘录保存在 DEVIATIONS.json。

## 1. 成功配方首先是实际启用的代码

[AMD 构建配方](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/build-modules.ps1#L63) 的普通 c32-wave1 开启 ROLL_WINDOW、ROLL_HIDDEN、VEC_INPUT、PACK8、RTZ_PAIR、PREFIX/FINISH_TAIL_VEC，以及边界 full-tile 与 byte stream。编译使用 llvm23，关闭 post-misched，选择 max-ilp。普通 c64-wave2 开启 QT_SMALL_MASK=3、QT_BATCH=4、HIDDEN_TILES=2、FRAGMENT_WEIGHTS、QKV_FUSE=3、LAUNDER_QKV、SCHED_FENCE、ROLL_QUERY、PACK8=6、HOIST_LOADS=15。

因此 C64/C128 执行的是四 QT 批处理与两块隐藏片滚动；不是头文件默认的逐 QT 路径。W2_FFN_W16=1 的宽权重读取用于 C256 exports；W16_SMALL 没有构建定义，C64/C128 新 exports 不存在，host 的 HasFn 检查保留旧布局。历史小通道 W16 减少约 6% load 指令，但 ABBA 方向不一致，未采用。FAST_NUM、DEFER_Q、FULL_WINDOW、诊断 PHASE_REP 与 persistent 的 QT_BATCH=2 均不混入普通成功配方。依据：[AMD Body 的实际选路](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/hip_reference_network.h#L533)、[W16 小通道未采用记录](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/results/w16-c64-c128-20260930/README.md#L1)。

## 2. window、wave、head、thread 的所有权

AMD 两条路线的窗口都是 8×8、64 token；QT 是其中连续 16 token。下表的 M 使用现役逻辑 padded canvas，窗口数为 M/64，边界仍按原坐标处理。

| 实现 | 编译网格 / WG | wave 与 thread 所有权 |
|---|---|---|
| AMD C32 cw_body | grid=M/64；WG32，1 wave32 | 一个 wave 完整拥有一个 window，四 QT 顺序处理，唯一 head 的全部 32 channel |
| AMD C64 swin_wave2_body | grid=M/64；WG64，2 wave32 | head=tid/32，两个 wave 各拥有一个完整 head、全部四 QT |
| AMD C128 swin_wave2_body | grid=M/64；WG128，4 wave32 | 四个 wave 分别拥有四个 head；跨 head 混合通过同 WG 的 LDS |
| Intel 现役主要 kernels | 每个 program 为 4 SG16、64 threads | program_id 或循环定义 branch/head；不能把四 SG 直接称为四 head |

AMD lane=tid%32，rc=lane&15，gr=lane>>4；每个 ci fragment 的 lane 持 rc 对应的一行、gr 对应的连续八 channel，ci=0/1 合起来是 32 channel。每 QT 的两个 fragment 共覆盖 16×32；四 QT 覆盖 64×32，既不删 padding key，也不重复 token。V 的 WMMA 操作数方向相反，后述 B-fragment 编码不能机械套用 A 的逻辑解释。依据：[AMD cw_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_c32.inc#L206)、[AMD swin_wave2_body](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L265)。

Intel 的具体 lane→矩阵元素由编译后的 fragment layout 决定，尚未取得 native ISA 的完整映射；上述是已证 program 所有权与编译线程数。已读现役相关 cache metadata 的 threads_per_warp=16、num_warps=4。r18 C64 serial 的实际选中 SPV 及 LLIR 还核对了 SG16 和 SubgroupMatrixMultiplyAccumulateINTEL；SPIR-V 不能独立证明最终 native DPAS issue 效率。

## 3. AMD C32：小 live set，所有中间量有明确去处

cw_body 先为当前 QT 取得输入，按原坐标 zero extension；VEC_INPUT 将连续八 channel 的 float 读组织为两个 b128，byte 读为一个 b64。它保留逐元素算术，收益来自读组织，而不是换公式。Prefix 的噪声/历史输入与 post 的 low/skip/scales 各有真实入口，不能用普通 body 的假计数替代。历史阶段账指出输入与 FFN 都值得优化，但其百分比是 wave 周期、不是当前 Intel 的墙钟份额。依据：[C32 阶段账与宽读回归](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/results/c32-wave-phase-20260926/README.md#L13)。

MLP 按八个 hidden16 片滚动：每片先完成完整 K32 的两次 K16 WMMA，执行原激活与 FP8 边界，立即收缩到两个 N16 输出累加器。expanded 从不写 global/LDS；不同时保留 hidden128。完成的 feature 一份以 half 存入 4 KiB residual LDS，另一份变成当前 QT 的 FP8 A fragments。再逐 Q/K/V 投影；Q/K 的平方、求和、rsqrt 与 Q scale 保持该实现的次序。

全部 QT 的 Q/K/V 留在寄存器：每个 head 一个 64×32 FP8 矩阵约 2 KiB，三者合计逻辑 6 KiB，即 wave32 平均 48 DWORD/lane。之后逐 QT 对全部四个 key16 tile 计算 score、原指数位变换、分母、概率与 P×V，再投影和残差。这里没有标准 softmax，也不能删零 padding key。V 通过反向 WMMA 操作数直接生成 P×V 需要的 B fragments，免去一次 V 转置。

qt 与 hidden 循环禁展开，QT 间有 compiler sched_barrier，压住未来片的 load/temporary 生命周期。只有一个 wave 的 C32 core 没有 WG 硬件 barrier；尾段使用 WG_FENCE，不能照搬成多个 Intel SG 之间的同步。prefix/finish 尾部重新分配 lane 为 pixel owner/channel group，以 ds_load_b128 取八 half、PACK8 后写八 byte；2×2 pooling、half/RTZ 次序、crop 与 RGB head 仍各按原算法。

## 4. AMD C64/C128：四 QT 权重复用与四次 WG barrier

[四 QT / T=2 内循环](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L417) 的 A/W 内循环是成功实现的关键：

~~~text
一个 WG 对应 window；每个 wave 对应 head32
stage 全部输入到 plane0；WG barrier
contract[4 QT][2 N16] = 0
for ht = 0,2,4,6:                    # hidden128，T=2
    expanded[4 QT][2 N16] = 0
    for kt = 0..C/16-1:              # 完整 C，递增 K16
        A[4 QT] = load plane0
        for tile in 0,1:
            W = 一次 fragment-native 权重读
            对四个独立 QT 各做一次 WMMA(A,W)
    逐片激活/打包，Wcontract 一次读复用四 QT
    立即累加 contract，不写展开结果
store contract 到 plane1；WG barrier
跨全部 head 投影 + 原残差，feature 覆盖 plane0；WG barrier
每 wave 生成自己的全部 Q/K/V，再算四 QT attention
AV 写入已不再需要 contract 的 plane1；WG barrier
每 wave 拥有自己的输出32列，读取所有 head AV，完成完整 C 投影
~~~

每 K step 的四个 A fragments 各复用到 T=2；每个 W fragment 复用到四 QT。独立累加器让调度器能交错 WMMA 依赖链，同时降低重复权重 load。它不同于 r18 把 N 扩到64、共享两个 branch 的 X；更不同于先算完一个 branch 再串行第二个。contract 与 expanded 各为八个 f8，合计 128 个逻辑 DWORD/AMD lane，尚未算输入、权重与地址；源码作用域只给出逻辑 live set，不保证实际 VGPR 峰值。

plane0/plane1 各为 64×C byte：C64 总8 KiB，C128 总16 KiB。四次 w2_sync 均为 fence→s_barrier→fence：输入可读、contract 可跨 head 读、feature 可跨 head 读、AV 可跨 head 读。plane1 从 contract 转为 AV 时，Q/K/V 仍在各 wave 私有寄存器，所以不存在覆盖别人尚在读取的 KV。这是生命周期设计，不是两个异步 K-stage ping-pong buffers。

QKV_FUSE=3 在 C64/C128 同一个 K loop 中做 Q/K/V：A 读一次供六个 N16 累加器，V 用相反方向 WMMA。Q/K 经 w2_serial_norm，query_words 收束保存 Q；ROLL_QUERY 禁止四 QT attention 整体展开。LAUNDER_QKV 的新鲜坐标与 SCHED_FENCE 防止 LLVM 提早搬入未来 QT 的输入和权重。注意力按 head 独立执行，最终跨 head projection 只通过 AV plane 汇合。依据：[QKV / norm / AV](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/hip/wave_owned_mh.inc#L555)。

## 5. 预排权重、PACK8 与“预取”的准确含义

[AMD host fragment packing](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/HIP/packed_weights.h#L61) 把已量化的权重做纯 byte permutation，tile 是 N16×K32=512 byte：

~~~text
offset = ((n/16)*(K/32)+k/32)*512
       + (((k%32)/16*2+(k%16)/8)*16+n%16)*8 + k%8
~~~

它与 w2_weight 的 lane/gr 地址吻合，一次连续 b64 就是 WMMA B fragment；不在热循环中做 row-major 到 fragment 的 scatter。grouped contraction 的省略依赖 host 的零权重结构检查，不能把“完整 K”随意缩为 head32。

PACK8 把八个值用四条成对转换装进两个 DWORD，取代逐 byte 插入与重复 clamp。当前 MH mode6 对乘法、+0、非有限值和 MODE 临界区有专门处理；C32 的 RTZ pair 也没有授权 packed-half 算术重结合。历史 PACK8 记录有逐位回归和 ABBA 收益，但 C512/ViT 的同类尝试未采用，已经说明“宽转换必快”不成立。依据：[PACK8 历史回归](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/results/pack8-20260927/README.md#L1)。

两份 include 没有显式 async 多 stage prefetch 队列。实际采用的是 branch-free 安全地址装载、一次读多次复用、T=2/B=4 独立计算链、受限展开和 compiler scheduling fences。HOIST_LOADS 解决了部分 load→wait 串行化；记录明确保留 FFN 权重 loop 的两 load/wait 问题，软件 pipeline 类尝试也有无收益结果。不能把 compiler max-ilp、两个 LDS plane 或 num_stages=2 写成已证“多级异步流水”。依据：[AMD 装载与 wait 记录](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/results/hoist-loads-20260930/README.md#L25)。

## 6. 现役完整路径与 r18 偏离

以下采用零 shift 的代表 canvas，其他 shift 使用真实 Hp/Wp；表内是 logical program 网格，不是 GPU SIMD 驻留数。

| 现役入口 | C32，384×640 | C64，192×320 | C128，96×160 |
|---|---|---|---|
| MLP producer | native_cubic_c32._kernel，BM32；7680 programs | batched._pairs，BM32/N32；grid[1920,2]，每 program 一个 branch | pairwise._pairs_pairwise，BM32；grid[480,4]，每 program 一个 branch |
| MLP projection | 同一个 C32 kernel | batched._project，BM16/BN64；[3840,1]，每 branch 加后 round half | BranchAccum._project_fp32，BM16/BN64；[960,2]，按原顺序 FP32 加 |
| QKV + norm + pack | native C32 projection；[7680,3]，每 family 一个 program | direct pack；[3840,6]，head×Q/K/V 为独立 segment | direct pack；[960,12]，同样独立 segment |
| attention/project/crop | 普通 C32 tail；[2,3840]，一个 head | _fused BM32；[2,960]，program 内顺序两个 head、输出全部64列 | _fused BM16；[4,240]，program 内顺序四 head、输出全部128列 |

来源：[现役 C32 _kernel](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/native_cubic_c32_v1.py#L24)、[现役 _pairs / _project](../../../experiments/2026-10-03/r18/source/experimental/fp8_unround_overlay/modules/batched_branched_mlp_v1.py#L22)、[现役 C128 _pairs_pairwise](../../../experiments/2026-10-03/r18/source/game/branched_mlp_pairwise_720_v1.py#L35)、[BranchAccum 实际 dispatcher](../../../experiments/2026-10-03/r18/source/game/branch_accum_native_720_v1.py#L614)、[现役 C64 direct pack](../../../experiments/2026-10-03/r18/source/game/c64_qkv_direct_pack_v2.py#L31)、[现役 C128 direct pack](../../../experiments/2026-10-03/r18/source/game/c128_qkv_direct_pack_one_v1.py#L30)、[现役 C64 _fused](../../../experiments/2026-10-03/r18/source/game/c64_attention_project_fused_720_v1.py#L19)、[现役 C128 _fused](../../../experiments/2026-10-03/r18/source/game/c128_attention_project_fused_720_v1.py#L19)。C32 普通路径见 [现役普通 C32 完整调用](../../../experiments/2026-10-03/r18/source/game/c32_repeated_tail_fusion_v1.py#L34)；pre、first decoder32、post 有各自融合外层，不能用普通七块入口覆盖其实际 callee。

r18 的差异可明确到实现：

- _mlp_pairs 的 shared-two C64 选中 BM8/warps8；serial 在37/39实际选到 BM32/warps4，却仍写同一个 global LATENT，再由独立 _mlp_project 写 global MLP。其 source 先完成一个 branch 的 N64，再完成另一个，削弱了可交错的独立链。X 的逻辑读次数由 r15 shared 的2C/row变4C/row；**现役原单 branch N32 路径是8C/row**，所以不能把这次相对 r15 的翻倍误报为相对 baseline 的新增带宽。是否受 ILP 限制待测。[r18 _mlp_pairs](../../../experiments/2026-10-03/r18/source/game/audit_swin_720_kernels_v1.py#L39)
- r18 _qkv 以相邻 N32 segments 拼 N64。C64 是三个 pair programs，不是 AMD 一个 head 内 Q/K/V 共用整个 K loop；C32 第四个 segment 被正确 mask，但仍有 N64 点积列的 padding 工作。QKV 仍为全局 FP16。现役 direct pack 已省去独立 HWC QKV/unpack 往返，不能把它说成完全未融合。
- r18 C64 attention 选 BM16，现役选 BM32：每 window 的 query programs 从2变4，每个仍读取完整 K/V。这是已证的 program 与逻辑读请求翻倍，缓存是否命中、实际 DRAM 增量和耗时份额未知。C128 BM16 没有这一差异，但 r18 wide score/value 的 live fragments 更宽，tail SLM 记录为8 KiB，现役相关 cache 为4 KiB。
- r18 C256 fused tail 的 BN128 将 attention 随两组输出列重复；split 把它改成独立 head attention，但加入真实 ATT 写读和16个额外 kernel launch。38b/39 的192个角色相对37的176个角色正包含这16个 attend。它与 AMD C64/C128 的 WG 内 AV plane 汇合不同；现役 C256 原路径已有独立中间边界，不能把新增量跨参照系混算。[r18 C256 split](../../../experiments/2026-10-03/r18/source/game/audit_swin_720_kernels_v1.py#L371)
- accepted Q/K 保留 ordered half norm；r18 有 finite FP32 norm/denominator 及 exceptional fallback。AMD FP8/RTZ、串行 norm、倒数修正也各有自己的数学。移植调度不自动授权替换这些公式。[r18 normalization](../../../experiments/2026-10-03/r18/source/game/audit_swin_720_kernels_v1.py#L208)

## 7. Intel SG16 可落地的转换边界与预算

直接借用：64-token 所有权、权重 fragment 冷重排、A/W 多 QT 复用、滚动 hidden、V 消费方向、phase scoped scratch、coalesced 写出。必须转换：wave32 的 lane ABI、WMMA→原生 FP16 DPAS fragments、FP8 plane→FP16 plane、AMD fence/MODE，以及半精度/异常值次序。

单 SG16 照搬 B4/T2，MLP 的 contract+expanded 就是256 DWORD/lane，AMD wave32 为128；这是数学尺寸预算，**不是256个物理 GRF**。全部 FP16 QKV 私有保存又要192 DWORD/lane。不能无预算地“整窗大 fusion”。

建议的受控映射为 C64/C128 每 head 两个 SG16：SG偶数拥有 QT0/1，奇数 QT2/3，WG仍为64/128线程。每 SG B2，先试 N16 rolling，但必须把两块 hidden16 合成原 K32 contraction step；不凭空改成两个可重结合的小 K reduction。C32 可先一 SG 顺序四 QT，QKV 暂存 SLM；对所有 tail 使用现有 crop/接口。

~~~text
cold: 原 FP16 权重纯重排为已核对的 DPAS B ABI，锁定对象/源/资源
WG(window); head=sg_id/2; owned_qt=2*(sg_id%2)+{0,1}
R = FP16 feature plane
arena = max(contract, full KV, AV) 的受控别名区

gather 原 padded view → R；barrier
for hidden K32 group，按原次序:
    生成两个 hidden16；cubic 保留原 half 边界
    组成原 K32 operand，DPAS 收缩到 owned QT 的 head32
store contract → arena；barrier
投影读取全部 branch，保留该 family 的逐 branch half/FP32 加法
feature 覆盖 R；barrier
共享每 K32 的 A 做 Q/K/V，ordered norm 与 Q scale 保持
Q 留 owned QT 寄存器，全部64-token K/V → arena；barrier

对 owned QT：
    score0/score1 覆盖32+32 keys，原 exp/分母/异常策略
    P×V 保持原递增 K32，两份相加后原 half 边界
    AV 暂留寄存器
barrier                         # 所有人结束 KV 读取，尚不能提前覆盖
store AV → 已死的 KV arena；barrier
每 SG 拥有的输出32列读取全部 head AV，完整 C 的 DPAS 累计
按原残差次序，ORDER/SY/SX crop，再交还真实 caller
~~~

这给 C64/C128 六次 WG 同步，比 AMD 四次多两次：KV 变为共享 SLM 后必须先发布，并在最后一个读者完成后才允许 AV 覆盖。可选择保留 KV 寄存器以减少同步，但代价就是上述 live set；两条路线都未测，不能预报更快。wholly-cropped query 可跳算，却仍须到达共享 WG barrier；所有64个 key始终保留。

| 预算，不含编译器 scratch/权重 stage | C32 | C64 | C128 |
|---|---:|---:|---:|
| AMD core LDS | 4 KiB | 8 KiB | 16 KiB |
| Intel 转换：R 与 QKV/共享KV arena | 16 KiB | 24 KiB | 48 KiB |
| 若额外双 slot 暂存各 head 的 K32×N32 FP16 weights | +4 KiB | +8 KiB | +16 KiB |

C64/C128 B2/T2 的 contract+expanded 为128 DWORD/SG lane；N16 rolling 主累加器可降为96，暂留第一块 hidden16 再加16，A/W 片约再加48，粗略逻辑峰值160，未含地址/转换。若再保留下一 K 的 A/W，约加48到208。MLP 累加器必须在 QKV 阶段前死亡，不能把各阶段预算简单相加。实际 GRF、SLM limits、bank conflict 与 resident WG 数均待 native compiler/运行证据验证。

双 slot prefetch 只作为后续有界诊断：先复现冷重排、B2 的真复用和生命周期，再比较显式下一 K operand 预装；保持 K32 累计次序，环 slot 不得越过 owner/barrier。它是 Intel 新转换方案，不是声称 AMD 已有 async pipeline。first decoder32 保持 [first decoder32 cold binding](../../../experiments/2026-10-03/r18/source/game/audit_swin_720_v1.py#L311) 的实际 gather callback，[现役 K8 外层接口](../../../experiments/2026-10-03/r18/source/game/post_attention_k8_combined_v1.py#L1) 的 RGB/history crop 仍由真实 caller 消费；不得只在 body override 记一个 tail hit。此处仅伪代码，没有落地新 kernel。

## 8. zeroSpill 仍慢：哪些已证，哪些待测

| Luna 原始结果 | full baseline→candidate，ms | raw baseline→candidate，ms |
|---|---:|---:|
| 37 C64 serial | 46.019511→50.237196，+4.217684 | 40.045409→43.057761，+3.012352 |
| 38b C256 split | 45.047300→54.863604，+9.816305 | 38.474086→47.940367，+9.466280 |
| 39 combined | 46.578913→50.305881，+3.726968 | 40.185409→43.237022，+3.051613 |

三 profile 都装完整 Swin、扩展 c64/c256 BranchAccum，并开启宽 QKV、FP32 norm/denominator 等共同选项。这些是整套 candidate 对 accepted_baseline，**不是 A/B 独立阶段消融**；不同 attempt 的 delta 不能相加。raw 已变慢，不能把 full/raw 差称作 CPU 独占耗时。

38b 的原始 C64 admission：BM32/w4 spill832、BM16/w4 spill320、BM8/w4 spill384，BM8/w8才为0；37/39 serial 的 BM32/w4已为0。各次记录的256-GRF build flag与 registers=256 仍不代表高 occupancy。QKV/attend 的 registers=0 也不证明不用寄存器。

已证的是更细 query 网格、全局边界仍在、serial 分支的重读/独立链改变、split 的真实写读与 launch、宽片及 SLM 差异。待测的是每阶段时间、native load/wait/DPAS 排程、cache hit/DRAM、GRF 实际分配、驻留与功耗时钟。AMD 的历史驻留工作使用硬件 ID/时间戳测并发；[AMD 旧版本驻留实测方法](https://github.com/lmxxf/dlss5-on-amd-9070xt-porting/blob/9ec741522d267c4d2365af53081716fb2943068a/Development/results/launch-occupancy-20260923/README.md#L3) 是旧四-wave C32，不能把那组数冒充当前 wave-owned 或 Intel。后续应按同 revision 的真实 off/on 小消融测这些量，保留完整 K、已准入资源和数值/历史门；本次不运行。

复核：30份源码记录与固定源核对，六份原始 RESULT 的 SHA 与 Main receipt 相符；CPU 8项62子例通过，覆盖实际 host packing 方程、lane/QT/LDS 覆盖、构建分支、四 barrier、默认关闭、26个实际网格、选中 SG16/SPV 与预算。候选均为44个真实 Swin site；176/192是角色标签数。提议的 SG 分区与预算检查只验证组织代数，metadata 测试非 GPU 资格，收益与品质均待 Luna。
