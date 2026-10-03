# Pre 后端验证记录（2026-09-08）

目标仍是可由 OptiScaler 和现有工具复用的完整 B580 NR 后端。**当前已完成整个pre主体的跨卡输出验证：以相同原生front特征为输入，四份素材的skip和down均逐字节一致。** 后续四个C32块也已完成连续输出验证，见[C32_BACKEND_STATUS.md](C32_BACKEND_STATUS.md)。reset front的3个噪声通道仍有小误差，更深网络与时序仍未移植；没有完整 B580 NR 画面，也没有实时性能结论。

最新验收报告：`reference/results/pre-holdouts-v1/pre-outputs-validation.json`。每份输入skip为3,276,800字节、down为819,200字节，gradient/checker/gray/Apex(seed17)在CPU与B580两端均零差异。对应原始CUBIN未修改；holdout输入/输出文件哈希及同源输入已核对，见同目录`pre-skip-validation.json`。此前仅覆盖部分注意力查询的隔离结果，被这个完整输出门槛进一步验证。

可复用入口是 `backend/nr_backend/pre_block.py` 的 `PreBlock(record)`：`forward_features_outputs(features)` 返回(skip,down)，两者为逻辑HWC32并已经过FP8边界；`forward_outputs(rgb,padded_size=(320,320),seed=...)` 还会使用便携front，此时不宣称逐位一致。当前是研究正确性路径，未优化XMX/INT8，也未接入OptiScaler。

## 固定实验范围

运行库为用户肉眼验收过的社区 SF-v2（DLL SHA256 `6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927`）。WEIGHTS_HT SHA256 为 `836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4`。仅提取 `block0.layer0.layer` 的 21696 字节用于实验。没有更改原 NR DLL 或驱动。

SM89 原始 CUBIN SHA256 `676d04897fe88e3b21f766b89da72b85f468fa414835b02a8d9215aaa4fb3b1e`。`reference/native-replay/pre_replay.cpp` 用 CUDA Driver API 重放 `cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8`：输入256、padding320、down160，40×40 CTA，每CTA32线程。RGBA16F normalized point texture，无历史/深度/运动纹理，控制参数为捕获值。尺寸推广公式尚未确认。

原始独立重放的 skip 3,276,800 字节和 down 819,200 字节分别与 NGX 捕获逐字节相同。证明见 `reference/results/pre-mlp-evidence-verification.json`。后续插桩只修改独立提取的研究 CUBIN，替换指令逐条经过官方 nvdisasm 回读验证。全部笔记本源码、输入、输出、临时文件和缓存位于 `E:\Codex-NR-Reference`。

## 自有 B580 组件

- `backend/nr_backend/front.py`：reset feature producer、投影。输入显式 HWC3/4 与 padded_size；噪声由 seed、坐标的整数哈希生成。默认投影走精确研究路径。
- `backend/nr_backend/tensor_math.py`：共享指数对齐、整数累加到 FP16 的单次舍入、cubic 激活。全部张量计算可留在 XPU；不依赖 B580 FP64。
- `backend/nr_backend/pre_mlp.py`：`PreMLP(record)` 解码权重并执行整个第一层 MLP。从输入投影先量化 E4M3，32→128扩展、cubic、E4M3、128→32收缩，再量化 E4M3。收缩的初始累加器是**未量化投影×skip_scale**，与公开图的残差位置不同。

这些组件优先用于逐层正确性验证。其整数/张量分批算法不是高性能 XMX 内核，不能把诊断耗时当成最终吞吐率。异常值不被静默清零；当前有限输入证据不覆盖所有 NaN/Inf 行为。

## 原生数值规则与布局

front 为 NHWC16 FP16：0–2三路高斯，3常数1，4–6归一化RGB，7–9 reset时重复RGB，10零，11–12一，13–14负一，15零。源纹理先转half；颜色 `(rgb-.5)*.125` 两步分别舍入half。只对越过右/下边界的输入纹理采样做单次反射，噪声仍用未反射的绝对坐标。噪声3通道尚未和 NVIDIA MUFU 逐位对齐。

投影权重两块half tile分别位于0x2010/0x2210。HMMA每K8累加到half，共K16；乘积与初始累加器按最大**操作数指数之和**对齐，向零截断至24个小数位，整数求和再half舍入。QMMA同理，每K16、13个小数位。规则由原生采样实验推导，当前不宣称是所有 CUDA GPU 的通用契约。

FP8权重按K32、N16分块，tile内32 lanes×16 bytes。lane=g*4+t；每N8 fragment的两寄存器各4字节，对应逻辑K `2t + byte%2 + 8*(byte//2) + 16*register`。输出N为`g+8*fragment`。扩展权重0..0xfff，收缩0x1000..0x1fff；当前固定record这两块均无E4M3 NaN编码。不要把公开加载器的通用“4个NaN”注释当成本record实测。

cubic按原生顺序：clamp到±4，两次HFMA2（每次融合乘加后只舍入一次half），再用未clamp的原值HMUL2。系数为half可精确表示的 `.055908203125/.447265625/.89453125`。

## 验证证据

素材：gradient/checker/gray，以及用户Apex截图的256×256原像素裁剪（1050,650到1306,906）。前三者seed0，Apex seed17。均输入256、padding320。

| 边界 | 每素材比较数 | CPU/B580结果 | 证据 |
|---|---:|---|---|
| native front→投影 | 3,276,800 FP16 | 四素材零差异 | `pre-mlp1-dump-v1/shared-exponent-validation.json` |
| MLP128扩展/激活（16/64像素每CTA） | 各3,276,800 FP16 | 四素材两边界零差异 | `pre-mlp1all-dump-v2/all-channel-validation.json` |
| native投影→完整MLP（全部像素） | 3,276,800 FP8 | 四素材零差异 | `pre-mlpout-dump-v1/full-chain-validation.json` |
| 整数缩放→half舍入 | 163,489值 | CPU/B580零差异 | `pre-mlp-evidence-verification.json` |

上表证据路径均相对于 `reference/results/`。舍入验证包含随机大整数、half相邻数中点、次正规边界和零；CPU FP64仅作为oracle。两组holdout下载共54个文件哈希已验证，输入与对应原始捕获逐字节相同。

包含便携 front 噪声的全链存在明确剩余差异：B580 gradient/checker/gray/Apex 的最终MLP FP8值分别有75/102/66/112个不同（每组3,276,800）；最大绝对差0.125。所有其他已隔离的投影/MLP边界零差异，不代表整个NR正确。

`pre-mlp1all-dump-v1` 是无效诊断，不作为参考：即时store没有等待Tensor Core固定延迟，读到旧寄存器值。v2增加显式延迟，且前16通道与之前保留原生前缀的独立快照逐字节一致。失败实验保留用于追溯，不覆盖有效数据。

## 后续

pre主体与down验证已完成，sequence13–16的四个C32块也已通过，详见C32记录。下一原生调用是sequence17的 `cc_tinlayout_fused_swin_2h_64_2_inpview_tilesync_fp8`，grid10×10、block32×2、88字节参数。随后推广其他blocks和尺寸、时序状态、控制参数，并解决front噪声精确性；最后回到OptiScaler资源链路与XMX/INT8优化。B580最终画面仍必须交用户肉眼审核一次。

## 注意力入口新增证据

`backend/nr_backend/attention.py` 新增 `PreAttentionFront`、`normalize_c32` 和 `score_exponential`，`tensor_math.py` 新增每窗口矩阵的 `sm89_f16_batched_dot`。这些均是部分组件。

- QKV实际起点0x2460，结束0x3060，已由原生96通道输出确认，公开0x2450偏移不能使用。顺序确认为Q/K/V。Q与K均做half归一化；Q还乘0x5060的float32转half scale（原float32值0.5997287631034851），再分别量化FP8。V后续边界仍在验证。
- `pre-qkv-dump-v1/qkv-validation.json`、`pre-qnorm-dump-v1/normalization-validation.json` 和 `attention-front-holdouts-validation.json`：四素材QKV每组2,457,600值、中间K组归一化每组819,200值，CPU/XPU零差异。后者54个下载文件已校验哈希和同源输入。
- `pre-score-dump-v1/score-input-chain-validation.json` 和 `attention-tile-holdouts-validation.json`：从同一原生MLP输出开始，四素材Q/K输入字节与原生一致；每CTA一个M16×N8分数块（含half bias），每素材204,800个half分数和指数值均零差异。holdout另54个文件哈希校验通过。
- 该块使用Q的A0 16像素与K的A1后8像素（窗口行序像素20–23、28–31）。bias来自0x3260 tile的第二N8 fragment，与捕获逐字节一致。
- 指数近似并非标准exp：half融合乘加 `score*.044921875+1.30078125`，clamp到`1.03125..1.5693359375`，half位型做`(bits<<5)+0x8000`后截断16位。原生用packed LEA完成，clamp保证跨half进位等价。有效参考是 `pre-exp-dump-v2`；v1误选了另一N块的第二输出寄存器，已排除。
- `pre-softmaxprep-dump-v1/softmax-prep-validation.json`：gradient从原生MLP输出计算Q/K，扩展到A0 16查询×全部64个key，1,638,400个指数值和204,800个部分行和，CPU/XPU全部零差异。这里**还没有执行完整softmax归一化**。

窗口pixel顺序：依次base=(0,4,32,36)，每base的word=(0,1)、g=0..7映射`base + g%4 + 8*(g//4) + 16*word`。bias按每M16×N16一块512字节，A块优先、N块次之，每lane8halfs；fragment/word的两half对应M=`g+8word`、N=`8fragment+2thread+half`。当前全部64key验证只覆盖A0查询。

64key指数按8个N8片分组，片内对应lane的两个half。原生部分和为`(((e0+e1)+(e2+e3))+(e4+e5))+(e6+e7)`，每一步half舍入。

## 完整pre主体的最终对齐

- 行归一化：部分和的4个half2按0、1、2、3依次累加，每步half；最后两个half相加、clamp到6.198883056640625e-5，FP32倒数再half，乘指数值后量化FP8。`pre-softmaxdenom-dump-v1`确认A0的16个查询倒数映射到lane0..15。`pre-valuetile-dump-v1/value-tile-validation.json`确认倒数、819,200个注意力输入字节、1,638,400个V输入字节和204,800个初始加权值全部零差异。
- V先由half量化E4M3，再计算64key加权和。`PreAttentionCore`执行完整64×64窗口注意力；gradient的A0加权V共819,200个half全部一致，见`pre-attended-dump-v1/attention-core-validation.json`。
- 输出投影权重0x5070..0x5470，skip scale在0x5470。初始累加器是**未量化MLP收缩输出×skip scale**。`PreMLP.forward_unquantized`保留该边界；原生A0未量化MLP捕获也确认逐位一致。
- 最后出现的191个half投影差异、5个FP8 skip差异，来自E4M3次正规操作数的指数：硬件对齐保留格式指数-6，普通`floor(log2(abs(value)))`会错误得到-7/-8/-9。修正后，完整pre输出门槛四素材两设备全部通过。`tensor_math.py`以`_fp8_operand_exponent`明确实现该规则。
- skip原始布局为`[H/4,W/4,32lanes,4words,4bytes]`。lane=g*4+t；局部像素(x,y)=`(g%4,g//4+2*(word%2))`，通道=`2t+byte%2+8*(byte//2)+16*(word//2)`。`encode_c32_quad_bytes`重打包后与原生完整缓冲区SHA256相同。
- down从**未量化**输出做2×2池化：`half(half(topLeft+topRight)+half(bottomLeft+bottomRight))`再half乘0.25，最后FP8。物理布局`[C/16,H,W,4threads,4bytes]`，每16通道内顺序是`0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15`。`encode_split_c16_bytes`与原生down原始字节逐位一致。先量化skip再池化的对照会在gradient产生167,253个错误字节，已排除。

关键产物：`pre_block.py`、`attention.py`、`tensor_math.py`、`pre_mlp.py`；最终验证脚本`reference/validate_pre_outputs.py`。所有参考数据仅是中间张量，没有用拟合单张结果或替代模型冒充原模型。
