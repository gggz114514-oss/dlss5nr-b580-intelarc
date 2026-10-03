# C32注意力归一化融合（2026-09-09）

本页保留C32融合当时34文件、668次检查点的记录；后续权重融合状态见ATTENTION_WEIGHTS_SPEED_STATUS.md。

已合入共用后端的显式triton/XPU路径。1080p成对完整模型实验由2.4620716833秒降到
2.3089370750秒/帧，速度比1.066322556，约快6.6%。原模型、权重、K8/K16矩阵
舍入及Swin/ViT指数路径均保留。当前仍非实时，完整迁移、XMX/INT8和生产工具接入未完成。

## 保留的数值行为

原32通道半精度平方/乘加、每步半精度舍入、XOR 4/2/1求和树、最小值钳位、
FP32倒平方根再转半精度、最终半精度乘法被放入一次GPU调用。仍复用经过验证的
精确半精度FMA辅助函数；没有采用普通连续求和、普通softmax、蒸馏或新量化模型。

倒平方根探针v1在全部有限半精度输入与当前PyTorch XPU相同，差异为NaN编码。
v2显式保留符号及有效载荷并置quiet位，65536种半精度输入逐字节相同，有限输入
还与独立FP64数学计算再转半精度相同。不据此声称所有输入、所有硬件的原生MUFU
位级等价。未采用查表；v1另一备选路径因本地sqrt_rn仅有FP64重载而编译失败，
失败日志保留。

归一化v1混合位模式测试有31498个半精度元素的位编码不同，分布在1077行；
这些元素两边都是NaN，有限数值无差异。v2对二元运算显式保留左NaN优先、
其次右NaN的现有XPU顺序。
原失败数据、所有NaN通道配对、四组新增随机混合位模式均通过。失败证据没有覆盖。

## 验证和速度

- 候选32组原语、每组4种配置全部字节一致；包含所有有限half常量行、8个位置的
  全部65536编码、NaN配对、不同dtype/步幅/维数/空输入，以及4个剖析尺寸。
- 原语配置根据[32768,32]合成输入的实测中位数选择16行、4 warps；此配置在4种
  实测尺寸约快5.25–15.44倍。这是局部函数计时，包含分配，不代表整帧提速。
- 两套独立模型同时驻留，各13帧；逐帧反转执行先后，采用预先定义的第1–12帧均值，
  包括最终reset，排除首帧即时编译；包括模型调用及同步，排除上传、IO与比对。
  候选第1运动帧较慢的记录仍保留在均值中，没有事后删样本。
- 26次成对实验的646963200个RGB字节及私有历史均重新读盘，与重复验证的4060
  原生捕获一致。该实验中的归一化入口覆盖attention、multihead_block、vit_block
  三个实际引用，每帧678次；half_fma计数1362→6，矩阵、FP8、cubic、指数计数不变。
- 正式候选包CPU10/GPU18全部通过，验证了公共分派、旧CPU/reference路径、按需
  导入、非法形状与输入所有权。默认reference路径及其函数体AST保留。
- 正式包：256/480/1080真实运动各13帧，512/480/1080/2559四组控制/UI各4帧，
  共55次；RGB、未合成UI的私有历史、重置、非法输入及调用方修改保护均通过。
- 主目录：CPU10及480p13帧通过；55帧与主目录13帧保存的数组均独立重读核验。
  正式目录1080p后续帧2.3168秒，主目录480p0.7679秒，2559控制/UI3.7029秒。
  这些不是与旧版交错执行的性能比较，小尺寸尚未证明提速。

## 源码与证据

主目录34个py与reference/experimental/attention-normalize-stage-v1/nr_backend一致。
只修改attention.py，新增triton_attention_normalize.py，其他32个旧文件逐字节不变。
前一版33个源码文件及600次检查点位于reference/experimental/revisions/
before-main-attention-normalize-v1。新日志、模型输出及编译缓存实际位于D盘。

关键SHA256：

| 文件 | SHA256 |
| --- | --- |
| backend/nr_backend/attention.py | ff719d3f146d741c8f6cf2f07b112a03836032ec5b404aaf57892064824e6721 |
| backend/nr_backend/triton_attention_normalize.py | 9a7a91f13d485392b8b3b4045600cd15633903532266c058cda7e50c1ab439cc |
| reference/backend-checkpoint.json | 9e188f59543a9878ba729408f1fa1a9fb4dfa579546e11525db9b97672bc2e9c |
| reference/attention-normalize-backend-promotion-v1.json | 111b45d910fe900a9dc67a34140928f600f02c48c371c1f4b0476a9aff70f61e |

数据根目录：D:/Codex-NR-Experiments/nr-b580/reference。
原语位于experimental/attention-normalize-v2/primitive.json；成对实验位于
results/attention-normalize-1080-v1/validation.json及saved-audit.json；正式包及主目录
审计位于experimental/attention-normalize-stage-v1/。所有本轮受控进程均正常结束；
早期rsqrt/归一化v1失败日志单独保留，不属于通过记录。

checkpoint_attention_normalize.py独立归档本版68次比较。旧收集器保持原内容，通过
历史源码快照核对归属；检查点共372次历史参考、668次优化比较（11版）以及26次
存档完整INT32图形共享比较。本版未新增图形资源、OptiScaler或ComfyUI生产入口证据。
这些比较包含重置复现，不能作为668个不同场景，也不证明全部控制组合已覆盖。

下一步速度工作：注意力行求和与权重归一化融合、减少数据搬运；继续保留每步舍入和
完整模型语义，再寻求有效XMX/INT8覆盖。旧剖析属于几何32源码版，区间有嵌套及主机
分派空隙，不能相加推断当前34文件版本的时间占比。
