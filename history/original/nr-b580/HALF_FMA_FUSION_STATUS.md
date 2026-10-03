# 半精度乘加融合候选（2026-09-08）

独立Triton内核保留TwoSum残差以及半精度舍入中点修正，将现有多次张量运算合并执行。
输入仍先转FP16，支持广播与非连续步幅；没有将原算法替换成先FP32相加再转half。
关闭编译器浮点融合，以保留中间运算顺序。已接入主后端的可选 Triton 分支；默认仍为 reference。

CPU解释器10组、B580编译GPU11组全部通过独立Python任意精度整数oracle比较：
24,007个原始随机/平方/舍入见证，两组分别覆盖全部63,488种有限FP16编码的构造，
转置广播、标量、空输入与块尾数。GPU另验证1,080,315元素的跨原临时工作区边界输入。
范围为有限操作数，不是穷举所有三元组，也不是完整模型已通过的证明。

| 元素数 | 原乘加中位数 | 融合中位数 | 算子加速 |
| --- | --- | --- | --- |
| 4,096 | 0.3904毫秒 | 0.0725毫秒 | 5.38倍 |
| 1,080,315 | 0.8519毫秒 | 0.1112毫秒 | 7.66倍 |

两版均先预热再测五次，每次结果重新与整数oracle输出核对。仅表示该算子的收益，
不能用作整帧加速或实时性能结论。

源码reference/experimental/half_fma_triton.py，验证脚本validate_half_fma_triton.py在同目录。
证据reference/half-fma-triton-cpu-v1.json和reference/half-fma-triton-xpu-v1.json。
完整模型副本reference/experimental/half-fma-stage-v1已通过四种尺寸各四帧。
显示RGB、私有历史以及重置、非法UI、调用方纹理改写、跨尺寸状态保护检查全部通过。
审计再次读取16份RGB和16份历史npy，匹配已认证的原生输出字节和收据。

| 尺寸 | INT32版热运行平均 | 新乘加融合平均 | 本轮加速 |
| --- | --- | --- | --- |
| 512×512 | 3.178秒 | 2.453秒 | 1.296倍 |
| 864×480 | 4.209秒 | 3.299秒 | 1.276倍 |
| 1920×1080 | 16.717秒 | 13.334秒 | 1.254倍 |
| 2559×1439 | 27.826秒 | 22.776秒 | 1.222倍 |

平均为各尺寸后三帧模型调用和同步，排除首次新形状编译，不包含输入上传、输出读回或
视频/游戏接入。测试使用与INT32版本相同的素材、控制及原生参考。仍未达到实时性能。
覆盖固定SF-v2、SDR、同尺寸UI、零深度、合成细运动及reset[1,0,0,1]；Apex来自重复截图。
此结果不自动扩展到所有控制组合、真实光流、深度/HDR或任意游戏。

默认reference分支通过源码AST对照及两组注意力CPU检查。新计数half_fma只在该融合
分派实际发生时出现；512/864/1080/Apex分别为1650/1755/3205/4444次。
完整证据reference/half-fma-stage-dimensions-v1.json及reference/half-fma-stage-full-audit-v1.json。
已按源码哈希从验证副本推广到主目录，旧主目录与检查点保存在
`reference/experimental/revisions/before-main-half-fma-v1`。七组主目录 CPU 检查通过，
主目录 512×512 四帧完整模型的 RGB、私有历史和状态保护也全部通过。
检查点新增本版本 16+4 次比较，总数为 372 次参考、140 次融合，按源码版本保留。
证据：`reference/half-fma-backend-promotion-v1.json`、`reference/half-fma-main-cpu-v1.json`、
`reference/results/half-fma-main-ui-dimension-512x512-temporal-rgb-b580-v1/validation.json`。
旧 INT32 的 80 帧控制结果不视为此版已重新测试。864/1080 的旧输入实际重复首帧，
测试范围修正见 [VIDEO_FIXTURE_CORRECTION.md](VIDEO_FIXTURE_CORRECTION.md)。
