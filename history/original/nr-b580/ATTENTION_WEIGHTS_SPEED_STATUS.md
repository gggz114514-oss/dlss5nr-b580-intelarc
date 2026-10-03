# 注意力求和与权重融合（2026-09-09）

已合入共用后端的显式triton/XPU路径。完整1080p成对实验从2.297754833秒降到
2.189442308秒/帧，速度比1.049470372，约快4.95%。保持原模型、权重和
每步舍入；未启用INT8/XMX，完整迁移和生产工具接口仍未完成。

## 保留的计算与验证

64项按原顺序先相邻行配对，逐次半精度合并，再按列分组逐次半精度合并。
权重入口继续执行原钳位、FP32倒数转半精度、半精度乘法及E4M3编码/解码边界。
原NaN传播次序、符号及有效载荷保留，原FP8边界的NaN处理也保留；没有改用普通
reduce、softmax或新量化模型。复用已验证的_nan_left和_round_fp8_half辅助函数。

ViT直接使用求和入口。其跨64-key块顺序、分子矩阵运算和最后的填充扣除源码均未改变。
不能提前屏蔽填充项，不能把多块一次求和后才舍入。

- 独立倒数检查：全部65536种half编码与当前XPU完全相同，有限输入还与FP64数学值
  转half相同。使用tl.div_rn(1,x)，不据此声称所有硬件原生倒数指令的一般位等价。
- 候选72项，每项4配置通过：有限half常量行、16个位置的65536编码、混合位模式、
  全NaN/无穷通道配对、随机尺度、dtype/布局/空输入、Swin/ViT指数输出和大张量。
  有限求和另与独立FP64逐步half舍入比较。配置先按合成大窗口计时选出，随后才跑整帧。
- 正式包CPU20/GPU36通过公共入口、原reference路径、CPU显式triton旧行为、按需导入、
  输入所有权和非法形状检查。这里CPU显式triton原本可回退，不错误添加拒绝条件。
- 两套独立模型各13帧，逐帧交替先后；第1–12帧均值保留候选第1运动帧较慢的样本，
  包含最终reset，排除首帧JIT。包括模型调用和同步，排除上传、IO和比对。
  26帧646963200个RGB字节及私有历史均独立重读，与重复验证的4060原生捕获一致。
- 本次缓存条件下首帧候选129.77秒、基线3.93秒，
  含即时编译。温态改善不代表首次启动开销消失。
- 正式55帧：256/480/1080运动各13，512/480/1080/2559控制/UI各4；主目录另13帧480p。
  RGB、私有历史、重置、非法参数保护及调用方修改保护均通过，保存数组独立重读通过。

主目录480p后续帧0.654175秒；正式目录1080p 2.194872秒，
2559控制/UI 3.570476秒。这些回归均值不是与旧版交错运行的性能对照，
不用于宣称小尺寸提速。当前仍非实时。

1080p每帧融合权重621次，独立ViT行求和80次；原FP8调用2447→1826，C32归一化678、
Swin/ViT指数621/8、dense1918、batched1258、cubic607、half_fma6全部保持原计数。
原实验基线的701次求和包括权重内部621次，候选直接求和计数80不包含已融合的内部求和。

## 当前源码与证据

主目录35个py与reference/experimental/attention-weights-stage-v1/nr_backend逐字节相同。
仅attention.py新增两个显式分派，新增triton_attention_weights.py，其他33个旧文件不变。
原reference函数体AST保留。求和默认32行/4warps，权重16行/1warp；矩阵仍4×32/1warp。
前一版34文件与668次检查点保存于reference/experimental/revisions/before-main-attention-weights-v1。
代码E盘，数据与缓存D盘，参考笔记本本轮没有操作。

| 文件 | SHA256 |
| --- | --- |
| attention.py | dbdf6440272e85d0f24d63a859f47831bb353ed8a9364ca74c1b796fb2ad729e |
| triton_attention_weights.py | 2bff6b4fe6b80425887c655ae695af032ee82fbd3ac3d848a14beff433e11213 |
| backend-checkpoint.json | a4614658ed381e3c1895cee84d2754dbb07aaa0680a40508563ee163998ea1fb |
| attention-weights-backend-promotion-v1.json | 8296481884052b23a0a0abd37ff77d8072abc044a11e79191856023425db57f1 |

数据根目录D:/Codex-NR-Experiments/nr-b580/reference，原语在experimental/attention-weights-v1，
成对实验在results/attention-weights-1080-v1，正式包及主目录审计在experimental/attention-weights-stage-v1。
正式本版68次加入检查点，共736次优化比较（12个源码版本）、372次历史参考、26次存档
完整INT32图形比较。成对26帧及新剖析2帧另存，未加入正式736次。
比较包含重置复现，不能作为736个独立场景，也不证明所有控制组合或当前版图形接口。

## 新版剩余耗时诊断

当前35文件主目录执行重置帧及下一张真实运动帧，第二帧插入GPU事件；两帧RGB及历史
独立重读通过。插桩帧耗时4.738409秒。下面是包含主机分派空隙的事件区间，
嵌套类别可能重叠，不能相加计算占比，也不能当作纯GPU硬件周期或正常帧延迟。

| 类别 | 调用数 | 包含式事件区间总毫秒 |
| --- | ---: | ---: |
| dense | 1918 | 1442.913 |
| batched | 1258 | 377.907 |
| fp8 | 1826 | 204.559 |
| normalize_c32 | 678 | 115.025 |
| normalize_attention_weights | 621 | 99.756 |
| score_exponential | 621 | 89.122 |
| cubic_fp8 | 607 | 52.768 |
| half_fma | 6 | 3.804 |
| attention_row_sum64 | 80 | 1.629 |
| vit_exponential | 8 | 1.238 |

原始记录在profiles/weights-pointwise-1080-v1/results.json与saved-audit.json；不包含新矩阵
操作数捕获。下一步依据当前实测热点继续优化矩阵/数据搬运和有效XMX覆盖，不把旧INT8
低命中率试验重复当作新的加速结果。原共享指数、逐乘积截断和K16舍入边界仍须保留。
