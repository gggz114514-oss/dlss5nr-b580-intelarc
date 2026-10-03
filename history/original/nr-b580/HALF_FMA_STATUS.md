# 单次半精度融合乘加舍入修正

2026-09-08。固定SF-v2原生RTX4060参考，B580正确性实现。

1920×1080原始RGB输入完整执行后，最终24,883,200字节与原生参考完全相同。
两边RGB SHA-256均为`11d5cb9879ad69922ae20b57165f44d07d11206b8456cff085f2ea18bf657aed`。
证据：`reference/results/reset-full-1920x1080-v3/validation.json`。

## 根因证据

首个失败算子为decoder call156、C128、shift(y=4,x=0)。原生输出原有73个FP8字节不同。
输入call155与原生完全一致。独立CUDA重放的完整decoder最终arena也与NGX捕获一致。
以下抓取均只修改独立实验CUBIN，所有注入指令先通过nvdisasm往返验证；未修改原DLL。

| 捕获边界 | 比较范围 | 修正前差异 |
|---|---:|---:|
| MLP半精度输出 | 4,669,440数值 | 0 |
| QKV半精度投影 | 14,008,320数值 | 0 |
| 已归一化Q/K的完整K与前32个Q | 7,004,160字节 | 1 |
| 前32个查询的指数结果 | 4,669,440数值 | 25 |

唯一错误K为CTA(y=4,x=16)、head3、key16、channel27。其归一化分母原生356.5，旧实现356.25。
原生的一项FMA为`(-6.125)^2 + half(2^-20)`。平方37.515625恰是半精度中点；加数决定向上舍入为37.53125。
旧路径先在FP32相加，把极小加数丢失，再转half，得到37.5。CPU double tensor再转half的旧诊断同样经历FP32转换，未能发现这类错误。

`tensor_math.half_fma`以FP32精确表示两个half的乘积，使用TwoSum保留加法残差，在half中点处决定正确舍入。
它不要求B580支持FP64，不读取捕获特征，也不针对图像坐标修补输出。已用于归一化、cubic激活、指数近似和解码器融合。
大张量按独立元素分批，限制临时显存；每个元素的数学运算相同。

`reference/validate_half_fma.py`以Python任意精度整数在2^-48单位下独立计算乘积、加法和ties-to-even舍入。
CPU和B580各24,007个数值案例，以及各1,080,315个跨分批边界案例，全部零差异。记录在`reference/half-fma-validation.json`。

## 捕获目录

- `reference/results/decoder156-mlp-dump-v2`
- `reference/results/decoder156-qkv-dump-v1`
- `reference/results/decoder156-qknorm-dump-v1`
- `reference/results/decoder156-exp-dump-v1`
- `reference/results/decoder156-normpartials-dump-v1`
- `reference/results/decoder156-normsum-dump-v1`
- `reference/results/decoder156-normscale-dump-v1`
- `reference/results/decoder156-normhalf-dump-v1`

MLP/QKV布局通过其他90个CTA建立，未用失败窗口拟合；指数饱和导致的歧义列由相邻唯一列的lane公式恢复。
失败的`decoder156-mlp-dump-v1`不能用于数值验证；低寄存器临时地址版本v2才成功。

这完成了当前1080p单帧对齐。任意尺寸、时序、控制参数、实时性能与OptiScaler集成仍需后续验证。
