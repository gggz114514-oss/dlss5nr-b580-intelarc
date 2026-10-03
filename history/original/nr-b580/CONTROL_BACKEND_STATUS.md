# 固定参考运行库的控制参数测量

自动遮罩、皮肤结构与局部结构联动的新增后端及验收范围见
[AUTO_MASK_BACKEND_STATUS.md](AUTO_MASK_BACKEND_STATUS.md)。显式控制遮罩及样式联动
已完成40帧B580完整时序验收，见[CONTROL_MASK_BACKEND_STATUS.md](CONTROL_MASK_BACKEND_STATUS.md)。
UI已有独立纹理参考测量，仍需后端实现。以下保留原有控制证据。

2026-09-08。默认控制之外，B580现已通过256×256固定样式1/2、强度0/0.5/1/2、局部
结构/色调0/2及六组连续取值/组合参数的完整时序验收。运行库仍固定为SF-v2，SHA-256为
`6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927`。

## 原生对照

复用已验收的256×256四帧、fine运动、零深度、重置[1,0,0,1]。十组设置分别运行
未插桩和插桩参考，每组四帧。输入RGB、运动和深度与既有默认参考完全相同；
全部传输文件哈希及每组插桩/未插桩RGB输出一致性已验证。参数在feature创建时设置。

设置包括默认值、强度0/0.5/2、局部结构0/2、局部色调0/2、样式1/2。
设置、实际传入元数据、内核名及输出对照在
`reference/control-sweep-v1-manifest.json`和`reference/control-sweep-v1-analysis.json`。
默认组四帧与先前已验证的fine参考全字节相同。所有参考端实验均在笔记本E盘。

## 前端控制通道

独立重放原生pre，另用已有front插桩内核保存完整320×320×16的half特征。
十组重置帧的原生pre skip/down均匹配NGX缓冲区；完整front符合下列关系，
十组共32,768,000字节全部一致：

| 前端通道 | 本轮实测关系 |
|---|---|
| 10 | half(style / 128)，样式0/1/2 |
| 11 | half(local tone)，0/1/2 |
| 12 | half(local structure)，0/1/2 |
| 其他通道 | 与相同种子及输入的默认重置front一致 |

强度0/0.5/1/2不改变重置front。证据：
`reference/control-front-constants-v1-validation.json`。这些是已测离散设置，尚未
证明全部连续取值、越界取值及运行中切换行为。

## 强度改变显示输出，内部历史另行保留

强度0和0.5使用`cc_tinlayout_fused_post_block_swin_1h_32_fp8_simple_blend_full_rect`
（312字节参数），默认和强度2使用普通post（184字节参数）。强度2的四帧输出与
默认1完全相同；强度0的四帧原始RGB输出与当前输入完全相同。这不代表已确认
所有越界强度的限制规则。

将低强度的最终显示输出当作下一帧历史重放，pre存在大量差异；失败记录保留在
`reference/results/control-front-replay-v1/validation.json`。改用默认强度流的
未混合历史后，强度0和0.5、各两张历史帧的原始pre skip/down全部逐字节匹配：
`reference/results/control-front-replay-v2/validation.json`中的四个intensity案例。

原生强度0/0.5/2的四帧pre skip/down，以及强度2可取得的post decoder输入，
共28个捕获区域均与默认流完全一致：`reference/intensity-history-v1-analysis.json`。
v5未保存simple-blend变体的before-post arena，报告明确标记缺失，未冒充已验收。

随后直接重放原生simple-blend内核，将私有/显示表面分别设为HALF和FLOAT。强度
0/0.5、各三帧，六组私有结果匹配普通NR，显示结果匹配实际NGX输出。恢复的顺序为：

```text
private_history = half_rz(raw_private)
display = half_rz(sat(fma32(raw_private - original_half, intensity, original_half)))
```

重置帧的raw_private先饱和；时序raw_private保留历史重采样造成的范围外值。
混合用尚未舍入为half的FP32值。若提前使用half私有历史混合，强度0.5三帧分别有
2971、2534、2706个分量不同。六组原生FLOAT显示与上述CPU公式逐字节相同，
HALF私有/显示也全部相同：`reference/results/intensity-post-replay-v1/validation.json`。
其输入使用已证明相同的默认流decoder/历史，首184字节参数的所有非指针字段、
arena相对指针及实际NGX显示输出均作了核对；没有把缺失的原生arena当作已捕获。

`IntensityMotionNR`现已通过强度0、0.5、1、2各四帧完整模型验收。十六帧显示RGB
和私有历史全字节一致，重置及返回张量修改隔离通过，约19.6–20.7秒/帧。
报告：`reference/results/{intensity0,intensity05,default,intensity2}-temporal-rgb-b580-v1/validation.json`。
已测强度1/2沿普通输出路径，不额外饱和或混合时序范围外值。
用法见[CONTROL_BACKEND_USAGE.md](CONTROL_BACKEND_USAGE.md)。

## 样式与剩余工作

样式1/2有额外copy/clear调用，pre序号分别为13/171/329/487，不能沿用默认的
12+156×frame。分析器已按实际内核名找到四组pre/post；此前固定序号引起的解析
失败属于分析器问题，原生参考运行本身成功。

使用最终显示输出作为历史的失败假设仍保留在v2报告，未改写为通过。原生普通post
的未处理输出才是样式流的私有历史。独立重置post加两张历史帧的pre/post重放，
六组pre skip/down全部匹配；原生`cg2r_post_process_kernel`连接该私有输出后，
两种样式各三张显示RGB也全部匹配NGX。证据在`style-post-reset-v1`、
`style-temporal-chain-v1`和`style-peripheral-replay-v1`的结果目录。

外围调色已经移植至`backend/nr_backend/style.py`。前段逐通道曲线的独立性经过
寄存器依赖检查，只枚举可分离区域；两轮RGB/HSL转换仍使用真实跨通道计算。
原生近似指令通过完整标量域资产恢复，无图像或模型中间特征缓存：

- 每种样式15361个half `[0,1]`输入的RGB曲线，各184332字节。
- 倒数全部8388608个归一化尾数，32MiB；独立指数及已采集操作数共2217629个值逐字节一致。
- 饱和度LG2/EX2往返覆盖 `[2^-126,1]`的全部1056964609个正正规FP32输入，
  无损int8 ULP差值表1056964609字节，差值范围-2..89，无例外记录。

资产在`model-assets/style-sm89-v1`，加载核对固定SHA-256。稀疏操作数查询仅用于
早期诊断，已从可执行后端排除。CPU完整域版本与六张参考图的FP32/FP16输出全字节相同。
B580外围六张原生私有图、四张独立合成边界/随机颜色测试也全部全字节相同，覆盖
灰色、色相分支、相邻half、负零及有限范围外输入。报告：
`reference/results/style-post-xpu-v2/validation.json`和`style-post-stress-xpu-v1/validation.json`。

`StyledMotionNR`现已用原权重完成样式1/2各四帧测试，显示RGB及私有历史均全字节相同，
只保留B580自己的私有历史；返回显示张量的修改不影响后续历史，重置复现通过。
报告：`reference/results/style1-temporal-rgb-b580-v2/validation.json`及
`style2-temporal-rgb-b580-v1/validation.json`。样式1的v1在首帧匹配后因测试脚本于
InferenceMode外修改张量而停止，已修正测试上下文，未更改模型数学。
接口、性能和尺寸限制见[STYLE_BACKEND_USAGE.md](STYLE_BACKEND_USAGE.md)。

## 局部控制和统一组合接口

局部色调0/2、局部结构0/2各四帧完整模型均逐字节匹配原生显示输出和私有历史：
`reference/results/{tone0,tone2,structure0,structure2}-temporal-rgb-b580-v1/validation.json`。
本轮约19.5–21.3秒/帧，峰值已分配显存约746MB。

随后新增强度0.25/0.75、局部色调0.5配结构1.5、样式1配强度0.5、样式2配强度0，
以及样式2/强度0.5/色调2/结构0，共六组。每组未插桩/插桩原生四帧均一致，全部
传输哈希和输入一致性通过：`reference/combined-control-sweep-v1-analysis.json`。
样式+低强度沿普通post输出私有half，再在`cg2r_post_process_kernel`末尾混合。
独立原生post/调色连续重放三组样式各三帧，九张显示结果均匹配NGX，为私有历史
提供独立参考：`reference/results/combined-private-chain-v1/validation.json`。

`ControlledMotionNR`已通过以上六组各四帧：新增24帧显示RGB和私有历史全部全字节
一致，并通过重置复现、返回张量修改隔离和非法尺寸不推进状态检查。结果在
`reference/results/{intensity025,intensity075,local05_15,style1i05,style2i0,style2i05t2s0}-combined-temporal-rgb-b580-v1/validation.json`。
调色接口增加强度参数后，默认强度的六个外围兼容案例也全部匹配：
`reference/results/style-post-xpu-v3/validation.json`。
统一接口用法和已测参数表见[CONTROL_BACKEND_USAGE.md](CONTROL_BACKEND_USAGE.md)。

运行中控制变化另已完成六组24帧验收，包括隐式重置与动态调色系数恢复，见
[LIVE_CONTROL_STATUS.md](LIVE_CONTROL_STATUS.md)。辅助参数与遮罩绑定切换又通过28帧，
见[LIVE_AUXILIARY_STATUS.md](LIVE_AUXILIARY_STATUS.md)。后续需补齐更大控制尺寸、
深度交互与UI等控制，现有静态资料
不能替代本机动态验证。
