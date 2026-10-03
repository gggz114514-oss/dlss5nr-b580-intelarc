# 全尺寸时序与归一化纹理坐标

2026-09-08：继续固定SF-v2、SDR、零深度与默认控制，将有状态后端扩展至用户
864×480原尺寸视频。零运动、fine运动各四帧的完整RGB时序已全部逐字节通过；
1080p独立边界与256/512旧尺寸兼容回归也已全部通过。1080p零运动和fine运动各
四帧完整时序均已逐字节通过。

## 原生参考

`reference/inputs/temporal-full-864x480-v1` 为用户480p视频的第100、101、102帧，
加回到第100帧的重置。无裁剪或缩放，RGB解码与先前参考一致。重置为[1,0,0,1]。
零运动和fine合成运动各有未插桩参考与v6插桩参考，原始输入、运动、深度和RGBA输出
逐字节相同。报告：`reference/temporal-full-864x480-{zero,fine}-capture-analysis.json`。

首次v5插桩因32MiB arena上限未保存中间缓冲区，保留了
`4060-temporal-full-864x480-zero-trace-v1.json` 的失败状态；模型推理本身正常。
改用已验证的v6采集器后，每组取得10个arena，完整参考为trace-v2。
参考端全部实验保存在笔记本E盘，原NR运行库和模型内核没有改动。

## 已恢复的额外数值规则

非2的整数次幂尺寸需要保留原生归一化步骤：像素中心乘原生尺寸倒数；运动乘宿主
FP32比例后融合相加；通过融合乘加计算cubic整数基点和局部分数。合并采样点经过
范围截断、模型归一化、区域映射及纹理归一化。提前约掉这些步骤会改变舍入。

独立探针枚举整数尺寸1至4096的原生 `MUFU.RCP`，16KiB标量表，不含图像或模型
特征。519个值与正确舍入FP32倒数相差1 ULP，例如1439。资产在
`model-assets/reciprocal-dimensions-sm89-v1`，加载检查固定SHA-256。

纹理单元的本次测量规则：先将正归一化坐标截断为21位小数定点数，然后与尺寸进行
整数乘法，最近舍入到1/256插值权重，正中点向上。像素中心偏移及clamp也按整数
系数处理。对于归一化横坐标u、宽度W，代码使用：

```text
n = floor(u * 2^21)
coefficient = clamp(((n * W + 4096) >> 13) - 128, 0, (W - 1) * 256)
```

这是本次SM89纹理配置的实验结果，不把它声明为任意CUDA纹理格式的通用规范。
独立0/1棋盘纹理直接读出插值权重，两个轴各229,376个位置均匹配21位截断规则；
16至24位固定小数/浮点有效位及多种舍入候选的对照在
`reference/results/texture-full-basis-v1/coordinate-precision-probe-v1.json`。
送入TEX之前的10个坐标值及6个cubic权重也逐字节一致。

只有非零权重的texel参与指数对齐和符号判断。所有有效输入均为负零时保留负零。
独立负零探针五个采样点共3,440,640个值在B580全部逐字节匹配：
`reference/results/texture-signed-zero-v1/xpu-validation.json`。

## 独立边界验收

- 四次原始pre内核在自有CUDA数组上重放，skip/down均与NGX arena一致。
- 四组真实历史图像、每组五个原生采样点，共13,762,560个FP32值全部一致：
  `reference/results/motion-full-864x480-replay-v1/texture-validation-v2.json`。
- B580的归一化采样连接完整pre/post后，四组pre skip、pre down和最终post RGB
  均逐字节一致：`reference/results/temporal-full-864x480-boundaries-xpu-v2/validation.json`。

边界报告v1使用了错误的pre-down解码布局，且在XPU上再次转FP8时改变了部分负零，
因此报告失败；v2改为已有验收使用的split-C16布局及CPU序列化已量化值。后端数学
未为修复该测试问题而改动。后续完整RGB验收不依赖这个FP8序列化步骤。

`MotionNR` 已接入864×480的归一化历史路径；256/512方形兼容路径保留。完整会话
只携带B580自己计算的历史，原生特征仅用于上述隔离诊断。两组共8帧，每帧4,976,640
个最终RGB32F字节全部一致，约43.4–44.9秒/帧，XPU已分配显存峰值约1.035GB。
报告：`reference/results/temporal-full-864x480-{zero,fine}-rgb-b580-v1/validation.json`。
包含重置后首帧复现、返回张量修改后的私有历史隔离、尺寸切换失败不推进状态。

1080p实际使用用户视频三个重复首帧加重置；旧相邻帧声明因时间戳问题撤回，见
[VIDEO_FIXTURE_CORRECTION.md](VIDEO_FIXTURE_CORRECTION.md)。两组原生未插桩/插桩参考均已完成
哈希和输入/输出逐字节校验。四组独立pre skip（每组70,778,880字节）、pre down
（17,694,720字节）和post RGB（24,883,200字节）全部逐字节匹配：
`reference/results/temporal-full-1920x1080-boundaries-xpu-v1/validation.json`。
它们使用原生历史/中间特征，不能替代只携带B580自身历史的完整模型验收。
256 fine v7、512 fine v2两组各四帧完整模型兼容回归全部通过，源码哈希已记录。
1080p零运动/fine两组完整八帧也已全字节通过（约180.8–183.1秒/帧，峰值约2.93GB），包含
历史累积、返回张量修改隔离、尺寸切换失败不推进状态和重置复现。
报告：`reference/results/temporal-full-1920x1080-{zero,fine}-rgb-b580-v1/validation.json`。
仍需完成更长全尺寸视频、
真实光流、深度/控制参数、XMX/INT8及OptiScaler/现有工具集成的验收。

2559×1439已使用用户原始Apex截图重复四帧，加入零运动/fine合成运动并测试重置。
这是静态截图时序探针，不是真实游戏录像。原生两组插桩/未插桩输入与输出全部匹配。
v6跟踪器受2GiB总捕获预算限制，每组保存六个arena；第1张历史帧pre/post齐全，
第2张历史帧的arena未保存。边界验收仅选第1张历史帧，完整模型验收仍比较全部四帧。
两组第1张历史帧的B580 pre skip（120,586,240字节）、pre down（30,146,560字节）、
post RGB（44,188,812字节）均已全字节匹配：
`reference/results/temporal-full-2559x1439-boundaries-xpu-v1/validation.json`。
只携带B580自身历史的完整模型四帧×两组现已全部逐字节通过。每帧44,188,812个
RGB32F字节，八帧共353,510,496字节，无字节或数值差异。耗时302.7–305.3秒/帧，
峰值已分配显存4,629,120,000字节；重置复现、返回张量修改后的历史隔离、无效尺寸
切换不推进状态均通过。报告：
`reference/results/temporal-full-2559x1439-{zero,fine}-rgb-b580-v1/validation.json`。
本次尺寸扩展后汇总为96次完整RGB比较（含重复重置输入）；后续控制验收继续更新
`reference/backend-checkpoint.json`。以上字节相同的输出按用户要求免于肉眼复核。
