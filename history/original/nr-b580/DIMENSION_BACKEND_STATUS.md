# 尺寸扩展与时序参考状态

2026-09-08。固定社区SF-v2、SDR、默认控制参数的正确性路径。尺寸一律保留，没有用256缩放或分块拼接代替完整模型。

| 输入 W×H | 原生内部 H×W | ViT位置数 | 当前证据 |
|---|---|---|---|
| 256×256 | 320×320 | 64 | 六份最终RGB零差异，用户视觉验收通过 |
| 512×512 | 512×576 | 96 | 完整RGB零差异，用户回复“画面正常，继续” |
| 864×480 | 512×896 | 128 | 完整人物帧RGB零差异；FMA修正后回归仍完全相同 |
| 1920×1080 | 1152×1920 | 640 | 完整RGB 24,883,200字节零差异，v3通过 |
| 2559×1439 | 1472×2560 | 960 | 完整 Apex RGB 44,188,812 字节零差异，v4通过 |

不能从这些采样尺寸推导任意尺寸padding公式。目前执行器保留明确尺寸表；未对齐尺寸仅用于研究。

512的原始FP32 RGB共3,145,728字节全部相同，耗时29.60秒。96位置注意力的八个块、CPU/B580共112个中间缓冲区和两次输出重排全相同。扩大尺寸后重新验证的256赛车(seed0xffffffff)仍完全相同。

864×480共4,976,640字节RGB全部相同，耗时40.01秒，峰值张量分配1,742,520,832字节。原生参考与采集参考的输入、输出、深度、运动文件均已校验。

1080p v3最终RGB逐字节一致，耗时176.46秒，峰值张量分配4,998,230,528字节；864×480修正后回归耗时41.48秒，峰值1,452,374,528字节。用户已明确完全一致时无需再次肉眼审核。

1080p首轮的95,156个RGB分量差异已定位到decoder call156归一化的一个FP8字节。此前完整编码器、所有skip、八个ViT和decoder110–155均完全一致。原生MLP与QKV半精度寄存器也全部一致；第一次错误来自半精度融合乘加被FP32加法双重舍入。通用修正、独立整数校验和抓取证据见 [HALF_FMA_STATUS.md](HALF_FMA_STATUS.md)。保留v1失败数值和v2显存不足记录，v3才是通过的整帧输出。

长序列注意力将临时乘积按头和查询行分批，保持每个输出的K累加顺序，避免按全部序列一次性生成巨大的乘积张量。补齐的键先参与半精度求和，再按原生规则扣除exp(0)贡献；提前屏蔽会改变结果。

Apex v1–v3先后在pre/post出现Level Zero资源不足，未产生有效RGB，失败记录保留。
v4将C32 MLP及QKV/归一化按独立像素分批，并将窗口注意力按独立窗口分批，
保持全模型的空间连接、窗口边界和累加顺序。最终44,188,812个RGB字节完全一致，
耗时304.47秒，峰值张量分配4,489,906,176字节。结果SHA-256：
`5fc16dd933c1598cd4cffae7de75034384fe6189abeb37169400b9ef117fa873`。
这解决了本次Apex全帧执行的资源错误，性能仍未达到实时要求。

分批改动后，六份256、512、完整480p/1080p及零运动四帧均重新通过字节比较。
1080p v4耗时182.17秒，峰值2,849,116,672字节；480p v4耗时45.26秒，
峰值1,012,855,296字节。此为正确性/显存改进，不是实时性能优化。

## 证据入口

- `reference/results/reset-executor-512-v1/validation.json` 与 `review.json`
- `reference/results/vit-dimension512-v1/vit-validation.json`
- `reference/results/dimension512-v1/256-regression.json`
- `reference/results/reset-full-864x480-v1/validation.json`
- `reference/results/reset-full-1920x1080-v1/validation.json`
- `reference/results/reset-full-1920x1080-v3/validation.json`
- `reference/results/reset-full-864x480-v3/validation.json`
- `reference/results/reset-full-2559x1439-v4/validation.json`
- `reference/results/reset-full-1920x1080-v4/validation.json`
- `reference/results/reset-full-864x480-v4/validation.json`
- `reference/results/half-fma-rgb-regression-v2/validation.json`
- `reference/results/dimension1920-v1/encoder-validation.json`
- `reference/results/dimension1920-v1/skip-validation.json`
- `reference/results/vit-dimension1920-v2/vit-validation.json`

## 四帧历史状态参考

笔记本E盘上的独立参考程序对256赛车裁剪执行reset=[1,0,0,1]。四帧输入对应水平裁剪位置975、974、973、975，深度和运动均为零，以隔离模型自身历史状态。不是完整运动补偿视频验证。

`reference/temporal-capture-v1-analysis.json`记录所有文件哈希及624次原生调用。四帧插桩与未插桩输出完全一致；中间两帧与逐帧reset输出明显不同；第四帧reset后与第一帧逐字节一致。原生pre/post在中间两帧启用历史RGB和运动纹理，seed依次0、1、2、0。B580已独立实现零运动历史输入、融合与重置，四帧最终RGB均完全一致，见 [TEMPORAL_BACKEND_STATUS.md](TEMPORAL_BACKEND_STATUS.md)。非零运动及长视频仍在开发。
