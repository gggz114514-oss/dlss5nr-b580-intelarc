# Magpie DLSSNR 降开销实现核查（2026-09-09）

用户链接对应[BV1EbtL62EGH](https://www.bilibili.com/video/BV1EbtL62EGH/)，公开接口返回标题“大力喜鹊0.6.0 DLSS5降低40-70%开销 深度估算降低25%”。已取得标题、简介及作者信息，没有观看/复现整段视频；不能把标题的幅度视为B580实测结果。

项目为SAOG0721/Magpie实验分支。核查固定v0.6.6提交 `9824d758b162ad3c5b5acc81e2e14c83f138e13d`。只下载7份文本（约174KB），没有安装或运行Magpie及其DLL。Git树中blob SHA-1和本地SHA-256均核对，原始文本保留在reference/community/magpie-review-v1。

0.6.0新增25%–100%输入分辨率选择，颜色、运动/深度按小尺寸送入NR，将输出差值重建回原图；文档提示过低分辨率会降低质量，建议至少480p。TensorRT深度估算是另一项可选优化，说明中的5070Ti/2K结果不适用于我们的B580后端。[0.6.0说明](https://github.com/SAOG0721/Magpie/blob/9824d758b162ad3c5b5acc81e2e14c83f138e13d/docs/RELEASE_NOTES_v0.6.0-experimental.md)

中性残差参数下可概括为：

`输出 = clamp(原始大图 + 放大(NR(缩小图) − 缩小图))`

当前实现的要点：

- 源码1893–1899行分别按百分比缩放宽和高。
- 185–223行颜色分两遍Lanczos-2 AA缩小；397–404行在低分辨率上计算处理差值，保留有符号FP16残差。
- 432–507行用Catmull-Rom分两遍放大残差，加回未缩小的原图并保留原alpha。
- 248–289行引导图降采样并按宽/高比例缩放运动向量；2292–2298行仍调用NGX NR。

这条代码没有把NR网络重新量化为INT8，改动在输入与输出周围。[固定源码](https://github.com/SAOG0721/Magpie/blob/9824d758b162ad3c5b5acc81e2e14c83f138e13d/src/Magpie.Core/DLSSNRFilter.cpp#L397)

Lanczos-2 AA/Catmull-Rom为0.6.5的更新，不冒充0.6.0使用了相同滤波器。0.6.5还提供跨SR/NR/FG光流共享并移除了深度估算。[0.6.5发布说明](https://github.com/SAOG0721/Magpie/releases/tag/v0.6.5-experimental)

对本项目的判断：低分辨率残差可以与B580的XMX/INT8后端叠加，原图细节仍在最终合成中，NR新增的高频细节和时序稳定性可能下降。我们的基准已使用预备的运动输入、未运行深度估算，因此后两项不能重复计入当前NR加速。方法不解决NR模型本身在B580上的矩阵执行，后端开发仍是主线。

像素数示例是数学推算：宽高均为63%时约剩39.69%像素，少60.31%。实际耗时还受填充、网络结构、采样及同步开销影响，不等于保证节省60%或固定提升2.5倍。本轮没有执行B580低分辨率NR画质/速度验证。

用户已允许高画面相似度换明显速度提升，见QUALITY_PERFORMANCE_ACCEPTANCE.md。后续以完整图像对照、运动片段、真实推理耗时和用户肉眼审查验证；优先保留原模型结构。

核查凭据位于D:\Codex-NR-Experiments\nr-b580\reference\community\magpie-review-v1：pin.json、tree.json、sources.json、v060-notes-source.json、bilibili-public-metadata.json。源许可证GPLv3，仅研究实现；若以后直接复用源码需保留相应许可。


---

## 2026-09-18 补充：与本项目实现的逐式对照

用户问「这个实现方法是不是和大力喜鹊不一样？」。已把**本地保留的原始源码**
（`reference/community/magpie-review-v1/src/Magpie.Core/DLSSNRFilter.cpp` +
`docs/experimental/todos/20260905-v0.6.5-r5-dlssnr-resampling-TODO.md`，固定提交 `9824d758b1…`）
与本项目 `nr-b580-int8/experimental/residual_scale_v1.py` **逐式对过**。

**结论：算法是同一套，架构不是。** 两个重采样核（Lanczos-2 AA 降采样、Catmull-Rom 4+4 残差上采样）
在坐标约定、support、核公式、tap 范围、归一化、clamp-to-edge 边界上**逐式相同**。
不同的是 9 条结构项，其中**根因是「方形画布」**：Magpie 的 NGX 能吃任意长宽比，
本项目的 NR 内核只有固定方形契约（`CallGuard` 硬钉 256×256、`ResidualScale` 只收 256/512），
于是必须把任意长宽比塞进正方形 ⇒ 才有了「有效区 + 复制边缘填充 + 运动在有效区外清零 + top 偏移」
这一整套 Magpie 里不存在的概念，也才有了「只有 256/512/原尺寸三个落点」而非 25%–100% 自由滑块。

**完整对照表（9 条结构差异 + 7 行公式对照）见
`reference/ressweep_v1/RESULT.md` §4.1**（该轮主题是输入分辨率滑块，§4 正是讲这些约束从哪来）。

**★ 由此推出的两条量化结论**：`ressweep_v1/RESULT.md` **§4.2** —— 256 画布只有 **55.5% 是有效像素**
（上下各 57 行是复制边缘 + 运动在区外清零）⇒ **与 Magpie 最实质的差距是"有效像素占画布的比例"（55.5% vs 100%），不是"分辨率高低"**；**§5** —— 路线裁决：**预编译常见档位（B 路）为现行方案**，
运行时任意尺寸（C 路）挂起。

⚠️ 边界：Magpie 那侧是**源码阅读 + 其自带 TODO 文档**，没有跑过它的代码
（`review-v1.json`：`b580_performance_measured=false`、`remote_code_executed=false`）；
本项目那侧是实际代码。**「逐式相同」是公式层面的对照，不是位级等价声明。**
