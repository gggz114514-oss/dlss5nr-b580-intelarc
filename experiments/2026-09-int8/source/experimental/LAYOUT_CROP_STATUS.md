# 采用布局与依赖裁剪组合（2026-09-10）

新的快速实验基线入口为 `nr256_selected_stack_v4.Stack`，直接导出已验证的
`layout_crop_stack_v1.Stack`，没有额外包装或构造变化。旧v1/v2/v3工厂及已执行
文件保持冻结。精确分支、公开仓库未改动，没有发布或完成OptiScaler等生产集成。

| 本轮完整调用 | ShortFP8基线 | 组合版 | 耗时减少 |
| --- | ---: | ---: | ---: |
| NR256连续运行 | 11.745086ms | 11.269206ms | 4.05% |
| NR256重置 | 11.517095ms | 11.077679ms | 3.82% |
| 1080p残差＋NR256连续运行 | 12.461203ms | 11.941503ms | 4.17% |
| 1080p残差＋NR256含重置 | 12.400846ms | 11.896700ms | 4.07% |

每项均为各自同次交替测试、三轮均改善。包括模型调用、历史提交和GPU完成等待，
残差测试另含缩放及合成；不包括解码、光流估算、上传、显示、JIT和游戏争用。
390帧验证不是独立性能基准。不能把以上时间倒推成游戏帧率，或混合不同轮差值。
固定4060 NR256历史参考3.566807ms仍明显更快，本次属于约半毫秒的稳定改进。

三项具体改动：C512投影直接消费注意力窗口布局、ViT投影直接消费head布局、
post只计算目标RGB及其完整注意力依赖窗口。保持当前FP16快速版数学和FP8规则。
作用域仅在GPU图构建时安装，正常重放不切这些补丁；进度回退沿用原有完整路径。
C512未量化full仍供末端池化，公共边界接口不变；未改C512 FFWD内部布局。

验证已通过：720次NR256完整输出、78次1080p残差及low NR全字节比较；390帧
独立连续历史的每个low NR字节与完整1080p合成哈希匹配已肉眼接受的视频。
末尾reset复现、seed、held输出、调用者修改输出不污染历史、输入/LUT不变、
非法motion和替换表拒绝、LUT版本保护及进度回退均通过。没有新画质变化、视频
编码或整帧缓存，因此按用户规则复用原肉眼验收。

每body的Triton调用683→651，独立FP8196→180。组合版6次body构建，C51296次、
ViT48次、post6次；新C512和ViT投影以及58项ShortFP8编译资源均零spill。
运行报告里的candidate_promoted=false是当时状态，本文在完整验证后记录采用。
本批字节一致性相对已审核快速版，不将其冒充精确分支对NVIDIA的保证。

主助手在用户通知后核对Luna终态、suite两子阶段的真实returncode与报告摘要，
复核父6项、残差487项、长序列489项源码及两子报告各269项exact_gate哈希。

数据根目录 D:/Codex-NR-Experiments/nr-b580/reference：

| 报告 | SHA256 |
| --- | --- |
| results/layout-crop-native-parity-v1/validation.json | e54f31291b1b628d2d988bf19bbc8fe43cdb8f0c19d80def21ec7f2418973109 |
| results/layout-crop-validation-suite-v1/validation.json | cb48355400195d65fa9aeb00fff0dea1415e1d05f7634f2bca2b2498add4c29c |
| results/layout-crop-residual256-v1/validation.json | b0d5b7e984ca530d36459b3c20651b3989f37719aae57bcee3bb6eb64e265fe8 |
| results/layout-crop-long1080-v1/validation.json | 193e041c029d04f53f6d31fe23ed930f160effb8e9e48690e9163a72d1ebdadd |

后续按用户顺序试保留FP8值/规则、用INT8执行可行部分，先分析真实操作数是否
能无损表达以及整数分解代价。仍无充分收益时再重新量化连续计算段，另做画质审核。
