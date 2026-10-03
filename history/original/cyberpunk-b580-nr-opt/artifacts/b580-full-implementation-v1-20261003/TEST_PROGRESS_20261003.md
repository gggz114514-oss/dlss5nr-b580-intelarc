# 现役 720p 审计改写：测试进度汇总（2026-10-03）

本表汇总 Main 已核对的结果回执。基准是现役 C512＋K8／all6、去舍入、融合真实运动历史、图重放；已验收的游戏桥纹理复用／GPU 接力保持不变。18 组源码合入实验树不表示 18 组已独立通过 GPU 验收。114 个测试档包含版本、控制组与组合，不是 114 项独立优化。

以下速度均为离线完整 `FullsizeGameModes.process` 的配对结果，包含该处理调用的工作；桥与游戏渲染不在本次模型计时范围。不能据此直接宣称游戏 FPS，也不能把独立 raw graph 与完整 process 的差值当作独占 CPU／转换时间。现役游戏未换版，实机基础帧时间仍采用用户此前验收的 55–60 ms。

| 已完成项目 | 完整处理实测 | 结果与采用状态 | 回执 |
| --- | --- | --- | --- |
| 前端／解码／post 组合，标准档独立 BCCB 复测 | 45.410→43.610 ms，快 **1.800 ms（3.96%）** | 两端配对均快；候选输出有数值变化，待游戏画面与实机收益验收。尚未部署。组合覆盖多组源码，不证明各组独立收益。 | [FDP_REPEAT_MAIN_REVIEW.json](FDP_REPEAT_MAIN_REVIEW.json) |
| 同组合，自然档 | 52.820→44.694 ms，快 8.126 ms | 单轮三 cycle 均快，输出有限，平均绝对误差约 6.77e-5；独立复测与肉眼验收待做。不能加入标准档收益。 | [FDP_CONDITIONAL_MAIN_REVIEW.json](FDP_CONDITIONAL_MAIN_REVIEW.json) |
| 同组合，电影档 | 53.176→45.182 ms，快 7.994 ms | 单轮三 cycle 均快，输出有限，平均绝对误差约 6.93e-5；独立复测与肉眼验收待做。不能加入其他风格收益。 | [FDP_CONDITIONAL_MAIN_REVIEW.json](FDP_CONDITIONAL_MAIN_REVIEW.json) |
| 早期 C512／ViT 完整组合 | 46.005→54.067 ms，慢 8.062 ms | 未采用；随后已拆分定位。 | [C512_COMPLETE_MAIN_REVIEW.json](C512_COMPLETE_MAIN_REVIEW.json) |
| 原生 QKV 分段改写 | 45.916→47.540 ms，慢 1.624 ms | 实际 16 处执行，13 帧输出／历史逐字节相同；未采用。 | [R13_SPLIT_MODULE_MAIN_REVIEW.json](R13_SPLIT_MODULE_MAIN_REVIEW.json) |
| C512 完整 FFN | 46.239→47.357 ms，慢 1.118 ms | 输出／历史逐字节相同；未采用。 | [R13_SPLIT_MODULE_MAIN_REVIEW.json](R13_SPLIT_MODULE_MAIN_REVIEW.json) |
| C512 编码段 | 46.325→47.750 ms，慢 1.425 ms | 输出有限、有微小数值变化；未采用。 | [R15_FULL_MODULE_MAIN_REVIEW.json](R15_FULL_MODULE_MAIN_REVIEW.json) |
| 历史调用检查方案 | 46.269→46.418 ms，慢 0.150 ms | 三 cycle 有快有慢，无可重复净收益。此轮输出／历史逐字节相同；此前冷热不一致根因仍 UNKNOWN，未宣称修好。 | [R15_FULL_MODULE_MAIN_REVIEW.json](R15_FULL_MODULE_MAIN_REVIEW.json) |
| Swin 完整模块原方案 | 46.007→54.631 ms，慢 8.624 ms | 44 个实际模块已命中；独立图本体也慢 8.017 ms。C64 寄存器压力／小 tile、C256 重复注意力为修正方向；原方案未采用。 | [R15_FULL_MODULE_MAIN_REVIEW.json](R15_FULL_MODULE_MAIN_REVIEW.json) |
| ViT 完整 FFN 原方案 | 45.937→55.122 ms，慢 9.185 ms | 输出／历史逐字节相同；实际 INT8 矩阵路径不等于高效。原方案行量化重复和累积组织已返修；未采用。 | [R15_FULL_MODULE_MAIN_REVIEW.json](R15_FULL_MODULE_MAIN_REVIEW.json) |
| C512 重复检查迁移，两组同源码控制 | 均值分别快 0.427／0.381 ms | 每组均有反向 cycle；尚非确认收益。相对未采用模块的控制组，不能算作现役游戏新增提速。 | [R16_HOT_MAIN_REVIEW.json](R16_HOT_MAIN_REVIEW.json) |
| ViT 行量化一次、K64／四段累积修正版 | 45.649→46.542 ms，慢 0.893 ms | 13 帧输出／历史逐字节相同，真实 8 个 FFN 位点、零 spill、只读缓存通过；完整处理无收益，未采用。 | [R18_VIT_THROUGHPUT_MAIN_REVIEW.json](R18_VIT_THROUGHPUT_MAIN_REVIEW.json) |
| ViT 行量化一次、K64／完整累积修正版 | 45.453→46.456 ms，慢 1.003 ms | 同上。独立图本体仅微量快约 0.054 ms，不当作完整帧收益。未采用。 | [R18_VIT_THROUGHPUT_MAIN_REVIEW.json](R18_VIT_THROUGHPUT_MAIN_REVIEW.json) |
| 原生 HDR 纹理桥 raw2 小样 | prepare／verify 各两帧实际 pack/export | raw 字节相同、消费者同步及安全回收通过；未运行模型、游戏或 Present，不是整桥验收。 | [NATIVE_RAW2_MAIN_REVIEW.json](NATIVE_RAW2_MAIN_REVIEW.json) |

当前待测／未完成：

- Luna 串行任务 37–39：C64 串行 MLP、C256 注意力／投影拆分、两者组合。修正版源码与 CPU 检查就绪；截稿时 37 在 baseline precompile，尚无新速度结论。
- 前端／解码组合的自然、电影档独立复测及游戏画面验收；标准档已有离线独立复测，游戏尚未部署。
- 原生桥完整模型／低强度／实机资格尚未验证；raw2 小样通过不能代替这些检查。
- AMD 最新报告的六项新方案目前是代码与结构分析，未执行性能测试；Sol 的现役逐核对照仍在补充。六项收益是工程预算，不能相加为帧率承诺。
- 尚未完成审计全部选项、条件与组合的 GPU 验收。本表仅对有实际回执的版本作结论，不把编译失败、源码覆盖或 CPU 检查当作 GPU 性能结果。

所有新候选均未改写 G: 现役游戏运行时。已有确认收益进入候选队列，负收益版本保留证据用于返修；没有 1 ms 采纳门槛。

## r18 Swin serial/split 最终复核

37／38b／39均完成离线资格；完整process分别慢4.218／9.816／3.727 ms，raw body也变慢。13帧output/history有限但不与baseline逐字节相同（MAE约4.526e-5、最大0.004883）；各臂自身冷热逐字节一致、只读cache miss/write均0。三项未采用，未部署游戏；单轮资格不扩大为重复回归。详见R18_SWIN_THROUGHPUT_MAIN_REVIEW.json。
