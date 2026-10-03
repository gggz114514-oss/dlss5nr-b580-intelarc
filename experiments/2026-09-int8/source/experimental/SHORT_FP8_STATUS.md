# 缩短等价 FP8 舍入（2026-09-10）

采用 `short_fp8_graph_v1.ShortFP8Graph`，叠加已选 WindowBlocks v3 与原 FP16 快速栈。
新的实验栈入口为 `nr256_selected_stack_v3.Stack`，构造方式与以下三份完整验证完全相同。
既有 v1/v2 工厂和所有已执行实验文件保留不变。精确分支、公开仓库均未修改，没有发布。
完整迁移与 4060 持平的目标仍未完成。

## 完整调用结果

同次测试交替运行基线与候选，单位 ms。均值来自三轮，三轮均改善。

| 范围 | WindowBlocks 基线 | ShortFP8 | 减少 |
| --- | ---: | ---: | ---: |
| NR256 连续帧完整调用 | 12.460293 | 11.563595 | 7.20% |
| NR256 reset 完整调用 | 12.179723 | 11.338717 | 6.90% |
| 1080p 缩放/残差合成加 NR256，连续帧 | 13.012945 | 12.032833 | 7.53% |
| 同上，包含 reset | 12.958815 | 11.984364 | 7.52% |

完整 NR 包括状态提交和完成等待，输入已驻留 GPU；残差测试再包括颜色/运动缩放和全尺寸合成。
不包括解码、运动估计、上传、显示、JIT 或游戏争用，不能据此宣称游戏中已有相应帧率。
4060 固定同一 NR256 输入的历史完整调用参考为 3.566807222 ms；当前约为其 3.24 倍。
这不是 480p 对比，仍不是架构突破。静态 body 筛选 10.197705→9.403620 ms（7.79%）
单独记录，不与完整调用混用，也不拿不同运行的差值当成外围分段耗时。

## 改动与正确性

原有 E4M3 FP8 舍入语义不变，包括饱和、符号零与既有 NaN 规范化规则。
normal 区间用整数位舍入；subnormal 区间利用 half 在偏置 2 附近的 1/512 间距，
通过加偏置、half 舍入、减偏置实现同一格点舍入。半精度输入在最小决定阈值附近的
间距为 2^-21，float32 在偏置附近为 2^-22，不会吞掉决定舍入方向的输入位。
v1 注释曾误写 2^-20；v2 已更正为 2^-21，计算实现没有变化。

GPU 穷举全部 65536 个 half 位模式，输出逐字节等于既有 rounder，包括所有 NaN 载荷。
静态 body 的正常输入和全零 front 输入均完整输出一致。候选只在自己的 NR256 图构建
期间替换 JIT 依赖，保留私有 globals/cache，之后恢复模块绑定。逐帧回放不再扫描或替换函数。
28 个 JIT 函数依赖发生变化，60 个模块绑定，完整测试实际产生 57 个变化后的编译特化。
全部 57 个在发射前取得零 spill；不将未修改的旧内核混入这一结论。

既有 FP8 来源证明和写别名保护仍启用；函数限定名、数据流合同不变。
每次 body 仍为 683 次 Triton 调用、196 次独立 FP8、427 次逻辑量化、231 次冗余消除。
收益来自更短的量化指令及其内联调用者，不是 INT8 化、删层或减少调用次数。

- NR256：720 次完整输出哈希与已冻结快速分支对应 reset/temporal 样本一致。
- 残差：3×13×2=78 次完整合成输出及 low NR 逐字节比较通过。
- 长序列：390 帧独立连续历史；每帧 low NR 全字节比较，1080p FP32 合成全字节 SHA256
  与已通过肉眼审核的视频一致。不是抽帧；没有重新编码视频或保存大幅全帧 dump。
- 输入/LUT、历史 seed、调用者修改输出、持有旧输出、reset、非法 motion/被替换表拒绝、
  progress 回退均通过相应完整测试。进度回调路径保留既有 rounder。
- CPU 框架回归验证模块别名去重、异常恢复全部绑定、检测干扰后仍恢复，及变化内核
  spill/未知报告拒绝。CPU 的资源门控用例是合成对象；真实资源证据来自 GPU 完整测试。

本批一致性是相对已审核的 FP16 快速分支，不声称它新获得了精确分支对 NVIDIA 的逐字节保证。
该串行实验适配器仍不是生产并发接口；OptiScaler/现有工具的完整集成尚未完成。

## 保留的失败与证据

body v1：门控错误地把未改变的 `native_cubic_batched_v1._project` 算进候选，
在其 spill=448 时拒绝；随后遇到重复模块绑定的恢复问题。没有证明新 rounder 溢出，
也没有设备故障。v2 只检查实际变化 JIT，按模块对象去重，并保证先恢复所有绑定再报告干扰。
native v1：720 次输出已比较，但末尾状态检查遗漏 provider 上下文，整份报告判失败。
native v2 补齐上下文后完整重跑通过；不采用 v1 的耗时作为最终成绩。
已执行的失败文件和报告均冻结，没有事后改写成成功。

数据根目录：`D:/Codex-NR-Experiments/nr-b580/reference`。

| 报告 | SHA256 |
| --- | --- |
| experimental/short-fp8-body-v1/validation.json（失败） | 661efbe2187e3bf50528fee1a33d1b24bce12d512e7746be18f0f0118f90e544 |
| experimental/short-fp8-body-v2/validation.json | 48ea3ee90be2f6e8166125a4917cfee1a46248a138f73bde90a1757b1a06aead |
| results/short-fp8-native-parity-v1/validation.json（失败） | 667af6f0741bd2a0715650dae2bcbd80f1e387bf63f8c4c9a8b1b1da8d660bc7 |
| results/short-fp8-native-parity-v2/validation.json | ddad6f922110fbb04775a3d7b2f65e131d03b1b553e5c1ab42fe854a7ca2f914 |
| results/short-fp8-residual256-v1/validation.json | fd7fe3b8f30e512a45fe766cf22be189fb5430193b72a588f5226957f00a588f |
| results/short-fp8-long1080-v1/validation.json | 38612d4dac943e0e533c319ddbf3fc4db862e4f8ddc847053958f472099ef8a1 |

运行报告中的 candidate_promoted=False 表示当时尚待后续门槛；本文件在完整测试与长序列
通过后记录采用决定。CPU checkpoint 另存源码、报告、lease、数组复核和新工厂的提交字节。
