# 完整图数据流与矩阵 FP8 融合实验（2026-09-09）

本批**不替换当前最快配置**。选用路径仍为 c5ac61b 的 FP8GraphRewrite；
92 处融合版完整 NR 略慢，排除寄存器溢出后的 84 处版只在独立 NR 测试中
改善约 0.68%，加入缩放、合成后基本持平。不能把这批称为性能突破。
完整迁移目标仍 ACTIVE；不发布，不改精确分支，不改变已审核画面。

## 同一轮配对实测

单位 ms；连续帧均值，两个 B580 变体交替执行，三轮。

| 测试 | c5ac61b 基线 | 新候选 | 判断 |
| --- | ---: | ---: | --- |
| 独立 NR256，92 处全融合 | 12.481519 | 12.551471 | 慢 0.56%，三轮均慢 |
| 独立 NR256，84 处无溢出融合 | 12.411754 | 12.327089 | 快 0.68%，三轮均快 |
| 1080p 残差链路、内部 NR256，84 处版 | 13.254136 | 13.256248 | 持平，未通过采用门槛 |

独立 NR 包括完整模型、状态提交和完成等待；输入驻留 GPU。残差链路另外包含
颜色/运动缩放和全尺寸合成。都不包含解码、运动估计、上传、显示、游戏争用或 JIT。
4060 固定同一 NR256 输入的原生完整调用参考仍为 3.566807222 ms。
84 处候选与其比值为 3.456057；当前基线同批测量约 3.48 倍。不是 480p 比较。

84 处候选独立本体诊断的 temporal host 均值 10.432470 ms，基线 10.499430 ms。
残差实验中本体单独回放中位数 10.342210 vs 10.445990 ms。这些单独回放诊断
不等于完整 NR 时间；不能将不同统计样本相减当作准确的逐帧分段计时。

## 获得了什么架构证据

1. `full_body_dataflow_v1.py` 在当前已选计算图上保留完整 reset/temporal 事件，
   包含存储生命周期、写版本、生产者、消费者、张量元数据、ATen 标量参数、
   Triton constexpr/launch 参数和最终逃逸输出。仍用弱引用区分存储生命周期，
   不因追踪而保留 GPU 存储。完整图每次 755 个 Triton 调用、232 次实际独立 FP8。
2. `plan_fp8_epilogues_v1.py` 对 167 处普通矩阵输出做全部用途检查：只有所有读取
   都经已审查的 FP16 视图、复制或常数零填充抵达相同 FP8 转换，才允许前移舍入。
   外部逃逸、其他算子直接读取、重解释 dtype、后续存储写入等都拒绝。
   reset/temporal 各选出 92 处，另外 75 处保留原路径。
3. `_matmul` 保持原来的点积、累加和显式 FP16 舍入，仅在最后存储前加已有
   `_round_fp8_half`。这不是 INT8 新模型，也不改变缩放/画质策略。
4. `MatrixEpilogues` 只作用于已拥有模型的 NR256 图构建，认证计划并核对每次调用
   的次序、形状、步长、参数和输出存储。新增已量化输出契约，让原来的数据流
   重写去掉后续冗余量化。每帧仍回放已有 GPU 图，不重新执行这些 Python 分析。
5. 92 处版：755→663 Triton，232→140 独立 FP8；编译器在其中 8 处报告 spills=192。
   84 处版只排除这 8 处：755→671，232→148；其余内核无报告的溢出。
   调用数、寄存器溢出和逻辑字节数都不能单独证明实际耗时。

`nr256_selected_stack_v1.Stack` 抽出当前串行实验的相同组件组合，已通过完整
首帧/时序帧对照。它仍是实验工厂，不是已完成并发、资源所有权和部署设计的后端 API。

## 正确性与边界

- 完整 reset/temporal trace 的低 NR 输出、全 1080p 合成、历史、seed、输入及 LUT 与冻结参考一致。
- 实际 GPU 对比全部 92×2 个矩阵输出：原矩阵→原量化与融合矩阵逐字节一致，
  共 73,506,816 字节；记录每处输出摘要和编译 LLIR，保存去重后的编译产物。
- 两个候选各有 720 次完整 NR 输出对照，全部与基线及冻结 3.7.2 快速分支摘要一致。
- 84 处版另有 78 个低 NR/全 1080p 输出对照；历史、seed、无效 motion、常量变更、
  caller mutation、持有输出、progress fallback 等已有边界检查通过。
- CPU 审核认证源文件、报告、租约、编译产物和已保存数组，并重放 8,613 条读取依赖。
  15 个规划器正反例覆盖直接量化、零填充、alias、逃逸、原始读取、修改、部分存储、
  dtype、INT8、批量矩阵和不同归约块等。不是独立 CPU 模型重算。
- 全部 GPU 作业正常结束。布局盘点 v1 有一次 CPU 计数断言错误：reset 有 51 处
  contiguous，temporal 有 52 处。保留原脚本和失败记录，v2 改成按模式核对并通过。
- 候选未采用，未继续追加 390 帧验证或编码新视频；不要求用户再审核相同输出。
  当前已选 c5ac61b 的 390 帧验收继续有效。

## 下一步：按网络块设计布局

`layout-boundaries-v2/analysis.json` 保留所有实际布局写入的输入生产者、输出消费者、
形状/步长/偏移和标量参数。完整时序图中有 52 次 contiguous、64 次 pad、12 次
repeat_interleave；还包含 window pack/unpack 等显式内核。分配和视图记录单列。
这些是候选边界盘点，不能认定每一次都可删除，也没有给它们分配耗时占比。

下一个实验应选择一组完整 encoder/decoder block，将块输入、窗口排列、投影和
残差输出定义成明确的内部布局，使消费者直接按这个布局读取。先验证是否能同时
减少拷贝、pack/unpack 和独立逐元素操作，再用完整调用计时决定是否采用。
保留 FP16/FP8 舍入点、边界零填充、窗口偏移、skip 多消费者和时序状态行为。
更激进的权重/结构或精度变化必须走另一个可视觉审核的候选，不能算作精确迁移完成。

不要在已经持平的矩阵末端融合上继续盲扫配置；也不要把换成 C++ 当作已测得的本体收益。
目标仍是减少真实 GPU 工作和调度，并以同输入、同完整调用口径追赶 4060。

## 冻结证据

数据根目录 `D:/Codex-NR-Experiments/nr-b580/reference`。源文件在 E 盘。

| 报告（相对数据根目录） | SHA256 |
| --- | --- |
| experimental/full-body-dataflow-v1/validation.json | 1f71df725b0360c1fe449c1b6a18e8f1090ae03d6b1edb38c81356efa1f8e064 |
| experimental/fp8-epilogue-plan-v1/plan.json | aa5b8db36edba548bb061fd91339e20df8f1723ab384c0ffc18d876d1f06a267 |
| experimental/fp8-epilogues-v1/validation.json | be2ace2418b866f76490994dba3da332516b4fa21412ad3c606800c6a81f11b3 |
| results/fp8-epilogues-native-parity-v1/validation.json | 72ee5d9f854fdc2e116ecea6e9077d9b01a035f5f397430719f0a1f90dd7828e |
| results/fp8-epilogues-native-parity-v2/validation.json | 2fe4b5161b981dfea06a33e9d75662993ca2a719211c358d822e146950ac83ba |
| results/fp8-epilogues-residual256-v2/validation.json | 94acf5b65f55597bbe66d4f44561eb20e6ef60469d932c913d4a007a0e8eda99 |
| experimental/layout-boundaries-v2/analysis.json | 7af8cb3d1de93f11855f27055ed5fdb2ce92479a28efb2e253a0f6326356c191 |

四个 `saved-audit-v1.json` 位于 primitive、native v1/v2 和 residual v2 的报告旁。
primitive 审核同时覆盖完整 trace/plan。所有文件、审核者、精确后端的最终验证清单
另存于 `experimental/fp8-epilogues-checkpoint-v1/saved-audit-v1.json`。
未向参考笔记本写入数据；未发布 GitHub；OptiScaler/现有工具的正式集成仍未完成。
