# 窗口布局与寄存器溢出检查（2026-09-09）

选用 **WindowBlocks v3**，叠加在 c5ac61b 的已选 FP16/FP8GraphRewrite 栈上。
v1/v2 保留为未采用实验。完整迁移目标仍 ACTIVE；精确后端、公开仓库未改动，也未发布。

## 实测结果

单位 ms，连续帧完整调用均值；同次运行交替比较基线和候选，三轮。

| 候选 | 基线 | 候选 | 结论 |
| --- | ---: | ---: | --- |
| v1：投影直接读取窗口，读取时量化 | 12.473569 | 12.615942 | 慢 1.14%，不采用 |
| v2：v1 的四处溢出改用小分块 | 12.401629 | 12.552274 | 慢 1.21%，不采用 |
| v3：注意力写出时量化，投影直接消费 | 12.426860 | 12.250884 | 快 1.416%，三轮均快 |
| v3：再包含 1080p 缩放和残差合成 | 13.260352 | 13.132500 | 快 0.964%，三轮均快 |

包含 reset 的残差均值 13.253318→13.094841 ms，约快 1.20%。
v3 native reset 均值 12.195740→12.025262 ms。
固定同一 NR256 原生输入的 4060 完整 host 调用参考仍为 3.566807222 ms，
v3 的连续帧耗时是其 **3.434692 倍**，未达到持平；这里不是 480p 对比。

完整 NR 计时包括状态提交和完成等待，输入已在 GPU；残差计时再包含颜色/运动缩放、
全尺寸合成。均不包含解码、运动估计、上传、显示、JIT 或游戏争用。
分离本体诊断：native temporal host 10.540370→10.396490 ms；
残差测试中的静态本体中位数 10.442070→10.270200 ms。它们不等于完整 NR GPU 时间，
也不能用不同统计样本的差值声称准确的逐帧外围分段耗时。

## 实际布局改动

覆盖 36 个拥有的 MultiHeadSwinBlock：C64 8 个、C128 12 个、C256 16 个。
MLP、QKV 投影、归一化、注意力数学和 skip/pooling 保留原来已审核的快速分支行为。

原衔接为：注意力窗口结果→HWC unpack→FP8→残差缩放的独立张量→投影→裁剪。
v3 的注意力直接将原始 half 结果经过相同 FP8 舍入后写成 head/window/pixel/32 排列；
投影按窗口中的像素作为独立矩阵行读取它，以相同 K32 顺序进行点积。
利用模型 pixel_order 将结果写到最终裁剪后的 HWC 位置，独立缩放残差的 FP16 舍入仍保留，
随后与 FP32 点积相加并舍入到 half。

因此省掉 36 个 unpack、36 个独立量化张量及 36 次独立残差乘法等中间工作。
每次完整本体：Triton 调用 **755→683**，独立量化 **232→196**，
原先 231 次冗余 FP8 消除仍保留。新量化函数调用 427 次，另 36 个逻辑量化由融合内核完成；
对外逻辑 dispatch 收据不变。调用次数不是 GPU 时间占比。

独立 MultiHeadAttention 仍返回完整张量。C512 的 forward_boundaries 等调试/边界 API
没有被替换成不完整结果。输入、历史、输出所有权继续由已有串行 GraphFront 约束。
只在图构建时应用 scoped Python 适配；逐帧回放不再逐块分析、编译或选择配置。

## 回应寄存器问题：编译前资源检查

`spill_preflight_v1.select` 用已固定版本 Triton 的 `warmup` 编译，再通过
`CompiledKernel._init_handles` 加载内核，取得驱动报告的 n_spills；此时尚未发射内核。
新投影先尝试选定分块，若溢出则尝试已审查的 16×32 分块；所有候选都溢出或报告不可用
则拒绝这个实验配置。新注意力也通过资源门槛。没有修改全局编译器/驱动选项。

真实测试不是伪造编译报告：复用 v1 的 80×80×64 溢出几何，在发射前检测到
64×64 配置 n_spills=832，选择 n_spills=0 的 16×32 配置；哨兵输出缓冲逐字节不变。
v3 的新注意力和投影在全部已测块中均报告零溢出。

n_spills 是驱动原始报告值，不是“坏掉的寄存器数”；记录中的 n_regs=0 也不能解释为
内核不使用寄存器。此门槛限定于这组已测试的新内核，不声称覆盖整个后端。
v2 已证明**零溢出不保证更快**：较小分块可能降低利用率，且 v1/v2 在每个投影输出通道
分块重复执行了 FP8 转换。v3 将它移到生产者只做一次，才通过完整性能门槛。

当前固定 Intel Triton 源码还暴露每内核 grf_mode 选项，但本批未强制切换模式，
也没有假定硬件支持所有源码列出的模式。今后若测试，应独立验证目标设备、输出和完整耗时。

## 验证证据

- v1/v2/v3 各验证首帧和时序帧的低 NR、全尺寸合成、输入、历史、seed 和 LUT。
- 每版各 72 个真实块输出与原块逐字节比较；另有 6 个 10×12、补边、零输入和
  非连续输入用例。v3 还逐个比较了生产者写出的 FP8 窗口张量与原注意力→量化→排列结果。
- 三版各 720 次完整 native NR 输出都匹配基线与冻结 3.7.2 快速分支摘要。
- v3 的 78 次残差完整输出对照，及 progress、无效 motion、常量变更、持有输出、
  caller mutation、历史/seed 等边界检查通过。
- v3 的 390 帧独立连续序列：每个低 NR 张量字节、全 1080p FP32 合成摘要都匹配
  已审核序列；重置、输入、独立历史、持有输出和 LUT 检查通过。
- 八份 CPU 审核认证源码、租约、报告、已保存数组和编译产物。长序列审核重读
  780 个去重数组、2,503,020,263 个存储字节。审核不是独立 CPU 模型重算。
- 本批没有失败的 GPU 作业/CPU 审核。所有实验过程与源文件冻结保存；未重编码视频，
  输出字节相同，无需新增肉眼审核。数据在 D 盘，代码在 E 盘，未使用参考笔记本。

## 使用与后续

已测完整组装见 `benchmark_window_blocks_native_parity_v3.py` 或
`validate_window_blocks_long1080_v3.py`：保留整个当前栈和 FP8GraphRewrite，
创建 `WindowBlocks(model, provider, head_layout)` 并将其安装在图构建上下文中。
`nr256_selected_stack_v1.Stack` 仍是原基线工厂；如使用它，显式加入上述组件。
不安装旧矩阵 epilogue 候选。仍未完成生产并发、OptiScaler 或现有工具的正式集成。

下一步以这个已选组合为性能基线，继续按块消除真实中间数据。
可以评估 C32 或 C512 衔接，但必须保留 C32 未量化残差、C512 的 crop/skip/pooling 和
边界 API 的全部可观察输出；不能仅改 channel 检查后推广，也不能以假张量替代边界结果。
保持完整调用、字节/时序验证与新画质的人工审核门槛。

## 冻结报告

数据根目录 `D:/Codex-NR-Experiments/nr-b580/reference`，表内路径后接 `/validation.json`。
相邻 `saved-audit-v1.json` 为对应 CPU 审核。

| 路径 | SHA256 |
| --- | --- |
| experimental/window-blocks-v1 | 07b31cda5201207dda13a29b436dcd4549f19e159f5d6dd621e04e238d118cf5 |
| experimental/window-blocks-v2 | 6dc31ff62f58f7879da68dfa43244fcecbc6f6c31f8f7ba2a466c19ade785fc2 |
| experimental/window-blocks-v3 | e0b7a5f9bf996bc9e825d439761f1cd19f8ed6ff4fa768d2dd28839a0f63f8af |
| results/window-blocks-native-parity-v1 | e32cc0f4a3dc2ca08e9bb497714f31d209e928d7bb3e2cc2cf21a0a2d9c7fc5f |
| results/window-blocks-native-parity-v2 | 17253db66723e130177daa141c21c4d5e1c88ef84ff716c8f2d61f527745c0c3 |
| results/window-blocks-native-parity-v3 | 6986ddfcaf1adbfe054ac415c69515327682fe40f3ba780d8e92e2786407a63d |
| results/window-blocks-residual256-v3 | c29c1a67b2e1d8314b0f8169a79c7ea51aa33ffbebecbb4078d23dbbf6316586 |
| results/window-blocks-long1080-v3 | 48decfda37749aa61f03d5783ffcc4ee654d9e7875ee32d2706b65779e02e296 |

最终文件/精确后端隔离检查清单保存为
`experimental/window-blocks-checkpoint-v1/saved-audit-v1.json`。
