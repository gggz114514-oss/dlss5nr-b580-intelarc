# 局部无损 INT8：数值检查通过，速度退步，不采用（2026-09-10）

用户询问进展后，主助手核对 Luna 的完成交接及结果、日志、lease、CPU 回归
四项哈希，重新认证 767 项 sources 和 269 项 exact_gate，并重算全部配对
中位数。v3 rc=0，phase=completed，5 个案例完成，无 error/finalization_error。

| 实际投影（144×512×512） | 当前 FP16 中位数 | 混合 INT8 中位数 | 耗时变化 |
| --- | ---: | ---: | ---: |
| encoder512.0.ffwd_projection | 0.010820 ms | 0.024571 ms | +127.09% |
| encoder512.1.ffwd_projection | 0.010870 ms | 0.021344 ms | +96.35% |

两组均为 7/7 配对轮混合更慢。每轮 20 次图重放，每图 64 次相同投影；常驻
输入、一次性权重打包，激活动态分类/转换计入内核，外部 FP8 舍入和 IO 不计。
这是局部投影计时，不是整模型、原生 NR 或 1080p 帧耗时。

全 FP16、全 INT8、极值混合控制和两个真实案例的 half 输出、外 FP8 边界
全部逐字节一致；真实基线与冻结 FFWD 边界一致。GPU 分支分类和 CPU 相同，
debug 与实测内核一致，输入改零/恢复、输出稳定及操作数不变检查通过。
这些有限输入不能证明所有帧或完整原生迁移的数学等价。

三个编译核 spill 均为 0。混合和 debug 寄存器报告为 256；baseline 为 0，
这个值不能解释成 GPU 不用寄存器，也不宜直接据此计算占用率差异。实测说明
分类、转换、尺度还原、混合分支和点积整体组合得不偿失；没有做逐项开销隔离，
不能把全部退步定量归因于某一个步骤。

第一个真实案例有 132/144 行-K32 块进入 INT8，但该计数包括零乘积块，不能
替代 CPU 报告排除零乘积后的 53.846% 有效块覆盖。第二个为 15/144。

Luna 交接里 v3 runner/CMD 的 expected hash 沿用了 v2 值。这是交接元数据
误报：启动前工具输出已记录正确 v3 哈希，CPU 回归报告也锁定正确 runner，
结果自身 source map 与现有源文件全部匹配。主助手确认 v3 正确值：

- runner：7463a8b7f8d3975f6965ebfba49ae20f0b29ba584171482d44db19b7ebb08f4c
- CMD：959e57ef86de6eaad3f8797f5fb082d1140b9f25ef433b4987134ae8015abf20

不修改既有执行文件或 Luna 原交接，以此补充说明保存审计过程。

决定：不推广混合原型，不再扩大这版测试。默认仍 nr256_selected_stack_v4。
按用户既定顺序，进入连续计算段重新量化；先对 ViT 扩张、激活、收缩链做
CPU 误差与尺度筛选，再决定 GPU 融合。该路线允许新量化误差，必须另做画质
和时序验证，不沿用本轮字节一致来免除未来视频审核。

结果：D:/Codex-NR-Experiments/nr-b580/reference/experimental/fp8-partial-int8-projection-v3/validation.json

SHA256 0d181b2f97b7d145627f03042d44f3a45fd27c49d4bd913838538f2a4fbb8c28。

日志 SHA256 3dd564a5f913cc73ea69828d1a0bc56bef33b51cd1274e49e7920ce3238b9300。

Lease SHA256 eb67ce428bc3cc466ed17db254c9907ec2005e875432a16b2eb6124c9a1fbda5。
