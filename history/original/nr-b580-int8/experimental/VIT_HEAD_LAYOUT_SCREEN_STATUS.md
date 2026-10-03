# ViT 注意力布局交接：小幅正收益，尚未晋升（2026-09-10）

用户通知 Luna 完成后，主助手核对完成交接、结果/log/lease 摘要与466项源码哈希。
此前原型文件中的“未测试”状态由本文更新；已执行的源码不回写。

仅八个ViT输出投影直接读取注意力产生的[32,64,32]连续head布局，省去转为
[64,1024]的复制；保留四个K分区、K32累积、有序half合并及FP8边界。
未启用post裁剪，未改变C512。所有八块的七个完整边界（56个张量）、head布局
映射和整body输出逐字节一致；全零front变化输入也一致且输出随输入变化。

七轮交替，每轮20次静态body重放，中位9.294255→9.205110ms，减少0.089145ms
（0.95914%）。七对均较快，但早期基线存在漂移，不用均值放大收益。这只是静态
body筛选，不是完整NR或时序速度，也不能同post的3.58%相加推算。

Triton调用683不变，ATen记录2059→2043；16个ATen记录不等于16个GPU内核。
独立FP8/逻辑FP8/消除数量仍196/427/231。新投影内核spill=0、shared=4096；
n_regs=0只是驱动报告值，不能理解为不使用寄存器。57个ShortFP8资源记录零spill。
3次构图各覆盖8块；输入、模型、history/seed、作用域恢复、图池外持久IO检查通过。

结果：D:/Codex-NR-Experiments/nr-b580/reference/experimental/vit-head-layout-body-v1/validation.json
SHA256：95faf6bb6818627d30f1aa5cf53fd0b30c01050d0130e6705dce96af206a2c26。
Luna交接在相邻vit-head-layout-body-v1-monitor-luna-v1目录，returncode=0。

默认仍是ShortFP8 Stackv3。接下来独立实现C512编码/解码窗口输出的直接投影消费者，
检查全部公开边界、末端未量化full与pool/final。不能以这次ViT结果宣称C512路线完成。
随后再按结果评估完整NR/残差和组合；两条独立路线得到有效结论后才转INT8后续阶段。
