# C512窗口布局交接：筛选正收益，尚未晋升（2026-09-10）

主助手在用户通知完工后核对Luna完成交接、result/log/lease摘要、466项源码与
269项exact_gate哈希。returncode=0，未修改已执行文件，默认仍为ShortFP8 Stackv3。

16个C512编码/解码块使用现有注意力生产者的FP8 half窗口输出；新的投影消费者
按shift和pixel_inverse直接读取，保留K32顺序、BM16/BN32、half残差舍入和
量化前full。减少unpack、裁剪后的连续复制、独立量化及缩放残差中间张量。
公共forward/forward_boundaries接口不变，未改FFWD内部布局，未启用ViT布局和post裁剪。

66个完整边界、16个量化前full、四种shift的独立CPU窗口映射全部逐字节一致。
固定输入及zero-front变化输入的完整body输出一致，变化输入输出不是旧缓存值。
末端encoder池化仍读取量化前full，pool/final完整边界已核对。

七轮交替，每轮20次静态body重放，中位9.757495→9.581700ms，减少0.175795ms
（1.80164%），七对均较快。不能与上一轮不同运行时段的ViT耗时横比，也不能直接
相加三个筛选百分比。这不是完整NR/1080p残差/长时序性能结论。

Triton调用683→651，独立FP8196→180，逻辑FP8调用427→395，消除231→215；
ATen记录2059→1851，不等于GPU内核数。候选3次构图，每次16块；基线未调用候选。
四个新投影特化spill=0/shared=3072，58个ShortFP8资源记录全零spill。
输入、模型、history/seed、绑定恢复、图池外持久IO及调用者输出检查通过。

结果：D:/Codex-NR-Experiments/nr-b580/reference/experimental/c512-window-layout-body-v1/validation.json
SHA256：d538d84f920fd39b31403d49996b6eceb02f4c0576a33cd670309a5d11b66268。
交接：相邻c512-window-layout-body-v1-monitor-luna-v1目录。

下一步将此消费者、ViT布局、post依赖裁剪组合，直接对照当前选定后端测完整调用。
先检查完整reset/temporal输出及状态，再确认1080p残差/历史/回退路径；未通过不得晋升。
这完成了上述具体交接点的独立筛选，不代表所有跨算子布局机会已穷尽。
