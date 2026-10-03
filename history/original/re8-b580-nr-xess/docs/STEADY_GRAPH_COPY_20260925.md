# RE8 540p 图内拷贝：按采样时间重读现有 VTune 结果

状态：对既有 VTune 结果做只读时间切片；尚未改变模型或生产运行时。这里的时间是 **GPU task 累计时长**，不是整帧墙钟，也不是把 `Compute` 等同矩阵算术。

源结果：`D:/Codex-NR-Experiments/nr-b580/re8-stage-profile-20260925/vtune-gpu-hotspots-540p-20260925/`。原采集包含程序启动、模型准备、16 帧预热、32 帧计时和 2 帧哈希校验，总任务累计：Compute 2.837460 s、显式 `zeCommandListAppendMemoryCopy` 0.404073 s（12.47%），拷贝 9.571 GB。原先把这 12.47% 视作稳态显式拷贝占比是不成立的。

用安装的 Intel VTune 2026.3 对**同一结果**执行 `-report hotspots -group-by computing-task-purpose -time-filter <秒:秒> -format csv`，获得：

| 相对采集起点 | Compute 任务累计 | H2D 类拷贝任务累计 | D2H 类拷贝任务累计 | 显式拷贝任务占该窗任务累计 | 显式拷贝字节量 |
|---|---:|---:|---:|---:|---:|
| 12–14 s | 365.578 ms | 2.005 ms | 0.514 ms | 0.68% | 0.858 GB |
| 14–16 s | 1570.528 ms | 12.608 ms | 2.063 ms | 0.93% | 4.351 GB |
| 16–18 s | 884.149 ms | 7.833 ms | 1.188 ms | 1.01% | 2.416 GB |
| **14–18 s 合计** | **2454.677 ms** | **20.441 ms** | **3.251 ms** | **0.96%** | **6.768 GB** |

14–18 秒窗口的 `source-computing-task` 查询仍有 `_five_tap` 44 次、`_kernel` 22030 次、`_matmul` 5717 次、`_pairs` 1674 次和显式拷贝 2826 次。因而它确实覆盖大量模型帧，而不是只看启动后的空闲区。`_kernel`、`_matmul`、`_pairs` 在此窗累计分别为 987.708、450.258、236.112 ms；这些通用名跨多个模块，不能直接定到某一层。

另按 `computing-task` 查询，5–10 秒窗口的显式拷贝任务累计约 360.200 ms，已占整个采集的 404.073 ms 的 **89.1%**。这一早期窗口包括启动／预热阶段，具体调用来源没有栈信息，不能直接声称是权重上传。后期窗口虽有大量拷贝字节，却只占 GPU task 累计时长约 1%。VTune 的 H2D/D2H 分类没有源／目标指针，不能据此断言每帧 CPU↔GPU 往返；应用层 `copy_` 或张量物化也可能被列为计算内核而非此类显式拷贝。

结论范围：**Level Zero 显式拷贝任务不是后段主要 GPU 时长**。这并不证明张量布局、量化、重排和核间读写可以忽略；它们很可能落在 Compute 类。下一步要按完整连续层组（特别是 C512／ViT／decoder→post）划分图内任务，先测时长再选语义不变的融合候选。图边界输入／输出缓冲的逻辑复制字节数不能直接从 VTune 的传输量中相减。

图边界字节审计已由 Luna 在相同 540p G: 生产运行时执行：`D:/Codex-NR-Experiments/nr-b580/re8-graph-io-audit-20260925/graph-io-byte-only.json`。该轮桌面 GPU 活动超过空闲门槛，脚本将所有耗时标记为**不可用于性能结论**，只核对 buffer 大小、重放计数和输出哈希。16 帧预热后 4/4 帧重放都进入有历史的图条目；两个重置帧的最终 RGB 哈希一致。每次有历史的重放，固定图输入复制 35,594,240 B，其中 `front` 20,971,520 B、RGB 6,266,880 B、上一帧 6,266,880 B、历史倒数 2,088,960 B；图内输出发布和返回克隆各 3,133,440 B，因此**已知图边界显式复制 41,861,120 B／帧**。这不是显存实测流量或时间，且不覆盖后续图内张量物化。方形 history-warp 图该轮重放 0 次。

结构审读发现生产图的 post 尾段仍执行 `repeat_interleave → 缩放 → skip 合并 → 补边 → attention unpack → 裁剪后连续化`；其中两次放大张量及裁剪后的连续化约有 104.86 MB／帧的逻辑写入。普通 decoder 的 gather 覆盖不包括 post。这是可验证的融合候选，但现有 VTune 中对应单个大尺寸 `_unpack` 仅约 0.2 ms／次，不能只凭字节量认定整个 post 是 54 ms 的头号瓶颈。先区分 C512、ViT、解码／post 的图内组级耗时，再决定先改哪段。

另一处待验证的生产路径差异：全尺寸会话在 `game/fullsize_session_v1.py` 中排除了选定栈的 `rewrite` 与 `compact_queries` 组件，改用自己的 C512/ViT 入口。不能把 NR256 栈曾经通过的改写自动计入这条 540p 游戏路径；也不能未经逐位对照就启用这些组件，因为它们可能按 NR256 几何或舍入边界设计。C512 入口先量化 attention 结果，`SplitProjection.forward_unquantized` 又对相同特征调用量化函数，存在可检查的重复量化；应先证明输入已符合量化函数幂等域，再单独测完整层组和连续帧，不能只凭源码删去。

进一步诊断使用模型原有的 13 个 progress 边界：同一会话追加 eager reset/temporal 与 graph reset/temporal 对拍，四帧的两个配对 RGB 哈希逐字节相同。完整记录在 `D:/Codex-NR-Experiments/nr-b580/re8-graph-io-audit-20260925/eager-group-probe.json`。eager 每组都显式同步，加上预检有 6.225% 桌面 GPU 活动，**耗时只作粗排，不是图重放分段时间**。两帧均值依次为 ViT 9.62 ms、RGB/post 8.32 ms、pre 8.06 ms、decoder C512 5.72 ms、encoder C512 5.30 ms；余下各组均约 3.05–5.19 ms。此结果提高了 ViT 与 post 作为候选的优先级，但不能换算生产整帧收益。

`tools/check_fp8_idempotence.py` 按当前 `nr_backend/triton_fp8.py::_round_fp8_half` 的位运算，用 CPU 对全部 65,536 种 FP16 输入位型做一次／二次舍入比较，发现 **0 个非幂等位型**。这是该公式的静态数值证据，尚未证明在生产 C512 入口绕过第二次 `q` 后布局、捕获、历史和整帧 RGB 保持一致。下一候选须先对 16 个 C512 块和最终连续帧做逐字节门，再做配对图重放计时；若完整链收益太小，应优先转向 ViT/post 的连续段融合。

复核方式：对上述 VTune 结果依次运行 `vtune -report hotspots -r <结果目录> -group-by computing-task-purpose -time-filter 14:16 -format csv` 和 `-time-filter 16:18`；对早期窗口使用 `-group-by computing-task -time-filter 5:10`。查询仅读已有结果，不重新采集或占用 B580。
