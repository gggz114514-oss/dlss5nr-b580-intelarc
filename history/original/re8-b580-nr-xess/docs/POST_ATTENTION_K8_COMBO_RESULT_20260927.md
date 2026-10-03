# Post 注意力融合与 K8 输出联合候选（2026-09-27）

对象是《村庄》现役 480/540p 三结构组合，不改变精确后端；候选在 E 盘隔离实现，[联合入口](../game/post_attention_k8_combined_v1.py) 复用[注意力融合核](../game/post_attention_fusion_v1.py)和现役 K8 头的数值算法。post 注意力输出直接进投影，写连续 C32 特征；K8 从这块连续特征只读取可见范围并计算实际消费的 4 个输出通道。它代替原 post K8 作用域的实例方法，退出时恢复原身份，让现役组合其余 C64/C32 路径不变。候选已在隔离验证后成套安装于 G 盘游戏运行包，保留了旧文件备份。

## 540p 首验

Luna 用同一 960×540 合成输入与非零运动，D 盘缓存准备后在新进程磁盘只读状态做 B-C-C-B 配对。每段 120 个计时帧全部为图重放，C64 八块、C32 七块和 post K8 的组合捕获门均通过；候选注意力与 K8 在捕获各命中，受检输出有限。prepare 及四段末帧的 RGB SHA-256 与基线一致。两张捕获图的物理 Triton 发射总数 1858→1852，净少 6 次；这不是独立的耗时估计。原始记录：[准备](D:/Codex-NR-Experiments/nr-b580/post-attention-k8-combo-20260927/prepare-540.json)、[计时](D:/Codex-NR-Experiments/nr-b580/post-attention-k8-combo-20260927/benchmark-540.json)。

| 540p 完整帧 | 现役三结构组合 | 联合候选 | 差异 |
| --- | ---: | ---: | ---: |
| 240 帧合并中位数 | 50.5343 ms | 49.5688 ms | **−0.9655 ms（−1.91%）** |
| P95 | 51.1908 ms | 50.2281 ms | **−0.9627 ms** |

两组配对中位收益分别为 0.8813 和 1.0004 ms，方向一致。此前单独 post 注意力融合约 −0.92 ms 但 P95 退化；本次合入 K8 后 P95 同向改善，不能用两轮绝对计时直接相减。当前只读回 prepare 及各段末帧，不宣称每个计时帧都已逐字节验过。

## 480p 扩档

沿用相同计算结构、480p 对应的 512×896 内部画布与 480×864 可见输出。Luna 在隔离缓存准备后重新做 B-C-C-B；两组配对的中位与 P95 都提速，捕获门、120/120 计时帧图重放、磁盘只读、有限值与受检 RGB 字节哈希均通过。两张准备图物理 Triton 发射数 1842→1836。原始记录：[准备](D:/Codex-NR-Experiments/nr-b580/post-attention-k8-combo-20260927/prepare-480.json)、[计时](D:/Codex-NR-Experiments/nr-b580/post-attention-k8-combo-20260927/benchmark-480.json)。

| 480p 完整帧 | 现役三结构组合 | 联合候选 | 差异 |
| --- | ---: | ---: | ---: |
| 240 帧合并中位数 | 37.5296 ms | 36.7650 ms | **−0.7646 ms（−2.04%）** |
| P95 | 38.7439 ms | 37.3390 ms | **−1.4049 ms** |

两组配对中位收益分别为 0.7715、0.7649 ms，P95 均改善。冻结 243 帧人脸和逐帧 DIS 运动的**每帧原始 float RGB 字节核对**随后在两档都通过：基线/候选各自保留历史，全部 243 帧哈希逐一相同，差异帧为 0；每臂 2 次捕获、241 次图重放、输出有限，全程用磁盘只读缓存。[480p 逐帧记录](D:/Codex-NR-Experiments/nr-b580/post-attention-k8-combo-20260927/video-bytes-480.json) · [540p 逐帧记录](D:/Codex-NR-Experiments/nr-b580/post-attention-k8-combo-20260927/video-bytes-540.json)。按用户此前约定，字节完全一致不需要再次肉眼审核这段视频。

E 盘组合入口已将新 post 候选嵌进现有 K8 作用域内，在切档/关闭时先退出新候选，再退出旧组合，并把 attention/K8 两个实际捕获计数纳入门。Luna 对该 E 组合完成 540 输入的 84 个 prepare 用例和独立新进程 84 个 DiskOnly check；480/540 新门通过、360 原路径不变，准备阶段新增编译为 0。[集成预检](D:/Codex-NR-Experiments/nr-b580/post-attention-k8-combo-20260927/precompile-overlay.json)。

G 盘游戏运行包的 `three_structure_combo_v1.py`、`post_attention_fusion_v1.py` 和新 `post_attention_k8_combined_v1.py` 已成套安装并做哈希核对；原有两文件备份于 `D:/Codex-NR-Experiments/nr-b580/game-post-attention-k8-backup-20260927/`，可用[部署/回退脚本](../tools/stage_post_attention_k8_game_20260927.ps1)恢复。G 安装包随后完成 168 个 prepare 与独立进程 168 个 DiskOnly 校验，覆盖 540/720 两种源分辨率、三档尺寸、控制和历史组合；480/540 五项捕获门通过，360 原路径不变，新编译核已全部进入安装缓存。[安装缓存报告](D:/Codex-NR-Experiments/nr-b580/post-attention-k8-combo-20260927/precompile-installed.json)。

安装版实机启动后，网页控制先在 480p 连续执行至少 160 帧，图重放为真、健康状态正常，面板平均约 38.03 ms；再切 540p 连续执行至少 127 帧，图重放为真、健康状态正常，面板平均约 50.77 ms；最后恢复原来的 360p 设置。此处是不同实机场景的短窗遥测，不与上述离线配对中位数相减，也不能代替灯光/切焦的人眼验收。两档冻结视频的原始 RGB 已逐帧字节相同，按用户约定无需重复对该视频肉眼审核。360p 不运行新融合。
