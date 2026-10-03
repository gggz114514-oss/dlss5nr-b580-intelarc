# 《村庄》快速链 C128 QKV 直写：单块判别

仅改 E 盘隔离候选，不动 G 盘游戏或精确分支。Astra 对照现役 G 代码锁定 `encoder[2][1]`：540p 的 QKV 输入是 88×136×128，480p 是 72×120×128，4 个 head、每 head 32 通道。候选把 FP16 XMX 的 QKV 矩阵结果直接写入原窗口布局，保留 FP32 累加后 FP16 截断、Q/K 归一化、Q 的 head scale、FP8 舍入和窗口位置置换；attention 与后续投影仍用现役核。代码在 [单块候选](../game/c128_qkv_direct_pack_one_v1.py)。

[现役边界局部探针](../tools/probe_c128_c256_qkv_boundary_v1.py)在 G 安装代码和磁盘只读缓存下测得：540p 的 12 个 C128 块矩阵＋pack 共约 2.049 ms，16 个 C256 块约 2.261 ms；480p 分别约 1.476 与 1.709 ms。这些是局部阶段成本，绝非可全数节省的时间。[540p 报告](D:/Codex-NR-Experiments/nr-b580/c128-c256-qkv-boundary-20260927/probe-540.json) · [480p 报告](D:/Codex-NR-Experiments/nr-b580/c128-c256-qkv-boundary-20260927/probe-480.json)。本轮暂不动 C256。

单块局部测试在两档均通过：Q/K/V 三路完整字节哈希一致、值有限、全新进程 DiskOnly 检查通过。每档 ABBA 四段、每段 120 次局部图重放的结果如下：

| 输入档 | 基线两段中位 | 候选两段中位 | 两段平均中位差 |
| --- | ---: | ---: | ---: |
| 540p | 0.1732 / 0.1757 ms | 0.1236 / 0.1233 ms | 约 −0.051 ms |
| 480p | 0.1295 / 0.1286 ms | 0.0963 / 0.0961 ms | 约 −0.033 ms |

准备期最初未采用安装运行包的 `fast_cache_options`，导致局部字节门虽过、第一次 DiskOnly 仍报缺核。仅修[局部测试脚本](../tools/check_c128_qkv_direct_pack_one_v1.py)的准备缓存选项后，重新准备并在新进程复测通过；优化内核未因该故障改动。[540p 准备/计时](D:/Codex-NR-Experiments/nr-b580/c128-c256-qkv-boundary-20260927/c128-one-540-benchmark.json) · [480p 准备/计时](D:/Codex-NR-Experiments/nr-b580/c128-c256-qkv-boundary-20260927/c128-one-480-benchmark.json)。

540p 单块[完整链门](../tools/benchmark_c128_qkv_direct_one_full_v1.py)也通过：现役 C64/C32/post 组合每臂两次捕获门、候选单块每段六次命中，重置及随后两帧 RGB 逐字节一致；每段 120 个整帧图重放样本、全新进程 DiskOnly、准备期无新增编译。B-C-C-B 两组中位净差分别为 +0.1005 ms、−0.0732 ms，平均仅 +0.0137 ms 且第二组反向；P95 平均略退化。一个约 0.05 ms 的局部收益在约 49 ms 的整帧噪声内，本轮**未据单块整帧宣称提速**，也未将单块候选装进游戏。[准备记录](D:/Codex-NR-Experiments/nr-b580/c128-c256-qkv-boundary-20260927/full-one-540-prepare.json) · [配对记录](D:/Codex-NR-Experiments/nr-b580/c128-c256-qkv-boundary-20260927/full-one-540-benchmark.json)。

单块局部改善不能乘 12 外推，但单块的数值和现役组合共存门已过，故下一门扩大到 [12 块隔离候选](../game/c128_qkv_direct_pack_all_v1.py) 并重新做完整链 ABBA、逐帧字节和 P95。如果整族仍无稳定净收益即停止该同构扩张；只有明显收益才做 480p、完整视频和实机验证。

## 十二块整帧配对

Luna 已在现役 480/540p 组合上完成 [全链测试脚本](../tools/benchmark_c128_qkv_direct_one_full_v1.py) 的十二块选项。两档 12 块均命中，原 C64/C32/post 捕获门通过，重置与随后两帧的 RGB 哈希相同；每段 120 个计时帧全走图重放、DiskOnly 无缺核。这里只检查前三帧字节，尚不代表全视频验收。

| 模型输入 | 第一组中位 / P95 净收益 | 第二组中位 / P95 净收益 | 两组中位平均 |
| --- | ---: | ---: | ---: |
| 540p | +0.802 / +0.725 ms | +1.250 / +3.308 ms | **+1.026 ms** |
| 480p | +0.471 / +0.692 ms | +0.483 / +0.026 ms | **+0.477 ms** |

净收益口径为同轮基线减候选；540p 第二组 P95 的较大差值含基线慢尾，不能当成候选稳定尾延迟收益。两档中位各组均同向改善。原始记录：[540p 配对](D:/Codex-NR-Experiments/nr-b580/c128-c256-qkv-boundary-20260927/full-all-540-benchmark.json) · [480p 配对](D:/Codex-NR-Experiments/nr-b580/c128-c256-qkv-boundary-20260927/full-all-480-benchmark.json)。Astra 静态核对 12 块编码/解码对象、四种移位画布和与现役 C64 嵌套的恢复顺序，未发现确定性偏差；并要求补两张图逐块命中及更长时序的字节门。随后用[243 帧逐帧检查](../tools/verify_c128_direct_video_bytes_v1.py)完成该验证。

243 帧冻结人脸视频在 480/540p 均通过：基线与候选各自保留历史，原始 float RGB 每帧字节哈希一致，差异 0 帧；每臂两次捕获、241 次重放，候选每张捕获图的 12 块调用计数都增加，现役组合门、有限值、DiskOnly 全通过。第 0 帧是历史重置，素材 DIS 向量为零；第 1–242 帧均为非零运动。这里如实称“非零运动连续历史”，不称每帧非零。[540p 视频记录](D:/Codex-NR-Experiments/nr-b580/c128-c256-qkv-boundary-20260927/video-bytes-540.json) · [480p 视频记录](D:/Codex-NR-Experiments/nr-b580/c128-c256-qkv-boundary-20260927/video-bytes-480.json)。按用户此前约定，字节一致的视频无需重复肉眼审核。

新 C128 入口已放入 [现役结构组合](../game/three_structure_combo_v1.py) 的 C64 作用域内部，480/540p 添加逐块捕获门，360p 原路径不变。E overlay 覆盖 540/720 两种源尺寸的准备 168/168 与独立进程 DiskOnly 168/168，通过全部控制/历史组合；新编译为 0。[overlay 记录](D:/Codex-NR-Experiments/nr-b580/c128-c256-qkv-boundary-20260927/precompile-overlay-c128.json)。三份源码已用[部署脚本](../tools/stage_c128_direct_game_20260927.ps1)成套装入 G 游戏运行包并哈希核对，原 combo 备份在 `D:/Codex-NR-Experiments/nr-b580/game-c128-direct-backup-20260927/`。G 安装版随后对两种源尺寸完成 prepare 168/168 与新进程 DiskOnly 168/168；新增编译 8 个特化、只读命中 487 次，480/540 新门和旧组合门全过、360 保留原路。[安装缓存记录](D:/Codex-NR-Experiments/nr-b580/c128-c256-qkv-boundary-20260927/precompile-installed-c128.json)。安装版游戏进程随后短窗连续执行 480p 至少 113 帧、540p 至少 87 帧，图重放为真、健康状态正常；面板平均约 38.73/50.87 ms，属于不同场景的实机遥测，不能与上述离线 ABBA 相减。设置已恢复原 360p，实机人眼观察未单独取得；冻结视频的字节门已按用户约定替代其画质复审。
