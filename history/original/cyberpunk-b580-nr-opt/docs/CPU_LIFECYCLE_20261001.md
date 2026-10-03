# 720p 快速链路 CPU 检查迁移

现役范围为《赛博朋克》720p、C512＋K8、去舍入、融合真实运动、计算图重放。2026-10-01 的 V6 已在实际游戏完成两套关闭→开启→开启→关闭配对：参考四项的 NR 处理及适配器交接合计平均 **45.7046 → 44.4875 ms，减少 1.2171 ms（2.66%）**；同样四项再加 C128 时 **47.4969 → 44.5733 ms，减少 2.9236 ms（6.16%）**。GPU 图时间基本不变，这是 CPU 检查的收益；尚未据此测得 RTSS 基础帧率提升。V6 的三个源码变更已备份合入 E 盘现役树，实际游戏已保留参考四项＋检查迁移开启。

## V5 实测边界

DLSS 性能档，帧生成关闭，同一场景静止镜头。两侧都勾选 DecoderInput、C32、分数历史坐标、Front 双精度清理，以及 C128 分支；唯一配对变量是“固定检查移到加载／建图（实验）”。按关闭、开启、开启、关闭顺序，每组采集 30 个唯一、已完成、实际重放的 NR 帧，排除切档建图帧。

| 顺序 | 固定检查迁移 | 顺序阶段合计平均 | 中位数 | P95 |
|---|---|---:|---:|---:|
| B1 | 关闭 | 47.8785 ms | 47.6363 ms | 49.4105 ms |
| C1 | 开启 | 45.5237 ms | 45.5040 ms | 46.0663 ms |
| C2 | 开启 | 45.4820 ms | 45.4041 ms | 46.0684 ms |
| B2 | 关闭 | 47.5298 ms | 47.5084 ms | 48.1387 ms |

顺序阶段合计为 process_wall_ms＋adapter_handoff_cpu_ms＋observer_finish_cpu_ms，三者按适配器执行顺序发生。GPU 图时间嵌套在 process 内，不能再加一次。两组开启值均低于两组关闭值；关闭侧均值漂移约 0.349 ms。

| 阶段 | 开启相对关闭 |
|---|---:|
| host process wall | −1.4771 ms |
| adapter CPU 线程交接 | −0.7247 ms |
| observer finish | ＋0.0005 ms |
| GPU 图（嵌套） | −0.0387 ms，处于漂移范围 |
| GraphFront 固定 buffer 扫描 | ＋0.0216 ms，仍约 1.7 ms |

开启侧未提供 numeric_guard_cpu_ms，保留 unavailable，未当作 0。结果仅证明这套 C128 组合的检查迁移收益，不能将旧候选的全部减速结论一并推翻，也不等同于完整游戏帧时间。Luna 已补测同一参考四项的 OFF→ON→ON→OFF，各 30 帧：process wall 减少 0.2075 ms、adapter handoff 减少 0.1858 ms、finish 增加 0.0003 ms，顺序合计减少 **0.3930 ms**。GPU 图增加约 0.0274 ms；没有 GPU 提速证据。该组也恢复参考组合＋关闭，并完成 8 帧健康检查。

测试已恢复四项参考组合和迁移关闭，并取得 8 个唯一完成重放帧，健康状态未失败，实际配置和 epoch 稳定。期间计数推进 25 帧、17 条未被采样连接，因此不声称取得连续逐帧事件轨迹。V5 源码、运行时、native 和缓存清单冻结前后相同。

## 已合并的改动

V5 包含 DecoderInput、C64/C128/C256 分支、History、ViT、Post、Front 的固定源码／符号、权重、拓扑、二进制及配置扫描迁移，和第三批同一调用栈的重复 guard、成功热帧纯诊断构造精简。加载、切档、实际建图前后与显式诊断继续完整核验。

历史、运动、随机种子更新、真实输入键、线程／资源归属、退役和 GPU 同步继续执行。History 路由记录有真实下游消费者，保留。HTTP 线程仅提交策略请求；NR 线程在串行锁内先退役旧图，再创建新策略会话并重置一次历史，不能给旧图直接换标志。

V5 修复了初始化顺序、正常退役以及作用域退出的检查边界：Post/Front 收尾在临时算术后端恢复后执行，永久归属检查不得要求临时后端仍开启；实际 dispatch 和 entry 消费仍检查正确后端。34 项集成 CPU 检查和前一包 10 项重复检查用例通过。

V5 bundle：3630d7b689df6119d806397ad73a2d2f96c9b341c608f4170faed8dd4ac74732。原生周期闪动修复、纹理桥、GPU 数学、模型及缓存未改。默认迁移关闭，网页独立勾选用于验证；14 个源码文件已完整备份并合入 E 盘现役树，回执为 D 盘本任务 replay-lifecycle-source-merged-v5-20261001/promoted.json。

## V6：GraphFront 常量扫描

V6 只在已封存、实际命中的重放 entry 内复用不可变模型常量清单。新会话、真实 key miss、捕获发布及显式检查仍扫描；原 forward 的 provider 检查、输入 key、输入复制、重放、完成与退役不改。使用现有迁移勾选项，不增加互斥数学档位。

主审阅发现 sidecar 冷审计直接调用完整 _validate，而实际 _select → numerical_scope.__enter__ → preflight 早于 session._installed()。V6 已把加载时审计限于固定源码、closed、signature 和 constants，实际推理仍由原 _validate 核验临时 provider。另修正计时器兼容：常量 sidecar 开启时不再用观察 wrapper 替换受保护的 _validate，保留 GPU 和 process 计时；该 CPU 分项 unavailable。12 项计时器 CPU 检查及 13 组实际冷作用域／原 provider 安装恢复边界回归全部通过。最终 V6 为 16 文件，bundle 30a711260c7fc378cec194dd99321b2f62e54ac25057c9995868e521df092682；2026-10-01 20:33（中国时间）用户正常退出后已完整备份安装并启动 PID 20560，用户进入同场景后由 GPT-6 Luna xhigh 独占实机与 API 测试。

两套各先完成关闭→开启→关闭的 8 帧就绪检查，再做关闭→开启→开启→关闭，每组 30 个唯一完成重放帧。主任务从原始逐帧记录独立重算顺序合计，结果与 Luna 汇总一致；每套的两个开启均值都低于两个关闭均值。

| V6 同配置配对 | 关闭平均 | 开启平均 | 减少 | 关闭／开启 P50 | 关闭／开启 P95 |
|---|---:|---:|---:|---:|---:|
| 参考四项 | 45.7046 ms | 44.4875 ms | 1.2171 ms | 45.6327／44.4030 ms | 46.4529／45.0407 ms |
| 参考四项＋C128 | 47.4969 ms | 44.5733 ms | 2.9236 ms | 47.4777／44.4794 ms | 47.9927／45.1328 ms |

合计仍为 process_wall＋adapter_handoff_cpu＋observer_finish_cpu；GPU 图嵌套在 process 中，不重复相加。P50/P95 为每侧合并 60 帧的中位数和线性插值百分位。两套关闭侧前后均值漂移分别为 ＋0.2608 ms、−0.1325 ms。GPU 图变化分别为 −0.0473 ms、＋0.0145 ms，没有 GPU 提速证据。开启侧的 GraphFront 校验计时缺失，记为 unavailable，未假作 0。

上述收益是 V6 整个检查迁移开关相对关闭的结果，包含 V5 已有迁移；不能再加上 V5 的收益，也不能直接把跨轮差值当成 V6 常量迁移的独立收益。本轮不证明 C128 数学候选比参考组合更快，最终仍保留参考四项，不勾选 C128。

最终 8 个新完成重放帧证明参考四项实际命中，health.failed=false，检查迁移 enabled=true／applied=true／pending=false、epoch 13。16 文件、native、host 及配置的冻结记录前后一致。本轮没有离线编译或修改缓存；冻结记录没有缓存清单条目，因此不声称对全部缓存文件做过哈希核验。收尾控制器首次因 registry 未初始化失败，尚未发设置请求；Luna 修正控制器后完成最终核验，未重跑成功的配对。

V6 三个变更为 lifecycle base、GraphConstants helper 和可选计时器；已合入 E 盘现役树，原文件备份与合并回执在 replay-lifecycle-source-merged-v6-20261001/promoted.json。游戏正在运行的 G 盘文件就是通过配对的版本，此次源码合并未再次修改游戏文件。

固定模型在单个会话中不可私自修改；更新需关闭旧会话再创建新会话。热路径不再用版本扫描检测私自原地写权重，显式审计和新建图仍拒绝变化。低强度缺少 float32 图时的原有 eager 回退不在本次修改范围。

## 可复现证据

- [V5 最终实机报告](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/replay-lifecycle-v5-unwind-live-v1-20261001/final-v5-c128-completed-20261001/RESULT.md)
- [四组逐样本合计分析](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/replay-lifecycle-v5-unwind-live-v1-20261001/main-summary-c128-20261001.json)
- [参考组合四组配对](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/replay-lifecycle-v5-unwind-live-v1-20261001/reference-only-v5-off-on-on-off-20261001/REPORT.md)
- [V5 来源和作用域修复](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/replay-lifecycle-controls-v5-unwind-20261001/REPORT.md)
- [V6 安装清单](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/replay-lifecycle-controls-v6-constants-20261001/payload-manifest.json)
- [V6 安装回执](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/replay-lifecycle-installed-v6-constants-20261001/installed.json)
- [V6 冷启动边界回归](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/replay-lifecycle-graph-constants-v1-cpu-20261001/ADDENDUM-v6-cold-select-provider-boundary-20261001.md)
- [V6 两套实机配对与最终状态](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/replay-lifecycle-v6-constants-live-20261001/REPORT.md)
- [主任务逐帧复算：参考组合](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/replay-lifecycle-v6-constants-live-20261001/main-reviewed-reference-20261001.json)
- [主任务逐帧复算：参考＋C128](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/replay-lifecycle-v6-constants-live-20261001/main-reviewed-c128-20261001.json)
- [V6 源码合并回执](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/replay-lifecycle-source-merged-v6-20261001/promoted.json)
- [原常量 sidecar 的 CPU 验证与限制](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/replay-lifecycle-graph-constants-v1-20261001/REPORT.md)
