# 赛博朋克 NR 启动默认档（2026-09-29）

用户要求新进程默认选择网页的“去舍入组合版”、融合采样＋真实运动、计算图重放。保持原有 540p 输入高度、标准风格及其他参数不变；用户仍可在网页内切换和对比。

- `src/asi.cpp` 的内置 NR 启动控制从 `ZERO_MOTION` 改为 `FUSED`；图重放仍为开启。
- `game/cyberpunk_nr_adapter.py` 的初始面板和原生控制映射默认选择 `backend_variant="unrounded"`。模型宿主由面板设置实际选择去舍入计算；启动后用户手动选择标准版仍可在当次进程生效。
- 未修改共享的 RE8 控制页全局默认值，也未修改精确后端。

构建使用 Release、14 并行任务；CTest 10/10 通过。游戏插件和适配器在替换前备份至 `D:/Codex-NR-Experiments/cyberpunk-opt/deployed-backups/before-default-unrounded-fused-20260929`。重启《赛博朋克》后，网页设置回报 `540p / unrounded / fused / graph=true`；实际执行状态也回报这四项，图重放 `graph_used=true`，连续超过 600 帧且 `health.failed=false`。后续运动探针在 3600 个采样点中出现非零值（例如第 420 次采样有 24 个），说明实际读取了游戏运动纹理；这不替代用户对画面与运动的肉眼确认。
