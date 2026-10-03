# 720p 实机周期闪动：诊断边界与安装

用户现象：在《赛博朋克》静止场景、没有调网页参数时，约每隔 5–10 秒闪一下，随后恢复，观感像一瞬间 NR 没有生效。2026-10-01 固定档位采集已证明单帧绕过 NR 的实际机制，见文末；修复后的视觉验收仍未完成。

独立 GPT‑6.1 Sol max 开发了 v2 被动诊断。原生桥保留有界的 128 个终结 SR 帧及 64 个异常事件；记录是否调用/完成 NR、是否提交合成、是否回退原图、资源/提交批次/代际、游戏 reset 和拒绝原因。Python 记录同帧的 reset 原因、模型 session/seed、历史是否存在、图重放次数和异常。以 NR frame ID、SR sequence 和 eval ID 精确关联；不以“最近一次全局帧”伪造关联。

19 项 CPU 检查通过；诊断 ASI 通过 MSVC Release、14 线程构建，PE 导出含 `NRB_GetPeriodicFlashDiagV2` 和 `NRB_GetPeriodicFlashContextV2`。安装前已验证实际 G host 与 E 被动 v1 插桩的差异，再验证 v2 中 reset 表达式及 `_bridge` / `_modes` 调用的 AST 不变。

安装仅涉及：

- `G:/epic/Cyberpunk2077/bin/x64/plugins/CyberpunkNRBridge.asi`
- 物理 `G:/SteamLibrary/steamapps/common/Resident Evil Village BIOHAZARD VILLAGE/nr-runtime/game/` 内 host、`periodic_flash_snapshot_v2.py`、`nr_temporal_diagnostics_v1.py`。

没有更换签名 OptiScaler loader、模型、内核或缓存，没有写经 Cyberpunk 的 `nr-runtime` junction。原 ASI/host 已备份到 `D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/periodic-flash-v2-installed-20261001/`；实际安装前后 SHA 见其中 `installed.json`。源码固定在 E 的 `artifacts/periodic-flash-v2-source-20261001/`；构建在 D。

诊断仅在本次游戏子进程设置 `NR_DIAG_PERIODIC_FLASH_V2=1`。现有 `/api/state` 的 `health.temporal_diagnostics.periodic_flash_v2` 返回窗口和计数，Luna 负责读取和保存：窗口 revision 改变时落盘、首末计数取样。与性能测试共用同一 GPU，串行执行。

证据边界：环形缓存有界，非阻塞原生记录可能丢事件，SR 提交不等同于 Present 或实际像素；Python/原生时钟未校准。未出现诊断异常不能证明屏幕没有闪动。若需像素证明，再针对已定位时间窗口做最小捕获，避免全程大量视频/读回。此阶段未修正画面行为；待实机事件证据定位后再改根因。

2026-10-01 15:06 +08：阻塞性能对比的离线缓存流程已闭合（20 候选＋18 历史组合在实际 G 新进程全部只读通过，0 编译/缺失）。主任务已以 `NR_DIAG_PERIODIC_FLASH_V2=1` 子进程环境启动 Cyberpunk，等待用户进入同一静止场景；90 秒自然闪动捕获由 Luna 在完成候选筛选后串行执行。此前切档缺缓存引起的失败锁存不用于归因自然闪动。源码审阅中跨命令列表的全局年龄超限仍是待验证线索，没有据此放宽资源状态门或重置历史。

## 固定档位的实机正面证据（2026-10-01）

20 项筛选完成后，Luna 恢复 reference 四项、720p C512＋K8、unrounded、融合真实运动、图重放、校验合并 ON，确认四个新完成帧，再串行采集 90 秒：15:26:25–15:27:55 +08，共 90 个接口取样。主任务仅读取已有 `end/duration` 的完成文件，不读取实时接口或监看 GPU。用户随后确认固定基准仍有“一闪就恢复”。

严格排除采集开始前保留的旧窗口后，首末 SR 为 16042→17395，共 1353 次超分调用；1337 次 NR 正常合成，**16 次 `source_state` 拒绝让该 SR 绕过 NR、回送原图**。16 个去重窗口与累计 fallback/skipped 增量完全相同。每个拒绝 SR 的前后相邻 SR 均正常 NR 合成，拒绝 SR 的 `nr_frame_id=0`、raw/skipped 为真；相邻 Python reset_causes/observations 为空。

16 次的颜色 age 为 66–247，source mask 均为 633；原检查要求的 Reset 已观察、未 Close、无 oversized、颜色 barrier 已见、plain/all-subresources/exact PSR 条件都成立，只有 age≤64 与派生 proven 两位未成立。正常相邻 SR mask 为 1017。本段 game_reset、generation_mismatch、invalidated、failure 增量均为 0，事件/frame dropped 增量也为 0。事件相邻间隔 0.288–12.655 秒，中位 3.976 秒；不是严格的 5–10 秒定时器。

因此，“全局命令计数年龄超过 64 → source gate 拒绝 → 一帧不执行 NR → 下一帧恢复”已是实机事实。结合用户现象，这成为闪动修复的具体依据；CPU SR 事件仍不等同于已与屏幕像素逐帧对齐，不声称补丁已消除视觉闪动。该年龄由所有命令列表共用的序号计算，既有 CPU 复现证明无关 list 活动也能使其超限。

主任务已让 GPT-6.1 Sol max 在独立、未安装的 `artifacts/periodic-flash-state-proof-v1-20261001/` 实现最小有界状态跟踪原型，避免无关 list 活动使有效颜色声明过期；保留真实颜色状态、Reset/Close、split/partial/aliasing、未知与容量拒绝及原 tail/texture/motion/queue 合同。没有直接删除或提高 64 门限，没有改历史、NR 算术或当前安装。主任务审阅/构建后再由 Luna 实机检查 fallback 与用户闪动是否一起消失。

完成原始采集：`live-web-perf-20261001/periodic-flash-v2-after-live-screens-v7-20261001.jsonl`，SHA `fac6752ea0a3e38c420309f32286ba8782defc99ffb5689db71b3d745c5c6172`。可复算工具为 `tools/summarize_periodic_flash_capture_v2_cpu.py`，仅用 stdlib 读取已完成 JSONL；首窗口边界、去重事件、前后 SR 与计数增量保存在同目录 `periodic-flash-v2-after-live-screens-v7-20261001-completed-summary.json`。此记录不混入筛选时切档引起的历史/图重建。

## 状态修复候选已安装（16:10 +08，实机验收待完成）

主任务审阅了独立 Sol max 交付的三份原生文件，并按 SHA 冻结到完整的独立 E 构建目录。新记录以成功 Reset 后的命令列表代次与被强引用保留的纹理身份为基础，不因其他列表消耗全局序号而过期；真实状态变更仍必须符合完整 transition，未知、容量、部分/拆分 barrier、别名及跨列表冲突保守拒绝。判断前核对 Reset/Close/ResourceBarrier hook 覆盖和实现身份。它只是已审阅的 legacy/direct、1280×720 R11G11B10 DLSS 记录状态合同，不声称覆盖 enhanced barrier、bundle、未观察的其他实现或 GPU 完成状态。

13 组针对性 CPU 行为用例及 5 项静态合同检查通过。主任务用 MSVC Release、`/MP14` 和 14 构建 workers 链接完整 ASI，并运行同一窄 CPU 用例。构建回执：`live-web-perf-20261001/periodic-flash-state-proof-v1-native-build/build-receipt.json`。新 ASI SHA 为 `c628d341b2c0d25087c926228b85d5506abc5b29d1fffc1df8d85b6a3955f3da`。

用户正常退出后，只替换 Cyberpunk 的 `plugins/CyberpunkNRBridge.asi`；旧 ASI 备份在 `periodic-flash-state-proof-v1-installed-20261001/`。模型、核、Python 源、缓存、控制选项和签名 loader 未修改。16:10 重启的子进程 PID18536 设置 `NRB_DLSS_COLOR_STATE_PROOF_SCOPE=legacy-direct-reviewed-v1`，保持原成本/闪动诊断子进程标记；无系统环境修改。编译支持存在但此精确 scope 缺失时仍走旧年龄门，新 proof unknown 时不以旧门补授权。

v2 诊断 ABI 不变。原始 source_mask 新增 bit10（新模式启用）、bit11（该记录状态已知）、bits16–31（拒绝原因）；旧 Python 解码器会将其显示为未知新增位，不能把它误判成新门未启用。Luna 应检查原始位和 fallback/skipped 计数，尤其是 age>64 时仍 ready/known/proven 且完成 NR 的正面记录。此处尚无新游戏性能、fallback 或视觉修复结果，后续由 Luna 采集及用户肉眼验收。

## 新门控启动失败，验收暂停

用户报告 NR 没有生效。Luna 的完成诊断 `periodic-flash-state-proof-v1-live-20261001/startup-failure-diagnostic.json` 显示：安装 SHA 正确、控制选项 enabled=true，但四次读取中 processing.frames=0，active_mode/source_geometry 为空，未观察到图重放或实际内核命中。90 秒采集没有启动，不接受“NR 已运行”或“闪动已修复”。

原 host 的诊断函数在首次 Python NR 回调前直接返回 `no_python_nr_call_yet`，因此本轮原生计数及新门拒绝原因是**不可用**，不能按零计数解释，也不能据此断言具体是容量、别名或实现身份导致拒绝。

主任务准备了 `artifacts/periodic-flash-startup-shadow-v1-source-20261001/`：原生新判断只旁路记录，实际 NR 授权恢复此前已跑通的旧入口（含原年龄门），通过精确子进程 scope `legacy-direct-shadow-v1` 启用。source_mask 新增 bit12 标识旁路，proven 位仍反映实际使用的旧入口，不能把它视为新状态判断已通过。原生状态观察与 NR 推理、运动、提交/回收合同保持分离。

同时移除诊断函数的首帧前返回：没有 Python NR 记录时提供空 Python 环，照常读取原生快照，另加 `python_nr_call_observed=false`。不伪造 NR 帧、不执行模型；host 除此被动函数外所有 AST 完全一致。14 组原生 CPU 行为检查与首帧前快照/缺失 DLL/空 Python 关联检查通过，ASI SHA 为 `6ee8c543e4284bb23763a55e644c25fa45fc0c672601d8996e75d0fc96c91a7e`。准备安装 ASI 与被动 host 两个文件，已核验现版及备份路径；等待游戏正常退出后替换。

该恢复版本仅用于重新获得连续 NR 和具体拒绝原因，**尚未安装、未实机验证，不宣称周期闪动消失**。旧年龄门的问题仍是已知待修项。构建/CPU 回执在 `live-web-perf-20261001/periodic-flash-startup-shadow-v1-native-build/`；后续仅由 Luna 读取实机结果。

16:40 +08，用户正常退出后已备份并安装上述 ASI 与被动 host，安装后的文件字节核对通过；备份/安装回执在 `periodic-flash-startup-shadow-v1-installed-20261001/`。新 host SHA 为 `d613d1ba88dd5c60833e48a2bc00778b9a274f775459319d46df4c548f0a7100`。游戏已重启（PID2204），`launch-startup-shadow-v1-20261001.json` 记录仅本次子进程的 shadow/成本/诊断环境；等待用户进入及 Luna 连续 NR 启动检查。这一安装动作仍不等于实机或闪动验收通过。

Luna 随后交付 `periodic-flash-startup-shadow-v1-live-20261001/one-shot-completed.json`（08:48:14 UTC）：NR 完成/合成/回收 6767 帧，图重放 6767 次，failure/generation_mismatch/invalidated 为 0，原生快照可读取。**此时实际是 540p baseline，不是 720p C512＋K8**；其约 43.84 ms 只用于证明恢复运行，不作为现役 720p 的速度结论。

当前及相邻正常 NR 帧 source_mask=464889，高 16 位为 7（aliasing），shadow 位为真，原已验证入口 proven 为真；颜色最新完整状态 PSR，age=3–23。新 observer 遇到任何 aliasing barrier 就调用 `invalidate_all(aliasing)`，把所有仍打开的列表持续标为拒绝，并在本次 Reset 前停止记录后续颜色 transition。该实机正面结果与代码定位共同解释了“新门直接授权时 0 帧，旁路后连续 NR”的启动失败。原年龄门仍在：该取样累计 original_fallback/skipped=76，不能声称闪动已消失。

## 根据实际原因准备访问状态修正版

微软文档区分了带 StateBefore/StateAfter 的 transition barrier 与用于重叠堆内存用途同步的 aliasing barrier；aliasing 描述本身不提供新的访问状态。新访问状态 observer 不再把 aliasing 通知解释成所有颜色的 StateAfter 永久失效。[ResourceBarrier 说明](https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-resource-barriers-to-synchronize-resource-states-in-direct3d-12)、[aliasing 参数及 NULL 范围](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_resource_aliasing_barrier)。

修正版明确只跟踪当前已验证 direct DLSS 路径的记录访问状态，不负责审计游戏的堆放置/物理别名内容正确性；仍使用游戏传入的当前活资源和既有前缀/私有/后缀提交、纹理、运动、生命期合同。它保留原已接受的入口，再用严格跟踪证据补充原 age 超限的帧；不因新诊断跟踪缺失而让全部原合法帧停止。age 超限时若跟踪仍未知/冲突，则仍拒绝，不盲目删年龄要求。

源码 `artifacts/periodic-flash-alias-state-v2-source-20261001/` 仅改三份原生文件及窄 CPU 用例；Python/模型/核/缓存未改。15 组 CPU 行为与启动观察检查通过，14 workers 构建的 ASI SHA 为 `8332bd30394266c1d81c25d4390e8332e25e05807b44cdb55b230197f6c3a3ed`。用户正常退出后已仅替换 ASI，旧版备份及安装字节回执在 `periodic-flash-alias-state-v2-installed-20261001/`。**已安装，实机验收待完成**；下一步由 Luna 比较同一固定场景的 NR 连续性、age>64 正面合成与 90 秒 fallback 增量，再由用户判断闪动。

此前 Luna 已恢复并确认真正的 720p C512＋K8、unrounded、融合真实运动、图重放与 reference 四项：20464–20471 共 8 个新完成帧命中实际路径，失败、回退、跳过增量为 0。这是 shadow 恢复版的短连续性证据，不是 alias-state v2 或 90 秒验收；见 `periodic-flash-startup-shadow-v1-live-20261001/RESULT-reference-eight-frame-confirmation.md`。

17:10 +08，已启动 alias-state v2，PID23144；`launch-alias-state-v2-20261001.json` 固定 ASI/host SHA 和仅本次子进程的 `legacy-direct-reviewed-v1`、成本、闪动诊断标记。当前 host SHA 仍为 `d613d1ba88dd5c60833e48a2bc00778b9a274f775459319d46df4c548f0a7100`。Luna 接手启动命中检查与固定场景 90 秒捕获；主任务只审阅完成报告。重启默认可能是 540p，因此必须先从实际执行记录确认恢复 720p，不能把开关状态当成执行成功。

## 实机定位第二处跨列表永久拒绝

alias-state v2 的完成取样证实真正 720p C512＋K8 已运行，failure=0；8 个新完成帧 8374–8381 的实际命中和 replay 成立。探针最初把 settings 与字段结构不同的 active_mode 直接比较，误报 wrong mode；按 model_controls 归一化后两者吻合。这是测试脚本比较错误，不是 NR 再次停止。

但 SR4968 的 age=102、mask=525945、高位 reason=8（cross_list），仍 raw fallback/skip；相邻 SR4969/4970 完成 NR 合成和回收，均没有 reset。短 8 帧的 age=6–34，因此没有回退；其新跟踪 reason 仍为 cross_list。90 秒修复验收暂停，不能把这轮判为闪动成功。完成证据在 `periodic-flash-alias-state-v2-live-20261001/STARTUP-RESULT.md`、`startup-eight-frame-completed.json` 和 `crosslist-readonly-addendum.json`。

代码定位：任何相同资源出现在另一张打开列表时，旧实现不但拒绝另一张列表的旧颜色记录，也把当前 own 整张列表设为 cross_list；随后 observe_color_barrier 提前返回，当前完整 transition 永远没有机会更新状态。这超出了记录访问状态的用途。D3D12 的 CPU 记录时间线允许并行，不能据它推断 GPU 执行时间线；此修正仍依赖已经验收的实际同帧提交/复制/资源合同。[微软迁移说明](https://learn.microsoft.com/en-us/windows/win32/direct3d12/porting-from-direct3d-11-to-direct3d-12)、[同一列表的状态声明检查](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-resourcebarrier)。

新 `artifacts/periodic-flash-list-state-v3-source-20261001/` 只改 `src/nr_sync_probe.cpp` 与 CPU 用例：跨列表活动只使其他列表中对应纹理的旧记录失效；当前完整本地 transition 重新声明状态时可恢复该记录。没有完整新声明的旧记录仍拒绝；资源身份/unknown/capacity/部分或拆分 barrier/矛盾链/lifecycle 等永久拒绝不被 cross_list 覆盖或清除。只有实际已知、完整 PSR 状态才能扩展 age>64 的帧。

17 组 CPU 行为及启动观察检查通过，14 workers 的 ASI SHA=`3f0667c5490bee16a5a721a8154a445c5546d6c7cc7eaca46e8416e76ab45647`。用户正常退出后已安装，原版备份及安装字节回执在 `periodic-flash-list-state-v3-installed-20261001/`；签名 loader、Python、模型、GPU 核及缓存仍不改。构建回执在 `periodic-flash-list-state-v3-native-build/`。

17:27 +08 已启动 v3，PID1036，子进程仍用 reviewed scope。启动 SHA/环境回执在 `launch-list-state-v3-20261001.json`，实机启动、超期帧正面合成与 90 秒固定场景验收交给 Luna；此时尚无修复通过结论。

## v3 仍闪动：观察范围收回当前列表

用户进入后确认仍闪。Luna 完成的 `periodic-flash-list-state-v3-live-20261001/one-shot-completed.json` 显示：实际 540p baseline 已处理 5143 帧，源资源仍是游戏 1280×720；累计原图回退/跳过 23，failure=0。当前 mask=526329、reason=8、known=false、shadow=false，当前列表的最新完整 PSR 声明也仍被其他列表的录制活动标为过期。这不是 720p 的计时证据。

v3 只修掉了“整列表永久拒绝”，仍把跨列表 CPU 录制顺序当成当前访问声明失效的依据。该 observer 的范围是**当前列表中声明的访问状态**；不能负责建立其他列表在 GPU 上执行的先后。v4 不再让其他列表录制改写这份局部声明。当前列表内真实 transition/部分或拆分 barrier/矛盾链、Reset/Close、资源身份、unknown、容量和实现身份门继续执行；已验证的实际提交/复制/等待/生命期合同继续负责执行路径。没有把 CPU 记录状态升级宣称为 GPU 执行状态证明。[微软的记录与执行时间线说明](https://learn.microsoft.com/en-us/windows/win32/direct3d12/porting-from-direct3d-11-to-direct3d-12)。

`artifacts/periodic-flash-local-decl-v4-source-20261001/` 仍只改 observer C++ 与 CPU 用例。17 组用例包含同一资源被其他列表持续录制、原诊断环覆盖及 age 扩展，同时确认当前列表自身的 UAV、部分/split、矛盾链等仍拒绝；启动观察用例也通过。14 workers 构建的 ASI SHA=`797e8c6fbee2f39017fccd4319e8e32b52d190180dac9299f34e724f6840a7a6`，安装前核验通过。完成证据输入另存到 `periodic-flash-local-decl-v4-native-build/completed-evidence-input.json`，避免修正测试派生字段时覆盖证据。

17:43 +08，用户正常退出后已备份并仅替换 ASI，安装字节核对通过；备份与安装回执在 `periodic-flash-local-decl-v4-installed-20261001/`。游戏已重启（PID12932），`launch-local-decl-v4-20261001.json` 固定本次启动的 ASI/host SHA 与子进程环境。Luna 接手实际 720p 连续帧、age 超期帧及固定 90 秒记录，视觉验收尚未完成。修复仍以 `legacy-direct-reviewed-v1` 子进程 scope 启用；普通无 scope 启动仍走旧门，不能宣称默认产品入口已更新。

Luna 的完成启动取样 `periodic-flash-local-decl-v4-live-20261001/startup-one-shot-completed.json`（09:48:27 UTC）确认 NR 已处理/合成/回收 4132 帧、failure=0，mask=4089：新观察开启、known、ready、非 shadow。此时仍为重启默认的 540p baseline；不作为 720p 性能证据。累计 9 次回退含首轮启动的 tail_unverified 窗口，不能混入接下来的固定场景增量。720p 配对连续帧及 90 秒验收尚待完成。

为避免接受修复后仍只能依赖临时环境启动，另备 `artifacts/periodic-flash-cp-default-v5-source-20261001/`。它只改变启动 scope 选择：无覆盖参数且程序名恰为 `Cyberpunk2077.exe` 时启用该已审阅路径；其他程序默认关闭，显式 reviewed/shadow/关闭或无效覆盖仍优先。当前列表 observer、访问状态/纹理/队列/等待/生命期合同与 GPU 运算均保留 v4 字节。18 组 CPU 行为和首帧前被动观察检查通过，14 workers 构建 ASI SHA=`5b5ebabc315390d5076a585263429a32aee993e1eeb5f83c85a63e3db2515c73`。此候选尚未安装，不替代当前 v4 的实机与视觉验收。

## v4 固定 720p 验收完成；普通启动版本待替换

用户在同一实机场景反馈“好像不闪了”。Luna 完成的 `startup-8frames-addendum-completed.json` 核对了 SR12343–12350 的 8 个真正新完成、合成、回收帧；720p C512＋K8、unrounded、融合真实运动、图重放与 reference 四项均实际命中。前一份短脚本比较失败另行保留，不覆盖旧回执。

09:57:09–09:58:39 UTC 的固定只读采集恰好 90 秒、90 样本：SR12735→14044，新超分 1309 次，NR 启动/处理/合成/回收增量均为 1309。**原图回退、跳过、failure、game_reset、generation_mismatch、invalidated 增量均为 0**，诊断丢事件/帧增量也为 0；启动期累计 9 次回退保持不变。首末 source mask=4089，开启/known/ready、非 shadow。原始 `flash-capture-90s.jsonl` SHA=`25f7dfdb409ab0c7abadd922c3e9285ff07ab49c540d7bcc85cdeb1fca3ce235`，主任务使用已完成文件 CPU 复算得到 `main-capture-summary-completed.json`，去重回退窗口为 0，与计数增量相符。

这轮在已测范围内通过连续 NR 检查，并有用户视觉改善反馈；不扩大成所有场景或永不闪动。窗口中可见记录最大 age=57，没有直接捕获 age>64 的正面完成帧；该边界的作用仍由针对性 CPU 用例确认。`negative_evidence_complete=false` 与非事务式计数限制保留，不把 CPU 事件当作屏幕逐帧证明。

主任务已将普通启动版的 8 份原生源码（CMake、ASI、sync observer、deferred 诊断接口及 CPU 用例）合入项目主源码，核对其余原生文件与已编译冻结源一致；旧源码逐文件备份与 SHA 在 `periodic-flash-cp-default-v5-canonical-backup-20261001/promoted.json`。只写 E 主源码，未写正在运行的 G ASI/Python/模型/核/缓存。v5 的安装前文件核验通过，等待游戏正常退出后仅替换 ASI，并核对无 scope 环境的实际启动。

18:06 +08，用户正常退出后已备份并仅换入 v5 ASI，实际安装字节核验通过（`periodic-flash-cp-default-v5-installed-20261001/installed.json`）。重启 PID20156 的 `launch-cp-default-v5-20261001.json` 明确 `scope_override_present=false`：仅对子进程移除该覆盖参数，保留本次成本/被动诊断，未修改系统环境。新 observer 应由确切 Cyberpunk 可执行文件 profile 自行启用；Luna 接手最小启动/720p 连续帧与 30 秒回退增量检查，普通入口验收待完成。

## 普通启动入口验证完成

用户确认已进入。Luna 的 `periodic-flash-cp-default-v5-live-20261001/completed-v5.json` 核对实际 ASI/host SHA 与无 scope 参数的启动回执相符，新判断 bit10/known bit11 为真、shadow bit12 为假、reason=0。恢复一次 720p C512＋K8、unrounded、融合真实运动、图重放及 reference 四项后，8 帧短窗口处理/合成/回收各 +8，回退/跳过/失败/reset 各 +0。

18:09:57–18:10:27 +08 的约 30 秒捕获（单调时钟 29.969 秒、30 样本）处理/合成/回收各 +397；source_proven、seen、delegated/submitted 增量也为 397。新增原图回退、跳过、failure、game_reset、generation_mismatch、invalidated 与诊断 dropped 均为 0。主任务仅对已完成原始文件 CPU 复算，去重回退窗口为 0，与计数增量一致。原始 `flash-capture-30s.jsonl` SHA=`ddb5f4c39b7045808399275197f9c569276e8489f47f67c036085340987b35f4`；复算输出为同目录 `main-capture-summary-completed.json`。

实际 C512、K8、DecoderGather、C32 与 front 命中、graph route=replay。静止取样的 history fractional 条件站点没有执行，准确记录为 `conditional_unexercised`，不把选中状态当作该条件站点已命中。非事务式计数、`negative_evidence_complete=false` 和 CPU 事件非像素证明的边界继续保留。

结论：当前已安装的 v5 在已测 Cyberpunk 1280×720 R11G11B10 direct DLSS 输入路径中，普通无 scope 启动即可启用修正；v4 的 90 秒连续 NR 与用户视觉改善、v5 的普通启动/短连续性均通过。原生修正已合入 E 主源码，回退 ASI 与源码保留；本次没有新的后端速度结论。后续性能实验继续沿现役 720p C512＋K8 开展，不把默认入口激活推广为其他游戏/尺寸/HDR 全面验收。
