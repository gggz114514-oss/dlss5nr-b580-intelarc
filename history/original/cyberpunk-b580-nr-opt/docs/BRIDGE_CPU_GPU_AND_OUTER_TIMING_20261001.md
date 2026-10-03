# 桥的 CPU/GPU 边界与网页外计时（2026-10-01）

适用现役《赛博朋克》720p C512＋K8、去舍入、融合真实运动、图重放。用户认可将约 15 ms 的整游戏/网页差值作为独立优化任务；GPT‑6.1 Sol max 分别实施 immutable 管线缓存、完整 C128/C256 MLP 和 ViT score/value 原生矩阵候选。主任务负责集成和外圈计时，Luna 负责按序 GPU/实机测试。

## 桥的图像处理已经在 GPU

`nr_hdr_proxy.cpp` 录制颜色代理、运动转换与残差合成计算着色器。现役纹理桥 `native/nr_texture_bridge_re8_v1.cpp` 使用 D3D12 DEFAULT heap 的共享缓冲、Win32 NT handle 导入、XPU device USM 指针，以及 GPU pack/unpack 与 device memcpy。不是将整帧回读 CPU、处理像素再上传。

CPU 仍负责 D3D12 录制/提交、Python 调用、配置和资源寿命，以及 `WaitForSingleObject` / `wait_and_throw` 完成等待。`HdrProxy` 当前每帧新建，重复编译同一 prepare/composite 着色器与创建 root/PSO。固定编译可以移冷；逐帧可写 descriptor/纹理、在飞 command allocator 和退休依赖不能一起无条件共享。

相同源码和 flags 的双入口编译离线均值为 4.8551 ms。它没有测量实机关键路径，也不证明整个约 15 ms 来自编译。记录见 `REMAINING_720_BENEFIT_FORECAST_20261001.md`。

## GPU 队列直接接力的可行性边界

本地 oneAPI 2026.1 的 `sycl/ext/oneapi/bindless_images_interop.hpp` 明确提供 `external_semaphore_handle_type::win32_nt_dx12_fence`；`bindless_images.hpp` 提供 import/release 与 queue 的 timeline wait/signal。因此工具链中存在把 D3D12 fence 导入 XPU、让 GPU 队列直接依赖的 API。

这仅证实头文件接口存在，尚未验证 B580 当前 Level Zero/Unified Runtime 的实现支持、驱动能力，以及现役 PyTorch 所借 SYCL context/queue 的兼容性。不能据此直接删除 CPU 等待。后续最小探针应只创建共享 fence 与小共享缓冲，验证 D3D12→XPU→D3D12 两方向的值和完成顺序，并在明确不支持时返回诊断；通过后才改 prepare/export，保留 consumer 退休和错误处理。即使减少 CPU 唤醒，依赖的 GPU 工作和等待也不会凭空消失。

本轮首先修复已确认的重复编译/创建，异步 fence 不混入该数学不变的缓存对照。

### 现役纹理桥里 CPU 实际做了什么

只读核对 `native/nr_texture_bridge_re8_v1.cpp` 的 `prepare` / `read_inputs` / `export_frame`，不是从 Python 墙钟推断执行设备：

| 顺序 | 像素/运动工作的位置 | 当前 CPU 完成等待 |
| --- | --- | --- |
| 输入纹理打包为共享 RGB/运动缓冲 | D3D12 compute shader | `submit()` 用 ready fence + `WaitForSingleObject` 等完成 |
| 共享缓冲复制到模型输入张量 | SYCL device `memcpy` | RGB、运动各一次 `wait_and_throw` |
| NR 图重放及历史处理 | XPU/GPU | 依现役模型/计时路径等待完成 |
| NR 结果复制到共享输出缓冲 | SYCL device `memcpy` | `wait_and_throw` |
| 缓冲解包为 D3D12 输出纹理 | D3D12 compute shader | `submit()` 再等 ready fence 完成 |

此外，`prepare`、`export_frame` 开头各有整条 XPU queue 的 `wait_and_throw`。这些等待的墙钟包含 GPU 前序工作，不能称作 CPU 算图，也不能直接认定全部是可省的 CPU 开销。两次设备内复制同样不是 GPU→CPU→GPU 上传下载。

GPU 接力候选的目标是用导入的 timeline fence 将上述跨 API 依赖交给队列，减少 CPU 唤醒/阻塞和过宽等待。它不会消除生产者和消费者的数据依赖；现役 Python/PyTorch 仍需 CPU 配置及提交，完全无人提交的 GPU 链路不是本轮改造。首个独立能力探针位于 `artifacts/dx12-xpu-fence-probe-v1-20261001`，要求使用 PyTorch 实际借出的 queue/context、相同 adapter LUID、两轮 timeline 与真实共享数据验证；尚未得到运行时支持或速度结论。

## 新的计时口径

隔离源码：`artifacts/bridge-outer-timing-v1-20261001/payload`。新增独立 `NRB_RecordTimes` / `NRB_GetRecordTimes`，原 `NRB_StageTimes` ABI 与现役 process 计时保持不变。

| 字段 | 测量范围 |
| --- | --- |
| record | Evaluate 中桥录制入口至 Pending 发布前的 CPU 墙钟 |
| resources | 前置/超分 allocator、list、completion fence 与颜色副本资源创建 |
| hdr_initialize | 颜色/运动代理初始化，包含重复着色器编译和 immutable 管线创建，也包含逐帧代理资源 |
| xess_record | 调用原 XeSS/Opti evaluator 向私有列表录制命令的 CPU 墙钟，不是 XeSS GPU 执行 |
| record_to_submit_gap | Pending 发布前至现有 native 处理计时起点；含其间游戏录制/提交，不能称纯桥耗时 |
| record_to_retire_span | 桥录制入口至该 NR 帧合成完成并 retire 返回；含游戏/排队/执行，仍不是 Present 或完整游戏帧时间 |

只在成功完成且录制/执行均启用计时、epoch 相同的帧发布。六个外圈字段用同一最近 32 个有效完成帧窗口和 `last_nr_frame_id`；切换计时或 NR controls 后清空窗口，缺失不填零。旧 native process 与外圈窗口可能在切换的首帧样本数不同，不能直接混加窗口均值。完整同帧满足 span≈record＋gap＋native process；独占 subphase 是 record 的子项，不能再次相加。

网页候选显示外圈 CPU 录制及其三项子阶段，同时注明 gap/span 包含游戏工作。完整约 15 ms 的定量闭合仍需同场景缓存 ON/OFF 的原生记录和 Present/RTSS 配对；网页 NR 时间快并不自动证明游戏帧时间等额减少。

## 验证状态

主任务完成纯 CPU 编译（14 build workers，保留 `/EHsc` 异常展开）、55 项时间窗口/非法样本/ABI 检查和 18 项现役状态证明检查。immutable 缓存 worker 已完成 CPU 验证并被合入 `artifacts/bridge-cache-integrated-v1-20261001/payload`，包含独立网页复选框；该合并版再次通过 CPU 构建及上述检查。缓存同时支持 OFF 对照与 ON 复用，不共享逐帧可写资源。随后由 Luna 完成 GPU 对照，主任务备份、定点试装并普通启动游戏；矩阵候选尚未安装。主任务没有执行 GPU 测试。

可复现脚本：`tools/prepare_bridge_outer_timing_v1_cpu.py`、`tools/build_bridge_outer_timing_v1_cpu.py`。构建回执和日志在 D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/bridge-outer-timing-v1-build。

## 缓存 GPU 最小验收与试装

Luna 已在同一 B580 D3D12 设备跑完 44 组 OFF/ON 对照（88 个 arm 帧），包括颜色 prepare、真实 RG16F 运动的缩放/关闭，以及零/非零残差 HDR composite；共比较 973,209,600 个有效字节，全部一致。稳态 ON 共 32 次初始化只编译两次、创建一个 root 和两个 PSO；后 31 次命中没有新增读文件/hash/编译/root/PSO。四次显式 OFF→ON 往返通过，OFF 的编译及创建计数按预期增加，返回 ON 复用旧管线。

这是固定合成输入/残差的 GPU 边界验收，不是整网画质、设备丢失或并发游戏线程验收，也没有 GPU graph/实机加速结论。首次探针的编译 shell 环境失败及两次测试代码编译修复保留了日志，没有修改产品来绕过门；通过回执为 [GPU-RECEIPT.json](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/bridge-cache-gpu-probe-v1-retry3-20261001/GPU-RECEIPT.json)。

主任务核对 CPU 合并版、GPU 实测 cache/shader pins、现役 ASI 及三份 Python 基线后，已备份并定点试装四个文件：ASI、插件 web/adapter、物理 nr-runtime/game 的控制面板。没有通过插件 junction 写入，也没有改后端模型、输入档、用户选项或同步/退休逻辑。新 ASI SHA256 为 `5a704ca4039e1a739eb0691fede12267d8cb39ad6b4485bf1067048b240ef6ff`。

备份与安装回执：[installed.json](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/bridge-cache-trial-v1/20261001T150507112380Z/installed.json)。默认缓存 ON，独立网页复选框可关闭作对照。用户进入同一静止场景后，Luna 完成了下面的实际配对。

## 实机缓存配对：当前 540p 设置

本轮保留启动时的用户配置：游戏源 1280×720，NR 实际输入 **540p**、unrounded、融合历史、图重放；720p 数学候选列表为空、生命周期迁移关闭。因此该结果不能标为 720p C512＋K8/reference4/V6 的验收。DLSS 性能、FG 关闭、同场景静止由用户确认。PID 14680，15:13:20–15:14:18 UTC。

OFF/ON/ON/OFF 四臂，每臂舍弃至少 8 秒及 35 个新完成帧，再采 30 个独立 RecordTimes 完成帧的 last 值。主任务从四份原始数据重算了均值、中位/P95、唯一递增帧号及缓存计数，与 Luna 汇总一致；前后 76 项 source pins 不变。

| 完成帧口径 | OFF 平均 ms | ON 平均 ms | 减少 ms |
| --- | ---: | ---: | ---: |
| HDR 初始化 CPU | 6.7898 | 1.8053 | 4.9845 |
| 网页计时之外的 CPU 录制总段 | 8.7414 | 3.7860 | **4.9554** |
| 录制入口至同一 NR 帧回收的跨度 | 54.9522 | 50.0121 | 4.9401 |
| native processing 段，独立样本集 | 45.0369 | 45.0632 | −0.0263 |

两个 ON 臂的 CPU 总录制均低于两个 OFF 臂；OFF 首尾漂移仅 0.0029 ms。ON 两臂未新增 shader read/hash/compile/root/PSO，分别命中 110/111 次；OFF 分别 102/103 次未缓存初始化，对应 204/206 次编译和 PSO 创建。全程 observed health.failed=false，编译失败增量 0；没有暴露的累计回退/失败/重置和 GPU graph cost 均记 unavailable，不据此填零。

这里的 **4.9554 ms 是新的实机 CPU 录制差值**，不是先前离线编译数据。录制至回收跨度包含游戏工作和等待，不是纯桥成本、Present 或完整游戏帧时间；没有同窗口 RTSS 读数，暂不声称游戏 FPS 或完整约 15 ms 差额已闭合。native 段几乎不变符合缓存优化发生在其计时之前的源码边界，网页 NR ms 看不出这项收益是预期。

已恢复缓存 ON 并取得 8 个新健康完成帧，原分辨率、模型/历史/图及计时设置保留。证据：[Luna REPORT](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/bridge-cache-live-pair-v1-20261001/REPORT.md)、[主任务原始重算](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/bridge-cache-live-pair-v1-20261001/MAIN_REVIEW.json)。可复现重算入口 `tools/review_bridge_cache_pairs_cpu_v1.py`。Luna 接着只在已验证的现役 720p 组合中复测缓存开关，结果单列。

缓存与计时的 12 个改动源文件已按构建 source pins 合入 E 项目/共享控制面板，默认缓存开启；合入没有改 G 正在加载的文件，没有新增数学改动。逐项核对原 canonical SHA、保存 D 备份后写入，写后字节核对通过。备份与回执：[promoted.json](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/bridge-cache-source-merged-v1/20261001T154016172170Z/promoted.json)。可复现定点合入脚本为 `tools/promote_bridge_cache_source_v1_cpu.py`。

## 剩余 CPU 工作的两项后续

720p 已在同一游戏进程和静止场景完成补测：C512＋K8、reference4＋probability＋C128、unrounded、真实运动融合、图重放、V6 已实际命中。仍为 OFF/ON/ON/OFF，每臂 30 个唯一递增完成帧，主任务原始重算通过。CPU 录制 **8.6306→3.6768 ms，减少 4.9538 ms**；HDR 初始化 **6.7095→1.7465 ms，减少 4.9630 ms**。录制至回收跨度 59.8465→55.1294 ms，native process 独立样本 50.0722→50.3326 ms，后者没有加速；不能把 CPU 缓存称为推理核提速。两个 ON 臂均低于两个 OFF 臂，76 项源 pin 不变，已恢复缓存 ON 和目标组合并取得 11 个健康新完成帧。成本分解/累计回退计数/同窗口 RTSS 未暴露，保留 unavailable。

证据：[720p RESULT](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/bridge-cache-live-720-pair-v1-20261001/RESULT.json)、[720p 原始重算](D:/Codex-NR-Experiments/cyberpunk-opt/numeric-game-stage-720-20260930/live-web-perf-20261001/bridge-cache-live-720-pair-v1-20261001/pair/MAIN_REVIEW.json)。两次过渡前 CPU 脚本错误已保留：误读 exporter 清单格式、把新增复制数误认为总缓存数；修正为 674=98 新增＋576 相同复用。两次都未发送 API 请求，不是新的 NR 故障。

用户认为 ON 后约 3.79 ms 仍大，要求尽可能由 GPU 完成。像素处理已经是 GPU；资源创建与命令录制仍为 host API 操作，GPU fence 接力不能自动删除这段 3.79 ms。当前 HDR 初始化仍每帧创建 RGBA32F、RG16F 两张中间纹理和私有五项 descriptor heap，具备明确的复用入口；单独测得的资源准备 CPU 约 0.949 ms，XeSS 命令录制约 0.637 ms。这些为嵌套子项，不能再加到 3.79 ms 或视为全部可省；现有资源准备计时也未覆盖所有后续 allocator/fence 创建。

`artifacts/bridge-resource-pool-v1-20261001` 开发默认关闭的有界帧资源池候选，只在精确 completion 成功且 processor retire 成功后归还；在飞 descriptor/纹理不重绑定，不改状态判断、批次拆分、历史/运动和闪动修复。先集中 HDR 纹理/heap，命令资源可保持原逐帧方式，避免一次引入无界缓存或资源寿命变化。没有 GPU 验收、实际减时或 FPS 结论。

另一路 `artifacts/dx12-xpu-fence-probe-v1-20261001` 验证实际借用 PyTorch SYCL queue/context 的两轮双向共享 fence 和共享数据。它针对跨 API CPU 阻塞等待，仍不保证驱动支持或删除全部 GPU 依赖时间。先做 CPU 编译，Luna 在游戏退出后独占 GPU 执行；不在现役桥中直接删等待。完整 MLP/ViT 计算候选继续并行，均由 Luna 串行测试。

共享 fence 探针主任务 CPU 编译已通过：SYCL DLL、MSVC ABI/数据公式/HLSL、独立 watchdog、导出/依赖检查及 source pin 通过，14 device-link 并行上限，未加载探针或运行 GPU。[CPU 回执](D:/Codex-NR-Experiments/cyberpunk-opt/dx12-xpu-fence-probe-v1-20261001/cpu-06xhjf2l/CPU_COMPILE_RECEIPT.json)。用户已正常退出游戏，Luna 获得单次有界运行 GO。现役 SYCL DLL 实际位于物理 runtime 的 `python/Library/bin/sycl9.dll`，不存在假定的 runtime/bin；不更新或替换库来伪造当前驱动支持。

## 2026-10-02：共享栅栏的具体故障与可用边界

Luna 在游戏退出后的独占窗口实际执行，主任务负责 CPU 编译、实现改动和原始数据复核，未运行 GPU。v1 在提交 XPU signal 时阻塞；监督器有界结束子进程，未据此宣布导入不支持。v2 先提交 D3D12 producer 再提交 XPU signal 后，两轮均完成，但第二轮的全部 256 个字与 `XPU modify → D3D12 produce → D3D12 consume` 完全相符，正确顺序应为 produce → modify → consume。v3 增加显式 handler 依赖，结果相同。

v3 的 UR trace 显示 waitValue **10/11**、signalValue **20/21** 均正确，不能把故障归为数值参数漏传。相同 in-order queue 的依赖在 UR 层没有额外 wait-list。仅改子进程 `SYCL_UR_USE_LEVEL_ZERO_V2=0` 的旧 adapter 对照也失败；这不是只凭远程新版源码就能归因给 V2 的证据。本轮未改变系统环境、驱动或已安装运行时。

v4 让两轮使用不同前向 D3D12 fence，**512/512 字正确**。随后 v5 保留同一个前向 D3D12 fence，仅分别导入为两个 external semaphore 对象，也取得 **512/512 字正确**；反向 signal fence 和 GPU feedback 仍复用。两轮使用现役 Torch 的同一 queue/context，全部 wait/signal 在 GPU 依赖图中，只有最后审计 readback 的有界 CPU 等待；清理已确认完成。由此收窄为“重复使用同一个前向导入对象等待新的 timeline 值失败”，不能称为 GPU 接力根本不可用，也不能称为实际游戏加速已经实现。

证据：[v3 失败与 trace](D:/Codex-NR-Experiments/cyberpunk-opt/dx12-xpu-fence-probe-v3-20261002/runtime-trace-v1/RUNTIME_RECEIPT.json)、[v4 主任务逐字复核](D:/Codex-NR-Experiments/cyberpunk-opt/dx12-xpu-fence-probe-v4-20261002/runtime-trace-v1/MAIN_ORDER_REVIEW.json)、[v5 实测](D:/Codex-NR-Experiments/cyberpunk-opt/dx12-xpu-fence-probe-v5-20261002/runtime-trace-v1/RUNTIME_RECEIPT.json)。所有旧失败、源及回执保留。随后 v6 扩大到同一 D3D12 fence、32 个独立导入、8192 字和反馈依赖，由 Luna 实测通过：95 次 GPU wait、96 次 GPU signal，只有一次最终 readback 审计 CPU 等待；同一借用 queue/context、in-order、终值 41/51/61 和实际清理通过，无 quarantine。主任务以独立整数公式重算全部 8192 字并核对编译源/DLL pins：[原始回执](D:/Codex-NR-Experiments/cyberpunk-opt/dx12-xpu-fence-probe-v6-20261002/runtime-v1/RUNTIME_RECEIPT.json)、[独立复核](D:/Codex-NR-Experiments/cyberpunk-opt/dx12-xpu-fence-probe-v6-20261002/MAIN_ORDER_REVIEW.json)。v6 仍在开始时导入全部对象，不证明动态逐帧导入/退休的资源寿命或整桥速度。

`artifacts/gpu-handoff-bridge-v1-20261002` 按上述结果实现独立的、默认关闭的 native 候选。计划取消的是帧内 host 阻塞接力，保持同一模型队列、GPU copy、运动/历史语义和消费完成证明。需要不同 pack/unpack 记录，禁止尚未完成便 reset allocator 或重绑定；逐帧前向导入的生命期保留到真实消费完成。现役 ASI 的 prepared CPU wait 仍存在，native helper 单独改变不能声称整桥已实现 GPU 接力。

候选 CPU 构建已通过：268 项 lease/完成状态检查、8 项 Python wrapper 检查、4 个原 HLSL 入口编译，以及现役旧 DLL ABI 导出保持。主任务审查了实际 prepare/export 和资源寿命，并独立复核 14 个构建源 pin、14 个输入 pin 与候选/探针二进制：[构建回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-bridge-v1-20261002/cpu-u3a97ea9/CPU_COMPILE_RECEIPT.json)、[主任务复核](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-bridge-v1-20261002/MAIN_BUILD_REVIEW.json)。Luna 已获独占运行 GO，下一门是实际旧 DLL／新 OFF／新 ON 的 34 帧纹理比较（其中 32 帧 GPU 接力），再跑同一 Torch current stream 的 32 帧图重放。尚无这两门的通过或实机减时结论。独立审核脚本为 `tools/review_gpu_handoff_product_probe_cpu.py`；它不加载 GPU DLL，只核对冻结文件和运行回执。

后续 owner 保留修复增加第 9 项 wrapper CPU 检查，形成 `cpu-5_vwajxn`。第一轮监督器在 GPU 启动前发现旧构建与正在修改的源码不一致，拒绝执行，子进程已回收。新构建的第二轮实际完成 34 帧／204 次颜色、运动和输出字节比较，32 次独立前向导入、真实退休和释放均匹配，32 次 prepare 在 gated producer 未完成时返回；最后配置关闭、idle/健康和实际消费者值也正确。然而 native 清理阶段 40 出现访问异常，`cleanup_ok=0`，Python 图重放尚未开始，**整个门仍失败，禁止据此安装或宣告加速**。[原始运行](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-bridge-v1-20261002/runtime-v2/RUNTIME_RECEIPT.json)、[最后完成计数](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-bridge-v1-20261002/runtime-v2/NATIVE_PROGRESS.json)。所有 14 个源及 14 个输入 pin 均未变化，监督器正常回收子进程、无游戏运行。

主任务将经过复核的 14 源冻结复制为 `artifacts/gpu-handoff-bridge-v1-20261002-main-frozen-cpu5`，消除开发与测试写入竞态。另建 `artifacts/gpu-handoff-cleanup-v2-20261002`，只增加逐个 helper close、USM free、模块卸载和 State 析构的诊断位置；native 候选、数学/字节门及原清理动作保持，下一轮用于确定访问异常具体发生在哪一步。没有绕过清理门，也没有替换 G: 游戏文件。

细化探针再次由 Luna 执行，最后位置为 stage 83：三个 helper close、全部 USM free、四个 NT handle close、两个 `FreeLibrary` 均返回成功，随后析构仍持有 SYCL event/queue 的 `State` 时访问异常。主任务据此推断是测试提前卸载 helper 代码、而借用的 Torch context 仍拥有相关对象。v3 仅在探针中以精确函数地址固定两个 helper 模块至借用 context 的进程寿命，仍执行原 helper close、全部 GPU/审计内存释放、平衡的 LoadLibrary 引用释放和完整 State 析构；这不是保留在飞 GPU 数据来跳过回收门。Win32 的模块固定行为见 [GetModuleHandleExW](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-getmodulehandleexw)。`cleanup_ok` 延后至 State 析构成功后才设置。native 产品候选源码未改，新的固定源码/CPU 回执和后续 GPU 结果独立保存于 `gpu-handoff-cleanup-v3-20261002`，尚未据此宣布退出或图重放通过。

## 共享纹理与图重放门已通过

Luna 的 cleanup-v3 实测正常退出，监督器未触发，14 个源码与 14 个输入 pin 前后相同。原生路径完成 34 帧（32 个 GPU 接力帧、2 个 OFF 对照）、204 项颜色/运动/输出字节比较、102 项非零运动检查和 128 项在飞拒绝检查。32 个前向导入均真正退休并释放，最终 idle、active=0、poisoned=0，State 析构完成。随后同一借用 Torch 队列完成 32 个共享纹理帧、3 次图捕获、96 次重放，64 项输出纹理、128 项输入及 96 项运动检查通过，张量引用与 GPU 资源实际回收。测试固定两个 helper 代码模块到借用 context 的进程寿命，仍明确关闭并释放其 GPU 资源；不宣称模块完全卸载。

主任务独立复核原始回执、计数、源及二进制：[运行回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-cleanup-v3-20261002/runtime-v1/RUNTIME_RECEIPT.json)、[主任务复核](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-cleanup-v3-20261002/MAIN_RUNTIME_REVIEW.json)。这是实际 native helper 与合成图的协议检查，未加载完整 NR 模型，未计性能，也不是游戏画质/持续运行验收。

游戏/网页接入候选已通过 76 项协议、48 项 host/web 和 6 项 CTest；主任务另核对 69 个 payload pin、61 个编译源码 pin、128 字节 native Info 字段及原 ABI：[接口复核](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-product-v1-20261002/MAIN_COUPLING_REVIEW_03.json)。定点安装器 `tools/install_gpu_handoff_trial_v1.py` 的八文件 dry-run 通过，GPU 接力与两槽资源池均默认 OFF。随后实际备份并安装八个文件，逐项核对写后 hash；数学模型源、输入档、用户参数和缓存未改：[安装回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-product-trial-v1-20261002/20261002T032023010926Z/installed.json)。旧离线 fixture 的前端源 pins 随这次有意安装而改变，不能冒充仍未改变或为通过旧守卫而回写旧文件。

2026-10-02 03:29:35 UTC 以普通 `--launcher-skip` 启动赛博朋克，PID 17484；启动器再核对八个已装 hash，并仅在游戏子进程移除临时 source-scope override，未改系统环境：[启动回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-product-trial-v1-20261002/launch-20261002T0329358916625Z.json)。同场景实机开关配对尚待 Luna 执行。模型末尾同步与最终批次等待保留；上一消费复用等待发生在现有 NR 计时起点之前，须同时看完整同帧跨度，不能把绕过次数称为省时或完整异步推理。

## 中间纹理池的检查状态

两槽、64 MiB 上限的独立资源池已完成 CPU 构建，127 项 lease 检查、8 项实际 C ABI 控制检查和 5 项 CTest 通过。新增 pool API 为 168 字节，不改变原 Controls/StageTimes/RecordTimes。网页勾选与“着色器管线复用”分离，默认关闭，HTTP 只切换 native 资源池，不重建模型或历史。主任务额外验证了 ABI、开关往返、非法类型/令牌、native 拒绝，以及模型设置不变。[网页 CPU 回执](D:/Codex-NR-Experiments/cyberpunk-opt/bridge-resource-pool-web-v1-20261002/cpu-01/CPU_CHECKS.json)。

原完整调试层 GPU 探针因本机 `D3D12 debug layer unavailable` 返回 77，**字节验收未执行**，保留该结果为未验证。[原探针结果](D:/Codex-NR-Experiments/cyberpunk-opt/bridge-resource-pool-v1-20261001/gpu-byte-luna-run02-result.json)。另一个独立 v2 探针保持原 96 项字节、绑定、真实完成栅栏和计数要求，仅允许缺少可选调试层时继续；将 `debug_layer_available=false` 与 `debug_errors=null` 明确写入，不能冒充原完整调试门通过。Luna 随后实测 v2 通过 96 次比较、16 次绑定变化及 OFF→ON→OFF，2 次创建、14 次同 canonical resource 命中，共 39,321,600 字节纹理，最终关闭且子进程正常退出。测试使用真实 HDR adapter 和合成 consumer，不是 NR/XPU 或游戏验收：[GPU 字节回执](D:/Codex-NR-Experiments/cyberpunk-opt/bridge-resource-pool-byte-probe-v2-20261002/gpu-byte-run01/gpu-byte-receipt.json)。主任务逐项复核源、二进制、shader、父监督器、字节 hash、CPU/Web ABI 和现役四文件 pins，`tools/install_bridge_resource_pool_trial_v1.py` 已完成 dry-run；截至本段记录尚未安装资源池、取得实机收益或完成真实 processor/游戏持续运行验收。

## 下一步接入边界

`artifacts/gpu-handoff-bridge-v1-20261002` 负责同一借用 Torch 队列的 native prepare/export；`artifacts/gpu-handoff-product-v1-20261002` 隔离实现游戏 prepared 等待的能力协商、当前 callback 的真实 producer fence/value 和独立网页复选框。原 Frame/Processor/Controls ABI 与默认等待路径保留，不能只靠网页勾选便跳过 producer 等待。必须由 owning NR callback 验证 native helper 已配置、同 queue/context/LUID 和每帧实际生产者证明后才能启用；之前的完成栅栏、命令列表状态、输出寿命及闪动修复仍是必要依赖。

本轮主任务未运行 GPU、未更换驱动或系统运行时。矩阵候选的真实现役中间张量导出也由 Luna 完成：原 export-01 因 GBK 默认读取工具链源而停止；export-02 仅以子进程 `-X utf8` 重试，155 个源 pin 和只读 baseline JIT guard 保留，13 帧完成、取得两个 owned MLP site 与真实 scaled ViT Q/K/V。该离线素材用于矩阵差异/模块时间，不是赛博朋克帧率或画质验收。单独的 `owned_mlp_report.py` 只记录差异和时间，不借任意数值门限自动宣告画质接受。

## 2026-10-02：纹理复用实机收益与 GPU 接力导入修复

Luna 在赛博朋克同场景 720p C512＋K8、去舍入、融合真实运动、图重放、现役六项组合和 V6 上执行纹理池 OFF→ON→ON→OFF。每臂预热至少 8 秒和 35 个完成帧，采 30 个不同的真实完成帧；主任务独立重算：CPU 录制平均 2.91673→1.75096 ms，省 **1.16577 ms**；录制到回收跨度平均 57.44341→55.99714 ms，省 **1.44627 ms**。两次 ON 均低于两次 OFF，OFF 漂移分别 0.12127/0.12349 ms。开启区间实际 383 次获取、382 次复用、1 次创建和 383 次归还，0 隔离、0 创建失败；仅占用一个 20 MiB 纹理槽。[原始重算](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/main-driver-run02/MAIN_POOL_RAW_REVIEW.json)、[Luna 运行核对](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/main-driver-run02/POOL_COMPLETED_RECEIPT.json)。跨度包含游戏等待，不能换算为 Present 帧率或纯桥 GPU 时间；仍待组合测试与用户画面检查。

同轮 GPU 接力 ON 未启用，host 明确返回 `OPTIONAL_NATIVE_ABI_UNAVAILABLE`，prepared bypass 为 0。NR 保持健康，脚本恢复两项 OFF。主任务查明 `fp8_unround_overlay/bootstrap.py` 把 overlay/modules 放在 game 之前：游戏实际导入的是旧 `experimental/fp8_unround_overlay/modules/nr_texture_bridge_v1.py`（SHA a3c326…），此前更新的 game 副本（SHA 8a0cf…）没有用于该对象。**本轮不能作为 GPU 接力提速或减速结论。** 79 个 G 源 pin 在运行后核对仍一致；E 测试 driver 的后续修改发生在 run02 结束后，旧 driver SHA 保留，不重标旧运行证据。

人类退出后，主任务仅替换这个实际导入的 Python 文件，沿用协议已通过的 8a0cf… wrapper；备份旧文件并逐项核对九文件安装。SourceFrame、TextureOutput 和旧纹理 ABI 保持，未重排全局搜索路径、未修改原 modules、模型数学或缓存。[单文件修复回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-import-repair-v1-20261002/install-01/installed.json)。以普通游戏入口重启 PID 16408，[启动回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-product-trial-v1-20261002/launch-20261002T0418538818826Z.json)。更新的测试 driver 通过 82 个源 pin 的 CPU 协议检查；人类确认回到同场景后，由 Luna 单独执行 run03 的两项开关配对和 90 秒连续运行。结果未出前不提升 GPU 接力为默认值。

run03 的纹理池四臂再次完成，两次 ON 都更快：录制均值 3.27003→2.03025 ms，省 **1.23978 ms**；录制到回收跨度 54.98474→53.18207 ms，省 **1.80267 ms**。这与前轮方向相同，仍不是 Present/FPS。[第二轮主任务重算](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/main-driver-run03/MAIN_POOL_RAW_REVIEW.json)。GPU 接力在 ON 切换时报 `GPU handoff setter requires its owning host thread`，NR 锁存失败、prepared bypass 仍为 0；测试脚本已关闭两个新开关，没有把失败前的 OFF 路径计为 GPU ON。

主任务另从同一批快照核对 `processing.frames == recording.last_nr_frame_id`，只比较匹配的网页上一帧时间，四臂保留 30/29/30/30 个样本，剔除一个不同步快照且未补零：网页处理均值 50.61921→50.02556 ms，省 0.59365 ms，两次 ON 仍均低于两次 OFF。该计时包含 NR 准备/调用/合成/尾段，排除了更早的 CPU 录制；它与上述跨度、录制节省互有重叠，**不得相加**。[网页时间原始重算](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/main-driver-run03/MAIN_POOL_WEB_RAW_REVIEW.json)。主任务仍未采 RTSS/Present，不能把这些值称为基础帧率实测。

最新 Python 栈（2026-10-02 04:22:45 UTC）与代码说明具体线程合同冲突：现役 `cyberpunk_nr_adapter.process` 在 game queue worker 改变时只更新 Python 的 `bridge.thread` 和 numeric owner，而 native bridge 的创建时 DWORD owner 未转移。主任务在隔离 `gpu-handoff-owner-transfer-v1-20261002` 实现显式 native owner 交接、对应 wrapper 和 adapter 调用；交接须同借用队列、旧 native owner 匹配、真实 consumer 与 XPU events 退休，busy 不改 owner；原正常入口线程检查保留。process/retire 使用同一 adapter 串行锁，线程交接不重新创建 GPU 队列、图、输入映射或历史。候选 DLL SHA 8875a437… 已以 14 个链接工作线程编译，原导出、数据 ABI 与 HLSL 保持：[CPU 构建回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-owner-transfer-v1-20261002/cpu-build-03c41unu/CPU_BUILD_RECEIPT.json)。此时尚未安装；下一门由 Luna 在游戏退出后测真实两线程交接和纹理输出，成功后再实机配对。

### 线程交接已通过并试装

首个两线程 GPU 探针 `runtime-2thread-01` 在第 2 帧的错误队列断言失败，子进程正常回收、未触发 watchdog。主任务定位为测试口径问题：三个错误调用共用一次计数快照，而每个病例之后的 `gpu_handoff_info` 会正常回收已完成的上一帧；下一病例把该正常回收误认为非法调用改状态。只修测试为每个错误调用立即前后单独取全字段快照，断言前记录原始值和差异；新增 CPU 回归确认真实非法计数变化仍能被检出。产品源和 DLL 未改变，失败记录保留。独立测试工作者在首轮运行结束后违反冻结要求又修改了测试，增加 frozen CPU5 比较和实际 SYCL 位置核对，未修改产品或 busy/字节断言；旧回执不得冒充当前版本。新一轮以完整重新编译的 `probe-cpu-hgs5ybzv` 为准，15 项 CPU 检查通过，不复用旧 source pins。

Luna 的 `runtime-2thread-02` 实测通过：两条持久真实线程（native ID 20824/12856），6 帧共享纹理（3 ON、3 OFF）、3 个计算图捕获、18 次重放、4 次实际交接；15 次未完成状态拒绝、6 次错误 owner/队列拒绝、3 次普通入口错线程拒绝。12 次输出字节比较、24 次输入比较和 18 次非零运动检查通过，比较输出 442,368 字节。最终 3 次开始/退休、3 次 forward import/release、3 次 pack/unpack/wait/signal/consumer registration 全部平衡，forward live 为 0，bridge 健康且空闲，线程退出、资源关闭，exit 0，无 watchdog/隔离。仍为真实 wrapper/adapter 配合合成数学图；未加载完整 NR，不是游戏画质或性能验收。helper 代码模块沿用经过验证的 context 生命周期固定方式，不宣称代码模块全部卸载。

主任务以只读脚本 `tools/review_gpu_owner_transfer_cpu.py` 重新核对原始子报告、真实完成栅栏、每病例计数、队列/线程身份、全部源/二进制及实际 SYCL：[主任务复核](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-owner-transfer-v1-20261002/runtime-2thread-02/MAIN_RUNTIME_REVIEW.json)。人类已退出后，定点备份并更新 adapter、game wrapper、实际 overlay wrapper 和 native DLL 四个文件，九文件安装 hash 全匹配，模型数学、运动语义和缓存未改：[安装回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-owner-transfer-v1-20261002/install-01/installed.json)。2026-10-02 05:05:41 UTC 普通入口启动 PID 16088；实际速度与连续运行仍待同场景实机配对，两个新开关维持默认 OFF。

### run04：纹理复用有效，GPU 接力资格仍被正常换线程清除

同一 720p C512＋K8 六项/V6 场景，Luna 再完成纹理池 OFF/ON/ON/OFF 四臂，各 30 个预热后的独立完成帧。主任务重算：录制 CPU **3.08275→1.98069 ms，减少 1.10207 ms**；录制至回收跨度 **53.50876→52.66021 ms，减少 0.84855 ms**，两次 ON 均低于两次 OFF。与前两轮一致的收益发生在网页计时之前。[原始记录重算](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/main-driver-run04/MAIN_POOL_RAW_REVIEW.json)。匹配同一完成帧的网页处理时间 **49.42092→49.59501 ms，慢 0.17409 ms**，OFF 首尾漂移 −0.59554 ms，不能宣称推理段加速，也不能将重叠口径相加：[网页时间复核](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/main-driver-run04/MAIN_MATCHED_WEB_POOL_REVIEW.json)。

GPU 接力 ON 预热时脚本报 `Bridge capability disappeared during warmup`，NR 本身 healthy、无新锁存错误；两个新开关恢复 OFF，取得健康新帧。不能将这个未持续启用的区间当作 GPU 接力性能结果。主任务从有界 12 秒 trace 的 130 份真实快照定位：采样完成帧 12774→12969（恢复 OFF 后达到 12981），资格新增确认 181 次、身份不匹配清除 180 次，只实际跳过 prepared CPU wait 14 次，观察到 15 个非零 CPU owner ID；借用 GPU 队列/上下文不变，运动读回等待计数 12 未增加。[主任务原始复核](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/capability-trace-run04-01/MAIN_TRACE_REVIEW.json)。这是正常提交 worker 更换引发旧线程绑定资格失效，不是运动数据精度或 XMX 数学问题。

隔离的 `artifacts/gpu-handoff-serial-owner-v2-20261002` 增加可选串行队列资格，旧 Arm 的严格线程合同保留；新入口只供已通过真实退休验证的 owner-transfer helper/串行 adapter 配对使用。相同 helper、GPU queue/context/device 保留资格，每次 CPU worker 交接仍须实际 native/XPU 完成；请求/processor/队列变化或失败清除资格。另修复 OFF 请求晚于本帧 prepared bypass 的竞态：完成该帧真实 GPU producer 等待，下一个 CPU-prepared 帧才关闭 native handoff。新串行资格每次重新确认须真实 CPU-prepared callback，已有 legacy 资格不能为新合同/替换 helper 提供免等待许可。原 prepared fence、consumer 退休、数学、着色器及闪动修复不改。最新 ASI `12fe0f89…` 已完成 14 线程 CPU 构建、9 类原 native CPU 协议和 48 项原 host/ctypes/panel 检查，旧导出保留，仅增加两个可选导出。[CPU 构建回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/cpu-build-03/CPU_BUILD_RECEIPT.json)。build-01/02 保留为后续竞态/资格修正前的历史结果，不用于本次安装。

新增独立 sidecar 已通过：真实候选 C++ 门控在两条持久 CPU 线程上的 **540 次断言**、新增 host/web **23 项测试**、原 C++/48 项 host 回归；不加载 ASI 或创建 GPU。首个 C++ sidecar 在假 callback 身份病例失败：两份假 process/retire 都返回 0，Release 可能将相同机器码合并，病例没有构造出不同地址。原冻结四文件及失败日志保留；主任务复制到 `tests-distinct-callbacks-v1`，仅令替换 callback 的假返回体分别为 1/2，并增加地址不同断言，全部原拒绝门保留，产品不为这项 fixture 修正改动。第二轮 4 个 CTest 全过，输入源前后不变：[完整 sidecar 回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/cpu-sidecar-02/CPU_SIDECAR_RECEIPT.json)。

主任务核对两份 CPU 结果、当前九文件安装、仍相同的已通过真实两线程 GPU 验证的 helper/adapter/wrapper/SYCL，以及数学/着色器/deferred/consumer 源 byte identity：[主任务 CPU 复核](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/cpu-sidecar-02/MAIN_CPU_REVIEW.json)。`tools/install_gpu_serial_handoff_trial_v2.py` 三文件 dry-run 已通过，只拟换 ASI、web、handoff host；`tools/benchmark_gpu_handoff_serial_live_v2.py` 强制 native 与 host 的 serial 资格均实际生效，并沿用真实 prepared bypass 增量、开关往返和 90 秒连续门，不能把回退标成 ON。截至本段用户退出请求已发出，尚未替换、未取得 v2 实机或性能结论。

人类确认退出后，2026-10-02 06:13:10 UTC 已备份并定点替换上述三个文件，九个安装目标写后 SHA 全部匹配：[serial v2 安装回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/install-01/installed.json)。新严格 driver 的安装后 CPU 协议检查通过 83 个 source pins：[检查回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/driver-CPU-installed9-01/CPU_RECEIPT.json)。普通入口已启动 PID 4036，人类随后确认进入同一实机场景；仅 Luna 获得 GPU/API GO，执行新的 `serial-driver-run05` 四臂配对、组合和 90 秒连续检查。两个新开关仍非默认，尚不宣称 serial v2 性能或实机验收通过。ViT worker 只刷新新九文件 front pins、CPU 检查及冻结，未获 GPU GO。

### run05：实际 GPU 接力已持续生效

Luna 严格 driver exit 0；83 个源 pin 前后一致。纹理池四臂、接力四臂和组合臂各 30 个独立完成帧，均预热超过 8 秒和 35 个新完成帧。主任务只读重算，未调用游戏 API 或执行 GPU：

| 独立比较与口径 | OFF 均值 ms | ON 均值 ms | 节省 ms |
| --- | ---: | ---: | ---: |
| 纹理复用：CPU 录制 | 3.32893 | 2.23820 | 1.09073 |
| 纹理复用：录制至回收跨度 | 54.99370 | 53.73219 | 1.26152 |
| 纹理复用：同完成帧网页处理 | 50.50388 | 50.26837 | 0.23551 |
| GPU 接力：CPU 录制 | 3.41941 | 3.43179 | −0.01238 |
| GPU 接力：录制至回收跨度 | 55.22693 | 53.43471 | 1.79222 |
| GPU 接力：同完成帧网页处理 | 50.64552 | 48.81951 | 1.82601 |

纹理复用的两个 ON 臂 CPU 录制均低于两个 OFF，OFF 首尾漂移 −0.01748 ms；GPU 接力两个 ON 臂的跨度和网页处理均低于两个 OFF，OFF 首尾漂移分别 +0.26616/+0.05056 ms。接力网页样本 30/30/29/30，排除一个不同完成帧快照，没有补零。两种开关覆盖的口径相互重叠，不能把表中差值相加；跨度包含游戏与等待，不是 Present/RTSS 帧率。

两个接力 ON 臂全部 30/30 快照同时具有 native serial 资格和 host serial 资格，实际 prepared bypass 增量 67/65，正常换 CPU worker 未再清除资格。组合臂纹理命中与实际 bypass 均增加 69。组合连续窗口 **90.39 秒、1,559 个新完成帧、177 个快照**，全部实际 ON、健康，新增 process/consumer 失败、identity mismatch、资源创建失败和 quarantine 为 0。结束后两个新开关 OFF，健康恢复；主任务重新核对全部原始帧/计数/快照和当前源：[协议复核](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/serial-driver-run05/MAIN_PROTOCOL_REVIEW.json)。[纹理统计](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/serial-driver-run05/MAIN_POOL_RAW_REVIEW.json)、[接力统计](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/serial-driver-run05/MAIN_HANDOFF_RAW_REVIEW.json)、[接力网页复核](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/serial-driver-run05/MAIN_MATCHED_WEB_HANDOFF_REVIEW.json)。

这里只证明前置 producer 等待已由 GPU 接力、结果连续回收；模型末端与 ASI consumer 尾段的既有同步仍在，不声称全部桥工作均脱离 CPU。为直接确认两项一起开启的净收益，新增独立 `benchmark_gpu_bridge_combined_live_v2.py`，在同一未改变的数学链路上做组合 OFF/ON/ON/OFF，各 30 完成帧；它引用本轮冻结的 90 秒协议复核，86 个 pin 的 CPU 检查通过，仅 Luna 获得 GO。结果单列，尚未提升组合为默认，尚无人类运动/灯光和 RTSS 验收。

### 组合四臂：净收益约 2.97 ms，未将局部差值相加

`combined-serial-run01` 已在原游戏进程完成 OFF/ON/ON/OFF，四臂各 30 个唯一完成帧、原预热门不变。CPU 录制 **3.38934→2.26551 ms，省 1.12383 ms**；录制至回收跨度 **54.94664→51.97559 ms，省 2.97105 ms**。两个 ON 均低于两个 OFF，OFF 首尾跨度漂移 +0.28007 ms。[组合原始重算](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/combined-serial-run01/MAIN_COMBINED_RAW_REVIEW.json)。

同一完成帧网页处理四臂均有 30 个匹配样本：**50.40297→48.46948 ms，省 1.93349 ms**，两个 ON 均更快，OFF 首尾漂移 +0.03447 ms。[组合网页重算](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/combined-serial-run01/MAIN_MATCHED_WEB_COMBINED_REVIEW.json)。普通网页 reducer 限制另一开关 OFF，因而正确拒绝组合数据；新增组合专用 CPU reducer，保留原脚本及拒绝记录，不放宽旧门。86 个源/证据 pin 前后及当前完全一致，120 个独立帧实际资格、命中/bypass、失败和恢复门通过：[组合协议复核](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/combined-serial-run01/MAIN_PROTOCOL_REVIEW.json)。这仍不是 Present/FPS，也不能把 CPU 录制与网页差值再相加为整帧成绩。

为人类运动验收恢复双 ON 的首个手动监看脚本误读 `processing.last_nr_frame_id`（实际路径为 `processing.stages.recording.last_nr_frame_id`），把健康且实际 serial ON 的运行误判为超时：该回执 processing.frames 25293→26324，native/host serial 均 true、无 process 失败。失败证据 `ON_FOR_USER_REVIEW.json` 保留，不能标为性能/产品失败。第二次在任何 API 之前因给 `pins()` 传 str 而非 Path 失败；`ON_FOR_USER_REVIEW_02.json` 保留。随后只修监看脚本的字段/类型，复用已通过的 frame/validate/applied/wait_state，产品源、G 安装、先前性能证据不改。恢复确认和人类视觉验收另列。

修正监看脚本后，`ON_FOR_USER_REVIEW_03.json` 已确认健康且实际双 ON：录制完成帧 **35224→35234**，83 个源 pin 前后不变，实际纹理命中与 prepared bypass 都新增 10，native/host serial 资格均 true、pending=false，无新增处理/consumer/identity mismatch/创建失败/quarantine。[恢复回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/combined-serial-run01/ON_FOR_USER_REVIEW_03.json)。人类已收到转镜头和焦点切换的视觉/稳定性检查请求，尚未收到结果。GPT‑6.1 Sol 仅准备成功桥源的定点 canonical 合入方案和 CPU dry-run；冻结测试源、当前 G 安装及默认值暂不改。

人类随后确认 **“画面和稳定性都正常”**，并报告当前 RTSS **“55–60 ms”**；换算约 **16.7–18.2 基础帧/秒**，按本轮约定帧生成关闭。这是同场景当前完整游戏链的用户读数，没有自动 Present trace，也没有与 API 同帧配对，不能把之前至当前的全部 RTSS 改善都归因于本次桥组合。[人类验收记录](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-live-pair-v1-20261002/combined-serial-run01/HUMAN_ACCEPTANCE.json)。纹理复用＋GPU 接力组合保留；当前九文件安装和隔离源已实际验收，canonical 合入继续，开机默认尚未修改。下一数学候选是准备就绪的 current-provider 完整 ViT score/value 比较，需游戏退出、Luna 独占后才给 GPU GO。

## 已验收桥优化合入主源码

人类退出后，完整模型首轮预编译已停止并保留测试来源误判的失败证据。主任务核对合入目标与当前 ViT 源/前端冻结及原始 155-pin exporter 无交叉，随后按已审阅的 `gpu-hooks-only` 计划备份并合入 **25 项 E 源码**，涵盖 ASI、纹理池、共享队列接力、owner transfer、Python 接口及 CMake 依赖。pre-XeSS host 只增加四处已测 GPU 调用，既有模型/历史/控制及其他诊断字节保留。G 安装和启动默认没有修改。

主源码重新构建 ASI 及九个 CPU 目标成功，**9/9 C++ 检查、48 项 Python 控制检查通过**，导出 ABI 与已测候选一致。canonical 的 Python 测试最初假定控制/host 在本工程 game 下，主任务只修正其寻找既有共享 RE8 源目录的路径及页面来源，保留全部断言；该测试文件的新 SHA 单独记录，不宣称与原 stage 测试字节相同。合入后 ViT `verify-freeze` 再次通过，仍为原 **448/551/9** 文件和原冻结 SHA。备份、最终源码 pin、编译与检查记录见 [主任务合入回执](D:/Codex-NR-Experiments/cyberpunk-opt/gpu-handoff-serial-owner-v2-20261002/source-integration-v2/MAIN_CANONICAL_SOURCE_RECEIPT.json)。新构建二进制只在 D 用于 CPU 验证，没有安装或重新声称 GPU/实机验收。

## 外层重复同步：独立候选已完成 CPU 检查

`terminal-sync-qualified-v1-20261002` 只尝试省去 FullsizeGameModes 在热图重放之后的外层 `torch.xpu.synchronize()`。GraphFront 原有完成等待、历史提交、输入 finite/range 检查、ASI consumer 尾段以及线程 owner 交接证明保留。资格绑定实际 callback、同一队列/context、serial lock、单帧 fence/epoch；冷帧、reset、新建图、未授权或不匹配时保留原等待或在回送前拒绝。结果、历史、模型、图及 numerical child 持有至真实消费完成，注册 retire 尚未完成不算安全复用。

Sol 的隔离代码及 17 项 source freeze 已完成；主任务逐段审读并独立运行受限 CPU runner，**19 个测试 / 44 个子案例通过**，源码及保护的 E/D 文件前后相同。[主任务 CPU 回执](D:/Codex-NR-Experiments/cyberpunk-opt/terminal-sync-qualified-v1-20261002/main-review-01/CPU_RECEIPT.json)。这些使用显式 CPU 张量/事件替身，不能当作 GPU 字节或速度证明。候选默认关闭，未部署；真实 GPU 验收 runner 正在独立实现，等完整注意力测试结束后由 Luna 串行使用 B580。省去此等待可能只把等待移到 export、consumer 尾段或下次 idle，额外资格查询也有成本；现阶段不计净收益，更不与已测 2.2655 ms CPU 录制相加。

后续在构建真实 GPU runner 时发现 v1 资格条件误写为 `experiment_720=baseline` 加 six opts；实际 host 工厂与已验收 live 设置要求 `c512_k8` 加 six opts，故合法现役路由不会取得 v1 资格。旧 CPU 案例预先创建 modes，没有覆盖真实 `_modes=None → _new_modes → process` 首次构建；原 19/44 仅证明其中的契约，不能说明 v1 能在现役跳过等待。旧冻结及结果不改，v1 不部署、不记收益。

Sol 新建 `terminal-sync-qualified-v2-20261002`，唯一产品差异是将资格路由改为真实 `c512_k8`，原 19/44 保留，增加首次构建、冷帧原等待、热帧资格和真实消费退休路径回归。worker 报告共 23 项 CPU 测试通过；主任务与真实 GPU 检查仍需分别完成。新 GPU runner 只引用 v2，不放宽 callback/queue/owner/fence/retirement 门；G 安装保持已验收版本。[v2 worker CPU 回执](D:/Codex-NR-Experiments/cyberpunk-opt/terminal-sync-qualified-v2-20261002/cpu-ready-01/CPU_RECEIPT.json)。

主任务随后独立执行 v2 受限 CPU runner，exit0，**23 项测试 / 44 个原子案例通过**（19 项原契约＋4 项实际工厂路由回归）。旧冻结、v2 source 与 protected E/D pins 前后不变，没有 framework/GPU/DLL/G/API 操作。GPU runner 继续准备，仍无 GPU 数值或净收益结论。[主任务 v2 CPU 回执](D:/Codex-NR-Experiments/cyberpunk-opt/terminal-sync-qualified-v2-20261002/main-review-01/CPU_RECEIPT.json)。
