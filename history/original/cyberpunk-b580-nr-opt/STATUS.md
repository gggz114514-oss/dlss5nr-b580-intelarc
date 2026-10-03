# Cyberpunk 2077 B580 NR → XeSS：当前状态（更新至2026-10-03）

## 18组完整候选已合并，开始GPU资格与速度验证（2026-10-03）

r9 Front/Decoder/Post完整组合实际通过：离线完整NR平均46.715→44.403ms，快2.312ms（约4.95%）；三个13帧计时周期分别快1.989/2.034/2.915ms，P95为47.794→45.087ms。输出有限，MAE0.00006588、最大绝对差0.00415039，13帧捕获/历史/只读缓存资格通过，零缓存写入。此结果尚需独立配对复测及用户视觉/赛博朋克实机验收，不当作游戏帧时间。原始结果为D:实验目录attempt-12-r9-fdp-runner4。

用户授权D盘不足时使用E盘。独立phase2-e测试器已完成46项CPU检查、13项比较检查和实际E路径preflight；2331个缓存文件约131.89MiB已校验复制到E，二进制/编译元数据保持原字节，仅编译组child_paths重定位。旧D缓存和结果保留；下一轮输出/缓存改为E:/Codex-NR-Experiments/cyberpunk-opt/b580-full-implementation-v1-20261003/luna/phase2，原始fixture和历史证据仍从D只读。Luna获授权先复测Front收益，再独立测C512及Swin。

r9补齐Decoder及ViT剩余旧provider冷计数检查：r8 Front实际通过公共post后又被Decoder旧门拒绝，不能报告其完整帧速度。两个实际方法补丁通过Main重跑14项/119子项回归及115项集成检查，258文件/103档新快照已冻结，Luna获授权使用r9串行复测Front、未融合FFN入口C512、Swin。候选数学内核、完整KV、控制/运动/历史语义不变；现役G:安装仍未改，本批尚无新实机整帧收益。

r8已冻结258文件/103档，Main的115项CPU检查及Swin实际方法13项选择回归通过。r7两组候选没有测得性能：Front在实际捕获后被公共post检查器残留的旧K8计数假设拒绝；C512融合FFN入口实际编译出现7168字节spill，发生在整帧测量前。r8已统一公共post的真实替代入口证明，并给Swin增加有限BM8/warps8冷选择；Luna串行复测Front、独立的未融合FFN入口C512档和Swin。融合FFN入口由6.1 Sol max继续修复，完整组合仍待其资格通过；未改G:安装，未宣称新增游戏提速。

r7续测：r6基线完整NR平均45.908ms，历史组合45.930ms，单轮差为−0.023ms，没有可确认净收益。Swin第二次实编仍有寄存器溢出：BM32/stages1为832字节，BM16/stages1为320字节，运行前停止、图/session退役通过；继续改冷分块。Front组合确实替换了K8 post，旧历史检查仍要求其旧计数，r7改为校验实际已编译替代入口与剩余旧入口；同时修正“图已销毁但外层session尚未关闭”时的合法恢复顺序。相关15项CPU回归、112项主集成检查通过。C512冷选现记录真实spill类型/值，并仅在合法正整数溢出时尝试工具链实际支持的GRF256配置，GPU资格待测。r7冻结258文件/103档，正在串行复测Front与C512，不改G:游戏安装。

Halley的3.161→0.217秒是CPU逐像素检测改为NumPy批量检测：真实两帧完整报告一致，约14.6倍；13帧检测约19.805→0.918秒。这不是模型推理提速。比较优化已合入官方测试runner，原GPU worker、数值/缓存/退役边界保留，原runner3字节归档；原46项检查和新增13项比较检查通过。

恢复测试后，历史 admission 的旧接口调用、正常退役标记与 broker 释放冲突已修正。完整模块基准已通过实际只读缓存/数值/捕获检查：整段 NR 平均约45.74ms，网络图主体P50约39.38ms，桥与游戏不在此计时范围。Swin 首次实编发现C64双分支核BM32/stages2有1280字节spill，r6改为冷选分块/阶段并绑定实际执行核；Front旧来源检查拒绝了新融合尾段，r6只准入实际Main/Front归属的绑定。两处修正分别通过9项与8项CPU回归，主集成与条件矩阵89项通过；修正后的实际GPU结果待Luna。C512有效查询注意力也在编译资格处失败，旧错误文本混淆未知与非零spill，正在细分真实值并改冷启动寄存器配置；未取得此候选整帧时间，不归为已测减速。

r6隔离源码冻结258文件/103档，其中20档为同风格/控制的条件配对。Luna先串行测试独立历史、Front和Swin；C512与完整组合待该处修复后继续。18组代码已合入隔离树，不等于18组已验证提速。本轮目前只有首批数值组合约0.61ms的单轮网络图差值，尚无可确认的新增游戏整帧收益。自然/电影风格、低强度及真实HDR桥仍须分别执行，不用标准风格的零调用代替验证。

唯一开发基线仍为固定720p **C512＋K8**、六项已验收计算选项、融合真实运动历史与图重放、纹理复用/GPU接力；用户此前实机55–60ms。五组6.1 Sol max的最终候选源码已合入E:隔离树：Swin/MLP、C512/ViT、front/Decoder/post、history/host/graph、native桥。主任务完成共享模型控制写入、实际新旧捕获角色、线程交接、图先退役再释放、typed控制/TLS读入、两帧native consumer协议及来源哈希组合。Main的79项CPU检查通过；worker CPU检查和真实GPU资格分开保存。

第一轮11个数值档全部完成预编译及新进程只读缓存复测，零缓存写入，实际2次capture＋11次replay，输出/私有历史有限。基准网络图P50 **39.4145ms**、P95 **39.7894ms**；数值清理组合P50 **38.8010ms**、P95 **39.4563ms**，13帧输出最大绝对差0.00390625、平均绝对差0.00006417。这是单轮离线网络图结果，排除历史/输入准备、桥和游戏Present；约0.61ms差尚待重复验证，不能当新增游戏收益。其全host冷13帧统计包含捕获/提交等不稳定成本，不用来推断稳态游戏FPS。

完整模块/布局快照已冻结258个文件（250个Python文件），83个候选/组合，先测30个优先项覆盖计算、producer/consumer布局和历史/资产，随后测其余替代实现。新runner另计完整`FullsizeGameModes.process`的XPU事件边界与CPU提交，保持同13帧真实输入、运动、reset/history/seed合同；桥真实HDR及consumer栅栏单独验证。GPU唯一执行者仍为Luna（gpt-6-luna xhigh），串行预编译→全新进程只读命中→误差/整段耗时。native新protocol资格仍0，CPU编译不冒充GPU执行。

[WORK.json](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/WORK.json)、[PHASE2.json](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/b580-full-implementation-v1-20261003/PHASE2.json)保存源码与测试矩阵。游戏安装暂未改；先720p，随后兼容输入分辨率缩放；NR256和精确4060分支未改。不设1ms门槛，只采纳可重复净收益，画质变化单列。D:重复中间输出在完成哈希/误差报告后清理，保留源码、脚本、报告与必要视觉结果。

## 现役720p全覆盖补审完成（2026-10-03）

五名6.1 Sol max完成12维逐节点补审，唯一对象为当前固定720p的C512＋K8优化集成后端＋已验收纹理复用/GPU接力桥。主任务完成来源、动态选择条件、共享归属与遗漏核对：2534个覆盖节点，137个保存编译变体全部关联，67条原始问题及改造记录合并为18组工作包。记录包含热路径、条件/稳定性和冷资产，不等于67处实测瓶颈。完整结论与实施顺序见 [FULL_BACKEND_AUDIT_20261003.md](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/docs/FULL_BACKEND_AUDIT_20261003.md)。

本轮完成的是审计及计划，没有改生产数学、部署或新增GPU收益。现役已验收完整基础帧时间仍为55–60ms。先处理ViT补偿FMA、Decoder四级输出舍入和分段half等数值遗留，再改完整计算段/布局；桥与CPU独立验证，GPU由Luna串行。有可重复净收益即保留，质量变化单列；先优化720p，再兼容输入分辨率缩放。外部库/驱动内部、物理流量、同帧Present关键路径及条件触发仍是测量项。

## v1现役后端重审记录（2026-10-03，本轮五组补审前）

唯一开发基线为720p **C512＋K8**，最近用户验收纹理复用＋GPU接力的实机基础帧时间为55–60ms。四名GPT‑6.1 Sol max完成Swin/MLP、C512/ViT、front/Decoder/post、history/host/graph的只读审计；主任务核验实际安装来源、有效选项、137个执行编译key及最终引用。共38项发现，已去重编排后续工作；17项A类为确定执行且值得优先验证，并非17项已测加速。

完整范围、现役残留、已有但关闭的候选、新写模块及最小实验见 [CURRENT_BACKEND_REAUDIT_20261003.md](E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/docs/CURRENT_BACKEND_REAUDIT_20261003.md)。此轮没有改游戏安装或数学源码。当前矩阵已有原生FP16/INT8 DPAS；Decoder输出FP8舍入、补偿FMA、有序half边界与重复加载/窄tile等分别记录。下一阶段先处理数值对齐，再按完整计算模块及消费者接口改造，GPU验证仍由Luna串行完成。

以下为早期接入记录；其中启动参数、现场计时和进行中状态仅属于注明的历史轮次。

## 向其他 DLSS 游戏迁移的第一步（同日，未部署）

新增独立 `GenericDLSSProbe.asi`：只读取 OptiScaler NGX 入口中少量帧的输入合同和调用方，原样转交超分调用；不启动 NR 或改写资源。它与已验收的 Cyberpunk 实机桥分离，不能在同一进程混装。14 线程 Release 编译和模拟 OptiScaler 的 12 次连续调用测试通过；第二款游戏的实机输入、命令顺序和画面尚未验证。可用候选位于 `artifacts/generic-dlss-probe/GenericDLSSProbe.asi`，D: 构建产物与其哈希一致。实施边界与新游戏接入条件见 [GENERIC_DLSS_PORT.md](GENERIC_DLSS_PORT.md)。

## 原生 XeSS 帧生成选项与显卡伪装（14:25 实机对照）

在 XeSS 性能档、游戏帧生成关闭时，`OptiScaler.ini` 的 `Spoofing.Dxgi=true`、`StreamlineSpoofing=true` 组合下，游戏图形菜单不显示原生 XeSS 帧生成选项。备份 INI 后，仅将全局 `Spoofing.Dxgi` 改为 `false`，保留 `StreamlineSpoofing=true`，重启游戏；用户确认原生 XeSS 帧生成选项立即恢复且可选。这支持全局 DXGI 显卡伪装是选项消失的原因，不需为 XeSS 路线伪装整台显卡。此次为隔离验证还关闭了 OptiScaler XeFG；其开关影响运行中的帧生成，不作为菜单选项消失的独立证据。原 INI 备份在 `D:\Codex-NR-Experiments\cyberpunk-opt\deployed-backups\before-native-xess-fg-check-20260926`。

此前在 `Dxgi=true` 时，另一路 `FrameGen.FGInput=upscaler`、`FGOutput=xefg` 成功创建并激活 XeFG 交换链，NR 连续处理；用户确认实机场景画面与 HUD 正常。该轮未打开 XeFG Debug View，不据此宣称已逐帧确认插帧标记。

随后游戏设置保存为 `FrameGeneration=XESS`、`XESS_FrameGeneration=true`；OptiScaler 帧生成保持关闭，移走 NR 执行标记后重启。用户确认在实机场景原生 XeSS 超分＋帧生成的画面、HUD、流畅度正常。无需重启，恢复 NR 标记；XeSS 尾部安全门仍为 `safe=1`，NR 540p 连续执行后通过网页切到 360p／融合历史／图重放。快照超过 793 帧，最近一帧约 34 ms，`health.failed=false`；用户确认转动镜头时画面、灯光、HUD、卡顿均正常。NR 执行标记已恢复，当前游戏配置为关闭全局 DXGI 伪装、保留 Streamline 伪装、关闭 OptiScaler 帧生成、开启游戏原生 XeSS 帧生成。尚未用插帧标记或独立显示帧计数直接测定生成帧数量；这里的验收是功能设置、连续 NR 和用户实机画面。

## 原生 XeSS 性能档（13:44 实机诊断）

游戏设置 `ResolutionScaling=XeSS`、`XESS=Performance`，帧生成为关。游戏调用了原生 `xessD3D12Execute`，独立桥观察到嵌套 NGX 超分调用，证据为 `xess_d3d12_execute`；输入颜色 R11G11B10、运动 RG16F、深度及曝光纹理，尺寸 1280×720，XeSS 输出 2560×1440。运动缩放为 (1280,720)，历史重置和抖动也已记录。这比从可选参数名猜测路由更强。

首轮 `build-deferred` 诊断构建没有启用原生 XeSS 的 NR，日志中的 `bypass=unsupported_input` 符合预期。八帧尾部探针显示每帧 NGX 调用后、命令列表关闭前还有 **一次资源复制**（关闭 OptiScaler 的 RCAS 和帧生成后依然存在，后续 dispatch 和 draw 均为零）；颜色尾部状态 2240→2112、输出 8→192，与 DLSS 路线不同。因此不能直接拿已验收的 DLSS 拆批次规则启用 NR。签名的 OptiScaler DLL 未改，原版 ASI 和 INI 均已在 D: 留存备份。

第二轮 8 帧诊断确认：唯一后续动作是从 **原始颜色**复制到另一张 1280×720 的 R11G11B10 纹理，不触碰 XeSS 输出，也没有 dispatch/draw。已在独立 `NRB_MULTI_ROUTE_LIVE` 候选中添加严格匹配该复制、前后状态及八帧一致性的 XeSS 放行条件；HDR 代理准备与合成也改为按 XeSS 的最终读状态切换，原 DLSS 路线默认状态不变。候选离线 10/10 检查通过，随后部署实测。

**后续实机更新：原生 XeSS→NR→XeSS SR 已运行并通过画面审核。** 候选在八帧 `safe=1` 后，原生 `xessD3D12Execute` 路由记录 `source=1`，开始在同帧拆批次执行 NR。540p／零运动档连续超过 1,000 帧、`health.failed=false`；进入实机场景约 89 ms/帧，用户确认画面、灯光和运动正常。网页切到 360p／融合历史／图重放后，实际处理超过 2,300 帧，稳态快照约 35–37 ms/帧、无失败；运动探针在实机场景读到 3600/3600 非零稀疏点，缩放 `(1280,720)`。用户再次确认画面正常。此处速度包含 NR 与当前 Cyberpunk 拆批次队列尾段，不能与 RE8 的 NR 独占时间直接比较。后续游戏原生 XeSS 帧生成与 NR 联动见文首更新；FSR 路线仍未实机放行。当前放行条件固定为本次 1280×720→2560×1440 的纹理与尾部形状，不等于已验证其他游戏或分辨率。

## FSR 3 性能档（14:35 只读基线）

**路线暂缓（用户 2026-09-26 决定）：** 不继续为 FSR 写独立替换桥；先用已验收的 DLSS 和原生 XeSS 输入路线。当前游戏是 FSR3 Performance、游戏帧生成关闭，NR 执行标记未部署，因此画面是游戏原生 FSR，不是 NR→XeSS。诊断 ASI 只读、限量记录，不改 FSR 输入/输出；游戏仍在运行，暂不热替换 ASI。下次正常退出后可从 `D:\Codex-NR-Experiments\cyberpunk-opt\deployed-backups\before-fsr-tail-probe-20260926\CyberpunkNRBridge.asi` 恢复上一实机版本。OptiScaler `DxgiFactoryWrapping` 已还原为 `auto`。

游戏设置已切到 FSR3 Performance、帧生成关闭，用户确认实机场景画面正常。NR 执行标记已移到 D: 备份，故此轮 **没有 NR**。ASI 的 `ffxFsr3UpscalerContextDispatch` 钩子安装成功并记录真实调用；但没有看到 OptiScaler 在该调用内执行 NGX evaluate。OptiScaler 日志提示 `ffxFsr3ContextCreate_Dx12 D3D12 device not found!`，其本地源码在设备指针为空时把 dispatch 原样交回游戏，因此现有 XeSS/NGX 桥不能直接接 FSR。FSR 路由仍 fail-closed，不能仅因 FSR hook 命中就宣称已经过 NR。

已在独立 ASI 中加入限量、只读的 FFX dispatch 资源描述记录，不改 FSR 参数或签名 OptiScaler DLL。新 ASI 已部署，游戏连续记录到第 600 次 dispatch：`render=1280×720`，颜色 FFX 格式 13（R11G11B10_FLOAT）、运动格式 14（R16G16_FLOAT）、深度格式 24（R32_FLOAT）均为 1280×720，输出目标 2560×1440；运动倍率 `(1280,720)`，资源状态为颜色/运动/深度 FFX 12（像素＋计算读取）、输出 FFX 2（UAV）。FFX 描述符镜像布局已与 OptiScaler 公开头文件对照：资源 176 字节、输出偏移 1592、渲染尺寸偏移 1784。这里只读到了 FFX 描述信息，尚未证明 D3D12 尾部状态、安全拆批次、NR 结果或 XeSS 输出。构建及 10/10 离线检查通过。两个 hook smoke 共用一个日志文件，已用 CTest 资源锁避免并行删除/写入互相干扰。

## FSR／原生 XeSS 入口扩展（实机确认中）

在独立 ASI 中为原生 `xessD3D12Execute` 和 FSR3 `ffxFsr3UpscalerContextDispatch` 加入外层 API 范围标记；只有真实 API 嵌套的 NGX 调用才获得对应路由证据。OptiScaler 0.9.4 的两处导出钩子在 13:07 启动的《赛博朋克》进程中均安装成功，游戏此时仍是 DLSS 档，故尚无 FSR／XeSS 实机帧。先前只凭可选参数名判断路由仍保持旁路。签名的 OptiScaler 主 DLL 未改。

已将命令尾部八帧证明改为随路由切换重新采集并按 DLSS／FSR／XeSS 分别记录，避免从 DLSS 的尾部状态推断其他两路。此改动与新入口日志已在 D: 编译；运行中的游戏仍加载先前 ASI，重启后才生效。新两路的拆批次 NR 代码由 `NRB_MULTI_ROUTE_LIVE` 独立控制；`build-deferred` 默认为 OFF，`build-multiroute` 已编译 ON 候选但**未部署**，待实机颜色格式、运动比例、尾部资源状态确认后再启用。当前已部署版本备份在 `D:\Codex-NR-Experiments\cyberpunk-opt\deployed-backups\before-fsr-xess-route-20260926-1307`。最近一轮离线完整测试 10/10 通过；新增按路由重采集测试通过。此处不宣称 FSR／XeSS 已运行 NR 或 XeFG。

## XeFG 实机接入（进行中）

已备份游戏的 `OptiScaler.ini` 至 `D:\Codex-NR-Experiments\cyberpunk-opt\deployed-backups\before-xefg-20260926-123649\OptiScaler.ini`，随后仅将 `[FrameGen]` 设为 `Enabled=true`、`FGInput=dlssg`、`FGOutput=xefg`。游戏内开启 DLSS 帧生成并重启；超分仍为 DLSS 性能输入经 OptiScaler 转 XeSS，NR 仍在 XeSS 前。游戏为 2560×1440 无 HDR 的无边框模式，XeFG 日志显示创建交换链、支持一次插帧并成功 `SetEnabled`。OptiScaler 菜单显示 `DLSSG via Streamline → XeFG`，`Active` 已勾选。控制页切到 360p／融合历史／图重放后，NR 连续超过 7,000 帧且 `health.failed=false`。

第一次实机日志反复出现 `Hudless state changed true -> false -> true`，XeFG 因此主动停用、等待十帧、再启用；菜单截图同时显示 `Reflex not hooked`。用户在高级帧生成设置中尝试 `Disable HUDless` 后主观报告顿挫消失、画面可接受，但第一次进程日志后续仍记录状态切换。已再次备份 INI 至 `D:\Codex-NR-Experiments\cyberpunk-opt\deployed-backups\before-hudless-fix-20260926-124622\OptiScaler.ini`，并将 `[FrameGen] DisableHudless=true` 写入游戏配置。

**第二次干净启动已完成这条接入链的实机验收。** OptiScaler 确认读取到 `FrameGen.DisableHudless: true`，支持一次插帧（2× 档）。NR 设为 360p／融合历史／图重放，连续处理超过 **4,100 帧**，`health.failed=false`。XeFG 自载入菜单后的短暂启停以外没有再记录 `Hudless state changed`、标签错误或缺资源告警。用户在实机场景打开 XeFG `Debug View`，确认看到生成帧标记；关闭调试后确认 HUD、运动与整体画面可接受。这证明 NR→XeSS SR→XeFG 在《赛博朋克》DLSS 性能档当前配置下实际工作。OptiScaler 菜单仍显示 `Reflex not hooked`，本轮未测独立的显示帧率、延迟或多游戏通用性；不能把 NR 网页耗时或菜单 FPS 直接当成 XeFG 帧率。由于禁用了 HUDless，其他 UI 场景仍可能需要画面复核。

## 真实运动矢量候选（已读到实机非零数据）

已在同帧 XeSS 前桥读取 Cyberpunk DLSS 调用给出的 RG16F 运动纹理、`MV.Scale.X/Y` 和历史重置标记。候选着色器在源尺寸 1280×720 上把运动值乘每轴缩放，形成 RE8 NR 后端要求的像素位移 RG16F；零运动与每帧重置仍可通过控制页对照。参考/融合历史仅在尺寸、格式、设备、有限缩放与无运动抖动条件满足时接入，否则退回游戏原 XeSS 路线。原始游戏运动纹理不修改，XeSS 仍使用自己的原运动输入。

2026-09-26 实机重启后，控制页 540p 零运动档曾连续处理 240 帧、约 83 ms/帧，`health.failed=false`。第一次切到“真实运动”后持续处理超过 1200 帧，控制页未报错；用户在转动镜头时肉眼确认画面、灯光和稳定性正常。切档后立即采集的四帧稀疏样本全部为零，因此加了有限的间隔采样。新诊断版部署后，真实运动档的第 60、120、180、240、300、360、420、480 次采样各读到 **3600/3600 个非零点**，缩放为 `(1280,720)`；截至观察时控制页连续处理超过 1300 帧、`health.failed=false`。这证明游戏运动确实到达转换后的 NR 输入，尚不能单靠非零数证明运动方向和时序完全正确；本轮移动画面仍需用户确认。采样到第 480 次为止，之后不再复制纹理。当前完整离线测试的两个 mock hook smoke 在开启实机诊断构建参数时以 `0xc0000409` 退出，不能宣称全套测试通过。

用户随后通过网页切到 360p／融合历史／图重放，控制页快照约 3300 帧、约 36 ms/帧、`health.failed=false`。这是不同输入尺寸和场景负载下的现场快照，不能直接当成 540p 真实运动档的配对提速结果。当前运行进程的可访问控制页为 `http://127.0.0.1:8765/`；重启后的测试构建仍默认 540p／零运动，网页可切到参考或融合历史。

同日分段计时版澄清了与 RE8 计时口径的差别。Cyberpunk 网页从准备列表提交前计到本批次 XeSS 列表的完成 fence，因此包含 XeSS；RE8 网页的 XeSS 前统计在 NR 输出就绪后结束。Cyberpunk 360p／融合历史／图重放第 1320、1440、1560、1680 帧的均值分别为前置准备 **3.84 ms**、Python/NR 运行时调用 **29.06 ms**、返回后至队列完成 **2.59 ms**、总计 **35.49 ms**。同一运行前半段第 120–1080 帧总计约 44–46 ms、运行时调用约 41 ms，说明场景/负载时序不能忽略；仅凭此日志不能把下降归于某个单一原因。RE8 既有 360p 游戏内约 28–30 ms 是另一计时边界与场景，不能直接算作 Cyberpunk 本体慢 6 ms。上述日志只每 120 帧抽样一次，未生成逐帧分布。

此次修复过一处导致首次候选卡住的锁递归：Evaluate 已持有 `state_mutex`，再次调用 `NRB_GetControls()` 会死锁；现在直接传递已持锁读取的控制值。当前运行中的修正版已通过连续帧验证。旧版备份位于 D: 的 `deployed-backups/before-real-motion-20260926-120226`，卡住的候选另有 `deadlock-candidate-20260926-1211.asi` 备份。

## 2026-09-26 实际 NR 接入（以下优先于早期探针快照）

《赛博朋克 2077》的 DLSS“性能”档向 OptiScaler 提供 1280×720 游戏输入；拆批次桥在同帧 XeSS 前运行 B580 快速 NR。540p 档先从 720p 画面构造 SDR 代理，模型输出的差值再合入保留高光的 R11G11B10 游戏颜色，交给 XeSS 放大到 2560×1440。本节记录初期零运动接入，真实运动候选的后续结果以上节为准。

第一版完成 1 帧后，RE8 运行时因不同 CPU 提交线程回收输出而拒绝下一帧。修正版让输出在发起推理的提交线程等待消费完成并退休，Cyberpunk 适配层在下一提交线程接续同一 XPU 流。Luna 只读日志确认修正版连续 **8 帧**完成 NR：`nr_frames=1…8`、`nr_failed=0`，提交失败、列表代际不匹配、失效计数均为 0。第 8 帧已提交，快照没有再次观察其退休。用户尚未完成这轮连续画面的肉眼审核。

这 8 帧首帧约 8.3 秒、第二帧约 731 ms，之后 6 帧约 **80.2–80.8 ms/帧**。这是短样本的整段处理时间，不代表长期稳态或游戏 FPS。现阶段仍不满足实时目标。

已在 E: 源码接入 RE8 的 `ControlPanel`，沿用原有三风格按钮、360/480/540p 滑块、模型参数、历史/图重放与计时；把完整设置映射到 Cyberpunk ASI 控制接口。当前桥不传真实运动，因此网页禁用相关历史选项。实际控制页尚待退出游戏后部署重启。离线验证包括真实 RE8 面板对 mock 原生接口的 HTTP 设置、无效运动模式拒绝，以及现有适配器/着色器/钩子检查。

源码在本项目，构建数据在 `D:\Codex-NR-Experiments\cyberpunk-opt`。本机游戏插件位于 `G:\epic\Cyberpunk2077\bin\x64\plugins`，临时 junction 指向已安装的 RE8 运行时；这还不是便携发行包。旧 ASI 备份在 D: 的 `deployed-backups`。OptiScaler 签名 DLL 未改动。FSR、原生 XeSS 输入桥与真实运动矢量尚未完成，不能由 DLSS 的 8 帧结果外推。

## 早期探针阶段记录（其“NR=0”等结论已过时）

**最新实机进展：** DLSS 720p 输入的同列表恒等回送已经录制、提交并
在 GPU 完成后回收，用户确认移动画面正常；详情见
[LIVE_DLSS_IDENTITY_RELAY.md](LIVE_DLSS_IDENTITY_RELAY.md)。尾部与批次探针
确认 XeSS 列表位于 9 个列表中的第 4 个；拆批次恒等版已完成离线构建，
尚未实机验收。真正 NR 帧仍为 0，
上述仅证明可替换 XeSS 前颜色输入，尚未解决外部 SYCL 推理与同帧提交
顺序。下文是此前探针阶段的记录，其“未部署”和 HDR 像素未知等描述
已被这次实机测试及单帧像素回读更新，不再代表最新状态。

**同步边界探针补记：** 已在 E: 源码加入只读的 D3D12 列表 Reset 代际、传统资源屏障、提交调用关联；D: 上 Debug/Release 各 6/6 离线测试通过，尚未部署 G: 或在游戏实测。`source_submission_proven=false`，实际 NR 仍为 0。完整边界见 [SYNC_BOUNDARY_PROBE.md](SYNC_BOUNDARY_PROBE.md)。下文原有“各 5/5”是该探针加入前的测试快照。

**结论：签名 OptiScaler 上的独立 ASI 已在真实游戏中截获 XeSS 和 DLSS 的超分前输入，但尚未执行任何 NR 帧。** 三条入口已拆为独立模块，共用 RE8 NR 后端合同和未来的 XeSS SR/FG 输出；FSR 目前只有模拟入口检查。游戏原生 XeSS SR/FG 的独立观察工作不在此目录，本项目不重复其实现。详见 [THREE_ROUTE_BRIDGES.md](THREE_ROUTE_BRIDGES.md)。

XeSS Performance 的实机输入为 1280×720、输出为 2560×1440；DLSS 也已在游戏命中，初期有一次 1505×847 过渡尺寸，随后重建为 1280×720→2560×1440。两路观察到 R11G11B10 浮点颜色、RG16F 运动；XeSS 的 Opti NGX scale 为 (1280,720)，DLSS 还记录到 auto-exposure=1。用户游戏 HDR 设置为关闭，格式与 HDR flag **不能证明实际像素 >1**。RE8 桥不支持此颜色格式，也不能把原运动直接当 RE8 的单位尺度。队列候选、当前帧生产者 fence 与消费者退休仍需证明。详见 [LIVE_XESS_DIAGNOSTIC.md](LIVE_XESS_DIAGNOSTIC.md) 和 [THREE_ROUTE_BRIDGES.md](THREE_ROUTE_BRIDGES.md)。

代码只在本项目 `src/`、`include/`、`game/`、`tools/`、`tests/`。构建和临时数据在 `D:\Codex-NR-Experiments\cyberpunk-opt`。**本轮三路拆分、路由证据与恒等安全门只在 E/D，未替换正在运行的游戏插件**；G 盘部署由用户单独负责。此前已部署的 ASI 副本仍是旧版诊断构建。官方 OptiScaler 主 DLL 保持签名；早前游戏独立 INI 从 Trace 调为 Info 的备份位于 D。未修改 RE8 原工程或模型，未发布。

验证：Debug、Release 各 5/5 离线测试通过，包含三路模拟、可选键缺失判 UNKNOWN、RE8 能力规划、DLSS 恒等复制安全门与 R11G11B10 CPU 解码；早前 XeSS 两轮实机诊断正常，DLSS 实机路由由用户独立确认。网页在旁路时可显示真实拦截计数、NR 完成数及三种风格按钮。`processed_frames=0`，无画面经过 NR，故尚无画质、速度、FG 联动验收。Luna 对先前快照的独立检查见 `D:\Codex-NR-Experiments\cyberpunk-opt\luna-qa\ROUND2_QA_REPORT.md`；其中加载锁、网页启动、风格按钮、输出/曝光诊断缺口已在当前源码修正，生产者顺序和安全退休仍未解决。

下一步先由用户的实机观察器在可信资源状态/完成 fence 下获取少量 R11G11B10 原始回读，交 `tools/analyze_r11g11b10.py` 判断像素范围；内部格式/flag 不能替代此证据。DLSS 同列表恒等复制虽已具备录制函数，但尚缺当前 Reset 代际、最后写入、源状态与消费者退休证明，未接入游戏。外部 RE8/SYCL NR 还需要更严格的**已提交当前帧**生产者 fence，不能把队列 1 的历史提交当证明。FSR/XeSS 的可选键只做诊断，外层 API TLS route tag 留待各路验收。早期离线构建说明保存在 [OFFLINE_PHASE_ARCHIVE.md](OFFLINE_PHASE_ARCHIVE.md)，其“未运行游戏”等描述仅适用于当时。

## 2026-09-26 XeSS/FSR3 入口标记与路由诊断离线回归

按用户要求仅运行离线构建和测试，未启动游戏，也未调用 GPU 推理或游戏渲染。使用 D: 的 `build-deferred`、Release 配置完整构建成功；CTest **10/10 通过**（nr_offline、sync/tail probe、HDR shader 源检查、hook/Streamline mock、面板/adapter/web/pixel probe）。`nr_hook_smoke` 验证 FSR/XeSS 可选 NGX 标记只增加诊断入口计数，仍以 `unknown_route` 旁路；日志有 route/evidence/queue/bypass 字段且诊断采样上限生效。

覆盖边界：离线 mock 没有调用真实 `xessD3D12Execute` 或 `ffxFsr3UpscalerContextDispatch`，因此未验证目标 DLL/导出在游戏中的可用性、ABI/签名、Detours 安装成功以及 TLS tag 嵌套时序；smoke 也只检查 evidence 字段存在，没有逐项断言 `xess_d3d12_execute`、`fsr3_upscaler_dispatch` 和 marker evidence 的 JSON 值。当前证据是编译与模拟分类通过，不等于游戏实机 FSR3/XeSS 路由验收。构建期间只有查询 MSBuild 项目时 `--target help` 不适用（尝试查找 help.vcxproj）；改用 ALL_BUILD 后成功，此为查询工具兼容性问题，不是构建/测试失败。

2026-10-03 — FDP BCCB E attempt13：Main 逐个校验四个实际结果 SHA 与只读缓存 0 写。基准45.600139→候选43.604841，候选43.615935←基准45.220210 ms，净收益1.995298 / 1.604275 ms；平衡均值45.410174→43.610388 ms，快1.799786 ms（约3.96%）。这是离线完整 NR，游戏尚未部署、肉眼未验收，不能换算游戏FPS。E attempt14 C512 因历史真实对象归属检查失败，attempt15 Swin 因C128融合核正spill（最低320）预编译失败，两者无合格候选计时；修复已分开交给6.1 Sol max，Luna负责下一冻结版GPU。实际摘要已支持D/E并校验4个合格arm。证据：artifacts/b580-full-implementation-v1-20261003/FDP_REPEAT_MAIN_REVIEW.json。


2026-10-03 — r10 三组返修合并：C512 FFN流式K32、原生QKV真实帧归属修复、Swin C128 N32已合入并冻结，Main96项CPU通过。259份源文件/104个测试档只是源就绪数量，不等于104项优化或GPU已通过。Luna串行测C512和Swin；正spill诊断独立标注，默认仍要求零spill。自然档完整离线NR约52.820→44.694 ms、电影档53.176→45.182 ms，同臂冷热13帧字节一致、输出与私有历史有限、只读cache 0 miss/0写；标准档独立BCCB收益仍是约1.800 ms，三种风格收益不能相加。两种新风格未经过肉眼/游戏验收。E attempt18 CPU直接调度冷热不一致已交源返修；不作速度判断。证据：artifacts/b580-full-implementation-v1-20261003/FDP_CONDITIONAL_MAIN_REVIEW.json 和 WORK.json。


2026-10-03 — C512 完整组合与原生 QKV 调用修复：C512/ViT完整组合离线NR约46.005→54.067 ms，三轮均慢约8 ms，图本体39.659→46.529 ms；13帧输出与私有历史有限、冷热一致、只读缓存与回收通过，但不采纳该组合。原生QKV另发现旧fused_dot闭包使实际QKV调用为0，现同时修正完整消费者和原有注意力入口；r13的7项调用回归及96项组合CPU检查通过。Luna接续独立测QKV、C512 FFN、ViT FFN与编码段，未获GPU实测前不宣称原生QKV提速。r11/r12中间归档不批准GPU运行；游戏G盘版本未改。证据：C512_COMPLETE_MAIN_REVIEW.json、WORK.json。


2026-10-03 — r13 分段实测与 r14 首尾段隔离：原生QKV已确认16处实际执行，离线完整NR45.916→47.540 ms、独立图本体39.523→40.022 ms；C512 FFN46.239→47.357 ms、图本体39.833→39.896 ms。两项各13帧输出与历史逐字节同基准、冷热一致、只读缓存/退出通过，但均不采纳；图外增加量尚不能直接命名为格式转换耗时。ViT独立测试在基准的冷热输出检查失败，候选尚未运行；失败帧诊断runner已冻结，保留严格断言与具体误差。Swin完整组也尚未计时：首个decoder32 post尾段绕过新body入口。r14单独保留该现役K8尾段来测试其余模块，完整首尾段由独立开发修正。游戏版本未改。证据：R13_SPLIT_MODULE_MAIN_REVIEW.json、WORK.json。


2026-10-03 — r15 实际完整模块与并行编译：Swin44个完整模块共132个QKV/MLP/tail捕获入口已实际命中，首个decoder32 block66经DecoderGather回调进入新body；它与最终post block70不同。离线完整NR46.007→54.631ms、独立图本体39.500→47.517ms，慢8.624ms，其中图本体已慢8.017ms；不采纳，并继续改进实际tile/数据布局吞吐。C512编码段46.325→47.750ms，图本体39.852→40.473ms，同样不采纳。ScreenedLaunch46.269→46.418ms、各轮有快有慢，暂无可重复净收益；历史冷热字节失败这轮未复现，根因仍UNKNOWN。上述均13帧、冷热一致、有限值、只读缓存0miss/0write及退出通过，Swin最大误差0.0048828125，编码段0.003173828125。r14报告错误保留原始失败证据，r15单独修复序列化及未选编译项误验收。实际8路编译峰值私有内存约4.24GiB、最低系统可用33.60GiB，下一批编译16路；GPU初始化/捕获/测量仍由Luna串行。现役游戏未改，实机基础帧时间仍为用户验收55–60ms；独立模块收益不能相加。证据：R15_FULL_MODULE_MAIN_REVIEW.json、WORK.json。


2026-10-03 — r15 ViT 完整 FFN 实测：完整NR45.937→55.122ms（慢9.185ms），独立图本体39.779→47.376ms（慢7.597ms）。实际13帧输出/历史逐字节同基准、冷热一致、只读缓存0miss/0write及退出通过，但没有速度收益，不采纳。开发代理继续核对实际新核消费、DPAS tile、布局及图内额外处理；字节相同本身不能证明INT8模拟FP16。证据：R15_FULL_MODULE_MAIN_REVIEW.json。


2026-10-03 — r16 同源码检查配对与 r17 吞吐源码：冻结r16（260文件/109档），真实manifest CPU 46项及13帧预检通过。C512 FFN/编码段分别按完全相同数学实现，只切重复检查选项，由Luna串行测试31/32；GPU结果单独记录，尚未计入收益。ViT行量化一次＋K64完整累加/四段INT32归并候选已三方合并到工作源码，保留r16热检查和16路编译；等待Swin组合后冻结新修订。实际原生桥首轮在GPU执行前因adapter提前读取配置失败，原记录保留，Halley在新raw-stage2补真实配置绑定。现役游戏未改，55–60ms实机记录仍有效。证据：R16_MAIN_DISPATCH.json、MAIN_VIT_THROUGHPUT_COMPOSITION_r17.json、WORK.json。


2026-10-03 — r16 检查配对实测与 r17 吞吐候选：同源码C512 FFN47.475→47.049ms、编码段47.077→46.696ms，13帧输出/历史逐字节、冷热一致、只读缓存0miss/0write、清理及热准入实际计数通过。三轮中均有小反向，不宣称重复净收益；两项相对未采用的完整模块，不当作现役55–60ms游戏新增提速，不能相加。r17已冻结261文件/114档，实际46项CPU及五候选预检通过，保留默认关闭、全K/运动历史和16路编译。真实raw2纹理探针补配置后排在Luna串行队列，随后测ViT行量化一次K64完整/四段归并；Swin独立C64/C256/组合源码已就绪待明确GPU安排。证据：R16_HOT_MAIN_REVIEW.json、WORK.json。

2026-10-03 — r17 编译兼容性修正与原生 HDR 小样

r17第33臂在候选冷编译阶段被Triton不支持的static_assert membership AST拦下，尚无候选速度/画质结论。Main仅改三处等价断言并更新源码指纹，r18冻结261文件/114档，46项CPU检查通过；Luna串行35/P4、36/full，零spill门和完整K/时序合同不变。原生HDR桥raw-stage2 prepare/verify各两帧实际pack/export通过，raw字节一致且消费者回收正确；未运行模型/游戏，不是整桥验收，low测试尚未授权。现役游戏55–60ms版本保持。

2026-10-03 — AMD 最新结构深审（6.1 Sol Max）

固定AMD提交9ec741522d267c4d2365af53081716fb2943068a：作者1080网络归档约9.5–10ms，不能直接当游戏Present90fps。Sol以现役accepted720 C512+K8/all6为基准提出六项新模块方案：有效行QKV、ViT完整流式尾部、C32整窗、C64/C128共读QKV输入、跨token tile共享权重、C256窗口依赖队列。Main复核基准constructor与r18相同，C51219712→15360行算量正确；已存在的库QKV/INT8/融合尾部/图重放/GPU接力与r15退化修复不算新增。收益预算未测、不可叠加为FPS承诺；快速版允许原生数学数值变化，保留全K/真实时序控制。详见AMD_LATEST_MAIN_REVIEW和review目录REPORT/PLAN。

2026-10-03 — r18 ViT 完整行量化/分块累积实测

Luna35/P4、36/full均完成，13输出与历史/冷热同臂逐字节一致，readonly cache零miss/写，实际8FFN sites、零spill。P4完整process45.649→46.542ms（慢0.893）；full45.453→46.456ms（慢1.003），三cycle都慢。raw分别39.237→39.166和39.230→39.176，仅微量差值、非完整帧收益；不以两独立口径相减认定CPU独占成本。两候选不采用，未修改现役游戏。下一批Luna串行37–39原Swin资源修复；AMD逐核补审并行只读。


2026-10-03 — 审计测试进度统一汇报

已完成测试、采用状态和未测范围统一记录在 artifacts/b580-full-implementation-v1-20261003/TEST_PROGRESS_20261003.md。标准档FDP组合独立复测快1.800ms；自然／电影单轮约8.126／7.994ms待独立复测与肉眼验收。原生QKV、C512 FFN／编码段、完整Swin、原ViT及r18两档ViT均已有完整帧结果，负收益未采用；r16重复检查均值微利但cycle有反向，不计确认收益。Luna37–39测Swin修正版，AMD六项新方案未测、逐核补审仍在进行。18组源码就绪与114档不是GPU全部通过；未部署新候选，现役游戏55–60ms记录不变。
