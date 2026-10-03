# Cyberpunk OptiScaler 桥：实机 XeSS 输入诊断（2026-09-26）

## 已验证

- 官方 SignPath 签名的 OptiScaler v0.9.4 主 DLL 保持原样。独立 `CyberpunkNRBridge.asi` 经 OptiScaler ASI 加载器注入；Detours 和便携 Python 初始化已移出宿主 `DllMain` 的加载锁。游戏进程中日志明确显示前 SR `NVSDK_NGX_D3D12_EvaluateFeature` detour 已连接，RE8 便携 Python 已导入。游戏正常响应，短诊断后通过正常关闭窗口退出。
- 游戏当前 XeSS Performance 配置的实机输入为 1280×720，SR 输出为 2560×1440。颜色与输出均为 `DXGI_FORMAT_R11G11B10_FLOAT` (26)，运动为 `DXGI_FORMAT_R16G16_FLOAT` (34)，深度为 `DXGI_FORMAT_R32G8X24_TYPELESS` (19)，曝光纹理为 1×1 的 `DXGI_FORMAT_R8_UNORM` (61)。颜色／运动／深度子矩形原点均为 (0,0)，MV scale 为 (1280,720)，HDR 标志为 1，运动低分辨率标志为 1。8 条有界的 XeSS 输入记录一致。
- 设备上观察到 9 条直通命令队列。上采样命令列表在前一次提交中的记录落在队列序号 1；命令列表指针可能复用，所以这只是**候选诊断**，不能证明当前帧生产者或消费者的提交次序。NR 帧数始终为 0，旁路状态为 `unsupported_input`；游戏画面未经过 NR。
- 网页在 NR 旁路时仍可启动，显示三入口截获数、NR 实际完成帧数和耗时。风格为三个独立按钮；360/480/540 是 NR 模型处理高度，不是游戏源高。当前游戏运行时通过 `CYBERPUNK_NR_RUNTIME` 指向已安装的 RE8 `nr-runtime`，不是可分发的独立安装配置。
- Debug/Release 各 4/4 离线测试通过；包含三入口模拟导出、FG handle 旁路、参数映射和网页 mock。模拟通过不代表 DLSS/FSR 已在游戏中执行。Luna 的独立 QA 对先前快照也得到两套 4/4，并指出生产者证明、输出退休代际、实机格式的剩余风险；加载锁、网页启动、输出/曝光元数据和风格按钮缺口已在当前代码修正。

## 为什么仍然不运行 NR

1. 现有 RE8 纹理桥不接受打包的 R11G11B10 HDR 颜色。直接把它当 RGBA8/32F 或把大于 1 的值简单截断，会改变灯光和 XeSS 的输入语义。需要独立的 GPU 格式／色域处理，并用实机像素范围与画面对照验收。深度和曝光仍须原样传给 XeSS；不能声称 RE8 NR 模型已消费深度。
2. RE8 的 RG16F/RG32F 运动读取按单位倍率处理；当前 XeSS 输入明确给出 (1280,720) 倍率。供 NR 使用的运动必须按该倍率、方向与历史约定转换，不能把未转换纹理直接接入。传给 XeSS 的原始运动纹理保持不变。
3. 当前 ASI 在 XeSS 评估回调看到的是**仍待提交的命令列表**及其资源指针。OptiScaler 的公开 ASI 加载器没有本帧颜色/运动写入已提交的回调；从命令列表反查不到可信生产队列。`source_submission_proven` 故意不会被用户布尔开关置真。即使把 9 条队列缩成候选 1，也不能安全启动使用独立队列的 RE8 推理。
4. NR 输出退休目前按命令列表指针＋提交序号诊断，尚无命令列表代际或消费者完成令牌。只有在获得可信当前帧生产／消费边界后才能将此路径打开；否则维持旁路，避免资源过早复用。

## 下一步实施边界

先确定可以证明当前帧源纹理已完成生产、以及替换后的 XeSS 消费已提交并完成的入口。可审计的最小接口须提供同设备的直通队列、帧 ID、当前颜色/运动资源、生产完成 fence/value 和消费者完成 fence/value；如果游戏仍在同一未关闭命令列表录制，则改成同列表内 NR 或在可控边界分段提交。单靠已签名 OptiScaler 的 ASI 加载功能无法提供此证明。随后再做 R11G11B10 HDR 和运动倍率的隔离适配，并逐帧比对画面；不碰 RE8 原工程和精确分支。DLSS/FSR 游戏入口尚需实机切换验证，不能由 XeSS 结果代替。

构建产物在 `D:\Codex-NR-Experiments\cyberpunk-opt\build-sol\Release\CyberpunkNRBridge.asi`，部署副本在游戏 `bin\x64\plugins`；诊断在同目录 `CyberpunkNRBridge.log`。之前产生的 OptiScaler 大日志已移入 D 盘任务备份目录，并把本游戏隔离配置的日志等级由 Trace 改为 Info，配置原件也已备份。未修改主 DLL，未发布。
