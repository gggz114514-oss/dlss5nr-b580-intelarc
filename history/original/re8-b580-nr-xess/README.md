# 生化危机 8：B580 NR → XeSS 实验接入

这个包将游戏送入 XeSS 前的画面交给 B580 上的 **快速版 NR**，再交还 XeSS 超分。网页可开关 NR，选择 360p／480p／540p 模型输入、三种风格、模型强度、局部色调/结构、历史方式和计算图重放，并显示 NR 处理耗时与估算吞吐。NR 输出的差值合回游戏原始渲染尺寸。**快速版并非 NVIDIA 原版或逐字节一致的精确版；画面会变化。**

`dxgi.dll` 是接入入口，推理还需要旁边的 `nr-runtime`（Python/XPU 算子、缓存与用户本地取得的模型）；单独复制 DLL 不会运行 NR。

## 安装

目前支持 Windows 上的 Intel Arc B580 和《Resident Evil Village》D3D12 路线；已准备的 XeSS 前游戏渲染输入是 **960×540 与 1280×720**，其他输入尺寸会安全跳过 NR。请先拥有游戏，并按 [OptiScaler 的 RE8 指南](https://github.com/optiscaler/OptiScaler/wiki/Resident-Evil-8-Village)安装这些原作者组件：REFramework 的 TemporalUpscaler 适用版本（`dinput8.dll`）、UpscalerBasePlugin **1.1.2**（`PDPerfPlugin.dll`）、OptiScaler 0.9.4（`OptiScaler.dll`、`libxess.dll`、`OptiScaler.ini`）。在 REFramework 中启用 TemporalUpscaler，确认纯 XeSS 游戏画面正常。安装器会检查这些文件；它们不在本仓库的发布包中。

从本仓库的 [Releases](https://github.com/gggz114514-oss/re8-b580-nr-xess/releases) 下载 `re8-b580-nr-xess-installer-v0.1.0-pre.zip`，解压到任意普通文件夹，双击 `Install.bat`，按提示粘贴游戏目录路径（含 `re8.exe`）。也可直接在 PowerShell 运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Install.ps1 -GameDirectory '你的游戏目录（含 re8.exe）'
```

安装器从既有 [B580 NR 运行时发布](https://github.com/gggz114514-oss/b580-dlss5-comfyui-video-nodes/releases/tag/v0.2.0-pre)获取并逐份校验运行时，再获取本仓库同版本游戏增量包。它会从[第三方 SF-v2 原始发布](https://github.com/RankFTW/rhi-repo/releases/tag/dlssnr-310.8.SF-v2)下载固定版本 DLL，在**你的电脑**上提取所需的 `WEIGHTS_HT` 资源并校验哈希；本仓库不提供该模型文件。已有同版 DLL 时可向安装器传入 `-ModelDll '完整路径'`。下载与解包需要较多磁盘空间；请预留至少约 12 GB，且保持游戏关闭。

安装器先在游戏目录下准备完整运行时，然后备份原有 `dxgi.dll`、NR 标记和 `OptiScaler.ini` 到 `.nr-backup`，最后才替换代理 DLL。它会把 OptiScaler 的 `ManualInputPolling` 设为 `true`：本机实测 NR 开启时，默认窗口消息接管在手柄/鼠标切换及游戏失焦时会崩溃；手动轮询输入的同场景复测稳定。代价是鼠标回到游戏窗口后通常需要点击一次才能重新控制游戏，OptiScaler 菜单打开时也可能无法阻止游戏接收输入。安装中断时不要手工删除暂存目录；可先看报错。安装后进入实机场景，访问 **http://127.0.0.1:8765/** 调整 NR。确认网页显示具体“当前档位”且“已处理帧数”持续增加；仅开关显示为打开并不能证明 NR 已执行。网页的“估算吞吐”是 NR 平均处理耗时换算值，**不是游戏实际帧率或帧生成速度**。首次选取新输入尺寸/参数可能需要准备，360p 默认是融合采样和计算图重放。游戏过程中不会现场编译缺失的算子：若网页报告 NR 失败，保存进度后退出游戏，查看 `nr-runtime/logs/nr-pre-xess-python-error.txt`。

恢复原文件（游戏关闭时）：双击 `Restore.bat`，或运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Restore.ps1 -GameDirectory '你的游戏目录（含 re8.exe）'
```

该命令校验当前代理 DLL，再从最近一次 `.nr-backup` 恢复原文件与原始输入设置；若你安装后又修改了 `OptiScaler.ini`，恢复器会保留你的修改并提示原文件的备份位置。它不会自动删除 `nr-runtime`，避免误删用户数据。

## 当前验证范围

详细的验证条件、结果与局限见 [VALIDATION.md](VALIDATION.md)。

- 在开发机 B580 上，独立运行时和游戏目录运行时通过 **960×540 合成输入的两帧 GPU 自检**；游戏目录又通过 **1280×720 来源、默认 360p 图重放的两帧纹理桥自检**。540p/720p 游戏来源与网页控制分支共 168 个组合通过重新启动进程后的预编译缓存只读复测。
- 《村庄》XeSS 前路线曾在开发机实机运行，360p／480p／540p 和网页开关经过人工观察；具体游戏版本、驱动、场景会影响表现。
- 本机实机对照中，NR 关闭而 XeSS 保留时，手柄/鼠标切换正常；NR 开启、OptiScaler 默认窗口消息接管时，游戏失焦可复现崩溃；仅切到 `ManualInputPolling = true` 后，NR 持续处理且用户确认切换稳定。其他电脑、其他游戏、不同版本驱动或非 B580 GPU 尚未完成验收。
- 若 NR 启用后画面/灯光异常，先在网页关闭 NR；游戏退出后可运行 `Restore.ps1`。

## 代码与资产

`game/` 是网页控制与 XeSS 前 NR 的游戏专用 Python 代码；`native/` 是纹理桥源码；`optiscaler/` 含修改 OptiScaler `7534ad0` 的补丁、附加源码和 GPL-3.0 许可。`tools/` 提供增量包制作、缓存路径重定位和合成帧自检。基础 NR 运行时来自[另一个公开项目](https://github.com/gggz114514-oss/b580-dlss5-comfyui-video-nodes)，在安装时按版本与哈希获取。

增量包内有预编译 B580 算子、编译缓存、游戏控制代码、纹理桥、修改后的 OptiScaler DLL 与标量查询表；**没有 `WEIGHTS_HT.bin` 或 NVIDIA DLL**。缓存对应本次测试使用的 B580 与驱动；其他驱动若不能复用缓存，NR 会报告失败，不会在游戏中长时间现场编译。编译缓存的机器路径在打包时抹除，安装到最终路径后重建。

本仓库的自有接入代码以 GPL-3.0 发布；模型及第三方组件不因本仓库许可而改变权利归属。第三方组件的获取地址和许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。本项目与 Capcom、NVIDIA、Intel、OptiScaler、REFramework 作者无官方关联。
