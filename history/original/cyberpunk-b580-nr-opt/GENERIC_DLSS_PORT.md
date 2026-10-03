# DLSS 游戏接入：独立诊断版与实机放行条件

截至 2026-09-26，《赛博朋克 2077》的 DLSS 性能档已验证 **游戏 1280×720 输入 → B580 NR → XeSS 超分 2560×1440 → XeFG 或游戏原生帧生成**。这证明本项目能在一款游戏内把 NR 插在超分之前；尚不能证明同一 ASI 可直接用于其他游戏。现有实机桥仍保留在《赛博朋克》目录，不与这里的候选混装。

## 新游戏第一步：只读 DLSS 输入探针

独立的 `GenericDLSSProbe.asi` 只挂接 OptiScaler 的 NGX Evaluate 入口，读取少量调用的参数和纹理描述，随后把 **原调用原参数**交回 OptiScaler。它不加载 NR 模型、不建立 GPU 队列钩子、不发 GPU 命令，也不修改颜色、运动、深度或输出纹理。日志写在 ASI 同目录的 `GenericDLSSProbe.log`，记录调用方模块、handle、输入/输出分辨率、资源格式和大小、运动倍率、抖动、曝光、历史重置等。前 8 次调用和之后按 2 的幂间隔采样；每类最多约 18 次，防止游戏菜单的过渡尺寸耗尽样本。

源文件为 `src/generic_dlss_probe.cpp`，构建开关为 `NRB_BUILD_GENERIC_DLSS_PROBE`，默认关闭以免改变常规构建。本机候选产物固定在 `artifacts/generic-dlss-probe/GenericDLSSProbe.asi`，与 D: 编译产物 SHA-256 相同。它依赖已安装、会调用 `InitializeASI` 的 OptiScaler ASI 加载方式；**单独放入游戏目录不会自动运行**。同一进程不要同时加载它和 `CyberpunkNRBridge.asi`，因为两者都挂接同一入口。当前 OptiScaler 源码与构建固定于 v0.9.4 `7534ad0`。尚未在第二款游戏安装或验收这个探针。

诊断日志里的 `route=unknown` 不等于游戏没有走 DLSS。比如 Streamline 的 `sl.common.dll` 从另一个模块发起调用时，本探针会记录 `external_untagged`，仍保存尺寸和纹理信息；不能只凭 handle 范围或可选参数键把它升级为可信的 DLSS 证明。`bypass=producer_or_consumer_order_unproven` 是探针的固定安全状态，表示它没有执行 NR。

## 从诊断到可用 NR 桥

新游戏至少还需核对：实际游戏场景是否连续命中同一超分入口；颜色与运动纹理的格式、范围、运动方向/单位和抖动；输入尺寸及 HDR/曝光；当帧输入何时写完、OptiScaler 何时读取；NR 输出应替换哪张纹理以及它何时可安全回收。先做同帧恒等回送，再开放 NR，最后验证画面、运动、HUD、超分与帧生成。任何不匹配都旁路并保留原画面。

目前 NR 执行端和游戏接入着色器还有 960×540、1280×720 等尺寸约束；《赛博朋克》的颜色尾部和资源状态判断也只针对当时实测的 1280×720→2560×1440。新游戏的输入若是 720p 以外、格式或命令顺序不同，需要单独适配，不能只改分辨率常量就放行。已验证的《赛博朋克》桥和 OptiScaler 签名主 DLL 在验证新游戏时保持原样。

本地验证：`nr_generic_probe_smoke` 使用模拟 OptiScaler 连续调用 12 帧，确认原调用成功返回并只记录限定数量的诊断行。它不代替新游戏实机测试。
