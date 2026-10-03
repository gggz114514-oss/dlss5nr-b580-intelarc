# 三条跨游戏入口桥与共同后端的边界（2026-09-26）

目标是三条**各自可跨游戏复用**的入口链：`DLSS→NR→XeSS SR→XeFG`、`FSR→NR→XeSS SR→XeFG`、`XeSS→NR→XeSS SR→XeFG`。三路共享 `NRB_Frame`、RE8 NR 处理器和未来的 XeSS 输出/退休合同；它们不能共享未经核对的运动倍率、输入子矩形或路由识别假设。Cyberpunk 只是验证游戏，不是公共合同的默认值。

| 层 | 当前代码 | 跨游戏复用范围 | Cyberpunk 证据 / 尚缺 |
|---|---|---|---|
| DLSS 入口 | `src/routes/dlss.cpp`，游戏/Streamline 调用来源记录 | 标准 NGX 颜色、运动、深度、jitter、reset、输出参数和调用来源 | 已在游戏实机命中。过渡尺寸约 1505×847，之后重建至 1280×720→2560×1440；颜色 fmt26、运动 fmt34、深度 fmt39、auto-exposure=1。仍未替换一帧。 |
| FSR 入口 | `src/routes/fsr.cpp` | FFX 的每帧 upscaleSize 可覆盖上下文最大输出尺寸；原输入参数保留 | 模拟入口通过。`FSR.frameTimeDelta` 只是可选诊断标记，尚无外层 API route tag 和游戏实机验收。 |
| XeSS 入口 | `src/routes/xess.cpp` | Opti 已折算进 NGX 的 MV scale 原样记录，不能视为原生 XeSS 的原始 velocity scale | 早前游戏实机命中：1280×720→2560×1440，fmt26/34/19。独立观察到 `xessSetVelocityScale=(2,-2)`，Opti 内部 NGX scale=(1280,720)；坐标/尺度转换未交给 RE8。缺外层确定性 tag。 |
| 共享帧合同 | `include/nr_bridge.h`、`src/nr_ngx_adapter.cpp`、`src/nr_core.cpp` | 纹理尺寸、子矩形、输出 base、格式、运动比例来源、曝光、历史和同设备校验；HDR 格式是合法的**入口数据** | 允许诊断 R11G11B10 浮点输入，但不表示现有 RE8 SDR 模型可处理。 |
| RE8 后端能力 | `include/nr_contract.h`、`src/nr_contract.cpp`、`src/nr_python.cpp` | 与游戏无关的能力表：已缓存的源尺寸、可读格式、运动尺度、曝光、HDR 语义等；列出所需适配而非默默转换 | Cyberpunk DLSS 至少要求颜色格式适配、HDR/SDR 范围判定、运动尺度适配。动态源尺寸还要求已准备的缓存。 |
| XeSS SR/FG 输出 | `src/asi.cpp` 中仅在安全条件满足时替换 NGX `Color` 并等待/退休 | 目标是三路共享一次输出接线 | 当前生产者提交证明和消费者代际退休都不足，`source_submission_proven=false`，实际 NR 仍为 0。FG 未做三路联动验收。 |

## 路由证据

当前公共 Evaluate 钩子把游戏主模块/Streamline 的 NGX 调用列为 DLSS 证据。Opti 内部的 `FSR.frameTimeDelta` 与 `XeSS.ExposureScaleTexture`/`ResponsivePixelMask` 只写入 `route_evidence` 诊断，不授权 NR：键可能缺失，也可能在共享参数容器里残留。其余内部/外部调用标 UNKNOWN 并透传。下一阶段在三个**外层 API 入口**建立有作用域的线程局部 route tag，分别覆盖 NGX DLSS、FFX FSR dispatch 与 XeSS execute；进入共同 Evaluate 时消费 tag，外层函数退出清除。必须先在各游戏验证实际 API 路由和 Opti 版本，不能靠一个游戏的键推导所有游戏。

## DLSS 先行的恒等替换

`include/nr_identity.h` 与 `src/routes/dlss_identity.cpp` 已编译一个**同命令列表、同格式** `CopyResource` 录制函数：仅在已证明当前列表 Reset 代际、该代中颜色的最后写入早于 Evaluate、源纹理当前状态、目标纹理起始 `COPY_DEST` 状态、同设备同布局，以及目标有 XeSS 消费完成退休跟踪时才允许录制；源状态随后恢复，目标转为 shader-read。离线测试验证上一代写入记录被拒绝。当前 ASI 没有这些证明的生产者，函数**没有接到游戏回调**，也没有 `identity_substitution` 实机结果。Opti 的 XeSSFeature 会根据自身配置/quirk 在 Evaluate 内转换颜色资源状态，不能据此倒推调用前的状态；不要通过网页布尔值伪造状态证明。`strength=0` 会触发 RE8 host 的 `[0,1]` 夹断，也不能当浮点恒等对照。此里程碑的下一步是可信状态跟踪与输出生命周期，再由用户在同一场景验收恒等替换画面。

## 像素范围与色调判定

游戏菜单 HDR 关闭，而内部纹理仍为 fmt26 且带 HDR flag。两者都不能证明实际像素大于 1。`tools/analyze_r11g11b10.py` 只分析**已经完成且经过 fence 的** R11G11B10 readback 文件，报告整帧或有界采样范围、>1 像素数、最大值与非有限值；它不读 GPU，也不证明回读的来源。后续只读 GPU 采样器须记录 DLSS 入口帧 ID、纹理尺寸/row pitch、源资源状态证据、同队列副本完成 fence/value、场景与曝光，并在亮灯、暗部、人脸各取少量完整帧。若来源仍在未关闭的游戏命令列表内，不能让独立队列提前拷贝。仅凭几个采样帧没有 >1，也不能证明所有游戏场景都是 SDR；确定范围后再设计以原 HDR 画面为底、只叠加有界 NR 残差的候选，配对肉眼审核。

## 验收口径

每条路分别核对外层 API tag、规范化帧合同、生产/消费顺序、源格式与运动转换、恒等替换、实际 NR 完成帧、XeSS SR 结果和 XeFG 历史稳定性。`intercepts>0` 仅证明钩子看到调用，`processed_frames>0` 才证明 NR 已运行；两者仍不足以替代画面验收。现有源码只在 E，构建/测试在 D；用户负责 G 盘部署及实机切换，本轮 CPU 工作不改 G、不占 GPU。
