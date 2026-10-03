# 第三方组件与边界

- [OptiScaler](https://github.com/optiscaler/OptiScaler) 为 GPL-3.0；本仓库发布的修改版 DLL 的对应修改源码见 `optiscaler/`，基础源码固定在 `7534ad00bf9e590eedb99e8dd9fd8c89dae3654f`。上游依赖与其自身许可随上游源码保留。
- [REFramework](https://github.com/praydog/REFramework) 和 [UpscalerBasePlugin](https://www.nexusmods.com/site/mods/502) 由用户从原作者处获取；本仓库不再分发 `dinput8.dll` 或 `PDPerfPlugin.dll`。版本与配置见 [OptiScaler 的 RE8 指南](https://github.com/optiscaler/OptiScaler/wiki/Resident-Evil-8-Village)。
- [XeSS](https://github.com/intel/xess) 文件由用户按 OptiScaler 指南准备；本仓库不再分发 `libxess.dll`。
- SF-v2 NR 模型由用户从[原发布页](https://github.com/RankFTW/rhi-repo/releases/tag/dlssnr-310.8.SF-v2)获取。安装器核对固定哈希并在本地提取资源。我们没有该模型的公开再分发许可，**来源标注不等于转发授权**。
- 既有便携 Python/XPU 运行时及其许可随[基础运行时发布](https://github.com/gggz114514-oss/b580-dlss5-comfyui-video-nodes/releases/tag/v0.2.0-pre)提供，安装器逐份校验后取得。
