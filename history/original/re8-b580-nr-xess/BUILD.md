# 从源码重建游戏接入

发布包的游戏专用源码位于 `game/`、`native/` 和 `optiscaler/`。便携 Python/XPU 基础运行时、其依赖与许可由 `runtime-assets.json` 锁定在另一个已发布版本；SF-v2 模型文件不在本仓库。

## OptiScaler 代理 DLL

使用 Visual Studio 2022 Build Tools、Windows SDK 和上游要求的依赖，获取 [OptiScaler](https://github.com/optiscaler/OptiScaler) 及子模块，将上游固定到 `7534ad00bf9e590eedb99e8dd9fd8c89dae3654f`。在上游 Git 根目录运行 `git apply` 加载 `optiscaler/optiscaler-7534ad0-re8.patch`，然后把 `optiscaler/added/` 下三个文件复制进上游 `OptiScaler/wrapped/`。构建 `OptiScaler/OptiScaler.vcxproj` 的 `Release|x64`；补丁已移除与 NR 无关的自动移动/打包事件，构建产物为修改版 `OptiScaler.dll`，安装时改名为游戏目录的 `dxgi.dll`。本仓库保留 GPL-3.0 许可文本。

## D3D12 ↔ XPU 纹理桥

`native/nr_texture_bridge_re8_v1.cpp` 与 `native/nr_texture_bridge_v1.h` 构成桥源码。使用 Intel oneAPI DPC++/C++、Windows SDK、Level Zero 头文件及与便携运行时匹配的 Python 头文件，以 C++17、共享库形式编译，链接 SYCL、D3D12、DXGI、D3DCompiler，输出 `nr_texture_bridge_re8_v1.dll`。这份桥只负责互操作与测试纹理；模型推理代码在 `game/` 和基础运行时中。

## 资产包

先关闭游戏，并用便携运行时的 Python 执行 `tools/precompile_game_modes.py --runtime <运行时目录> --report <校验报告.json>`。该脚本准备 540p 与 720p 游戏来源、360p／480p／540p NR、普通执行／图重放、标准／融合历史，以及三种风格与滑块所触发的不同控制分支，并在新进程用禁止编译模式再测一遍。必须完整通过 168 个组合。

然后运行 `tools/package_overlay.py --runtime <运行时目录> --dll <修改版 OptiScaler.dll> --validation <校验报告.json> --output <re8-nr-overlay.zip>`。打包器核对被测游戏代码和缓存，拒绝只测部分档位的包。它从本仓库 `game/` 取源码，加入桥 DLL、预编译 B580 缓存和标量查询表；它拒绝 `WEIGHTS_HT` 文件，并把缓存元数据中的本机绝对路径替换成安装时占位符。`tools/relocate_game_cache.py` 在最终目录里将占位符解析为本机路径。将包的文件长度和 SHA-256 写入 `game-assets.json` 后方可发布。
