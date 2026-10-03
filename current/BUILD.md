# 现役源码、构建与依赖边界

模型验证与执行见 [测试入口](../tests/README.md)。从仓库根运行 CPU 源码检查时，无需 Intel SDK 或游戏。

`bridge/` 为 Windows x64 C++20／CMake 源码。外供 OptiScaler SDK 固定为
`7534ad00bf9e590eedb99e8dd9fd8c89dae3654f`，包括其 NGX 头和 Detours 库。
冻结源码的默认关闭诊断构建有一个条件声明缺口：`diagnostic_dlss_scope`
只在身份复制／尾段探针启用时声明，公共状态记录也会使用它。
[构建准备工具](../tools/prepare_bridge_build.py) 核验 162 个源文件 SHA，
复制到仓库外，仅为关闭诊断的分支补声明，并写出原始／派生 SHA。
历史实机配置启用该条件分支，仍使用原来的声明。冻结 `bridge/` 不改写。

请在已配置 MSVC x64 与 Windows SDK 的终端使用独立源码、构建目录：

```powershell
python -I -B tools/prepare_bridge_build.py --output C:/nr-build/prepared-source
cmake -S C:/nr-build/prepared-source -B C:/nr-build/bridge -A x64 -DOPTISCALER_SOURCE=C:/deps/OptiScaler
cmake --build C:/nr-build/bridge --config Release --parallel 16
ctest --test-dir C:/nr-build/bridge -C Release --output-on-failure -R "^(nr_offline_tests|nr_sync_probe_tests|nr_color_state_proof_tests|nr_record_timing_tests|nr_gpu_handoff_cpu_tests)$"
```

默认构建不启用游戏专属的实机接入开关。历史实机版使用
`NRB_XESS_IDENTITY_TEST=ON`、`NRB_TAIL_ORDER_PROBE=ON`、
`NRB_DEFERRED_IDENTITY_TEST=ON`、`NRB_LIVE_NR_TEST=ON`、
`NRB_MULTI_ROUTE_LIVE=ON`、`NRB_DLSS_COLOR_STATE_PROOF_V1=ON`。
这些选项来自现役构建来源，不能据名称推断它们在其它游戏已经通过验证。
未知游戏先用被动 DLSS 输入探针；原生 XeSS 桥与 DLSS 桥已分别在 Cyberpunk 验收，FSR 实机桥尚未完成。

完整历史实机配置的编译入口如下；这只生成二进制，不自动安装或运行游戏：

```powershell
cmake -S C:/nr-build/prepared-source -B C:/nr-build/live -A x64 -DOPTISCALER_SOURCE=C:/deps/OptiScaler -DNRB_XESS_IDENTITY_TEST=ON -DNRB_TAIL_ORDER_PROBE=ON -DNRB_DEFERRED_IDENTITY_TEST=ON -DNRB_LIVE_NR_TEST=ON -DNRB_MULTI_ROUTE_LIVE=ON -DNRB_DLSS_COLOR_STATE_PROOF_V1=ON
cmake --build C:/nr-build/live --config Release --parallel 16
```

上面的五项测试为 CPU 合同检查。其它 CTest 项含 DLL／ASI 加载或资源探针，
须按自己的隔离测试环境选择，不属于这次 CPU 构建验收范围。

2026-10-03 已在 MSVC 14.44.35207／Windows SDK 10.0.26100.0、16 路并行下
完成默认编译、五项 CPU 测试以及六开关实机配置的编译；
实机配置中原始与派生 `asi.cpp` 的归一化预处理文本 SHA 相同。
这是构建验证，没有加载 ASI、运行 GPU 或替换游戏文件。
[构建回执](../evidence/2026-10-03/reproduction/BRIDGE_BUILD_REVIEW_V2_PUBLIC.json)
保留编译器、SDK、工具与输出身份。

`native/` 是已核对源码 SHA 的共享纹理／GPU 接力 helper，采用原仓库 GPL-3.0；需要 oneAPI SYCL 编译器、Level Zero 头、Windows D3D12 SDK。已记录构建使用 oneAPI 2026.1、`-fsycl -std=c++20 -shared -O2 -fexceptions -fcxx-exceptions -fsycl-device-code-split=per_kernel`，以及 `-ld3d12 -ldxgi -ld3dcompiler`。并行设备链接参数为 `-fsycl-max-parallel-link-jobs=14`。外供 `ze_api.h` 与编译器并未附在本仓库；请显式传入头目录并把输出放在仓库外。

来源清单证明源码与历史 build input 的对应关系，不证明其它机器新构建的 DLL 与安装二进制逐字节一致。本次复现测试直接执行模型，不替代原生桥、游戏画面或游戏帧时间验证。
