# 社区整数化与后端复用范围（2026-09-08）

用户问“社区有把dlss5 int化的项目吗？我们的做出来以后通用吗？”；结合此前INT8话题，
本次按整数化/INT8理解。目前搜索未找到公开、可复现且已验证在B580执行完整原始NR模型的
INT8项目。此结论是当前检索结果，不是对所有私人或未来项目不存在的断言。

- [DLSS-NR-on-AMD](https://github.com/danielblnc/DLSS-NR-on-AMD)：作者说明为AMD HIP运行时重实现，
  面向RX9000/7000。已查README及[发布记录](https://github.com/danielblnc/DLSS-NR-on-AMD/releases/tag/v0.2.15)
  未证明模型已经INT8量化；也没有承诺全字节一致。不能将“AMD可运行”直接解释为“INT8后端可移植到Intel”。
- [dlss5-as-inpainting](https://github.com/taowen/dlss5-as-inpainting)：作者区分原生逐位路径、
  不完整语义图和可移植蒸馏近似模型。后者不满足本任务原模型精确迁移目标，也不是已完成的原模型INT8移植。
- [video2dlssnr](https://github.com/DaniilSokolyuk/video2dlssnr)：D3D12原生NR调用、视频/图片和
  ComfyUI包装，依赖NVIDIA执行环境；属于工具接入参考，不是独立Intel INT8执行核心。
- [MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS)：Apple Silicon的MLX/Metal、PyTorch和
  Core ML重实现，README描述半精度/FP8舍入及浮点执行，未找到INT8完成声明。
  作者报告NR与DLL存在非零MAE，且新增部分视频时序启发式；不能视为原生全字节/完整时序等价。
  可供跨平台架构和算子优化参考，本次仅阅读，未下载运行或将其模型替换进本项目。

本项目主后端已接入的INT32路径只缩短有范围证明的对齐乘积转换/求和，之后扩回INT64累加与舍入。
模型权重、激活数值格式未换成INT8，主后端没有调用XMX/DPAS。

2026-09-08再次核对上述项目的公开说明，尚未找到可复现的完整原始NR INT8 B580项目。
本地新增了独立的INT8 XMX算子候选：能无损整数表示且满足累加精度条件的K16组用INT8，
其余输出保留旧整数算法。9组GPU测试和3组基准的每次输出均逐字节一致；编译IR包含DPAS及
SubgroupMatrixMultiplyAccumulateINTEL调用。这不是完整NR模型重量化，也未接入主后端。
基准中满足无损条件的比例仅3.4%–6.6%，速度为旧算子的0.705–0.877倍，因变慢而暂不采用。
证据：reference/xmx-int8-exact-xpu-v2.json，源码和IR快照在
reference/experimental/revisions/xmx-int8-attempt-v2。
另有完整INT32累加/舍入候选正在验证，属于保值整数运算优化，不能称为INT8模型量化。
真正INT8重量化仍需另做误差、时序与性能验证；B580峰值INT8算力不能直接换算本模型帧率。

复用目标是固定SF-v2 NR核心加工具适配器：OptiScaler、XeSS/ComfyUI、视频工具可共用
同一执行核心，对新素材无需重新训练，4060仅开发对照使用。此目标尚未全部接入。
跨工具仍须处理纹理/显存共享、运动输入、颜色格式及历史重置；其他Intel型号、其他GPU架构、
其他DLSS5模型/运行库版本需要单独适配验证。当前证据不等于所有游戏/硬件/版本即插即用，
也不意味着任意AI模型可直接调用NR专用图。

[OptiScaler_DLSSNR](https://github.com/Dagherbou/OptiScaler_DLSSNR)已提供DX12 NR接入方向的
社区参考；它的存在不代表本项目已接好B580后端。共享核心还需各调用方的GPU资源、运动方向/
单位、颜色与历史重置适配。普通用户在已验证的B580配置上使用该核心时，4060仅是开发测试
参考机，不需要逐帧联网调用；实时游戏性能仍须单独达标。
