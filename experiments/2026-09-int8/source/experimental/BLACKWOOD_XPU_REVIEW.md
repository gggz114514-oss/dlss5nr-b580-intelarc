# Blackwood416 资料与 NR 后端的对应关系

2026-09-09 源码评估。结论：值得采用其数据布局、ESIMD/DPAS 和验证方法作为下一轮实验参考；没有证据表明安装整个扩展就能加速 NR。本轮未运行外部内核、未安装扩展、未更换 Torch/编译器、未发布。

当前性能基线仍为 fee6d0d 的 WindowBlocks v3 + 已选 FP16/FP8GraphRewrite 栈：同次配对 NR256 完整连续调用 12.426860→12.250884 ms，1080p 残差路径 13.260352→13.132500 ms。这些是此前本地优化的成绩，不是此次外部项目带来的收益。完整迁移目标仍未完成。

## 固定来源

只读快照在 `D:/Codex-NR-Experiments/nr-b580/reference/research/blackwood-review-v1`。

| 来源 | 固定提交 | 与本任务的关系 |
| --- | --- | --- |
| [omni-xpu-kernel](https://github.com/Blackwood416/omni-xpu-kernel) | c6de520af4899830bd592aa8ad16f7fa8ca07209 | SYCL 扩展、DPAS 注意力、oneDNN INT8、目标架构分派 |
| [A770-Kernel-Optimization](https://github.com/Blackwood416/A770-Kernel-Optimization) | b05a20a67660a57b1a49658fd0d604f1d30c5f57 | GEMM 布局片段、寄存器/SLM 和负收益记录 |
| [ComfyUI-OmniXPU](https://github.com/Blackwood416/ComfyUI-OmniXPU) | 08a10a3980e09ad891beb6eadbc23ffa1a46de74 | 上层形状分派、常量缓存、现有 ComfyUI 集成参考 |
| [Aila](https://github.com/Blackwood416/Aila) | e00c2adeedbb9f7ef31d5769860393c54f0cb476 | SYCL/oneDNN 常驻算子、低比特融合和 C API |
| [InferRef](https://github.com/Blackwood416/InferRef) | ae91e11534b6b7708d4d6260b9333eeab7db8bec | 分层参考、测试用例提取、首次差异和时序场景 |
| [博客源码](https://github.com/Blackwood416/Blackwood416.github.io) | 6ee90e05bbd33d028b210cfb9ac5b6d1d0de3f6d | XPU GEMM 长文、oneAPI 学习日志 |

博客域名 `https://blog.blackwood.cv` 在网页工具中无法打开，改读作者公开仓库内文章原文。只下载了七篇相关 Markdown，没有下载博客图片或模型。GitHub 账号的 16 个仓库已做目录筛选；语音应用、USB 屏幕、Android/Linux 等不作为 NR 性能优先项。A770 仓库中的指南按外部参考资料审阅，未安装为本任务 skill。

## 1. 最值得做：按 DPAS 操作数布局传递数据

文章 `src/content/blog/xpu-trick-01.md`，尤其“操作数直通布局与寄存器重排消除”和“循环地址步进优化与 4 级缓冲屏障减半”，以及 A770 仓库 `references/api/code-snippets.md` 的 ESIMD 16×16 core，提供了可检查的实现片段：

- B 采用矩阵指令所需的打包排列，A 按操作数段组织；从 SLM 到寄存器直接整块加载，减少 select 拼接。
- 循环外计算地址，K 循环只做固定步进；实验比较同步频次、缓冲深度和工作组形状。
- 文中更大 GRF、更深缓冲和软件预取均有负收益例子，不能把它们直接当默认开关。

这些是作者在 A770、BF16、特定预打包 GEMM 上的结果。博客计时包含主机发射及完成等待，但输入遵循预打包约定；其中动态 A 的打包成本不能在我们的整帧比较中省略。静态权重可以加载时打包；逐帧激活应由上游直接写出所需排列，或者把额外打包成本完整计入。

我们刚完成的 WindowBlocks v3 已经让注意力输出直接交给投影。进一步候选是让投影/QKV/MLP 的输入布局与 XMX 操作数一致，并用 ESIMD 明确控制加载和点积。需要从真实捕获形状开始；不能把 BF16 示例简单改名为 FP16，就声称保持现有舍入顺序或相同字节。

参考：[文章固定版本](https://github.com/Blackwood416/Blackwood416.github.io/blob/6ee90e05bbd33d028b210cfb9ac5b6d1d0de3f6d/src/content/blog/xpu-trick-01.md)、[代码片段](https://github.com/Blackwood416/A770-Kernel-Optimization/blob/b05a20a67660a57b1a49658fd0d604f1d30c5f57/references/api/code-snippets.md)。

## 2. INT8 有融合参考，但不能据名称判断计算精度

`omni_xpu_kernel/csrc/onednn_int8.cpp` 的 scaled primitive 设置逐行激活尺度、逐列或标量权重尺度、可选 bias 和 FP16/BF16 输出；缓存 primitive，使输出缩放不必单独提交一个内核。`utils.h` 从 Torch 当前 XPU stream 取得 SYCL queue，适合研究与现有运行时的衔接；这本身不证明我们的图捕获兼容性。

关键限制：该文件设置 `attr.set_fpmath_mode(dnnl::fpmath_mode::any, true)`。oneDNN 文档说明第二个参数使整数 primitive 应用浮点数学模式，并引入隐式权重上转换。因此不能仅凭 s8 输入或注释就宣称一定执行整数 DPAS。实际采用前要记录实现名、编译产物和输入/输出完整成本。

我们已有 `fused_activation_int8_v1.py`，已将有限形状的激活量化、INT8 点积、两次有序缩放和初始累加融合。新项目不是首次提供这个思路。我们的量化使用对称半值远离零规则；它的 ConvRot 示例使用 rnde，还包含 Hadamard 旋转，不能直接替换并宣称字节相同。那条 DG2 融合 ConvRot 路线在源码中还明确保留 BMG 未实测的限制。

参考：[INT8 源码](https://github.com/Blackwood416/omni-xpu-kernel/blob/c6de520af4899830bd592aa8ad16f7fa8ca07209/omni_xpu_kernel/csrc/onednn_int8.cpp)、[oneDNN 数学模式文档](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_fpmath_mode.html)。

## 3. 有 BMG 源码，但没有 B580 NR 成绩

首页强调 A770；`setup.py` 实际包含 `bmg`、`ptl-h`、`dg2` 目标，核心也有 BMG 专用实现。`device_utils.h` 仅将 E210/E211 分派为 b60、E223 分派为 b70，其他 ID 为 unknown；`bmg_kernel_policy.h` 的通用 BMG 策略沿用 B70。应为 B580 单独实测，不能将 B60/B70 策略视为已验证的 B580 调度。

现成 `sdp` 入口要求 batch=1、head_dim=64 或 128，签名只有 q/k/v。我们的已选 Swin 核心是每窗口 64 token、head_dim=32，包含每头 64×64 bias 和固定 half/FP8 边界。尺寸、偏置和数学契约都不匹配。其 attention 内核可提供打包与调度参考，不能作为现成替换函数。

参考：[设备分派](https://github.com/Blackwood416/omni-xpu-kernel/blob/c6de520af4899830bd592aa8ad16f7fa8ca07209/omni_xpu_kernel/csrc/device_utils.h)、[SDP 入口](https://github.com/Blackwood416/omni-xpu-kernel/blob/c6de520af4899830bd592aa8ad16f7fa8ca07209/omni_xpu_kernel/csrc/sdp.cpp)。

## 4. 其他项目的用途

**Aila：** `src/ops/Linear.cpp` 缓存 oneDNN primitive、scratchpad、部分调用参数，并在单 token 时选择 GEMV；提供 bias/GELU 融合。`Bnb4BitLinear.cpp` 的 NF4 路线在 SLM 中反量化为 BF16，再执行 joint_matrix。它是量化存储加浮点矩阵计算的例子，不能用它证明全程整数计算收益。可参考常驻执行与布局管理；单 token 语言模型加速不能外推到 NR。

**ComfyUI-OmniXPU：** 可参考形状/目标选择、量化权重副本缓存及生命周期。我们 NR 固定模型已常驻 GPU，不能把大模型 CPU 卸载节省直接计为 NR 收益。当前任务继续优先本体，避免全局 monkey patch。

**InferRef：** 更适合保存“参考运行→最小分层用例→候选→首次差异”的过程，契合用户将来发行复刻方法的需求。当前前端基于 TorchDispatchMode，不会自动恢复闭源 NR DLL 内部图，也不能替代我们的自定义 Triton 发射捕获。默认 numeric comparator 是容差判定，不等于字节相同；精确分支仍需完整字节、历史、seed 和输入所有权证据。暂不迁移全部已有数据格式。

## 下一轮实验选择

1. 先做已有快速分支中真实 QKV/投影矩阵的 ESIMD 小型候选：固定 B580 目标，权重加载时打包，激活布局由上游生产者提供，保留当前 half/FP8 数学顺序。至少比较原 Triton、打包但不融合、生产者直接布局三种完整成本，避免只比较预打包矩阵乘法。
2. 编译阶段保留现有 spill preflight 思路；ESIMD 路径补足同样的资源/产物报告。检查矩阵指令、寄存器重排、SLM 访问和屏障，静态指令数只作为诊断，不能替代设备计时。
3. 若单块正确且成本有收益，接入 WindowBlocks v3 基线做完整 NR256、1080p 残差的配对验证。输出相同复用用户审核；改变数值的候选按已有要求提供完整视频。INT8/近似候选不算精确迁移完成。

上述实验尚未执行，不承诺速度倍率。全量后端替换、全局安装、运行作者整套 ComfyUI 工作流均不在本次评估的必要步骤中。
