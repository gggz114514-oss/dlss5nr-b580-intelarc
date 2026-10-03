# allanmeng 的 XPU 项目与 NR 的可用部分

2026-09-08。已检查四个 allanmeng 仓库，并沿其明确引用检查两个底层项目。
结论：对 Windows XPU 集成和内核调度有参考价值，尚未发现可直接执行固定 NR
模型并保持 NVIDIA 算术语义的现成后端，也没有因此获得新的 NR 提速证据。

本次只读取源码，没有运行社区代码、安装插件、加载其 DLL 或修改 Python 环境。
GitHub API 限流、Git 延迟取 blob 失败后，使用固定提交的 raw 源码与本地 Git 树
中的 blob ID 逐一校验。127 个文本文件共 1,536,030 字节；下载不等于全部逐行审查，
实际重点检查的函数和范围列在下方。两个 source-manifest.json 保存逐文件 SHA256。

| 仓库 | 固定提交 | 本次关注 |
| --- | --- | --- |
| allanmeng/ComfyUI-AIMDO-XPU | 92d11c477575d45cad81c73f5cbad3a801a035da | Python 显存缓存、CUDA shim；README 指向原生 Level Zero 上游 |
| allanmeng/ComfyUI-Qwen3TTS-XPU | 26b7caef49e9fb0f0d82eec232e8280c02bcbad1 | nodes.py 的编译、量化；形状预热实验说明 |
| allanmeng/llama-cpp-python-sycl-windows | 015df3536c716a59689dec5c25af60a6e8c5a2e8 | Windows 编译和运行库装载文档；该 Git 树主要是文档，非其自研矩阵内核源码 |
| allanmeng/ComfyUI-Aila-XPU | 740fcd50be35ab172e19f98db06f7a172a577cc5 | C API/工作进程封装；README 指向 Blackwood416/Aila |
| xiangyuT/comfy-aimdo-xpu | 908ea465c6614896612bf51ede2d1dabe7248d1f | dev/xpu-level-zero-vbar 分支的队列、显存、LUID |
| Blackwood416/Aila | e00c2adeedbb9f7ef31d5769860393c54f0cb476 | XMX NF4 GEMM、oneDNN primitive 缓存、运行上下文及数值测试 |

**Qwen3TTS 的 INT8 不是我们要找的 INT8 矩阵内核。**
[nodes.py:418](https://github.com/allanmeng/ComfyUI-Qwen3TTS-XPU/blob/26b7caef49e9fb0f0d82eec232e8280c02bcbad1/nodes.py#L418)
按行求 scale、round/clamp 后保存 int8 权重；forward 的第437行先转回输入浮点
类型，再调用 F.linear。这个源码路径没有显式 INT8 dot/XMX 算子。编译器最终如何
降低该图未实测，不能仅凭仓库标题认定用了 INT8 矩阵计算。量化本身改变一般权重值，
不符合当前固定原模型/权重的迁移目标。界面说明第804行也写了节省显存但速度略降。
torch.compile 可作为局部融合候选，不能假定它保留我们所有舍入边界。

**Aila 的底层内核值得研究，但需分清数据格式与调度。**
[Bnb4BitLinear.cpp:136](https://github.com/Blackwood416/Aila/blob/e00c2adeedbb9f7ef31d5769860393c54f0cb476/src/ops/Bnb4BitLinear.cpp#L136)
有真实 joint_matrix_load/mad/store 调用，NF4 解码为 BF16 后进行矩阵运算。
第270行开始的 wide 版本将装载块扩大到 BK32/64，内部仍按 TK16 顺序调用矩阵
累加，使每次共享局部内存同步对应更多计算。这是值得在 NR 上验证的调度思路。
第431行开始的这一 NF4 GEMM XMX 路由用 A770 名称作为门槛；不能把其 A770
分块和开关直接视为 B580 最优配置，也不能概括成 Aila 的所有 XMX 路径都仅支持 A770。

该路径保持的是自身 NF4/BF16 算法，未证明符合 NR 的 SM89 shared-exponent/
K16 舍入规则。tests/ops/Bnb4BitLinearKernelTests.cpp:101 使用
max(0.06, 0.005*abs(expected)) 容差；生成相同 token 也不代表中间张量或 NR RGB
字节相同。采纳时只能先借鉴装载、分块、缓存和同步方式，保留我们的严格算术。
Linear.cpp:74 缓存形状对应的 oneDNN primitive 和 scratchpad，183行按单 token
GEMV 与多 token GEMM 分流；NR 是图像矩阵负载，不能套用它的 TTS/LLM 加速倍数。
Context.hpp 默认自行创建 SYCL 队列；若参考其算子封装，必须改为我们已验证的
实际 PyTorch 队列/上下文，不能直接混用两个上下文中的 USM 指针。

**AIMDO 上游提供更直接的集成经验。**
[dispatch.cpp:859](https://github.com/xiangyuT/comfy-aimdo-xpu/blob/908ea465c6614896612bf51ede2d1dabe7248d1f/src-xpu/dispatch.cpp#L859)
复用 PyTorch 提供的 SYCL queue，取得原生 Level Zero context/device。
第655–766行按设备、大小和队列管理缓存块，用完成事件保护跨队列复用；第770行
开始查询 SYCL LUID/node mask，供 DXGI 匹配实际适配器。comfy_aimdo/control.py
从当前 Python 的 Library/bin 选择运行库，和我们已经验证的运行库选择方向一致。
VBAR 使用显式 fault()、zeVirtualMemReserve/zePhysicalMemCreate/zeVirtualMemMap；
不能把 README 的“缺页换入”表述扩展为任意 GPU 访问自动触发迁移。

我们 v3 共享缓冲区探针目前按 B580 名称选择 DXGI 适配器，后续生产接口应改成
SYCL/DXGI LUID 一致性验证，并结合 WDDM Budget/CurrentUsage 管理游戏与 NR
共享显存。这有助于设备匹配和资源生命周期，未证明能减少当前主要算术耗时。
不在逐帧路径套用 Python CUDA shim、全局 empty_cache 或模型来回卸载策略。

**与现有性能证据的关系。**
我们此前 INT8 候选的编译中间表示已经包含 DPAS/矩阵调用，但测试输出分组只有
约3.4–6.6%通过无损证明，每个16×16块仍触发修复计算，因此总体慢1.14–1.42倍。
不是只换成 joint_matrix 就能消除这部分工作。可优先研究一次装载供多组K16顺序
运算、减少重新装载/转换、在提交 dot 前避开无收益的分组；是否可行需新候选证明。
尤其不能把多个原本需分别舍入的K16组直接合并成一个累加段。

后续顺序：先完成已经通过64轮的D3D12共享缓冲区桥接的完整NR帧验证；结合上述
LUID和生命周期思路完善接口。算术优化另建候选，保留固定原模型/权重和4060原始
RGB/时序比较门槛，按真实NR形状测热启动总时间。尚未移植或测量Aila内核。

源码保留在 reference/community/allanmeng-review-v1。Aila 根许可证为 MIT；
AIMDO 原生上游为 GPL-3.0。这里只记录来源与设计分析；若以后复制实现，需保留
其相应许可/归属。没有因此引入新的用户审批要求。
