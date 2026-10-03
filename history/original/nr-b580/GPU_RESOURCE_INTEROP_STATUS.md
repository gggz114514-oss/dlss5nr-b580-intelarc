# D3D12 与 NR 计算端共享资源（2026-09-08）

以下完整模型共享资源证据对应已存档的完整INT32版本。当前主后端已合入注意力求和/权重融合、C32归一化、注意力指数融合、4×32单warp的K16分块、FP8转换及cubic激活融合；
检查点保留这26次测试的原源码归属，不将其计作当前注意力融合版本的图形资源回归。

本机B580已通过共享D3D12缓冲区执行完整NR：864×480和1920×1080各13帧，
共26帧。D3D12读回的RGB16F转换为RGB32F后，与固定4060参考的388,177,920个
RGB字节完全一致；私有RGB16F历史也一致。独立CPU审计重新读取了全部保存结果。
这已超出简单算术探针，仍未接入游戏纹理、OptiScaler或现有视频工具的生产接口。

| 路径 | 当前证据 |
| --- | --- |
| D3D11纹理 → OpenCL → D3D11 | RGBA32F、RGBA16F各4轮变化内容通过 |
| D3D12 → D3D11共享纹理 → OpenCL → D3D12 | 由D3D11创建NT共享纹理，再由D3D12打开；两种格式各4轮通过，D3D12负责输入和读回 |
| D3D12共享缓冲区 → PyTorch/Triton张量 → D3D12 | FP32、FP16各32轮，4096元素/轮；共786432字节独立读回复核通过 |
| D3D12共享RGB/MV → 完整NR → 共享输出 → D3D12读回 | 480p和1080p各12张真实相邻图像加一次首帧重置，26帧RGB/历史全部字节一致 |

完整模型路径使用三个持续复用的缓冲区：HWC RGB32F输入、HWC2 MV32F输入和
HWC RGB16F输出。每个缓冲区在实际PyTorch SYCL队列中导入，经DLPack直接暴露
同一指针；模型返回值通过GPU copy_写入输出缓冲区。D3D12读回输出之前没有插入
torch.xpu.synchronize或Tensor.cpu；模型内部输入合法性检查仍可能发生CPU同步，
所以不能据此声称整个调用都无主机等待。

输入图像和运动数据由CPU上传作为本次测试种子，输出由CPU读回验收；模型输入
没有先经torch.from_numpy(...).to('xpu')复制到另一份张量。每轮也额外读回输入，
验证模型未改写它们。输出缓冲区每次先覆盖为-7，然后写入完整模型结果；私有历史
只来自B580自身。显式重置、模型返回值被调用方清零、共享输出再次覆盖后，历史
仍正确。成功关闭所有导出和资源；基本顺序/存活导出保护检查通过，尚非完整故障恢复。

SYCL提供的LUID用于EnumAdapterByLuid创建D3D12设备，随后再次核对设备LUID。
本机测试三份资源均为0000000000010d3c，单节点mask=1。该诊断标识不是跨机器常量。
生产接口须使用调用方的设备/队列并核对身份，不能继续自行创建这些测试用队列。

| 尺寸 | 本次首帧秒数 | 后12帧平均秒数 |
| --- | --- | --- |
| 864×480 | 4.413 | 2.677 |
| 1920×1080 | 12.652 | 11.011 |

计时从本轮输入提交到D3D12输出读回，包含测试上传/读回与可能的JIT，后续输入读回
和历史审计不计入。这次已存在持久编译缓存；首帧数值不是清空缓存后的冷启动基准。
没有性能提升或实时吞吐声明，也不能把它与先前仅模型+同步的时间直接相减当桥接开销。
PyTorch峰值计数不包含所有D3D12/驱动分配；不作为总显存占用。

第三条早期算术探针不经过OpenCL：在实际PyTorch当前SYCL队列/上下文中导入D3D12 committed
buffer，用DLPack暴露相同指针。PyTorch执行两次原地运算，Triton再执行一次原地运算，
随后D3D12读回。每轮输入模式不同，输出由独立CPU公式核对。CPU上传和读回用于
初始化与验收，计算端交接没有通过CPU数组重建张量。未测量驱动内部是否产生复制。

最终采用三个时间线：D3D12生产者栅栏、SYCL生产者栅栏，以及只用于CPU验收的D3D12
栅栏。PyTorch队列为in-order；GPU外部等待和发信号排序，交接点不等待SYCL事件。
导出张量仍存活时拒绝销毁资源。每轮仍有最后的CPU验收等待，因此不是生产流水线
延迟/吞吐基准。多帧并发、取消、异常销毁仍需验证。

v1由多个队列交替发信号到同一栅栏，在读回时超时。v2等待SYCL事件后，有3轮FP32
字节一致，但随后原生栅栏值未按预期推进。v3拆分时间线后，两种精度64轮异步交接
通过。这支持采用单一生产者时间线，未证明驱动内部的具体故障原因。失败证据均保留。

运行环境差异：独立oneAPI 2026.1探针只枚举到OpenCL，报告SYCL导入能力不可用。
NR实际使用Python环境内的SYCL 2026.0，队列为Level Zero；bindless image、外部显存
和外部栅栏能力均为真。已记录实际DLL路径及哈希。Level Zero内存导入位掩码196支持
opaque Win32、D3D12 heap和D3D12 resource；图像导入212还包含D3D11 texture。
能力查询之后，已实际完成上述线性缓冲区导入。

D3D12自行创建的首版纹理在OpenSharedResource1处返回E_INVALIDARG；换成D3D11
创建共享纹理后通过。不代表任意现成游戏纹理都可直接导入。微软说明了
[跨运行库资源描述要求](https://microsoft.github.io/DirectX-Specs/d3d/ResourceHeaps.html#sharing-with-other-runtimes)，
以及[共享栅栏接口](https://learn.microsoft.com/en-us/windows/win32/api/d3d11_4/nf-d3d11_4-id3d11device5-opensharedfence)。
OpenCL部分遵循[Khronos的资源获取/释放规则](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clEnqueueAcquireD3D11ObjectsKHR.html)。

证据位于reference/integration：d3d12-opencl-probe-v1/v2、torch-queue-probe-v1，
以及torch-d3d12-buffer-v1/v2/v3（v3为当前通过版本）。resource-interop-audit-v1.json
重新读回64轮输出并按独立公式复核；这些探针给完整NR帧比较数增加0帧。

完整模型新证据：reference/integration/torch-d3d12-nr-v1（480p）及v2/results/1920x1080。
shared-nr-864x480-full-audit-v1.json和shared-nr-1920x1080-full-audit-v2.json重新读回
保存的RGB32F、共享输出RGB16F及私有历史；run.log的两个lease均返回0。
backend-checkpoint.json把26次共享资源完整模型比较单独记录，复用已有真实视频
输入，不扩写成26张新增独立输入，也不改写此前372参考/260算术融合比较的范围。

下一步实现纹理与线性张量的GPU布局转换、调用方资源状态和队列交接、多帧资源池、
异步取消/失败恢复及OptiScaler适配。guide分辨率/子矩形、HDR、非默认控制交叉
组合和性能也仍需验证。尚不能宣称插件可用。
