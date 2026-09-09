# 复现与代码导航

本预览包含能阅读、导入和检查的实际项目源码，不包含完整模型运行所需的第三方资产。请先区分以下三种复现层级。

## A. 不需要模型或显卡的检查

```text
python tools/verify_release.py
python tools/fma_witness.py
python tools/compare_bytes.py reference.bin candidate.bin
```

第一个检查发布清单与源码文件身份；第二个用标准库和独立整数舍入展示half FMA中点反例；第三个比较完整文件字节。它们不是完整NR测试。

安装或使用已有NumPy/PyTorch后，可以运行不加载模型权重的标量验证：

```text
python tools/check_half_fma.py --device cpu
python tools/check_half_fma.py --device xpu
```

默认CPU，共24007组固定样例；XPU版本需要Intel兼容的PyTorch。工具没有自动安装依赖、占用外部参考机或修改原始数据。

## B. 运行已发布精确后端

本地实测环境为Windows、Intel Arc B580、PyTorch 2.13.0+xpu及Intel Triton/oneAPI工具链。详细版本快照见[evidence/environment.json](../evidence/environment.json)。这记录已测试环境，不承诺其他wheel或驱动组合可用，也不提供一键升级脚本。

需要自行准备如下资产目录：

```text
assets/
  sf-v2/WEIGHTS_HT.bin
  noise-sm89-v2/manifest.json
  noise-sm89-v2/radius.f32.bin
  noise-sm89-v2/sin.f32.bin
  noise-sm89-v2/cos.f32.bin
  sigmoid-sm89-v1/sigmoid.f32.bin
  reciprocal-sm89-v1/reciprocal.f32.bin
  reciprocal-dimensions-sm89-v1/reciprocal.f32.bin
```

reset路径只需要权重和噪声表；有运动的时序路径还需要其余标量表。加载器检查域、大小和固定哈希，不会接受任意同名文件。表的意义与范围见[数值规则](03-numerics-and-layouts.md)。当前预览没有把提取/枚举这些资产打包为完整自动化流程；生成器源码供研究参考，不把缺资产的clone称为即装即用。

准备与原生输入合同一致的浮点HWC3 NPY后：

```text
python tools/run_exact.py --assets /path/to/assets --input input.npy --output new-run --device xpu --arithmetic triton
```

这个新增公开CLI只负责调用和保存，不是历史验证宿主。发布时完成参数/语法检查；完整模型证据属于所附原样后端及历史验证链，而不是未经宣称的“新CLI全模型回归”。冷启动含JIT，输出目录必须不存在，不自动缩放或转换色彩。

研究者直接使用时序接口：

```python
import sys
sys.path.insert(0, "backend")
from nr_backend import MotionNR, use_arithmetic_backend

model = MotionNR.from_assets(
    "assets/sf-v2/WEIGHTS_HT.bin",
    "assets/noise-sm89-v2",
    "assets/sigmoid-sm89-v1",
).to("xpu").eval()

# rgb: HWC3浮点；motion: HWC2，当前到上一帧像素位移；均已在xpu。
with use_arithmetic_backend("triton"):
    first = model(rgb, motion, reset=True)
    second = model(next_rgb, next_motion, reset=False)
```

`MotionNR`属于默认SDR/零深度等已记录合同。风格/UI/遮罩在其他类中，不能假设这个示例已涵盖所有控制。调用后的`next_seed`及私有历史由会话管理。

## C. 重做原生恢复实验

需要相同参考DLL、匹配权重、4060环境、Windows D3D12宿主和CUDA开发工具。过程按[参考方法](02-reference-method.md)执行：先证明原始宿主与捕获一致，再证明未修改CUBIN的独立重放一致，最后使用研究副本插桩。

`reference/native-replay/*.cpp`是实际研究程序源码，通常接收输入目录和新输出目录。每个程序的`Usage`与`read_file`列出所需文件。它们需要CUDA Driver API及Windows开发环境；源码不等于已经在新机器上编译并运行。

`reference/native-trace/nr_nvapi_trace.cpp`依赖Windows/D3D12及外部MinHook，须接入自有宿主；公开的这一版本是早期小尺寸诊断快照，存在捕获预算上限，不能直接用于全部大尺寸实验。

原始捕获数据和私有素材未公开，因此第三方仅靠这个仓库不能独立验证每一份历史哈希。可以复用方法和源码，在自己的合法资产、输入与环境上建立新的证据链。记录运行库/源码/工具/输入/输出哈希、重放等价性、完整张量差异及状态检查。

## 快速内核快照

`snapshots/fast-kernels/`保留最新NR256实验实际使用模块的本地依赖闭包。它们通过上下文适配器接入精确模型，其中仍有研究目录配置、标量表认证和常量注册要求。此预览提供源代码身份与结果，不把它们包装成稳定公共API。

尤其`cubic_lut_constant_v1.py`仍保留实验表及报告的固定哈希；不要删除校验后声称复现了原实验。快速FP16与INT8、精确模式及低分辨率残差合成是分开的配置，不应混报性能或误差。
