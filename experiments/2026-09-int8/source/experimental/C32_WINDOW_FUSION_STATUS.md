# C32 窗口融合与编译器定位（2026-09-09）

本轮只推进实验。已发布 v0.1.0-pre、精确后端和正式快速路径没有改动；同256输入的完整host性能基线仍是 B580 22.143602ms、4060 3.566807ms。下面的微基准不是完整NR推理成绩。

## 输入与正确性依据

从已认证的 NR256 时序 frame181 输入恢复当前快速模型，捕获实际调用的7个完整 C32SwinBlock。整网输出与原有完整half输出逐字节一致，逻辑dispatch一致，输入和LUT没有变化。捕获23,171,470原始字节，内容寻址压缩文件在D盘复用。CPU审计重读76个唯一数组；不是独立CPU执行模型。

4个encoder模块的shift是(0,0)、(4,4)、(0,4)、(4,0)，3个decoder模块是后3种shift。decoder.3.0.body和post.body复用了参数，但没有调用这个完整block方法，其输入/输出合同不同。capture v1错误地把它们算进调用集合，整网字节校验先通过，末尾覆盖断言失败；v2明确排除这两个不同合同并通过。失败记录保留。

捕获：`D:/Codex-NR-Experiments/nr-b580/reference/experimental/c32-window-blocks-v2/validation.json`，SHA256 `b99326949733fe839431341cc743ea5903abda8f95ef3ca795b23ed0f2ca08cd`。

## 没有采用的整块融合

一个64像素窗口程序尝试完成零扩展/原生像素布局、MLP、FP8边界、QKV、half归一化、Swin、FP8边界、输出投影、原始MLP残差和裁剪写回。整数/half边界与当前FP16快速版保持一致，未改变精确后端。

本机 Triton3.7.2 在 RemoveLayoutConversions 触发 `isIntOrFloat()` 断言。循环展开、两段MLP/QKV与attention/projection融合、显式双逆排列均未绕过。单独的attention/projection尾段也触发同类错误。每次失败保留独立源码、报告、编译IR及租约收据；未改写冻结版本。

进程内编译诊断通过官方inspection hook临时省略三次相关优化调用，变更单独计入缓存键，退出恢复hook，没有改装已安装的Python包。首个完整模块4种配置全部字节一致，但原模块0.169380ms，融合配置0.520750/0.440700/0.540910/0.410970ms。最好的也慢约2.43倍，因此通过本任务GPU租约的STOP机制停止其余编译，并清除自有STOP标记。宽覆盖实验保持passed=false；仅4个已完成比较和首模块计时可引用。

进一步按编译阶段定位：第1次layout pass通过，第2次失败；只省略第2次仍在后续同类pass失败。未把跳过优化当成生产修复。

上游[PR6931](https://github.com/intel/intel-xpu-backend-for-triton/pull/6931)修复了同一类指针位宽检查断言；这是相符的公开线索，不是本机二进制源码回溯证明。原安装的libtriton.pyd和Python文件均与3.7.2的RECORD匹配，不能归因为旧3.3包文件混装。

## 隔离新工具链

使用[PyTorch官方索引](https://download.pytorch.org/whl/nightly/xpu/triton-xpu/)的 `triton_xpu-3.8.0+git1e2d42a0-cp313-cp313-win_amd64.whl`，只解压到D盘实验目录，通过进程内sys.path选择。PyTorch仍为2.13.0+xpu；没有pip升级原环境。索引的download-r2镜像返回403，标准download.pytorch.org同文件可下载，已核对S3提供的SHA256。

轮子SHA256 `c363a2c6e5b0450015a18237fe58d830ab1d80f995947a470917f87ae26b37d8`；247,000,543下载字节，1,264,905,705解压字节。目录 `D:/Codex-NR-Experiments/nr-b580/reference/toolchains/triton-xpu-3.8.0-git1e2d42a0/site`；provision-v1.json记录全部文件哈希，SHA256 `e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07`。新缓存单独在D盘triton-cache-c32-triton38-v1。

3.8默认编译流程成功完成整块融合，不需要诊断hook。首模块两种配置输出与冻结3.7.2捕获完全一致；同次基准原模块0.181950ms，整块融合0.194300/0.197920ms，仍未胜出。不同进程基线波动，不把跨进程数字拼成精确的编译器加速倍数。

## 有希望的较小融合

`fused_c32_attention_projection_v1.py`保留现有MLP和QKV核，仅融合attention、FP8、output projection、half残差和裁剪scatter，消除attended HWC张量及解包步骤。在3.8默认流程中首模块两种配置字节一致；同次基准原模块0.192280ms，4warps0.145530ms，8warps0.193400ms。4warps初筛约24%收益；扩大测试时首模块基线不同，不能把24%当成稳定整网收益。

初筛均为5轮轮换顺序，每轮10次图重放，计时含GPU完成与完整输出拷贝。输出持有缓冲位于共享图池之外，每轮重读完整字节。模型加载、JIT、CPU校验不在计时内。

扩大验证已通过：7个实际模块和3个合成尺寸(8x8、32x48、192x192)，共10个完整输出比较，全部逐字节一致；实际输入覆盖4种shift，192x192跨越旧32768行分块边界。所有5轮重放、外部持有输出、输入与LUT检查通过，CPU审计重新读取完整数组和IR。

7个实际模块各自中位数相加由1.275530ms降到1.100960ms，减少0.174570ms/13.69%；逐模块下降11.04%–16.96%。这是分别计时的和，不是整网计时，也不能把初筛24%当成稳态全模型收益。两版在同一个3.8进程内轮换测量。扩大验证报告 `D:/Codex-NR-Experiments/nr-b580/reference/experimental/c32-tail-triton38-complete-v1/validation.json`，SHA256 `df5640c358b105b4b90abaaa3556463d4ca55a2173c1e2a513a1f5d656a1a508`，saved-audit-v1通过。

尚未创建整网block适配器，也没有把3.8或新tail装进正式GraphFront/ResidualScale路径。下一步必须分别验证：现有整个模型在3.8的字节与时序保持不变、新tail实例适配器加入后的整网字节/逻辑dispatch、独立长时序与状态所有权、同口径完整host速度。只有这些通过并有收益，才考虑接入快速路径。没有宣称对任意输入或全部NaN编码等价。

所有GPU作业已关闭；宽覆盖整块v5是主动停止的负结果，STOP标记已清除。未操作4060；没有新的肉眼审核请求。当前正式速度和发布内容不变。
