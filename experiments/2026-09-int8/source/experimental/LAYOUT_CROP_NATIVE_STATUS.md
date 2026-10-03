# 布局与依赖裁剪组合：完整NR256减少约4%耗时（2026-09-10）

用户通知完工后，主助手核对Luna终态交接、result/log/lease摘要、483项源码及
269项exact_gate文件哈希。returncode=0；已执行代码冻结，尚未晋升默认后端。

基线是已选用的ShortFP8 Stackv3；候选组合C512窗口消费者、ViT head布局和
post依赖裁剪。三项仅在GPU body构图时安装，正常重放不切这些作用域。

| 完整调用模式 | 基线均值 | 组合均值 | 耗时减少 |
| --- | --- | --- | --- |
| reset | 11.517095ms | 11.077679ms | 0.439416ms / 3.81533% |
| temporal | 11.745086ms | 11.269206ms | 0.475880ms / 4.05174% |

每种模式三轮、每轮每侧60次，交替运行；六组轮均值均更快。没有把三个静态
筛选收益相加。输入是驻留GPU的NR256固定RGB与零运动；包含完整模型调用、
历史提交和GPU完成等待，排除解码、光流估算、上传、显示、JIT及游戏争用。
不是480p基准，也还不是1080p残差或390帧视频结果。

720次完整输出全部匹配已有快速分支冻结哈希，并两侧逐字节一致；reset复现、
私有history/seed、修改caller输出不污染history、held输出、输入与LUT不变、
共享图池外持久IO及绑定恢复检查通过。没有改变精确分支或重新声明NVIDIA精确性。

候选6次body构建，C51296次（16块各6）、ViT48次（8块各6）、post6次。
Triton683→651、独立FP8196→180；候选4个C512投影特化、1个ViT投影特化与
58项ShortFP8资源均零spill。post少一个逻辑C32分块，因此每种模式的逻辑计数
均减少fp8=3、dense=3、cubic_fp8=1、attention_normalize_c32=2；不是漏算输出。

结果：D:/Codex-NR-Experiments/nr-b580/reference/results/layout-crop-native-parity-v1/validation.json
SHA256：e54f31291b1b628d2d988bf19bbc8fe43cdb8f0c19d80def21ec7f2418973109。
Luna交接：DREF/experimental/layout-crop-native-parity-v1-monitor-luna-v1。

下一步验证1080p残差完整调用、运动/状态/进度回退与390帧已审核序列；这些通过后
才决定纳入快速分支。4%的收益尚不足以接近4060；保留用户后续FP8规则/部分INT8，
再连续段重新量化的路线，不能将本次结果包装成架构突破。
