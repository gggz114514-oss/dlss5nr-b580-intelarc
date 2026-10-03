# 作者资料的实测跟进与当前本体热点

2026-09-10。作者博客和五个相关项目已在 [初次源码评估](BLACKWOOD_XPU_REVIEW.md) 中固定来源。本次把此前的本地实验结论与新的当前版本分段测量接起来。当前运行时仍为 fee6d0d 的 WindowBlocks v3 + c5ac61b FP8 栈，完整迁移与 4060 性能持平目标尚未完成。

## 资料已经带来的具体帮助

- [博客 GEMM 长文](https://github.com/Blackwood416/Blackwood416.github.io/blob/6ee90e05bbd33d028b210cfb9ac5b6d1d0de3f6d/src/content/blog/xpu-trick-01.md) 与 [A770 调优片段](https://github.com/Blackwood416/A770-Kernel-Optimization/blob/b05a20a67660a57b1a49658fd0d604f1d30c5f57/references/api/code-snippets.md) 提供按 DPAS 操作数排列、减少寄存器拼接和循环寻址的参考。其 A770/BF16 预打包条件不能直接作为 B580/NR 收益。
- [Aila](https://github.com/Blackwood416/Aila) 和 [ComfyUI-OmniXPU](https://github.com/Blackwood416/ComfyUI-OmniXPU) 的常量缓存、执行生命周期与按形状分派值得参考；[InferRef](https://github.com/Blackwood416/InferRef) 的分层用例和首次差异定位更贴近将来发行复刻过程的需求。它们尚未被整体安装或接入当前工具。
- [omni-xpu-kernel](https://github.com/Blackwood416/omni-xpu-kernel) 的 INT8 和 ESIMD 提供实现参考；现成注意力的尺寸、偏置与数学边界并不匹配 NR。详情与代码定位见初次评估。

布局思路已在我们的本地代码中验证，详见 [单矩阵实验](ESIMD_DENSE_STATUS.md) 与 [完整 ViT 实验](ESIMD_VIT_STATUS.md)。部分深矩阵计入动态输入打包后更快；随后生产者直接写私有布局，八个 ViT 块接入完整 body。v2 同次配对 NR256 连续调用 12.411984→12.307339 ms（−0.843%），但 1080p 残差完整链路 12.974752→13.045585 ms（+0.546%），因此没有选用。输出匹配的是已审核快速分支，不能据此称其等同 NVIDIA 精确分支。

这些结果表明，单矩阵和寄存器资源报告不足以决定整帧方案。残差回退的具体缓存或调度原因仍未测定。

## 新的当前版本诊断

先做 `probe_xpu_graph_events_v1.py`：图外计时标记有效，但捕获到图内的标记在重放后查询报错，运行库明确表示 recording-state submission 返回的事件没有 profiling 信息。普通图的三组改变输入重放仍正确，进程正常退出。这条方法不能用于当前图内阶段计时，没有启用此前不稳定的 XPU profiler。

随后运行 `profile_selected_body_stages_v1.py`，采用真实连续帧的冻结 body 输入，完整保留当前 WindowBlocks v3、ViT64 和 FP8 消除。分段输入按完整存储复制，保留别名、stride、offset；FP8 证明来自完整计算的数据来源记录。每段热身和捕获均检查 Triton 内核顺序、形状、启动参数、量化调用及消除次数。整段和各段公开结果位于共享临时池之外。

完整 body 与分段拼接的输出均匹配冻结快速参考；683 次 Triton 调用、427 次逻辑量化、231 次消除保持。19 个完整阶段数组已保存用于独立 CPU 对照。计时为七轮轮换顺序，每样本十次静态图重放后等待完成，校验和编译不在计时内。

| 独立重放范围 | 中位耗时 |
| --- | ---: |
| 完整 body | 10.258780 ms |
| pre | 0.845090 ms |
| encoder C32 | 0.691090 ms |
| encoder C64 | 0.485020 ms |
| encoder C128 | 0.550230 ms |
| encoder C256 | 0.664400 ms |
| encoder C512 | 1.002840 ms |
| 8 个 ViT 块 | 1.381710 ms |
| decoder C512，含输入合并 | 1.020070 ms |
| decoder C256 | 0.672330 ms |
| decoder C128 | 0.538620 ms |
| decoder C64 | 0.493900 ms |
| decoder C32 | 0.681930 ms |
| post | 1.176830 ms |

分段重放改变缓存和调度边界，包含各段独立公开输出复制，**不能相加成整帧耗时，也不能用来扣算外围开销**。完整 body 也不包含动态 front、历史 warp、模型校验、历史提交、缩放合成、运动估算、上传和显示。10.258780 ms 不是新的完整 NR 延迟或优化收益。

较大阶段分散在 ViT、C512 编解码和 pre/post，继续只替换八次扩展矩阵的收益范围有限。下一步先细分这些阶段内部的精确 K8 点积、C32 注意力、RGB 时序混合和布局转换，再决定更大范围的生产者/消费者融合。当前数据尚不能把 pre/post 的全部耗时归因于 K8。

## 冻结证据

全部实验数据在 `D:/Codex-NR-Experiments/nr-b580/reference/experimental`，没有新视频或远程作业。

- `xpu-graph-events-v1/validation.json`：`7aa167f727f8b10ceb3727daf1f5f3c07c3262c611b038f94d02177ba94360bb`
- `selected-body-stages-v1/validation.json`：`f2553253c77b49e535c50f176afec1246abf5ee44784643bde1d25a820661e66`
- `selected-body-stages-v1/physical-sequence.json`：完整 Triton 发射描述，hash 记入上述报告。
- `audit_selected_body_stages_v1.py`：检查作业退出、来源、数组、顺序与计时统计；输出到 `selected-body-stages-checkpoint-v1/saved-audit-v1.json`。该 CPU 审计不重新执行模型。

执行过的脚本和结果保留原字节；改进另建版本。此次没有选择新的运行时，也没有发布 GitHub。
