# 整数分组与实验数据共享（2026-09-09）

主后端仍为35文件，检查点736次优化比较；本轮没有合入候选，没有新增正式完整帧比较。完整迁移与生产接口仍未完成。

普通循环的每组4项/8项共24个算子检查全部逐字节一致，但6项成对测速全部慢于现版。展开8项循环在第一组小矩阵出现365字节差异，保留失败样本；原因未确定，不能归因于已证实的编译器错误。

## 成对真实算子测速

| 捕获与分组 | 现版 ms | 候选 ms | 现版/候选 |
|---|---:|---:|---:|
| actual-0-4x32w1-g4 | 0.398313 | 0.769867 | 0.5174 |
| actual-0-4x32w1-g8 | 0.431250 | 1.018630 | 0.4234 |
| actual-1-4x32w1-g4 | 0.285638 | 0.581992 | 0.4908 |
| actual-1-4x32w1-g8 | 0.328502 | 0.745395 | 0.4407 |
| actual-2-4x32w1-g4 | 0.434223 | 0.778585 | 0.5577 |
| actual-2-4x32w1-g8 | 0.426463 | 1.020580 | 0.4179 |

同一轮交替/逆序8轮，每轮20次温态调用，包含分配与分派，排除上传/JIT/读写/比较。候选仍用INT32 SIMD；没有使用INT8 XMX。

## 资源证据

v1整数乘积候选的 n_spills 为14784/14784/14912；v3普通4项循环为704/704/896，普通8项循环为0。字段按运行库原样记录，不推断单位；n_regs=0 不代表物理寄存器使用为零。8项没有报告溢出却仍较慢，说明减少溢出不足以让这一写法超过现版。额外读取、归约、循环和整数对齐成本仍须分析；没有硬件周期归因。

## 保存方式

新实验使用 immutable_artifacts_v1.py：按完整文件SHA-256命名，相同数组/IR引用同一文件；已有内容不一致时拒绝覆盖。原始数组全部保留，没有用哈希替代原始字节。
v3逻辑输出与IR共72242099字节，实际唯一文件21917625字节；独立审计已重新读取全部报告引用并核对。v2失败与v3成功均保留，连同新缓存、本轮报告累计约32.11 MiB（记录时）。
未删除任何旧实验；用户上一轮询问的是清理建议。11.89GiB候选清单仍在 STORAGE_CLEANUP_CANDIDATES.md。

## 对“把整个模型改成INT”的判断

以整数精确模拟FP8/半精度语义，与重新量化成通常的INT8矩阵图，是两项不同工作。前者仍须处理逐项指数对齐、截断、K16舍入；整数表示本身不保证能利用XMX。后者通过权重/激活尺度、图中连续低精度区域和算子融合有可能提升吞吐，但可能改变输出；不能作为已经满足当前4060字节一致目标的结果。
本轮不是整模型INT8实验。OpenVINO文档仅用于解释一般量化/混合精度原理，不证明NR已可直接转换或可以获得特定倍数加速。
- https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/precision-control.html
- https://docs.openvino.ai/2024/documentation/openvino-ir-format/intermediate-representation-int8-inference.html

## 证据

- `D:\Codex-NR-Experiments\nr-b580\reference\experimental\integer-products-v3\milestone-v1.json`
- `D:\Codex-NR-Experiments\nr-b580\reference\experimental\integer-products-v2\saved-audit.json`
- `D:\Codex-NR-Experiments\nr-b580\reference\experimental\integer-products-v3\saved-audit.json`
- GPU租约92041终止exit1；90301终止exit0。无活动GPU任务，参考笔记本未使用。
- 下一步优先研究减少跨算子解码/预打包开销及符合XMX的精确矩阵表示；不能把不修正的DPAS或重数量化近似当成迁移完成。
