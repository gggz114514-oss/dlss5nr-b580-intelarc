# 当前源码与复现边界

新增快照位于[snapshots/2026-09-14](../snapshots/2026-09-14/README.md)。旧顶层backend和fast-kernels保留2026-09-09身份，不静默把新实验替换成旧回归覆盖。

| 新快照内路径 | 职责 |
| --- | --- |
| `nr-b580/backend/nr_backend/` | 当前精确计算与时序/控制实现 |
| `nr-b580-int8/product/branched_exact_v1/`及`exact_*` | 本次精确优化基线：融合、布局、前后处理、图执行 |
| `nr-b580-int8/product/qkv_exact_v1/` | QKV候选源码；历史256²验收，不冒充本次大尺寸速度基线 |
| `nr-b580-int8/backend/nr_backend/` | 快速分支使用的后端依赖，须与精确分支隔离进程导入 |
| `nr-b580-int8/experimental/nr256_product_stack_v1.py`及依赖 | 固定快速算术与内核组合 |
| `nr-b580-int8/experimental/face480_residual_scale_v1.py`、`residual_scale_v1.py` | **画面变化很大的NR256缩放/残差实验路线** |
| `nr-b580/reference/review_fast_fullsize_v1.py` | 本次原尺寸适配，复用量化参数、按行批处理FFN、完整注意力 |
| `nr-b580/reference/diagnose_fourway_geometry_v1.py`、`diagnose_residual_oracle_v1.py` | 精确NR256和六帧残差归因 |
| `nr-b580/reference/collect_fast_fullsize_v1.py`、`run_fast_fullsize_review_v2.py` | 规格收集、14进程编译、验收、视频流水线 |
| `nr-b580/reference/benchmark_fullsize_optimized_pair_v2.py` | **本次正确测速入口**，包装v1但替换为branched精确优化基线 |

这些是可审计的研究源码快照，不是脱离资产即可启动的成品。已包含静态导入可解析的本地模块依赖；部分实验脚本仍采用原研究目录布局、D盘数据目录、FFmpeg和工具链路径，运行前需要逐项配置。不要把`benchmark_fullsize_pair_v1.py`单独作为最新基准，它保留的是被纠正的旧精确选择。

## 最小公开检查

```text
python tools/verify_release.py
python tools/verify_milestone.py
python tools/fma_witness.py
```

这些检查文件哈希、Python语法、公开证据统计和独立FMA案例，不加载权重、不运行完整模型。

## 完整GPU实验还需要什么

- 原固定模型资产及噪声、sigmoid、倒数/控制表；参见[资产说明](06-reproduction.md)。
- 与当前INT8打包合同对应的冻结量化profile。源码公开重打包规则，但profile包含派生模型常量，不随本次公开快照附带；不要用随意重新校准的profile声称重现本次结果。
- 已认证的共同RGB/运动输入、对应精确/快速输出，以及目录中的validation索引。诊断脚本有意要求哈希相符，原私有视频、张量和实验数据库未分发。
- Windows B580的匹配PyTorch XPU、Intel Triton运行环境；编译缓存必须在匹配环境准备。本次流水线使用14个独立编译worker，不能以设置OMP线程数替代。

为第三方实现通用的资产配置、视频读入和错误提示是后续产品工作。本次没有发布或修改ComfyUI节点安装包，也未发布私有GPU Block/DIS、驱动、NVIDIA DLL或模型权重。

## 源码身份

[新快照源码清单](../evidence/source-manifest-2026-09-14.json)记录原项目相对路径、原SHA和公开SHA；若为隐去本机账户路径而变更，会显式标出，不能当作原字节。编译和性能验收对应原研究源码。原报告摘录与公开文件的哈希各自独立。
