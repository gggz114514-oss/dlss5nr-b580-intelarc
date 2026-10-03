# W8A8→FP8→C32 MLP 跨算子链路结果

日期：2026-09-23。范围：原尺寸快速线的单链路捕获与 B580 微基准；未运行 243 帧完整视频，未改产品、冻结源或私有 GPU Block/DIS。

## 结论

选定的真实调用链已通过运行时张量身份验证。候选保留了 W8A8→FP8→C32 MLP 的数值语义，将 INT32 累加、row/column scale 还原、残差相加和 FP16 舍入后直接编码为 E4M3 byte；下一核解码量化输入，并在 load 时完成 `(4,4)` 零填充。外部 `scale_out=1`，指数已编码在 E4M3 byte 中。

数值检查全部通过，但候选两核链未比已有融合 W8A8 快：设备事件中位数 **0.407656 ms**，现有 fused 链为 **0.387656 ms**（候选慢 **0.020000 ms / 5.16%**）；墙钟中位数分别为 **0.712900 ms** 和 **0.689700 ms**。因此不满足事前门槛，不升级完整 243 帧视频，也不建议进入集成评估。

## 生产者、消费者及数值检查

CPU trace 240 的 index 17–19 对应所选 encoder 形状链；index 815–817 是相同形状的重复链。运行时选择到的 producer 权重名是 `encoder.0.0.output_weight`，其调用栈落在 `c32_block.py:39` 的最终投影及 `forward_outputs`，真实输入为 `256×448×32`，W 为 `32×32`，带原 residual。producer 输出立即进入 FP8 量化，随后作为下一 C32 MLP 的输入；消费者的窗口填充形状为 `264×456×32`。D 盘捕获记录中 producer 输出与量化输入的 `data_ptr` 相同且字节相同；量化结果与消费者内部区域逐字节相同，四边 padding 全零。

逐项检查：候选 FP16 producer 输出与原 fused W8A8 相同；候选 E4M3 解码值与已有 FP8 quantizer 相同；候选 C32 MLP 输出同时匹配现有 fused 链和原 harness 捕获输出，均为逐字节相等。114,688 个激活 row scales 与旧量化器相同，32 个权重 column scales 与已打包权重相同；split 与 fused W8A8 的最终输出也逐字节相等。FP16 与 W8A8 结果不同（消费者输出 `max_abs=0.239258`、`RMSE=0.008197`），这是 INT8 权重/激活量化差异。

| 捕获张量端到端路径 | Triton 核数* | 设备事件中位数 | 墙钟中位数 |
|---|---:|---:|---:|
| 旧 split W8A8 + FP8 + pad + C32 MLP | 4 | 0.571771 ms | 0.836100 ms |
| 已有 fused W8A8 + FP8 + pad + C32 MLP | 3 | 0.387656 ms | 0.689700 ms |
| FP16 + FP8 + pad + C32 MLP | 3 | 0.379583 ms | 0.642300 ms |
| 新 W8A8→E4M3 producer + C32 MLP | **2** | 0.407656 ms | 0.712900 ms |

* 计数仅列 Triton 核；前三条的 ATen `pad` 也包含在端到端计时中，候选在消费者内实现同样的零填充。编译不计时：各路径及 debug 校验核均在五次预热阶段编译/预热，之后交错计时 15 次。

## 验证与边界

捕获使用 D 盘原尺寸 harness 数据集的 frame 0（`limit=1`），验证和源 SHA 均通过；量化/消费者身份以实际数组和指针确认。捕获子进程在报告写完后以 `3221226505`（Windows teardown fail-fast）退出；工程原始 `a_insitu` 入口已记载该拆卸期现象。此处只容忍该已知退出码，条件是 harness `passed=true`、capture 报告存在且通过、CPU 运行时链路核验通过。数据流入口自己汇总的 `calls` 为 0，是本捕获 wrapper 未转接汇总计数器；逐调用记录仍显示选中 producer `kernel=_fused`、`packed=true`，设备 profile 也记录 fused launch。计数器没有用于性能/正确性判定。

GPU capture 与微基准分别通过 `gpu_lease_nr.py` 独占租约运行；第二次租约只使用已捕获张量。B580 环境为 Torch `2.13.0+xpu`、Triton `3.8.0`。子进程使用 `vcvars64.bat` 的 PATH/LIB/INCLUDE，`CC=CXX=C:/Program Files (x86)/Intel/oneAPI/compiler/2026.1/bin/icx-cl.exe`；编译缓存和 TEMP/TMP 均在本轮 D 盘 run 下。

原始产物、缓存、日志和逐张量 `.npy` 捕获均保存在 `D:/w8a8-chain-v1/run-20260923-01/`。入口报告、完整验证、两次 lease 日志及其 stdout/stderr 均保留；没有生成视频。

## 代码与证据 SHA-256

新实验代码：

- `PLAN.md`: `a5daea148a1f8e06651acbceae86e9e4a254c5d64ee4c223d889fcdf8f44989a`
- `capture_chain.py` capture-execution version: `4174095bebcb5e52606481338e83c6a7489923c1b466d64d13fb8860609165c7`; final CPU-checker version: `bd6a0c4de79b179f73bd2356dfb894f3053d7a91df4c9dd157c1f2bc215562ad` (only extended trace assertions to cover selected indices 17–19 as well as 815–817; no GPU capture logic changed).
- `chain_kernels.py`: `6238fb93d4fbd85513f808081067d35fccdaf2bb95443b1139319f9cbaf37e6f`
- `benchmark_chain.py`: `122831d5aaa152e90b8fcc9beef48cc127b0c6bcc7e6c75fe915a6421728af60`
- `pipeline.py`: `1d12ce7362ca78f0e0a2906066b01082af821d90d2e435fae3a82b59513a8a8c`
- `launch_lease.py`: `ac1d7ab0b19a5d8ceefce35c00a73b40aba9ee9d644acd385dba88d5afa7b6f3`

主要原始结果：

- `capture/capture.json`: `ecddf99c0e9688befa123de36ffb8173bf3ec647e001dc229afac62555626d69`
- `capture/a_insitu.json`: `da97326374d343c52805c1c8d113e714ec64160bfdb6dbbcb90c996eb8560c1c`
- `capture/run/validation.json`: `cca7a1c2d9c5840e73247ef7b467916f4731904090b34b61f7d04e9c9831f018`
- `result.json`: `543bf47b689e3559e0f2e2ae39d680f4a4f01811be7a4d2c1bed1c74edd4505c`
- trace `D:/w8a8-dataflow-v1/run-20260923-02/full-abba-02/fused-a/a_insitu.json`: `53d9a27844de156afcbd027763c58e21a7ef465f57a1e9c93cdba0d5868043c2`

已检验源码 SHA 记录在 D 盘 `capture/capture.json` 与 `result.json`；其中包括 `w8a8_dataflow_v1/entry.py`、`a_insitu_entry_v1.py`、`fast_matrices_v3.py`、`fused_activation_int8_v1.py`、`c32_block.py`、`pre_mlp.py`、`triton_fp8.py` 和 `fused_c32_mlp_lut_v1.py`。
