# 第二层「拆同步」—— RESULT

> 目录：`nr-b580/reference/syncfast_v1/` ｜ 数据：`D:/syncfast-v1/r1/`（8 轮）
> 裁决产物：`D:/verdict-v1/syncfast-{metrics,video}/`
> 日期：2026-09-17 ｜ 机器：Intel Arc B580 ｜ 素材：480×864 × 243 帧

---

## 1. 一句话

**拆掉 16 次/帧的 `torch.xpu.synchronize`，帧墙从 82.72 ms 降到 75.86 ms（省 6.867 ms，1.0905×），
8 轮 × 243 帧逐字节零差异。**
其中 13 个 stage fence 贡献 3.908 ms，另外 2 个周期性 fence **单独拆是零收益甚至负收益**，
但和前者一起拆时贡献 2.96 ms —— **串联的 fence 不可逐个拆，收益不叠加。**

---

## 2. 最终数据（8 轮 ABBA，帧 4..N 的中位帧墙）

| 变体 | 中位帧墙均值 | 两次离散度 | 加速比 | 省 | 字节门 |
|---|---:|---:|---:|---:|---:|
| `off` 对照 | 82.7235 ms | **0.0080%** | 1.0000× | — | 243/243 |
| `mark`（13 stage fence） | 78.8155 ms | 0.1293% | 1.0496× | **3.908 ms** | 243/243 |
| `c32`（2 个周期性 fence） | 83.0870 ms | 0.7957% | 0.9956× | **−0.363 ms** | 243/243 |
| **`all`（两处一起）** | **75.8562 ms** | 0.6380% | **1.0905×** | **6.867 ms** | 243/243 |

逐轮：

```
idx  tag        median ms       min ms   frames    byte-eq
0    off          82.7202      77.8323      243        243
1    mark         78.7645      76.2880      243        243
2    c32          82.7564      78.4762      243        243
3    all          75.6142      71.7629      243        243
4    all          76.0982      72.2008      243        243
5    c32          83.4175      78.3858      243        243
6    mark         78.8664      74.0413      243        243
7    off          82.7268      78.6762      243        243
```

**`off` 两次跑的离散度只有 0.0080%（6.6 微秒）** —— 这是本项目至今最稳的对照臂，
远低于 2.8% 的噪声地板。所有变体的离散度都在噪声地板之内。

**与 `FRAME_BUDGET_REPORT.md` 的 K1/K2 消融对照（报告是 32 帧）：**

| | 本轮（243 帧 × 2） | 报告（32 帧） | 差 |
|---|---:|---:|---:|
| K1 `mark` | **3.908 ms** | 4.1472 ms | 0.24 ms |
| K2 `all` | **6.867 ms** | 7.2885 ms | 0.42 ms |

**一致。** 本轮的贡献是用 243 帧 × 8 轮 ABBA + 每帧逐字节门把报告的单次消融**复核**了一遍。

**抑制计数（全部轮次相加，逐轮逐位一致）：**

| 变体 | `skipped` | 明细 | `kept` | 明细 |
|---|---:|---|---:|---|
| `all` | 7290 | `executor.mark` 6318（13/帧）+ `c32.projected` 972（2/帧） | 980 | 计时窗口边界 2/帧 |
| `mark` | 6318 | `executor.mark` 6318 | 1952 | `c32` 2/帧 + 边界 2/帧 |
| `c32` | 972 | `c32.projected` 972 | 7298 | `mark` 13/帧 + 边界 2/帧 |

**`kept` 精确等于"不该拆的次数"** —— 计时窗口边界（`materials_entry_v1.py:240/246`）与会话收尾
（`graph_front_v1.py:96`、`graph_history_warp_v1.py:97` 的 `close()`）**一次都没被误伤**。
抑制器是"按调用点精确打击"，不是"全局关掉同步"。

---

## 3. 诊断

### 3.1 同步点人口普查（`syncfast_probe_v1.py`，6 帧 106 次）

| 调用点 | calls | total_ms | **mean_us** | median_us | 性质 |
|---|---:|---:|---:|---:|---|
| `nr_backend/executor.py:72`（`mark()`） | 78 | 44.281 | **567.7** | 66.0 | 13/帧，stage 边界 |
| `fused_c32_projection_native_half_v1.py:99` | 12 | 26.624 | **2218.7** | 2167.2 | 2/帧，周期性 |
| `materials_entry_v1.py:240` | 6 | 0.448 | 74.7 | 74.4 | 1/帧，计时窗口起点 |
| `materials_entry_v1.py:246` | 6 | 0.247 | 41.2 | 41.5 | 1/帧，计时窗口终点 |
| `graph_front_v1.py:96` | 3 | 0.316 | 105.4 | 68.9 | 会话收尾 `close()` |
| `graph_history_warp_v1.py:97` | 1 | 0.089 | 89.1 | 89.1 | 会话收尾 `close()` |

仪器自身成本 0.157 ms，不计入。

**⚠️ 后三个调用点不在逐帧路径上。** `close()` 是会话收尾，`240/246` 是计时窗口边界
（**必须保留**，否则计时窗口失效）。**别把"普查里出现过的行"都当成逐帧开销。**

### 3.2 统计量选错：偏斜分布不能用中位数（已修）

**我先前报的"13 个 stage fence 每次只等 66 微秒"是错的**，正确值 **567.7 µs/次**。
错因：`report()` 用了 `statistics.median`，而这 13 个点的分布**极度偏斜**——
4 个点（`pre` / `RGB` / `decoder C32` / `encoder C32`）扛住几乎全部设备等待 12.65 ms
= 净设备等待的 92.4%，其余 9 个 `|wait| ≤ 0.18 ms`。
**中位数落在那一大堆"0"里，把整体低估 8.6 倍。**

对照：`fused_c32_projection_native_half_v1.py:99` 的 mean 2218.7 / median 2167.2 —— **不偏斜**。

**修正**：判定列改为 `mean_us`（= `total_ms / calls`），`median_us` 保留但降级为偏斜指示，
表尾加注。**判别规则：先看 `max/mean`，比值 >> 1 就直接用均值。**

### 3.3 两处同步的真实语义（读源码得出）

**① `executor.py:71` 的 `mark()`：**
```python
def mark(name):
    if rgb.device.type == 'xpu': torch.xpu.synchronize(rgb.device)   # ← 先无条件同步
    if progress is not None: progress(name)                          # ← 再判断
```
13 个调用点（`pre` / encoder C32·C64·C128·C256 / C512 / ViT / decoder C512 / C256·C128·C64·C32 / RGB）。

**没有进度回调时同步照样做。** 核实：`Session.process(rgb, motion, *, reset=False)` **没有 `progress` 参数**；
且 `materials_entry_v1.py:156` 走 `stack = session._stack`、直接调 `stack.model(...)`，
**绕过 `Session.process`** ⇒ `progress` 恒为 `None`。

**② `fused_c32_projection_native_half_v1.py:98`：**
```python
if attention.C32AttentionFront.forward is not body.c32_front:
    for _ in range(chunks // 8): torch.xpu.synchronize(features.device)
```
两条注释说明动机：`c32_chunk_layout_v2.py` 写 *"capture_body.c32_front removes only explicit
periodic synchronization. **Keep that exact scheduling distinction** for ordinary progress/eager calls."*；
`capture_body_v1.forward_front` 写 *"**Progress/fences belong outside capture**"*
⇒ **同一条 dataflow 已在捕获路径上零 fence 跑通。**

### 3.4 为什么摘掉同步不可能改字节

1. `synchronize` 是**纯等待**，不写任何数据；
2. eager 路径**单流**（无自定义 `torch.xpu.Stream`；`graph.py:22` 那个是捕获路径的）；
3. 无 `from torch.xpu import synchronize` ⇒ 属性补丁可达所有调用点；
4. 副作用只有两个且可接受：异步错误推迟到下一个同步点才抛（仍会抛）、
   分段计时粒度变粗（`progress` 本来就没传进来）。

**实测证实：8 轮 × 243 帧逐字节相同。**

---

## 4. 实现（`syncfast_cuts_v1.py`）

**按调用点精确抑制，不做全局禁用**（全局关会连累计时窗口边界）。

做法：看**直接调用者**的 `(co_filename 后缀, co_name)`，命中白名单才跳过，其余一律照原样调用。

```python
TARGETS = (
    ('nr_backend/executor.py', 'mark', 'executor.mark', True),
    ('fused_c32_projection_native_half_v1.py', 'c32', 'c32.projected', False),
    ('fused_c32_projection_pack_v1.py', 'c32', 'c32.packed', False),
    ('c32_chunk_layout_v2.py', 'prepare', 'c32.chunk_layout', False),
)
```

`mark` 那一处**额外要求 `progress is None`**（防御性：万一将来有人传 progress，就按原样同步）。

**自检确认可行**：`f_locals` 对闭包会带出自由变量 ——
`locals_keys = ["name", "progress", "torch"]`、`has_progress = true`。
拿不到就按"可能有人传了"处理（宁多同步，不可少同步）。

开关：`SYNCFAST_CUTS=off|mark|c32|all`。`off` = 对照臂，走**同一入口、同一序幕、同一套断言**。

---

## 5. ⚠️ 本轮最重要的发现：串联的 fence 不可逐个拆

| 拆法 | 相对对照 | 预算（报告 §4） |
|---|---:|---:|
| 只拆 13 个 stage fence（`mark`） | **−3.908 ms** | 4.1472 ms（K1 实测） |
| 只拆 c32 那 2 个（`c32`） | **+0.363 ms（噪声内，≈ 0）** | 5.1283 ms |
| 两处一起拆（`all`） | **−6.867 ms** | 7.2885 ms（K2 实测） |

**`c32` 有 5.13 ms 的预算，单独拆一分钱都拿不回来（甚至是 −0.363 ms）；但和 `mark` 一起拆时它贡献 2.96 ms。**

**原因：多个 fence 串联时，只有最上游那个"真正堵住"的拆了才有用，后面的拆了会被上游的等待补上。**
这正是报告那句 *"fence 是把等待搬来搬去，它不制造等待"* 的另一面 ——
**等待会顺着流水线往上游找新的堵点。**

**⇒ 做消融时必须有一个"全拆"臂（`all`）作为上界，不能只做逐个拆然后相加。**
逐个拆得到的数只能说明"这一处是不是当前最堵的"，不能说明"这一处值多少"。

---

## 6. 裁决产物

**指标（`D:/verdict-v1/syncfast-metrics/`）：判定 PASS**

| 检查 | 观测 | 门限 |
|---|---:|---:|
| D1 单帧 PSNR(cand, native) ≥ 基线最差帧 − 1.0 dB | **37.3602** | 36.3602（帧 107） |
| D2 单帧 SSIM(cand, native) ≥ 基线最差帧 − 0.01 | **0.9884** | 0.9784（帧 107） |
| D3 帧间步长比 ≤ 1.2× | **1.0000** | 1.2000 |
| R1（仅报告）单帧 max_abs(cand, ref) | **0.0000** | 0.2500 |

**决定性证据：**

- **`cand_vs_ref` 差异像素数 = 0、`all_identical = True`** ⇒ 拆同步一个字节都没改。
- **`cand_vs_native` 与 `ref_vs_native` 的每个数逐位相同**
  （mean `42.51361762840425` / median `42.13933456704723` / worst `37.36022406255206`）
  ⇒ 拆同步没有引入任何新差异。
- `delta_psnr_to_native` 的 mean / median / worst **全 0.0**。

**视频（`D:/verdict-v1/syncfast-video/`）：**

| 文件 | 尺寸 | 大小 |
|---|---|---:|
| `comparison-fourway-24fps.mp4` | 1728×1056 | 7.6 MB |
| `comparison-face-24fps.mp4` | 1536×1632 | 16.1 MB |
| `comparison-diff-24fps.mp4` | 1728×1056 | 15.9 MB |

四路：① 原片 ② 精确线锚点（B580 逐字节等价 4060）③ 现状快速版（未拆同步）④ 拆同步后（本轮）。
**③ 与 ④ 逐字节相同** —— 这正是"没有改画面"的铁证。

**按已定口径，最终以用户人工查看为准。**

---

## 7. 结论与下一步

### 7.1 这一层值不值得留

**值得留**：6.867 ms、1.0905×、逐字节零差异、`off` 离散度 0.0080%（极稳）、可复现。

**累积（第一层 + 第二层）：**

| 层 | 省 | 累计帧墙 |
|---|---:|---:|
| 起点 | — | 82.72 ms |
| 第一层 压 host | 2.385 ms | 80.34 ms |
| 第二层 拆同步 | 6.867 ms | **75.86 ms** |

**⚠️ 两层能不能直接相加，尚未验证**（各自是独立 A/B，未做过"两层同时开"的臂）。
按第二层的发现（串联的等待会被上游补上），**很可能不能相加** —— 必须做一次合并臂才算数。

### 7.2 ⚠️ 执行顺序的修正（用户 2026-09-17 提出，已采纳）

**用户的问题：GPU 侧要换算法换精度，会不会影响 CPU 派发？如果会，是不是该先改 GPU？**

**答案：会，而且最大的那个点正好是"双料"的。**

`FRAME_BUDGET_REPORT.md` §5.3 + §6.3：

- `nr_backend.triton_fp8._kernel` = **383 次 / 7.6132 ms**，**占全部启动成本 34.0%**，两侧完全相同。
- §6.3 结论一原话：*"`triton_fp8.py:42/43` 两行 = 794 op/帧（21.9%）……
  **这是全帧最大的单点，host 侧与设备侧同时最大。**"*
- §8.2 第 1 行给的收益：*"启动次数减半 ≈ **3.8 ms**"* —— 这是 **CPU 侧**的收益。

⇒ **改 `triton_fp8` 的批大小/行数，会同时减少 CPU 启动成本和 GPU 执行时间。**
**先做 CPU 侧的深度优化，有可能在 GPU 改动落地后白做。**

**按"会不会被 GPU 改动影响"重排 CPU 侧的工作：**

| CPU 侧工作 | 类型 | 会不会被 GPU 改动影响 |
|---|---|---|
| 第二层拆同步（**已落袋 6.867 ms**） | 改同步，与 launch 结构无关 | **不受影响** ✅ |
| 第一层刀 1（**已落袋 2.385 ms**） | 每笔启动成本 | 不作废，但笔数减半 ⇒ 收益减半 |
| host 第 2 行做透（≈5.45 ms，未做） | 每笔成本 | **同样缩水** ⇒ **暂缓** |
| `select`（2.85 ms，未做） | 189 次/帧 | 次数可能随 launch 结构变 ⇒ **暂缓** |

**⇒ 修正后的下一块：`triton_fp8._kernel`（第三层 3b），不是"host 第 2 行做透"。**

**代价与门槛**：3b 改行数/批大小 → 改 K 累加顺序 → **会改字节**。
按报告 §8.3：*"必须走 `local` → `gate` → `full` 的逐字节门槛，不得用本报告的毫秒当依据。"*
并且按已定口径，**最终以用户人工查看视频为准**。

### 7.3 ⚠️ 第二个问题的核实结果：线程池是满的，但确实"用不上"

**用户的问题：CPU 慢有可能是没有调用多线程？**

**实测（本机）：**

| | 值 |
|---|---|
| 物理核 / 逻辑核 | **16 / 16** |
| `torch.get_num_threads()` | **16** |
| `omp_get_max_threads()` | **16** |
| `mkl_get_max_threads()` | **16** |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | 均未设置（走默认 = 核数） |

⇒ **不是单线程配置，线程池是满的。**

**但线程配满 ≠ 用得上。** 按 §7 的 CPU 四分区，逐段看为什么：

| 段 | ms | 为什么多线程帮不上 |
|---|---:|---|
| `spill_preflight_v1.select` | 2.83 | 纯 Python，**GIL 锁死** |
| Triton 启动路径（`JITFunction.run`） | 22.37 | 纯 Python，**GIL 锁死** |
| eager ATen 图（净） | 24.17 | 全是 `to` / `empty_like` / `contiguous` / `slice` —— **派发型 op，本身异步，不是 CPU 计算** |
| 模型 Python 残差（净，去掉同步） | ≈19.1 | 纯 Python，**GIL 锁死** |
| （其中）16 次同步等待 | 13.71 | **不是计算，是等待** |

⇒ **约 44 ms 是 Python 解释器开销（GIL 锁死），24 ms 是"把工作丢给 GPU"的派发开销（本身异步），
13.7 ms 是等待。没有一段是"可以被多线程加速的 CPU 计算"。**

**⇒ "加线程"解决不了这个问题。但用户的直觉指向了一个真实的东西：16 个核里有 15 个是闲着的。**

**那 15 个核能被用起来吗？—— 只有一条路：把 CPU 侧的调度搬出 Python。**

**账（报告 §1 的恒等式）：**

```
帧墙 82.21  =  host 提交 68.98  +  fence 税 6.87  +  帧尾 6.36
```

**完美并行（零同步）的帧墙 = `max(68.98, 57.71) + 6.36` = 75.34 ms。**
**而第二层实测 82.72 → 75.86（省 6.867）—— 正好把 fence 税 6.87 ms 吃干净。** ✅ 对得上。

⇒ **剩下的 75.86 ms 里，68.98 是 CPU 单线程的串行调度。这就是新的墙。**

**要突破它，只有两条路：**

1. **减少 CPU 要做的调度笔数 / 每笔成本** —— 刀 1 在做（每笔 4.0 → 2.1 µs）；
   **`triton_fp8` 的 383 次是最大的一笔**（7.61 ms，占启动成本 34%）。
2. **把调度搬出 Python**（C++ 重写 / 图捕获，即"host 步骤直接写到 GPU"那个质变）。

**第 2 条是唯一能同时利用那 15 个闲核的办法。**

### 7.4 还欠的

- **两层合并臂**（第一层 + 第二层同时开）—— 验证能否相加
- **3b `triton_fp8._kernel` 的批大小改动** —— 会改字节，需要 `local → gate → full` 流程
- **第三层 3a / 3c**（无算术 11.0 ms / 三个 `fused_*` 30 次 8.43 ms）
- 1920×1080 素材（需先改 `materials_entry_v1.py` 的形状断言与 `verdict_metrics_v1.py` 的 `NATIVE_SHAPE`）
- LPIPS（需用户批准单独建隔离 venv）
- 阈值定死（`TOL_PSNR/TOL_SSIM/TOL_FLICKER`，等用户看视频后裁定）
