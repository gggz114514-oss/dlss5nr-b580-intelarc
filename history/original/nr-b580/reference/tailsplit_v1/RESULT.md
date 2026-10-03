# tailsplit_v1 —— 7.4 ms「帧尾」到底是什么

> 一轮一目录。装置：`tailsplit_entry_v1.py`（18 个内存补丁）＋ `tailsplit_run_v1.sh`（一个 GPU 租约）＋ `tailsplit_analyze_v1.py`（零租约）。
> 数据：`D:/tailsplit-v1/{off,hook,sync-count,sync-skip,sync-control}/`。
> **产品源 0 改动**：沿用「源码补丁 + `exec`」；`materials_entry_v1.py` 里 `@triton.jit` 计数 = 0 ⇒ 不移动任何内核起始行号 ⇒ 不引入 JIT。

---

## 0. 一句话

**帧尾 7.60 ms 里 7.30 ms（96.0%）是 `torch.xpu.synchronize()` 在等设备**，它是**排水**——宿主做完最后一次提交时设备还欠的活——**不是"固定收尾成本"**。

⇒ **帧尾不是一项独立可优化的成本**；它由宿主的提交曲线派生（§4）。上一轮"帧尾 = 固定成本，不是排水"的说法**撤回**（§3）。

---

## 1. 装置与判据

四段全部是**宿主侧 `perf_counter()`**（成本 ≈0.1 µs/个，不动设备）：

| 记号 | 位置 |
|---|---|
| `t_model` | `stack.model(...)` 返回 |
| `t_float` | `low.float()` 返回（= `submitted`，即最后一次 Python 级提交） |
| `t_sync` | 最终 `torch.xpu.synchronize()` 返回（**仍在 `with` 内**） |
| `t_exit` | `with` 块退出（含 `installed().__exit__` / `call_guard.validate()`） |

⇒ `帧尾 = (t_model − t_a) + (t_float − t_model) + (t_sync − t_float) + (t_exit − t_sync)`

`t_a`（最后一次 Triton 提交）只有**时间线开着**才拿得到 ⇒ 两臂：`ARM=off`（无扰动，给绝对毫秒）、`ARM=hook`（+3.8 ms，给 `t_a`，让分解精确闭合）。

**闭合判据（预注册）**：四段之和必须闭合到 `帧尾`（误差 ≤0.3 ms），否则说明窗口边界找错了。

**结果：中位残差 +0.0000 ms，最大 |resid| 0.0000 ms —— 全部五臂逐帧精确闭合。** 窗口边界找对了。

---

## 2. 四段分解

### 2.1 `ARM=off`（无扰动，32 帧，稳态 = `frames[4:]`，n=57 打点）

| 段 | 中位 (ms) | 占帧墙 |
|---|---:|---:|
| 帧墙 `elapsed` | **44.2849** | 100% |
| A 模型主体（到 `stack.model` 返回） | 36.6586 | 82.8% |
| B `low.float()` | 0.0089 | 0.02% |
| **C 最终 `synchronize()`** | **7.2959** | **16.5%** |
| D `with` 退出 | 0.3043 | 0.7% |
| 提交窗口 `submit_all` | 36.6687 | |
| **帧尾 `tail`** | **7.6005** | **17.2%** |

⇒ **帧尾 7.6005 = C 7.2959 + D 0.3043 + 0.0003**。即 **96.0% 是 `synchronize()` 等待，4.0% 是 `with` 退出**。

⚠️ 各列是**独立取中位**，所以 `36.6687 + 7.6005 = 44.2692` 与中位帧墙 `44.2849` 差 **0.016 ms**；
**逐帧**的 `提交窗口 + 帧尾 == 帧墙` 与 `四段和 == 帧墙` 都是**精确**的（残差 0.0000 ms，见 §1）。

### 2.2 `ARM=hook`（+3.8 ms，但能看到 `t_a`）

| 段 | 中位 (ms) |
|---|---:|
| 帧墙 | 47.8713 |
| 帧尾（本臂重算） | 7.5257 |
| ├ 模型收尾 `(t_model − t_a)` | **0.3730** |
| ├ `low.float()` | 0.0088 |
| ├ 最终 `synchronize()` | **6.8400** |
| └ `with` 退出 | 0.3144 |

⇒ 两臂同形：**`synchronize()` 等待占 91–96%**，模型自身的收尾（最后一次 Triton 提交之后到 `stack.model` 返回）只有 **0.37 ms**。

### 2.3 预注册预测命中

> 预注册：若"固定收尾"主要是 `synchronize()` 的等待，则 `(t_sync − t_float)` 占大头；若主要是模型收尾的 eager ATen 段，则 `(t_model − t_a)` 占大头。

**命中第一支**：6.84 vs 0.37，差 **18 倍**。

---

## 3. ★ 撤回上一轮：帧尾**就是**排水

上一轮（`fp8phase_v1` §7.9/§10、`HANDOVER` §5.25⑦⑨/§7.1/§7.2）写过：

> 「帧尾 = **固定成本，不是排水**」，证据是「探针把提交窗口 +11.05 ms 而帧尾只 −0.39 ms ⇒ **Δ帧尾/Δ提交 ≈ −0.035**，若帧尾是排水应 ≈ −1，差 30 倍」。

**这个判据不判别。** 见 §4 的模型：宿主在帧**中段**多花 11 ms，**不改变"宿主最后一次提交时设备欠多少活"**——那由最后那一段提交曲线决定，不由总时长决定。所以排水也预测 Δ帧尾 ≈ 0。

⇒ **判据作废，结论撤回。** 四段分解给出的正面证据（96% 落在 `synchronize()` 里）才是决定性的：`synchronize()` 返回当且仅当设备队列排空 ⇒ **帧尾 = 设备排水**。

**旁证**（同一份数据，本来就有）：`materials_entry_v1` 里那个被剖析器膨胀到 **1824 ms** 的帧，帧尾只有 **0.4238 ms**（C 0.1065 + D 0.3173）——宿主跨度足够长时设备早就排空了。**排水会在宿主变慢时消失，固定成本不会。**

---

## 4. ★★ 模型：帧墙 = 提交窗口 + 排水，而**排水是派生量**

设 `S(t)` = 到宿主时刻 t 为止**已提交**的设备工作量（ms），`B` = 一帧的设备忙时总和，`T` = 提交窗口长度。对 work-conserving 单服务台：

```
帧墙 W = max_{0≤t≤T} ( t + B − S(t) )
排水   = W − T = max_t [ (B − S(t)) − (T − t) ]  ≥ 0
```

即 **排水 = 整帧里"宿主落后设备"的最大缺口**（未提交的设备活 − 剩余宿主时间）。

**三条推论，全部与实测吻合：**

1. **减宿主成本 ⇒ 帧墙 1:1 下降，而帧尾几乎不变。** 实测：`hook`(T=41.05, tail=7.15) → `off`(T=37.23, tail=7.54)：ΔT = −3.82，ΔW = −3.43，**Δtail 只有 +0.39**。
2. **`帧墙 ≈ 提交窗口 + 7.57`** —— 与 `MEMORY §5` 早已记下的经验式**逐位一致**（36.6687 + 7.6005 = 44.2692 ≈ 44.2849）。
3. **帧尾没有独立杠杆。** 想砍它，只能砍 `S(t)` 的落后量，而那等价于砍宿主的提交曲线 ⇒ **和砍宿主成本是同一件事**。

**⚠️ 一条反直觉但重要的推论**：若把宿主工作放在**排水 argmax 之后**削减，`T` 减小而 argmax 处的 `B−S` 不变 ⇒ **帧墙不变、帧尾变大**。⇒ **"哪一段宿主工作值得砍"取决于它在帧内的位置**，不是所有 host 削减都 1:1。（本轮**没有**定位 argmax——需要 per-launch 设备时间，现有仪器给不出，见 §6。）

---

## 5. 已关闭的旁支：产品的周期 `synchronize()`（内存护栏）**不触发**

产品后端有三处**刻意**的周期性设备同步，注释写明是为了界住 scratch 内存：

| 位置 | 条件 | 注释 |
|---|---|---|
| `attention.py:95` `swin_attention_windows` | 每 `groups_per_sync` 组窗口 | — |
| `attention.py:116` `C32AttentionFront.forward` | 每 8 个 32768 行块 | "Keep the large image's QKV/normalization scratch bounded" |
| `pre_mlp.py:114` `forward_unquantized` | 每 8 个 32768 行块 | "Retain only one bounded expansion/cubic workspace" |
| `executor.py:75` `mark()` | 仅当 `progress is not None` | "Without a callback it is pure waiting, so it is not paid" |

它们本来是最像"设备空转来源"的嫌疑。**用一个只包 `torch.xpu.synchronize` 属性的钩子直接量**（`inmodel` 区间计数 + 可空转）：

| 臂 | 帧墙 (ms) | 提交窗口 | 帧尾 | 帧内 sync 次数 | 其中宿主等待 |
|---|---:|---:|---:|---:|---:|
| `count`（只数，放行） | 44.0424 | 36.2245 | 7.7648 | **0** | 0.0000 |
| `skip`（帧内直接空转） | 44.1701 | 36.5070 | 7.6737 | **0** | 0.0000 |
| `control`（窗口内强行插一次） | 44.2795 | 36.7022 | 7.6775 | **1** | 0.0413 |

- **阳性对照通过**：`control` 臂 `sync_n = 1`（min 1 / max 1）⇒ 钩子确实接上了 ⇒ `count`/`skip` 的 **0 是真 0**。
- ⇒ **本配置下产品在 `stack.model()` 内部一次 `torch.xpu.synchronize()` 都不调用**（行分批改变了 `numel` 判据 / `groups_per_sync` 没走到）。`skip` 与 `count` 帧墙差 0.13 ms ⇒ 这三个护栏**完全不在帧内成本里**。
- 顺带拿到一个单价：**空队列上一次 `synchronize()` 只花 0.041 ms**（宿主侧）。

⇒ **该方向关闭**（🔴 无可兑现项），但这是**有阳性对照的负结果**，不是"没测出来"。

---

## 6. 诚实边界

1. **排水 argmax 的位置没定位。** 需要 `S(t)`（per-launch 设备时间）；`nowsplit3` 的 `dev_cum_ms` 在本栈**全为 None**（`dev_total/submit/tail` 也是 None），`torch.profiler` 的 `key_averages()` 只有聚合值。⇒ §4 推论 3 的"位置"问题**本轮没有答案**。
2. **7.3 ms 排水里有多少是"设备关键路径的最后一坨"、多少是"宿主提交太晚"，没分开。** 两者在 `synchronize()` 里长得一模一样。
3. **剖析遍不能测帧尾**（技能 197）：宿主被抬高 25×（`wall_ms_inflated = 958` vs 真 44.3）而设备工作量几乎不变 ⇒ 设备在宿主跨度内就排空了，队列结构被破坏。本轮的 0.4238 ms 就是证据。
4. **绝对毫秒不作加速比证据。** 本轮只读帧内结构；跨臂差只用来做"是否随 X 变化"的定性判据。
5. `sync_n = 0` 只覆盖**显式** `torch.xpu.synchronize()`；隐式同步（`.item()` / `assert` 张量 / D2H 拷贝等待）不在计数内。
6. `count`/`skip`/`control` 三臂的 `byte_equal` 都是 1/32 —— 这是 `materials_entry_v1` 这个诊断入口的既有状态（它比对的是 `fullsize-rows-v1` 的 `fast`），**不是本轮引入的**，也不影响四段分解（三臂互相一致）。

---

## 7. 复现

```bash
cd E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/tailsplit_v1

# 四段（两臂，各一个 GPU 租约）
LIMIT=32 ARM=off  RUN=D:/tailsplit-v1/off  bash tailsplit_run_v1.sh
LIMIT=32 ARM=hook RUN=D:/tailsplit-v1/hook bash tailsplit_run_v1.sh

# 同步钩子（三臂）
LIMIT=32 ARM=off SYNCHOOK=count   RUN=D:/tailsplit-v1/sync-count   bash tailsplit_run_v1.sh
LIMIT=32 ARM=off SYNCHOOK=skip    RUN=D:/tailsplit-v1/sync-skip    bash tailsplit_run_v1.sh
LIMIT=32 ARM=off SYNCHOOK=control RUN=D:/tailsplit-v1/sync-control bash tailsplit_run_v1.sh

# 分析（零租约）
python tailsplit_analyze_v1.py --run D:/tailsplit-v1/off
```

**⚠️ `EXACT` 必须指向当前产品源的派生记录**：`materials_entry_v1` 逐字节校验 `loaded_sources`，不符当场 `AssertionError`。
刀① + 刀 E 改过 6 个受保护源 ⇒ `D:/quanttrim-land-v1/exact-derived/validation.json`（33 件里 **4 件不符**）已过期，用 **`D:/fp8fast-land-v1/exact-derived/validation.json`（33/33 全对）**。

---

## 8. 下一刀

| 候选 | 预期 | 状态 |
|---|---|---|
| **定位排水 argmax**（帧内哪一段宿主工作"值钱"） | 决定后续 host 削减该砍哪里；**可能推翻"任意 host 削减都 1:1"** | 🔴 需要新仪器（per-launch 设备时间：给 launch 钩子加内核名 + 用剖析器的 per-kernel 设备时间反推 `S(t)`） |
| 产品周期同步护栏 | 0（本配置不触发） | ✅ **已关闭** |
| 图捕获 E3 | ~~≤7.6 ms~~ 上界已撤回；装置要新入口 | 🔴 本刀不做（见 `fp8phase_v1` §10） |
| 批量化 E2 | ≈0.11 ms，且落在设备侧 | 🔴 优先级低 |
| **降提交宿主成本** | **1:1，余量 8.18 ms ⇒ 硬底 36.62 ms = 27.3 fps** | ✅ 仍是唯一已兑现的杠杆 |
