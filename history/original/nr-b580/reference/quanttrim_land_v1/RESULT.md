# 刀 ①（quanttrim）**源码落地**：175 次/帧的空转被摘掉，画面逐字节未变

> **轮次**：`quanttrim_land_v1`（2026-09-19）
> **对象**：`nr-b580` @ `nr/exact` + `nr-b580-int8` @ `nr/int8`（**现行产品**）
> **性质**：**落地 + 双证据链**（逐字节门 243 帧 × 2 臂；ABBA 8 臂配对速度）
> **数据**：`D:/quanttrim-land-v1/`（跑完只留报告与脚本）
> **上游**：`quanttrim_v1/RESULT.md`（诊断轮，靶子在这里挖出来）

---

## §0 一句话

诊断轮挖出的「**块出口 `q(...)` → 下个块入口又 `q(...)`**」重复量化，在
**11 个产品侧调用点（175 次/帧）**整条短路成 `return x`：

| | 结果 |
|---|---|
| **画面** | 243 帧 × 2 臂 **逐字节 == 冻结参考**，`max_abs = 0.0`（**零风险**）；<br>产品 env 下 8 臂逐帧偏差序列**完全相同** ⇒ 两臂是同一个画面（第二重核验） |
| **速度** | 两批 ABBA（门 env + 产品 env）**池化 8 个配对观测**：中位 **−3.8558 ms**，<br>95% CI **[−4.0871, −3.2775]**，**8/8 同号** |
| **帧墙** | 门 env `50.53 → 46.93`；**产品 env `49.00 → 45.16 ms`**（22.14 fps） |

> **⚠️ 两个 env 必须分清**：逐字节门**必须** `NR_MOTION_SCOPE=off`（冻结参考早于 N4c 落地），
> 而**产品实际跑 `on`**。所以速度跑了两批，见 §4。

---

## §1 ★★ 落地时踩的坑：Triton 缓存键含**内核起始行号**

### §1.1 症状

第一版落地（`quanttrim_land_apply_v1.py`）在 `nr-b580-int8/experimental/batched_branched_mlp_v1.py`
的 import 段**新增了一行** `from nr_backend.pre_mlp import quantize_fp8_grid as qg`。
冒烟直接死在：

```
RuntimeError: Missing fast artifact; prepare offline before video processing: _pairs
  （fast_cached_runtime_v1.py:39）
```

### §1.2 机制（读源确认，不是推的）

`triton/runtime/jit.py`：

```python
@property
def cache_key(self) -> str:                       # JITFunction
    ...
    self.hash = dependencies_finder.ret + str(self.starting_line_number)
    self.hash = hashlib.sha256(self.hash.encode("utf-8")).hexdigest()
```

其中 `DependenciesFinder.__init__` 是 `self.hasher = hashlib.sha256(src.encode("utf-8"))`，
`src` = **该内核自己的源码文本**（`inspect.getsource`）。而
`triton/compiler/compiler.py` 的 `ASTSource.hash()` 再把它与签名/常量拼进磁盘缓存键。

⇒ **内核的起始行号是缓存键的一部分**。`_pairs` 从第 20 行被推到第 21 行 ⇒ 键全变 ⇒
`DiskOnly` 在冻结的 `D:/Codex-NR-Experiments/nr-b580/reference/triton-cache-c32-triton38-v1`
里查不到 ⇒ 拒绝在推理期编译（这正是它的设计目的）。

### §1.3 ★ 由此得到一条落地硬约束

> **在含 Triton 内核的文件里，改动必须是「同行文本替换」——一行都不能多、不能少。**

本轮的处置：

| 文件 | v1 的做法 | 修正（v2） |
|---|---|---|
| `experimental/batched_branched_mlp_v1.py` | 新增 import 行 ⇒ `_pairs` 20→21、`_project` 40→41 | **整字节还原**，调用点改用已导入的 `multihead.quantize_fp8_grid` |
| 另 3 个 `experimental/*.py` | 新增 import 行（**无内核 ⇒ 无害**，但规则不统一） | 同样改成「已导入模块的属性」：`blocks.quantize_fp8_grid` / `multihead.quantize_fp8_grid` / `decoder.q_grid` |
| 载体模块 | — | `multihead_block.py:5`、`decoder.py:5` 在**已有 import 行末尾**追加名字（同行、零位移） |

* `pre_mlp.py` 新增 25 行**但该文件不含任何内核** ⇒ 行号位移对缓存键无害，保持原样。
* 复核脚本（`quanttrim_land_apply_v2.py`）把这条约束写成了**会 FAIL 的断言**：
  `MUST_KEEP_LINECOUNT` + 逐个内核比对起始行号（技能 137）。

### §1.4 第二个坑：受保护源清单

`materials_entry_v1.py:141` 有 `assert paths.sha(path) == digest, path` ——
它把采集那一刻的**产品源文件 sha256** 钉在 `--exact-validation` 里。
改了产品源码就必须**重新派生采集记录**（技能 123 只对源码改动生效）：

```
[derive] 243 input frames verified byte-for-byte against disk
[derive]   backend/nr_backend/pre_mlp.py          0a7c9689 -> 571c0b0f
[derive]   backend/nr_backend/split_block.py      aa74edd6 -> 820857ba
[derive]   backend/nr_backend/c32_block.py        6083c1c3 -> 678fa0a5
[derive]   backend/nr_backend/post.py             a8c4551e -> 2491d1bc
[derive]   backend/nr_backend/multihead_block.py  f5fcd992 -> 304e7882   ← 本轮新增
[derive]   backend/nr_backend/decoder.py          a851e670 -> 0621efcf   ← 本轮新增
```

派生脚本断言「**恰好**这 6 个变了」；多一个少一个都 FAIL。

---

## §2 落地形态

### §2.1 新函数（`nr_backend/pre_mlp.py`，两树）

```python
_GRID_GUARD_OFF = os.environ.get('NR_QUANTTRIM_GUARD', 'on').strip().lower() in ('', 'off', 'none', '0', 'false')

def quantize_fp8_grid(x: Tensor) -> Tensor:
    if _GRID_GUARD_OFF or x.dtype != torch.float16 or not x.is_contiguous():
        return quantize_fp8(x)              # off 臂 = 逐字复现落地前
    record_arithmetic_dispatch('fp8')       # ★ 账照记 ⇒ 算术账本与落地前逐项相同
    return x
```

两个**廉价前置条件**（dtype / 连续性）是诊断轮 §6.1 五项里的两项，留作运行时保险；
任一不成立就退回真转换。`NR_QUANTTRIM_GUARD=off` 可整体关掉。

### §2.2 11 个调用点（实测 175 次/帧）

| 文件 | 行 | 次/帧 | 落地后写法 |
|---|---:|---:|---|
| `int8/experimental/window_blocks_v3.py` | 29 | 36 | `blocks.quantize_fp8_grid(features)` |
| `int8/experimental/window_blocks_v3.py` | 48 | 36 | `blocks.quantize_fp8_grid(features)` |
| `int8/experimental/batched_branched_mlp_v1.py` | 73 | 36 | `multihead.quantize_fp8_grid(features)` |
| `backend/nr_backend/split_block.py` | 36 | 32 | `qg(features)` / `qg(residual)` |
| `int8/experimental/native_half_head_layout_v1.py` | 31 | 16 | `multihead.quantize_fp8_grid(features)` |
| `backend/nr_backend/c32_block.py` | 35 | 7 | `quantize_fp8_grid(features)` |
| `int8/experimental/decoder_gather_scope_v1.py` | 33 | 5 | `decoder.q_grid(skip)` |
| `int8/experimental/decoder_gather_scope_v1.py` | 60 | 4 | `decoder.q_grid(features)` |
| `int8/experimental/decoder_gather_scope_v1.py` | 51 | 1 | `decoder.q_grid(features)` |
| `backend/nr_backend/post.py` | 47 | 1 | `qg(features)` |
| `backend/nr_backend/post.py` | 48 | 1 | `qg(skip)` |
| **合计** | | **175** | |

载体（两树各一份、逐字节相同，均**不含内核**、**零行数变化**）：

```
multihead_block.py:5  from .pre_mlp import cubic_quantize,_decode_weights,quantize_fp8,quantize_fp8_grid
decoder.py:5          from .pre_mlp import _decode_weights,quantize_fp8 as q,quantize_fp8_grid as q_grid
```

> **与计划的偏差**：诊断轮 §8.2 说「两树只共改 `{pre_mlp,split_block,c32_block,post}.py`」。
> 实际多了 `multihead_block.py` / `decoder.py` 两个**载体**。原因就是 §1.3 那条硬约束：
> 在含内核的文件里不能新增行，所以只能借用「调用点已经导入的模块」。
> 这两个改动**本身是惰性的**（只在 import 行末尾加一个名字），但它们进了受保护源清单。

---

## §3 证据链 A：逐字节门（243 帧 × 2 臂）

| 臂 | 帧数 | 逐字节 == 冻结参考 | `max_abs` | 非零帧 | 叶子转换/帧 |
|---|---:|---:|---:|---:|---:|
| `off`（`NR_QUANTTRIM_GUARD=off`） | 243 | **243/243** | 0.0 | 0 | **383.00** |
| `on`（`NR_QUANTTRIM_GUARD=on`） | 243 | **243/243** | 0.0 | 0 | **208.00** |

* `off` 臂 243/243 = **地基自证**：这套 env 能复现冻结参考（`NR_MOTION_SCOPE=off` 是硬要求 ——
  `sampling.py:18` 的 N4c 守卫默认就是 `on`，不关掉输出不再等于冻结参考）。
* `on` 臂 243/243 + `max_abs = 0.0` ⇒ **本刀构造性逐位等价，画面零变化**。
* 计数自证精确命中期望：`383 − 175 = 208`。两臂都装同一个**只读计数包装器**（打在
  `triton_fp8.quantize_fp8` 这个**叶子**上），所以对 Δ 无偏。

---

## §4 证据链 B：ABBA 配对速度（两批 × 8 臂 × 12 帧）

### §4.0 为什么必须跑两批

逐字节门**必须** `NR_MOTION_SCOPE=off`（`sampling.py:18` 的 N4c 守卫**默认 `on`**，
但冻结参考早于它落地 —— 不关掉输出不再等于冻结参考，地基自证就做不了）。
**而产品实际跑 `on`（省 +1.6995 ms）** ⇒ 门的 env **不是**产品的 env。

所以速度跑了两批，AB 脚本把 `MOTION_SCOPE` 参数化了：

```bash
MODE=abba LIMIT=12                  bash quanttrim_land_ab_run_v1.sh   # 批1 门 env
MODE=abba LIMIT=12 MOTION_SCOPE=on  bash quanttrim_land_ab_run_v1.sh   # 批2 产品 env
```

### §4.1 批 1：门 env（`NR_MOTION_SCOPE=off`）

| 臂 | 帧墙中位（`frames[4:]`） | 范围 | spread |
|---|---:|---|---:|
| `off` | **50.5316 ms** | [49.8616, 50.5977] | 0.7360 |
| `on` | **46.9296 ms** | [46.4765, 47.2411] | 0.7646 |

配对差（`on − off`，按 **tag** 取方向）：`−4.0461 / −2.6835 / −3.8596 / −3.3566`
⇒ 中位 **−3.6081 ms**，4/4 同号，|Δ| ≈ 4.9× 臂内 spread（技能 127 ✓）。

### §4.2 批 2：**产品 env**（`NR_MOTION_SCOPE=on`）

| 臂 | 帧墙中位 | 范围 | spread |
|---|---:|---|---:|
| `off` | **48.9966 ms** | — | — |
| `on` | **45.1570 ms** | — | — |

配对差：`−3.5058 / −4.0552 / −4.0995 / −3.8521` ⇒ 中位 **−3.9537 ms**，4/4 同号。

> ★ **副产品（有价值的第二重核验）**：批 2 里 `byte_eq` 只有 1/12、`max_abs = 2.783e-02`
> —— 那是 `NR_MOTION_SCOPE=on` 相对冻结参考的差（**预期**）。但 **8 个臂的逐帧 `max_abs`
> 序列去重后只有 1 种** ⇒ **`on` 臂与 `off` 臂在产品 env 下也是同一个画面**。
> 这比"门 env 下 243/243"更贴近产品实际配置。

### §4.3 池化（技能 139）

| 批 | 配对观测 | 中位 | sd（族内） |
|---|---|---:|---:|
| 批 1（门 env） | −4.0461 / −2.6835 / −3.8596 / −3.3566 | −3.6081 | 0.6094 |
| 批 2（产品 env） | −3.5058 / −4.0552 / −4.0995 / −3.8521 | −3.9537 | 0.2706 |
| **池化 n=8** | | **−3.8558** | **0.4841** |

* 池化均值 **−3.6823 ms**，95% CI **[−4.0871, −3.2775]** ⇒ **排除 0** ✓
* **8/8 同号** ✓
* 两批中位差 −0.3456 ms ⇒ 在噪声内 ⇒ **Δ 迁移到产品 env 成立**（这正是跑批 2 要回答的问题）
* ★ **族间 sd 又差 2.25×**（0.6094 vs 0.2706）—— 技能 139 的第二次命中：
  **同族不同批次 sd 可差数倍，点估计必须池化。**

> **引用口径**：引用 **池化中位 −3.86 ms（CI [−4.09, −3.28]，n=8）**；
> 分环境说就报批 2 的 **−3.95 ms**（产品 env）与批 1 的 **−3.61 ms**（门 env）。
> `off` 臂的 50.5316 与账上记的 50.54 吻合 ⇒ 装置本身没有引入可观测偏差。

**帧墙**：门 env `50.5316 → 46.9296`（−7.1%）；**产品 env `48.9966 → 45.1570 ms`
（22.14 fps）**。后者的 `off` 臂 48.9966 与 `nowsplit_v1` 在现行产品上实测的 48.4406
同量级（差 0.56 ms = 会话漂移 + 计数包装器）⇒ 互证。

### §4.1 ★ 修正诊断轮 §7 的一个用词

诊断轮 §7 把 `191 × 17.5 µs = 3.35 ms` 叫作「**收益上界**」。本刀实测：
**175 次 ⇒ 3.61 ms**（每次 20.6 µs），**超过了按 17.5 µs 折算的 3.06 ms**。

⇒ 那个数**不是上界**，是**孤立调用**（设备空闲、每次前后同步）的价。真实帧里 host 是瓶颈侧
（host/device ≈ 1.19×），一次 `quantize_fp8` 的 host 价 ≈ **20.6 µs**。
`quanttrim_v1/RESULT.md` §7 的「上界」一词应在下次引用时改口为「下界/孤立价」。

### §4.2 与补丁形态的关系

补丁形态（v1，白名单 13 点 = 191 次）ABBA 中位 `−3.6604 ms`；落地形态（11 点 = 175 次）
中位 `−3.6081 ms`。两者差 0.05 ms ⇒ **在噪声内**。

⚠️ 由此得到一条口径：**参考侧复放装置那 2 个点（16 次/帧）的收益在本装置上不可分辨**
（预期 0.28 ms，实测差 0.05 ms）。所以「复放装置的点不动」这个决定**没有代价**，
但也**不能用本装置去估它们的收益**。

---

## §5 诚实边界

* **两条证据链是独立的**（技能 138）：逐字节门回答「能不能落地」，ABBA 回答「值不值得落地」。
  两者都在**落地形态**下重测，不用补丁形态的数字顶替。
* 速度绝对值含**计数包装器**的开销（两臂都装 ⇒ Δ 基本无偏；`on` 臂少 175 次包装调用，
  约 0.02 ms，可忽略）。
* **1 个装置、1 份素材（`clip480.mp4` 243 帧）、2 个 env × 4 对 = 8 个配对观测。**
  8/8 同号 + |Δ| ≈ 5–13× 臂内 spread 足以支撑「正收益」这一档结论；
  池化值按技能 139 报，且**已报族间 sd（2.25×）**。
* ⚠️ 批 2（产品 env）的 `byte_eq` 只有 1/12，**不要误读成"本刀改了画面"** ——
  那是 `NR_MOTION_SCOPE=on` 相对冻结参考的差，**8 个臂的逐帧偏差序列完全相同**
  ⇒ 两臂互相之间仍是同一个画面。
* **参考侧镜像的 2 个点没落地**（`materials_entry_v1.py:199/200`，16 次/帧）。
  它们的**产品对应位置**是 `vit_block.py:74/75`，但镜像把 FFN 换成了
  `stack.int8_vit.ffn(index, x)` —— **输入生产者不同** ⇒ 镜像上的「已在网格上」结论
  **不能直接搬**过去。要落地得对产品自己的 ViT 路径单独做一次探针。**列为后续**。
* 「11 个点 / 175 次」是**本快照**的结论。产品结构一变，清单必须重算
  （诊断轮 §9 已声明；本刀复核了计数在**两批、16 个臂**里都是 383/208，没漂）。
* 本刀**不动**任何冻结参考、不改历史入口、不发布上传。

---

## §6 复现

```bash
cd E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/quanttrim_land_v1

# 0) 落地（含行数/内核行号断言；先 --check）
python -X utf8 quanttrim_land_apply_v2.py --check
python -X utf8 quanttrim_land_apply_v2.py

# 1) 重新派生采集记录（改了产品源就必须做，否则 materials_entry 的清单断言会挂）
python -X utf8 quanttrim_land_derive_exact_v1.py --out D:/quanttrim-land-v1/exact-derived

# 2) 冒烟（2 帧）
RUN=D:/quanttrim-land-v1/smoke MODE=gate  LIMIT=2   bash quanttrim_land_ab_run_v1.sh
# 3) 逐字节门（243 帧 × 2 臂）
RUN=D:/quanttrim-land-v1/gate  MODE=gate  LIMIT=243 bash quanttrim_land_ab_run_v1.sh
# 4) ABBA 配对速度（8 臂 × 12 帧）—— **两批都要跑**
RUN=D:/quanttrim-land-v1/abba  MODE=abba LIMIT=12                  bash quanttrim_land_ab_run_v1.sh
RUN=D:/quanttrim-land-v1/abbaP MODE=abba LIMIT=12 MOTION_SCOPE=on  bash quanttrim_land_ab_run_v1.sh

# 出结论
python -X utf8 quanttrim_land_summary_v1.py --run D:/quanttrim-land-v1/gate  --mode gate
python -X utf8 quanttrim_land_summary_v1.py --run D:/quanttrim-land-v1/abba  --mode abba
python -X utf8 quanttrim_land_summary_v1.py --run D:/quanttrim-land-v1/abbaP --mode abba
```

| 脚本 | 作用 |
|---|---|
| `quanttrim_land_apply_v2.py` | **落地（修正版）**：还原 + 同行替换 + 载体 import；断言行数守恒与内核行号 |
| `quanttrim_land_apply_v1.py` | 第一版落地（**已被 v2 取代**，保留作 §1 的现场证据） |
| `quanttrim_land_derive_exact_v1.py` | 派生采集记录（6 个受保护源） |
| `quanttrim_land_entry_v1.py` | 落地形态入口：只读计数叶子 + 调 `materials_entry_v1.main` |
| `quanttrim_land_ab_run_v1.sh` | 一个租约里跑完所有臂（`gate` / `abba`） |
| `quanttrim_land_summary_v1.py` | 读臂目录出结论（含配对方向修正） |
| `quanttrim_land_gate_v1.py` | **补丁形态**的旧入口（白名单已失效，不要再跑） |
| `backup/` | 18 个文件的落地前字节（回退用） |

**数据清理**：`D:/quanttrim-land-v1/` 只留 `summary.txt`（5 份）、`apply-v2.json`、
`exact-derived/validation.json`（**派生链的一环，下一轮从它接着派生**）。
所有 `frames/` 已删（**3.2 GB → 2.1 MB**），诊断轮的 `D:/quanttrim-v1/` 一并清了帧。

---

## §7 结论与下一步

### §7.1 已成立

1. **落地成立**：11 个产品侧点、175 次/帧整条短路；243 帧 × 2 臂逐字节 == 冻结参考；
   产品 env 下 8 臂逐帧偏差序列完全相同。
2. **速度成立**：两批池化 **8 个配对观测、中位 −3.8558 ms、95% CI [−4.0871, −3.2775]、8/8 同号**；
   **Δ 迁移到产品 env 成立**（批间中位差 −0.35 ms，在噪声内）。
3. **帧墙账**：`82.2 → 64.82（N2b）→ 57.88 → 51.96（N4）→ 50.54（刀 1）→ **45.16**`
   （末段是**产品 env** 的实测：48.9966 → 45.1570）。
4. **一条可复用的硬约束**：含 Triton 内核的文件里改动必须同行 —— 缓存键含内核起始行号。
5. **一条可复用的口径**：逐字节门要求的 env 可能 ≠ 产品的 env ⇒ **速度要在产品 env 里另测一批**。

### §7.2 待裁决

* **保留落地**（建议）。`NR_QUANTTRIM_GUARD` 默认 `on`，`off` 可一键回退到落地前。
* **是否送人工审核**：本刀逐字节等价 ⇒ 画面**没有变化**，按 `VERDICT_PROTOCOL.md`
  的定位这属于「不需要审核」的一档。但**是否要出一份送审素材**由用户定。
* **参考侧镜像的 2 个点**（16 次/帧）：要不要对产品 `vit_block.py:74/75` 单独做探针？

### §7.3 实时门的剩余距离

实时门 = XeSS-FG 输入 40/60 fps ⇒ **25.0 / 16.7 ms**。当前 **45.16 ms**（产品 env 实测），
距 40 fps 还差 **20.16 ms**。这一刀贡献了 **3.86 ms（池化）**。
