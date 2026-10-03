# `fullchain_timing_v1` —— NR×XeSS 全链计时基线

> 日期 2026-09-20 ｜ 状态：**原尺寸快速线已实测（45.83 ms，243/243 逐字节）；
> 且已接进实时链并端到端实测（§7：1728×960+FG = 56.07 ms/输入帧，NR 占 85%）**
> 起因（用户）：*「你先测一下，然后计划两个项目正式，这次合并以后，nr-xess全链的时间我们都需要监控」*
> 目标链（用户 2026-09-20 明确）：**`GPU Block Lite V4 → NR → XeSR → XeFG`**，
> 其中 NR 是**「现在优化的原尺寸快速版」**。
> **§7 是项目②（接实时链）的第一步成果；§1–§6 是项目①（NR 自身帧墙）的结论。**

---

## 0. 一句话

**「原尺寸快速版」能跑，本轮实测帧墙 = `45.8256 ms`（243/243 逐字节等于冻结参考）。**
先前那份"三条入口全都跑不起来"的结论**是我用错了入口**：原尺寸快速线的入口是
`reference/materials_v1/materials_entry_v1.py`（所有落地轮都用它，`--exact-validation` 可覆盖），
**不是** `fullsize_rows_v1/rows_entry_v1.py`（那是 09-15/09-16 去分批轮的入口，它把
exact+fast 两份冻结记录一起钉住，而 fast 记录没有派生链 ⇒ 09-19 起才不可用）。

**真正被堵住的只有一条**：产品快速线（`nr_worker_v1.dll --gpu-mode sr-fg`），
它的 Triton 缓存是共享缓存 `reference/triton-cache-c32-triton38-v1`，有效条目止于 **09-14 17:xx**。

---

## 1. ★★★ 本轮实测：原尺寸快速线的当前帧墙

**装置**：`measure_nr_v1.sh`（本轮新建）→ 租约 → `materials_entry_v1.py`

```
--phase full --c512-arith int8 --frames 243
--exact-validation D:/fp8fast-land-v1/exact-derived/validation.json
--baseline-validation D:/fullsize-rows-v1/r1/full-both/validation.json
--cache D:/fullsize-rows-v1/r1/triton-cache
```

**环境**（产品源码默认值，只把外挂 cuts 显式关掉 —— 与历次落地轮同口径）：

```
PREPFAST_CUTS=off SELECTFAST_CUTS=off SYNCFAST_CUTS=off MOTION_SCOPE_CUTS=off ALLMERGE_CUTS=off
NR_LAUNCH_FASTPATH="run,getitem,launch,md,book"   (= LaunchFastPath.DEFAULT，非覆盖)
NR_SELECT_CACHE=on NR_FMA_SCALAR_CACHE=on NR_ALLMERGE_GUARD=on
NR_QUANTTRIM_GUARD=on NR_PADGUARD=on NR_QKVCACHE=on NR_MOTION_SCOPE=off
```

| 量 | 值 |
|---|---:|
| **帧墙中位 `median_ms`** | **45.8256 ms** |
| 均值 / 最小 / 最大 | 45.9405 / 42.1695 / 64.0532 ms |
| 样本 | 239（frames 4..242） |
| **逐字节门** | **`identical_frames = 243 / 243`**，`max_abs = 0.0` |
| `cache_mode` / `cache_hits` | `disk_only` / **164** ⇒ **零 JIT，计时有效** |
| `ffn_calls` 每帧 | `{c512: 16, vit: 8}` —— 与契约一致 |
| `fs_guard` | `installed: true` ⇒ **缓存未被 removedirs 删** |
| `error` | `None` |
| 数据 | `D:/fullchain-timing-v1/nr-current/run/validation.json` |

⇒ **45.83 ms ≈ 21.8 fps**。与 `quanttrim_land_v1/RESULT.md` 的 45.16 同口径、同量级
（单臂 vs 配对臂的差别，本轮未做 ABBA 配对）。

---

## 2. ★★★ 三处更正（我先前说错的）

### 更正 ①：原尺寸快速线的入口不是 `rows_entry_v1.py`

| | 用 `rows_entry_v1.py`（我先前） | 用 `materials_entry_v1.py`（正确） |
|---|---|---|
| 钉几份记录 | **exact + fast 两份**（`:224`） | **只钉 exact**（`:140-141`） |
| 可覆盖 | ❌ 无 `--exact-validation` 参数 | ✅ `--exact-validation` |
| 结果 | 09-19 起 `AssertionError` | **能跑**（本轮实测 243/243） |
| 谁在用 | 09-15/09-16 的去分批轮 | **所有落地轮**：quanttrim / padguard / qkvcache / fp8fast / matmulcache2 / fp8phase |

证据：`padguard_land_ab_run_v1.sh:45` 原文 ——
*「旧记录会让 `materials_entry_v1.py:141` 的 sha 断言当场炸（第一次跑就是这么炸的）」*。

### 更正 ②：精确记录**不需要我派生** —— 上一个会话今天 13:24 已经派生好了

`D:/fp8fast-land-v1/exact-derived/validation.json`：

| 量 | 值 |
|---|---|
| mtime | **09-20 13:24**（= `pre_mlp.py` 最后一次改动那一刻） |
| `loaded_sources` | 33 |
| **与磁盘不符** | **0** |

⇒ **"先做派生"这件事已经不需要做了。** 派生链（`D:/<round>-land-v1/exact-derived/`）：
`warpsync → allmerge → motion-scope → quanttrim → padguard → fp8fast`，最新一份就是当前状态。

### 更正 ③：撤回「XeFG 输入硬门 40 fps = 25.0 ms」

**这条没有来源。** 查了两处：

| 查的地方 | 结果 |
|---|---|
| `xess-tools/sdk/official/doc/*.md` | 无 "40 fps" / "minimum frame rate" 之类下限 |
| `src/realtime/vpl_gpu_full_sr_fg.cpp` | `input_fps` 默认 **82.5**；`fps_cap = input_fps × fg_multiplier`（`:1731`）；**无下限** |

⇒ 我先前对用户说的「差 1.8×」「够不到 25.0 ms 门」**撤回**。
`realtime_ingest_v1/PLAN.md` §1.5 已经写过同一条精神的话：
**XeFG 自身的 GPU 成本从未被测出 ⇒ FG 这一环「既不能判成立也不能判不成立」。**

---

## 3. 唯一的真故障：产品快速线的 Triton 缓存失效

| 项 | 值 |
|---|---|
| 产品快速线的缓存 | `reference/triton-cache-c32-triton38-v1`（`fast_cached_runtime_v1.py:12` 硬设） |
| 非空条目 | 1293 |
| **最新有效** | **09-14 17:xx** |
| 源码改动 | 09-19 15:49 ~ 09-20 13:24（两树 backend 9 个 + int8 experimental 9 个 + `product/nr_runtime_v1.py`） |
| 症状 | `RuntimeError: Missing fast artifact; prepare offline before video processing: _kernel` |
| | `[full-gpu] input=0 output=0 expected=0 encoded=0 wall=3.412s pass=0` |

**★ 关键旁证：整条 XeSS 侧是好的**（失败前全部初始化成功）：

```
[xefg] libxess_fg.dll version 1.2.0
[xefg] proxy swap chain: 1728x960, 4 buffers, format 28
[gpu-block] motion source ready: current->previous, repair=refine, center-bias=0.002000
[nr-scale] quality=101 input=864x480 output=1728x960 valid=1
[sr] XeSS context ready input=864x480 output=1728x960 quality=performance
```

⇒ **XeFG / XeSS SR / GPU Block / 捕获全部就绪，断点只在 NR 的第一个内核。**

**修法**（`HANDOVER.md` §1677 已给出，精确线用过）：**record-only 派生，零重编译**
—— 只刷新包/记录里的源码哈希，缓存产物复用（落地件都是"同行替换"，Triton 缓存键含内核起始行号，
行号不移动 ⇒ 键不变）。精确线的 `quanttrim_exact_derive_v1.py` 是现成模板。

---

## 4. ★★ 环境陷阱：这个卷的 `rmdir` 会递归删

`rows_fscache_guard_v1.py` 原文：

> Triton's `put()` ends with `os.removedirs(temp_dir)` **after** having moved the payload
> out of temp_dir into the cache directory. `os.removedirs` removes the leaf and then
> **walks up the path pruning empty parents**, and it depends on `rmdir(parent)`
> ***raising*** once a parent is non-empty to stop walking.
> **On this volume `rmdir()` does not raise for a non-empty directory: it removes it recursively.**

**⇒ 任何没装 `rows_fscache_guard_v1` 的 Triton 运行，都会在每次 `put()` 时顺着往上删缓存树。**

**本轮记账**：第一版探针（走产品 `fast_cached_runtime_v1`，**没装守卫**）在 16:11–16:17
创建了 **1239 个缓存目录，其中只有 4 个非空**。共享缓存里 `09-20_16` 时段只有 **4** 个非空条目。

| 缓存 | 非空 | 最新 |
|---|---:|---|
| 共享 `reference/triton-cache-c32-triton38-v1` | 1293 | 09-14 17:xx |
| 原尺寸 `D:/fullsize-rows-v1/r1/triton-cache` | **359** | 09-20 13:xx |

⚠️ **未决**：09-14 之后到 09-19 之间共享缓存是否有过写入、本轮是否删掉了它们，**未证实**。
**下一次碰共享缓存前必须先装守卫。** 好消息：`materials_entry_v1.py:61-69` **自带**这个守卫
（本轮 `fs_guard.installed = true`）。

---

## 5. 装置清单

| 文件 | 作用 | 状态 |
|---|---|---|
| `measure_nr_v1.sh` | **原尺寸快速线当前帧墙**（本轮主装置，已跑通） | ✅ 实测 45.83 ms |
| `probe_arm_v1.py` | 产品全链臂探针（`sr1728` vs `srfg1728` **只差 `--gpu-mode`** ⇒ 差 = 纯 XeFG） | 🟡 装置好，源被堵 |
| `freeze_sources_v1.py` | 冻结当前源（本轮已冻 **742** 个文件） | ✅ |
| `summarize_v1.py` | 汇总各臂 + 增量 + **守卫列**（防"空转臂"被当成"快臂"） | ✅ |

**臂定义**（`probe_arm_v1.py`）：

| 臂 | gpu-mode | 输出 | 深度 | 用途 |
|---|---|---|---|---|
| `sr864` | sr | 864×480 | 在线 | |
| **`sr1728`** | **sr** | **1728×960** | **在线** | **与下一行配对** |
| **`srfg1728`** | **sr-fg** | **1728×960** | **在线** | **差值 = XeFG 增量** |
| `srfg864` | sr-fg | 864×480 | 在线 | FG 在 1× |

⚠️ **这套装置测的是产品 Session（NR256）**。测原尺寸快速版要走 `materials_entry_v1`（见 §1）。

---

## 6. 诚实边界

1. §1 的 45.83 ms 是**单臂 243 帧**，未做 ABBA 配对（跨轮比较用配对中位差，本轮不是）。
2. §3 的产品症状是**实测报错原文**；"缓存失效是根因"是**推断**（缓存时间戳 + 症状吻合），
   未做逐内核比对。
3. §4 的"本轮删了缓存"是**部分证实**：1239 个新目录确实只有 4 个非空；
   "是否连带删了旧条目"**未证实**。
4. §2 更正③ 的撤回有**两处独立查证**（SDK 文档、工具箱源码）。
5. 未改产品源、未改冻结参考、未做 git 提交。
6. **仍未取得**：全链（NR→SR→FG）的端到端数字。原因是 §3（产品侧缓存失效）
   ＋ 原尺寸 NR 还不在产品链里（那是项目②）。
   > **⚠️ 本条已被 §7 取代（2026-09-20 17:48）**：原尺寸 NR 已接进实时链，端到端数字已实测。

---

## 7. ★★★ 项目② 第一步：原尺寸 NR 已接进实时链并实测

> 追加于 2026-09-20 17:15–17:48。起因（用户）：*「先把原尺寸栈接进实时链，
> 先完成 480nr-xess sr1080p-fg 的链路，现在太需要一次游戏实测让确定 nr 需要优化到多少毫秒了」*

### 7.1 结论：三个数

**装置**：`run_fullsize_arm_v1.sh`（租约）→ `run_arms_v1.py` → `probe_arm_v1.py`
→ `nr_worker_v1.dll`（**未改动**，与 Sep 11 基线同一个二进制）。
243 帧、`DiskOnly`（计时期禁 JIT）、预热后。

| 链 | 输出 | ms/**输入**帧 | ms/输出帧 | 输出 fps | n | 极差 |
|---|---|---:|---:|---:|---:|---:|
| **原尺寸 NR → XeSS SR perf → XeFG** | **1728×960** | **56.07** | 28.04 | **35.7** | 4 | 0.8% |
| 同上，**1080 行** | **1944×1080** | **56.68** | 28.34 | 35.3 | 2 | 0.4% |
| **NR 隔离**（864×480 进出、无 SR/FG/depth） | 864×480 | **47.62** | 47.62 | 21.0 | 3 | 1.0% |

⇒ **NR 占全链 `47.62 / 56.07 = 85%`。** 其余（SR+depth+FG+呈现）
只占 **`8.45 ms/输入帧`**。
> ⚠️ **本行的分解已于当晚修正 → 见 §8.4**。正确分解 = AI depth 6.75 + **FG present 1.21** + SR 0.53
> = **8.49**（闭合 0.5%）。**旧写法漏了 FG 的 present，且误把 motion 0.40 算进来**（它在 NR 隔离臂里本来就有）。

**参照**：Sep 11 同一链路但 **NR256** 栈 = 34.44 ms/输入帧（不同 harness、不同源码状态，
只作数量级参照，**不作差值的被减数**）。

### 7.2 ★★★ 预算：NR 要优化到多少毫秒

FG 是 2× 输出（243 → 485）。每输入帧预算 = `1000 / (目标输出 fps / 2)`；
NR 上限 = 预算 − 8.45（非 NR 部分）。

| 目标输出 | 输入 fps | 每输入帧预算 | **NR 上限** | 距当前 47.62 |
|---:|---:|---:|---:|---:|
| **60** | 30 | 33.33 ms | **24.9 ms** | −22.7（**−48%**） |
| 90 | 45 | 22.22 ms | **13.8 ms** | −33.8（−71%） |
| 120 | 60 | 16.67 ms | **8.2 ms** | −39.4（−83%） |
| 144 | 72 | 13.89 ms | **5.4 ms** | −42.2（−89%） |

⇒ **一句话：要让这套链上 60 fps 输出，原尺寸 NR 必须从 47.6 ms 砍到 ≈25 ms。**

⚠️ 这条算式的**被减数 8.45 里 AI depth 占 6.75（79.5%）**。若 depth 能免掉或降档，
60 fps 的 NR 上限可放宽到 **≈31.6 ms**。这是唯一一个"不动 NR 也能腾出预算"的杠杆。
（**分解已修正 → §8.4**；`motion 0.40` 不在被减数里，FG 的 present 1.21 在里面。）

### 7.3 怎么接进去的：**产品源零改动**

| 文件 | 作用 |
|---|---|
| `fullsize_session_v1.py`（新） | 包一层产品 `Session`，对外暴露**同形** `process(rgb,motion,reset=)` / `close()`。内部：**不建、不套 `Face480Scale`** + `rows_scopes.install(stack,'both',None,None)` + 7 个猴子补丁（`SplitSwinBlock.forward/forward_boundaries`、`VitBlock.forward`）。三个闭包**逐字移植**自 `materials_entry_v1.py:172-230`。 |
| `probe_arm_v1.py`（改） | 新增 `NR_FULLSIZE`（默认 **off**，默认行为与之前完全一致）与**预热**。 |
| `run_arms_v1.py`（新） | 一个租约跑多臂、全量落 `stdout.log`/`stderr.log`、阶段级环境覆盖。 |

**没碰**：`nr_worker_v1.dll`（二进制未变）、`nr_runtime_v1.py`、`nr-b580/backend/nr_backend/*`
（精确线预编译包的 35 个受保护源）。
**静态核对**：移植块与原文件逐行比对 —— `vforward`/`installed` **零差异**，
`c512` 只少 fp16 分支（3 行，正是 int8 参考侧）；两个文件 `@triton.jit` 计数 = **0**。

### 7.4 ★★ 两个必须先说清的事实

#### 事实 ①：不预热，臂必在 **5 帧**处死 —— 根因是硬编码 30 秒

首跑（17:21）**只跑了 5 帧**就失败，`block = "decoder_queue_timeout"`。

- `nr_worker.cpp:2101` 解码线程只等 `std::chrono::seconds(30)` 就要消费者腾出队列槽位；
- 原尺寸路径的**内核集与 NR256 不同**（`product/precompile/README.md:37-39` 已警告），
  首帧要现场编译几十个内核 ⇒ 远超 30 s ⇒ 解码线程先判超时。

**修法**：worker 之前先 `session.warmup(3)`（零输入、1 帧 `reset=True` + 2 帧稳态，
用零输入因为内核按**形状**特化、与像素值无关），编译搬到流水线之外。
**没有改 `nr_worker_v1.dll`** —— 改二进制会毁掉与 Sep 11 基线的可比性。
预热耗时 1.7–2.3 s（缓存已热时）。

#### 事实 ②：**864×480 到不了 1920×1080**（几何硬门）

`nr_worker.cpp:427` 的判据是**整数硬相等**，不是"接近"：

```cpp
const bool valid = ... && uint64_t(in_w)*out_h == uint64_t(in_h)*out_w;
```

| 组合 | 判据 | 结果 |
|---|---|---|
| 864×480 → 1728×960 | `864*960 = 480*1728 = 829440` | ✅ valid |
| 864×480 → **1920×1080** | `864*1080=933120` ≠ `480*1920=921600` | ❌ **invalid** |
| 864×480 → **1944×1080** | `864*1080 = 480*1944 = 933120` | ✅ valid |

运行期实证（stderr）：

```
[nr-scale] quality=101 input=864x480 output=1728x960  minimum=752x418 maximum=1728x960  valid=1
[nr-scale] quality=101 input=864x480 output=1944x1080 minimum=846x470 maximum=1944x1080 valid=1
```

源是 **1.8:1**（864/480），1920×1080 是 **16:9** ⇒ **源的比例决定了输出比例，SR 不换比例**。
要真正的 16:9 1920×1080，**必须换 16:9 的源**（如 `960×540 → 2.0× → 1920×1080`，
或 `864×486 → 2.222× → 1920×1080`）。注意 1944×1080 的输入下限 `846×470` 离 `864×480`
只差 2%，即这条臂已贴在 perf 档的边界上。

### 7.5 ★★ 测量纪律：为什么只取 ABBA 那一组

同一组臂跨租约读数**漂了 8%**，而同租约内 ABBA 交替只漂 0.2–1.0%：

| 臂 | 非 ABBA 租约 | **ABBA 租约（采信）** |
|---|---|---|
| `single864` | 47.63 / **50.36 / 50.89 / 51.49** | **47.53 / 47.62 / 47.99** |
| `srfg1944` | 56.91 / **61.79 / 58.71 / 62.22** | **56.79 / 56.56** |
| `srfg1728` | 55.61 / 55.81 | 56.39 / 56.01 / 56.14 / 55.91 |

⇒ **§7.1/§7.2 只用 ABBA 租约的读数。** 那些高值出现在我**同时在跑诊断命令**
（遍历上千个缓存目录的 `find`、反复起 Python）的租约里；`single864` 是**纯 host 侧**
的臂（无 SR/FG/depth 吸收抖动），所以对宿主争用最敏感。
**这不是"臂本身不稳"，是"测的时候机器不安静"** —— 与技能里"同一时刻只有一个 AI 测速"同一条。
**结论对漂移不敏感**：NR 要砍一半才能到 60 fps，8% 的漂移改变不了这个判断。

### 7.6 本轮新增/改动的装置

| 文件 | 作用 | 状态 |
|---|---|---|
| `fullsize_session_v1.py` | 原尺寸 NR 会话（+`warmup()`） | ✅ 243 帧跑通 |
| `run_arms_v1.py` | 一租约多臂驱动器（+全量日志、+阶段级 env、+旧键剔除） | ✅ |
| `run_fullsize_arm_v1.sh` | 租约包装（参数转发） | ✅ |
| `probe_arm_v1.py` | +`NR_FULLSIZE`、+预热、+`sr1944`/`srfg1944` 臂 | ✅ |
| `triton_host_cache_probe_v1.py` | CPU-only 宿主模块缓存键诊断 | ✅ |

**臂定义（新增）**：`single864` = `--gpu-mode sr` + `single=True` ⇒ 输入=输出=864×480、
`sr_seconds=0`、`fg=0`、`depth=0`（stderr 无 `[sr]`/`[nr-scale]` 行）⇒ **NR 隔离臂**。
`sr1944`/`srfg1944` = 输出 1944×1080 的 SR / SR+FG 对。

**顺手修掉的两个真缺陷**：
① 共享缓存 `6L4GK…`（`arch_utils`）目录**被清空**（mtime 09-20 16:13），
从 `D:/fullsize-rows-v1/r1/triton-cache` 同键目录把 30208 B 的 `.pyd` 搬回（**仅新增，未删任何东西**）；
② 驱动器原来只留 4 行 stderr，把 worker 的判决行 `[full-gpu] ... pass= block=` 截掉了
⇒ 改为**全量落盘**并直读 `native-report.json` 的 `block`。

### 7.7 诚实边界

1. **这不是游戏实测**：输入是 243 帧 h264 文件。worker 的"捕获"是
   **自捕获自己的 present**（`runtime.present_capture`），报告里 `wgc_obs: NOT_TESTED`、
   `live_wgc_obs: NOT_TESTED`。**真·游戏实测需要活捕获接入（WGC/DXGI 或 SDK 内集成），本 harness 没有。**
2. **harness 每输出帧都做 h264_qsv 编码、并解码源文件**；真实游戏链没有编码器。
   `encode_worker_wait` 仅 ≈0.8 ms/输入帧（异步、基本被吸收），所以去掉编码器大约只省 1–2 ms/输入帧，
   **不改变 §7.2 的量级**。
3. **`xefg_gpu_seconds` 是 `null`** —— 但**不是"忘了测"：SDK 根本没有这个出口**（`xefg_swapchain.h`
   无任何计时设施；`native_gpu_timers.h` 只有四个累加器；`vpl_gpu_full_fg.cpp:1822` 字面写 `null`）。
   而 `srfg` 与 `sr` 的**同租约配对**差（**+1.3663 / +1.4238 ms/输入帧**）**不是 XeFG 的 GPU 时间** ——
   它 **90.7% 落在 `present_s`** 上，而四个被计时的 GPU 段**一点没动**。**详见 §8。**
4. **`single864` 的 `blocked_on = strict_gpu_depth_input`、`gate_status = PASS_COLOR_MOTION_DEPTH_PENDING`**
   是 `single` 模式的预期状态（无 depth），不是故障。
5. **NR 隔离臂的 47.62 含解码+编码+呈现**，不是"纯 NR 内核时间"。它与 `rows` harness 的
   45.83（`materials_entry_v1`，另一套装置）同量级、可互相印证，但**两者不可直接相减**。
6. **未改产品源、未改冻结参考、未改 `nr_worker_v1.dll`、未做 git 提交。**
7. **仍未做**：真 16:9 1920×1080（需 16:9 源）；XeFG GPU 时间（P2b）→ **见 §8**；
   `NR_FULLSIZE` 在 960×540/864×486 下能否成立（当前会话硬锁 `FULLSIZE=(480,864)`）。

---

## 8. ★★★ P2b 结案：XeFG 的 GPU 时间**测不出来**，而那 1.37 ms 是 present

> 追加 2026-09-20 晚。**零新租约** —— 全部从 §7 已落盘的 `arms-summary.json` +
> `native-report.json` 读出来，外加对工具箱源码的一次静态核对。

### 8.1 先说结论

| 问题 | 答案 |
|---|---|
| XeFG 的 GPU 时间是多少？ | **测不出来，而且不是"没测"，是没有出口**（§8.2） |
| 那 `srfg − sr` 的差是什么？ | **present 阶段的宿主墙钟**，**90.7%** 落在它身上（§8.3） |
| 四个被计时的 GPU 段动了吗？ | **没动**（Δ ≤ 0.02 ms，0.7% 以内） |
| 要为它排一轮新实验吗？ | **不需要** —— 它不改变 §7.2 的预算结论（§8.5） |

### 8.2 为什么测不出来：SDK 没有出口，不是忘了测

三层证据，一层比一层硬：

1. **API 层**：`sdk/official/inc/xess_fg/xefg_swapchain.h` 里**没有任何**
   timestamp / statistics / gpu_time 设施 —— 只有 `SetPresentId` / `GetLastPresentStatus` /
   `SetLatencyReduction` / `TagFrameResource` 这类**提交与打标**接口。
2. **累加器层**：`src/gpu/native_gpu_timers.h` 的 `NativeGpuTimers` **只有四个累加器**
   （`motion_seconds` / `mask_seconds` / `depth_post_seconds` / `sr_seconds`）
   —— **结构体里根本没有 xefg 字段可填**。
3. **输出层**：`src/gpu/vpl_gpu_full_fg.cpp:1822` 把 `"xefg_gpu_seconds": null` 与
   `"openvino_gpu_model_seconds": null` **字面写在 printf 串里**
   ⇒ 这两个 `null` 是**设计**，不是缺失。

**★ 顺手查到的可用余量**：`NativeGpuTimers::init` 建的是 `slots*8` 个 timestamp，
但 `resolve` 只解析 **7 个**（point 0..6），已用 4 段
（`motion=t1−t0` / `mask=t2−t1` / `depth_post=t3−t2` / `sr=t6−t5`）
⇒ **`t4→t5` 未归因、point 7 建了却从未写入**。
**若将来真要测，不需要新建 query heap，补两个 `mark` 即可。**

**★ 但 FG 不在我们的命令列表里**：XeFG 是 **swapchain 级 API**，生成工作在
`Present` 路径内部完成。我们的 `EndQuery` 只能标自己的 list
⇒ 想在 GPU 侧夹住 XeFG，只能"present 前后在同队列各插一次标记"，
**而那测到的是 present 的墙钟跨度，不是 XeFG 的忙时**（技能 161：
墙钟时间戳原理上区分不了"忙"与"闲"）。

### 8.3 但那 +1.37 ms 是 present，不是 GPU

同租约配对（唯一可采信的口径，§7.5）：

| 租约 | `sr1728` | `srfg1728` | **Δ 墙钟** | 四个 GPU 段之和 Δ |
|---|---:|---:|---:|---:|
| `timing` | 54.2395 | 55.6058 | **+1.3663** | **−0.0039** |
| `timing-rep` | 54.3815 | 55.8053 | **+1.4238** | **−0.0223** |

⇒ **墙钟长了 1.37–1.42 ms，而四个被计时的 GPU 段一点没动。**

归属证据：`present_s` **当且仅当 `fg=243` 时非零** —— 8 个 FG 臂全落在
**1.2028–1.2744 ms**，**所有非 FG 臂全为 `0`**。逐项拆 `timing` 那一对
（`native-report.json` 的 `stages` 是 243 帧**总计秒**）：

| 项 | Δ（s，243 帧） | 占 Δ |
|---|---:|---:|
| `stages.present_s` | **+0.2937** | **90.7%** |
| `stages.copy_submit_s` | +0.0037 | 1.1% |
| `stages.record_submit_s` | +0.0072 | 2.2% |
| 其余 | +0.0191 | 5.9% |
| `stages.decode_s`（外层） | **+0.3237** | = 总 Δ 的 97.5% |

⇒ 每**输出**帧的 present 价 = `0.2937 s ÷ 485 = **0.6055 ms**`。
**`record_submit_s` 几乎不动（+0.0072 s）** ⇒ **FG 完全不影响我们自己命令列表的录制。**

⚠️ **诚实边界**：`present_s` 是宿主 `perf_counter` 量到的 **present 调用墙钟**。
按技能 161，**墙钟时间戳区分不了"忙"与"闲"** ⇒ 这 1.21 ms 里有多少是 XeFG 的 GPU 忙时、
多少是 swapchain 的宿主开销，**本装置答不了**。
`present_required_wait_count = 0` 说明没有显式必需等待，**倾向于宿主侧，但这是旁证不是证据**。

### 8.4 ★★ 顺手把非 NR 的 8.45 闭合了 —— 并纠正 §7.2 的分解

§7.2 原写"8.45 里 AI depth 占 6.7、SR 0.53、**motion 0.40**"。**这一行有两处错：**

- **motion 0.40 不该进来**：它在 NR 隔离臂 `single864` 里**本来就有**
  （那是 NR 自己的运动估计），相减时自动抵消；
- **FG 的 present 阶段 1.21 被漏掉了**。

正确分解（全部来自 §7 同一次实测的字段）：

| 项 | ms/输入帧 | 来源字段 |
|---|---:|---|
| AI depth — OpenVINO 推理 | 4.53 | `depth_inference_ms_per_input` |
| AI depth — GPU 后处理 | 2.22 | `gpu_ms_per_input.depth_post_seconds` |
| **FG — present 阶段** | **1.21** | `stage_ms_per_input.present_s` |
| SR（XeSS 消费者列表） | 0.53 | `gpu_ms_per_input.sr_seconds` |
| **合计** | **8.49** | vs 实测 **8.45** ⇒ **闭合到 0.5%** |

**★ 副产品（值得单独记）**：输出几何从 **864×480 → 1728×960（4× 像素）**，
在这本账上等于 **0** —— 8.49 里没有任何一项与像素数成比例。
（原因：NR 画布固定，长出来的只是 scaler / composite / 编码，而它们在本装置里已被吸收。）

### 8.5 预算结论不变，但杠杆排序更清楚了

| 目标输出 | 每输入帧预算 | **NR 上限** | 距现状 47.62 |
|---:|---:|---:|---:|
| **60 fps** | 33.33 ms | **24.9 ms** | −22.7（**−48%**） |
| 90 fps | 22.22 ms | 13.8 ms | −33.8（−71%） |
| 120 fps | 16.67 ms | 8.2 ms | −39.4（−83%） |

**非 NR 那 8.49 里 AI depth 占 6.75 = 79.5%** ⇒ 免掉/降档 depth，60 fps 的 NR 上限放宽到
**≈31.6 ms**（§7.2 的 ≈31.5 成立，出处从 "6.7" 改为 "6.75"）。
而 **FG 的 1.21 是 present 路径开销** —— **把 XeFG 的推理做得更快也不会让它变小**
⇒ 它不是优化对象，而是**需要被预算接纳的固定项**。

### 8.6 ★ 一个反直觉但重要的读法：FG 不是成本，它是"买输出帧率"的手段

| 臂类 | `ms_per_input_frame` | `ms_per_output_frame` |
|---|---:|---:|
| 非 FG（243 出帧） | 47.53 – 54.38 | 同左 |
| **FG（485 出帧）** | **55.91 – 56.91**（安静租约） | **27.86 – 28.51** |

⇒ **FG 把每输出帧成本降到 ≈0.51×（对 `sr1728`）/ ≈0.59×（对 `single864`）**，
因为重的 NR + SR 每输入帧只付一次，多出来的那一帧只花 **≈1.37 ms**（其中 1.21 是 present）。

**⇒ 别把 FG 那 1.37 ms 当"要砍的对象"。** 它是 2× 帧率的门票价，
而且这张票把每输出帧成本砍了近一半。

### 8.7 仍然没做 / 不能做

1. **XeFG 的 GPU 忙时**：本 harness 答不了（无 API、FG 不在我们的命令列表里、
   present 墙钟 ≠ 忙时）。要测需要 **(a)** 在 present 前后插标记 + **同时**用 profiler 的
   设备忙时做交叉校验，或 **(b)** 同一输入下 FG on/off 的**设备忙时**差分。
   **两条都要新租约，且 (a) 要动 DLL** ⇒ 列为待定，**不进本轮结论**。
2. **`openvino_gpu_model_seconds` 同样是字面 `null`** ⇒ 那 4.53 ms 的 depth 推理
   **也只有宿主墙钟，没有 GPU 时间**（同 §8.2 的三层证据）。
3. §7.7 的其余各条（非游戏实测、含编码器、未改产品源、未改冻结参考等）**依然全部成立**。
