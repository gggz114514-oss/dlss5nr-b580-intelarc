# corr_strip_v1 结案报告

**日期**：2026-09-23 ｜ **状态**：**结案（否证）** ｜ **原始读数**：`D:/corr-strip-v1/r9/`

---

## 0 一句话结论

**「切除为对齐 4060 而写的保一致层」这条线索否证**：换成原生实现的 nat 臂比基线**慢 +1.7835 ms**
（阈值 −1.0 ms，效应/不确定度 18.8×），而 P2（alg 臂）在数学上不可能翻盘。
保一致层**不是性能税，它是一个更优实现** —— 它不改画面，却比替代方案快 1.78 ms。

---

## 1 先说一件必须纠正的事：本轮入场时的前提不成立

本轮的入场指令是「直接开 corr_strip，**但要先修掉尺子上那处注定在该臂上失败的断言**」。
这句话描述的问题**在本轮开工之前就已经不存在了**：

| 时间 | 轮次 | 当时 nat 臂的状态 | 真因 |
|---|---|---|---|
| 09-22 15:00–17:15 | r3–r7 | `passed` 假、`close_error` | 收尾期 `Fork.verify_restored` 的归约断言（替换名与别的装置 bindings 同名） |
| 09-22 17:30 | r8 | `passed=True, error=None` | — |
| 09-22 17:35 | r9 | 五臂**全部** `passed=True, error=None` | — |

修法是 `NR_CORR_RESTORE_BEFORE_CLOSE`（在 `stack.close` 之前把替换掉的全局名还原），
它已在 r8/r9 生效。旧稿 `PARALLEL-COMPILE.md` §6 把 nat 臂失败归因给「派发计数断言」，
**该归因是错的，已作废。**

⇒ **结论：r9 早已产出了有效判决。本轮不需要修尺子，只需要把 r9 的判决读出来并结案。**
（另有一处尺子缺陷是**真的**且仍未修：`gpu_kernels` 按**裸内核名**聚合，同名跨模块合并 ——
本轮 `_kernel` 一行混了 8 个模块、`_project` 混了 3 个。它影响逐内核归因的干净度，
但不影响 P1 这个「整臂汇总差」的判决。）

---

## 2 判决（PLAN §5 的判据）

### 2.1 P1（主判据）：判负

| 项 | 值 |
|---|---|
| 臂设计 | ABBA 四臂（`01-off` / `02-nat` / `03-off` / `04-nat`），另加预热臂 `00-manifest-nat` |
| 帧数 | 每臂 3 帧，off 组 6 帧 / nat 组 6 帧 |
| off 设备忙时均值 | 39.1220 ms |
| nat 设备忙时均值 | 40.9055 ms |
| **Δ(nat − off)** | **+1.7835 ms** |
| 阈值 | −1.0 ms |
| 判定 | **换原语无用**（且方向是反的：更慢） |

**离散度**：σ(逐帧配对差) = **0.0950 ms**，逐帧差范围 [1.7028, 1.8881]，**三项全为正**
（index 9: +1.7596｜10: +1.7028｜11: +1.8881）。效应 / 不确定度 = **18.8×**。

### 2.2 Q1 形状自检：通过

同 index 下 nat 与 off 的 `gpu_instances` **逐帧相等** ⇒ 替换没有改变派发形状，速度差可信。
（这是必要的自检：若替换改变了内核数量，速度差就无法归因到「实现差异」上。）

### 2.3 P2（次判据）：数学上已被前置决定必败

alg 臂本轮没有运行，但**它不可能翻盘**：alg 在 nat 之上只能再动 `_normalize` / `normalize_c32`，
而 Census 里整个 `_normalize` 只值 **0.0503 ms**（nat 下实测 −0.0115 ms）。
即使 alg 把这一项**完全清零**，也只会落在 ≈ **+1.77 ms**，离 −3.0 ms 的阈值差着一个数量级。

⇒ **不需要为了 P2 再花 GPU 时间。**

---

## 3 归因：为什么换原生反而更慢

### 3.1 差集中在 4 个模块，占 91%

| 模块 | off ms | nat ms | Δ ms | off 次 | nat 次 |
|---|---:|---:|---:|---:|---:|
| `_kernel` | 13.8510 | 14.9357 | **+1.0847** | 299 | 299 |
| `_pack` | 1.9969 | 2.2158 | **+0.2189** | 52 | 52 |
| `_pairs` | 3.8752 | 4.0545 | **+0.1793** | 36 | 36 |
| `_project` | 3.2851 | 3.4281 | **+0.1430** | 88 | 88 |
| 四项合计 | | | **+1.626** | | |
| `_normalize` | 0.0503 | 0.0388 | −0.0115 | 16 | 16 |

四个模块合计 **+1.626 ms**，占全部效应 +1.7835 ms 的 **91%**；而**启动次数逐帧完全相同**
⇒ 差异来自**内核体**，不来自派发。这四项恰好就是承载修正原语的模块。

### 3.2 机制：这条链上的 fp8 转换不是硬件指令

`tt.fp_to_fp` → fp8 在 Intel 后端的 XPU 链上**不被映射成一条硬件 cvt**，而是展开成约 20 条软件序列：
`llvm.nearbyint.f16`（舍入库调用）+ `llvm.usub.sat.i16` + `llvm.umin.i16` + `uitofp` / `fptoui` +
`fmul half` —— 逐 lane 标量、跨整数与浮点管道、走 16 位通道。

而原来的复刻层全程用 **i32** 加原生 `umin` / `umax` / `smax.i32`。

⇒ **复刻层的指令条数更多（LLIR 208 条 vs 164 条），却更快。**
**指令条数不是成本，指令种类才是。**
（同源证据：先前 A 臂「INT8 XMX」实测 +4.96 ms —— B580 上整数 ALU 比硬件 fp8 cvt 更快。）

### 3.3 度量陷阱（重要，别再踩）

| 度量 | 本例读数 | 能否预测速度 |
|---|---|---|
| TTIR 字符数 | **−80%**（13823 → 2816） | 不能，方向是反的 |
| LLIR 条数 | −21% | 不能，条数相同也可能快慢相反 |
| LLIR 指令**种类** | i32 逻辑 + 原生 min / max | 能，与慢的方向一致 |

---

## 4 证据与可复现

| 项 | 位置 |
|---|---|
| 报告（markdown / json） | `D:/corr-strip-v1/r9/corr_strip_report.md` / `.json` |
| 五臂原始留痕 | `D:/corr-strip-v1/r9/00-manifest-nat/` … `04-nat/` 各自的 `corr_strip.json` |
| 逐帧产物 | `D:/corr-strip-v1/r9/0X-*/run/validation.json`、`run/frames/*.npz` |
| 重算（不花 GPU） | `python reference/corr_strip_v1/corr_strip_report_v1.py --root D:/corr-strip-v1/r9 --out <新目录>` |

**画面读数（R1）**：报告器 §6 里的 `rmse` / `max_abs` 是**每帧对冻结参考**的读数
（`validation.json` 的 `exact_validation` / `baseline_validation`），**不是 nat 与 off 之间的比对** ——
两臂都不等于冻结参考（`byte_equal` 0/24），因为冻结参考来自更早的血统。
`validation.json` 对此有明文：`byte_assertion` = "recorded against the supplied frozen baseline; not a failure"，且 `frozen_reference` = False ⇒ 0/24 **是预期内的**，不是异常。

**真正该看的是「nat 与 off 彼此是否逐字节相同」，答案是：逐字节相同。**
`validation.json` 每帧都带输出 `sha256`，逐帧比对如下：

| 比对 | 输入 sha256 一致 | 输出 sha256 一致 |
|---|---|---|
| `01-off` vs `02-nat` | 12 / 12 | **12 / 12** |
| `03-off` vs `04-nat` | 12 / 12 | **12 / 12** |

⇒ **两对独立的 off/nat 臂、12 帧全部输出逐字节相同**，`rmse` / `max_abs` 也逐帧全等。
**这坐实了「保一致层替换完全不改画面」**：既有 1.7835 ms 的代价，又没有任何画面收益，
PLAN §0 裁定 A 下不判通过/不通过，但这一条直接决定了它**没有任何落地理由**。

**臂有效性**：五臂 `passed` 全为 `True`、`error` 为空；01/02/03/04 的 `cache_mode` 均为 `disk_only`
（编译没有混进计时窗）；nat 臂替换计数 130，off 臂 0。
`validation.json` 里还带着 `fs_guard.installed = True` —— 即 r9 的运行**装着**那个守卫。

---

## 5 顺带修正的一处留痕缺陷

r9 的 nat 臂在 `corr_strip.json` 里写着 `hardware_cvt: 'unavailable, fell back'`，
看起来像「探针失败、回落到了另一套整数模拟」。**这是误报 —— 实际装进去的就是 cvt 版本。**
（TTIR 里能查到 `tt.fp_to_fp ... -> f8E4M3FN`。）

**代码缺陷**：`corr_strip_nat_v1.install()` 的顺序是
`natives, info = _make_natives()`（此步已把 `natives['_round_fp8_half']` 绑成 cvt）
→ `_select_round_fp8_half(info)`（**只重绑模块全局名，不更新 bounds 字典**）
→ `d[name] = natives[name]` ⇒ **无论探测结果如何都装 cvt**。

**已修**：在 `_select_round_fp8_half()` 之后回写 `natives['_round_fp8_half']`，
并新增留痕字段 `round_fp8_half_installed`，使留痕与实现强制一致。

**对结论的影响**：**没有影响，反而是好消息。** 正因实际装的是 cvt，
r9 测的**就是**目标杠杆（「硬件 cvt 替掉整数复刻」），而不是「拿一套整数模拟换另一套」。
P1 的判决因此站得更稳。

⚠️ **注意**：上面这处修改改变了模块行为（修后探测失败会**真的**回落到 fallback）。
**r9 的数字来自修改前的版本**，引用 r9 时必须带这句话。

---

## 6 本轮的一次事故与完整恢复（必须记录）

### 6.1 事故

2026-09-23 **00:56:24 – 01:07:33** 之间，**D:/corr-strip-v1 全树 13,695 条被移入回收站**
（`D:/$RECYCLE.BIN/S-1-5-21-3146757892-1026985121-1119832181-1001/`）。

**根因是我的疏忽**：新写的驱动 `cvt_probe_run_v1.sh` 同时违反了**两条已有纪律** ——
① **没有装 rows_fscache_guard_v1**；② 把 `TRITON_CACHE_DIR` 设在了**交付物树内部**
（`D:/corr-strip-v1/cache-cvtprobe`）。
于是 Triton `FileCacheManager.put()` 末尾的 `os.removedirs(temp_dir)` 从
`<CACHE>/<key>/tmp.pid_*` 向上剪空父目录，而**本卷 rmdir 不抛错、直接递归删** ⇒ 一路上溯到树根。
（这与 2026-09-22 吞掉 `E:` 上 `.git`、以及吞掉 `D:/fullsize-rows-v1` 是同一条机制，是**第三次**。）

### 6.2 恢复（无损，已完成）

| 步骤 | 结果 |
|---|---|
| 解析回收站 `$I` 记录 | 13,695 条，**实体缺失 0 条**（全部可还原） |
| 还原 r1–r9 子树 | 命中 501 条记录 ⇒ 移回 **60 目录 + 410 文件**、合并进已存在目录 57 处、**跳过 0、坏记录 0** |
| 逐文件尺寸对表 | 抽查 393 个文件，与 `$I` 记录的原尺寸**不符 0 个** |
| 文件数逐臂对账 | r1–r9 = 20/24/76/29/14/15/80/81/81，与回收站记录的实体文件数**逐项完全一致** |
| 语义级复核 | 在还原数据上**重跑报告器**，§2/§3/§7 的**每一个数字与判定逐字相同** |
| 全部判据 | **无损** |

还原用 `os.rename`（同卷元数据搬运，逐字节不变），**全程没有调用任何 rmdir / rmtree**，
冲突一律 skip-if-exists，绝不覆盖。脚本：`reference/corr_strip_v1/restore_from_recycle_v1.py`。

**仍在回收站、未取回**：`cache2`–`cache5` 共 **13,104 条** Triton 编译缓存（可再生，取回无价值）。

### 6.3 新增的两道防线（都已实测）

1. **进程内硬断言**（`corr_strip_cvt_probe_v1.py` 顶部，在任何 `import triton` 之前）：
   装 `rows_fscache_guard_v1` 并**自检** `os.removedirs.__name__ == 'removedirs_leaf_only'`；
   另加 `_assert_cache_outside_trees()`，命中交付物树即拒跑。
2. **驱动层 shell 断言**（`cvt_probe_run_v1.sh`）：`_check_tree` 校验 `CACHE` 与 `TMP/TEMP`
   不在任何交付物树内；缓存与临时目录分别改到树外的 `D:/corr-strip-cache`、`D:/corr-strip-tmp`。

另修了导致前两次微基准**整轮作废**的坑：运行时扩展必须**整份播种**（`spirv_utils` /
`extension_utils_impl` / `arch_utils` **三个**），只播 `spirv_utils` 仍然失败；
默认缓存只有 156 KB，全量 `cp -rn` 即可。

---

## 7 本轮的全部写操作清单

**产品源（nr-b580-int8/）零改动。** 全部改动都在实验目录与记忆里：

| 文件 | 动作 |
|---|---|
| `reference/corr_strip_v1/corr_strip_cvt_probe_v1.py` | 加守卫 + 树外缓存硬断言 + `import os` |
| `reference/corr_strip_v1/cvt_probe_run_v1.sh` | 树外缓存/临时目录 + `_check_tree` + 全量播种 + 默认 ROOT 改 r3 |
| `reference/corr_strip_v1/corr_strip_nat_v1.py` | 修 `natives` 回写 + 新增留痕字段 |
| `reference/corr_strip_v1/corr_strip_report_v1.py` | 臂自动发现（alg 可选、不再因缺臂中止出报告）。**副作用**：§1 有效性表由 5 臂变 6 臂（多列 `00-manifest-off`）；§2/§3/§7 的数字与判定**逐字不变**（已用重算对比验证） |
| `reference/corr_strip_v1/restore_from_recycle_v1.py` | 新增（回收站无损还原，默认 dry-run） |
| `reference/corr_strip_v1/to_recycle_v1.py` | 新增（移入回收站，不看 rc 看源路径是否消失） |
| `reference/corr_strip_v1/scan_loss_v1.py` | 新增（只读损失面盘点） |
| `D:/corr-strip-v1/r9/**` | 回收站还原；测试残留 `regress-out.md` 已再移入回收站 |

---

## 8 待办与留给后面的东西

1. ~~确认报告 §6 那张表「拿谁跟谁比」~~ **已解决**（见 §4）：比较对象是冻结基线；nat 与 off 之间已由 12/12 帧输出 sha256 逐帧比对确认逐字节相同。
2. **gpu_kernels 按裸内核名聚合**这一处尺子仍未修 ⇒ 逐内核归因要按 `模块.内核` 记账。
3. **corr_strip_v1 可归档。** 若将来要重跑，用修好的 `cvt_probe_run_v1.sh`
   （它会拒跑树内缓存）；微基准脚本 `corr_strip_cvt_probe_v1.py` 含 noop 基线，
   可回答「这些内核是访存受限还是 ALU 受限」—— 但 P1 已结案，**不建议再花 GPU 时间**。
4. **值得单独立案的新线索**：`to(float8e4nv)` 是这条链上的**慢路径**。Triton 层只有
   `nr_backend/triton_fp8.py` 一处用它（已被复刻层覆盖）；但 `nr_backend/pre_mlp.py` 用的是
   torch 的 `float8_e4m3fn` 转换，落到设备侧是那批 `_ZTSN2at6native3xpu...` 内核 ——
   **若它在热路径上，可能正在付同样的代价。**
