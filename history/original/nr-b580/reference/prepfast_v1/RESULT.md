# 第七层 `prepfast_v1`：把层 1 自己的准备期做透

**日期**：2026-09-17 ｜ **轮次目录**：`E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/prepfast_v1/`
**数据**：`D:/prepfast-v1/{r1,r2,split,mdgc*,guard,guardcost,...}` —— ⚠️ **2026-09-17 20:45 已整目录清除**
（用户要求「只留结论和脚本」）。本文件里的数字即最终留档；**要复核请按 §8 / §10.9 的命令重跑**（命令会自动重建目录）。
**一句话**：在层 1+2 之上再削 **4.545 ms（1.0585×）**，**243 帧逐字节不变**；其中 `md`（跳过 `launch_metadata`）
一项占 2.85 ms。**五刀里建议只留三刀** —— `keyof` 实测为零，`stream` 在噪声内且是唯一带假设的一刀。

> ### ⚠️ 当日补测：`md` 那 2.85 ms 里只有约 **0.8 ms** 是产品级收益
>
> 见 **§10**。A/B 的测量车（`materials_entry_v1`）每帧套
> `rows_scopes_v1.dispatch_guard(VARIANT)`，它在帧内把 `CompiledKernel.launch_metadata`
> 换成 `checked`；`md` 刀一开就**连这层包装一起绕过**。该包装在**产品树里 0 处引用**
> ⇒ 不是产品的一部分。**修正后：五刀的产品级总收益 ≈ 3.0–3.3 ms（≈1.04×），不是 4.545 ms。**
> 其余四把刀（`launch`/`book`/`stream`/`keyof`）不受影响，数字照旧。

---

## 1. 靶子从哪来

`prepleft_v1` 的探针把 `HostFastCuts._run` 换成逐字复刻但带打点的版本，在 3 帧 / 2717 笔真启动上
切了 11 段。本轮逐项对着它改，**每一项单独一臂**，好让「哪一项没生效」可判。

| 开关 | 改什么 | 机制 |
|---|---|---|
| `keyof` | 逐参 Python 循环 → 每计划 `exec` 出一段**直线代码** | 判别式逐字不变，只去掉 `FOR_ITER`+`LIST_APPEND` 的每参固定税 |
| `launch` | ① `grid` 不可调用时免建 `dict(zip(names,args))`；② `kernel.run`/`.function`/`.packed_metadata` 存进记忆条目；③ 缓存 `driver.active` | ①`values` 本来就是 `args`；②三者在 `_init_handles()` 之后恒等 |
| `stream` | `(device, stream)` 缓存，**每个帧边界失效一次** | 以 `torch.xpu.synchronize` 为帧边界 |
| `md` | hook 链为空时**不调** `kernel.launch_metadata`，直接传 `None` | 见 §2 |
| `book` | ① 计划单态内联缓存；② 抽样改倒计数；③ 去掉 `hash(key)`+`memo.get(key)` 的重复哈希 | 纯记账 |

### 2. `md` 的机制：Triton 里那个「提前返回」从来不生效

`triton/compiler/compiler.py:512-514`：

```python
def launch_metadata(self, grid, stream, *args):
    if knobs.runtime.launch_enter_hook is None:
        return None
```

看上去「没装 hook ⇒ 零成本」。但 `triton/knobs.py:515` 的类属性默认值是

```python
launch_enter_hook: HookChain[LaunchHook] = HookChain()
```

—— **一个实例，永远不是 `None`**。所以这个提前返回**在生产里从不成立**，每笔启动都会：
取值 hook 链 → `self._init_handles()`（`module is not None` 早返回）→
**建 `LazyDict({"name":…, "function":…, "stream":…})`（1 dict + 1 LazyDict + 1 list，三次分配）**
→ `isinstance(self.src, ASTSource)` → `self.src.fn.launch_metadata is None` → 返回。

`launch_metadata` 的**唯一消费者是启动 hook**（proton 之类）。本产品从不装 hook
（实测 `hook_enter_calls=0`、`hook_exit_calls=0`、`hook_enter_is_none=false`），
所以只要链为空，传 `None` 就是等价的。

**三重证据**：
1. **冒烟（`launchmd_smoke_v1.py`）**：同一个内核，`manual_run(want_metadata=True)` 与
   `want_metadata=False` 的 `torch.equal` 为 **True**，且都等于走原版路径的结果。
2. **分解探针（`launchmd_split_v1.py`）**：三变体
   `build_pass`（构造并传）/ `build_drop`（构造后丢掉，传 `None`）/ `skip_both`（不构造）
   的输出**逐位相同**（`build_pass_vs_skip_both=true`、`build_drop_vs_skip_both=true`）。
3. **端到端**：`md` 臂 **243/243 帧逐字节等于冻结参考**。

---

## 3. 协议与证据

照抄 `hostfast_v1` / `fp8fast_v1`（唯一带逐字节门的计时轮）：

* **回文 A/B，8 个 tag，16 轮**（每个 tag 在两个半程各出现一次；`off` 在两端，`all` 在中缝）：
  `ab1` = `off base keyof launch stream md book all` ／ `ab2` = `all book md stream launch keyof base off`
* **基线是 `base` = `run,getitem`（层1+层2）而不是 `off`** —— 本轮靶子是层 1 自己的准备期，
  按项目规则「该报的数永远是**上一层之上**那一行」，边际收益必须相对当前已知最好来报。
  `off` 只作绝对锚点。

  > **⚠️⚠️ 2026-09-17 22:50 更正：上面「（层1+层2）」这个标签是错的 —— `base` 只是「层1」。**
  >
  > `run,getitem` 是 **`HOSTFAST_CUTS`** 的取值（见 `hostfast_v1`），即**层1 = `hostfast`**。
  > **本文件自己有两处自证**：
  > ① §4 的交叉校验 —— `base` 相对 `off` 省 **1.278 ms**，而 `merge_v1` 的 `host` 臂
  >    （**同一配置 `HOSTFAST_CUTS=run,getitem`**）实测省 **1.216 ms**，两轮复现到 0.06 ms 以内。
  >    **若 `base` 含层2，它应该省 ~7.8 ms，而不是 1.28 ms。**
  > ② §7 的「还欠的」原话 —— *"本轮只测了「五刀在层1 之上」的边际。要得到产品级数字，
  >    还需与 `merge_v1` 的 `sync` 层…"* —— 明确说层2 **没装**。
  >
  > ⇒ **`base` = 层1（82.2459 ms）**，**`all` = 层1 + 五刀（77.7008 ms）**，**不含层2**。
  > ⇒ §4 表格里 `base`（层1+层2）/`all`（层1+层2+五刀）两行的括号说明，以及 §0 的表述，**均以此更正为准**。
  > ⇒ **「层1+层2+五刀」的合成值从未在同一次 session 里测过**（需新开一轮）。
  > ⇒ 对外引用时写「**层1 + 五刀**」，不要写「层7 之后的全量」。
  >
  > 另：`base` 82.2459 vs `merge_v1` 的 `host` 81.5446 相差 **0.70 ms** —— 正是本机 session 偏移量级
  > （0.6–1.1 ms），两者互证。
* **字节门**：两侧都对 `D:/fullsize-rows-v1/r1/full-both/validation.json`（243 帧冻结参考）逐帧比。
  **16/16 轮 `identical_frames = 243`** ⇒ 五刀全部不改字节。
* **闸门（`--gate`，每个开关都要有可观测证据）**：**16/16 PASS**。
  `key_check=96/0`（每轮：生成键 vs 循环键在真实实参上对 96 次、零失败，4 轮共 384 次）、
  `stream_generation=4135`（每轮：帧边界失效确实在跑）、
  每一臂的 `flags` 与请求的开关**逐项相同**。
* **计数不变量**：`base` 与 `all` 的 `hits=553212 / misses=828 / launch=461434` **逐位相同**
  ⇒ 五刀没有改变缓存行为（不是"少算了"，是"少付了"）。
* **耗时口径**：分进程、4 帧热身、统计帧 4..N、计时期禁 JIT。
* **结论表用配对口径**（先在 session 内配对再跨 session 平均），因为本机有 session 级偏移。

---

## 4. 结果（★ 配对口径，每臂 2/2 session 完整，符号全部一致）

| tag | 相对 `base` 省 (ms) | r1 | r2 | 判读 |
|---|---:|---:|---:|---|
| `md` | **+2.846** | +3.428 | +2.263 | **最大项，两 session 同号** |
| `launch` | **+1.249** | +1.183 | +1.315 | **最稳的一项（离散 0.13 ms）** |
| `book` | +0.562 | +0.950 | +0.175 | 同号但离散 0.78 ms |
| `stream` | +0.262 | +0.085 | +0.439 | 同号，但一臂 0.085 **在噪声内** |
| `keyof` | +0.135 | +0.011 | +0.260 | **≈ 0**（一臂 0.011 明确是零） |
| **`all`** | **+4.545** | **+4.511** | **+4.579** | **两 session 相差 0.068 ms（1.5%）** |

**可加性**：五项边际之和 5.054 ms ｜ 五项全开实测 4.545 ms ｜ 偏差 **0.509 ms（+11.2%）**
⇒ 有轻度重叠（`launch` 与 `md` 打在同一段 `_launch` 上），但**五项近似独立**。

**绝对帧墙**（中位帧墙的均值，帧 4..243）：

| tag | 帧墙 (ms) | vs `off` |
|---|---:|---:|
| `off`（产品基线，一把不装） | 83.5241 | — |
| `base`（层1+层2） | 82.2459 | **1.0155×**（省 1.278 ms） |
| **`all`（层1+层2+五刀）** | **77.7008** | **1.0749×**（省 5.823 ms） |

⇒ `all` = **12.87 fps**（`off` = 11.97 fps）。

**交叉校验**：`base` 相对 `off` 省 **1.278 ms**，`merge_v1` 的 `host` 臂（同一配置
`HOSTFAST_CUTS=run,getitem`）实测省 **1.216 ms** —— 两轮独立复现到 0.06 ms 以内。

---

## 5. ★ 未解矛盾：探针的**差值**也不可外推（2.9×）

`launchmd_split_v1.py` 用一个 **15 个运行时实参**的内核（与 `fast_matrices_v3._matmul` 同形）
把 `md` 那一刀切成两半（交替取样，每变体 4 组取中位）：

| 变体 | µs/笔 |
|---|---:|
| `build_pass`（构造 + 传递） | 11.0655 |
| `build_drop`（构造后丢掉） | 10.7130 |
| `skip_both`（不构造） | 10.0415 |

    「传递」的代价（C++ 侧） = 0.353 µs
    「构造」的代价（Python 侧）= 0.671 µs
    合计                     = 1.024 µs/笔

**但端到端实测是 3.00 µs/笔**（`md` 臂配对均值 2.846 ms ÷ 949.5 笔/帧；逐 session 是
3.61 / 2.38 µs），**差 2.9 倍**。
而且方向还是反的：探针用的内核有 **15** 个实参，产品里按调用数加权平均只有 **~7.5** 个
（`triton_fp8._kernel` 383 次/帧只有 4 个实参）⇒ 探针本该**高估**，实际却**低估**。

**已排除的解释**：
* 不是「C++ 侧每笔调一次 hook」—— 探针的 `build_drop` 已经传 `None`，hook 同样不会被调，
  而这一段的代价只有 0.353 µs，撑不起 2.6 µs 的缺口。
* 不是「少算了」—— 字节门 243/243、`ffn_calls` 断言（c512=16/vit=8）全过、计数逐位相同。

**仍然成立的候选解释**（未验，按可疑度排序）：
1. **分配压力 → GC**：`launch_metadata` 每笔多 3 次分配（dict + LazyDict + list）
   × 949.5 笔/帧 = **2849 次/帧**。真实管线里的存活对象远多于探针进程，
   gen-0 回收的遍历成本可能被放大 —— 探针测不到这一项。
2. **探针的测量上下文与产品不同**：探针是 200 次背靠背启动的紧循环（每笔基线 10 µs，
   其中大部分是启动器对 15 个张量的编组）；产品里 949.5 笔散布在 82 ms 的帧里，
   中间夹着别的 host 工作与设备等待 —— 边际 Python 开销能否被设备等待吸收，
   两种上下文不一样。**探针里能被吸收的，产品里未必。**
3. 探针内核的实参模式（15 个同 dtype、同形状、连续）与真实内核不同。

⇒ **可复用的判据（本轮新固化）**：
**探针的「同口径相对占比」可以信；探针里两个变体之间的「差值」不能外推成 ms/帧。**
上一轮的教训是「名义值要 ÷3」；这一轮更硬：**÷3 那个折算率本身也不是常数**
（同一轮里 `md` 是 2.9×，`book` 反而是 0.45×）。**只有带字节门的端到端 A/B 能定 ms。**

---

## 6. 建议：留三刀，砍两刀

| 决定 | 项 | 理由 |
|---|---|---|
| **留** | `md` +2.846（**产品级 ≈0.83**，见 §10） | 两 session 同号，机制已用三重证据钉死；**但产品级只是第三大项，不是最大项** |
| **留** | `launch` +1.249 | 最稳的一项（离散 0.13 ms），改动最保守（只是不建一个本来就多余的 dict） |
| **留** | `book` +0.562 | 同号；纯记账，无风险 |
| **砍** | `keyof` +0.135 | **实测为零**。⇒ 结论：键构造的成本在 `a.dtype` / `a.data_ptr()` 这两个 C 调用上，**不在 Python 循环开销上**。砍掉它同时消掉整轮唯一的 `exec` 生成代码 |
| **砍** | `stream` +0.262 | 一臂在噪声内；且它是**唯一带假设**的一刀（"同一帧内流恒定"），代价是 monkeypatch `torch.xpu.synchronize`。收益 0.26 ms 换一个假设，不值 |

砍掉两刀后按 `md+launch+book = 4.657 ms` 计（实测 `all` 是 4.545 ms），
**放弃 0.11 ms 换掉两个最脆的部件**。

**风险与红线**：
* 三刀都只替换 `JITFunction.run` 一个属性，`uninstall()` 原样放回；不写文件、不改产品、
  不改冻结参考、不改工具链。
* `md` 带 hook 非空守卫（`if enter.calls or exit_.calls`）⇒ 将来装了 proton 也不会被静默改行为。
* **`md` 依赖一个版本相关事实**（`knobs.runtime.launch_enter_hook` 是 `HookChain` 实例而非 `None`）。
  升级 Triton 后必须**重跑冒烟**，不能假设它仍成立。

---

## 7. 本轮修掉的真实错误（都发生在正式跑之前）

1. **`_Plan2` 的 `__slots__` 与 `__init__` 那处编辑没落盘** ⇒ 运行期
   `AttributeError: '_Plan2' object has no attribute 'check_left'`。
   紧接着 `report()` 那处编辑也没落盘 ⇒ `KeyError: 'env'`，看起来像**两个 bug，其实是一个**。
   ⇒ 已固化为规矩：**每处 `Edit` 后立刻 grep 该编辑独有的标记**。
2. **`RUN` 目录没建** ⇒ bash 重定向先失败（rc=1），错误信息长得像「臂失败了」，实际连进程都没起来。
   ⇒ 运行脚本自己 `mkdir -p "$RUN"`。
3. **租约调用的 rc 被 `| tail` 换掉** ⇒ 内层 `rc=1`、外层报 `LEASE_RC=0`，失败被读成通过。
   ⇒ 改重定向到文件。
4. **键自检写在 `try` 外面** ⇒ 不可哈希实参（设计好的回落信号）会在前 3 次调用**崩溃**而不是回落。
   ⇒ 自检移入 `try`，且失败抛 `RuntimeError`（不被该 `except` 捕获）。

前 3 条已写进技能 `triton-xpu-frame-attribution` 的陷阱 **76–78**，第 4 条是 **80**。

---

## 8. 产物与复现

| 文件 | 作用 |
|---|---|
| `prepfast_cuts_v1.py` | 五刀的实现（`PREPFAST_CUTS` 开关，每项独立） |
| `prepfast_entry_v1.py` | 入口（序幕 + 装刀 + **每开关闸门**） |
| `prepfast_ab_run_v1.sh` / `prepfast_queue_v1.sh` | 回文 A/B + 租约排队 |
| `prepfast_ab_summary_v1.py` | 汇总（残缺闸门 + 配对口径 + 可加性） |
| `prepfast_keyof_selftest_v1.py` | **离线**单元测试：生成键 ≡ 循环键（7 例全过，0 GPU） |
| `launchmd_smoke_v1.py` / `launchmd_split_v1.py` | `launch_metadata` 的正确性冒烟与成本分解 |
| `AB_SUMMARY.txt` | 完整汇总输出 |

```bash
PY="E:/ComfyUI-aki-v3-IntelArc_20260722/ComfyUI-aki-v3-IntelArc/python/python.exe"
cd E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/prepfast_v1

"$PY" -X utf8 prepfast_keyof_selftest_v1.py                 # 离线，0 GPU
RUN=D:/prepfast-v1/r1 bash prepfast_queue_v1.sh ab1         # 约 21 分钟（含租约）
RUN=D:/prepfast-v1/r2 bash prepfast_queue_v1.sh ab2
"$PY" -X utf8 prepfast_ab_summary_v1.py --run D:/prepfast-v1/r1 --run D:/prepfast-v1/r2 --mode ab
```

**没出对比视频**：16 轮全部 243/243 逐字节等于冻结参考 ⇒ **没有画面差异需要人工审核**。

## 9. 还欠的

* ~~§5 那个 2.9× 缺口未定位~~ ⇒ **当日已定位并结案，见 §10**（是测量脚手架 `dispatch_guard`，不是 GC）。
* **§10.7 那个新的 2.3×**：离线单价 × 笔数（0.548 ms/帧）vs 产品实测（1.23 ms/帧）。
  最直接的下一步：在产品里给 `checked` 加自计时（`perf_counter` 累计），
  直接读出它的真实每笔成本，而不是靠外部差分。
* 本轮只测了「五刀在层1 之上」的边际。要得到产品级数字，还需与 `merge_v1` 的 `sync` 层、
  `selectfast_v1` 的 `select` 层合并测一次（三层同开）。
* ~~帧墙 77.70 ms 与「四分区 68.50 / 探针 77.36 / 帧墙 82.00」那个老口径矛盾仍未定位。~~
  ⇒ **2026-09-17 23:05 已解，不是矛盾**：①「68.50」是**笔误** —— 四分区合计实为 **82.2083**（`FRAME_BUDGET_REPORT.md:367,375` 原话「四项之和 = Phase 0b 自己的帧墙」），68.50 疑似 `host 提交 68.9837` 抄丢一位；
  ② 四分区是**排他分解**（四项之和 **=** 帧墙）⇒ **分解项不能与总量相加**，所以「77.36 + 68.50 ≫ 82.00」是重复计数造出来的伪矛盾。
  详见 `handover_v1/HANDOVER.md` §7.3 缺口 7。

---

## 10. 当日补测：§5 那个 2.9× 缺口已定位 —— **是测量脚手架，不是产品**

### 10.1 结论先行

`md` 刀在 A/B 里量到的 **2.846 ms/帧**里，**只有约 0.8 ms 是产品级收益**。

原因：A/B 的测量车 `materials_entry_v1.one_frame` 每帧套一层
`rows_scopes_v1.dispatch_guard(VARIANT)`（`:242`），而它干的第一件事就是
**在帧内把 `CompiledKernel.launch_metadata` 换成 `checked`**（`rows_scopes_v1.py:206,239`）。
`md` 刀一开，`launch_metadata` 整个不被调用 ⇒ **这层包装也被一并绕过**。

`grep -rl dispatch_guard nr-b580-int8/` = **0 个文件**（`rows_scopes_v1.py` 只存在于
`reference/fullsize_rows_v1/`）⇒ **它不是产品的一部分，是测量脚手架。**

### 10.2 为什么两个探针都复现不出（而这是对的）

| 探针 | 内核模块 | 走的路径 | 给的 md 代价 |
|---|---|---|---:|
| `launchmd_split_v1.py` | `__main__` | 裸 `launch_metadata` | 1.024 µs/笔 |
| `launchmd_guard_v1.py`（套**真** guard） | `__main__` | `checked` 的**通用路径** | 0.973 µs/笔 |
| `guard_cost_offline_v1.py`（真 guard + 假 kernel） | `fast_matrices_v3` | 通用路径 | **0.422 µs/笔** |
| 同上 | `c512_int8_ffn_rows_v1` | 通用路径 **+ `ROWS_EXTRA` 分支** | **1.080 µs/笔** |

`checked` 里有两条分支，探针的裸内核**只走得到第一条**：

```python
if family is not None:                              # MODULE_FAMILY 只覆盖 4 个 FFN 模块
    seen['kernels'][name] = seen['kernels'].get(name, 0) + 1
if candidate and name in ROWS_EXTRA:                # ROWS_EXTRA 的 9 个键全在 *_rows_v1 里
    bound = dict(zip(jit.arg_names, args))          # ← 每笔建一个 dict
    ... DEBUG / PARTS / M 三项检查 + 桶计数
```

产品每帧 950 笔启动里，**224 笔**（23.6%）属于 `c512_int8_ffn_rows_v1` /
`int8_ffn_segment_rows_v1`，**两条分支都命中**；`ROWS_EXTRA` 分支比通用路径贵
**+0.658 µs/笔**（上表实测）。离线单价 × 每帧笔数：

    通用路径  950 笔 × 0.422 µs = 0.401 ms/帧
    ROWS_EXTRA 分支 224 笔 × 0.658 µs = 0.147 ms/帧
    ────────────────────────────────────────────────
    合计 0.548 ms/帧

**探针给的 0.42 µs 是下界，不是答案。**

### 10.3 怎么量的：把实验搬进产品（同进程三条件逐帧轮转）

探针里加两次变量都复现不出 ⇒ 换设计，不再猜上下文。新探针
`prepfast_mdgc_v1.py` **不动产品、不动刀**，只替换 `rows_scopes_v1.dispatch_guard`
一个模块属性 —— 同一个钩子既定帧、又切条件（`one_frame` 每帧恰好调它一次；
`torch.xpu.synchronize` 每帧有 ~15 次，**不能**当帧边界）：

| 条件 | md 刀 | `launch_metadata` | guard | 对应 |
|---|---|---|---|---|
| `skip` | 开 | 不调 | 装但用不到 | A/B 的 `base,md` 臂 |
| `full` | 关 | 调 | **装** | A/B 的 `base` 臂 |
| `noguard` | 关 | 调 | **不装** | **产品真实情形** |
| `trivial` | 关 | 调 | 只加一层空包装 | 拆「多一层 Python 帧」 |

条件按 **K3 欧拉回路**（`noguard,trivial,full,noguard,full,trivial`）逐帧轮转：
每个**有序**对各出现一次，正反向互为对照，条件不再与「帧在周期里的位置」混淆。
**243 帧全部逐字节等于冻结参考**（`identical_frames=243`、`max_abs=0.0`）。

### 10.4 数字

组中位数（两个独立轮次，n≈70/条件）：

| 条件 | 轮次 A 中位 (ms) | 轮次 B 中位 (ms) |
|---|---:|---:|
| `skip` | 79.708 | — |
| `noguard` | 80.535 | 80.568 |
| `trivial` | — | 80.750 |
| `full` | 81.762 | 81.833 |

| 净效应 | 组中位数 | 配对（同进程） | A/B（跨进程） |
|---|---:|---:|---:|
| `full − skip`（A/B 量到的） | **+2.054** | +2.662 | +2.846 |
| `full − noguard`（**guard 包装**） | **+1.227 / +1.265** | +2.026 | — |
| `noguard − skip`（**`launch_metadata` 本体**） | **+0.827** | +0.575 | — |
| `trivial − noguard`（多一层 Python 帧） | +0.182 | — | — |

两个独立轮次的组中位数给出 **+1.227 / +1.265**（差 3%）—— 这是最可复现的一行。
配对估计受位置偏置影响（正反向不一致 0.35–1.40 ms），故以组中位数为准。

**⇒ `md` 的产品级收益 = `noguard − skip` ≈ 0.6–0.85 ms/帧（不是 2.85 ms）。
拆分比：guard 包装 ≈ 2/3，`launch_metadata` 本体 ≈ 1/3。**

### 10.5 GC 假设被否掉

每个帧边界记 `gc.get_stats()`，并在 15 个帧边界计时 `gc.collect(0)`：

    gc_threshold0            = 2000
    额外 gen0 回收/帧（full − skip） = +0.091
    gc.collect(0) 单价        = 0.080 ms（中位）
    ⇒ 额外 GC 成本            = 0.0073 ms/帧 = **2.662 ms 的 0.27%**

「`launch_metadata` 每笔 3 次分配 → gen0 回收」**不成立**。

### 10.6 修正后的账

`dispatch_guard` 只在「`launch_metadata` 被调用」时才收费 ⇒ A/B 八臂里
**只有两行被污染**（`md` 与 `all`），其余四把刀两侧都付同样的 guard，**数字照旧**。

| | 原报 | 修正后（产品级） |
|---|---:|---:|
| `launch` / `book` / `stream` / `keyof` | 1.249 / 0.562 / 0.262 / 0.135 | **不变**（guard 在两侧抵消） |
| `md` | 2.846 | **0.6–0.85** |
| 五刀合计 | **4.545 ms（1.0585×）** | **≈ 3.0–3.3 ms（≈1.04×）** |

绝对帧墙（`base` 与 `all` 两侧不同待遇，必须换算）：

    harness_base 82.2459 = product_base + guard
    harness_all  77.7008 = product_all              （md 开 ⇒ 不付 guard）
    ⇒ product_base ≈ 80.996（取 guard = 1.25）  ⇒ 产品级 base → all = **≈3.30 ms**

加法口径独立复核：`launch+book+stream+keyof = 2.208` + `md(0.83)` = **3.04 ms**（差 8%）。

**⇒ 本轮真正的产品级数字是 ≈3.0 ms（≈1.04×），不是 4.545 ms。**
`md` 仍然该留（0.8 ms 是实打实的），但它**不是最大的一项** —— 排序变成
`launch(1.249) > md(0.83) > book(0.562) > stream(0.262) > keyof(0.135)`。

### 10.7 仍未解释的（诚实记账）

**离线单价 × 每帧笔数给 guard 0.548 ms/帧，而产品里实测 1.23 ms/帧 —— 还差 2.3×。**
也就是说：原来看起来是「探针 vs 产品差 2.9×」的那个缺口，**定位之后并没有消失，
只是从一层挪到了下一层**（探针测得出机制，测不出量级）。

同一轮里 `launch_metadata` 本体却是**对得上**的（离线 0.75–1.02 µs/笔 →
0.71–0.97 ms/帧，实测 0.827 ms/帧）。
⇒ **「探针单价 × 笔数」这个折算对纯 C 调用成立，对 Python 层调用不成立。**
候选解释（未验）：产品堆大得多（数百 MB 活跃张量）⇒ 属性查找 / 短字符串哈希的
缓存命中率远低于探针进程。**要验证得在产品里给 `checked` 加自计时**，本轮没做。

### 10.8 顺带发现：`md` 刀让 guard 的 FFN 契约断言空转

`md` 一开，`launch_metadata` 不被调用 ⇒ `checked` 不执行 ⇒ `seen['kernels']` /
`seen['rows']` 全空 ⇒ **`dispatch_guard` 的 FFN 契约检查在 `md` 臂里其实是空转的**。
字节门与 `FfnCounters`（`counts['c512']==16 and counts['vit']==8`）仍然有效，
所以 A/B 的结论不受影响；但这是一处**覆盖盲区**，要记下来。

### 10.9 本轮新增产物

| 文件 | 作用 |
|---|---|
| `prepfast_mdgc_v1.py` / `prepfast_mdgc_run_v1.sh` | 同进程多条件轮转探针（帧边界 = `dispatch_guard`） |
| `launchmd_guard_v1.py` | 用**真** guard 量包装成本（含「探针内核不触发 ROWS_EXTRA 分支」这一发现） |
| `guard_cost_offline_v1.py` | **零 GPU**：真 guard + 假 kernel，量两条分支的单价 |
| `launchmd_smoke_v1.py` | `launch_metadata` 传 `None` 与传真 `LazyDict` 的等价性冒烟 |

```bash
PY="E:/ComfyUI-aki-v3-IntelArc_20260722/ComfyUI-aki-v3-IntelArc/python/python.exe"
cd E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/prepfast_v1
"$PY" -X utf8 guard_cost_offline_v1.py --out D:/prepfast-v1/guardcost   # 0 GPU
RUN=D:/prepfast-v1/mdgc4 bash prepfast_mdgc_run_v1.sh full              # 243 帧，约 3 分钟
```

数据：`D:/prepfast-v1/{mdgc4,mdgc6,guard,guardcost}` —— ⚠️ **已清除**（见文件头说明；命令重跑即可重建）。

---

## 11. 污染面排查：这个坑**只埋在 `md` 一处**（grep 级成本，零 GPU）

§10 定位到的污染有一个直接推论：**所有走 `materials_entry_v1.one_frame` 的层都带同一层脚手架**，
所以要逐层问一遍「它是不是也把 guard 绕过了」。

### 11.1 判据是构造性的，不用逐层重测

`dispatch_guard` **只补一个东西**：

```python
# rows_scopes_v1.py:239
CompiledKernel.launch_metadata = checked
```

（`:209-235` 的 `checked` 体全是断言与字典自增；`:243` 恢复。）
⇒ **guard 的全部成本都发生在 `CompiledKernel.launch_metadata` 被调用的那一刻。**

而 `launch_metadata` 的调用次数 = **真启动次数**，不是 `JITFunction.run` 的次数：

```python
# triton/runtime/jit.py:774-777
if not warmup:                                   # ← warmup 路径到此为止
    ...
    launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
```

（另一个调用点 `compiler/compiler.py:530` 在 `CompiledKernel.__getitem__` 的 `runner` 里，
产品走 `JITFunction.run`，不经过它。）

> **判据**：一层优化的实测收益被污染 ⟺ **该层减少了 `launch_metadata` 的调用次数**。
> 反之，只要两侧调用次数相同，guard 的代价在差分里**自动抵消**，报数就是干净的。

### 11.2 逐层核（全部走同一辆车，只有 `block1_v1` 自己 import guard）

| 层 | 刀 | 改什么 | 减少 `launch_metadata` 调用？ | 判 |
|---|---|---|---|---|
| 1 `hostfast_v1` | `run` | 复刻 `jit.py:768-780` 启动段 | **否** —— `hostfast_cuts_v1.py:341` **保留了** `kernel.launch_metadata(grid, stream, *values)` | ✅ 净 |
| 1 `hostfast_v1` | `getitem` | 按 `(id(jit), grid)` 复用闭包 | 否（仍走 `run` 的启动段） | ✅ 净 |
| 2 `syncfast_v1` | — | 搬 `torch.xpu.synchronize` 的位置 | 否（不碰 Triton 类属性；其 RESULT 亦自称「与 launch 结构无关」） | ✅ 净 |
| 3 `merge_v1` | 1+2 | 同 | 否；且不变量表里 `hostfast launch` = **230717** 在 3 个 `host` 轮**逐位相同** | ✅ 净 |
| 4 `selectfast_v1` | `select` | 缓存 `select` 结果 ⇒ 去掉 **189 次/帧** `jit.warmup` | **否** —— `jit.warmup` ⇒ `run(warmup=True)` ⇒ 落在 `if not warmup:` 的**假分支**，**根本到不了 `launch_metadata`** | ✅ 净 |
| 5 `fp8fast_v1` | `disp`/`half`/`pool` | `quantize_fp8` 的**外围 CPU 段**（`x.half()` / `.contiguous()` / `empty_like()`） | 否（`_kernel` 启动数 383/帧不变） | ✅ 净（且整层已砍） |
| 6 `block1_v1` | — | C512 FFN 算术 INT8↔FP16 | 否（内核数不变；int8 侧 243/243 逐字节复现冻结参考） | ✅ 净 |
| 7 `prepfast_v1` | `launch`/`book`/`stream`/`keyof` | 准备期查表 / 键 / 流 / 同步 | 否（`launch` 刀在 `:351-354` 两分支里**都**调 `launch_metadata`） | ✅ 净 |
| 7 `prepfast_v1` | **`md`** | `:350-354`：hook 链为空时 `md = ... else None` ⇒ **整个跳过调用** | **是：950/帧 → 0**（guard 自计 `launches_per_frame`） | ❌ **污染** |

### 11.3 结论

> **污染是孤立的，只有 `md` 一项。**
> 此前各层的报数 —— `hostfast` **2.385 ms**、`syncfast` **6.867 ms**、`merge` **8.07 ms** ——
> **都不需要修改**。§10.6 那次修正是本轮**唯一**需要做的修正。

**为什么 `md` 偏偏是那个例外**：它是五刀里**唯一**动「调用次数」的一刀。
其余四刀都在动「每次调用花多少」（准备期、键、流、同步），次数不动 ⇒ guard 在两侧等量收费。
`selectfast` 的 `select` 看起来像在减次数（189 次/帧），但减的是 **warmup**，
而 warmup 路径被 `if not warmup:` 挡在 `launch_metadata` 之前 —— **减了次数，却没减到 guard 头上**。

### 11.4 顺带确认的一个覆盖缺口

§10.8 那条（`md` 臂里 guard 的 FFN 契约断言空转）**只影响 `md` 臂**；
其余各层的两臂里 `checked` 都照常执行，契约断言有效。

### 11.5 一条可复用的规矩（已写进技能）

> **凡是用「插桩/包装」做 A/B 的脚手架，必须回答：被测的每一刀是否改变了插桩点的触发次数。**
> 若某刀减少触发次数，则它测到的是「刀的真实收益 + 脚手架在那些次数上的开销」，
> 必须单独扣除。**判据最好从插桩的实现本身推出来（这里 = guard 只补 `launch_metadata`），
> 而不是靠逐层重跑** —— 前者是构造性的，后者只是抽样。

---

## 12. 数据清理记录（2026-09-17 20:45）

按用户要求**只留结论与脚本**，`D:/prepfast-v1/` **整目录已删除**（回收 **10.8 GB**，D: 可用 17 → 27.2 GB）。
删掉的是：`r1` `r2`（8 臂 × 2 session 回文 A/B 原始数据，4.8 G × 2）、`mdgc` `mdgc2` `mdgc3` `mdgc4` `mdgc5` `mdgc6`、
`guard` `guardcost` `smoke` `smoke1` `split`，以及全部租约/外层日志。

**保留**（均在 `E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580/reference/prepfast_v1/`，`E:` 侧不受影响）：

| 文件 | 作用 |
|---|---|
| `RESULT.md` | 本文件 —— 全部结论与数字（**唯一留档**） |
| `prepfast_cuts_v1.py` / `prepfast_entry_v1.py` | 五刀实现 + 入口 |
| `prepfast_ab_run_v1.sh` / `prepfast_ab_summary_v1.py` / `prepfast_queue_v1.sh` | §8 的 16 轮 A/B 复现链 |
| `prepfast_mdgc_v1.py` / `prepfast_mdgc_run_v1.sh` | §10/§11 的同进程多条件轮转探针 |
| `launchmd_guard_v1.py` / `guard_cost_offline_v1.py` / `launchmd_smoke_v1.py` / `launchmd_split_v1.py` | guard 定价与等价性冒烟 |

**⚠️ 后果（已知并接受）**：本轮所有数字**只剩本文件的记录，无法再复核原始帧**。
任何需要重算的场合（换门槛、加素材、查某一帧）都必须按 §8 / §10.9 的命令**重跑**，命令会自动重建 `D:/prepfast-v1/` 下的目录。

### 12.1 追加：其它轮次数据也一并清除（20:55，用户「继续清」）

| 清除 | 体积 |
|---|---:|
| `materials-v1` / `fp8fast-v1` / `merge-v1` / `block1-v1` / `syncfast-v1` | 11.0 / 6.6 / 4.8 / 1.2 / 0.78 G |
| `warpsync-v1` / `fullsize-devsplit-v1` / `prepleft-v1` / `selectfast-v1` / `hostfast-v1` | 零头（合计 < 60 M） |

**只保留两个**：
* `D:/fullsize-rows-v1` —— **冻结参考（硬约束，绝不能删）**
* `D:/verdict-v1` —— **用户人工审核素材**（`baseline-video` / `hostfast-video` / `syncfast-video` /
  `block1-c512-fp16-video` / `materials-*-video` + 指标）。按项目协议「助手只出指标 + 对比视频，**裁决权在用户**」
  ⇒ 它是**交付物**，不是中间产物。

**D: 可用 16.4 → 51.09 GB（共回收 34.7 GB）。**
⚠️ **读数陷阱**：删完后 `df` **在头 20 秒内一直不变**（5 次采样全是 27.19–27.20 GB），
30 秒后才跳到 51.09 GB。**在 Windows 上核验大批删除必须等 30 秒，并看「已用」列的绝对变化。**
