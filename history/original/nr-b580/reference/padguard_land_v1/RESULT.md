# padguard_land_v1 —— 刀 A：`F.pad` 全零元组短路

> 轮次：`nr-b580/reference/padguard_land_v1/`　数据：`D:/padguard-land-v1/`
> 依据：`opcensus_v1`（按调用点普查，79 次/帧的 `F.pad` 全部归到行，其中 **40 次/帧 pad 元组全零**）
> 前情：`nowsplit3_v1`（N7）判定 8.18 ms 设备空转 **散在帧中** ⇒ 修法 = **降「每次提交」的 host 成本**。

---

## 1. 结论

| 项 | 结果 |
|---|---|
| **逐字节** | **64/64 帧**（on 臂 vs off 臂）**逐帧逐元素相等** ⇒ **逐字节中性**（实测，不是推断） |
| **速度** | **−0.35 ~ −0.48 ms/帧**（两个口径同向、**CI 都不含 0**） |
| 作用位置 | host 提交侧（**28 次** `F.pad` dispatch + 28 次分配 + 28 次真复制/帧 全部消失） |
| 开关 | `NR_PADGUARD`（默认 `on`）；`off` 逐字复现落地前 |
| 改动 | 5 个文件、**两树同改**（4 个 `backend/nr_backend/*` 同行替换 + `pre_mlp.py` 末尾追加）+ 1 个 int8-only 调用点 |

> **⚠️ 短路次数是 28，不是 40。** 落地前写的"40 次/帧全零"是**普查标签污染**造成的
> （`_bump` 只记每个链桶第一次调用的 pad 元组）⇒ 已撤回，真值由落地前后**总量差**算出。
> 完整推导见 `opcensus_v1/RESULT.md` §10。**下面 §2 的表已按真值改写。**

**速度的两个口径（必须一起看）**

| 口径 | 估计 | 95% CI | 同号 |
|---|---:|---|---|
| 按帧号配对（8 off 臂 vs 8 on 臂 × 20 稳态帧） | 中位 **−0.4255 ms** | [−2.0019, −0.2898] | 15/20 |
| 池化（每个「臂 × 稳态帧」当独立样本，剔离群） | 均值 **−0.3490 ms** | [−0.6069, −0.0880] | off n=158 / on n=160 |

* 两个口径**方向一致、量级一致**，CI **都不含 0** ⇒ 这是**解出来的**结果，不是噪声。
* **8 臂版本解不出来**（4+4 × 8 稳态帧）：中位 −0.1961 ms，CI [−3.1426, +0.2190] 跨 0，同号 5/8。
  ⇒ 16 臂（`REPEATS=4`）是**必要的**，不是"多跑一遍更保险"。
* ★ **实测 −0.35 ms 比微基准预测的 0.151 ms 大 2.3 倍**（28 次 × 5.39 µs）。
  这**符合技能 149**：孤立调用价是**下界**。真帧里那次 `F.pad` 还带来一次分配与一次
  真复制，省掉它同时让下游直接吃原本就在缓存里的张量。
  ⇒ **帧级实测才是数字，微基准只用来排序。**

---

## 2. 改了什么

`pre_mlp.py`（**末尾追加**，该文件无 Triton 内核）：

```python
_PAD_GUARD_OFF=os.environ.get('NR_PADGUARD','on').strip().lower() in ('','off','none','0','false')

def pad_or_identity(x: Tensor,pad) -> Tensor:
    if _PAD_GUARD_OFF or any(pad):
        return torch.nn.functional.pad(x,pad)
    return x
```

调用点（**全部同行替换**）。下表的"短路次数"由**落地前后总量差**逐点算出
（落地前调用数 − 落地后存活数），**不依赖普查标签**：

| 文件 | 行 | 落地前 次/帧 | 短路 次/帧 | 存活 次/帧 |
|---|---|---:|---:|---:|
| `experimental/window_blocks_v3.py`（**仅 int8**） | 48 | 36 | **9** | 27 |
| `backend/nr_backend/vit_block.py` | 39 | 8 | **8** | 0 |
| `backend/nr_backend/vit_block.py` | 40 | 8 | **8** | 0 |
| `backend/nr_backend/multihead_block.py` | 108 | 3 | **3** | 0 |
| `backend/nr_backend/decoder.py` | 70 | 0 | 0 | 0（保留，防未来出现全零元组） |
| **合计** | | **55** | **28** | **27** |

合计 **28 次/帧**（= 全帧 79 → 51 的差额）。
`window_blocks_v3.py:48` 是**同一行三种 pad 元组**
（18 全零 / 15 真 / 3 全零 —— 这三个数**也是标签**，只用于说明"必须按值判断"）
⇒ **守卫必须按值判断**，不能静态改常数。
★ 注意该行只短路 **9** 次：**"这行有三种元组"不等于"这行大部分是全零"**。

---

## 3. ★ 落地流程踩的两条硬约束（本轮最大的收获）

参考侧 `materials_entry_v1.py` 对 `nr_backend/*` 有**两条互相拉扯的约束**：

| # | 约束 | 位置 | 表现 |
|---|---|---|---|
| 1 | `loaded_sources` sha256 清单（**只钉 exact 树**） | `materials_entry_v1.py:141` | 改 exact 树 ⇒ 清单过期 ⇒ 一开跑 `AssertionError` |
| 2 | `duplicate_source_agreement()`（**两树逐字节相同**） | `materials_entry_v1.py:95` | 只改 int8 ⇒ 两树分叉 ⇒ 一开跑 `AssertionError` |

三次尝试：

| 版本 | 做法 | 结果 |
|---|---|---|
| `apply_v1` | 两树同改，**没派生记录** | ✗ `AssertionError: ...nr-b580\backend\nr_backend\pre_mlp.py` |
| `apply_v2` | **只改 int8**，以为"两树相同"这条惯例已过时 | ✗ `AssertionError: ['decoder.py','multihead_block.py','pre_mlp.py','vit_block.py']` |
| `apply_v3` | **两树同改 + 派生新记录** | ✓ 64/64 逐字节门通过 |

⇒ **合法做法只有一种**：两树同改（满足 #2）**然后**派生新记录（满足 #1）。
派生是 CPU-only：`padguard_land_derive_exact_v1.py`
（base = `D:/quanttrim-land-v1/exact-derived/validation.json`，**恰好**推进 4 个 `loaded_sources`，
243 个输入帧逐字节复核，不重采）。产物
`D:/padguard-land-v1/exact-derived/validation.json`（`sha256 4ff5176a…`）。

★ **旧记录不能用**：`--exact-validation` 必须换成上面这一份。
换记录这件事**不是可选的收尾**，是跑得起来的前提。

---

## 4. 装置与判据

* `padguard_land_apply_v3.py` —— 落地（`--check` 是真实 dry-run：从备份起算）。
  复核：两树 `backend/nr_backend/*` 落地后**仍逐字节相同**、可编译、**内核起始行号不变**。
* `padguard_land_entry_v1.py` —— 一臂到底，报告 `pre_mlp._PAD_GUARD_OFF` 作**臂身份自证**。
  **不在产品里塞计数器**（本刀砍的就是 host 侧，计数器会污染计时）。
* `padguard_land_ab_run_v1.sh` —— `MODE=gate`（2 臂）/ `MODE=abba`（`REPEATS`×4 臂），一个租约。
* `padguard_land_summary_v1.py` —— 配对逐字节门 + 两个口径的速度估计（配对 / 池化）。

**逐字节门用的是「两臂互相相等」，不是「等于冻结参考」**：本轮的 env 与冻结参考本来就
不逐字节相等（普查实测 `max_abs` 从第 1 帧起非零）。对一个**可证明恒等**的改动，
两臂互相相等才是正确的门。

| 批 | RUN | 规模 | 用途 |
|---|---|---|---|
| r1 / r2 | — | — | ✗ apply_v1 / apply_v2 时代的失败跑（记录在案） |
| r3 / r4 | — | — | ✗ 同上 |
| r5 | `D:/padguard-land-v1/r5` | 2 臂 × 64 帧 | **逐字节门 64/64** |
| r6 | `D:/padguard-land-v1/r6` | 8 臂 × 12 帧 | 速度（**解不出来**，留作对照） |
| r7 | `D:/padguard-land-v1/r7` | **16 臂 × 24 帧** | **速度（结论）** |
| r5（普查） | `D:/opcensus-v1/r5` | 24 帧 | **开关自证**：`pad` **79 → 51 次/帧**、全零 **28 → 0**（对照 r4 = 落地前） |

---

## 5. 复现

```bash
W=E:/ComfyUI-aki-v3-IntelArc_20260722
PY=$W/ComfyUI-aki-v3-IntelArc/python/python.exe
cd $W/nr-b580/reference/padguard_land_v1

"$PY" -X utf8 padguard_land_apply_v3.py --check      # 落地 dry-run
"$PY" -X utf8 padguard_land_apply_v3.py              # 正式落地（两树）
"$PY" -X utf8 padguard_land_derive_exact_v1.py --out D:/padguard-land-v1/exact-derived

RUN=D:/padguard-land-v1/r8 MODE=gate LIMIT=64 bash padguard_land_ab_run_v1.sh
RUN=D:/padguard-land-v1/r9 MODE=abba LIMIT=24 REPEATS=4 bash padguard_land_ab_run_v1.sh

"$PY" -X utf8 padguard_land_summary_v1.py --run D:/padguard-land-v1/r8 --mode gate
"$PY" -X utf8 padguard_land_summary_v1.py --run D:/padguard-land-v1/r9 --mode abba
```

回滚：`backup/{exact,int8}/backend/nr_backend/*.py` 是**落地前的整字节原件**，
`NR_PADGUARD=off` 也逐字复现落地前（两臂走同一份源码，只差环境变量）。

---

## 6. 诚实边界

1. **绝对帧墙不作产品数字。** 本轮的绝对值（off 中位 **47.04 ms**、on 中位 **46.56 ms**）
   与 `HANDOVER` 记的 45.16 ms **不同 harness、不同 env** ⇒ **只能报配对 Δ**（技能 138）。
2. **`submit_all_ms` 这一轮拿不到**：落地轮的逐帧记录只有 `ms`；
   `submit_all_ms` / `sync_tail_ms` 是**普查轮补丁**加的。⇒ "省在 host 侧"这句话
   在**机制上**成立（40 次 dispatch/分配/复制消失），但本轮**没有直接测到 host 分量**。
3. **离群已剔除并记录**：池化口径剔了 2 个（`off` 臂一帧 78.7 ms，另一臂 1 帧），
   判据是「> 该臂中位的 1.5 倍」。剔除前后两个口径都报（配对口径不剔）。
4. **`decoder.py:70` 在本轮 0 次全零**，改它只是为将来兜底 —— 不计入 40 次/帧。
5. **`multihead_block.py:92` 在快速线里被 `WindowBlocks.apply` 顶替**，本轮 0 次全零；
   同样只是兜底。
6. **预编译包审计：8/35 → 9/35 不符**（新增 `vit_block.py`）。
   包在本刀之前就**已经不可用**（`CacheOnly.__enter__:212` 会抛），
   处置是「用 record-only 派生件替换 package.json + catalog.json」——
   **这是待裁定的第 ② 项，本轮不动。**
7. **仍有 51 次/帧 真复制 pad 没动** —— `window_blocks_v3.py:48` 存活 27
   （`(0,0,4,4,4,4)`）＋ `c32_block.py:36` 6 ＋ `post.py:49` 1 ＋ `executor.py:83/88` 17
   （`(0,0,0,4,0,0)`）。存活数**由构造保证是真复制**（守卫只放行 `any(pad)` 为真的调用）。
   ⚠️ 原稿写"39 次/帧"，那是**标签污染**下的错值（见 `opcensus_v1/RESULT.md` §10）；
   **真值是 51**。动它们会改画面或需要融合内核。其中 17 次/帧 的**归属本身还没确认**（见 `opcensus_v1/RESULT.md` §7）。
   ★ 这也意味着：**"剩下的 pad"比原以为的大 31%**，但它已经不是"空操作"这一类了。

---

## 7. 下一刀

| 刀 | 位置 | 次/帧 | 逐字节 | 状态 |
|---|---|---:|---|---|
| B | `fused_qkv_pack_native_half_v1.py:79` 的 3 个 `empty` 按 (h,w,heads) 缓存 | 156 | 需验证（别名风险） | 待做 |
| C | `fast_matrices_v3.py:93` 输出缓冲缓存 | 115 | 需验证（别名风险） | 待做 |
| D | `c32_block.py:38/39/46` 换 `quantize_fp8_grid`（补刀① 漏的 12 次/帧） | 12 | 需**探针**证明值已在网格上 | 待做 |
| E | 剩余 208 次/帧 真转换的融合 | 208 | 需先分相测 `.contiguous()` | 待做 |

★ 刀 B/C 与本刀同族（**省 host 提交**），但都带**别名风险**：
把输出缓冲缓存起来复用，等于让两个帧的中间张量共享内存。
**必须先确认没有消费者跨帧持有引用**（时域环里 `previous` 是活的），再谈速度。
