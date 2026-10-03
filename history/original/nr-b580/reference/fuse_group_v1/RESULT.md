# 纵向融合 A1（`_groups` 接线）—— 只测不落地

> 轮次目录 `reference/fuse_group_v1/`（2026-09-21）。数据：`D:/fuse-group-v1/r1/{seq40,seq80swap,pool.json}`。
> 上游：`merge_survey_v1/RESULT.md` §5.1 / §9-A（本轮的立案书）；
> `kernel_split_v1`（同 (模块, grid) 相邻帧可差 19 倍 ⇒ 只能用跨帧/跨臂对比）。
> **这是诊断跑**：产品源 0 改动、冻结缓存 0 改动、不产正确性产物、不进任何门槛。
> 本轮**没有**把任何东西接进产品代码 —— 结论是「可以接线」，不是「已经接线」。

---

## §0 一句话

`c512_int8_ffn_rows_v1.py:103` 那个**已写好、没接线**的 `_groups`：**可以接线**。
它**逐字节等价**（同输入双跑 32/32 块全等、`mismatch=0`、差值 0）、**零溢出**，
每帧**省 16 次启动**（189 → 173，每帧都少、零例外），
帧墙合并口径**降 0.308 ms（2.19σ，占 0.70%）**。

**但这条路的量就到这里。** 它把调度层已经不多的余量再削 0.3 ms，
离 10 ms 还差 **约 4.5 倍** —— 与 `merge_survey_v1` §6 的硬结论完全一致：

> 帧墙 44.9 ms、设备忙时 36.0 ms ⇒ 主机侧全部可兑现上限 8.92 ms；
> 并核 0.9 + 纵向 4.0 已是天花板，**10 ms 要求设备工作量掉 3.6 倍，那是算术改动**。

本轮把「纵向 4.0 ms」这个上界里的**第一个单元**量出来了：**0.31 ms**。

---

## §1 结论表

| # | 结论 | 读数 | 判定 |
|---|---|---|---|
| 1 | **P1/P2 逐字节等价** | dual 同输入双跑 **32/32 块相等**，`mismatch=0`，`max_abs=0` | ✅ **通过** |
| 2 | **P5 零溢出** | `_groups` `spills=0`；`shared_bytes=2048`（`_expand` 4096 + `_reduce` 4096） | ✅ 通过 |
| 3 | **启动账** | 每帧 `select()`：off **189** / group **173** ⇒ **每帧少 16**，零例外；dual 205（多一次候选） | ✅ 与源码预测一致 |
| 4 | **帧墙（合并两次跑）** | Δ均值 **+0.308 ms**，σ 0.140，**+2.19σ**，占 **0.70%** | 🟡 **方向一致、幅度贴噪声底** |
| 5 | 兑现率 | 0.308 / 0.47（主机侧上界 = 16 × 29.4 µs）≈ **66%** | 量级自洽 |
| 6 | 设备侧顺带 | QH 中间缓冲消失：**29.3 MB/帧**的写 + 读不再发生（带宽口径 ≈ 0.06 ms） | 小 |
| 7 | **副作用发现** | `NR_LAUNCH_FASTPATH` 的 `md` 刀使 `launch_metadata` 在无 hook 时**不被调用** ⇒ `dispatch_guard` / `Dataflow` 在未剖析帧里**全空转** | ⚠️ 见 §7 |
| 8 | 画面 | 本轮的 `group` arm **等价于**`off` arm ⇒ **不改画面** | ✅ |

**事前登记的否证条件一条都没触发**（`PLAN.md` §1）：没有溢出、没有不等价、启动数正好是 64（c512 族 80 → 64）。

---

## §2 方法：三处不变量

### 2.1 产品源码 0 改动 —— **内存替换模块**

`rows_scopes_v1` 直到 `materials_entry_v1.py:89` 才被 import。入口补丁在它**之前**
把原文件读进来、按 3 个锚点做文本替换、`exec` 成一个模块对象塞进 `sys.modules`
⇒ 之后所有 `import rows_scopes_v1` 拿到的是打补丁那份，**磁盘上一个字节没变**。
入口另外做 6 个锚点、并**断言拿到的是打了补丁的模块**
（磁盘上的 `rows_scopes_v1` 没有 `GROUP_STATE` 属性，所以这本身就是把手）—— 对不上就停。

两级补丁都报 `source_sha256` / `patched_sha256`，写在报告里可复核。

### 2.2 正确性门 —— **同输入双跑**（dual）

不能拿「两帧的输出」比：帧是**时序序列**，同一帧跑两遍会毁掉时间维状态。
做法是**同一个 block、同一份输入**上跑两条路：

- 参考路 `_expand` → `_reduce` → `qg`（**模型继续消费它**，所以 dual 帧画面不变）
- 候选路 `_groups` → `qg_candidate`（临时缓冲）
- 逐元素 `!=` 计数

这是**同输入**的等价性，不是同帧的等价性 —— 所以它是一次真正的门，而不是相关性。

### 2.3 三种 arm，一个进程

| 帧位置 | arm | 干什么 |
|---|---|---|
| `0 .. OFF-1` | `off` | 原路。对冻结基线做逐字节复现 ⇒ C1 |
| `OFF .. OFF+DUAL-1` | `dual` | 同输入双跑 ⇒ C4；顺带把 `_groups` 编译/热身 |
| 其余 | `off` / `group` **交替** | 帧墙 A/B |

**两次跑、奇偶对调**（`NR_FFN_GROUP_SWAP`）：交替 arm 的隐患是「负载本身有奇偶周期性」时
会把周期读成 arm 差。本段素材的 `reset` 只在帧 0（已核），所以这是保险；
第二次跑把奇偶对调 ⇒ 偏置方向相反、合并时相消。实测对调生效：两次跑的 off 帧位置**分别落在偶/奇**。

---

## §3 自检（C1–C7，全过）

| # | 检查 | 判定 | 读数 |
|---|---|---|---|
| C1 | `off` 臂复现冻结基线 | PASS | reset 帧 `byte_equal=True` / `max_abs=0.0`；帧 1 `max_abs=6.103515625e-05`；**与上游未打补丁的 merge_survey_v1 同帧逐位相同** |
| C2 | 逐帧启动账 | PASS | `off=[189]` / `dual=[205]` / `group=[173]`，**每臂只有一个值**；`off−group=16`、`dual−off=16` |
| C3 | 融合路真的在跑 | PASS | `ffn_selections['groups_debugFalse']` 已选 `[16,64]`；`_groups` 的 `_groups_dual` 与 `_groups_debugFalse` **同一个 hash** |
| C4 | dual 等价 | PASS | `calls=32, equal=32, mismatch=0, max_abs=0` |
| C5 | arm 记账 | PASS | 40 帧 = off 20 + dual 2 + group 18；80 帧 = off 40 + dual 2 + group 38 |
| C6 | `select` 自检 | PASS | `verify_fail=0`（`verify_ok=227/227`） |
| C7 | `_groups` 资源 | PASS | `spills=0`；`shared_bytes=2048` |

**C1 的诚实口径**：上轮清理时删掉了 `merge_survey_v1` 的 `frames/*.npz`，
所以本轮**没有**做原始字节比 —— 一致性是**指纹级**的：`reset` 帧 `byte_equal` 与
`max_abs=0.0` 完全相同，帧 1/2 的 `max_abs` 与上游也逐位相同。登记在 §9。

---

## §4 主结果 1：逐字节等价

| 量 | 值 |
|---|---|
| 双跑帧数 | 2（每跑 2 帧，两次跑都做） |
| 累计 block 比较次数 | **32** |
| 相等 | **32** |
| 不等 | **0** |
| 最大差值 | **0** |

**为什么这是可预期的**：两边的 K 累加序同构。
`_expand` 对每个 part 沿 K 走 `start=0,1`（两段 32），`_reduce` 依次把 `part=0..7`
的 `tl.dot` 累加进 `total`；`_groups` 就是「对 `part=0..7`：先两段 32 展开、`_q` 之后**立刻**
`tl.dot` 进 total」。INT32 累加精确、`_q` 确定 ⇒ 逐字节相等。
**但「预期」不是「已验证」** —— 现在是已验证。

⚠️ 一个**非平凡**的观察：`_groups` 选中的配置是 **(BM,BN)=(16,64)**，
而 `_expand` / `_reduce` 各选 **(32,64)**。融合后 M 方向反而要**更小的 tile**
（多了一个 32 宽的累加器 + 8 次 `part` 循环的活跃区间）。这不影响等价性（已逐字节验证），
但它说明**融合内核不是两段路的机械拼接**。

---

## §5 主结果 2：启动账

`spill_preflight_v1.select()` 在**每次 launch 里恰好被调用一次**（`rows_scopes` 的
`c512_ffn.launch` / `vit_ffn.launch` 都是先 `select` 再发），而它自己记着
`_STATS['calls']` ⇒ **逐帧差分这个计数器就等于逐帧数启动，且零额外开销**。

| arm | 每帧 `select()` | 与 off 之差 | 归因 |
|---|---:|---:|---|
| off | **189** | 0 | 5 内核 × 16 块 + 4 内核 × 8 块 = 80 + 32 = 112，**其余 77 次来自其它 rows 模块** |
| group | **173** | **−16** | 4 内核 × 16 块 + 32 = 96 |
| dual | **205** | **+16** | 6 内核 × 16 块 + 32 = 128（多跑一次候选） |

**判据只用可归因的差**：`off−group = 16` **恰好等于 c512 块数**，
且**每一帧都少 16、零例外**（每臂的差分集合只有一个元素）。
两臂之间唯一的源码差别就是「`_expand` + `_reduce` 并成 `_groups`」。

⚠️ 77 次非 FFN 的 `select()` 调用本轮**没有**逐一归因（`nr-b580-int8/experimental` 里
有十几个模块 import 了 `select`）。它不影响判据 —— 判据是**两臂之差**，而两臂跑的是同一份会话。

**顺带**：`_groups` 让 `qh = torch.empty((rows, 2048), dtype=torch.int8)` 直接不再分配。
本轮 `rows=448`（由 `_linear` 的 grid `(14, 8)` 反推：14 × 32）⇒ 每块 **917 KB**，
16 块 ⇒ **14.7 MB/帧**的分配与写，加上 `_reduce` 读回来的 **14.7 MB** ⇒
**29.3 MB/帧**的显存流量消失（B580 带宽口径约 0.06 ms）。小，但方向一致。

---

## §6 主结果 3：帧墙（两次跑，奇偶对调）

帧墙 = `synchronized processing call`，含 Python / scope / guard；不含上传、回读、IO、motion、编解码。

| 跑 | arm | n | 中位 ms | 均值 ms | σ ms |
|---|---|---:|---:|---:|---:|
| `seq40`（SWAP=0） | off | 18 | 44.007 | 44.098 | 0.781 |
| | group | 18 | 44.127 | 43.845 | 0.798 |
| | **Δ** | | **−0.120** | **+0.253** | 0.263（0.96σ） |
| `seq80swap`（SWAP=1） | off | 38 | 44.177 | 44.224 | 0.879 |
| | group | 38 | 43.746 | 43.894 | 0.526 |
| | **Δ** | | **+0.431** | **+0.330** | 0.166（**1.99σ**） |
| **合并（逆方差加权）** | | 112 | | **+0.308** | **0.140（+2.19σ）** |

**怎么读这张表（不美化也不埋掉）**：

- **均值口径两次同号**（+0.253 / +0.330），方向与机理（少 16 次提交）一致。
- **中位口径两次不一致**（−0.120 / +0.431）⇒ 单跑的中位数还埋在噪声里。
- 合并均值 **+0.308 ms = 0.70%**，**2.19σ**。事前登记的 0.15 ms 阈值在均值口径上两次都过。
- **SWAP=1 那一跑替我们排掉了一个偏差**：那一跑里 off 是每对中的**第二个**；
  若存在「越跑越快」的漂移，漂移本身会把 Δ 推向**负**，而实测是 **+0.330**
  ⇒ 这一跑的正差**不可能**由漂移造成。这是本轮唯一一个「不靠 σ 也能站住」的论证。
- 与主机侧上界对照：16 × 29.4 µs = **0.47 ms**（`fastline_arch_v1/ARCHITECTURE.md` §4.3），
  实测 **0.308** ⇒ **兑现率 ≈ 66%**。剩下的 34% 大概是「省下的提交被重叠吸收」。

**§9-listed warning**：σ 是**样本标准误**，只有 2 次跑，它不是置信区间的严格陈述。
它只回答一个问题：**这个差有没有被噪声解释掉。** 答案：合并口径没有（2.19σ），单跑口径基本有。

---

## §7 顺带发现（重要，与 A1 无关但影响历史结论的可信度）

当 `NR_LAUNCH_FASTPATH` 打开时，`CompiledKernel.launch_metadata` **在未剖析的帧里根本不被调用**。

本轮第一版把启动账挂在 `rows_scopes.dispatch_guard(VARIANT)` 的 `seen` 上（它包的就是
`CompiledKernel.launch_metadata`）。40 帧全部读回 `{}` —— 一次启动都没数到。

原因在那一刀的源码里（`nr-b580-int8/experimental/launch_fastpath_v1.py:363-367`）：

```
if 'md' in self.cuts:
    md = (kernel.launch_metadata(grid, stream, *values)
          if (enter.calls or exit_.calls) else None)
```

`enter` / `exit_` 是 `knobs.runtime.launch_enter_hook` / `launch_exit_hook`。
没有注册 launch hook 时（生产态的常态），`launch_metadata` **被整段跳过**。

⇒ 两个后果，**都登记在此**：

1. `rows_scopes.dispatch_guard` 与 `quantization_dataflow_v1.Dataflow` **在未剖析的帧里是空转的。**
   前者数不到东西，后者的 `assert name in CONTRACTS, ('Unreviewed kernel', name)` 也**不会触发**
   —— 「未注册内核 fail closed」这条保证，在这个环境下**不是靠这条路径实现的**。
   （`torch.profiler` 会注册 hook，所以**被剖析的那一帧**里它们大概是活的；
   但本轮**没有**做隔离实验证明这一点，只登记为「大概率」。）
2. 本项目所有沿用这一套环境变量的诊断轮（`kernel_split_v1` / `dense_ab_v1` / `merge_survey_v1`）
   如果曾经把 `dispatch_guard` 的 `seen` 当成过读数，那些读数都该重看。
   本轮 C2 因此改挂 `select_stats()['calls']`。

⚠️ 这条**不影响** `merge_survey_v1` 的主结论：那一轮的依赖探针挂的是
`JITFunction.run`（记到 776 次，C6 自检通过），不是 `launch_metadata`。

---

## §8 对「10 ms」意味着什么

| 量 | 值 |
|---|---|
| 接线 `_groups` 的实测收益 | **0.308 ms** |
| 帧墙（约） | 44.2 → **43.9 ms** |
| 目标 | 10 ms |
| 还差 | **≈ 4.4 倍** |

**结论不变**：调度层（并核 0.9 + 纵向 4.0 上界，本轮把纵向第一个单元量到 0.31）**加起来也不到 5 ms**。
10 ms 只能来自**设备工作量的算术级下降**。

---

## §9 边界（诚实清单）

1. **只跑 40 + 80 帧、一段素材**（864×480）。依赖与等价性由源码决定，但本轮只报「这一段」。
2. **帧墙口径**含 Python / scope / guard；**不含**上传、回读、IO、motion 生成、编解码。
   不与历史数字比，只与同轮同进程的另一臂比。
3. **dual 帧有观测者效应**（每块多一次候选启动 + 一次 `!=` 归约与同步）⇒ **dual 帧不参与计时**。
4. **σ 是样本标准误，只有 2 次跑**。不是置信区间。
5. `--allow-compile` **关掉了 DiskOnly 对参考侧的保护**。因为 `_groups` **不在冻结缓存里**
   （缓存普查：`grep '"_groups"'` 命中 **0**，而 `_expand`/`_reduce` 命中）——
   这同时**确证**了它此前是死代码。已有内核仍命中磁盘缓存，但这条登记在此。
6. **C1 是指纹级一致**，不是原始字节比（上游 `frames/*.npz` 已在上一轮清理时删除；
   本轮的 `frames/*.npz` 同样已按惯例清掉，只留 `report.json` / `validation.json` / `pool.json` 这些派生读数）。
7. 77 次非 FFN 的 `select()` 调用**未归因**（不影响判据：判据是两臂之差）。
8. **不改画面 ⇒ 本轮不落地。** `rows_scopes_v1.py` 与 `c512_int8_ffn_rows_v1.py` 磁盘上一个字节没动；
   若将来要落地，**唯一的改动面**是给 `rows_scopes_v1.c512_ffn` 加一个默认关的环境变量分支
   （本轮已经把这段代码写完并验过，见 `fuse_group_patch_v1.py` 的 `B_LAUNCH`）。
9. **没有设备侧逐内核读数**：本轮没装 profiler（装了就会污染帧墙）。所以「QH 少 29.3 MB」
   是**字节账**，不是实测带宽。

---

## §10 下一步

| 级别 | 动作 | 本轮新增的依据 | 预期 | 改画面？ |
|---|---|---|---|---|
| **A1** | 接线 `_groups`（默认关的环境变量分支） | **本轮：等价 + 零溢出 + 0.308 ms** | **0.31 ms** | **否** |
| A2 | 融 `batched_branched_mlp_v1` 的 `_pairs` + `_project`（36 次/帧，设备 4.39 ms） | `merge_survey_v1` §5 | 未量 | 否（要先过同款等价门） |
| **B** | **L3-1 拆包装**打在 10 个 fused/window 调用点（24.39 ms / 305 次启动） | `XMX_ABLATION` 自证 **8.0×**；`l3_1_unwrap_v1/RESULT.md` | 唯一能动 3.6× 的 | 可能是 |
| C | L3-3 改算术/结构 | —— | —— | **是** ⇒ 必须批 |

**我的建议：A 到此为止。** A1 已经量到 0.31 ms、A2 要新写内核而预期同样在零点几毫秒量级
—— 把预算继续投在 A 上是**用开发量换 0.3 ms**。
真正该动的是 **B**，而 B 之前卡着你的两件事：

1. **D5 第 3 项（数值容差门是否数值化）** —— 不定就没有验收标尺；
2. **B 到底改不改画面** —— 若改，必须走你的审批。

---

## §11 复现

```bash
# 静态自检（CPU-only，零租约）
python -m py_compile reference/fuse_group_v1/*.py
python reference/fuse_group_v1/fuse_group_patch_v1.py --check     # 锚点唯一 + 可编译 + arm 调度
python reference/fuse_group_v1/fuse_group_entry_v1.py --check     # 锚点唯一 + 跨锚点变量名自洽

# 两次跑（各一个短租约，奇偶对调）
NR_FFN_GROUP_MODE=seq NR_FFN_GROUP_SWAP=0 LIMIT=40 TAG=seq40 \
  bash reference/fuse_group_v1/fuse_group_run_v1.sh
NR_FFN_GROUP_MODE=seq NR_FFN_GROUP_SWAP=1 LIMIT=80 TAG=seq80swap \
  RUN=D:/fuse-group-v1/r1 bash reference/fuse_group_v1/fuse_group_run_v1.sh

# 离线（不占 GPU，幂等）
python reference/fuse_group_v1/fuse_group_report_v1.py --dir D:/fuse-group-v1/r1/seq40 \
  --json D:/fuse-group-v1/r1/seq40/report.json --md D:/fuse-group-v1/r1/seq40/report.md
python reference/fuse_group_v1/fuse_group_pool_v1.py \
  --run D:/fuse-group-v1/r1/seq40 --run D:/fuse-group-v1/r1/seq80swap \
  --json D:/fuse-group-v1/r1/pool.json
```

**产物**：`PLAN.md`（事前登记）、`RESULT.md` / `RESULT.html`（本报告）、
`fuse_group_patch_v1.py`（内存替换 `rows_scopes_v1` + dual 记录器 + arm 调度）、
`fuse_group_entry_v1.py`（`materials_entry_v1.py` 的补丁入口）、
`fuse_group_run_v1.sh`（租约驱动）、`fuse_group_report_v1.py`（C1–C7 + 帧墙）、
`fuse_group_pool_v1.py`（两次跑合并）、`fuse_group_html_v1.py`（md→html + mdsafe 自检）。
数据：`D:/fuse-group-v1/r1/{seq40,seq80swap}/`、`pool.json`。
产品源码 **0 改动**、冻结缓存 **0 改动**。
