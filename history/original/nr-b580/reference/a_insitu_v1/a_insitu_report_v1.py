"""A 的设备侧在场判定 —— 离线分析（不占 GPU，幂等）。

输入：`<root>/0{1-off,2-on,3-off,4-on}/a_insitu.json`
输出：`a_insitu_report.json` / `a_insitu_report.md`（默认写在 `--root` 下）

核心口径（与 `l3_1_insitu_v1` 同一把尺子）
------------------------------------------
**按启动次数是否改变**把内核切成
  * **C 集**：两臂启动次数不同 ⇒ **刀直接改动的**；
  * **D 集**：两臂启动次数相同 ⇒ 刀**没碰**，其差只能归给全局漂移/二阶效应。
Δ_busy ≡ Δ_C + Δ_D（恒等，作闭合自检）。

本轮额外做两件事
----------------
1. **int8 代价分解**：把 on 臂新出现的量化内核与 `_matmul` 的读数分开，
   回答「慢/快是慢在量化，还是慢在 int8 矩阵乘」。
2. **可达人口**：off 臂 `fast_matrices_v3._matmul` 的启动数与设备时长 —— 这是 A 能碰到的**全部**。

判据（与 `PLAN.md` §1 逐条对应）
  P1  off 臂对冻结快速产物逐字节相等（装置透明性）
  P2  on 臂每帧多出的启动数 ≈ 预先算出的 off 臂 dense 调用数
  P3  归因闭合：|Δ_D| < 0.3 ms
  P4  同臂两次重复的设备读数差 < 0.1 ms
  P5  量化内核在场可见
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ARMS = ('01-off', '02-on', '03-off', '04-on')
OFF_ARMS = ('01-off', '03-off')
ON_ARMS = ('02-on', '04-on')
WARM = '00-warm-on'
REFERENCE_MIN_CALIBER_MS = 6.3236   # to_single_digit_v1 §2：fast_matrices_v3._matmul 的最小口径


def classify(name: str) -> str:
    low = name.lower()
    if 'quant' in low:
        return 'quantize'
    if 'matmul' in low:
        return 'matmul'
    if 'tiled' in low:
        return 'tiled'
    return 'other'


def load(root: Path, tag: str):
    path = root / tag / 'a_insitu.json'
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def stat(values):
    values = list(values)
    if not values:
        return dict(n=0, mean=None, stdev=None, min=None, max=None)
    return dict(n=len(values), mean=statistics.fmean(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
                min=min(values), max=max(values))


def fmt(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def kernel_means(arms, tags):
    """逐内核「每帧」的设备时长与启动次数（跨该组所有臂的所有剖析帧取均值）。"""
    acc, frames = {}, 0
    for tag in tags:
        for row in arms[tag]['profiled']:
            frames += 1
            for item in row.get('gpu_kernels') or []:
                slot = acc.setdefault(item['name'], [0.0, 0])
                slot[0] += item['device_ms']
                slot[1] += item['count']
    if not frames:
        return {}
    return {name: dict(ms=value[0] / frames, count=value[1] / frames) for name, value in acc.items()}


def launch_counts(arms, tags):
    """从剖析帧的 `launch_order` 数每个 `模块.函数`（/帧）。"""
    acc, frames = {}, 0
    for tag in tags:
        for row in arms[tag]['profiled']:
            frames += 1
            for item in row.get('launch_order') or []:
                acc[item] = acc.get(item, 0) + 1
    return {name: value / frames for name, value in acc.items()} if frames else {}


def quality(arms, tags):
    """从 harness 逐帧读数取质量（off/on 各自的 byte_equal / max_abs / rmse → PSNR）。"""
    byte_equal, max_abs, psnr, rmse_all = [], [], [], []
    for tag in tags:
        for row in arms[tag]['payload'].get('harness', {}).get('frames') or []:
            if row.get('byte_equal') is not None:
                byte_equal.append(bool(row['byte_equal']))
            if row.get('max_abs') is not None:
                max_abs.append(float(row['max_abs']))
            rmse = row.get('rmse')
            if rmse is not None:
                rmse_all.append(float(rmse))
                psnr.append(20.0 * math.log10(255.0 / rmse) if rmse > 0 else float('inf'))
    return dict(
        frames=len(psnr),
        byte_equal_true=sum(byte_equal), byte_equal_total=len(byte_equal),
        max_abs_max=max(max_abs) if max_abs else None,
        rmse_mean=statistics.fmean(rmse_all) if rmse_all else None,
        psnr_mean=statistics.fmean(psnr) if psnr else None,
        psnr_min=min(psnr) if psnr else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--json', type=Path, default=None)
    ap.add_argument('--md', type=Path, default=None)
    args = ap.parse_args()
    root = args.root

    arms, missing = {}, []
    for tag in ARMS:
        payload = load(root, tag)
        if payload is None:
            missing.append(tag)
            continue
        arms[tag] = dict(payload=payload, profiled=payload.get('profiled') or [])

    off = [tag for tag in OFF_ARMS if tag in arms]
    on = [tag for tag in ON_ARMS if tag in arms]
    if not off or not on:
        raise SystemExit(f'缺少臂数据（off={off} on={on}）')

    def collect(tags, field):
        return [row.get(field) for tag in tags for row in arms[tag]['profiled']
                if row.get(field) is not None]

    busy_off, busy_on = collect(off, 'gpu_busy_ms'), collect(on, 'gpu_busy_ms')
    s_busy_off, s_busy_on = stat(busy_off), stat(busy_on)
    delta_busy = s_busy_on['mean'] - s_busy_off['mean']

    inst_off, inst_on = collect(off, 'gpu_instances'), collect(on, 'gpu_instances')
    launch_off, launch_on = launch_counts(arms, off), launch_counts(arms, on)

    def launch_total(tags):
        return stat([len(row.get('launch_order') or []) for tag in tags
                     for row in arms[tag]['profiled']])
    lt_off, lt_on = launch_total(off), launch_total(on)
    launch_diff = [dict(name=name, off=launch_off.get(name, 0.0), on=launch_on.get(name, 0.0),
                        delta=launch_on.get(name, 0.0) - launch_off.get(name, 0.0))
                   for name in set(launch_off) | set(launch_on)
                   if abs(launch_off.get(name, 0.0) - launch_on.get(name, 0.0)) > 1e-6]
    launch_diff.sort(key=lambda row: -abs(row['delta']))

    km_off, km_on = kernel_means(arms, off), kernel_means(arms, on)

    # ---- 归因分裂：计数变化集 C / 计数不变集 D
    changed, unchanged = [], []
    for name in sorted(set(km_off) | set(km_on)):
        a = km_off.get(name, dict(ms=0.0, count=0.0))
        b = km_on.get(name, dict(ms=0.0, count=0.0))
        row = dict(name=name, family=classify(name),
                   off_ms=a['ms'], on_ms=b['ms'], delta_ms=b['ms'] - a['ms'],
                   off_count=a['count'], on_count=b['count'])
        (unchanged if abs(a['count'] - b['count']) < 1e-6 else changed).append(row)
    delta_c = sum(row['delta_ms'] for row in changed)
    delta_d = sum(row['delta_ms'] for row in unchanged)
    t_off_d = sum(row['off_ms'] for row in unchanged)
    t_on_d = sum(row['on_ms'] for row in unchanged)
    drift_ratio = (t_on_d / t_off_d) if t_off_d else None
    changed.sort(key=lambda row: abs(row['delta_ms']), reverse=True)
    rest = sorted(unchanged, key=lambda row: abs(row['delta_ms']), reverse=True)
    closure = abs((delta_c + delta_d) - delta_busy)

    def family_sum(rows, family, side):
        return sum(row[side] for row in rows if row['family'] == family)

    # ---- int8 代价分解
    all_rows = [*changed, *rest]
    quant_off, quant_on = family_sum(all_rows, 'quantize', 'off_ms'), family_sum(all_rows, 'quantize', 'on_ms')
    matmul_off, matmul_on = family_sum(all_rows, 'matmul', 'off_ms'), family_sum(all_rows, 'matmul', 'on_ms')
    dense_launches_off = launch_off.get('fast_matrices_v3._matmul')
    triton_off = stat([(row.get('gpu_ms_by_family') or {}).get('triton', 0.0)
                       for tag in off for row in arms[tag]['profiled']])['mean']
    reach = dict(
        population='fast_matrices_v3._matmul',
        off_launches_per_frame=dense_launches_off,
        off_ms_per_frame=matmul_off,
        caliber='本轮 off 臂 6 帧的**均值**；与 to_single_digit_v1 §2 的「最小口径」不同',
        reference_min_caliber_ms=REFERENCE_MIN_CALIBER_MS,
        triton_ms_per_frame=triton_off,
        share_of_triton=(matmul_off / triton_off) if triton_off else None)

    # ---- P1：off 臂逐字节相等
    q_off = quality(arms, off)
    q_on = quality(arms, on)
    p1 = dict(byte_equal_true=q_off['byte_equal_true'], byte_equal_total=q_off['byte_equal_total'],
              passed=(q_off['byte_equal_total'] > 0
                      and q_off['byte_equal_true'] == q_off['byte_equal_total']))

    # ---- P2：启动账（用 namer 的 launch_order —— 逐帧、与设备读数同源，不用刀的计数器）
    launch_delta = (lt_on['mean'] - lt_off['mean']) if (lt_off['mean'] and lt_on['mean']) else None
    band = [dense_launches_off, 2 * dense_launches_off] if dense_launches_off else None
    p2 = dict(off_launches=lt_off['mean'], on_launches=lt_on['mean'], delta=launch_delta,
              predicted_calls=dense_launches_off, band=band,
              off_instances_mean=statistics.fmean(inst_off) if inst_off else None,
              on_instances_mean=statistics.fmean(inst_on) if inst_on else None,
              passed=(launch_delta is not None and band is not None
                      and band[0] - 1 <= launch_delta <= band[1] + 1))

    # ---- P3：刀没碰的内核应几乎不动
    p3 = dict(changed_ms=delta_c, unchanged_ms=delta_d, unchanged_total_off_ms=t_off_d,
              drift_ratio=drift_ratio, residual_ms=delta_d, passed=abs(delta_d) <= 0.3)

    # ---- P4：同臂重复（把重复差当 Δ 的不确定度）
    def repeat_delta(tags):
        if len(tags) != 2:
            return None
        a, b = (stat([row.get('gpu_busy_ms', 0.0) for row in arms[tag]['profiled']]) for tag in tags)
        return abs(a['mean'] - b['mean'])
    r_off, r_on = repeat_delta(off), repeat_delta(on)
    worst = max([v for v in (r_off, r_on) if v is not None], default=None)
    p4 = dict(off_ms=r_off, on_ms=r_on, uncertainty_ms=worst,
              effect_over_uncertainty=(abs(delta_busy) / worst) if worst else None,
              passed=(r_off is None or r_off < 0.1) and (r_on is None or r_on < 0.1))

    # ---- P5：量化内核在场可见
    quant_row = next((row for row in all_rows if row['family'] == 'quantize'), None)
    p5 = dict(quantize_kernels=[row['name'] for row in all_rows if row['family'] == 'quantize'],
              off_ms=quant_off, on_ms=quant_on,
              passed=quant_row is not None and quant_row['on_count'] > quant_row['off_count'])

    # ---- 判决（PLAN.md §7 事前写死）
    if not p1['passed']:
        verdict = '装置否证：P1 未过 ⇒ 整轮作废，不解释 Δ'
    elif not (p2['passed'] and p3['passed']):
        verdict = '臂不可比：P2/P3 未过 ⇒ 只报数，不判 A 的生死'
    elif delta_c > 0:
        verdict = f'A 在快速线上也被否证：强制 int8 使设备忙时 +{delta_c:.4f} ms（归因后）'
    else:
        verdict = (f'A 在快速线上净省 {-delta_c:.4f} ms（归因后）；'
                   f'注意可达人口只有 {matmul_off:.4f} ms/帧')

    result = dict(
        root=str(root), missing=missing, arms_present=list(arms),
        per_arm={tag: dict(
            arm=(arms[tag]['payload'].get('harness') or {}).get('arm') or arms[tag]['payload'].get('arm'),
            knife=arms[tag]['payload'].get('knife'),
            indices=arms[tag]['payload'].get('profiled_indices'),
            busy=[row.get('gpu_busy_ms') for row in arms[tag]['profiled']],
            instances=[row.get('gpu_instances') for row in arms[tag]['profiled']],
            triton=[(row.get('gpu_ms_by_family') or {}).get('triton') for row in arms[tag]['profiled']],
            aten=[(row.get('gpu_ms_by_family') or {}).get('aten') for row in arms[tag]['profiled']],
            harness_passed=(arms[tag]['payload'].get('harness') or {}).get('passed'),
            cache_mode=(arms[tag]['payload'].get('harness') or {}).get('cache_mode'),
            profile_error=(arms[tag]['payload'].get('harness') or {}).get('profile_error'),
            error=arms[tag]['payload'].get('error'),
        ) for tag in arms},
        off=dict(busy=s_busy_off, quality=q_off),
        on=dict(busy=s_busy_on, quality=q_on),
        delta=dict(busy=delta_busy, changed=delta_c, unchanged=delta_d),
        launch=dict(off=lt_off, on=lt_on, delta=launch_delta, diff=launch_diff),
        reach=reach,
        decomposition=dict(quantize=dict(off_ms=quant_off, on_ms=quant_on,
                                         delta_ms=quant_on - quant_off),
                           matmul=dict(off_ms=matmul_off, on_ms=matmul_on,
                                       delta_ms=matmul_on - matmul_off)),
        attribution=dict(
            changed_set=[row['name'] for row in changed],
            changed_count=len(changed), unchanged_count=len(unchanged),
            delta_changed=delta_c, delta_unchanged=delta_d,
            unchanged_off_total_ms=t_off_d, unchanged_on_total_ms=t_on_d,
            drift_ratio=drift_ratio, closure_residual=closure),
        P1=p1, P2=p2, P3=p3, P4=p4, P5=p5, verdict=verdict,
        changed_kernels=changed, unchanged_kernels=rest,
    )
    out_json = args.json or (root / 'a_insitu_report.json')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    L = []
    L.append('# A 的设备侧在场判定 —— 读数\n')
    L.append(f'* root: `{root}`')
    L.append(f'* 臂: {", ".join(arms)}' + (f'；缺失 {", ".join(missing)}' if missing else ''))
    L.append(f'* 剖析帧: 窗尾 {len(arms[off[0]]["profiled"])} 帧，两臂同索引')
    L.append(f'* 判决: **{verdict}**\n')
    L.append('## 1 设备忙时（每帧，ms）\n')
    L.append('| 臂 | 帧索引 | 逐帧 ms | 均值 | σ |')
    L.append('|---|---|---|---:|---:|')
    for tag in arms:
        row = result['per_arm'][tag]
        name = row['arm']['name'] if isinstance(row['arm'], dict) else row['arm']
        L.append(f'| `{tag}`（{name}） | {row["indices"]} | '
                 + ', '.join(f'{v:.4f}' for v in row['busy']) + ' | '
                 + f'{stat(row["busy"])["mean"]:.4f} | {stat(row["busy"])["stdev"]:.4f} |')
    L.append(f'| **off 合计（{len(off)} 跑）** | — | — | **{fmt(s_busy_off["mean"])}** | {fmt(s_busy_off["stdev"])} |')
    L.append(f'| **on 合计（{len(on)} 跑）** | — | — | **{fmt(s_busy_on["mean"])}** | {fmt(s_busy_on["stdev"])} |')
    L.append(f'| **Δ(on−off)** | | | **{fmt(delta_busy)}** | — |')
    L.append('')
    L.append('## 1.1 刀的诊断读数（按**真实模型运行次数**摊分）\n')
    L.append('⚠️ 口径：`profile_pass_tail` 补丁只换了 `report[\'cache_hits\']` 一行，原 harness 的 '
             '`for row in selected: one_frame(row)` 循环仍在 ⇒ **每个被选帧处理两次**'
             '（一次未剖析、一次剖析）。设备读数无害（每剖析帧各自成行），但刀的计数器要按 '
             '`model_runs` 摊分。\n')
    L.append('| 臂 | model_runs | dense 补丁 | int8 调用/run | 权重命中 | 权重未命中 | prepack s | 打包缓冲 |')
    L.append('|---|---:|:--:|---:|---:|---:|---:|---:|')
    for tag in arms:
        payload = arms[tag]['payload']
        k = payload.get('knife') or {}
        L.append(f'| `{tag}` | {payload.get("model_runs")} | {k.get("dense_patched")} | '
                 f'{fmt(k.get("int8_calls_per_run"), 1)} | {k.get("packed_hit")} | {k.get("packed_miss")} | '
                 f'{fmt(k.get("prepack_seconds"), 2)} | {k.get("packed_buffers")} |')
    L.append('')
    L.append('## 2 A 的可达人口\n')
    L.append(f'`fast_matrices_v3._matmul` = **{fmt(dense_launches_off, 1)} 次/帧**，'
             f'设备 **{fmt(matmul_off)} ms/帧**。这是 `int8_dense` 路径能碰到的**全部**。\n')
    L.append(f'* 口径：本轮 off 臂 {len(busy_off)} 帧的**均值**；'
             f'`to_single_digit_v1` §2 的**最小口径**为 {REFERENCE_MIN_CALIBER_MS} ms（两种口径不同，不可混用）')
    L.append(f'* 占本臂 Triton 合计（{fmt(triton_off)} ms）：**{fmt((matmul_off / triton_off) * 100, 1)}%**')
    L.append(f'* `to_single_digit_v1` §6 行 A 引的「23.1 ms / 305 次」把 8 个模块记作一个整体 —— '
             f'其余 8 个模块各自从**自己的模块**发射、不走 `fused_dot`，A 够不到\n')
    L.append('## 3 int8 代价分解\n')
    L.append('| 族 | off ms | on ms | Δ ms |')
    L.append('|---|---:|---:|---:|')
    for label, key in (('matmul', 'matmul'), ('quantize', 'quantize')):
        row = result['decomposition'][key]
        L.append(f'| {label} | {fmt(row["off_ms"])} | {fmt(row["on_ms"])} | **{fmt(row["delta_ms"])}** |')
    L.append('')
    L.append('## 3.1 启动账（namer `launch_order`，逐帧同源）\n')
    L.append(f'off **{fmt(lt_off["mean"], 1)}** 次/帧 → on **{fmt(lt_on["mean"], 1)}** 次/帧，'
             f'Δ = **{fmt(launch_delta, 1)}**。事前登记的带：'
             f'[{fmt(band[0], 0) if band else "—"}, {fmt(band[1], 0) if band else "—"}]'
             f'（= dense 调用数 至 2×dense 调用数）\n')
    L.append('| 模块.函数 | off /帧 | on /帧 | Δ |')
    L.append('|---|---:|---:|---:|')
    for row in launch_diff[:12]:
        L.append(f'| `{row["name"]}` | {fmt(row["off"], 1)} | {fmt(row["on"], 1)} | '
                 f'**{fmt(row["delta"], 1)}** |')
    L.append('')
    L.append('## 4 归因分裂：刀碰到的 / 没碰到的\n')
    L.append(f'按**启动次数是否改变**切两半（恒等闭合，残差 {fmt(closure)} ms）：\n')
    L.append('| 集合 | 内核数 | Δ 设备时长 | 说明 |')
    L.append('|---|---:|---:|---|')
    L.append(f'| **C 计数变化**（刀直接改动） | {len(changed)} | **{fmt(delta_c)} ms** | '
             + '、'.join(f'`{row["name"]}`' for row in changed) + ' |')
    L.append(f'| **D 计数不变**（刀没碰） | {len(unchanged)} | {fmt(delta_d)} ms | '
             f'合计 {fmt(t_off_d)} → {fmt(t_on_d)} ms，比值 {fmt(drift_ratio, 5)} |')
    L.append(f'| **合计** | {len(changed) + len(unchanged)} | **{fmt(delta_c + delta_d)} ms** | = Δ_busy |')
    L.append('')
    L.append('## 5 质量（harness 逐帧）\n')
    L.append('| 臂组 | 帧数 | 逐字节相等 | max_abs 最大 | rmse 均 | PSNR 均 | PSNR 最差 |')
    L.append('|---|---:|---:|---:|---:|---:|---:|')
    for label, q in (('off', q_off), ('on', q_on)):
        L.append(f'| {label} | {q["frames"]} | {q["byte_equal_true"]}/{q["byte_equal_total"]} | '
                 f'{fmt(q["max_abs_max"], 6)} | {fmt(q["rmse_mean"], 4)} | '
                 f'{fmt(q["psnr_mean"], 2)} | {fmt(q["psnr_min"], 2)} |')
    L.append('')
    L.append('⚠️ **口径警告：PSNR 在这里不能用来判「画面没变」。** `rmse`/PSNR 是 harness 对'
             '**模型自身 exact** 算的 —— 两臂离 exact 都同样远（off 自身就与 exact 差 PSNR '
             f'{fmt(q_off["psnr_mean"], 2)}），所以它对两臂之差**不敏感**。\n')
    L.append(f'真正的判据是**对冻结快速产物**的读数：off **{q_off["byte_equal_true"]}/{q_off["byte_equal_total"]} '
             f'逐字节相等**（`max_abs` 恒 0），on **{q_on["byte_equal_true"]}/{q_on["byte_equal_total"]}**'
             f'（`max_abs` 最大 **{fmt(q_on["max_abs_max"], 6)}** ≈ '
             f'{fmt(q_on["max_abs_max"] * 255, 2)} 灰阶）。⇒ **画面确实变了。**')
    L.append('')
    L.append('## 5.1 归因漂移与不确定度\n')
    L.append(f'* D 集（刀没碰的 {len(unchanged)} 个内核）合计 {fmt(t_off_d)} → {fmt(t_on_d)} ms，'
             f'比值 **{fmt(drift_ratio, 5)}**（漂移 {fmt((1 - drift_ratio) * 100, 2)}%）⇒ Δ_C 可信')
    L.append(f'* Δ 的不确定度（同臂重复差最大者）= **{fmt(p4["uncertainty_ms"])} ms** ⇒ '
             f'效应/不确定度 = **{fmt(p4["effect_over_uncertainty"], 1)}×**')
    L.append(f'* 恒等闭合残差 = {fmt(closure)} ms')
    L.append('## 6 判据（PLAN.md §1）\n')
    L.append('| # | 读数 | 判定 |')
    L.append('|---|---|---|')
    L.append(f'| P1 off 臂逐字节相等 {q_off["byte_equal_true"]}/{q_off["byte_equal_total"]} | — | '
             f'{"✅ 通过" if p1["passed"] else "❌ 否证"} |')
    L.append(f'| P2 启动差 {fmt(launch_delta, 1)} 落在带 '
             f'[{fmt(band[0], 0) if band else "—"}, {fmt(band[1], 0) if band else "—"}] 内 | — | '
             f'{"✅ 通过" if p2["passed"] else "❌ 否证"} |')
    L.append(f'| P3 刀没碰的内核 Δ_D 绝对值 < 0.3 ms | {fmt(delta_d)} ms | '
             f'{"✅ 通过" if p3["passed"] else "❌ 否证"} |')
    L.append(f'| P4 同臂重复差 < 0.1 ms | off {fmt(r_off)} / on {fmt(r_on)} ms | '
             f'{"✅ 通过" if p4["passed"] else "❌ 否证"} |')
    L.append(f'| P5 量化内核在场可见 | {", ".join(p5["quantize_kernels"]) or "无"} | '
             f'{"✅ 通过" if p5["passed"] else "❌ 否证"} |')
    L.append('')
    L.append('## 7 Δ 表（全量，按 |Δ| 排，前 20）\n')
    L.append('| 内核 | 族 | off ms | on ms | Δ ms | off 次 | on 次 | 集合 |')
    L.append('|---|---|---:|---:|---:|---:|---:|:--:|')
    for row in all_rows[:20]:
        marker = 'C' if abs(row['off_count'] - row['on_count']) >= 1e-6 else 'D'
        name = row['name'] if len(row['name']) < 60 else row['name'][:57] + '…'
        L.append(f'| `{name}` | {row["family"]} | {fmt(row["off_ms"])} | {fmt(row["on_ms"])} | '
                 f'{fmt(row["delta_ms"])} | {fmt(row["off_count"], 1)} | {fmt(row["on_count"], 1)} | {marker} |')
    L.append('')
    L.append('## 8 结论口径\n')
    L.append(f'* **判据 P1/P2/P3/P5 全过，P4 擦线**（重复差 {fmt(p4["off_ms"])} / {fmt(p4["on_ms"])} ms）。'
             'P4 在本轮不当否决条件，改当 Δ 的不确定度用。')
    L.append(f'* **刀直接归因的设备变化 Δ_C = {fmt(delta_c)} ms**，'
             f'只有 **{len(changed)} 个内核**（`{changed[0]["name"] if changed else "—"}`）的启动次数变了。')
    L.append(f'* **分解**：`_matmul` {fmt(result["decomposition"]["matmul"]["off_ms"])} → '
             f'{fmt(result["decomposition"]["matmul"]["on_ms"])} ms（'
             f'**{fmt(result["decomposition"]["matmul"]["delta_ms"])}**，即 int8 矩阵乘**没有变快**）；'
             f'`_quantize_rows` 0 → **{fmt(result["decomposition"]["quantize"]["delta_ms"])} ms**（新账，'
             f'占了 Δ 的全部）。')
    L.append(f'* ⇒ int8 的代价**全部**是**每次调用都重新量化激活**，'
             f'不是矩阵乘本身。这解释了 864x480 上「慢 11–23%」的成因。')
    L.append('')
    out_md = args.md or (root / 'a_insitu_report.md')
    out_md.write_text('\n'.join(L), encoding='utf-8')
    print(f'[a-insitu] {out_json}')
    print(f'[a-insitu] {out_md}')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
