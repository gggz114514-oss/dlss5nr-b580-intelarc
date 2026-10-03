"""A 的对账 —— 用既有落盘证据复检「8 个 C32 dense 走 INT8 XMX」。

零 GPU、零产品改动、幂等。只读 `fast-precision-864x480-v1/validation.json`。
判据与决策规则见 `PLAN.md` §1。
"""
from __future__ import annotations
import argparse, hashlib, json, statistics as st, sys
from pathlib import Path

REF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
CASE = REF / 'results' / 'fast-precision-864x480-v1' / 'validation.json'
TARGET = Path('E:/ComfyUI-aki-v3-IntelArc_20260722/nr-b580-int8/experimental/fast_matrices_v3.py')
PROVIDER_NAME = 'fast_matrices_v3.py'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def warm_mean(rows, key='seconds'):
    """帧 1..12 的均值（丢冷启动帧 0）。与报告自报的 matching_warm_mean_seconds 口径一致。"""
    vals = [r[key] for r in rows[1:]]
    return st.mean(vals), len(vals)


def load(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else CASE
    if not path.exists():
        raise SystemExit(f'找不到证据文件：{path}')
    return json.loads(path.read_text(encoding='utf-8'))


def analyse(d: dict) -> dict:
    out: dict = {}

    # ---- Q2：源码同一性 -------------------------------------------------
    recorded = {Path(k).name: v for k, v in d['sources'].items()}
    live = sha256(TARGET) if TARGET.exists() else None
    rec = recorded.get(PROVIDER_NAME)
    out['q2_source_identity'] = dict(
        provider=PROVIDER_NAME,
        recorded=rec,
        live=live,
        identical=(rec is not None and rec == live),
    )
    changed = []
    for key, h in d['sources'].items():
        p = Path(key)
        if p.exists() and sha256(p) != h:
            changed.append(p.name)
    out['q2_changed_siblings'] = sorted(changed)

    # ---- Q3 / Q4：时序与质量 --------------------------------------------
    arms = list(d['runs'])
    timing, quality = {}, {}
    for mode, rows in d['runs'].items():
        mean, n = warm_mean(rows)
        timing[mode] = dict(warm_mean_s=mean, warm_frames=n,
                            reported_s=d['matching_warm_mean_seconds'].get(mode),
                            paired_speedup=d['paired_speedup'].get(mode),
                            per_frame_s=[r['seconds'] for r in rows])
        ps = [r['psnr_db'] for r in rows if r.get('psnr_db') is not None]
        ss = [r['ssim'] for r in rows if r.get('ssim') is not None]
        mx = [r['max_abs'] for r in rows if r.get('max_abs') is not None]
        quality[mode] = dict(
            psnr_mean=st.mean(ps) if ps else None,
            psnr_worst=min(ps) if ps else None,
            ssim_worst=min(ss) if ss else None,
            max_abs_median=st.median(mx) if mx else None,
            byte_equal=sum(1 for r in rows if r.get('byte_equal')),
            frames=len(rows),
        )
    out['timing'] = timing
    out['quality'] = quality

    # ---- Q3 判定 ---------------------------------------------------------
    base, f16, i8 = timing['baseline']['warm_mean_s'], timing['fp16_xmx']['warm_mean_s'], timing['int8_dense']['warm_mean_s']
    # 第二口径：剔除各臂的偏热帧（帧 1），只用帧 2..12
    def trim(rows):
        return st.mean([r['seconds'] for r in rows[2:]]), len(rows) - 2
    t_base, t_f16, t_i8 = trim(d['runs']['baseline']), trim(d['runs']['fp16_xmx']), trim(d['runs']['int8_dense'])
    out['q3'] = dict(
        baseline_s=base, fp16_xmx_s=f16, int8_dense_s=i8,
        int8_vs_fp16_ratio=i8 / f16,                      # >1 ⇒ int8 更慢
        int8_slower_than_fp16=(i8 > f16),
        pct_slower=(i8 / f16 - 1.0) * 100.0,
        trimmed=dict(frames=t_f16[1], baseline_s=t_base[0], fp16_xmx_s=t_f16[0], int8_dense_s=t_i8[0],
                     int8_vs_fp16_ratio=t_i8[0] / t_f16[0],
                     pct_slower=(t_i8[0] / t_f16[0] - 1.0) * 100.0),
    )

    # ---- Q5：这份三臂数据到底还能推出什么（严格说：推不出多少）------------
    # B, F, G 三个可观测量；模型 R + D/f = F, R + D/g = G, R + D = B 有 4 个未知量
    # ⇒ 1 个自由度。唯一不依赖该自由度的结论是下面两条不等式。
    s16 = base - f16
    s8 = base - i8
    out['q5'] = dict(
        saving_fp16_s=s16, saving_int8_s=s8,
        r_over_s16=s8 / s16,
        ratio_is_identity=True,
        identity='G/F = β − r·(β−1)，β=B/F、r=(B−G)/(B−F) 均为可观测量 ⇒ 恒等式，非独立证据',
        f_lower_bound=base / f16,                 # dense 占比 ≤ 1 ⇒ f ≥ B/F
        dense_share_lower_bound=s16 / base,       # 若备选实现「免费」则 dense ≥ S16/B
        note='单靠三臂数据无法把「int8 在 dense 上比 fp16 慢多少」与「dense 占比」分开；'
             '要分开必须再加一个独立可观测量（例如直接测设备侧）',
    )

    # ---- 元数据 ----------------------------------------------------------
    out['meta'] = dict(
        case=str(CASE), dimension=d['dimension'], scope=d['scope'],
        timing_scope=d['timing_scope'], execution_orders=len(d['execution_orders']),
        passed=d['passed'], promoted=d['promoted'],
        complete_migration=d['complete_migration'], human_review=d['human_review'],
        prepack_seconds=d['prepack_seconds'], prepacked_buffers=d['prepacked_buffers'],
        prepacked_bytes=d['prepacked_bytes'],
        inputs_unchanged=d['inputs_unchanged'],
    )
    return out


def render_md(a: dict) -> str:
    L: list[str] = []
    L.append('# A 的对账 —— 用既有落盘证据复检「8 个 C32 dense 走 INT8 XMX」\n')
    m, q2, q3, q5 = a['meta'], a['q2_source_identity'], a['q3'], a['q5']
    t, q = a['timing'], a['quality']
    tr = q3['trimmed']

    L.append('## §0 一句话\n')
    L.append(f"`int8_dense` 在**同一装置**上被 `fp16_xmx` **严格支配**："
             f"**慢 {q3['pct_slower']:.1f}%（剔偏热帧后 {tr['pct_slower']:.1f}%）**"
             f"（暖帧 {q3['int8_dense_s']:.4f} vs {q3['fp16_xmx_s']:.4f} s），"
             f"且 PSNR 低 **{q['fp16_xmx']['psnr_mean'] - q['int8_dense']['psnr_mean']:.2f} dB**。")
    L.append(f"⇒ `to_single_digit_v1` §6 的「若 1.5–2.5× ⇒ 省 9–14 ms」（其自标为**量级估计**）"
             f"**方向与此相反，须视为未对账**。")
    L.append(f"⇒ 但**本轮不宣布 A 死**：装置不是快速线，只能把 A 从「推荐」降级为"
             f"「须先做一次可判定的设备侧测量」。\n")

    L.append('## §1 证据来源\n')
    L.append('| 项 | 值 |')
    L.append('|---|---|')
    L.append(f"| 文件 | `{m['case']}` |")
    L.append(f"| 维度 / 口径 | {m['dimension']}｜{m['scope']} |")
    L.append(f"| 计时口径 | {m['timing_scope']} |")
    L.append(f"| 轮转执行序 | {m['execution_orders']} 组 |")
    L.append(f"| passed / promoted / complete_migration | {m['passed']} / {m['promoted']} / {m['complete_migration']} |")
    L.append(f"| human_review | {m['human_review']} |")
    L.append(f"| prepack | {m['prepack_seconds']:.2f} s｜{m['prepacked_buffers']} 缓冲｜{m['prepacked_bytes']/2**20:.1f} MiB |")
    L.append('')

    L.append('## §2 Q2：这份证据作用的 provider 与今天是否同一份\n')
    L.append(f"- 记录时 `{q2['provider']}` 的 sha256：`{(q2['recorded'] or '?')[:16]}…`")
    L.append(f"- 当前磁盘上同一文件的 sha256：`{(q2['live'] or '?')[:16]}…`")
    L.append(f"- **判定：{'✅ 完全一致' if q2['identical'] else '❌ 已变'}**"
             f" ⇒ 「模式间比较」{'可逐字适用于今天的实现' if q2['identical'] else '只能记为历史版本证据'}")
    L.append(f"- 另有 {len(a['q2_changed_siblings'])} 个同批源文件已变：`{', '.join(a['q2_changed_siblings'])}`")
    L.append('  （这些变化同时作用于两臂，不改变模式间的相对结论，但会改变绝对量级）\n')

    L.append('## §3 Q3：时序（同装置、轮转顺序、三模型常驻）\n')
    L.append('| 臂 | 暖帧均值 s | 报告自报 s | 相对 baseline | 相对 fp16_xmx |')
    L.append('|---|---:|---:|---:|---:|')
    for mode in ('baseline', 'fp16_xmx', 'int8_dense'):
        d = t[mode]
        rel = d['paired_speedup']
        vs = '—' if mode == 'baseline' else f"{d['warm_mean_s'] / t['fp16_xmx']['warm_mean_s']:.4f}×"
        L.append(f"| `{mode}` | {d['warm_mean_s']:.4f} | {d['reported_s']:.4f} | {rel:.4f}× | {vs} |")
    L.append('')
    L.append(f"⇒ **`int8_dense` 比 `fp16_xmx` 慢 {q3['pct_slower']:.1f}%**"
             f"（{q3['int8_dense_s']:.4f} / {q3['fp16_xmx_s']:.4f} = {q3['int8_vs_fp16_ratio']:.4f}）。")
    tr = q3['trimmed']
    L.append(f"⇒ 剔除各臂的偏热帧后（只用帧 2..12）**慢 {tr['pct_slower']:.1f}%**"
             f"（{tr['int8_dense_s']:.4f} / {tr['fp16_xmx_s']:.4f} = {tr['int8_vs_fp16_ratio']:.4f}）。")
    L.append(f"两条口径**方向一致**；报告自报口径偏保守（`fp16_xmx` 的帧 1 为 0.834 s，"
             f"显著高于其帧 2..12 的 ~0.365 s，抬高了 fp16 的均值）。\n")
    L.append('逐帧秒（帧 0 是冷启动，不计入）：\n')
    L.append('| 臂 | ' + ' | '.join(f'f{i}' for i in range(13)) + ' |')
    L.append('|---|' + '---:|' * 13)
    for mode in ('baseline', 'fp16_xmx', 'int8_dense'):
        row = ' | '.join(f"{s:.3f}" for s in t[mode]['per_frame_s'])
        L.append(f"| `{mode}` | {row} |")
    L.append('')

    L.append('## §4 Q4：质量（相对 exact 基线）\n')
    L.append('| 臂 | PSNR 均 | PSNR 最差 | SSIM 最差 | max_abs 中位 | 逐字节相等帧 |')
    L.append('|---|---:|---:|---:|---:|---:|')
    for mode in ('baseline', 'fp16_xmx', 'int8_dense'):
        d = q[mode]
        f = lambda v: '—' if v is None else f'{v:.2f}'
        g = lambda v: '—' if v is None else f'{v:.4f}'
        h = lambda v: '—' if v is None else f'{v:.3e}'
        L.append(f"| `{mode}` | {f(d['psnr_mean'])} | {f(d['psnr_worst'])} | {g(d['ssim_worst'])} | "
                 f"{h(d['max_abs_median'])} | {d['byte_equal']}/{d['frames']} |")
    L.append('')
    L.append(f"⇒ **`int8_dense` 质量也更差**（PSNR 均 {q['int8_dense']['psnr_mean']:.2f} vs "
             f"{q['fp16_xmx']['psnr_mean']:.2f}，低 {q['fp16_xmx']['psnr_mean'] - q['int8_dense']['psnr_mean']:.2f} dB；"
             f"SSIM 最差 {q['int8_dense']['ssim_worst']:.4f} vs {q['fp16_xmx']['ssim_worst']:.4f}）。")
    L.append('⇒ 合起来：**`int8_dense` 更慢且更差 ⇒ 被 `fp16_xmx` 严格支配。**\n')

    L.append('## §5 Q5：这份三臂数据**还能推出什么**（严格说：推不出多少）\n')
    L.append('模型：`R + D/f = F`、`R + D/g = G`、`R + D = B`'
             '（`B/F/G` = 三臂实测，`D` = baseline dense 秒数，`R` = 其余，`f`/`g` = 两模式在 dense 上的加速比）。')
    L.append('三个可观测量、**四个未知量** ⇒ **只剩 1 个自由度**。\n')
    L.append(f"⚠️ **必须登记的一处自我更正**：我原本打算由 `β=B/F` 与 `r=(B−G)/(B−F)` 解出一个"
             f"「int8/fp16 dense 时间比」的保守界，写作 `r + (1−r)·f`。核对后发现 —— "
             f"`G/F = β − r·(β−1)` **是代数恒等式**（代入 `β`、`r` 即恒等于 `G/F`），"
             f"**它不携带任何新信息，不能当证据卖**。故本轮不用它。\n")
    L.append('真正能给出的、不依赖自由度选择的，只有两条不等式：\n')
    L.append('| 结论 | 值 | 依据 |')
    L.append('|---|---:|---|')
    L.append(f"| fp16 在 dense 上的加速比下界 `f ≥ B/F` | **≥ {q5['f_lower_bound']:.4f}×** | dense 占比不能超过 100% |")
    L.append(f"| dense 在 baseline 中的占比下界 | **≥ {q5['dense_share_lower_bound']*100:.1f}%** | 若备选实现「免费」 |")
    L.append('')
    L.append(f"⇒ **单靠这份三臂数据，无法把「int8 在 dense 上比 fp16 慢多少」与「dense 占比」分开**。")
    L.append(f"要分开，必须**再加一个独立可观测量** —— 最直接的就是**在快速线上直接测设备侧**。")
    L.append('这正是 §6 第 5 条给出的下一步。\n')

    L.append('## §6 诚实边界\n')
    L.append('1. **装置不是快速线**：该 `baseline` 为 0.6075 s/帧，快速线为 44.8 ms/帧，差 13.6 倍。')
    L.append('   **绝对毫秒禁止跨线引用**；本轮**不把任何「省多少 ms」折算到快速线**。')
    L.append('2. **能外推的只有模式间的比值**（两臂只差 provider 模式，轮转顺序已消偏）。')
    L.append('3. 该批 `sources` 里另有 **' + str(len(a['q2_changed_siblings'])) +
             ' 个**同批文件已变 ⇒ 绝对量级可能漂移；比值受影响较小但**未量化**。')
    L.append('4. 本轮**不宣布「A 死」**。它证明的是：`to_single_digit_v1` §6 的 9–14 ms **未经对账**，'
             '且既有同一实现上的实测**方向相反**。')
    L.append('5. 因此正确的下一步不是按 9–14 ms 立项，而是**用快速线自己的设备尺子做一次可判定的测量**'
             '（装置见 `l3_1_insitu_v1`；`FastMatrices.select(\'int8_dense\')` 与 `prepack()` 已就绪）。\n')

    L.append('## §7 复现\n')
    L.append('```bash')
    L.append('PY="C:/Users/REFERENCE_USER/.workbuddy/binaries/python/versions/3.13.12/python.exe"')
    L.append('"$PY" -B -X utf8 a_recheck_v1.py --json a_recheck.json --md a_recheck.md')
    L.append('```')
    return '\n'.join(L) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', default=str(CASE))
    p.add_argument('--json', default='a_recheck.json')
    p.add_argument('--md', default='a_recheck.md')
    a = p.parse_args(argv)
    d = load(a.case)
    res = analyse(d)
    Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    Path(a.md).write_text(render_md(res), encoding='utf-8')
    print(f"[a_recheck] Q2 源码一致 = {res['q2_source_identity']['identical']}")
    print(f"[a_recheck] int8_dense vs fp16_xmx = {res['q3']['int8_vs_fp16_ratio']:.4f}x "
          f"({'更慢' if res['q3']['int8_slower_than_fp16'] else '更快'}, {res['q3']['pct_slower']:+.1f}%)")
    print(f"[a_recheck] 写出 {a.json} / {a.md}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
