"""离线：把启动序里记下的 (模块名, grid, 调用点) 与 chrome trace 的设备时长 zip 起来。

纯后处理：不占 GPU、不建模型、不参与任何门槛。

zip 只在两条序列**等长**时才有效，所以这里**断言**而不是假设；
再用逐名计数、单流有序、时长闭合、逐名归组闭合四条交叉核对 ——
**任何一条不过就不出调用点表**。

用法
----
  python dense_ab_zip_v1.py --dir D:/dense-ab-v1/r1/probe \
      --json <out.json> --md <out.md>
"""
import argparse
import collections
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bare(name):
    """'nr_backend.triton_fp8._kernel' -> '_kernel'；裸名原样返回。"""
    return name.rsplit('.', 1)[-1] if '.' in name else name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path, required=True,
                        help='一轮的输出目录，含 validation.json 与 DENSE_AB_TRACE 写的 trace')
    parser.add_argument('--json', type=Path, default=None)
    parser.add_argument('--md', type=Path, default=None)
    parser.add_argument('--tolerance-ms', type=float, default=0.05,
                        help='逐名归组闭合的绝对容差（ms）')
    args = parser.parse_args()

    report = load(args.dir / 'validation.json')
    profiled = report.get('profiled')
    if not profiled:
        raise SystemExit('validation.json 里没有 profiled —— 诊断遍没跑成；'
                         '看 report["profile_error"]')

    order = profiled['launch_order']
    grids = profiled.get('launch_grids') or []
    sites = profiled.get('launch_sites') or []
    trace_info = profiled.get('trace') or {}
    if 'path' not in trace_info:
        raise SystemExit(f"trace 没导出：{trace_info}")
    trace = load(Path(trace_info['path']))
    events = trace['traceEvents'] if isinstance(trace, dict) else trace

    kernels = sorted((e for e in events if e.get('cat') == 'kernel'), key=lambda e: e['ts'])
    pids = sorted({e.get('pid') for e in kernels})
    tids = sorted({e.get('tid') for e in kernels})
    triton = [e for e in kernels if not e['name'].startswith('_ZTS')]
    aten = [e for e in kernels if e['name'].startswith('_ZTS')]

    gpu_busy_ms = profiled['gpu_busy_ms']
    keyavg = {row['name']: row for row in profiled['gpu_kernels']}
    trace_device_ms = sum(e['dur'] for e in kernels) / 1000.0

    out = dict(
        dir=str(args.dir),
        frame_index=profiled.get('index'),
        source_sha256=report.get('source_sha256'),
        dense_ab=report.get('dense_ab'),
        launch_fastpath_env=report.get('launch_fastpath_env'),
        cache_mode=report.get('cache_mode'),
        gpu_busy_ms=gpu_busy_ms,
        gpu_ms_by_family=profiled['gpu_ms_by_family'],
        gpu_instances_by_family=profiled['gpu_instances_by_family'],
        gpu_distinct_kernels=profiled['gpu_distinct_kernels'],
        trace_kernels=len(kernels), trace_triton=len(triton), trace_aten=len(aten),
        trace_pid=pids, trace_tid=tids,
        trace_device_ms=trace_device_ms,
        recorded_launches=len(order),
        recorded_grids=len(grids),
        recorded_sites=len(sites),
        note='诊断的后处理；不是证据、不参与门槛',
    )

    # ---------------- C1..C6 ----------------
    checks = {}

    # C3 单流有序（先于 zip，因为 zip 的前提就是它）
    checks['C3_single_in_order_queue'] = dict(
        ok=(len(pids) == 1 and len(tids) == 1), pid=pids, tid=tids)

    # C1 计数对齐
    checks['C1_count_match'] = dict(
        ok=(len(order) == len(triton)),
        recorded=len(order), trace_triton=len(triton))

    # C2 逐名计数对齐（比 C1 强：能抓出"总数对、名字错"）
    counter_order = collections.Counter(bare(name) for name in order)
    counter_trace = collections.Counter(e['name'] for e in triton)
    only_order = {k: v for k, v in (counter_order - counter_trace).items()}
    only_trace = {k: v for k, v in (counter_trace - counter_order).items()}
    checks['C2_per_name_count_match'] = dict(
        ok=(not only_order and not only_trace),
        recorded_names={k: v for k, v in sorted(counter_order.items())},
        trace_names={k: v for k, v in sorted(counter_trace.items())},
        only_in_recorded=only_order, only_in_trace=only_trace)

    # C4 时长闭合
    delta = abs(trace_device_ms - gpu_busy_ms)
    checks['C4_duration_closure'] = dict(
        ok=(delta < 0.01 * max(gpu_busy_ms, 1e-9)),
        trace_device_ms=trace_device_ms, gpu_busy_ms=gpu_busy_ms,
        delta_ms=delta, delta_pct=100.0 * delta / max(gpu_busy_ms, 1e-9))

    zip_valid = checks['C1_count_match']['ok'] and checks['C2_per_name_count_match']['ok'] \
        and checks['C3_single_in_order_queue']['ok']
    out['zip_valid'] = zip_valid

    have_sites = (len(sites) == len(order)) and len(order) > 0
    out['sites_aligned'] = have_sites

    if zip_valid:
        by_module = collections.defaultdict(lambda: dict(device_ms=0.0, count=0))
        by_module_grids = collections.defaultdict(
            lambda: collections.defaultdict(lambda: dict(device_ms=0.0, count=0)))
        by_site = collections.defaultdict(lambda: dict(device_ms=0.0, count=0))
        by_module_site = collections.defaultdict(
            lambda: collections.defaultdict(lambda: dict(device_ms=0.0, count=0)))
        for position, (name, event) in enumerate(zip(order, triton)):
            row = by_module[name]
            row['device_ms'] += event['dur'] / 1000.0
            row['count'] += 1
            grid = grids[position] if position < len(grids) else None
            grow = by_module_grids[name][grid]
            grow['device_ms'] += event['dur'] / 1000.0
            grow['count'] += 1
            if have_sites:
                site = sites[position]
                srow = by_site[site]
                srow['device_ms'] += event['dur'] / 1000.0
                srow['count'] += 1
                mrow = by_module_site[name][site]
                mrow['device_ms'] += event['dur'] / 1000.0
                mrow['count'] += 1
        out['triton_by_module'] = {
            k: dict(device_ms=v['device_ms'], count=v['count'],
                    us_per_launch=v['device_ms'] * 1000.0 / v['count'])
            for k, v in sorted(by_module.items(), key=lambda kv: -kv[1]['device_ms'])}
        out['triton_by_module_total_ms'] = sum(v['device_ms'] for v in by_module.values())
        out['triton_by_module_grids'] = {
            module: {grid: dict(device_ms=g['device_ms'], count=g['count'])
                     for grid, g in sorted(rows.items(), key=lambda kv: -kv[1]['device_ms'])}
            for module, rows in sorted(by_module_grids.items(),
                                       key=lambda kv: -sum(g['device_ms']
                                                           for g in kv[1].values()))}
        if have_sites:
            out['triton_by_site'] = {
                k: dict(device_ms=v['device_ms'], count=v['count'])
                for k, v in sorted(by_site.items(), key=lambda kv: -kv[1]['device_ms'])}
            out['triton_by_module_sites'] = {
                module: {site: dict(device_ms=g['device_ms'], count=g['count'])
                         for site, g in sorted(rows.items(), key=lambda kv: -kv[1]['device_ms'])}
                for module, rows in sorted(by_module_site.items(),
                                           key=lambda kv: -sum(g['device_ms']
                                                               for g in kv[1].values()))}
            # 每个调用点的**平均 grid**（同一调用点若 grid 唯一，就直接可反推形状）
            site_grids = collections.defaultdict(collections.Counter)
            for position, name in enumerate(order):
                if position < len(sites) and position < len(grids):
                    site_grids[(name, sites[position])][grids[position]] += 1
            out['site_grids'] = {f'{n} @ {s}': dict(c)
                                 for (n, s), c in sorted(site_grids.items())}

        # C5 逐名归组闭合：zip 出的每名字设备 ms 之和 == key_averages 的同一名字
        closure = {}
        worst = dict(name=None, abs_ms=0.0)
        for name, counter in sorted(counter_trace.items()):
            zipped = sum(row['device_ms'] for resolved, row in by_module.items()
                         if bare(resolved) == name)
            reference = (keyavg.get(name) or {}).get('device_ms')
            diff = None if reference is None else zipped - reference
            closure[name] = dict(zipped_ms=zipped, key_averages_ms=reference,
                                 diff_ms=diff)
            if diff is not None and abs(diff) > worst['abs_ms']:
                worst = dict(name=name, abs_ms=abs(diff))
        checks['C5_per_name_closure'] = dict(
            ok=(worst['abs_ms'] < args.tolerance_ms),
            tolerance_ms=args.tolerance_ms, worst=worst, per_name=closure)

        out['family_closure'] = dict(
            triton_zipped_ms=out['triton_by_module_total_ms'],
            triton_key_averages_ms=profiled['gpu_ms_by_family']['triton'],
            triton_diff_ms=out['triton_by_module_total_ms']
            - profiled['gpu_ms_by_family']['triton'],
            aten_key_averages_ms=profiled['gpu_ms_by_family']['aten'])
    else:
        checks['C5_per_name_closure'] = dict(ok=None, note='zip 无效，不做闭合')

    # C6 交叉计数：层 1 自己的 stats（**累计量取差**）
    stats = profiled.get('fastpath_stats')
    if stats is None:
        checks['C6_fastpath_launch_count'] = dict(ok=None, note='拿不到层 1 单例（加分项，不算失败）')
    else:
        delta = stats.get('delta') or {}
        checks['C6_fastpath_launch_count'] = dict(
            ok=(delta.get('launch') == len(order)),
            fastpath_launch=delta.get('launch'), recorded=len(order),
            fastpath_warmup=delta.get('warmup'),
            fastpath_hits=delta.get('hits'), fastpath_misses=delta.get('misses'),
            fastpath_verify_fail=delta.get('verify_fail'))

    out['checks'] = checks
    out['all_checks_ok'] = all(
        row.get('ok') is not False for row in checks.values())

    # ATen 侧按 `aten::` 算子归组：**ATen 事件带 `External id`**，能链回它的 `cpu_op`。
    op_of = {}
    for e in events:
        if e.get('cat') == 'cpu_op':
            ext = (e.get('args') or {}).get('External id')
            if ext is not None:
                op_of.setdefault(ext, e['name'])
    by_op = collections.defaultdict(lambda: dict(device_ms=0.0, count=0))
    unlinked = dict(device_ms=0.0, count=0)
    for e in aten:
        op = op_of.get((e.get('args') or {}).get('External id'))
        if op is None:
            unlinked['count'] += 1
            unlinked['device_ms'] += e['dur'] / 1000.0
            continue
        row = by_op[op]
        row['device_ms'] += e['dur'] / 1000.0
        row['count'] += 1
    out['aten_by_op'] = {k: dict(device_ms=v['device_ms'], count=v['count'])
                         for k, v in sorted(by_op.items(), key=lambda kv: -kv[1]['device_ms'])}
    out['aten_unlinked'] = unlinked

    # ---------------- 输出 ----------------
    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding='utf-8')

    lines = []
    add = lines.append
    add('# 设备侧调用点归因（`_matmul` 的 115 次到底从哪来）')
    add('')
    add('诊断，不是证据。无门槛、无正确性产物、无加速比结论。'
        '这里的毫秒是设备时间构成，不是加速比。')
    add('')
    add(f'- 被剖析帧号 {out["frame_index"]} ｜ 缓存模式 `{out["cache_mode"]}`')
    add(f'- `gpu_busy_ms` = {gpu_busy_ms:.4f} ms ｜ '
        f'triton {profiled["gpu_ms_by_family"]["triton"]:.4f} '
        f'({profiled["gpu_instances_by_family"]["triton"]} 次) / '
        f'aten {profiled["gpu_ms_by_family"]["aten"]:.4f} '
        f'({profiled["gpu_instances_by_family"]["aten"]} 次)')
    add(f'- trace：{len(kernels)} 个内核事件，pid={pids}，tid={tids}')
    add(f'- 记录到的启动 = {len(order)}，调用点 = {len(sites)}，'
        f'trace 非 `_ZTS` = {len(triton)} ⇒ zip_valid = {zip_valid}')
    add('')
    add('## 自检')
    add('')
    add('| # | 检查 | 结果 | 读数 |')
    add('|---|---|---|---|')
    row = checks['C1_count_match']
    add(f'| C1 | 计数对齐 | {"PASS" if row["ok"] else "FAIL"} | '
        f'记录 {row["recorded"]} vs trace {row["trace_triton"]} |')
    row = checks['C2_per_name_count_match']
    add(f'| C2 | 逐名计数对齐 | {"PASS" if row["ok"] else "FAIL"} | '
        f'仅记录有 {row["only_in_recorded"]}；仅 trace 有 {row["only_in_trace"]} |')
    row = checks['C3_single_in_order_queue']
    add(f'| C3 | 单流有序 | {"PASS" if row["ok"] else "FAIL"} | pid={row["pid"]} tid={row["tid"]} |')
    row = checks['C4_duration_closure']
    add(f'| C4 | 时长闭合 | {"PASS" if row["ok"] else "FAIL"} | '
        f'trace {row["trace_device_ms"]:.4f} vs busy {row["gpu_busy_ms"]:.4f} '
        f'（差 {row["delta_ms"] * 1000:.1f} µs，{row["delta_pct"]:.3f}%） |')
    row = checks['C5_per_name_closure']
    if row.get('ok') is None:
        add(f'| C5 | 逐名归组闭合 | — | {row.get("note")} |')
    else:
        add(f'| C5 | 逐名归组闭合 | {"PASS" if row["ok"] else "FAIL"} | '
            f'最差 `{row["worst"]["name"]}` {row["worst"]["abs_ms"] * 1000:.3f} µs'
            f'（容差 {row["tolerance_ms"] * 1000:.0f} µs） |')
    row = checks['C6_fastpath_launch_count']
    if row.get('ok') is None:
        add(f'| C6 | 层1 交叉计数 | — | {row.get("note")} |')
    else:
        add(f'| C6 | 层1 交叉计数 | {"PASS" if row["ok"] else "FAIL"} | '
            f'层1 launch={row["fastpath_launch"]} vs 记录 {row["recorded"]} |')
    add('')

    if zip_valid and have_sites:
        add('## 按模块（与 kernel_split 同口径，作交叉核对）')
        add('')
        add('| module.function | 启动 | 设备 ms | µs/次 |')
        add('|---|---:|---:|---:|')
        total = out['triton_by_module_total_ms'] or 1.0
        for k, v in out['triton_by_module'].items():
            add(f'| `{k}` | {v["count"]} | {v["device_ms"]:.4f} | {v["us_per_launch"]:.2f} |')
        add(f'| **合计** | **{sum(v["count"] for v in out["triton_by_module"].values())}** | '
            f'**{total:.4f}** | |')
        add('')
        add('## 全部调用点（按设备时间）')
        add('')
        add('| # | 调用点 | 启动 | 设备 ms | 占 Triton |')
        add('|---:|---|---:|---:|---:|')
        for index, (k, v) in enumerate(out['triton_by_site'].items(), 1):
            add(f'| {index} | `{k}` | {v["count"]} | {v["device_ms"]:.4f} | '
                f'{100.0 * v["device_ms"] / total:.1f}% |')
        add('')
        add('## 逐模块 → 调用点')
        add('')
        for module, rows in out['triton_by_module_sites'].items():
            add(f'**`{module}`**')
            add('')
            add('| 调用点 | 启动 | 设备 ms | µs/次 | grid（该调用点用到的） |')
            add('|---|---:|---:|---:|---|')
            for site, g in rows.items():
                gs = out.get('site_grids', {}).get(f'{module} @ {site}', {})
                gt = ', '.join(f'`{gg}`×{c}' for gg, c in sorted(gs.items(), key=lambda kv: -kv[1]))
                add(f'| `{site}` | {g["count"]} | {g["device_ms"]:.4f} | '
                    f'{g["device_ms"] * 1000.0 / g["count"]:.2f} | {gt} |')
            add('')
    else:
        add('## ZIP 或调用点无效 —— 不出调用点表')
        add('')
        add(f'- zip_valid={zip_valid} sites_aligned={have_sites} '
            f'len(order)={len(order)} len(sites)={len(sites)}')

    if args.md:
        args.md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
