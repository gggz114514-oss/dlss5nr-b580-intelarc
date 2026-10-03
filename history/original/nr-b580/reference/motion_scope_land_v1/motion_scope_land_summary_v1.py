"""落地轮 ABBA 汇总 —— `motion_scope` 短路在产品里还剩多少？

与 `motion_scope_v1/motion_scope_summary_v1.py` 的区别
------------------------------------------------------
那份汇总**外挂轮**（读 `cuts.json` 的 `motion_scope.stats`，因为刀装在外挂层）。
本轮的刀已经**写进产品源码** ⇒ 没有 `cuts.json.motion_scope`，臂目录里多出来的是
`gate_probe.json`（开关自证）。所以这里自己算，做四件事：

  1. 逐臂帧墙 + 逐字节门 + 开关自证；
  2. **配对口径**（先在 session 内配对，再跨 session 平均）—— 本机有 session 级漂移；
  3. 池化统计量（n、mean、sd、SE、t、95% CI）—— 点估计必须跨批次池化（技能 139）；
  4. 与外挂轮记录值 `+1.683 ms` 及同变体 spread 的比较（技能 127）。

用法：
    python motion_scope_land_summary_v1.py --run D:/motion-scope-land-v1/r1 [--run ...] \
        [--out motion_scope_land_v1/ab_summary.txt]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
       14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
       20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060}

# 外挂轮（送审那一版）的记录值，只用于对照，不参与判定
OUT_OF_TREE_DELTA = 1.683


def _load(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else None


def collect(run_dir: Path):
    arms = []
    for arm in sorted(run_dir.iterdir()):
        if not arm.is_dir():
            continue
        m = re.match(r'^(\d+)-(off|on)$', arm.name)
        if not m:
            continue
        val = _load(arm / 'run' / 'validation.json')
        if val is None:
            continue
        probe = _load(arm / 'gate_probe.json') or {}
        timing = val.get('timing') or {}
        arms.append(dict(
            round=run_dir.name, index=int(m.group(1)), tag=m.group(2), arm=arm.name,
            median_ms=timing.get('median_ms'),
            identical_frames=val.get('identical_frames'),
            compared=len(val.get('frames') or []),
            passed=val.get('passed'),
            error=val.get('error'),
            probe_ok=probe.get('ok'), probe_flag=probe.get('flag'),
            probe_env=probe.get('env_value'),
            near_vs_pre=probe.get('near_degenerate_max_abs_vs_pre'),
            sub_vs_pre=probe.get('subpixel_max_abs_vs_pre')))
    return arms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, type=Path)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    lines = []

    def say(text=''):
        lines.append(text)
        print(text, flush=True)

    arms = [a for rd in args.run for a in collect(rd)]
    if not arms:
        print('没有可汇总的轮次', file=sys.stderr)
        return 1

    say('=== 落地轮 ABBA：NR_MOTION_SCOPE off/on（同 session、位置对称）===')
    say('臂序：0-off 1-on 2-on 3-off | 4-off 5-on 6-on 7-off')
    say()

    say('--- 逐臂（median_ms = 稳态 frames[4:] 中位）---')
    say('%-8s %-5s %12s %10s %10s %8s %8s %12s'
        % ('round', 'tag', 'median ms', 'ident', 'compared', 'probe', 'flag', 'near_vs_pre'))
    for a in arms:
        say('%-8s %-5s %12s %10s %10s %8s %8s %12s' % (
            a['round'], a['tag'],
            ('%.4f' % a['median_ms']) if a['median_ms'] is not None else '-',
            a['identical_frames'], a['compared'],
            'OK' if a['probe_ok'] else 'FAIL',
            a['probe_flag'],
            ('%.3e' % a['near_vs_pre']) if isinstance(a['near_vs_pre'], (int, float)) else '-'))
    say()

    # ---- 每轮配对 Δ = mean(off) − mean(on) ----
    say('--- 逐轮配对 Δ（正 = on 比 off 省）---')
    say('%-8s %12s %12s %10s   %s'
        % ('round', 'off 均', 'on 均', 'Δ', 'off / on 各臂'))
    deltas = []
    for rd in args.run:
        rows = collect(rd)
        offs = [a['median_ms'] for a in rows if a['tag'] == 'off' and a['median_ms'] is not None]
        ons = [a['median_ms'] for a in rows if a['tag'] == 'on' and a['median_ms'] is not None]
        if not offs or not ons:
            continue
        d = statistics.mean(offs) - statistics.mean(ons)
        deltas.append(d)
        say('%-8s %12.4f %12.4f %+10.4f   %s / %s'
            % (rd.name, statistics.mean(offs), statistics.mean(ons), d,
               ' '.join('%.3f' % x for x in offs), ' '.join('%.3f' % x for x in ons)))
    say()

    # ---- 同变体 spread ----
    say('--- 同变体 spread（每臂中位帧墙的极差）---')
    spread = {}
    for tag in ('off', 'on'):
        xs = [a['median_ms'] for a in arms if a['tag'] == tag and a['median_ms'] is not None]
        if xs:
            spread[tag] = max(xs) - min(xs)
            say('  %-4s n=%d  mean=%9.4f  spread=%.4f (%.3f%%)'
                % (tag, len(xs), statistics.mean(xs), spread[tag],
                   100 * spread[tag] / statistics.mean(xs)))
    say()

    # ---- 判据 ----
    say('--- 判据 ---')
    if deltas:
        n = len(deltas)
        mean = statistics.mean(deltas)
        sd = statistics.stdev(deltas) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else float('nan')
        t = mean / se if se == se and se > 0 else float('nan')
        crit = T95.get(n - 1, 2.0)
        same_sign = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        worst = max(spread.values()) if spread else 0.0
        say('  逐轮 Δ：%s' % ' '.join('%+.4f' % d for d in deltas))
        say('  Δ = %+.4f ms  n=%d  sd=%.4f  SE=%.4f  t=%.2f  CI95=[%+.4f,%+.4f]  %s'
            % (mean, n, sd, se, t, mean - crit * se, mean + crit * se,
               '可判' if abs(t) > crit else '不可判'))
        say('  符号一致：%s' % ('是' if same_sign else '否'))
        say('  |Δ| vs spread(off)=%.4f：%s   vs spread(on)=%.4f：%s'
            % (spread.get('off', float('nan')),
               '>' if abs(mean) > spread.get('off', 0) else '≤',
               spread.get('on', float('nan')),
               '>' if abs(mean) > spread.get('on', 0) else '≤'))
        say('  ★ 保守判据（对较大的 spread）：|Δ| %s spread ⇒ %s'
            % ('>' if abs(mean) > worst else '≤',
               '可判' if (same_sign and abs(mean) > worst) else '⚠️ 不可判 —— 需加轮次'))
        say()
        say('  对照：外挂轮（送审那一版）记录值 Δ = +%.3f ms' % OUT_OF_TREE_DELTA)
        say('       落地轮池化 Δ = %+.4f ms ⇒ 落地形态是外挂形态的 %.0f%%'
            % (mean, 100.0 * mean / OUT_OF_TREE_DELTA if OUT_OF_TREE_DELTA else float('nan')))
        hi = mean + crit * se
        say('       池化 CI 上界 = %+.4f ⇒ %s +%.3f'
            % (hi, '排除' if hi < OUT_OF_TREE_DELTA else '不排除', OUT_OF_TREE_DELTA))
    say()

    # ---- 逐字节门 ----
    say('--- 逐字节门（vs 冻结参考）---')
    for tag in ('off', 'on'):
        xs = [a for a in arms if a['tag'] == tag]
        if not xs:
            continue
        say('  %-4s ident=%s  compared=%s' % (
            tag, [a['identical_frames'] for a in xs], [a['compared'] for a in xs]))
    off_bad = [a for a in arms if a['tag'] == 'off'
               and not (a['identical_frames'] == a['compared'] and a['passed'] is True)]
    if off_bad:
        say('  ⚠️ off 臂没有全部逐字节 ⇒ **落地文件里的 off 分支 ≠ 落地前的产品**，'
            '速度基线不成立。')
        say('     %s' % [(a['arm'], a['identical_frames'], a['compared']) for a in off_bad])
    else:
        say('  ✅ off 臂全部逐字节 ⇒ 落地文件里的 off 分支 = 落地前的产品（速度基线成立）。')
    on_ok = [a for a in arms if a['tag'] == 'on'
             and a['identical_frames'] == a['compared']]
    if on_ok:
        say('  ⚠️ on 臂竟然有逐字节的：%s ⇒ 短路没生效？' % [a['arm'] for a in on_ok])
    else:
        say('  ✅ on 臂全部非逐字节 ⇒ 短路确实在改画面（与"不是逐位恒等"一致）。')
    say()

    # ---- 开关自证 ----
    say('--- 开关自证（gate_probe.json）---')
    bad = [a for a in arms if not a['probe_ok']]
    for a in arms:
        say('  %-10s tag=%-4s env=%-4s flag=%-5s near=%-12s sub=%-12s %s'
            % (a['arm'], a['tag'], a['probe_env'], a['probe_flag'],
               ('%.3e' % a['near_vs_pre']) if isinstance(a['near_vs_pre'], (int, float)) else '-',
               ('%.3e' % a['sub_vs_pre']) if isinstance(a['sub_vs_pre'], (int, float)) else '-',
               'OK' if a['probe_ok'] else 'FAIL'))
    if bad:
        say('  ⚠️ %d 个臂的开关自证不过 ⇒ 这些臂的帧墙差值不可用。' % len(bad))
    else:
        say('  ✅ 全部臂自证通过。')

    if args.out:
        args.out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print('\n写入 %s' % args.out)
    return 2 if (bad or off_bad) else 0


if __name__ == '__main__':
    sys.exit(main())
