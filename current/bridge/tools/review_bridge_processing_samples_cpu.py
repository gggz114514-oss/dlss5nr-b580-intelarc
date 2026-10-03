"""Read matching completed-frame web times; no API or GPU imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arms', nargs=4, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--switch', choices=('pool', 'handoff'), default='pool')
    args = p.parse_args()
    arms = []
    ids = set()
    for path, on in zip(args.arms, (False, True, True, False)):
        raw = json.loads(path.read_text(encoding='utf-8-sig'))
        expected_pool = on if args.switch == 'pool' else False
        expected_handoff = on if args.switch == 'handoff' else False
        if (raw.get('completed') is not True or raw['pool_enabled'] is not expected_pool or
                raw['handoff_enabled'] is not expected_handoff):
            raise ValueError('Not the completed four-arm ' + args.switch + ' comparison')
        values, excluded = [], 0
        for state in raw['snapshots']:
            processing = state.get('processing') or {}
            recorded = ((processing.get('stages') or {}).get('recording') or {}).get('last_nr_frame_id')
            if processing.get('frames') != recorded:
                excluded += 1
                continue
            if type(recorded) is not int or recorded < 1 or recorded in ids:
                raise ValueError('Missing or duplicate completed-frame identity')
            ids.add(recorded)
            value = processing.get('last_ms')
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Missing actual last-frame timing')
            if (state.get('health') or {}).get('failed') is not False:
                raise ValueError('Failed NR frame')
            values.append(value)
        if len(values) < 20:
            raise ValueError('Fewer than twenty matched frames in an arm')
        ordered = sorted(values)
        arms.append({'input': str(path), 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'pool_on': expected_pool, 'handoff_on': expected_handoff,
                     'switch_on': on, 'raw_ms': values, 'n': len(values),
                     'mismatched_snapshot_frame_ids_excluded': excluded,
                     'mean_ms': statistics.fmean(values), 'p50_ms': statistics.median(values),
                     'p95_ms': ordered[math.ceil(.95 * len(ordered)) - 1]})
    off = [arms[i]['mean_ms'] for i in (0, 3)]
    on = [arms[i]['mean_ms'] for i in (1, 2)]
    result = {'status': 'matching_completed_web_samples_reviewed', 'switch': args.switch, 'arms': arms,
              'paired': {'off_ms': statistics.fmean(off), 'on_ms': statistics.fmean(on),
                         'saved_ms': statistics.fmean(off) - statistics.fmean(on),
                         'both_on_below_both_off': max(on) < min(off), 'off_drift_ms': off[1] - off[0]},
              'scope': 'processing.last_ms only; includes NR prep/process/composite/tail, excludes earlier CPU recording; not Present or RTSS.',
              'API_or_GPU_executed': False}
    target = args.output.resolve()
    if target.drive.casefold() != 'd:' or target.is_junction() or target.is_symlink():
        raise ValueError('Task result belongs on a physical D path')
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'n': [a['n'] for a in arms], 'paired': result['paired']}))


if __name__ == '__main__':
    main()
