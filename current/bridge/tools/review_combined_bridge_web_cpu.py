"""Reduce actual same-completed-frame web times for combined bridge arms."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import benchmark_gpu_handoff_serial_live_v2 as serial

base = serial.base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs=4, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    arms, seen = [], set()
    for path, on in zip(args.arms, (False, True, True, False)):
        base.physical(path)
        if path.resolve().drive.casefold() != 'd:':
            raise ValueError('Read physical task-owned D evidence')
        raw = base.read(path)
        if (raw.get('completed') is not True or raw['pool_enabled'] is not on or
                raw['handoff_enabled'] is not on):
            raise ValueError('Expected combined OFF/ON/ON/OFF')
        values, excluded = [], 0
        for state in raw['snapshots']:
            if (state.get('health') or {}).get('failed') is not False or not serial.applied(state, on, on):
                raise ValueError('Unhealthy or unqualified combined frame')
            processing = state.get('processing') or {}
            fid = base.frame(state)
            if processing.get('frames') != fid:
                excluded += 1
                continue
            if type(fid) is not int or fid < 1 or fid in seen:
                raise ValueError('Missing/duplicate completed frame')
            seen.add(fid)
            value = processing.get('last_ms')
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Missing actual completed-frame timing')
            values.append(value)
        if len(values) < 20:
            raise ValueError('Insufficient matching completed frames')
        ordered = sorted(values)
        arms.append({'input': str(path), 'input_sha256': base.sha(path), 'switch_on': on,
                     'n': len(values), 'mismatched_snapshot_frame_ids_excluded': excluded,
                     'raw_ms': values, 'mean_ms': statistics.fmean(values),
                     'p50_ms': statistics.median(values),
                     'p95_ms': ordered[math.ceil(.95 * len(values)) - 1]})
    off, on = [arms[i]['mean_ms'] for i in (0, 3)], [arms[i]['mean_ms'] for i in (1, 2)]
    result = {'status': 'combined_matching_completed_web_samples_reviewed', 'arms': arms,
              'paired': {'off_ms': statistics.fmean(off), 'on_ms': statistics.fmean(on),
                         'saved_ms': statistics.fmean(off) - statistics.fmean(on),
                         'both_on_below_both_off': max(on) < min(off), 'off_drift_ms': off[1] - off[0]},
              'scope': 'Same completed processing.last_ms; prep/process/composite/tail; earlier CPU recording excluded; not Present/RTSS.',
              'reviewer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'API_or_GPU_executed': False, 'unavailable_metrics_substituted': False}
    base.physical(args.output)
    if args.output.resolve().drive.casefold() != 'd:':
        raise ValueError('Write physical D task result')
    base.dump(args.output.parent, args.output.name, result)
    print(json.dumps({'status': result['status'], 'n': [a['n'] for a in arms], 'paired': result['paired']}))


if __name__ == '__main__':
    main()
