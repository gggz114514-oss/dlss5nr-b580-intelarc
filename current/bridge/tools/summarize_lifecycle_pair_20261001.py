"""Summarize completed B-C-C-B rows; no game, API, or GPU access."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(path):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if 'candidates' in data:
        if data.get('completed') is not True or len(data['candidates']) != 1:
            raise ValueError('expected one completed pair block')
        data = data['candidates'][0]
    arms = data['arms']
    if [arm['role'] for arm in arms] != ['B', 'C', 'C', 'B']:
        raise ValueError('expected the completed B-C-C-B sequence')
    if [arm['policy_enabled'] for arm in arms] != [False, True, True, False]:
        raise ValueError('expected actual OFF-ON-ON-OFF policies')
    output = []
    for arm in arms:
        measure = arm['measurement']
        rows = measure['samples']
        ids = [row['frame_id'] for row in rows]
        if len(rows) != 30 or len(set(ids)) != 30 or measure['unique_completed_samples'] != 30:
            raise ValueError('each arm must contain 30 unique completed frames')
        totals = []
        for row in rows:
            snapshot = row['lifecycle_snapshot']
            if (row['network_replay_count'] != 1 or snapshot['frame_id'] != row['frame_id'] or
                    snapshot['lifecycle_trial'] is not arm['policy_enabled']):
                raise ValueError('actual replay/policy/frame receipt mismatch')
            total = sum(row[name] for name in ('process_wall_ms', 'adapter_handoff_cpu_ms',
                                             'observer_finish_cpu_ms'))
            if not math.isfinite(total) or total < 0:
                raise ValueError('invalid sequential adapter duration')
            totals.append(total)
        ordered = sorted(totals)
        output.append({'arm': arm['arm'], 'role': arm['role'], 'policy': arm['policy_enabled'],
                       'count': len(rows), 'mean_ms': statistics.mean(totals),
                       'p50_ms': statistics.median(totals),
                       'p95_ms': ordered[math.ceil(0.95*len(ordered))-1]})
    baseline = statistics.mean(output[i]['mean_ms'] for i in (0, 3))
    candidate = statistics.mean(output[i]['mean_ms'] for i in (1, 2))
    return {'completed': True, 'source': str(path), 'candidate': data['candidate'],
            'selected': data['selected'], 'arms': output,
            'adapter_total': {'baseline_ms': baseline, 'candidate_ms': candidate,
                              'c_minus_b_ms': candidate-baseline,
                              'improvement_percent': 100*(baseline-candidate)/baseline,
                              'both_candidates_below_both_baselines':
                                  max(output[i]['mean_ms'] for i in (1, 2)) <
                                  min(output[i]['mean_ms'] for i in (0, 3))},
            'existing_stage_deltas': data['paired_cost_delta'],
            'definition': 'process_wall + adapter_handoff_cpu + observer_finish_cpu; GPU is nested inside process',
            'limits': 'Same selected combination, 30 frames/arm. No RTSS base-FPS claim. Missing child telemetry remains unavailable.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('preserving completed summary')
    result = summarize(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'arms': result['arms'], 'adapter_total': result['adapter_total']}))
