"""Review saved serial bridge evidence only; no API or GPU execution."""
import argparse
import math
from pathlib import Path

import benchmark_gpu_handoff_serial_live_v2 as serial

base = serial.base


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base.physical(args.result_root)
    root = args.result_root.resolve(strict=True)
    require(root.drive.casefold() == 'd:', 'Read task-owned D evidence')
    result = base.read(root / 'RESULT.json')
    require(result.get('completed') is True, 'Incomplete live run')
    pins = result['source_hashes_before']
    require(pins == result['source_hashes_after'], 'Source changed during run')
    for filename, digest in pins.items():
        base.physical(Path(filename))
        require(base.sha(Path(filename)) == digest, 'Current source drift: ' + filename)
    controls = result['preserved_user_controls']
    seen, arms, evidence = set(), [], {}
    specs = [(f'pool-arm{i:02d}', on, False) for i, on in enumerate((False, True, True, False), 1)]
    specs += [(f'handoff-arm{i:02d}', False, on) for i, on in enumerate((False, True, True, False), 1)]
    specs += [('combined-arm', True, True)]
    for name, pool, handoff in specs:
        path = root / (name + '.json')
        data = base.read(path)
        require(data.get('completed') is True and data['name'] == name, 'Incomplete arm: ' + name)
        require(data['pool_enabled'] is pool and data['handoff_enabled'] is handoff, 'Wrong arm switch')
        require(data['warmup']['seconds'] >= 8 and data['warmup']['new_completed_frames'] >= 35, 'Insufficient warmup')
        rows, states = data['recording_samples'], data['snapshots']
        require(len(rows) >= 30 and len(rows) == len(states), 'Missing completed samples')
        ids = [row['last_nr_frame_id'] for row in rows]
        require(all(type(v) is int and v > 0 for v in ids), 'Invalid completed frame')
        require(all(b > a for a, b in zip(ids, ids[1:])) and not seen.intersection(ids), 'Duplicate/regressed frame')
        seen.update(ids)
        for row, state in zip(rows, states):
            require(row == state['processing']['stages']['recording'], 'Recording snapshot mismatch')
            require(all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) and row[k] >= 0 for k in base.METRICS), 'Unavailable timing')
        for state in [data['before'], *states, data['after']]:
            base.validate(state, controls)
            require(serial.applied(state, pool, handoff), 'Switch/capability not actually applied')
        delta = {group: base.counters_delta(data['before'], data['after'], group) for group in ('bridge_resource_pool', 'gpu_handoff', 'bridge_cache')}
        require(delta == data['counter_delta'], 'Saved counter delta mismatch')
        if pool:
            require(delta['bridge_resource_pool'].get('hits', -1) > 0, 'No actual texture reuse')
        if handoff:
            require(delta['gpu_handoff'].get('prepared_bypasses', -1) > 0, 'No actual GPU handoff')
        for group, keys in (('gpu_handoff', ('process_failures', 'previous_consumer_wait_failures')),
                            ('bridge_resource_pool', ('creation_failures', 'quarantines')),
                            ('bridge_cache', ('shader_compile_failures',))):
            require(all(delta[group].get(key) == 0 for key in keys), 'Missing/nonzero failure counter')
        arms.append({'name': name, 'n': len(rows), 'actual_pool_hits': delta['bridge_resource_pool'].get('hits'),
                     'actual_prepared_bypasses': delta['gpu_handoff'].get('prepared_bypasses')})
        evidence[str(path)] = base.sha(path)
    path = root / 'continuity-90s.json'
    continuity = base.read(path)
    states = continuity['snapshots']
    require(continuity['seconds'] >= 90 and len(states) > 1, 'Missing 90-second continuity')
    for state in [continuity['start'], *states]:
        base.validate(state, controls)
        require(serial.applied(state, True, True), 'Continuity capability lost')
    ids = [base.frame(continuity['start']), *(base.frame(s) for s in states)]
    require(all(b >= a for a, b in zip(ids, ids[1:])), 'Continuity frame regression')
    require(ids[-1] - ids[0] == continuity['new_completed_frames'] and continuity['new_completed_frames'] >= 100, 'Continuity completion count mismatch')
    for group, keys in (('gpu_handoff', ('process_failures', 'previous_consumer_wait_failures', 'identity_mismatch_waits')),
                        ('bridge_resource_pool', ('creation_failures', 'quarantines'))):
        delta = base.counters_delta(continuity['start'], states[-1], group)
        require(all(delta.get(key) == 0 for key in keys), 'Continuity failure/capability loss')
    require(result.get('final_experiments_OFF') is True, 'No final OFF restoration')
    base.validate(result['final_state'], controls)
    require(serial.applied(result['final_state'], False, False), 'Restoration not actually applied')
    evidence[str(path)] = base.sha(path)
    evidence[str(root / 'RESULT.json')] = base.sha(root / 'RESULT.json')
    base.physical(args.output)
    require(args.output.resolve().drive.casefold() == 'd:', 'Write review on D')
    review = {'status': 'serial_handoff_live_protocol_reviewed', 'source_pin_count': len(pins),
              'current_source_hashes_checked': True, 'arms': arms, 'unique_timed_frames': len(seen),
              'continuity_seconds': continuity['seconds'], 'continuity_new_completed_frames': continuity['new_completed_frames'],
              'continuity_snapshots': len(states), 'serial_actual_capability_held': True,
              'final_experiments_OFF': True, 'evidence_sha256': evidence,
              'reviewer_sha256': base.sha(Path(__file__)), 'main_API_or_GPU_executed': False,
              'visual_or_Present_acceptance': False, 'combined_paired_performance_pending': True}
    base.dump(args.output.parent, args.output.name, review)
    print(__import__('json').dumps(review))


if __name__ == '__main__':
    main()
