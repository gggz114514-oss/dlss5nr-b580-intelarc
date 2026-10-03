"""Review saved combined serial bridge activation; no API/GPU calls."""
import argparse
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
    require(root.drive.casefold() == 'd:', 'Use task-owned D results')
    result = base.read(root / 'RESULT.json')
    require(result.get('completed') is True and result.get('final_experiments_OFF') is True, 'Incomplete combined run')
    pins = result['source_hashes_before']
    require(pins == result['source_hashes_after'], 'Sources changed during run')
    for filename, digest in pins.items():
        base.physical(Path(filename))
        require(base.sha(Path(filename)) == digest, 'Current source drift: ' + filename)
    prior_path = Path(result['prior_protocol_review'])
    require(prior_path.as_posix() in {Path(p).as_posix() for p in pins}, 'Prior review not pinned')
    prior = base.read(prior_path)
    require(prior['status'] == 'serial_handoff_live_protocol_reviewed' and prior['continuity_seconds'] >= 90, 'Prior continuity not accepted')
    controls, seen, arms, evidence = result['preserved_user_controls'], set(), [], {}
    for ordinal, on in enumerate((False, True, True, False), 1):
        path = root / f'combined-arm{ordinal:02d}.json'
        data = base.read(path)
        require(data.get('completed') is True and data['pool_enabled'] is on and data['handoff_enabled'] is on, 'Wrong combined arm')
        require(data['warmup']['seconds'] >= 8 and data['warmup']['new_completed_frames'] >= 35, 'Insufficient warmup')
        rows, states = data['recording_samples'], data['snapshots']
        require(len(rows) >= 30 and len(rows) == len(states), 'Missing completed samples')
        ids = [row['last_nr_frame_id'] for row in rows]
        require(all(type(v) is int and v > 0 for v in ids) and all(b > a for a, b in zip(ids, ids[1:])), 'Invalid/duplicate completed frame')
        require(not seen.intersection(ids), 'Frame reused between arms')
        seen.update(ids)
        require(all(row == state['processing']['stages']['recording'] for row, state in zip(rows, states)), 'Saved recording differs from snapshot')
        for state in [data['before'], *states, data['after']]:
            base.validate(state, controls)
            require(serial.applied(state, on, on), 'Actual combined switches/capability lost')
        delta = {g: base.counters_delta(data['before'], data['after'], g) for g in ('bridge_resource_pool', 'gpu_handoff', 'bridge_cache')}
        require(delta == data['counter_delta'], 'Saved delta mismatch')
        if on:
            require(delta['bridge_resource_pool'].get('hits', -1) > 0 and delta['gpu_handoff'].get('prepared_bypasses', -1) > 0, 'No actual reuse/handoff')
        for group, keys in (('gpu_handoff', ('process_failures', 'previous_consumer_wait_failures', 'identity_mismatch_waits')),
                            ('bridge_resource_pool', ('creation_failures', 'quarantines')),
                            ('bridge_cache', ('shader_compile_failures',))):
            require(all(delta[group].get(key) == 0 for key in keys), 'Missing/nonzero failure counter')
        arms.append({'name': data['name'], 'n': len(rows), 'actual_pool_hits': delta['bridge_resource_pool'].get('hits'),
                     'actual_prepared_bypasses': delta['gpu_handoff'].get('prepared_bypasses')})
        evidence[str(path)] = base.sha(path)
    base.validate(result['final_state'], controls)
    require(serial.applied(result['final_state'], False, False), 'Final OFF did not apply')
    evidence[str(root / 'RESULT.json')] = base.sha(root / 'RESULT.json')
    base.physical(args.output)
    require(args.output.resolve().drive.casefold() == 'd:', 'Write main review on D')
    review = {'status': 'combined_serial_live_protocol_reviewed', 'source_pin_count': len(pins),
              'unique_timed_frames': len(seen), 'arms': arms, 'prior_protocol_review': str(prior_path),
              'current_sources_checked': True, 'final_experiments_OFF': True, 'evidence_sha256': evidence,
              'reviewer_sha256': base.sha(Path(__file__)), 'main_API_or_GPU_executed': False,
              'visual_or_Present_acceptance': False}
    base.dump(args.output.parent, args.output.name, review)
    print(__import__('json').dumps(review))


if __name__ == '__main__':
    main()
