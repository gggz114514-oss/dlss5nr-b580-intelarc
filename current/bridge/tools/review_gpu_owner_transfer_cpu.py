"""Review Luna's real two-thread result using only saved data and file hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = read(args.runtime_receipt)
    require(run.get('passed') is True and run['exit_code'] == 0 and
            run['watchdog_terminated'] is False and run['process_reaped'] is True and
            run['process_tree_contained'] is True, 'The GPU process did not exit normally')
    child_path = args.runtime_receipt.parent / 'CHILD_RESULT.json'
    child = read(child_path)
    require(child == run['child_result'], 'Saved child result and receipt disagree')
    require(child.get('passed') is True and child.get('GPU_attempted') is True and
            child.get('whole_model_or_game_test') is False and
            child.get('performance_measured') is False, 'Wrong test scope or failed child')
    cpu_path = Path(run['CPU_receipt'])
    require(sha(cpu_path) == run['CPU_receipt_sha256'], 'CPU probe receipt changed')
    cpu = read(cpu_path)
    build_path = Path(cpu['candidate_build_receipt'])
    require(sha(build_path) == cpu['candidate_build_receipt_sha256'], 'Native build receipt changed')
    build = read(build_path)
    require(cpu['status'] == 'CPU_passed_GPU_UNTESTED' and
            build['status'] == 'native_CPU_compiled_GPU_UNTESTED', 'Missing successful CPU build')
    pins = {}
    for group in ('test_source_pins', 'product_source_pins', 'protocol_source_pins'):
        pins.update(cpu[group])
    pins.update(build['source_pins'])
    pins.update(build['input_pins'])
    pins.update({cpu[key]: cpu[digest] for key, digest in
                 (('candidate_DLL', 'candidate_sha256'), ('probe_DLL', 'probe_sha256'),
                  ('old_DLL', 'old_sha256'))})
    pins[cpu['SYCL']['path']] = cpu['SYCL']['sha256']
    for path, expected in pins.items():
        require(sha(path) == expected, 'Reviewed input changed: ' + path)
    require(run['candidate_sha256'] == cpu['candidate_sha256'] == build['candidate_sha256'] and
            run['probe_sha256'] == cpu['probe_sha256'], 'Runtime binary identity mismatch')
    require(child['actual_loaded_SYCL'] == cpu['SYCL'], 'Wrong loaded SYCL runtime')
    proof = child['owner_transfer_probe']
    require(proof['actual_loaded_SYCL'] == cpu['SYCL'], 'Owner phase used another runtime')
    require(proof['status'] == 'passed' and proof['NR_model_loaded'] is False and
            proof['validation_model_stubbed'] is True and
            proof['actual_product_wrapper_and_adapter_process_used'] is True,
            'The actual product wrapper/adapter was not exercised')
    for key in ('real_native_completion_observed', 'cleanup_ok', 'host_threads_joined'):
        require(proof[key] is True, 'Incomplete runtime proof: ' + key)
    require(proof['quarantined'] is False, 'Resources were quarantined')
    threads = [proof['host_A'], proof['host_B']]
    native_ids = {t['native_thread'] for t in threads}
    python_ids = {t['python_thread'] for t in threads}
    require(len(native_ids) == len(python_ids) == 2 and all(native_ids | python_ids),
            'Two actual persistent threads were not observed')
    expected_counts = {'shared_texture_frames': 6, 'ON_frames': 3, 'OFF_frames': 3,
                       'graph_captures': 3, 'graph_replays': 18,
                       'output_byte_comparisons': 12, 'input_byte_comparisons': 24,
                       'nonzero_motion_checks': 18, 'retire_lock_checks': 6}
    for key, expected in expected_counts.items():
        require(proof[key] == expected, 'Incomplete test count: ' + key)
    require(len(proof['migrations']) == 4, 'Incomplete actual thread migrations')
    for migration in proof['migrations']:
        require(migration['previous_native_thread'] in native_ids and
                migration['adopted_native_thread'] in native_ids and
                migration['previous_native_thread'] != migration['adopted_native_thread'] and
                migration['previous_python_thread'] in python_ids and
                migration['adopted_python_thread'] in python_ids and
                migration['previous_python_thread'] != migration['adopted_python_thread'],
                'Migration did not change actual thread ownership')
    frames = proof['frames']
    require([f['frame'] for f in frames] == list(range(1, 7)), 'Wrong frame order')
    for frame in frames:
        require(frame['owner_native_thread'] in native_ids and
                frame['actual_source_completed'] >= frame['frame'] and
                frame['actual_consumer_completed'] >= frame['frame'] * 3 and
                frame['output_comparison_bytes'] == 64 * 36 * 4 * 4 * 2,
                'Input/output/consumer completion proof missing')
    require(proof['output_bytes_compared'] == sum(f['output_comparison_bytes'] for f in frames),
            'Output comparison byte sum mismatch')
    wrong = proof['wrong_identity']
    require(len(wrong) == 6 and {w['enabled'] for w in wrong} == {0, 1},
            'Missing OFF/ON wrong-identity cases')
    expected_labels = {'previous_native_high_bits', 'different_SYCL_queue', 'different_D3D12_queue'}
    for enabled in (0, 1):
        require({w['case'] for w in wrong if w['enabled'] == enabled} == expected_labels,
                'Missing wrong-identity rejection')
    for case in wrong:
        require(case['rc'] != 0 and case['adopted'] == 0 and case['error'] and
                case['changed_stats'] == {} and case['before_stats'] == case['after_stats'],
                'Rejected call changed state')
    busy = proof['busy_observations']
    require(len(busy) == 15, 'Missing busy ownership rejection')
    for stage, ids in (('producer_and_SYCL_work_in_flight', {2, 3, 4}),
                       ('output_borrowed_consumer_unregistered', set(range(1, 7))),
                       ('consumer_registered_but_fence_incomplete', set(range(1, 7)))):
        require({b['frame'] for b in busy if b['stage'] == stage} == ids, 'Missing busy stage: ' + stage)
    for case in busy:
        require(case['rc'] == 0 and case['adopted'] == 0 and not case['error'],
                'Busy ownership was adopted or poisoned')
        if case['stage'] == 'consumer_registered_but_fence_incomplete':
            require(case['actual_consumer_completed'] < case['registered_consumer_value'] and
                    case['actual_consumer_gate'] < case['frame'], 'Consumer was not actually blocked')
        if case['stage'] == 'producer_and_SYCL_work_in_flight':
            require(case['actual_source_completed'] < case['frame'], 'Producer was not actually blocked')
    require(len(proof['normal_owner_rejections']) == 3 and
            all(r['rc'] != 0 and 'owning host thread' in r['error']
                for r in proof['normal_owner_rejections']), 'Ordinary thread checks were bypassed')
    info, stats = proof['final_info'], proof['final_stats']
    require(info['enabled'] == info['active'] == info['poisoned'] == 0 and
            info['healthy'] == info['reuse_safe_idle'] == 1 and
            stats['active'] == stats['poisoned'] == stats['forward_live'] == 0,
            'Final bridge not healthy and retired')
    for key in ('frames_started', 'frames_retired', 'forward_imports', 'forward_releases',
                'pack_submits', 'unpack_submits', 'xpu_waits', 'xpu_signals', 'consumer_registrations'):
        require(stats[key] == 3, 'Unbalanced native counter: ' + key)
    require(info['borrowed_sycl_queue'] == proof['same_Torch_stream_queue'] == stats['borrowed_queue'],
            'Queue identity changed')
    out = args.output.resolve()
    require(out.drive.casefold() == 'd:', 'Keep review data on D')
    for part in (out, *out.parents):
        require(not part.is_junction() and not part.is_symlink(), 'Redirected review path')
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'runtime_owner_transfer_reviewed',
              'reviewed_utc': datetime.now(timezone.utc).isoformat(),
              'runtime_receipt': str(args.runtime_receipt), 'runtime_receipt_sha256': sha(args.runtime_receipt),
              'child_result': str(child_path), 'child_result_sha256': sha(child_path),
              'CPU_build_receipt': str(build_path), 'CPU_build_receipt_sha256': sha(build_path),
              'candidate_DLL_sha256': build['candidate_sha256'],
              'source_hashes_unchanged': True, 'verified_hashes': pins,
              'byte_checks_passed': True, 'two_actual_threads_observed': True, 'normal_exit': True,
              'thread_ids': threads, 'counts': expected_counts, 'busy_rejections': len(busy),
              'migrations': len(proof['migrations']), 'native_final_counters': stats,
              'GPU_or_API_executed_by_main': False, 'game_performance_accepted': False,
              'scope': 'Shared texture, real motion, synthetic graph; full NR/game speed remains untested'}
    with out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'review': str(out), 'counts': expected_counts}))


if __name__ == '__main__':
    main()
