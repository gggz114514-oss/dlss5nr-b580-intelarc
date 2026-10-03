"""Luna-only combined OFF/ON/ON/OFF on the already qualified live chain."""
from pathlib import Path
import benchmark_gpu_handoff_serial_live_v2 as serial

base = serial.base
_serial_pins = serial.pins
PRIOR = Path(r'D:\Codex-NR-Experiments\cyberpunk-opt\gpu-handoff-live-pair-v1-20261002\serial-driver-run05')


def pins(receipt):
    result = _serial_pins(receipt)
    prior = base.read(PRIOR / 'RESULT.json')
    review = base.read(PRIOR / 'MAIN_PROTOCOL_REVIEW.json')
    if prior.get('completed') is not True or review.get('status') != 'serial_handoff_live_protocol_reviewed':
        raise ValueError('Prior serial qualification is incomplete')
    if any(result.get(path) != digest for path, digest in prior['source_hashes_before'].items()):
        raise ValueError('Prior serial source identity differs')
    for path in (Path(__file__).resolve(), PRIOR / 'RESULT.json', PRIOR / 'MAIN_PROTOCOL_REVIEW.json'):
        base.physical(path)
        result[str(path)] = base.sha(path)
    return result


def run(args):
    root = args.output.resolve()
    base.physical(root)
    if root.drive.casefold() != 'd:':
        raise ValueError('Write task-owned live output on D')
    root.mkdir(parents=True, exist_ok=False)
    before_pins = base.source_pins(args.install_receipt)
    api = base.API()
    api.connect()
    initial = api.call('/api/state')
    base.dump(root, 'initial.json', initial)
    controls = {key: initial['settings'][key] for key in base.CONTROL_KEYS}
    try:
        # Never reconfigure the mathematical chain during this extra comparison.
        base.validate(initial, controls)
        for ordinal, on in enumerate((False, True, True, False), 1):
            base.arm(api, root, f'combined-arm{ordinal:02d}', controls, on, on)
        base.request(api, False, False)
        restored = base.wait_state(api, lambda s: base.applied(s, False, False), controls)
        restored_id = base.frame(restored)
        restored = base.wait_state(api, lambda s: base.applied(s, False, False) and base.frame(s) - restored_id >= 8, controls)
        after_pins = base.source_pins(args.install_receipt)
        if before_pins != after_pins:
            raise ValueError('Source changed during combined comparison')
        result = {'status': 'combined_live_pairs_completed_pending_main_review', 'completed': True,
                  'source_hashes_before': before_pins, 'source_hashes_after': after_pins,
                  'prior_protocol_review': str(PRIOR / 'MAIN_PROTOCOL_REVIEW.json'),
                  'preserved_user_controls': controls, 'final_state': restored,
                  'final_experiments_OFF': True, 'finished_utc': base.utc(),
                  'visual_or_Present_acceptance': False, 'executed_by': 'assigned GPU tester'}
        base.dump(root, 'RESULT.json', result)
        print(__import__('json').dumps({'status': result['status'], 'output': str(root)}), flush=True)
    except BaseException as exc:
        try:
            base.request(api, False, False)
            recovered = api.call('/api/state')
        except Exception as cleanup:
            recovered = {'recovery_error': type(cleanup).__name__}
        base.dump(root, 'FAILED.json', {'status': 'failed', 'error_type': type(exc).__name__,
                  'error': str(exc), 'recovered': recovered, 'source_hashes_before': before_pins,
                  'last_state': getattr(exc, 'last_state', None), 'finished_utc': base.utc()})
        raise


if __name__ == '__main__':
    serial.pins = pins
    base.run = run
    serial.main()
