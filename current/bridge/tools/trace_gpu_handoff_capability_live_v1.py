"""Luna's bounded live capability trace; never treats requested as enabled."""
import argparse
import importlib.util
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('cpu-check', 'run'))
    parser.add_argument('--install-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    driver = Path(__file__).with_name('benchmark_gpu_handoff_live_v1.py')
    spec = importlib.util.spec_from_file_location('frozen_bridge_live_driver', driver)
    live = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(live)
    pins = live.source_pins(args.install_receipt)
    pins[str(Path(__file__).resolve())] = live.sha(Path(__file__).resolve())
    if args.mode == 'cpu-check':
        print(json.dumps({'source_pins': len(pins), 'API_or_GPU_executed': False}))
        return
    out = args.output.resolve()
    live.physical(out)
    if out.drive.casefold() != 'd:':
        raise ValueError('Trace data belongs on D')
    out.mkdir(parents=True, exist_ok=False)
    api = live.API()
    api.connect()
    initial = api.call('/api/state')
    controls = {k: initial['settings'][k] for k in live.CONTROL_KEYS}
    live.validate(initial, controls)
    rows, failure, final = [], None, None
    try:
        live.request(api, False, True)
        start = time.monotonic()
        while time.monotonic() - start < 12:
            state = api.call('/api/state')
            live.validate(state, controls)
            stages = state['processing']['stages']
            native = state['health'].get('temporal_diagnostics', {}).get('periodic_flash_v2', {})
            rows.append({'elapsed_seconds': time.monotonic() - start,
                         'completed_frame_id': live.frame(state),
                         'processing_frames': state['processing'].get('frames'),
                         'gpu_handoff': state.get('gpu_handoff'),
                         'health_failed': state['health']['failed'],
                         'native_counters': native.get('counters'),
                         'runtime_stage_last': {k: v for k, v in (stages.get('runtime') or {}).items()
                                                if k.endswith('last_ms')}})
            time.sleep(.08)
    except BaseException as error:
        failure = {'error_type': type(error).__name__, 'error': str(error)}
    finally:
        live.request(api, False, False)
        final = live.wait_state(api, lambda s: live.applied(s, False, False), controls)
        first = live.frame(final)
        final = live.wait_state(api, lambda s: live.applied(s, False, False) and
                               live.frame(s) - first >= 8, controls)
        after_pins = live.source_pins(args.install_receipt)
        after_pins[str(Path(__file__).resolve())] = live.sha(Path(__file__).resolve())
        live.dump(out, 'TRACE.json', {'status': 'capability_trace_completed' if failure is None else 'failed',
                  'samples': rows, 'error': failure,
                  'initial_gpu_handoff': initial.get('gpu_handoff'),
                  'final_gpu_handoff': final.get('gpu_handoff'),
                  'initial_frame_id': live.frame(initial), 'final_frame_id': live.frame(final),
                  'restored_OFF_and_healthy': True, 'source_hashes_before': pins,
                  'source_hashes_after': after_pins, 'source_hashes_unchanged': pins == after_pins,
                  'scope': 'Capability/owner changes only, not a timing or visual acceptance'})
    if failure or pins != after_pins:
        raise RuntimeError('Trace failed or source changed; inspect saved trace')
    print(json.dumps({'status': 'capability_trace_completed', 'snapshots': len(rows),
                      'file': str(out / 'TRACE.json')}))


if __name__ == '__main__':
    main()
