"""Independently review saved current-provider timings; no framework/GPU calls."""
from pathlib import Path
import json
import math
import sys

STAGE = Path('E:/ComfyUI-aki-v3-IntelArc_20260722/cyberpunk-b580-nr-opt/artifacts/vit-current-provider-gpu-runner-v4-20261002')
sys.path.insert(0, str(STAGE))
import runner_common as c
import result_review

def main():
    c.verify_sources()
    path = c.DATA / 'compare-01/RESULT.json'
    result = c.read(path)
    c.require(result['status'] == 'FULLMODEL_OPERATOR_CHECKS_PASSED_GAME_PENDING' and
              result['completed'] and not result['game_performance_evidence'], 'No completed full-body result')
    ready = c.checked(result['prepared'])
    prepared = c.verify_prepared(ready.parent)
    children = {tag: c.read(c.checked(row)) for tag, row in result['children'].items()}
    for tag, row in result['processes'].items():
        process = c.read(c.checked(row))
        c.require(process['returncode'] == 0 and not process['timed_out'], 'Unfinished child: ' + tag)
    for tag, child in children.items():
        result_review.validate_arm(child, tag, prepared)
    raw = {tag: c.stats(children[tag]['raw']['samples_ms']) for tag in ('B1', 'C1', 'C2', 'B2')}
    c.require(raw == result['graph_raw'], 'Saved raw summary differs')
    b = (raw['B1']['mean_ms'] + raw['B2']['mean_ms']) / 2
    candidate = (raw['C1']['mean_ms'] + raw['C2']['mean_ms']) / 2
    c.require(result['graph_comparison']['delta_ms'] == b - candidate, 'Saved paired delta differs')
    errors = {}
    for kind, values in result['full_thirteen_frame_errors'].items():
        frames = values['frames']
        c.require([f['frame_id'] for f in frames] == list(range(13)) and
                  all(f['finite'] for f in frames), 'Incomplete thirteen-frame errors')
        count = sum(f['elements'] for f in frames)
        recomputed = dict(max_abs=max(f['max_abs'] for f in frames),
                          mae=sum(f['sum_abs'] for f in frames) / count,
                          rmse=math.sqrt(sum(f['sum_squared'] for f in frames) / count))
        c.require(all(values[key] == number for key, number in recomputed.items()), 'Saved error reduction differs')
        errors[kind] = dict(frames=13, finite=True, **recomputed)
    details = {}
    for tag in ('B1', 'C1'):
        child = children[tag]
        details[tag] = {}
        for name in ('vit.0.score', 'vit.0.value'):
            site = child['provider_observation']['sites'][name]
            key = site['actual_observed_compiler_keys'][0]
            actual = child['cache_gate']['actual_compiler_keys'][key]
            details[tag][name] = dict(source=actual['source'], compiler_key=key, binary=site['binary'],
                                     specialization=json.loads(actual['specialization_data']),
                                     cache_policy=child['cache_gate']['policy'])
    out = c.DATA / 'compare-01/MAIN_RAW_AND_PROTOCOL_REVIEW.json'
    report = dict(status='MAIN_CPU_RAW_PROTOCOL_REVIEW_PASSED_NO_CONFIRMED_SPEEDUP',
                  result=c.record(path), prepared=c.record(ready), freeze=c.record(STAGE / 'FREEZE.json'),
                  arms_raw=raw, baseline_mean_ms=b, candidate_mean_ms=candidate, saved_ms=b-candidate,
                  baseline_first_last_drift_ms=raw['B2']['mean_ms']-raw['B1']['mean_ms'],
                  paired_saved_ms=dict(first=raw['B1']['mean_ms']-raw['C1']['mean_ms'],
                                       last=raw['B2']['mean_ms']-raw['C2']['mean_ms']),
                  errors_from_saved_frame_reductions=errors, actual_matrix_compilation=details,
                  decision='Retain candidate as experiment; no measured gain to promote in current route',
                  one_ms_threshold_used=False, old_local_reference_gain_used=False,
                  metric=result['graph_comparison']['metric'],
                  GPU_executed_now=False, game_calls=0, G_writes=0, game_acceptance=False)
    c.write(out, report)
    print(json.dumps(dict(status=report['status'], baseline_mean_ms=b, candidate_mean_ms=candidate,
                          saved_ms=b-candidate, errors=errors, report=c.record(out))))

if __name__ == '__main__':
    main()
