"""Prewritten CPU-only audit of stored stage diagnostics; never execute the model."""
import hashlib, json, math, statistics, sys, traceback
from datetime import datetime
from pathlib import Path

M = Path('D:/Codex-NR-Experiments/nr-b580/reference/results/repaired-body-stages-v1-monitor-luna-v1/request-from-main.json')
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
m = js(M)
cache = {}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def file(path, expected=None):
    p = Path(path).resolve()
    key = str(p).casefold()
    if key not in cache:
        h, n = hashlib.sha256(), 0
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
                n += len(block)
        cache[key] = dict(path=str(p), sha256=h.hexdigest(), bytes=n)
    item = cache[key]
    assert expected is None or item['sha256'] == expected, p
    return item


assert Path(m['log']).is_file(), 'Known-launched log missing'
if sys.argv[1:] == ['--startup']:
    assert not Path(m['startup']).exists()
    write(m['startup'], dict(read_at=datetime.now().astimezone().isoformat(),
          manifest=str(M), log=m['log'], log_bytes=Path(m['log']).stat().st_size))
    print('Startup recorded')
    sys.exit(0)
assert not sys.argv[1:] and not Path(m['handoff']).exists()
a = dict(passed=False, phase='auditing', no_model_execution=True, no_rerun=True)
try:
    r, lease = js(m['result']), js(m['lease'])
    assert lease['owner'] == m['owner'] and lease['returncode'] == 0
    assert r['passed'] and r['phase'] == 'completed'
    assert not r.get('error') and not r.get('finalization_error')
    for key in ('candidate_promoted', 'new_quantization', 'recalibration', 'new_video', 'tensor_dumps'):
        assert r[key] is False
    for key in ('default_unchanged', 'diagnostic_medians_are_not_additive',
                'full_staged_physical_sequence_equal', 'full_staged_output_byte_equal',
                'stage_trace_summaries_cover_full_body', 'full_graph_output_byte_equal',
                'persistent_io_outside_shared_pool', 'stage_inputs_unchanged',
                'history_seed_constants_static_inputs_and_graphs_unchanged'):
        assert r[key] is True, key
    for p, h in m['critical_sources'].items():
        file(p, h)
    for group in ('sources', 'exact_gate'):
        assert r[group]
        for p, h in r[group].items():
            file(p, h)
    prior_path = Path('D:/Codex-NR-Experiments/nr-b580/reference/results/int8-ffn-range-repair-v1/validation.json')
    file(prior_path, 'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
    prior = js(prior_path)
    expected = {v['frame']: v for v in prior['paired']['runs']['repaired_int8'] if v['round'] == 0}
    assert len(r['warmup']) == 2
    for i, row in enumerate(r['warmup']):
        assert row['frame'] == i and row['byte_equal']
        for key in ('low_raw_sha256', 'full_raw_sha256'):
            assert row[key] == expected[i][key]
    assert r['expected_output_sha256'] == expected[1]['low_raw_sha256']
    counts = dict(triton_calls=643, standalone_fp8=172, quantization_calls=395, elided_fp8=223)
    assert r['full_trace_summary'] == counts
    assert r['capture_build_counts'] == [counts]*6
    names = ['pre', 'encoder_C32', 'encoder_C64', 'encoder_C128', 'encoder_C256',
             'encoder_C512', 'ViT_8_blocks', 'decoder_C512_with_input', 'decoder_C256',
             'decoder_C128', 'decoder_C64', 'decoder_C32', 'post']
    assert [s['name'] for s in r['stages']] == names
    trace = r['physical_sequence']
    file(trace['path'], trace['sha256'])
    assert len(js(trace['path'])) == 643
    for key in ('triton_calls', 'quantization_calls', 'elided_fp8'):
        assert sum(s[key] for s in r['stages']) == counts[key]

    def samples(values, median):
        assert len(values) == 7 and all(math.isfinite(v) and v > 0 for v in values)
        assert math.isclose(statistics.median(values), median, abs_tol=1e-9)

    samples(r['full_body_samples_ms'], r['full_body_median_ms'])
    for row in r['stages']:
        samples(row['samples_ms'], row['median_ms'])
        assert row['physical_sequence_and_fp8_elisions_equal_full_stage'] and row['captured_output_byte_equal']
        for meta in row['inputs'] + row['outputs']:
            if meta is not None:
                assert meta['bytes'] > 0 and len(meta['shape']) == len(meta['stride'])
                assert len(bytes.fromhex(meta['raw_sha256'])) == 32
    assert r['stages'][-1]['outputs'][0]['raw_sha256'] == r['expected_output_sha256']
    assert len(r['orders']) == 7
    workloads = ['full_body'] + names
    for i, order in enumerate(r['orders']):
        offset = i*3 % len(workloads)
        assert order == workloads[offset:] + workloads[:offset]
    assert len(r['graphs']) == 2 and sum(v['replays'] for v in r['graphs']) == 2
    assert all(v['persistent_inputs_and_output_verified_outside_pool'] for v in r['graphs'])
    assert all(group and all(v['spills'] == 0 for v in group.values()) for group in r['resources'].values())
    final = json.loads(Path(m['log']).read_text(encoding='utf-8').splitlines()[-1])
    assert final['passed'] and final['full_body_median_ms'] == r['full_body_median_ms']
    measured = {s['name']: s['median_ms'] for s in r['stages']}
    assert final['stages_ms'] == measured
    a.update(passed=True, phase='completed', primary={k: file(m[k]) for k in ('result', 'log', 'lease')},
        unique_files_checked=len(cache), full_body_median_ms=r['full_body_median_ms'],
        full_body_samples_ms=r['full_body_samples_ms'], stage_medians_ms=measured,
        ranking=sorted(measured, key=measured.get, reverse=True),
        diagnostic_medians_are_not_additive=True, candidate_promoted=False,
        scope='Stored source/numerical/timing audit. No model reexecution; stage graphs are isolated diagnostics.')
except BaseException:
    a.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    write(m['handoff'], a)
print(json.dumps(dict(passed=True, handoff=m['handoff'])))
