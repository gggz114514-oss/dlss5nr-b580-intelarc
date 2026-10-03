"""CPU audit of the selected stage diagnostics and unsupported graph timing tags.

Authenticates executed source/lease receipts and saved complete arrays. The
stage outputs are independently compared with the older authenticated fixture,
while GPU capture checks are receipts from the frozen runner, not reexecution.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import compressed_arrays_v1 as arrays

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
PUBLIC = ROOT.parent / 'nr-b580-public'
D = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = D / 'experimental/selected-body-stages-checkpoint-v1'
assert not OUT.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
git = lambda root, *args: subprocess.check_output(['git', '-C', str(root), *args])
assert git(ROOT, 'rev-parse', 'HEAD').strip() == b'2e277b3f9216fbed2bd9cc69403830209f3b3342'
assert not git(ROOT, 'diff', '--name-only') and not git(ROOT, 'diff', '--cached', '--name-only')
for root, head in ((EXACT, b'7355848c4b8952fb5e383fe29d3e2fb3cdd00b1d'),
                   (PUBLIC, b'695ae22a32830c6d266c0636d9fd8fb3ffb46a3a')):
    assert git(root, 'rev-parse', 'HEAD').strip() == head
    assert not git(root, 'status', '--porcelain')

pins = {
    'xpu-graph-events-v1': '7aa167f727f8b10ceb3727daf1f5f3c07c3262c611b038f94d02177ba94360bb',
    'selected-body-stages-v1': 'f2553253c77b49e535c50f176afec1246abf5ee44784643bde1d25a820661e66',
}
sources, reports, artifacts, records = {}, {}, {}, {}
raw_bytes = 0


def walk(value):
    global raw_bytes
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            path, digest = value['path'], value['sha256']
            if path not in artifacts:
                assert sha(path) == digest
                if value.get('format') == 'npy+zlib':
                    raw_bytes += arrays.load(value).nbytes
                artifacts[path] = digest
            else:
                assert artifacts[path] == digest
        for item in value.values():
            walk(item)
    elif isinstance(value, list):
        for item in value:
            walk(item)


for name, digest in pins.items():
    p = D / 'experimental' / name / 'validation.json'
    assert sha(p) == digest
    r = js(p)
    assert r['passed'] and not r['complete_migration']
    records[name] = r
    lease = p.parent.with_suffix('.log.lease.json')
    assert js(lease)['returncode'] == 0 and 'reason' not in js(lease)
    for file in (p, lease, p.parent.with_suffix('.log')):
        reports[str(file)] = sha(file)
    for file, h in r['sources'].items():
        if file not in sources:
            assert sha(file) == h
            sources[file] = h
        else:
            assert sources[file] == h
    walk(r)

event = records['xpu-graph-events-v1']
assert not event['captured_tags_usable'] and not event['captured_samples']
assert event['capture_phase'] == 'replay_and_query'
assert 'Profiling information is unavailable' in event['capture_error']
assert len(event['untagged_samples']) == 3
assert [s['initial'] for s in event['untagged_samples']] == [0, 7, -3]
assert len({s['output_sha256'] for s in event['untagged_samples']}) == 3
assert event['outside_control']['first_ms'] > 0 and event['outside_control']['second_ms'] > 0

r = records['selected-body-stages-v1']
for flag in ('full_staged_physical_sequence_equal', 'full_staged_output_byte_equal',
             'persistent_io_outside_shared_pool', 'stage_inputs_unchanged',
             'diagnostic_medians_are_not_additive', 'history_seed_model_constants_and_static_inputs_unchanged'):
    assert r[flag]
assert not r['candidate_promoted'] and len(r['stages']) == 13
summary = r['full_trace_summary']
assert (summary['triton_calls'], summary['standalone_fp8'], summary['quantization_calls'], summary['elided_fp8']) == (683, 196, 427, 231)
sequence = js(r['physical_sequence']['path'])
assert len(sequence) == 683
assert sum(s['triton_calls'] for s in r['stages']) == 683
assert sum(s['quantization_calls'] for s in r['stages']) == 427
assert sum(s['elided_fp8'] for s in r['stages']) == 231
fixture_path = D / 'experimental/current-body-stages-v2/validation.json'
assert sha(fixture_path) == '144d6f4aa8a28f37a4870af671821a3a30ab22b23614eea23f442c1e3aa1c448'
fixture = js(fixture_path)
assert r['body_inputs'] == fixture['body_inputs'] and r['expected_output'] == fixture['expected_output']
old_stages = {s['name']: s for s in fixture['stages']}
comparisons = 0
for stage in r['stages']:
    assert stage['physical_sequence_and_fp8_elisions_equal_full_stage'] and stage['captured_output_byte_equal']
    old = old_stages[stage['name']]
    assert len(stage['outputs']) == len(old['outputs'])
    for actual, expected in zip(stage['outputs'], old['outputs']):
        # ViT's diagnostic boundary is now flat; the value order is unchanged.
        assert arrays.load(actual).tobytes() == arrays.load(expected).tobytes()
        walk(expected)
        comparisons += 1
    samples = stage['samples_seconds']
    assert len(samples) == 7 and all(math.isfinite(s) and s > 0 for s in samples)
    assert statistics.median(samples) == stage['median_seconds']
assert comparisons == 19
samples = r['full_body_samples_seconds']
assert len(samples) == 7 and all(math.isfinite(s) and s > 0 for s in samples)
assert statistics.median(samples) == r['full_body_median_seconds']
names = ['full_body', *[s['name'] for s in r['stages']]]
assert r['orders'] == [names[(i * 3) % 14:] + names[:(i * 3) % 14] for i in range(7)]

# Check the trace helper against the already committed source even if an older
# runner's transitive source inventory did not include it explicitly.
helper = HERE / 'full_body_dataflow_v1.py'
assert helper.read_bytes() == git(ROOT, 'show', 'HEAD:experimental/full_body_dataflow_v1.py')
sources[str(helper)] = sha(helper)
new = [p for p in git(ROOT, 'ls-files', '--others', '--exclude-standard').decode().splitlines()]
new_hashes = {p: sha(ROOT / p) for p in new}
ast_count = 0
for name in new:
    if name.endswith('.py'):
        ast.parse((ROOT / name).read_text(encoding='utf-8-sig'))
        ast_count += 1
report = dict(passed=True, scope=__doc__, complete_migration=False, candidate_promoted=False,
    sources=sources, reports=reports, artifacts=artifacts, raw_array_bytes_reread=raw_bytes,
    complete_stage_arrays_compared=19, graph_internal_tags_usable=False,
    full_body_median_ms=r['full_body_median_seconds'] * 1000,
    stage_median_ms={s['name']: s['median_seconds'] * 1000 for s in r['stages']},
    new_files=new_hashes, python_ast_count=ast_count)
OUT.mkdir()
(OUT / 'saved-audit-v1.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k not in ('sources', 'reports', 'artifacts', 'new_files')}, indent=2))
