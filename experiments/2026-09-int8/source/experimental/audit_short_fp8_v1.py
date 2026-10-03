"""CPU artifact audit of equivalent rounding, full-call timing and long history.

Saved arrays are reread completely; runtime-only output comparisons are frozen
GPU receipts. This does not rerender video or turn the fast branch into exact.
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
EXACT,PUBLIC = ROOT.parent/'nr-b580',ROOT.parent/'nr-b580-public'
D = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = D/'experimental/short-fp8-checkpoint-v1'
assert not OUT.exists()
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
git = lambda root,*args:subprocess.check_output(['git','-C',str(root),*args])
assert git(ROOT,'rev-parse','HEAD').strip() == b'c16bb9f854a5d7d51b1183af5a28a4677e73c97d'
assert not git(ROOT,'diff','--name-only') and not git(ROOT,'diff','--cached','--name-only')
for root,head in ((EXACT,b'7355848c4b8952fb5e383fe29d3e2fb3cdd00b1d'),(PUBLIC,b'695ae22a32830c6d266c0636d9fd8fb3ffb46a3a')):
    assert git(root,'rev-parse','HEAD').strip() == head and not git(root,'status','--porcelain')
pins = {
    'experimental/short-fp8-body-v1':('661efbe2187e3bf50528fee1a33d1b24bce12d512e7746be18f0f0118f90e544',False),
    'experimental/short-fp8-body-v2':('48ea3ee90be2f6e8166125a4917cfee1a46248a138f73bde90a1757b1a06aead',True),
    'results/short-fp8-native-parity-v1':('667af6f0741bd2a0715650dae2bcbd80f1e387bf63f8c4c9a8b1b1da8d660bc7',False),
    'results/short-fp8-native-parity-v2':('ddad6f922110fbb04775a3d7b2f65e131d03b1b553e5c1ab42fe854a7ca2f914',True),
    'results/short-fp8-residual256-v1':('fd7fe3b8f30e512a45fe766cf22be189fb5430193b72a588f5226957f00a588f',True),
    'results/short-fp8-long1080-v1':('38612d4dac943e0e533c319ddbf3fc4db862e4f8ddc847053958f472099ef8a1',True),
}
sources,reports,artifacts,records = {},{},{},{}
raw_bytes = 0

def walk(value):
    global raw_bytes
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            p,h = value['path'],value['sha256']
            if p not in artifacts:
                assert sha(p) == h
                if value.get('format') == 'npy+zlib':
                    raw_bytes += arrays.load(value).nbytes
                artifacts[p] = h
            else:
                assert artifacts[p] == h
        for item in value.values():walk(item)
    elif isinstance(value,list):
        for item in value:walk(item)

for name,(h,passed) in pins.items():
    path = D/name/'validation.json'
    assert sha(path) == h
    r = js(path);records[name] = r
    assert r['passed'] is passed and not r['complete_migration']
    lease = path.parent.with_suffix('.log.lease.json')
    assert js(lease)['returncode'] == (0 if passed else 1) and 'reason' not in js(lease)
    for p in (path,lease,path.parent.with_suffix('.log')):reports[str(p)] = sha(p)
    for p,h in r['sources'].items():
        if p in sources:assert sources[p] == h
        else:assert sha(p) == h;sources[p] = h
    walk(r)

framework_path = D/'experimental/short-fp8-framework-v1/validation.json'
framework = js(framework_path)
assert framework['passed'] and framework['no_gpu_execution']
assert all(framework[k] for k in ('module_alias_deduplicated','exception_cleanup_restores_every_binding',
    'interference_detected_and_all_bindings_restored','registry_unchanged'))
for p,h in framework['sources'].items():
    if p in sources:assert sources[p] == h
    else:assert sha(p) == h;sources[p] = h
reports[str(framework_path)] = sha(framework_path)
body = records['experimental/short-fp8-body-v2']
e = body['exhaustive']
assert e['encodings'] == 65536 and e['different_words'] == 0 and e['byte_equal']
assert arrays.load(e['old']).tobytes() == arrays.load(e['new']).tobytes()
assert arrays.load(body['changed_input_checks'][0]['output']).tobytes() == arrays.load(body['changed_input_checks'][1]['output']).tobytes()
assert arrays.load(body['changed_input_checks'][0]['output']).tobytes() != arrays.load(body['expected_output']).tobytes()
assert all(len(v) == 7 and all(math.isfinite(x) and x>0 for x in v) for v in body['samples_seconds'].values())
assert all(statistics.median(v) == body['median_seconds'][n] for n,v in body['samples_seconds'].items())
assert all(a>b for a,b in zip(body['samples_seconds']['selected'],body['samples_seconds']['short_fp8']))
assert body['bindings_restored'] and body['all_outputs_byte_equal']
assert len(body['resources']) == 57 and all(r['spills'] == 0 for r in body['resources'].values())
assert 'Graph session requires its own installed arithmetic provider' in records['results/short-fp8-native-parity-v1']['error']
assert 'spill' in records['experimental/short-fp8-body-v1']['error'].lower()

native = records['results/short-fp8-native-parity-v2']
residual = records['results/short-fp8-residual256-v1']
long = records['results/short-fp8-long1080-v1']
legacy = js(D/'results/nr256-native-parity-v1/validation.json')
expected = {(r['round'],r['mode'],r['sample']):r['output_raw_sha256'] for r in legacy['measured']['native_half']}
for n,rows in native['measured'].items():
    assert len(rows) == len(expected) == 360
    assert {(r['round'],r['mode'],r['sample']):r['output_raw_sha256'] for r in rows} == expected
    for mode in ('reset','temporal'):
        values = [r['host_ms'] for r in rows if r['mode']==mode]
        assert all(math.isfinite(v) and v>0 for v in values)
        assert statistics.mean(values) == native['summary'][n][mode]['mean_host_ms']
        assert [statistics.mean(r['host_ms'] for r in rows if r['mode']==mode and r['round']==i) for i in range(3)] == native['summary'][n][mode]['round_mean_host_ms']
prior = js(D/'results/residual-scale-fp16_xmx-256-v1/validation.json')
for n,rows in residual['runs'].items():
    assert len(rows) == 39
    for row in rows:
        old = prior['runs'][row['frame']]
        assert row['byte_equal'] and row['output'] == old['output'] and row['low_nr'] == old['low_nr']
    for mode in ('all','temporal'):
        subset = [r for r in rows if mode=='all' or not r['reset']]
        assert all(math.isfinite(r['host_ms']) and r['host_ms']>0 for r in subset)
        assert statistics.mean(r['host_ms'] for r in subset) == residual['summary'][n][mode]['mean_host_ms']
        assert [statistics.mean(r['host_ms'] for r in subset if r['round']==i) for i in range(3)] == residual['summary'][n][mode]['round_mean_host_ms']
for report in (native,residual,long):
    candidate = report['candidate']
    assert len(candidate['resources']) == 57 and all(r['spills'] == 0 for r in candidate['resources'].values())
    assert candidate['capture_scopes'] == 6 and len(candidate['forks']) == 28 and candidate['bindings'] == 60
    assert report['no_dependency_swaps_on_steady_replay']
    builds = report['builds'] if report is long else [b for v in report['builds'].values() for b in v]
    assert all((b['triton_calls'],b['standalone_fp8'],b['quantization_calls'],b['elided_fp8']) == (683,196,427,231) for b in builds)
assert native['full_outputs_verified'] == 720 and residual['full_outputs_verified'] == 78
approved = js(D/'results/batched-residual-long1080-review-v2/validation.json')
assert len(long['frames']) == long['frames_completed'] == 390 and long['graph_replays'] == 393
for i,(row,old) in enumerate(zip(long['frames'],approved['frames'])):
    assert row['frame'] == i and row['next_seed'] == i+1 and row['reset'] == (i==0)
    assert row['low_nr'] == old['low_nr'] and row['motion'] == old['motion']
    assert row['full_raw_sha256'] == old['runs']['batched']['full_raw_sha256']
    assert row['low_raw_sha256'] == old['runs']['batched']['low_raw_sha256']
    assert row['input_rgb8_sha256'] == old['input_rgb8_sha256']
    assert all(row[k] for k in ('low_bytes_equal','full_hash_equal','private_byte_equal_low','held_outputs_unchanged','inputs_unchanged'))
assert long['independent_uninterrupted_history'] and long['reset_reproduces_first_frame']
assert not long['new_quality_change'] and not long['new_video_encoded'] and not long['new_output_arrays']
changes = {}
for label,r in (('native',native),('residual',residual)):
    changes[label] = {}
    for mode in r['summary']['selected']:
        old,new = r['summary']['selected'][mode],r['summary']['short_fp8'][mode]
        assert all(a>b for a,b in zip(old['round_mean_host_ms'],new['round_mean_host_ms']))
        changes[label][mode] = dict(selected_ms=old['mean_host_ms'],short_fp8_ms=new['mean_host_ms'],
            change_percent=(new['mean_host_ms']/old['mean_host_ms']-1)*100)
new_files = git(ROOT,'ls-files','--others','--exclude-standard').decode().splitlines()
new_hashes = {p:sha(ROOT/p) for p in new_files}
ast_count = 0
for p in new_files:
    if p.endswith('.py'):
        ast.parse((ROOT/p).read_text(encoding='utf-8-sig'));ast_count += 1
report = dict(passed=True,scope=__doc__,sources=sources,reports=reports,artifacts=artifacts,
    raw_array_bytes_reread=raw_bytes,complete_migration=False,candidate_promoted=True,
    selected_factory='experimental/nr256_selected_stack_v3.py',changes=changes,
    exhaustive_half_encodings=65536,native_full_outputs=720,residual_full_outputs=78,long_frames=390,
    new_files=new_hashes,python_ast_count=ast_count)
OUT.mkdir()
(OUT/'saved-audit-v1.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('sources','reports','artifacts','new_files')},indent=2))
