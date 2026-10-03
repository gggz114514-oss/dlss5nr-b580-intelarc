"""Authenticate native ViT integration; independently compare saved boundaries.

This rereads saved complete boundary/reference arrays and verifies whole-frame
GPU comparison receipts and timing arithmetic. It does not reexecute the GPU
model. The 1080p residual performance regression prevents runtime promotion.
"""
import ast,hashlib,json,math,statistics,subprocess
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';PUBLIC=ROOT.parent/'nr-b580-public'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/esimd-vit-checkpoint-v1';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
git=lambda root,*args:subprocess.check_output(['git','-C',str(root),*args])
assert git(ROOT,'rev-parse','HEAD').strip()==b'153fa4051f45df10a5b098eb6395073ba4e61ff1'
assert not git(ROOT,'diff','--name-only') and not git(ROOT,'diff','--cached','--name-only')
assert git(EXACT,'rev-parse','HEAD').strip()==b'7355848c4b8952fb5e383fe29d3e2fb3cdd00b1d'
assert git(PUBLIC,'rev-parse','HEAD').strip()==b'695ae22a32830c6d266c0636d9fd8fb3ffb46a3a'
assert not git(EXACT,'status','--porcelain') and not git(PUBLIC,'status','--porcelain')
pins={
 'esimd-vit-native-parity-v1':'41c049efe45167bd277a3fe6bd30f240f83bfb7a5ecad31f9f1a4ce5fc958523',
 'esimd-vit-native-parity-v2':'42dff3710e3db9b5f0312c63bd234acfe49c6aeafea1a2690040cb00db9092f3',
 'esimd-vit-residual256-v2':'d06ead3c94755c56daa35fe5e863ab00305c2d49014804877322f7b3c2b269bb',
}
sources={};reports={};artifacts={};raw_bytes=0;records={};summary={}

def walk(value):
    global raw_bytes
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            path=value['path'];h=value['sha256']
            if path not in artifacts:
                assert sha(path)==h
                if value.get('format')=='npy+zlib':raw_bytes+=arrays.load(value).nbytes
                artifacts[path]=h
            else:assert artifacts[path]==h
        for v in value.values():walk(v)
    elif isinstance(value,list):
        for v in value:walk(v)

for name,h in pins.items():
    p=D/'results'/name/'validation.json';assert sha(p)==h
    r=js(p);records[name]=r
    assert r['passed'] and not r['complete_migration'] and not r['candidate_promoted']
    lease=p.parent.with_suffix('.log.lease.json');assert js(lease)['returncode']==0
    for path in (p,lease,p.parent.with_suffix('.log')):reports[str(path)]=sha(path)
    for path,digest in r['sources'].items():
        if path not in sources:assert sha(path)==digest;sources[path]=digest
        else:assert sources[path]==digest
    walk(r)

legacy_path=D/'results/nr256-native-parity-v1/validation.json'
assert sha(legacy_path)=='5008dfcf5316aabe984e1e30d1db1c933bb3536c5e4a54a3020390c4845ff007'
legacy=js(legacy_path)
for name in ('esimd-vit-native-parity-v1','esimd-vit-native-parity-v2'):
    r=records[name];assert r['complete_outputs_compared']==720
    assert r['full_boundaries_compared']==56 and r['packed_outputs_compared']==8
    assert len(r['boundaries'])==8 and [b['block'] for b in r['boundaries']]==list(range(8))
    for row in r['boundaries']:
        assert row['passed'] and len(row['outputs'])==len(row['references'])==7
        for actual,reference in zip(row['outputs'],row['references']):
            assert arrays.load(actual).tobytes()==arrays.load(reference).tobytes()
        packed=arrays.load(row['packed']);out=arrays.load(row['outputs'][-1])
        assert packed.shape==(8,64,8,16) and packed.transpose(0,2,1,3).reshape(64,1024).tobytes()==out.tobytes()
    for flag in ('inputs_unchanged','reset_reproduces_first','all_candidate_outputs_byte_equal','all_outputs_match_frozen_372','held_outputs_unchanged','lut_bytes_unchanged'):
        assert r[flag]
    pools=set();calculated={}
    for variant,rows in r['measured'].items():
        assert len(rows)==360
        for row,old in zip(rows,legacy['measured']['native_half']):
            assert (row['round'],row['mode'],row['sample'])==(old['round'],old['mode'],old['sample'])
            assert row['output_raw_sha256']==old['output_raw_sha256']
            assert row['seed']==(1 if row['mode']=='reset' else 22+row['sample'])
            assert row['private_equal'] and row['caller_independent'] and math.isfinite(row['host_ms']) and row['host_ms']>0
        calculated[variant]={mode:dict(mean_host_ms=statistics.mean(x['host_ms'] for x in rows if x['mode']==mode),
            median_host_ms=statistics.median(x['host_ms'] for x in rows if x['mode']==mode),
            round_mean_host_ms=[statistics.mean(x['host_ms'] for x in rows if x['mode']==mode and x['round']==i) for i in range(3)]) for mode in ('reset','temporal')}
        graphs=r['graphs'][variant];assert len(graphs)==2
        assert sorted(g['replays'] for g in graphs)==[241,248]
        assert all(g['persistent_inputs_and_output_verified_outside_pool'] for g in graphs)
        assert [g['captured_dispatch'] for g in graphs]==[g['captured_dispatch'] for g in r['graphs']['selected']]
        pools.update(g['pool'] for g in graphs)
    assert len(pools)==1 and calculated==r['summary']
    assert arrays.load(r['reset_output']).tobytes()==arrays.load(legacy['reset_output']).tobytes()
    assert r['chain_calls']==6
    for v in r['resources'].values():assert v['spill_bytes']==0
    for v in r['preflight'].values():assert all(a['spills']==0 for a in v['attempts'])
    for variant,builds in r['builds'].items():
        assert len(builds)==6
        for b in builds:
            assert b['elided_fp8']==231
            if variant=='selected':assert (b['triton_calls'],b['standalone_fp8'],b['quantization_calls'])==(683,196,427)
            else:assert (b['triton_calls'],b['standalone_fp8'],b['quantization_calls'],b['native_dense_calls'])==(668,188,419,8)
    summary[name]=dict(summary=calculated,temporal_change_percent=(calculated['esimd']['temporal']['mean_host_ms']/calculated['selected']['temporal']['mean_host_ms']-1)*100,
        reset_change_percent=(calculated['esimd']['reset']['mean_host_ms']/calculated['selected']['reset']['mean_host_ms']-1)*100,
        all_three_temporal_rounds_faster=all(n<o for o,n in zip(calculated['selected']['temporal']['round_mean_host_ms'],calculated['esimd']['temporal']['round_mean_host_ms'])))

r=records['esimd-vit-residual256-v2'];assert r['full_outputs_verified']==78
for flag in ('packed_weights_unchanged','held_outputs_survive_replay','caller_ownership_guards_passed','inputs_unchanged','lut_bytes_unchanged'):assert r[flag]
prior_path=D/'results/residual-scale-fp16_xmx-256-v1/validation.json'
assert sha(prior_path)=='056251b4e2a839fcc44845d9646d2cd8ae1e9ed7f99fd1858dc7e3d51e3e50f9'
prior=js(prior_path);calculated={};pools=set()
for name,rows in r['runs'].items():
    assert len(rows)==39 and len(r['graphs'][name])==2
    for ordinal,row in enumerate(rows):
        assert (row['round'],row['frame'])==divmod(ordinal,13)
        old=prior['runs'][row['frame']]
        assert row['output']==old['output'] and row['low_nr']==old['low_nr']
        assert row['byte_equal'] and row['private_byte_equal_low_nr']
        assert row['next_seed']==(1 if row['reset'] else row['frame']+1)
        assert math.isfinite(row['host_ms']) and row['host_ms']>0
        dispatch=dict(old['runtime_dispatch']);assert dispatch.pop('xpu_graph_replay')==1
        for k,c in old['captured_dispatch'].items():
            if k!='backend':dispatch[k]=dispatch.get(k,0)+c
        assert row['effective_dispatch']==dispatch
    calculated[name]={mode:dict(mean_host_ms=statistics.mean(row['host_ms'] for row in rows if mode=='all' or not row['reset']),
        round_mean_host_ms=[statistics.mean(row['host_ms'] for row in rows if row['round']==i and (mode=='all' or not row['reset'])) for i in range(3)]) for mode in ('all','temporal')}
    assert all(g['persistent_inputs_and_output_verified_outside_pool'] for g in r['graphs'][name])
    pools.update(g['pool'] for g in r['graphs'][name])
assert len(pools)==1 and calculated==r['summary']
assert len(r['progress_fallback'])==2 and all(g['byte_equal'] and g['native_chain_bypassed'] for g in r['progress_fallback'])
assert len(r['guards'])==4
assert all(g['rejected_before_launch'] and g['output_unchanged'] for g in r['guards'][:2])
assert all(g['invalid_motion_and_table_rejected'] and g['history_seed_and_replays_unchanged'] for g in r['guards'][2:])
assert len(r['lut_graph_guards'])==2 and all(g['rejected'] and g['history_and_seed_unchanged'] and g['no_graph_replay'] for g in r['lut_graph_guards'])
summary['residual']=dict(summary=calculated,temporal_change_percent=(calculated['esimd']['temporal']['mean_host_ms']/calculated['selected']['temporal']['mean_host_ms']-1)*100,
    all_frames_change_percent=(calculated['esimd']['all']['mean_host_ms']/calculated['selected']['all']['mean_host_ms']-1)*100)
assert all(n>o for o,n in zip(calculated['selected']['all']['round_mean_host_ms'],calculated['esimd']['all']['round_mean_host_ms']))
assert summary['residual']['temporal_change_percent']>0

for p in (D/'experimental/esimd-vit-isa-v1').rglob('*'):
    if p.is_file():artifacts[str(p)]=sha(p)
isa=(D/'experimental/esimd-vit-isa-v1/.text._ZTS17NRDpasVitExpandV1ILb1EE.asm').read_text()
assert 'dpas.8x8' in isa and 'r13:hf' in isa
assert records['esimd-vit-native-parity-v1']['preflight']['merge']['attempts'][0]['shared_bytes']==1024
assert records['esimd-vit-native-parity-v2']['preflight']['merge']['attempts'][0]['shared_bytes']==0
names='''Build-EsimdVitExpandV1.cmd
Run-EsimdVitNativeParityV1.cmd
Run-EsimdVitNativeParityV2.cmd
Run-EsimdVitResidual256V2.cmd
benchmark_esimd_vit_native_parity_v1.py
benchmark_esimd_vit_native_parity_v2.py
benchmark_esimd_vit_residual256_v2.py
esimd_vit_chain_v1.py
esimd_vit_chain_v2.py
esimd_vit_graph_v1.py
esimd_vit_graph_v2.py
esimd_vit_expand_v1.cpp
nr256_selected_stack_v2.py
audit_esimd_vit_v1.py
ESIMD_VIT_STATUS.md'''.splitlines()
new_files={str(HERE/n):sha(HERE/n) for n in names};parsed=[]
for n in names:
    if n.endswith('.py'):ast.parse((HERE/n).read_text(encoding='utf-8'));parsed.append(n)
assert set(git(ROOT,'ls-files','--others','--exclude-standard').decode().splitlines())=={f'experimental/{n}' for n in names if not n.endswith('.cpp')}
backend={}
for p in (ROOT/'backend/nr_backend').glob('*.py'):
    rel='backend/nr_backend/'+p.name;assert p.read_bytes()==(EXACT/rel).read_bytes()==git(EXACT,'show','HEAD:'+rel)
    backend[rel]=sha(p)
assert len(backend)==35
lease_root=ROOT.parent/'xess-tools/work/r4-route-completion-20260907'
assert not any((lease_root/n).exists() for n in ('STOP','gpu-owner.json'))
record=dict(passed=True,complete_migration=False,candidate_promoted=False,selected_runtime_changed=False,
    selected_runtime='fee6d0d WindowBlocks v3 + c5ac61b FP8 stack',prior_head='153fa4051f45df10a5b098eb6395073ba4e61ff1',
    performance_decision='Reject both as selected runtime: v1 weak/inconsistent native benefit; v2 regressed the paired complete 1080p residual path.',
    long_390_not_run='Paired residual performance gate failed; no promotion and no visual change exposed to user.',
    sources=sources,reports=reports,artifacts=artifacts,raw_array_bytes_authenticated=raw_bytes,summary=summary,
    new_files=new_files,python_ast=parsed,exact_backend_files=backend,public_repo_unchanged=True,gpu_leases_closed=True,auditor_sha256=sha(__file__))
OUT.mkdir();(OUT/'saved-audit-v1.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,sources=len(sources),reports=len(reports),artifacts=len(artifacts),raw_array_bytes=raw_bytes,files=len(new_files),python_ast=len(parsed),
    checkpoint_sha256=sha(OUT/'saved-audit-v1.json'),performance=summary),indent=2))
