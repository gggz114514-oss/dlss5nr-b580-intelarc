"""CPU evidence audit for the ESIMD feasibility and rounding experiments.

Reopens saved full arrays, authenticates executable sources and native modules,
recomputes recorded timing summaries, and preserves rejected runs. This does
not replay the model or independently reexecute native candidate real matrices.
"""
import ast,hashlib,json,math,statistics,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';PUBLIC=ROOT.parent/'nr-b580-public'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/esimd-dense-checkpoint-v1';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
git=lambda root,*args:subprocess.check_output(['git','-C',str(root),*args])
import numpy as np
import compressed_arrays_v1 as arrays
assert git(ROOT,'rev-parse','HEAD').strip()==b'87e9fd2b56e6cc738607cf0ef63e2fe35a9f3466'
assert not git(ROOT,'diff','--name-only') and not git(ROOT,'diff','--cached','--name-only')
assert git(EXACT,'rev-parse','HEAD').strip()==b'7355848c4b8952fb5e383fe29d3e2fb3cdd00b1d'
assert git(PUBLIC,'rev-parse','HEAD').strip()==b'695ae22a32830c6d266c0636d9fd8fb3ffb46a3a'
assert not git(EXACT,'status','--porcelain') and not git(PUBLIC,'status','--porcelain')
sources={};artifacts={};reports={};summary={};raw_bytes=0

def authenticate(path,h):
    assert sha(path)==h,path
    if path in sources:assert sources[path]==h
    sources[path]=h

def walk(value):
    global raw_bytes
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            path=value['path'];h=value['sha256'];assert sha(path)==h
            if path not in artifacts:
                if value.get('format')=='npy+zlib':
                    data=arrays.load(value);raw_bytes+=data.nbytes
                if 'stored_bytes' in value:assert Path(path).stat().st_size==value['stored_bytes']
                artifacts[path]=h
        for x in value.values():walk(x)
    elif isinstance(value,list):
        for x in value:walk(x)

pins={
    'esimd-dense-v3':'8b8214b33aea961cab0655913748b9a42b655d4c38da6cbc4b42ba38474cd6eb',
    'esimd-dense-v4':'f4863eca80a395d6f678b9a55721d359ca248ade78095492a008cd2a90f7a764',
    'esimd-rounding-v2':'b4a31a828bb2d564e55edca4f6ae06d78408efe453346bd64576ff7a9100e3cb',
}
expected={'esimd-dense-v1':False,'esimd-dense-v2':False,'esimd-dense-v3':True,'esimd-dense-v4':True,'esimd-rounding-v1':False,'esimd-rounding-v2':True}
loaded={}
for name,passed in expected.items():
    p=D/'experimental'/name/'validation.json';r=js(p);loaded[name]=r
    if name in pins:assert sha(p)==pins[name]
    assert r['passed']==passed and not r['complete_migration']
    lp=p.parent.with_suffix('.log.lease.json');lease=js(lp)
    assert lease['returncode']==(0 if passed else 1)
    for path in (p,lp,p.parent.with_suffix('.log')):reports[str(path)]=sha(path)
    for path,h in r['sources'].items():authenticate(path,h)
    walk(r)

for name in ('esimd-dense-v3','esimd-dense-v4'):
    r=loaded[name];assert not r['candidate_promoted'] and r['all_real_routes_byte_equal']
    assert r['general_half_equivalence_rejected'] and not r['smoke_byte_equal']
    assert len(r['cases'])==32 and sum(row['count'] for row in r['cases'])==163
    for resource in r['resources'].values():assert resource['spill_bytes']==0 and resource['max_group']>=16
    table=[];smoke=[]
    for row in r['cases']:
        assert row['inputs_unchanged']
        assert set(row['routes'])=={'triton','esimd_direct_a','esimd_prepacked_only','esimd_with_pack'}
        assert len(row['orders'])==5 and all(set(order)==set(row['routes']) for order in row['orders'])
        for route,checks in row['routes'].items():
            assert all(checks[x] for x in ('byte_equal','graph_captures_native_work','restored_input_replay'))
            samples=row['samples_ms'][route]
            assert len(samples)==5 and all(math.isfinite(x) and x>0 for x in samples)
            assert row['median_ms'][route]==statistics.median(samples)
        table.append(dict(key=row['key'],shape=row['shape'],count=row['count'],initial=row['operands']['initial'] is not None,
                          median_ms=row['median_ms'],pack_inclusive_change_percent=(row['median_ms']['esimd_with_pack']/row['median_ms']['triton']-1)*100))
    for row in r['smoke']:
        counts={}
        for route,meta in row['routes'].items():
            actual=arrays.load(meta['actual']);reference=arrays.load(meta['reference'])
            equal=actual.tobytes()==reference.tobytes();assert equal==meta['byte_equal']
            counts[route]=int(np.count_nonzero(actual.view('u2')!=reference.view('u2')))
        smoke.append(dict(shape=row['shape'],differing_half_elements=counts))
    summary[name]=dict(real_cases=32,represented_captured_calls=163,smoke=smoke,table=table,
        weighted_standalone_diagnostic_not_model_ms={route:sum(row['count']*row['median_ms'][route] for row in r['cases']) for route in r['cases'][0]['routes']})
assert summary['esimd-dense-v3']['smoke'][1]['differing_half_elements']=={'False':1,'True':1}
assert summary['esimd-dense-v4']['smoke'][1]['differing_half_elements']=={'False':0,'True':0}
assert summary['esimd-dense-v4']['smoke'][2]['differing_half_elements']['True']>0

rounding=loaded['esimd-rounding-v2'];assert len(rounding['cases'])==3
for row in rounding['cases']:
    assert row['inputs_unchanged']
    for left in row['comparisons'].values():assert all(count==[0,0,0] for count in left.values())
    for meta in row['native'].values():
        final=arrays.load(meta['post_initial']);half=arrays.load(meta['half'])
        assert final.astype('f2').tobytes()==half.tobytes()
case=rounding['cases'][1];assert case['shape']==[9,32,32]
assert not case['triton']['32']['matches_original']
diff=case['differing_elements'];assert len(diff)==1 and diff[0]['index']==[1,27]
cpu={name:arrays.load(meta) for name,meta in case['inputs'].items()}
exact=cpu['a'].astype('f8')@cpu['w'].astype('f8')+cpu['initial'].astype('f8')
assert float(exact[1,27])==diff[0]['exact_f64']
assert diff[0]['original_half']==2.181640625 and diff[0]['values']['native_True']['half']==2.1796875
summary['rounding']=dict(observed_difference=diff[0],instrumentation_changed_baseline=True,
    first_initial_candidate_fixed_k32_but_failed_other_random_shapes=True,general_half_replacement_rejected=True)

isa=D/'experimental/esimd-rounding-isa-v1'
b32=(isa/'baseline/.text._matmul.asm').read_text();b128=(isa/'baseline-k128/.text._matmul.asm').read_text()
instrumented=(isa/'instrumented/.text._probe.asm').read_text()
assert 'r69:f         r61:f' in b32 and 'dpas.8x8' in b32
assert 'r100:f        null:f' in b128 and 'r22.0<1>:f    r100.0<1;1,0>:f' in b128
assert 'r62:f         null:f' in instrumented and 'r24.0<1>:f    r62.0<1;1,0>:f' in instrumented
for folder in (isa,D/'experimental/esimd-dense-isa-v1'):
    for p in folder.rglob('*'):
        if p.is_file():artifacts[str(p)]=sha(p)
for version in range(1,6):
    folder=D/'experimental'/f'esimd-dense-build-v{version}'
    for p in folder.glob('*'):
        if p.suffix.lower() in ('.dll','.log'):artifacts[str(p)]=sha(p)
for version in (1,2):
    for p in (D/'experimental'/f'esimd-dense-accum-build-v{version}').glob('*'):
        if p.suffix.lower() in ('.dll','.log'):artifacts[str(p)]=sha(p)

names=set()
for pattern in ('Build-Esimd*.cmd','Run-Esimd*.cmd','esimd_dense*.cpp','esimd_dense_adapter_v*.py','benchmark_esimd_dense_v*.py','probe_esimd_rounding_v*.py'):
    names.update(p.name for p in HERE.glob(pattern))
names.update(('audit_esimd_dense_v1.py','ESIMD_DENSE_STATUS.md'))
files={str(HERE/n):sha(HERE/n) for n in sorted(names)};parsed=[]
for n in sorted(names):
    if n.endswith('.py'):ast.parse((HERE/n).read_text(encoding='utf-8'));parsed.append(n)
untracked=set(git(ROOT,'ls-files','--others','--exclude-standard').decode().splitlines())
assert untracked=={f'experimental/{n}' for n in names if not n.endswith('.cpp')}
backend={}
for p in (ROOT/'backend/nr_backend').glob('*.py'):
    rel='backend/nr_backend/'+p.name
    assert p.read_bytes()==(EXACT/rel).read_bytes()==git(EXACT,'show','HEAD:'+rel)
    backend[rel]=sha(p)
assert len(backend)==35
lease_root=ROOT.parent/'xess-tools/work/r4-route-completion-20260907'
assert not any((lease_root/n).exists() for n in ('STOP','gpu-owner.json'))
record=dict(passed=True,complete_migration=False,selected_runtime_changed=False,candidate_promoted=False,
    selected_runtime='fee6d0d WindowBlocks v3 + prior selected FP8 stack',prior_head='87e9fd2b56e6cc738607cf0ef63e2fe35a9f3466',
    sources=sources,reports=reports,artifacts=artifacts,raw_array_bytes_authenticated=raw_bytes,summary=summary,
    new_files=files,python_ast=parsed,exact_backend_files=backend,public_repo_unchanged=True,
    gpu_leases_closed=True,auditor_sha256=sha(__file__))
OUT.mkdir();(OUT/'saved-audit-v1.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,sources=len(sources),reports=len(reports),artifacts=len(artifacts),raw_array_bytes=raw_bytes,
                     files=len(files),python_ast=len(parsed),checkpoint_sha256=sha(OUT/'saved-audit-v1.json')),indent=2))
