"""CPU receipt audit and immutable source checkpoint; no repeated GPU execution.

Recompute timing summaries and compare all saved output hashes/metadata against
frozen references. Actual full-array comparisons are the authenticated GPU run
receipts; this script does not claim another replay or full-array reread.
"""
import ast,hashlib,json,statistics,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=D/'experimental/layout-crop-checkpoint-v1'
assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
git=lambda root,*args:subprocess.check_output(['git','-C',str(root),*args]).decode().strip()
assert git(ROOT,'rev-parse','HEAD')=='a6d3ef0ea5765188420d9bcd9f03a395e0909df5'
assert not git(ROOT,'diff','--name-only') and not git(ROOT,'diff','--cached','--name-only')
for name,head in (('nr-b580','7355848c4b8952fb5e383fe29d3e2fb3cdd00b1d'),
                  ('nr-b580-public','695ae22a32830c6d266c0636d9fd8fb3ffb46a3a')):
    assert git(ROOT.parent/name,'rev-parse','HEAD')==head
    assert not git(ROOT.parent/name,'status','--porcelain')
pins={
 'native':('layout-crop-native-parity-v1','e54f31291b1b628d2d988bf19bbc8fe43cdb8f0c19d80def21ec7f2418973109'),
 'suite':('layout-crop-validation-suite-v1','cb48355400195d65fa9aeb00fff0dea1415e1d05f7634f2bca2b2498add4c29c'),
 'residual':('layout-crop-residual256-v1','b0d5b7e984ca530d36459b3c20651b3989f37719aae57bcee3bb6eb64e265fe8'),
 'long':('layout-crop-long1080-v1','193e041c029d04f53f6d31fe23ed930f160effb8e9e48690e9163a72d1ebdadd')}
sources={};reports={};loaded={}
for name,(folder,h) in pins.items():
    path=D/'results'/folder/'validation.json';assert sha(path)==h
    r=js(path);assert r['passed'] and not r.get('error') and not r.get('finalization_error')
    reports[str(path)]=h;loaded[name]=r
    for key in ('sources','exact_gate'):
        for p,digest in r.get(key,{}).items():
            if p in sources:assert sources[p]==digest
            else:assert sha(p)==digest;sources[p]=digest
    if name in ('native','suite'):
        lease=path.parent.with_suffix('.log.lease.json')
        assert js(lease)['returncode']==0
        for p in (lease,path.parent.with_suffix('.log')):reports[str(p)]=sha(p)
assert [p['name'] for p in loaded['suite']['phases']]==['residual','long']
for phase in loaded['suite']['phases']:
    assert phase['passed'] and phase['returncode']==0
    assert sha(phase['result']['path'])==phase['result']['sha256']
legacy=js(D/'results/nr256-native-parity-v1/validation.json')
expected={(r['round'],r['mode'],r['sample']):r['output_raw_sha256'] for r in legacy['measured']['native_half']}
for rows in loaded['native']['measured'].values():
    assert len(rows)==360 and {(r['round'],r['mode'],r['sample']):r['output_raw_sha256'] for r in rows}==expected
prior=js(D/'results/residual-scale-fp16_xmx-256-v1/validation.json')
for rows in loaded['residual']['runs'].values():
    assert len(rows)==39
    for row in rows:
        old=prior['runs'][row['frame']]
        assert row['byte_equal'] and row['output']==old['output'] and row['low_nr']==old['low_nr']
approved=js(D/'results/batched-residual-long1080-review-v2/validation.json')
long=loaded['long'];assert len(long['frames'])==long['frames_completed']==390
for i,(r,old) in enumerate(zip(long['frames'],approved['frames'])):
    assert r['frame']==i and r['next_seed']==i+1 and r['reset']==(i==0)
    assert r['low_nr']==old['low_nr'] and r['motion']==old['motion']
    assert r['low_raw_sha256']==old['runs']['batched']['low_raw_sha256']
    assert r['full_raw_sha256']==old['runs']['batched']['full_raw_sha256']
    assert all(r[k] for k in ('low_bytes_equal','full_hash_equal','private_byte_equal_low','held_outputs_unchanged','inputs_unchanged'))
assert long['reset_reproduces_first_frame'] and long['independent_uninterrupted_history']
changes={}
for kind in ('native','residual'):
    r=loaded[kind];rows_key='measured' if kind=='native' else 'runs'
    changes[kind]={}
    for name,rows in r[rows_key].items():
        for mode,summary in r['summary'][name].items():
            subset=[v for v in rows if (v['mode']==mode if kind=='native' else (mode=='all' or not v['reset']))]
            assert statistics.mean(v['host_ms'] for v in subset)==summary['mean_host_ms']
            assert [statistics.mean(v['host_ms'] for v in subset if v['round']==i) for i in range(3)]==summary['round_mean_host_ms']
    for mode,old in r['summary']['selected'].items():
        new=r['summary']['layout_crop'][mode]
        assert all(a>b for a,b in zip(old['round_mean_host_ms'],new['round_mean_host_ms']))
        changes[kind][mode]=dict(selected_ms=old['mean_host_ms'],layout_crop_ms=new['mean_host_ms'],
            change_percent=(new['mean_host_ms']/old['mean_host_ms']-1)*100)
new_files={p:sha(ROOT/p) for p in git(ROOT,'ls-files','--others','--exclude-standard').splitlines()}
for p in new_files:
    if p.endswith('.py'):ast.parse((ROOT/p).read_text(encoding='utf-8-sig'))
factory=ast.parse((HERE/'nr256_selected_stack_v4.py').read_text())
assert len(factory.body)==2 and isinstance(factory.body[1],ast.ImportFrom)
assert factory.body[1].module=='layout_crop_stack_v1' and [(n.name,n.asname) for n in factory.body[1].names]==[('Stack',None)]
report=dict(passed=True,scope=__doc__,sources=sources,reports=reports,new_files=new_files,changes=changes,
    selected_factory='experimental/nr256_selected_stack_v4.py',candidate_promoted=True,complete_migration=False,
    native_full_outputs=720,residual_full_outputs=78,long_frames=390,no_gpu_execution=True)
OUT.mkdir();path=OUT/'saved-audit-v1.json'
path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,source_count=len(sources),new_files=len(new_files),checkpoint=str(path),sha256=sha(path))))
