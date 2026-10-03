"""Freeze the selected window-block layout and prove local source/report isolation."""
import ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';PUBLIC=ROOT.parent/'nr-b580-public'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/window-blocks-checkpoint-v1';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args])
assert git(ROOT,'rev-parse','HEAD').strip()==b'5cbaca544e427f1a6c6510dba48cceb26e363af9'
assert git(EXACT,'rev-parse','HEAD').strip()==b'7355848c4b8952fb5e383fe29d3e2fb3cdd00b1d'
assert git(PUBLIC,'rev-parse','HEAD').strip()==b'695ae22a32830c6d266c0636d9fd8fb3ffb46a3a'
assert not git(EXACT,'status','--porcelain') and not git(PUBLIC,'status','--porcelain')
assert not git(ROOT,'diff','--name-only') and not git(ROOT,'diff','--cached','--name-only')
names='''Run-WindowBlocksLong1080V3.cmd
Run-WindowBlocksNativeParityV1.cmd
Run-WindowBlocksNativeParityV2.cmd
Run-WindowBlocksNativeParityV3.cmd
Run-WindowBlocksResidual256V3.cmd
Run-WindowBlocksV1.cmd
Run-WindowBlocksV2.cmd
Run-WindowBlocksV3.cmd
audit_window_blocks_long1080_v3.py
audit_window_blocks_native_parity_v1.py
audit_window_blocks_native_parity_v3.py
audit_window_blocks_residual256_v3.py
audit_window_blocks_v1.py
audit_window_blocks_v2.py
audit_window_blocks_v3.py
benchmark_window_blocks_native_parity_v1.py
benchmark_window_blocks_native_parity_v2.py
benchmark_window_blocks_native_parity_v3.py
benchmark_window_blocks_residual256_v3.py
checkpoint_window_blocks_v1.py
spill_preflight_v1.py
validate_window_blocks_long1080_v3.py
validate_window_blocks_v1.py
validate_window_blocks_v2.py
validate_window_blocks_v3.py
window_block_attention_v3.py
window_block_projection_v1.py
window_block_projection_v3.py
window_blocks_v1.py
window_blocks_v2.py
window_blocks_v3.py
WINDOW_BLOCK_LAYOUT_STATUS.md'''.splitlines()
assert set(git(ROOT,'ls-files','--others','--exclude-standard').decode().splitlines())=={f'experimental/{n}' for n in names}
new_files={str(HERE/n):sha(HERE/n) for n in names};parsed=[]
for n in names:
    if n.endswith('.py'):ast.parse((HERE/n).read_text(encoding='utf-8'));parsed.append(n)
backend=list((ROOT/'backend/nr_backend').glob('*.py'));assert len(backend)==35;exact={}
for p in backend:
    rel='backend/nr_backend/'+p.name;content=p.read_bytes()
    assert content==(EXACT/rel).read_bytes()==git(EXACT,'show','HEAD:'+rel);exact[rel]=sha(p)
lease=ROOT.parent/'xess-tools/work/r4-route-completion-20260907'
assert not any((lease/n).exists() for n in ('gpu-owner.json','STOP'))
pins={
 'experimental/window-blocks-v1':('07b31cda5201207dda13a29b436dcd4549f19e159f5d6dd621e04e238d118cf5','64cdc3a6d4813a800ae78566548108412a86c5c7ea81b413d25081b3cd77b1a0','audit_window_blocks_v1.py'),
 'experimental/window-blocks-v2':('6dc31ff62f58f7879da68dfa43244fcecbc6f6c31f8f7ba2a466c19ade785fc2','869fd1ce407f7f35381600723789d08e2eb304cb409145a0827b98c089319a46','audit_window_blocks_v2.py'),
 'experimental/window-blocks-v3':('e0b7a5f9bf996bc9e825d439761f1cd19f8ed6ff4fa768d2dd28839a0f63f8af','ea46f04dea7d0760d622a3449aae517ea88d975f1eae3dcf6eab7b282a79dec3','audit_window_blocks_v3.py'),
 'results/window-blocks-native-parity-v1':('e32cc0f4a3dc2ca08e9bb497714f31d209e928d7bb3e2cc2cf21a0a2d9c7fc5f','20a2b19fa0839a95ddf83359c2d85cb51350f7c302ddadd8d51db6bea289c888','audit_window_blocks_native_parity_v1.py'),
 'results/window-blocks-native-parity-v2':('17253db66723e130177daa141c21c4d5e1c88ef84ff716c8f2d61f527745c0c3','59923d41a58093ebbccf9a763c585e86d760792f6d69f8c0461193219f660fe2','audit_window_blocks_native_parity_v1.py'),
 'results/window-blocks-native-parity-v3':('6986ddfcaf1adbfe054ac415c69515327682fe40f3ba780d8e92e2786407a63d','ec3fe37fcccfd1ba135f4d95caec05056180c49ca3251fcf3ab544f2eda5f661','audit_window_blocks_native_parity_v3.py'),
 'results/window-blocks-residual256-v3':('c29c1a67b2e1d8314b0f8169a79c7ea51aa33ffbebecbb4078d23dbbf6316586','fbbb4e35479fcc3c589dcc5a0eba723844cecd7e73ca3eab119588a66dd4baba','audit_window_blocks_residual256_v3.py'),
 'results/window-blocks-long1080-v3':('48decfda37749aa61f03d5783ffcc4ee654d9e7875ee32d2706b65779e02e296','482c22ef2831741ccdaed2472364f436e6c011256aab9212e2411ed3d3ca811a','audit_window_blocks_long1080_v3.py')}
sources={};reports={};artifacts={}
for folder,(digest,adigest,auditor) in pins.items():
    p=D/folder/'validation.json';ap=p.with_name('saved-audit-v1.json');lp=p.parent.with_suffix('.log.lease.json')
    assert sha(p)==digest and sha(ap)==adigest;r,a=js(p),js(ap)
    assert r['passed'] and a['passed'] and not r['complete_migration'] and not a['complete_migration']
    assert a['report_sha256']==digest and a['auditor_sha256']==sha(HERE/auditor)
    assert a['lease_sha256']==sha(lp) and js(lp)['returncode']==0
    for path in (p,ap,lp):reports[str(path)]=sha(path)
    for path,h in r['sources'].items():
        if path in sources:assert sources[path]==h
        else:assert sha(path)==h;sources[path]=h
    for path,meta in a.get('artifacts_authenticated',{}).items():
        assert sha(path)==meta['sha256'];artifacts[path]=meta['sha256']
    for path,h in a.get('llir_authenticated',{}).items():assert sha(path)==h;artifacts[path]=h
native=js(D/'results/window-blocks-native-parity-v3/saved-audit-v1.json')
old=native['summary']['previous']['temporal'];new=native['summary']['quant_graph']['temporal']
assert all(n<o for n,o in zip(new['round_mean_host_ms'],old['round_mean_host_ms']))
paired=js(D/'results/window-blocks-residual256-v3/saved-audit-v1.json')['temporal_only']
assert all(n<o for n,o in zip(paired['quant_graph']['round_mean_ms'],paired['previous']['round_mean_ms']))
long=js(D/'results/window-blocks-long1080-v3/saved-audit-v1.json');assert long['frames_verified']==390
record=dict(passed=True,complete_migration=False,selected_runtime_changed=True,selected='c5ac61b stack + window_blocks_v3.WindowBlocks',
    prior_commit='5cbaca544e427f1a6c6510dba48cceb26e363af9',sources=sources,reports=reports,artifacts=artifacts,
    new_files=new_files,python_ast=parsed,exact_backend_files=exact,gpu_leases_closed=True,public_repo_unchanged=True,
    native_continuous_ms=dict(previous=old['mean_host_ms'],selected=new['mean_host_ms']),
    native_change_percent=native['temporal_change_percent'],residual_continuous_ms=paired,
    full_long_frames_verified=390,all_reported_spill_and_correctness_checks_passed=True,
    rejected_performance_candidates=['window_blocks_v1','window_blocks_v2'],checkpoint_script_sha256=sha(__file__))
OUT.mkdir();(OUT/'saved-audit-v1.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,files=len(new_files),python_ast=len(parsed),exact_files=len(exact),sources=len(sources),reports=len(reports),artifacts=len(artifacts),checkpoint_sha256=sha(OUT/'saved-audit-v1.json')),indent=2))
