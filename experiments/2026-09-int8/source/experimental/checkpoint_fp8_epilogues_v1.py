"""Freeze this non-promoted architecture experiment and verify exact source isolation."""
import ast,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';PUBLIC=ROOT.parent/'nr-b580-public'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=D/'experimental/fp8-epilogues-checkpoint-v1';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args])
assert git(ROOT,'rev-parse','HEAD').strip()==b'c5ac61bc474c38fd19c7203a278736e3b6143455'
assert git(EXACT,'rev-parse','HEAD').strip()==b'7355848c4b8952fb5e383fe29d3e2fb3cdd00b1d'
assert git(PUBLIC,'rev-parse','HEAD').strip()==b'695ae22a32830c6d266c0636d9fd8fb3ffb46a3a'
assert not git(EXACT,'status','--porcelain') and not git(PUBLIC,'status','--porcelain')
assert not git(ROOT,'diff','--name-only') and not git(ROOT,'diff','--cached','--name-only')
names='''Run-FP8EpiloguesNativeParityV1.cmd
Run-FP8EpiloguesNativeParityV2.cmd
Run-FP8EpiloguesResidual256V2.cmd
Run-FP8EpiloguesV1.cmd
Run-FullBodyDataflowV1.cmd
analyze_layout_boundaries_v1.py
analyze_layout_boundaries_v2.py
audit_fp8_epilogues_native_parity_v1.py
audit_fp8_epilogues_residual256_v2.py
audit_fp8_epilogues_v1.py
benchmark_fp8_epilogues_native_parity_v1.py
benchmark_fp8_epilogues_native_parity_v2.py
benchmark_fp8_epilogues_residual256_v2.py
checkpoint_fp8_epilogues_v1.py
fp8_epilogue_graph_v1.py
fp8_epilogue_graph_v2.py
fp8_epilogue_matmul_v1.py
full_body_dataflow_v1.py
nr256_selected_stack_v1.py
plan_fp8_epilogues_v1.py
trace_full_body_dataflow_v1.py
validate_fp8_epilogues_v1.py
FP8_EPILOGUE_ARCHITECTURE_STATUS.md'''.splitlines()
expected={f'experimental/{n}' for n in names}
actual=git(ROOT,'ls-files','--others','--exclude-standard').decode().splitlines()
assert set(actual)==expected,(set(actual)-expected,expected-set(actual))
new_files={str(HERE/n):sha(HERE/n) for n in names};parsed=[]
for n in names:
    if n.endswith('.py'):ast.parse((HERE/n).read_text(encoding='utf-8'));parsed.append(n)
backend=list((ROOT/'backend/nr_backend').glob('*.py'));assert len(backend)==35
exact={}
for p in backend:
    rel='backend/nr_backend/'+p.name;content=p.read_bytes()
    assert content==(EXACT/rel).read_bytes()==git(EXACT,'show','HEAD:'+rel)
    exact[rel]=sha(p)
lease_root=ROOT.parent/'xess-tools/work/r4-route-completion-20260907'
assert not any((lease_root/n).exists() for n in ('gpu-owner.json','STOP'))
pins={
 'experimental/full-body-dataflow-v1/validation.json':'1f71df725b0360c1fe449c1b6a18e8f1090ae03d6b1edb38c81356efa1f8e064',
 'experimental/fp8-epilogue-plan-v1/plan.json':'aa5b8db36edba548bb061fd91339e20df8f1723ab384c0ffc18d876d1f06a267',
 'experimental/fp8-epilogues-v1/validation.json':'be2ace2418b866f76490994dba3da332516b4fa21412ad3c606800c6a81f11b3',
 'experimental/fp8-epilogues-v1/saved-audit-v1.json':'c4d06bce490a1c7fe44890ab2de99f6af45d6b1835317390d1a75d2cad109e7e',
 'results/fp8-epilogues-native-parity-v1/validation.json':'72ee5d9f854fdc2e116ecea6e9077d9b01a035f5f397430719f0a1f90dd7828e',
 'results/fp8-epilogues-native-parity-v1/saved-audit-v1.json':'07fa370359b625dd2fc8480daa72158fd9d4f275d361279a05e7625e7079ba22',
 'results/fp8-epilogues-native-parity-v2/validation.json':'2fe4b5161b981dfea06a33e9d75662993ca2a719211c358d822e146950ac83ba',
 'results/fp8-epilogues-native-parity-v2/saved-audit-v1.json':'cd35ac1017cbd431672e8d8597a2e9d0bab3e82d626b641aae7b125140edcc2d',
 'results/fp8-epilogues-residual256-v2/validation.json':'94acf5b65f55597bbe66d4f44561eb20e6ef60469d932c913d4a007a0e8eda99',
 'results/fp8-epilogues-residual256-v2/saved-audit-v1.json':'20b30af6cd960de995df0845605baa0e1d4c99436ffe8760ef5c4f42135d8c11',
 'experimental/layout-boundaries-v2/analysis.json':'7af8cb3d1de93f11855f27055ed5fdb2ce92479a28efb2e253a0f6326356c191'}
sources={};reports={}
for rel,digest in pins.items():
    path=D/rel;assert sha(path)==digest;r=js(path);assert r['passed'] and not r.get('complete_migration',False)
    reports[str(path)]=digest
    for p,h in r.get('sources',{}).items():
        if p in sources:assert sources[p]==h
        else:assert sha(p)==h;sources[p]=h
    if path.name=='validation.json':
        lp=path.parent.with_suffix('.log.lease.json');assert js(lp)['returncode']==0;reports[str(lp)]=sha(lp)
    if path.name=='saved-audit-v1.json':
        assert r['report_sha256']==sha(path.with_name('validation.json'))
        auditor={'fp8-epilogues-v1':'audit_fp8_epilogues_v1.py',
          'fp8-epilogues-native-parity-v1':'audit_fp8_epilogues_native_parity_v1.py',
          'fp8-epilogues-native-parity-v2':'audit_fp8_epilogues_native_parity_v1.py',
          'fp8-epilogues-residual256-v2':'audit_fp8_epilogues_residual256_v2.py'}[path.parent.name]
        assert r['auditor_sha256']==sha(HERE/auditor)
        for p,h in r.get('artifacts_authenticated',{}).items():assert sha(p)==h
plan=js(D/'experimental/fp8-epilogue-plan-v1/plan.json')
assert plan['planner_sha256']==sha(HERE/'plan_fp8_epilogues_v1.py')
trace=js(D/'experimental/full-body-dataflow-v1/validation.json')
layout=js(D/'experimental/layout-boundaries-v2/analysis.json')
for mode,g in layout['graphs'].items():
    for site in g['layout_sites']:
        e=trace['graphs'][mode]['events'][site['event']]
        assert e['reads']==[site['source']] and e['outputs']==site['outputs']
        assert e['arguments']==site['arguments'] and e['keyword_arguments']==site['keyword_arguments']
        for use in site['consumers']:
            ue=trace['graphs'][mode]['events'][use['event']]
            assert ue['name']==use['name'] and ue['reads'][use['role']]==use['value']
failure_path=D/'experimental/layout-boundaries-v1-failure/failure.json';f=js(failure_path)
assert not f['passed'] and not f['gpu_run'] and sha(f['source'])==f['source_sha256'];reports[str(failure_path)]=sha(failure_path)
record=dict(passed=True,complete_migration=False,selected_runtime_changed=False,
    selected_runtime_checkpoint='c5ac61bc474c38fd19c7203a278736e3b6143455',
    reason='92 epilogues regressed native complete-host time; 84 improved native by 0.68% but did not improve residual pipeline.',
    sources=sources,reports=reports,new_files=new_files,python_ast=parsed,exact_backend_files=exact,
    gpu_leases_closed=True,public_repo_unchanged=True,layout_inventory_cpu_failure_recorded=True,
    no_new_long390_or_visual_review_for_unselected_candidate=True,checkpoint_script_sha256=sha(__file__))
OUT.mkdir();(OUT/'saved-audit-v1.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,files=len(new_files),python_ast=len(parsed),exact_files=len(exact),sources=len(sources),reports=len(reports),checkpoint_sha256=sha(OUT/'saved-audit-v1.json')),indent=2))
