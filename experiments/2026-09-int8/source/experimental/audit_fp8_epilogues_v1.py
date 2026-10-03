"""Audit saved full-use proof, FP8 epilogue validation, artifacts and rejection guards.

This authenticates actual GPU comparisons; it is not an independent CPU NR replay.
Synthetic negative cases test the planner's conservative use and storage rules.
"""
import copy,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays
from plan_fp8_epilogues_v1 import plan,MATRIX,QUANTIZE
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
tp=D/'experimental/full-body-dataflow-v1/validation.json'
pp=D/'experimental/fp8-epilogue-plan-v1/plan.json'
vp=D/'experimental/fp8-epilogues-v1/validation.json'
target=vp.with_name('saved-audit-v1.json');assert not target.exists()
pins={str(tp):'1f71df725b0360c1fe449c1b6a18e8f1090ae03d6b1edb38c81356efa1f8e064',
      str(pp):'aa5b8db36edba548bb061fd91339e20df8f1723ab384c0ffc18d876d1f06a267',
      str(vp):'be2ace2418b866f76490994dba3da332516b4fa21412ad3c606800c6a81f11b3'}
assert all(sha(p)==h for p,h in pins.items())
t,p,v=js(tp),js(pp),js(vp);sources={};artifacts={};edges=0
for path,report in ((tp,t),(vp,v)):
    assert report['passed'] and not report['complete_migration'] and not report['timing_measured']
    lp=path.parent.with_suffix('.log.lease.json');assert js(lp)['returncode']==0;pins[str(lp)]=sha(lp)
    for name,digest in report['sources'].items():
        assert sha(name)==digest
        if name in sources:assert sources[name]==digest
        sources[name]=digest
    assert report['lut_unchanged'] and len(report['frames'])==2
    for i,row in enumerate(report['frames']):
        assert row['frame']==i and row['next_seed']==i+1
        assert row['byte_equal'] and row['history_equal'] and row['inputs_unchanged']
        for key in ('low_nr','output'):
            meta=row[key]
            if meta['path'] not in artifacts:
                a=arrays.load(meta);assert np.isfinite(a).all()
                artifacts[meta['path']]=sha(meta['path'])
assert p['trace_sha256']==sha(tp) and sha(Path(__file__).with_name('plan_fp8_epilogues_v1.py'))==p['planner_sha256']
assert p['passed'] and not p['complete_migration']
for mode,g in t['graphs'].items():
    assert mode in ('reset','temporal');states={};ordinals=Counter()
    for i,e in enumerate(g['events']):
        assert e['id']==i
        reads=e['reads'].values() if e['kind']=='triton' else e['reads']
        for a in reads:
            state=(a['revision'],a['producer'])
            if a['storage'] not in states:
                assert state==(0,None) and a['domain'] is None;states[a['storage']]=state
            assert state==states[a['storage']] and (a['producer'] is None or a['producer']<i)
            edges+=1
        writes=e['writes'].values() if e['kind']=='triton' else e['writes']
        for a in writes:
            assert a['revision']==states.get(a['storage'],(0,None))[0]+1 and a['producer']==i
            states[a['storage']]=(a['revision'],i)
        for a in e.get('outputs',[]):assert states[a['storage']]==(a['revision'],a['producer'])
        if e['kind']=='triton':
            assert e['kernel_ordinal']==ordinals[e['name']];ordinals[e['name']]+=1
        else:assert isinstance(e['arguments'],list) and isinstance(e['keyword_arguments'],dict)
    assert states[g['result']['storage']]==(g['result']['revision'],g['result']['producer'])
    s=g['summary'];assert (sum(ordinals.values()),ordinals[QUANTIZE],ordinals[MATRIX])==(755,232,167)
    assert (s['triton_calls'],s['standalone_fp8'],s['elided_fp8'])==(755,232,231)
    assert plan(g)==p['plans'][mode]
    selected=p['plans'][mode]['selected']
    assert len(selected)==92 and len(p['plans'][mode]['rejected'])==75
    assert p['plans'][mode]['removable_quantizers']==92
assert len(t['builds'])==len(v['builds'])==6
for b in v['builds']:
    assert (b['triton_calls'],b['standalone_fp8'],b['elided_fp8'])==(663,140,323)
    assert b['epilogues']['matrix_calls']==167
    assert b['epilogues']['fused_ordinals']==[s['kernel_ordinal'] for s in p['plans']['reset']['selected']]
assert v['history_and_seed_unchanged_by_matrix_validation'] and len(v['matrix_validation_builds'])==2
proofs=[]
for b in v['matrix_validation_builds']:
    ps=b['epilogues']['proofs'];assert len(ps)==92
    for proof,selected in zip(ps,p['plans']['reset']['selected']):
        assert proof['all_bytes_equal'] and proof['ordinal']==selected['kernel_ordinal']
        assert proof['shape']==selected['output']['shape'] and proof['bytes']==selected['output']['logical_bytes']
        artifact=proof['llir'];assert sha(artifact['path'])==artifact['sha256']
        assert Path(artifact['path']).stat().st_size==artifact['stored_bytes']
        artifacts[artifact['path']]=artifact['sha256'];proofs.append(proof)

# Build a small valid use graph; independently add one violating consumer each time.
template=next(e for e in t['graphs']['reset']['events'] if e['kind']=='triton' and e['name']==MATRIX and e['kernel_ordinal']==p['plans']['reset']['selected'][0]['kernel_ordinal'])
m=copy.deepcopy(template);m['id']=0;m['kernel_ordinal']=0
value=m['writes']['OUT'];value.update(storage=1,revision=1,producer=0)
other=copy.deepcopy(value);other.update(storage=2,producer=1)
q=dict(id=1,kind='triton',name=QUANTIZE,reads={'X':copy.deepcopy(value)},writes={'Y':copy.deepcopy(other)})
base=dict(events=[m,q],result=copy.deepcopy(other))
def accepted(g):return len(plan(g)['selected'])==1
assert accepted(base);guards={'direct_quantizer_accepted':True}
def reject(label,edit):
    g=copy.deepcopy(base);edit(g);assert not accepted(g),label;guards[label]=True
reject('raw_kernel_use_rejected',lambda g:g['events'].append(dict(id=2,kind='triton',name='unknown.raw',reads={'X':copy.deepcopy(value)},writes={})))
reject('escaping_output_rejected',lambda g:g.update(result=copy.deepcopy(value)))
reject('later_storage_write_rejected',lambda g:g['events'].append(dict(id=2,kind='aten',name='aten.zero_.default',reads=[],writes=[dict(value,revision=2,producer=2)],outputs=[])))
reject('partial_storage_rejected',lambda g:g['events'][0]['writes']['OUT'].update(offset=1))
reject('float_output_rejected',lambda g:g['events'][0]['writes']['OUT'].update(dtype='torch.float32'))
reject('int8_matrix_rejected',lambda g:g['events'][0]['constants'].update(INT8=True))
reject('batched_matrix_rejected',lambda g:g['events'][0]['constants'].update(BATCHED=True))
reject('different_reduction_tile_rejected',lambda g:g['events'][0]['constants'].update(BK=64))
reject('unquantized_leaf_rejected',lambda g:g['events'].pop())
def through_pad(g,val,dtype='torch.float16'):
    middle=dict(other,storage=3,dtype=dtype)
    pad=dict(id=1,kind='aten',name='aten.pad.default',reads=[copy.deepcopy(value)],writes=[middle],outputs=[middle],arguments=[{'tensor_argument':True},[1,1],'constant',val],keyword_arguments={})
    g['events'][1]['id']=2;g['events'][1]['reads']['X']=copy.deepcopy(middle)
    g['events'].insert(1,pad)
g=copy.deepcopy(base);through_pad(g,0);assert accepted(g);guards['zero_padding_accepted']=True
reject('nonzero_padding_rejected',lambda g:through_pad(g,0.3))
reject('dtype_conversion_rejected',lambda g:through_pad(g,0,'torch.float32'))
g=copy.deepcopy(base)
alias=dict(id=2,kind='aten',name='aten.transpose.int',reads=[copy.deepcopy(value)],writes=[],outputs=[copy.deepcopy(value)])
g['events'].append(alias);assert accepted(g);guards['half_alias_accepted']=True
alias['outputs'][0]['dtype']='torch.int16';g['events'][-1]=alias
assert not accepted(g);guards['reinterpret_alias_rejected']=True
audit=dict(passed=True,complete_migration=False,report_sha256=sha(vp),trace_sha256=sha(tp),plan_sha256=sha(pp),
    auditor_sha256=sha(__file__),pins=pins,sources_authenticated=len(sources),artifacts_authenticated=artifacts,
    value_read_edges_verified=edges,matrix_output_proofs=len(proofs),proof_bytes=sum(x['bytes'] for x in proofs),
    spills=sorted(set(x['spills'] for x in proofs)),planner_guards=guards,limitation=__doc__)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in audit.items() if k not in ('artifacts_authenticated','pins')},indent=2))
