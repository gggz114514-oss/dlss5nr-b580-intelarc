"""Inventory real layout materializations in the authenticated selected NR256 body.

These are logical tensor footprints and launch requests, NOT measured DRAM bytes,
GPU time, or predictions of speedup. Allocation/view metadata is kept separate.
Use producer/consumer identities to design explicit block input/output layouts;
do not remove copies merely because their operation name looks redundant.
"""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
tp=D/'experimental/full-body-dataflow-v1/validation.json'
ap=D/'experimental/fp8-epilogues-v1/saved-audit-v1.json'
out=D/'experimental/layout-boundaries-v1';assert not out.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(tp)=='1f71df725b0360c1fe449c1b6a18e8f1090ae03d6b1edb38c81356efa1f8e064'
t=js(tp);a=js(ap)
assert t['passed'] and a['passed'] and a['trace_sha256']==sha(tp)
assert all(sha(p)==h for p,h in t['sources'].items())
assert all(a['planner_guards'].values()) and a['value_read_edges_verified']==8613
key=lambda v:(v['storage'],v['revision'])
materializations={'aten.contiguous.default','aten.pad.default','aten.repeat_interleave.self_int'}
result={}
for mode,graph in t['graphs'].items():
    events=graph['events'];consumers=defaultdict(list);operations=defaultdict(lambda:dict(calls=0,logical_read_bytes=0,logical_write_bytes=0))
    for e in events:
        reads=list(e['reads'].values()) if e['kind']=='triton' else e['reads']
        writes=list(e['writes'].values()) if e['kind']=='triton' else e['writes']
        row=operations[e['kind']+' '+e['name']];row['calls']+=1
        row['logical_read_bytes']+=sum(v['logical_bytes'] for v in reads)
        row['logical_write_bytes']+=sum(v['logical_bytes'] for v in writes)
        refs=e['reads'].items() if e['kind']=='triton' else enumerate(e['reads'])
        for role,v in refs:consumers[key(v)].append(dict(event=e['id'],kind=e['kind'],name=e['name'],role=role,value=v))
    sites=[]
    for e in events:
        if e['kind']!='aten' or e['name'] not in materializations or not e['writes']:continue
        assert len(e['reads'])==1
        source=e['reads'][0]
        uses=[]
        for target in {key(v) for v in e['writes']}:
            uses.extend(consumers[target])
        sites.append(dict(event=e['id'],operation=e['name'],source=source,
            source_producer_name=None if source['producer'] is None else events[source['producer']]['name'],
            outputs=e['outputs'],arguments=e['arguments'],keyword_arguments=e['keyword_arguments'],consumers=uses))
    counts=Counter(s['operation'] for s in sites)
    assert counts=={'aten.contiguous.default':52,'aten.pad.default':64,'aten.repeat_interleave.self_int':12}
    result[mode]=dict(operations=dict(operations),layout_sites=sites,layout_site_counts=dict(counts),
        allocation_only_operations={name:row for name,row in operations.items() if name.startswith('aten aten.empty')},
        escaping_result=graph['result'])
report=dict(passed=True,complete_migration=False,timing_measured=False,scope=__doc__,
    source_trace_sha256=sha(tp),source_audit_sha256=sha(ap),analyzer_sha256=sha(__file__),graphs=result,
    sources={str(p):sha(p) for p in (tp,ap,Path(__file__))})
out.mkdir();(out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,analysis_sha256=sha(out/'analysis.json'),counts={m:g['layout_site_counts'] for m,g in result.items()}),indent=2))
