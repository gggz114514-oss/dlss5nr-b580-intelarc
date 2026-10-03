"""Authenticate unary/capture/primitive receipts and reread all referenced data.

The unary audit also independently indexes the certified table by saved input
bits. Fused matrix comparisons were executed on GPU; this is a saved-artifact
audit, not an independent CPU reimplementation of the complete model.
"""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays
HERE=Path(__file__).resolve().parent
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['unary','capture','mlp']);parser.add_argument('sha256');args=parser.parse_args()
folders={'unary':'native-half-cubic-v1','capture':'current-cubic-operands-v1','mlp':'native-cubic-mlp-v1'}
out=D/'experimental'/folders[args.stage];target=out/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=out/'validation.json';assert sha(p)==args.sha256
r=json.loads(p.read_text(encoding='utf-8'));assert r['passed'] and r['complete_migration'] is False
lease=out.with_suffix('.log.lease.json');l=json.loads(lease.read_text(encoding='utf-8'))
assert l['returncode']==0 and l['finished_unix']>=l['started_unix']
assert all(sha(p)==h for p,h in r['sources'].items())
seen={};texts={}
def walk(v):
    if isinstance(v,dict):
        if v.get('format')=='npy+zlib':
            key=v['path']
            if key in seen:assert seen[key]==v
            else:arrays.load(v);seen[key]=v
        elif all(k in v for k in ('path','sha256','stored_bytes')):
            assert sha(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['stored_bytes'];texts[v['path']]=v
        else:
            for x in v.values():walk(x)
    elif isinstance(v,list):
        for x in v:walk(x)
walk(r)
details={}
if args.stage=='unary':
    table=np.load(D/'experimental/cubic-fp8-lut-v3.npy',allow_pickle=False)
    assert len(r['cases'])==6 and r['all_complete_outputs_equal']
    for c in r['cases']:
        bits=arrays.load(c['input']).view('u2');expected=arrays.load(c['expected'])
        assert len(np.unique(bits))==65536 and expected.tobytes()==table[bits].tobytes()
        assert c['elements']==len(bits) and {m['method'] for m in c['methods']}=={0,1,2}
        for m in c['methods']:
            assert m['byte_equal'] and m['differences']==0 and not m['first_differences']
            assert arrays.load(m['output']).tobytes()==expected.tobytes()
            if m['method']==1:assert 'llvm.fma.f16' in Path(m['ir']['llir']['path']).read_text()
    for t in r['timings']:
        assert t['all_graph_outputs_equal'] and t['persistent_io_outside_pool']
        for k,v in t['samples_ms'].items():assert len(v)==5 and statistics.median(v)==t['median_ms'][k]
    details=dict(complete_output_comparisons=18,all_65536_half_encodings=True,native_half_fma_in_llir=True)
elif args.stage=='capture':
    assert r['byte_equal_prior'] and r['history_unchanged'] and r['vit_qkv_calls']==8
    assert len(r['cases'])==13 and {c['family'] for c in r['cases'].values()}=={'split','c32','batched'}
    size=0
    for c in r['cases'].values():
        assert c['count']>0
        size+=sum(v['raw_bytes'] for v in c['operands'].values())+c['expected']['raw_bytes']
        for name,v in c['operands'].items():
            d=c['descriptors'][name];assert d['shape']==v['shape'] and len(d['stride'])==len(d['shape'])
    assert size==r['operand_raw_bytes']<=r['maximum_operand_raw_bytes']
    details=dict(cases=13,raw_case_bytes=size,calls_by_family={f:sum(c['count'] for c in r['cases'].values() if c['family']==f) for f in ('split','c32','batched')})
else:
    assert len(r['cases'])==13 and (r['rounds'],r['calls_per_graph'],r['replays_per_sample'])==(5,4,10)
    for c in r['cases']:
        assert c['operands_unchanged'] and c['candidates']['previous']['byte_equal']
        for label,samples in c['samples_ms'].items():
            assert len(samples)==5 and statistics.median(samples)==c['median_ms'][label]
            m=c['candidates'][label];assert m['byte_equal'] and m['graph_byte_equal'] and m['persistent_io_outside_pool']
            if label!='previous':assert m['native_half_fma'] and 'llvm.fma.f16' in Path(m['llir']['path']).read_text()
        eligible=[label for label in c['samples_ms'] if label!='previous' and c['median_ms'][label]<=.95*c['median_ms']['previous'] and all(a<=.98*b for a,b in zip(c['samples_ms'][label],c['samples_ms']['previous']))]
        selected=min(eligible,key=lambda x:c['median_ms'][x]) if eligible else 'previous'
        assert selected==c['selected']
        assert c['weighted_saving_ms']==c['count']*(c['median_ms']['previous']-c['median_ms'][selected])
    assert r['weighted_local_saving_ms']==sum(c['weighted_saving_ms'] for c in r['cases'])
    assert r['selected_changes']==sum(c['selected']!='previous' for c in r['cases'])
    details=dict(complete_eager_comparison_receipts=sum(len(c['candidates']) for c in r['cases']),selected=r['selected_changes'],weighted_local_saving_ms=r['weighted_local_saving_ms'])
audit=dict(passed=True,complete_migration=False,report_sha256=sha(p),lease_sha256=sha(lease),auditor_sha256=sha(__file__),sources_authenticated=len(r['sources']),unique_arrays_reread=len(seen),stored_bytes_reread=sum(m['stored_bytes'] for m in seen.values()),ir_artifacts_authenticated=len(texts),limitation=__doc__,**details)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2))
