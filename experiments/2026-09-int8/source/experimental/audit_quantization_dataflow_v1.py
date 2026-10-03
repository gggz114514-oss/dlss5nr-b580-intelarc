"""Audit value edges, immutable sources, saved stages and FP8 rewrite receipts.

Complete redundancy inputs and outputs were compared in the primitive GPU run.
This audit replays metadata edges and reads saved stage/unary arrays; it does
not independently execute the NR arithmetic or regenerate the trace.
"""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays

D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
parser=argparse.ArgumentParser();parser.add_argument('folder');parser.add_argument('sha256');args=parser.parse_args()
out=(D/args.folder).resolve();assert out.is_relative_to(D.resolve())
p=out/'validation.json';target=out/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(p)==args.sha256
r=json.loads(p.read_text(encoding='utf-8'));lp=out.with_suffix('.log.lease.json');lease=json.loads(lp.read_text(encoding='utf-8'))
assert r['passed'] and not r['complete_migration'] and lease['returncode']==0
assert all(sha(p)==h for p,h in r['sources'].items())
for flag in ('all_instrumented_stage_outputs_match','assembled_byte_equal_prior','stage_inputs_unchanged','history_unchanged','lut_unchanged'):assert r[flag]
oldpath=D/'experimental/current-body-stages-v2/validation.json'
assert sha(oldpath)=='144d6f4aa8a28f37a4870af671821a3a30ab22b23614eea23f442c1e3aa1c448'
old=json.loads(oldpath.read_text(encoding='utf-8'));assert len(r['stages'])==len(old['stages'])==13
seen={};totals=Counter();producer_totals=Counter();verified_edges=0
def reread(m,finite=True):
    if m is not None and m['path'] not in seen:
        value=arrays.load(m)
        assert not finite or np.isfinite(value).all()
        seen[m['path']]=m['stored_bytes']
for row,previous in zip(r['stages'],old['stages']):
    assert row['name']==previous['name'] and row['instrumented_byte_equal']
    assert row['output_sha256']==previous['output_sha256']==[m['raw_sha256'] for m in row['outputs']]
    for m in row['inputs']+row['outputs']:reread(m)
    events=row['execution']['events'];summary=row['execution']['summary'];states={}
    def reference(v,index):
        global verified_edges
        actual=(v['revision'],v['producer'])
        if v['storage'] in states:assert actual==states[v['storage']]
        else:
            assert actual==(0,None) and v['domain'] is None
            states[v['storage']]=actual
        assert v['producer'] is None or 0<=v['producer']<index
        assert len(v['shape'])==len(v['stride']) and v['offset']>=0
        verified_edges+=1
    for index,event in enumerate(events):
        assert event['id']==index and event['kind'] in ('aten','triton')
        reads=list(event['reads'].values()) if event['kind']=='triton' else event['reads']
        for v in reads:reference(v,index)
        writes=list(event['writes'].values()) if event['kind']=='triton' else event['writes']
        for v in writes:
            previous_revision=states.get(v['storage'],(0,None))[0]
            assert v['revision']==previous_revision+1 and v['producer']==index
            states[v['storage']]=(v['revision'],index)
            if v['domain'] is not None:
                assert v['domain']=='satfinite_e4m3_decoded_half' and v['dtype']=='torch.float16'
                assert v['offset']==0 and v['logical_bytes']==v['storage_bytes']
        for v in event.get('outputs',[]):assert states[v['storage']]==(v['revision'],v['producer'])
    fp8=[e for e in events if e['name']=='nr_backend.triton_fp8._kernel']
    for e in fp8:assert e['redundant_fp8_proof']==(e['reads']['X']['domain']=='satfinite_e4m3_decoded_half')
    producers=Counter('external' if e['reads']['X']['producer'] is None else events[e['reads']['X']['producer']]['name'] for e in fp8)
    counts=dict(triton_calls=sum(e['kind']=='triton' for e in events),aten_calls=sum(e['kind']=='aten' for e in events),standalone_fp8=len(fp8),proven_redundant_fp8=sum(e['redundant_fp8_proof'] for e in fp8))
    assert all(summary[k]==v for k,v in counts.items()) and summary['fp8_input_producers']==dict(producers.most_common())
    totals.update(counts);producer_totals.update(producers)
    if 'rewrite' in row:
        rewrite=row['rewrite'];assert row['rewritten_byte_equal']
        assert row['all_redundancy_inputs_equal_quantized_outputs']==counts['proven_redundant_fp8']
        assert rewrite['elided_fp8']==len(rewrite['elisions'])==counts['proven_redundant_fp8']
        assert rewrite['triton_calls']==counts['triton_calls']-rewrite['elided_fp8']
        assert rewrite['standalone_fp8']==counts['standalone_fp8']-rewrite['elided_fp8']
assert r['execution_totals']==dict(totals)
for m in r['body_inputs'].values():reread(m)
reread(r['expected_output']);assert r['expected_output']['raw_sha256']==r['stages'][-1]['output_sha256'][0]
if 'all_half_fp8_idempotent' in r:
    unary=r['all_half_fp8_idempotent'];assert unary['passed']
    assert np.array_equal(arrays.load(unary['inputs']).view(np.uint16),np.arange(65536,dtype=np.uint16))
    assert arrays.load(unary['once']).tobytes()==arrays.load(unary['twice']).tobytes()
    for k in ('inputs','once','twice'):reread(unary[k],finite=False)
    assert len(r['rewrite_guards'])==4 and all(r['rewrite_guards'].values())
    assert r['stage_rewrite_totals']==dict(triton_calls=774,standalone_fp8=251,elided_fp8=212)
    assert len(r['graph_rewrite_builds'])==6
    for b in r['graph_rewrite_builds']:
        assert (b['triton_calls'],b['standalone_fp8'],b['quantization_calls'],b['elided_fp8'])==(755,232,463,231)
        assert len(b['elisions'])==231 and all(e['input']['domain']=='satfinite_e4m3_decoded_half' for e in b['elisions'])
audit=dict(passed=True,complete_migration=False,report_sha256=sha(p),auditor_sha256=sha(__file__),lease_sha256=sha(lp),
    sources_authenticated=len(r['sources']),unique_arrays_reread=len(seen),stored_bytes_reread=sum(seen.values()),
    value_read_edges_verified=verified_edges,execution_totals=dict(totals),fp8_input_producers=dict(producer_totals.most_common()),
    rewrite_validated='all_half_fp8_idempotent' in r,limitation=__doc__)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2),flush=True)
