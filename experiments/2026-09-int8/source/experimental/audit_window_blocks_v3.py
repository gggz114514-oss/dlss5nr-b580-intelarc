"""Authenticate full frame and per-block GPU receipts and saved block arrays.

This rereads the saved inputs/results and checks trace receipts; it is not an
independent CPU NR arithmetic replay. No speed conclusion follows from counts.
"""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');out=D/'experimental/window-blocks-v3'
p=out/'validation.json';target=out/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
parser=argparse.ArgumentParser();parser.add_argument('report_sha256');args=parser.parse_args()
assert sha(p)==args.report_sha256;r=js(p);lp=out.with_suffix('.log.lease.json')
assert r['passed'] and not r['complete_migration'] and not r['timing_measured'] and js(lp)['returncode']==0
assert all(sha(p)==h for p,h in r['sources'].items())
assert r['lut_unchanged'] and r['history_and_seed_unchanged_by_block_validation']
assert len(r['frames'])==2 and len(r['builds'])==6
old=js(D/'results/residual-scale-fp16_xmx-256-v1/validation.json')
verified={};llir={}
def read(meta):
    if meta['path'] not in verified:
        a=arrays.load(meta);assert np.isfinite(a).all()
        verified[meta['path']]=dict(sha256=sha(meta['path']),bytes=Path(meta['path']).stat().st_size,shape=list(a.shape),dtype=a.dtype.str)
    return verified[meta['path']]
for i,row in enumerate(r['frames']):
    assert row['frame']==i and row['next_seed']==i+1
    assert all(row[k] for k in ('byte_equal','history_equal','inputs_unchanged'))
    for k in ('low_nr','output'):assert row[k]==old['runs'][i][k];read(row[k])
for b in r['builds']:
    assert (b['triton_calls'],b['standalone_fp8'],b['elided_fp8'],b['quantization_calls'])==(683,196,231,427)
calls=r['normal_block_calls'];assert len(calls)==216
first=calls[:36];assert len({x['module'] for x in first})==36
assert Counter(x['shape'][-1] for x in first)=={64:8,128:12,256:16}
for i in range(6):assert calls[i*36:(i+1)*36]==first
proofs=r['block_proofs'];assert len(proofs)==72
for i,proof in enumerate(proofs):
    info=first[i%36]
    assert all(proof[k]==v for k,v in info.items())
    assert proof['all_bytes_equal'] and proof['unquantized_crop_equal']
    for k in ('input','output'):
        desc=read(proof[k]);assert desc['shape']==proof['shape'] and desc['dtype']=='<f2'
    assert proof['packed_bytes_equal'] and proof['spills']==proof['attention_spills']==0
    desc=read(proof['packed']);hp,wp,c=proof['padded_shape']
    assert desc['shape']==[c//32,hp//8,wp//8,64,32]
    attention=proof['attention_llir'];assert sha(attention['path'])==attention['sha256'];llir[attention['path']]=attention['sha256']
    for selection in (proof['projection_selection'],proof['attention_selection']):
        assert selection['attempts'][-1]['spills']==0 and selection['selected']==selection['attempts'][-1]['config']
    artifact=proof['llir'];assert sha(artifact['path'])==artifact['sha256']
    assert Path(artifact['path']).stat().st_size==artifact['stored_bytes'];llir[artifact['path']]=artifact['sha256']
spill=r['spill_guard'];assert spill['selected']==[16,32] and [a['spills'] for a in spill['attempts']]==[832,0]
assert spill['no_gpu_dispatch'] and spill['sentinel_output_unchanged']
guards=r['shape_guards'];assert len(guards)==6
assert {(g['channels'],g['kind']) for g in guards}=={(c,k) for c in (64,128,256) for k in ('zero','strided')}
assert all(g['all_bytes_equal'] and g['shape']==[10,12,g['channels']] for g in guards)
audit=dict(passed=True,complete_migration=False,report_sha256=sha(p),auditor_sha256=sha(__file__),lease_sha256=sha(lp),
    sources_authenticated=len(r['sources']),real_block_outputs_verified=72,shape_cases_verified=6,
    artifacts_authenticated=verified,llir_authenticated=llir,unique_array_bytes=sum(v['bytes'] for v in verified.values()),
    spills=sorted({b['spills'] for b in proofs}),spill_guard=spill,limitation=__doc__)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k not in ('artifacts_authenticated','llir_authenticated')},indent=2))
