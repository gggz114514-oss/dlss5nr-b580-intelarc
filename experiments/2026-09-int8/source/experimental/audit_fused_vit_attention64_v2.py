"""Authenticate saved ViT64 attention comparisons and reread complete references.

The GPU run compared candidate intermediate tensors and final outputs in full.
This auditor checks those receipts, sources, IR and saved inputs/reference
outputs; it does not independently recompute attention on CPU.
"""
import argparse,hashlib,json,statistics
import numpy as np
from pathlib import Path
import compressed_arrays_v1 as arrays
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
out=D/'experimental/fused-vit-attention64-v2';p=out/'validation.json';target=out/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('sha256');args=parser.parse_args();assert sha(p)==args.sha256
r=json.loads(p.read_text(encoding='utf-8'));lease=out.with_suffix('.log.lease.json');l=json.loads(lease.read_text(encoding='utf-8'))
assert r['passed'] and r['complete_migration'] is False and l['returncode']==0 and l['finished_unix']>=l['started_unix']
assert r['all_candidates_byte_equal'] and all(sha(p)==h for p,h in r['sources'].items())
assert (r['rounds'],r['calls_per_graph'],r['replays_per_sample'])==(5,4,10)
assert [c['name'] for c in r['cases']]==[f'block{i}' for i in range(8)]+['zero','signed_zero','positive_saturation','mixed_fp8_extremes','contiguous_actual','strided_channels']
seen={};irs={}
exponent=r['all_half_exponent'];assert exponent['byte_equal']
bits=arrays.load(exponent['inputs']).view(np.uint16)
assert np.array_equal(bits,np.arange(65536,dtype=np.uint16))
assert arrays.load(exponent['expected']).tobytes()==arrays.load(exponent['actual']).tobytes()
for meta in (exponent['inputs'],exponent['expected'],exponent['actual']):
    seen[meta['path']]=meta['stored_bytes']
meta=exponent['llir'];assert sha(meta['path'])==meta['sha256']
assert 'llvm.fma.f16' in Path(meta['path']).read_text(encoding='utf-8');irs[meta['path']]=meta
for row in r['cases']:
    assert row['operands_unchanged'] and set(row['candidates'])=={'0','1','2'}
    assert len(row['strides'])==3 and all(min(s)>0 for s in row['strides'])
    assert set(row['boundaries'])=={'scores','exponential','numerator','denominator','reciprocal'}
    for m in [*row['operands'].values(),*row['boundaries'].values(),row['expected']]:
        if m['path'] not in seen:arrays.load(m);seen[m['path']]=m['stored_bytes']
    for c in row['candidates'].values():
        assert c['byte_equal'] and c['debug_and_production_match']
        assert set(c['boundaries'])==set(row['boundaries']) and all(c['boundaries'].values())
        meta=c['llir'];assert sha(meta['path'])==meta['sha256'];irs[meta['path']]=meta
        assert 'llvm.fma.f16' in Path(meta['path']).read_text(encoding='utf-8')
    if row['timed']:
        assert row['all_graph_outputs_match'] and row['persistent_io_outside_pool']
        assert set(row['samples_ms'])=={'previous','fused_compensated','0','1','2'}
        for name,ts in row['samples_ms'].items():assert len(ts)==5 and min(ts)>0 and statistics.median(ts)==row['median_ms'][name]
        assert all(set(order)==set(row['samples_ms']) and len(order)==5 for order in row['orders']) and len(row['orders'])==5
timed=[c for c in r['cases'] if c['timed']];assert len(timed)==8
assert r['sum_median_ms']=={label:sum(c['median_ms'][label] for c in timed) for label in ('previous','fused_compensated','0','1','2')}
audit=dict(passed=True,complete_migration=False,report_sha256=sha(p),lease_sha256=sha(lease),auditor_sha256=sha(__file__),
    sources_authenticated=len(r['sources']),exponent_encodings_verified=65536,actual_qkv_cases=8,synthetic_and_layout_cases=6,complete_candidate_comparisons=42,
    mathematical_boundary_comparisons=252,unique_arrays_reread=len(seen),stored_bytes_reread=sum(seen.values()),ir_artifacts_authenticated=len(irs),
    sum_median_ms=r['sum_median_ms'],limitation=__doc__)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2))
