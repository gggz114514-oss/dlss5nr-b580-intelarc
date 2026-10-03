"""Authenticate completed eight-case QKV comparisons and reread saved tensors.

GPU comparisons covered complete outputs. This authenticates those receipts;
it does not independently execute matrix products or normalization on CPU.
"""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays

out=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/fused-vit-qkv-v1')
path=out/'validation.json';target=out/'saved-audit-v1.json'
assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(path)=='0b69ebfe1d836a30724dfd4a92794cd3cbf4bfd29e647486958736b4d053c8e6'
r=js(path);lease=out.with_suffix('.log.lease.json')
assert r['passed'] and r['all_candidates_byte_equal'] and not r['complete_migration']
assert js(lease)['returncode']==0 and r['eight_blocks_match_prior'] and r['history_unchanged']
assert all(sha(p)==h for p,h in r['sources'].items())
assert len(r['cases'])==8
verified={};labels={'previous','b16n32','b16n64','b32n32','b32n64'}
for index,row in enumerate(r['cases']):
    assert row['index']==index and row['inputs_unchanged']
    assert set(row['candidates'])==set(row['samples_ms'])==labels
    for label,c in row['candidates'].items():
        assert c['eager_byte_equal'] and c['graph_byte_equal'] and c['persistent_io_outside_pool']
        assert len(row['samples_ms'][label])==5 and all(0<t<100 for t in row['samples_ms'][label])
        assert statistics.median(row['samples_ms'][label])==row['median_ms'][label]
    for meta in [*row['operands'].values(),*row['outputs']]:
        value=arrays.load(meta)
        assert value.dtype==np.dtype('f2') and np.isfinite(value).all()
        verified[meta['path']]=meta['stored_bytes']
    assert [m['shape'] for m in row['outputs']]==[[64,32,32]]*3
    assert row['operands']['a']['shape']==[64,1024]
    assert row['operands']['w']['shape']==[1024,3072]
    assert row['operands']['scale']['shape']==[32]
assert r['sum_median_ms']=={label:sum(row['median_ms'][label] for row in r['cases']) for label in labels}
audit=dict(passed=True,report_sha256=sha(path),lease_sha256=sha(lease),auditor_sha256=sha(__file__),
    actual_cases=8,candidate_tiles=4,complete_qkv_eager_comparisons=32,
    unique_arrays_reread=len(verified),stored_bytes_reread=sum(verified.values()),
    sum_median_ms=r['sum_median_ms'],scope=__doc__,complete_migration=False)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
