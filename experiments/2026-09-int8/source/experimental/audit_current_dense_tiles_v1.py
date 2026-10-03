"""Authenticate complete dense-output comparisons and recompute tile selection.

Saved arrays are reread; the GPU performed actual candidate arithmetic. The
weighted local timing estimates are not complete NR latency measurements.
"""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays

root=Path('D:/Codex-NR-Experiments/nr-b580/reference')
out=root/'experimental/current-dense-tiles-v1'
path=out/'validation.json';target=out/'saved-audit-v1.json';assert not target.exists()
p=argparse.ArgumentParser();p.add_argument('report_sha256');args=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(path)==args.report_sha256
r=js(path);lease=out.with_suffix('.log.lease.json')
assert r['passed'] and js(lease)['returncode']==0 and not r['complete_migration']
assert all(sha(p)==h for p,h in r['sources'].items())
capture_path=root/'experimental/current-dense-operands-v2/validation.json'
assert sha(capture_path)=='46a16ba64e9f84425928d0a0fd93862e99574910b031ba098c789f0ca69e7290'
capture=js(capture_path)
assert len(r['cases'])==len(capture['cases'])==34 and r['all_original_dense_calls_accounted']
assert r['nr_input']==[256,256] and r['source_canvas']==[1920,1080]
assert (r['rounds'],r['calls_per_graph'],r['replays_per_sample'])==(5,4,10)
assert sum(row['count'] for row in r['cases'])==167
verified={};rejected=0;comparisons=0
for row in r['cases']:
    old=capture['cases'][row['key']]
    assert row['operands']==old['operands'] and row['descriptors']==old['descriptors'] and row['expected']==old['output']
    assert row['shape']==old['shape'] and row['count']==old['count'] and row['inputs_unchanged']
    assert len(row['candidates'])==8
    for label,candidate in row['candidates'].items():
        assert candidate['tile'][2:]==[32,4]
        if not candidate['byte_equal']:
            rejected+=1;assert label not in row['samples_ms'];continue
        comparisons+=1
        assert candidate['graph_byte_equal'] and candidate['persistent_io_outside_pool']
        samples=row['samples_ms'][label]
        assert len(samples)==5 and all(np.isfinite(t) and 0<t<10000 for t in samples)
        assert statistics.median(samples)==row['median_ms'][label]
    labels=list(row['samples_ms'])
    for repetition,order in enumerate(row['orders']):
        offset=repetition%len(labels);expected=labels[offset:]+labels[:offset]
        if repetition%2:expected.reverse()
        assert order==expected
    assert len(row['orders'])==5
    base=row['baseline'];medians=row['median_ms']
    eligible=[name for name in labels if name!=base and medians[name]<medians[base]*.95
              and all(b/c>1.02 for b,c in zip(row['samples_ms'][base],row['samples_ms'][name]))]
    selected=min(eligible,key=medians.get) if eligible else base
    assert row['selected']==selected and row['speedup']==medians[base]/medians[selected]
    assert row['weighted_saving_ms']==row['count']*(medians[base]-medians[selected])
    for meta in [*row['operands'].values(),row['expected']]:
        if meta is not None and meta['path'] not in verified:
            value=arrays.load(meta);assert value.dtype==np.dtype('f2') and np.isfinite(value).all()
            verified[meta['path']]=meta['stored_bytes']
assert r['selected_changes']==sum(c['selected']!=c['baseline'] for c in r['cases'])
assert r['weighted_saving_ms']==sum(c['weighted_saving_ms'] for c in r['cases'])
audit=dict(passed=True,report_sha256=sha(path),lease_sha256=sha(lease),auditor_sha256=sha(__file__),
    geometries=34,dense_calls=167,complete_eager_comparison_receipts=comparisons,rejected_candidates=rejected,
    selected_changes=r['selected_changes'],weighted_local_saving_ms=r['weighted_saving_ms'],
    unique_arrays_reread=len(verified),stored_bytes_reread=sum(verified.values()),
    limitation=__doc__,complete_migration=False)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2))
