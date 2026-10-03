"""Authenticate the native-half attention paired pipeline and reread every unique saved array.

Actual candidate-versus-reference comparisons occurred on complete tensors in
the GPU runs. Reused artifacts are not an independent CPU arithmetic replay.
"""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
parser=argparse.ArgumentParser();parser.add_argument('folder');args=parser.parse_args()
out=(DREF/args.folder).resolve();assert out.is_relative_to(DREF.resolve())
path=out/'validation.json';target=out/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
r=js(path);lease=out.with_suffix('.log.lease.json')
assert r['passed'] and not r['complete_migration'] and js(lease)['returncode']==0
assert all(sha(p)==h for p,h in r['sources'].items())
verified={};cache={}
def signature(meta,cast=None,allow_nonfinite=False):
    key=(meta['path'],meta['sha256'],cast,allow_nonfinite)
    if key not in cache:
        assert sha(meta['path'])==meta['sha256']
        a=arrays.load(meta) if meta.get('format')=='npy+zlib' else np.load(meta['path'],allow_pickle=False)
        assert allow_nonfinite or np.isfinite(a).all()
        if cast:a=a.astype(cast)
        cache[key]=(a.shape,a.dtype.str,hashlib.sha256(a.tobytes()).hexdigest())
        verified[meta['path']]=Path(meta['path']).stat().st_size
    return cache[key]
summary={}
assert 'full_outputs_verified' in r


assert r['lut_bytes_unchanged'] and r['lut_shared_before_capture']
guards=r['lut_graph_guards']
assert [g['kind'] for g in guards]==['buffer_replacement','in_place_version_change']
assert all(g['rejected'] and g['history_and_seed_unchanged'] and g['no_graph_replay'] for g in guards)
assert set(r['runs'])=={'previous','native_half'}
residual='residual' in out.name
old=(js(DREF/'results/residual-scale-fp16_xmx-256-v1/validation.json')['runs'][:13] if residual
     else js(DREF/'results/fast-precision-864x480-v1/validation.json')['runs']['fp16_xmx'])
n=len(r['runs']);assert n==2 and r['full_outputs_verified']==39*n
assert r['caller_ownership_guards_passed'] and r['held_outputs_survive_replay']
pools=set();temporal={}
for name,rows in r['runs'].items():
    assert len(rows)==39 and len(r['graphs'][name])==2
    assert all(g['persistent_inputs_and_output_verified_outside_pool'] for g in r['graphs'][name])
    pools.update(g['pool'] for g in r['graphs'][name])
    for ordinal,row in enumerate(rows):
        assert (row['round'],row['frame'])==divmod(ordinal,13)
        expected=old[row['frame']]
        assert row['byte_equal_prior'] and row['next_seed']==(1 if row['reset'] else row['frame']+1)
        if residual:
            assert signature(row['output'])==signature(expected['output'])
            assert signature(row['low_nr'])==signature(expected['low_nr']) and row['private_byte_equal_low_nr']
            dispatch=dict(expected['runtime_dispatch']);dispatch.pop('xpu_graph_replay')
            for key,count in expected['captured_dispatch'].items():
                if key!='backend':dispatch[key]=dispatch.get(key,0)+count
        else:
            assert signature(row['output'],'f2')==signature(expected['actual'],'f2')
            assert row['private_byte_equal_output'];dispatch=expected['dispatches']
        assert row['effective_dispatch']==dispatch
    assert statistics.mean(row['seconds'] for row in rows)==r['matching_mean_seconds'][name]
    temporal[name]=dict(mean_ms=statistics.mean(row['seconds'] for row in rows if not row['reset'])*1000,
        round_mean_ms=[statistics.mean(row['seconds'] for row in rows if row['round']==j and not row['reset'])*1000 for j in range(3)])
assert len(pools)==1 and len(r['progress_fallback'])==n
assert all(row['byte_equal'] and row['fused_pairs'] and row['fused_c32'] for row in r['progress_fallback'])
for name,value in r.get('body_only_diagnostic',{}).items():
    assert all(row['output_unchanged'] and row['median_seconds']==statistics.median(row['samples_seconds']) for row in value)
summary.update(full_outputs_verified=39*n,temporal_only=temporal)
summary.update(passed=True,report_sha256=sha(path),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
               unique_arrays_reread=len(verified),unique_array_bytes=sum(verified.values()),
               limitation=__doc__,complete_migration=False)
target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
