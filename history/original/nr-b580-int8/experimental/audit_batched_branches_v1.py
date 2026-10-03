"""Authenticate batched branch experiments and reread every unique saved array.

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
if 'source_frame' in r:
    assert r['source_frame']==181 and len(r['cases'])==36
    assert r['entire_body_byte_equal'] and r['all_modules_captured'] and r['inputs_and_lut_unchanged']
    assert len({v['name'] for v in r['cases']})==36
    assert r['raw_bytes']==50147328 and r['raw_bytes']<=r['maximum_raw_bytes']
    for row in r['cases']:
        vals={n:signature(m) for n,m in row['arrays'].items()}
        assert all(v[1]=='<f2' for v in vals.values())
        assert vals['features'][0]==vals['output'][0]==tuple(row['shape'])
    for meta in r['body_inputs'].values():signature(meta)
    signature(r['expected_output'])
    summary.update(real_modules_captured=36,entire_body_byte_equal=True)
elif 'primitive_comparisons' in r:
    assert r['lut_unchanged'] and r['primitive_comparisons']==324 and len(r['cases'])==54
    small=js(DREF/'experimental/small-branched-mlp-operands-v1/validation.json')
    wide=js(DREF/'experimental/wide-mlp-lut-v1/validation.json')
    configs=[dict(pair_bm=bm,pair_stages=stage,project_bm=16,project_bn=bn)
             for bm,stage,bn in ((16,1,32),(16,2,32),(16,1,64),(16,2,64),(32,1,64),(32,2,64))]
    shapes=set()
    for i,row in enumerate(r['cases']):
        src=small['cases'][i] if i<36 else wide['pair_cases'][i-36]
        assert row['scope']==('small' if i<36 else 'full') and row['name']==src['name']
        assert row['synthetic']==(i>=48) and row['operands_unchanged']
        vals={n:signature(m) for n,m in row['arrays'].items()}
        assert all(vals[n]==signature(m) for n,m in src['arrays'].items())
        shape=(row['rows'],row['channels'])
        assert row['timed']==(i>=36 or shape not in shapes)
        shapes.add(shape)
        assert [v['config'] for v in row['candidates']]==configs
        for v in row['candidates']:
            assert v['byte_equal'] and v['dispatch']==row['baseline_dispatch'] and len(v['kernels'])==2
            for k in v['kernels']:
                m=k['ttgir'];assert sha(m['path'])==m['sha256']
                assert Path(m['path']).stat().st_size==m['stored_bytes']
                assert 'ttig.dpas' in Path(m['path']).read_text()
        if row['timed']:
            samples=row['samples_seconds']
            assert len(samples)==7 and all(len(s)==7 and min(s)>0 for s in samples)
            assert row['median_seconds']==[statistics.median(s) for s in samples]
            assert row['orders']==[list(range(7))[j:]+list(range(7))[:j] for j in range(7)]
    for scope in ('small','full'):
        for c in (64,128,256):
            rows=[v for v in r['cases'] if v['scope']==scope and v['channels']==c and v['timed'] and not v['synthetic']]
            best=min(range(6),key=lambda n:statistics.mean(v['median_seconds'][n+1] for v in rows))
            assert r['selected_configs'][scope][str(c)]==configs[best]
    assert [v['selection'] for v in r['body_checks']]==['small','full']
    for v in r['body_checks']:
        assert v['byte_equal'] and sum(v['calls'].values())==36 and v['dispatch']==small['dispatch']
    for meta in small['body_inputs'].values():signature(meta)
    signature(small['expected_output'])
    summary.update(primitive_comparisons=324,real_cases=48,synthetic_cases=6,
                   whole_body_configurations_verified=2,selection=r['selected_configs'])
elif 'full_outputs_verified' in r:

    assert r['lut_bytes_unchanged'] and r['lut_shared_before_capture']
    guards=r['lut_graph_guards']
    assert [g['kind'] for g in guards]==['buffer_replacement','in_place_version_change']
    assert all(g['rejected'] and g['history_and_seed_unchanged'] and g['no_graph_replay'] for g in guards)
    assert set(r['runs'])=={'previous','batched'}
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
else:
    raise AssertionError("Unsupported report type")
summary.update(passed=True,report_sha256=sha(path),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
               unique_arrays_reread=len(verified),unique_array_bytes=sum(verified.values()),
               limitation=__doc__,complete_migration=False)
target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
