"""Authenticate wide-MLP cubic LUT experiments and reread every unique saved array.

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
if 'pair_cases' in r:
    assert r['lut_unchanged'] and r['total_primitive_comparisons']==148
    assert len(r['pair_cases'])==18 and len(r['split_cases'])==19 and len(r['body_checks'])==4
    capture=js(DREF/'experimental/branched-mlp-operands-v2/validation.json')['cases']
    split_capture=js(DREF/'experimental/fused-split-ffwd-v1/validation.json')['cases']
    pair_configs=[dict(bm=bm,warps=4,stages=stage) for stage in (1,2) for bm in (16,32)]
    split_configs=[dict(bm=bm,bn=bn,stages=1) for bm,bn in ((16,32),(16,64),(32,32),(32,64))]
    timings=[]
    for family in ('pair','split'):
        for i,row in enumerate(r[family+'_cases']):
            vals={n:signature(m) for n,m in row['arrays'].items()}
            assert row['operands_unchanged'] and all(v[1]=='<f2' for v in vals.values())
            assert [v['config'] for v in row['candidates']]==(pair_configs if family=='pair' else split_configs)
            assert all(v['byte_equal'] for v in row['candidates'])
            for v in row['candidates']:
                m=v['ttgir'];assert sha(m['path'])==m['sha256'] and Path(m['path']).stat().st_size==m['stored_bytes']
                assert 'ttig.dpas' in Path(m['path']).read_text()
            if family=='pair':
                c=row['channels'];assert c in (64,128,256)
                assert vals['features'][0]==vals['output'][0] and vals['features'][0][-1]==c
                assert row['synthetic']==(i>=12) and row['timed']
                assert row['baseline']==('original_small_fallback' if row['rows']<1024 else 'current_streaming_pair')
                if i<12:
                    assert all(vals[n]==signature(m) for n,m in capture[row['name']]['arrays'].items())
                else:
                    source=next(v for v in capture.values() if v['channels']==c)['arrays']
                    assert row['rows'] in (19,577)
                    for name in ('features','output'):
                        expected=arrays.load(source[name]).reshape(-1,c)[:row['rows']].copy()
                        assert arrays.load(row['arrays'][name]).tobytes()==expected.tobytes()
                    assert all(vals[n]==signature(source[n]) for n in ('expand','reduce','project','skip_scale'))
            else:
                old=split_capture[i]
                assert row['name']==old['name'] and row['synthetic']==old['synthetic']
                assert all(vals[n]==signature(m) for n,m in old['arrays'].items())
                assert row['timed']==(i in (0,7,8,15,16,17,18))
                assert row['channels']==512 and row['baseline']=='current_streaming_group'
            if row['timed']:
                samples=row['samples_seconds']
                assert len(samples)==5 and all(len(v)==5 and min(v)>0 for v in samples)
                assert [statistics.median(v) for v in samples]==row['median_seconds']
                best=min(range(4),key=lambda n:row['median_seconds'][n+1])
                timings.append(dict(family=family,name=row['name'],baseline_ms=row['median_seconds'][0]*1000,
                    best_ms=row['median_seconds'][best+1]*1000,speedup=row['median_seconds'][0]/row['median_seconds'][best+1]))
    for c in (64,128,256):
        rows=[v for v in r['pair_cases'] if v['channels']==c and not v['synthetic']]
        best=min(range(4),key=lambda n:statistics.mean(v['median_seconds'][n+1] for v in rows))
        assert r['selected_primitive_configs']['pairs'][str(c)]==pair_configs[best]
    for key,small in (('small',True),('large',False)):
        rows=[v for v in r['split_cases'] if v['timed'] and (v['rows']<=512)==small]
        best=min(range(4),key=lambda n:statistics.mean(v['median_seconds'][n+1] for v in rows))
        assert r['selected_primitive_configs']['split'][key]==split_configs[best]
    assert [v['name'] for v in r['body_checks']]==['pairs_only','split_only','both','both_all_rows']
    assert all(v['byte_equal'] and v['split_calls']==16 for v in r['body_checks'])
    assert len({json.dumps(v['dispatch'],sort_keys=True) for v in r['body_checks']})==1
    body=js(DREF/'experimental/fused-body-stages-v2/validation.json')
    for m in body['body_inputs'].values():signature(m)
    signature(body['expected_output'])
    summary.update(primitive_comparisons=148,real_cases=28,synthetic_cases=9,
        whole_body_configurations_verified=4,primitive_timings=timings,selection=r['selected_primitive_configs'])
elif 'full_outputs_verified' in r:
    assert r['lut_bytes_unchanged'] and r['lut_shared_before_capture']
    guards=r['lut_graph_guards']
    assert [g['kind'] for g in guards]==['buffer_replacement','in_place_version_change']
    assert all(g['rejected'] and g['history_and_seed_unchanged'] and g['no_graph_replay'] for g in guards)
    assert set(r['runs'])=={'previous','pairs','wide'}
    residual='residual' in out.name
    old=(js(DREF/'results/residual-scale-fp16_xmx-256-v1/validation.json')['runs'][:13] if residual
         else js(DREF/'results/fast-precision-864x480-v1/validation.json')['runs']['fp16_xmx'])
    n=len(r['runs']);assert n==3 and r['full_outputs_verified']==39*n
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
    assert 'frames_completed' in r
    large=r['dimension']=='1920x1080';count=390 if large else 243
    assert r['frames_completed']==len(r['frames'])==count
    assert r['independent_history'] and r['reset_reproduces_first_frame'] and r['caller_ownership_guards_passed']
    old=js(DREF/('results/long-precision-1080-fp16_xmx-v1/validation.json' if large else 'results/long-precision-480-v1/validation.json'))
    review_path=Path(r['human_review_reused_from']);review=js(review_path)
    assert review['fp16_xmx_visually_accepted'] and review['byte_identical_follow_up_outputs_may_reuse_this_visual_review']
    for i,row in enumerate(r['frames']):
        frame=old['frames'][i];expected=frame if large else frame['runs']['fp16_xmx']
        assert row['frame']==i and row['next_seed']==i+1 and row['reset']==(i==0)
        assert row['byte_equal_approved'] and row['private_byte_equal_output']
        assert row['input_rgb8_sha256']==frame['input_rgb8_sha256']
        assert signature(row['output'])==signature(expected['output'])
        assert signature(row['motion'])==signature(frame['motion'])
        assert row['effective_dispatch']==expected['effective_dispatch' if large else 'dispatches']
    assert len(r['graphs'])==2 and len({g['pool'] for g in r['graphs']})==1
    assert all(g['persistent_inputs_and_output_verified_outside_pool'] for g in r['graphs'])
    summary.update(full_frames_verified=count,complete_output_and_motion_arrays_verified=count*2,
                   approved_bytes_reused=True,user_review_sha256=sha(review_path),new_output_bytes=0)
summary.update(passed=True,report_sha256=sha(path),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
               unique_arrays_reread=len(verified),unique_array_bytes=sum(verified.values()),
               limitation=__doc__,complete_migration=False)
target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
