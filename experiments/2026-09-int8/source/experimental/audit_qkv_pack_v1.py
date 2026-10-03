"""Authenticate QKV normalization/FP8/window fusion experiments and reread every unique saved array.

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
if 'candidates' in r:
    count=r['real_module_count'];assert count>0
    assert len(r['cases'])==count+2 and len(r['candidates'])==(count+2)*4 and len(r['body_checks'])==4
    for i,row in enumerate(r['cases']):
        nonfinite=row['nonfinite_test'];assert nonfinite==(i==count)
        arrays_by_name={n:signature(m,allow_nonfinite=nonfinite) for n,m in row['arrays'].items()}
        h,w,heads,three,channels=row['shape'];assert three==3 and channels==32 and h%8==w%8==0
        assert arrays_by_name['z'][0]==tuple(row['shape']) and arrays_by_name['scale'][0]==(heads,)
        assert arrays_by_name['order'][0]==(64,) and arrays_by_name['order'][1]=='<i8'
        order=arrays.load(row['arrays']['order']);assert np.array_equal(np.sort(order),np.arange(64))
        for n in ('q','k','v'):assert arrays_by_name[n][0]==(heads,h//8,w//8,64,32) and arrays_by_name[n][1]=='<f2'
        assert row['synthetic']==(i>=count)
        if nonfinite:assert len(np.unique(arrays.load(row['arrays']['z']).view('u2')))==65536
        candidates=[c for c in r['candidates'] if c['case']==i]
        assert {c['config']['rows'] for c in candidates}=={8,16,32,64}
        assert len(candidates)==4 and all(c['byte_equal'] and all(c['qkv_matches']) for c in candidates)
    assert all(row['byte_equal'] and row['layout_calls']['qkv_pack']==count for row in r['body_checks'])
    assert {row['config']['rows'] for row in r['body_checks']}=={8,16,32,64}
    for row in r['timings']:
        assert len(row['samples_seconds'])==5 and all(len(v)==5 for v in row['samples_seconds'])
        assert [statistics.median(s) for s in row['samples_seconds']]==row['medians_seconds']
        assert [row['medians_seconds'][0]/t for t in row['medians_seconds'][1:]]==row['speedups']
    summary.update(real_modules_verified=count,candidates_verified=(count+2)*4,whole_body_configurations_verified=4,half_bit_patterns=65536)
elif 'full_outputs_verified' in r:
    residual='residual' in out.name
    old=(js(DREF/'results/residual-scale-fp16_xmx-256-v1/validation.json')['runs'][:13] if residual
         else js(DREF/'results/fast-precision-864x480-v1/validation.json')['runs']['fp16_xmx'])
    n=len(r['runs']);assert n in (2,4) and r['full_outputs_verified']==39*n
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
