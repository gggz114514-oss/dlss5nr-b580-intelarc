"""Authenticate bounded C32 projection/window scatter experiments and reread every unique saved array.

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
    assert len(r['cases'])==12 and len(r['candidates'])==48 and len(r['body_checks'])==4
    chunks=0
    for i,row in enumerate(r['cases']):
        vals={n:signature(meta) for n,meta in row['arrays'].items()}
        h,w,c=row['shape'];assert c==32 and h%8==w%8==0
        assert vals['x'][0]==(h,w,32) and vals['qkv_weight'][0]==(32,96) and vals['scale'][0]==(1,)
        assert vals['order'][0]==vals['inverse'][0]==(64,)
        order=arrays.load(row['arrays']['order']);inverse=arrays.load(row['arrays']['inverse'])
        assert np.array_equal(order[inverse],np.arange(64)) and np.array_equal(np.sort(order),np.arange(64))
        for n in ('q','k','v'):assert vals[n][0]==(h//8,w//8,64,32) and vals[n][1]=='<f2'
        assert row['synthetic']==(i>=10) and row['chunks']==(h*w+32767)//32768
        if i<10:chunks+=row['chunks']
        else:assert (h,w)==((512,896),(1152,1920))[i-10]
        candidates=[v for v in r['candidates'] if v['case']==i]
        assert len(candidates)==4 and {v['config']['rows'] for v in candidates}=={8,16,32,64}
        assert all(v['byte_equal'] and all(v['qkv_matches']) for v in candidates)
        for v in candidates:
            assert v['dispatch']==dict(backend='triton',dense=row['chunks'],batched=0,attention_normalize_c32=2*row['chunks'],fp8=2*row['chunks']+1)
    assert len(r['edge_checks'])==4
    for n,meta in r['edge_arrays'].items():signature(meta,allow_nonfinite=True)
    assert len(np.unique(arrays.load(r['edge_arrays']['z']).view('u2')))==65536
    for edge in r['edge_checks']:
        assert edge['all_half_bits'] and len(edge['chunks'])==6
        start=0
        for count,row in zip((19,1,37,211,256,244),edge['chunks']):
            assert row['start']==start and row['count']==count and row['complete_buffer_matches'] and row['guards_unchanged']
            start+=count
        assert start==768
    assert {v['config']['rows'] for v in r['edge_checks']}=={8,16,32,64}
    assert all(v['byte_equal'] and v['layout_calls']['c32']==10 and v['layout_calls']['c32_chunk_pack']==chunks for v in r['body_checks'])
    assert {v['config']['rows'] for v in r['body_checks']}=={8,16,32,64}
    for row in r['timings']:
        assert len(row['samples_seconds'])==5 and all(len(v)==5 for v in row['samples_seconds'])
        assert [statistics.median(v) for v in row['samples_seconds']]==row['medians_seconds']
        assert [row['medians_seconds'][0]/t for t in row['medians_seconds'][1:]]==row['speedups']
    summary.update(real_fronts_verified=10,candidates_verified=48,whole_body_configurations_verified=4,partial_scatter_checks=24,half_bit_patterns=65536)
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
