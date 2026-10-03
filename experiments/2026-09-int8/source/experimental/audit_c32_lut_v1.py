"""Authenticate embedded cubic LUT experiments and reread every unique saved array.

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
if 'all_half_encodings' in r:
    assert len(r['cases'])==8 and len(r['body_checks'])==4
    assert r['constant_identity_and_version_guards'] and r['lut_unchanged'] and r['lut_bytes']==131072
    edge=r['all_half_encodings']
    assert edge['count']==65536 and edge['byte_equal_current_helper']
    table_path=DREF/'experimental/cubic-fp8-lut-v3.npy'
    assert sha(table_path)=='7eedd128e03576a99a21e52cf618aec3be445bad134a4dd62002ae024bbbd651'
    signature(edge['output'],allow_nonfinite=True)
    assert arrays.load(edge['output']).tobytes()==np.load(table_path,allow_pickle=False).tobytes()
    configs=[dict(bm=bm,warps=4,stages=stages) for stages in (1,2) for bm in (16,32)]
    capture=js(DREF/'experimental/c32-mlp-operands-v1/validation.json')['cases']
    timings=[]
    for i,row in enumerate(r['cases']):
        vals={n:signature(m) for n,m in row['arrays'].items()}
        assert vals['features'][0]==vals['output'][0]==tuple(row['shape']) and row['shape'][-1]==32
        assert vals['expansion'][0]==(32,128) and vals['contraction'][0]==(128,32) and vals['skip_scale'][0]==(32,)
        assert all(v[1]=='<f2' for v in vals.values()) and row['operands_unchanged']
        assert row['synthetic']==(i>=6)
        if i<6:
            assert all(vals[n]==signature(m) for n,m in capture[row['name']]['arrays'].items())
        else:
            count=19 if i==6 else 65539
            first=next(iter(capture.values()))['arrays']
            for field in ('features','output'):
                x=arrays.load(first[field]).reshape(-1,32)
                expected=np.tile(x,((count+len(x)-1)//len(x),1))[:count].copy()
                assert arrays.load(row['arrays'][field]).tobytes()==expected.tobytes()
        assert [c['config'] for c in row['candidates']]==configs
        for c in row['candidates']:
            assert c['byte_equal']
            meta=c['ttgir'];text=Path(meta['path']).read_text()
            assert sha(meta['path'])==meta['sha256'] and Path(meta['path']).stat().st_size==meta['stored_bytes']
            assert 'ttig.dpas' in text
        samples=row['samples_seconds']
        assert len(samples)==5 and all(len(s)==5 and min(s)>0 for s in samples)
        assert row['median_seconds']==[statistics.median(s) for s in samples]
        timings.append(dict(name=row['name'],direct_ms=row['median_seconds'][0]*1000,
                            selected_lut_ms=row['median_seconds'][2]*1000,
                            speedup=row['median_seconds'][0]/row['median_seconds'][2]))
    body=js(DREF/'experimental/fused-body-stages-v2/validation.json')
    for meta in body['body_inputs'].values():signature(meta)
    signature(body['expected_output'])
    assert [c['config'] for c in r['body_checks']]==configs
    assert all(c['byte_equal'] and sum(c['calls'].values())==10 for c in r['body_checks'])
    summary.update(half_bit_patterns=65536,real_cases=6,synthetic_cases=2,primitive_candidates_verified=32,
                   whole_body_configurations_verified=4,primitive_timings=timings)
elif 'full_outputs_verified' in r:
    assert r['lut_bytes_unchanged'] and r['lut_shared_before_capture']
    guards=r['lut_graph_guards']
    assert [g['kind'] for g in guards]==['buffer_replacement','in_place_version_change']
    assert all(g['rejected'] and g['history_and_seed_unchanged'] and g['no_graph_replay'] for g in guards)
    assert set(r['runs'])=={'previous','lut'}
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
