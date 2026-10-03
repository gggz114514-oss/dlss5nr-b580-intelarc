"""Finite prewritten Luna audit. Read stored artifacts only; never run a model."""
import hashlib,json,math,statistics,sys,traceback
from datetime import datetime
from pathlib import Path

M=Path('D:/Codex-NR-Experiments/nr-b580/reference/results/int8-ffn-range-repair-v1-monitor-luna-v1/request-from-main.json')
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
m=js(M)
def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert Path(m['log']).is_file(),'Known-launched log missing'
if sys.argv[1:]==['--startup']:
    assert not Path(m['startup']).exists()
    write(m['startup'],dict(read_at=datetime.now().astimezone().isoformat(),manifest=str(M),log=m['log'],log_bytes=Path(m['log']).stat().st_size))
    print('Startup recorded');sys.exit(0)
assert not sys.argv[1:] and not Path(m['handoff']).exists()
a=dict(passed=False,phase='auditing',no_model_execution=True,no_rerun=True)
cache={}
def file(path,expected=None,size=None):
    p=Path(path).resolve();key=str(p).casefold()
    if key not in cache:
        h=hashlib.sha256();n=0
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
        cache[key]=dict(path=str(p),sha256=h.hexdigest(),bytes=n)
    v=cache[key]
    assert expected is None or v['sha256']==expected,p
    assert size is None or v['bytes']==size,p
    return v
def arrays(value):
    if isinstance(value,dict):
        if {'path','sha256','stored_bytes'}<=value.keys():file(value['path'],value['sha256'],value['stored_bytes'])
        for v in value.values():arrays(v)
    elif isinstance(value,list):
        for v in value:arrays(v)
try:
    r,l=js(m['result']),js(m['lease'])
    assert l['owner']==m['owner'] and l['returncode']==0
    assert r['passed'] and r['phase']=='completed' and not r.get('error') and not r.get('finalization_error')
    for key in ('default_unchanged','same_int8_compiled_kernels_and_resources','all48_calibration_and16_heldout_ffns_match_cpu',
                'all_frozen_face_references_reproduced','reset_reproduces_first_frame','inputs_constants_histories_held_outputs_checked'):assert r[key]
    assert r['candidate_promoted'] is False and r['new_quality_approved'] is False and r['human_review']=='pending'
    for p,h in m['critical_sources'].items():file(p,h)
    for key in ('sources','exact_gate'):
        assert r[key]
        for p,h in r[key].items():file(p,h)
    cal=r['calibration']
    assert cal['face_frames']==[48,144,242] and cal['car_samples']==[0,5,10] and cal['margin']==1.25
    assert len(cal['samples'])==6 and len(cal['layers'])==8
    assert {(s['scene'],s['frame']) for s in cal['samples']}=={(kind,i) for kind,ids in [('face',cal['face_frames']),('car',cal['car_samples'])] for i in ids}
    assert r['heldout_targets']==[96,192] and set(r['heldout_targets']).isdisjoint(cal['face_frames'])
    assert r['heldout_same_face_clip']==[i for i in range(243) if i not in cal['face_frames']]
    for layer in cal['layers']:
        assert layer['calibration_clip_fractions_after']==[0.0]*6
        assert layer['hidden_scale']['shape']==[1,4096] and layer['hidden_scale']['dtype']=='<f4'
        assert 0<=layer['changed_channels']<=4096
    for sample in cal['samples']+r['heldout_probes']:
        assert sample['eager_matches_graph'] and [v['block'] for v in sample['ffns']]==list(range(8))
        for v in sample['ffns']:assert v['cpu_matches_gpu'] and 0<=v['hidden_clip_fraction']<=1
    assert [v['frame'] for v in r['heldout_probes']]==[96,192]
    assert all(v['own_video_output_reproduced'] for v in r['heldout_probes'])
    arrays(cal);arrays(r['heldout_probes']);arrays(r['frames'])
    assert r['paired']['passed'] and set(r['paired']['runs'])=={'selected','original_int8','repaired_int8'}
    for name,rows in r['paired']['runs'].items():
        assert len(rows)==39
        for index,v in enumerate(rows):
            assert (v['round'],v['frame'])==divmod(index,13)
            assert math.isfinite(v['host_ms']) and v['host_ms']>0 and v['own_repeat_equal'] and v['private_history_matches_low']
        for mode in ('all','temporal'):
            actual=statistics.mean(v['host_ms'] for v in rows if mode=='all' or not v['reset'])
            assert math.isclose(actual,r['paired']['summary'][name][mode]['mean_host_ms'],abs_tol=1e-9)
    assert len(r['frames'])==r['frames_completed']==243 and len(r['images'])==7
    for i,row in enumerate(r['frames']):
        assert row['frame']==i and row['reset']==(i==0) and row['next_seed']==i+1
        for flag in ('frozen_references_reproduced','private_history_matches_low','inputs_unchanged','held_outputs_unchanged'):assert row[flag]
        for q in row['metrics'].values():
            for region in ('full','roi'):
                assert math.isfinite(q[region]['rgb_rmse_8bit']) and q[region]['rgb_rmse_8bit']>=0
    for subset in ('all','heldout'):
        for name in ('original_int8','repaired_int8'):
            rows=[v for v in r['frames'] if subset=='all' or not v['calibration_frame']]
            value=statistics.mean(v['metrics'][name]['roi']['rgb_rmse_8bit'] for v in rows)
            assert math.isclose(value,r['quality_summary'][subset][name]['mean_roi_rgb_rmse_8bit'],abs_tol=1e-9)
    for p,h in r['images'].items():file(p,h)
    for name,graphs in r['graphs'].items():
        assert len(graphs)==2 and sum(v['replays'] for v in graphs)==(287 if name=='repaired_int8' else 41)
        assert all(v['persistent_inputs_and_output_verified_outside_pool'] for v in graphs)
    for resources in (r['candidate']['ffn_resources'],r['candidate']['capture']['resources']):
        assert resources and all(v['spills']==0 for v in resources.values())
    v=r['video_finalization'];p=v['probe']
    assert v['passed'] and v['packet_timestamps_equal'] and v['all_decoded_frames_byte_equal'] and v['nclx']==[1,1,1]
    assert v['vui']==dict(colour_primaries=1,transfer_characteristics=1,matrix_coefficients=1,video_full_range_flag=0)
    assert (p['width'],p['height'],p['avg_frame_rate'],int(p['nb_frames']))==(2592,1352,'24/1',243)
    assert abs(float(p['duration'])-10.125)<.001 and p['color_range']=='tv'
    assert all(p[k]=='bt709' for k in ('color_space','color_transfer','color_primaries'))
    assert len(v['decoded_frames'])==243 and all(x['frame']==i and x['byte_equal'] and x['bytes']==2592*1352*3//2 for i,x in enumerate(v['decoded_frames']))
    assert r['video']==v['video'];file(r['video']['path'],r['video']['sha256'],r['video']['bytes'])
    final=json.loads(Path(m['log']).read_text(encoding='utf-8').splitlines()[-1]);assert final['passed'] and final['frames']==243
    a.update(passed=True,phase='completed',primary={k:file(m[k]) for k in ('result','log','lease')},
        unique_files_checked=len(cache),video=r['video'],quality=r['quality_summary'],timing=r['paired']['summary'],
        heldout_clip_fractions={str(v['frame']):[x['hidden_clip_fraction'] for x in v['ffns']] for v in r['heldout_probes']},
        human_review='pending',candidate_promoted=False)
except BaseException:
    a.update(passed=False,phase='failed',error=traceback.format_exc());raise
finally:write(m['handoff'],a)
print(json.dumps(dict(passed=True,handoff=m['handoff'],video=m.get('video'),human_review='pending')))
