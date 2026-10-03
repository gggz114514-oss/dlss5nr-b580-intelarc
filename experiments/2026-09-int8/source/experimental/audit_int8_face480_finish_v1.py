"""Prescribed one-shot Luna audit of the CPU-only face-video recovery."""
import hashlib,json,sys,traceback
from datetime import datetime
from pathlib import Path

M=Path('D:/Codex-NR-Experiments/nr-b580/reference/results/int8-ffn-face480-finish-v1-monitor-luna-v1/request-from-main.json')
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
m=js(M);log=Path(m['log']);assert log.is_file(),'Known-launched log missing'
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+chr(10),encoding='utf-8')
if sys.argv[1:]==['--startup']:
    assert not Path(m['startup']).exists()
    write(m['startup'],dict(read_at=datetime.now().astimezone().isoformat(),manifest=str(M),actual_log=str(log),log_bytes=log.stat().st_size))
    print('Startup recorded');sys.exit(0)
assert not sys.argv[1:] and not Path(m['handoff']).exists()
a=dict(passed=False,phase='auditing',human_review='pending',no_model_execution=True,no_video_execution=True)
cache={}
def file(path,expected=None,size=None):
    p=Path(path).resolve();key=str(p).casefold()
    if key not in cache:
        h=hashlib.sha256();n=0
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
        cache[key]=dict(path=str(p),sha256=h.hexdigest(),bytes=n)
    v=cache[key];assert expected is None or v['sha256']==expected,p
    assert size is None or v['bytes']==size,p
    return v
try:
    r=js(m['result']);l=js(m['lease'])
    assert l['owner']==m['owner'] and l['returncode']==0
    assert r['passed'] and r['phase']=='completed' and not r.get('error') and not r.get('finalization_error')
    for p,h in m['critical_sources'].items():file(p,h)
    for key in ('sources','exact_gate'):
        assert r[key]
        for p,h in r[key].items():file(p,h)
    for row in r['frames']:
        for meta in [row['motion']]+[v['low'] for v in row['runs'].values()]:file(meta['path'],meta['sha256'],meta['stored_bytes'])
    for p,h in r['images'].items():file(p,h)
    assert len(r['frames'])==len(r['decoded_frames'])==r['frames_completed']==243 and len(r['images'])==7
    assert all(row['frame']==i and row['byte_equal'] and row['bytes']==2592*1352*3//2 for i,row in enumerate(r['decoded_frames']))
    for key in ('all243_decoded_frames_byte_equal','all_packet_timestamps_equal','original_files_unchanged','no_model_execution','no_gpu_execution','no_reencoding','original_failed_report_preserved'):assert r[key]
    assert r['original_run']['returncode']==1 and js(r['original_run']['report'])['passed'] is False
    assert r['human_review']=='pending' and r['new_quality_approved'] is False and r['candidate_promoted'] is False
    atom=r['corrected_mp4_nclx'];assert (atom['primaries'],atom['transfer'],atom['matrix'],atom['full_range'])==(1,1,1,False)
    assert r['corrected_h264_vui']==dict(colour_primaries=1,transfer_characteristics=1,matrix_coefficients=1,video_full_range_flag=0)
    p=r['probe'];assert (p['width'],p['height'],p['avg_frame_rate'],int(p['nb_frames']))==(2592,1352,'24/1',243)
    assert all(p.get(k)=='bt709' for k in ('color_space','color_transfer','color_primaries')) and p['color_range']=='tv'
    v=r['video'];assert Path(v['path']).samefile(m['video']);a['video']=file(v['path'],v['sha256'],v['bytes'])
    final=json.loads(log.read_text(encoding='utf-8').splitlines()[-1])
    assert final['passed'] and final['frames']==243 and final['video']==v
    a.update(passed=True,phase='completed',primary={k:file(m[k]) for k in ('result','log','lease')},
        frames=243,source_counts={k:len(r[k]) for k in ('sources','exact_gate')},unique_files_checked=len(cache),
        all_decoded_bytes_equal=True,probe=p,images=r['images'],original_failed_report_preserved=True,
        scope='Completed stored inference evidence and CPU remux/decode verification; no new model or decode run by auditor')
except BaseException:a.update(passed=False,phase='failed',error=traceback.format_exc());raise
finally:write(m['handoff'],a)
print(json.dumps(dict(passed=True,handoff=m['handoff'],frames=243,human_review='pending')))
