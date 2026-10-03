"""CPU-only recovery after complete face inference and MP4 color-tag failure.

The original H264 SPS has correct BT709 values, but MP4 nclx primaries/transfer
are unspecified. A separate FFmpeg9 stream-copy remux fixes container metadata.
Compare every decoded YUV byte and packet timestamp. Preserve original failed
evidence; all model checks below are audits of its stored execution evidence.
"""
import hashlib,json,re,subprocess,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
D=Path('D:/Codex-NR-Experiments/nr-b580/reference/results')
OLD=D/'int8-ffn-face480-review-v1';OUT=D/'int8-ffn-face480-finish-v1'
assert not OUT.exists()
FF=BASE/'xess-tools/work/d3d12-media-pipeline/deps/ffmpeg-lgpl-shared-9.0/ffmpeg-n9.0-latest-win64-lgpl-shared-9.0/bin/ffmpeg.exe'
FP=FF.with_name('ffprobe.exe')
oldvideo=OLD/'comparison-full-and-face-24fps.mp4';video=OUT/oldvideo.name
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pins={OLD/'validation.json':'ecfd4c8153a45482fcef21aa02b23162fc49e96fb36214be14ff1e4659a39fd7',
    D/'int8-ffn-face480-review-v1.log':'75e38252dbb8309b86666a6e25a78fa8b7bcfadaff358353041b070c760a070d',
    D/'int8-ffn-face480-review-v1.log.lease.json':'f3f3a7233a55f7ae28c43535ecb14c5fd4b90e861dcd7704e058693d1bfb3ee0',
    oldvideo:'c37d4568d4c3f756245d978ca7beee24057f14ae9f8c2635fde49f74f9fc1d04'}
for p,h in pins.items():assert sha(p)==h,p
original=js(OLD/'validation.json')
assert original['passed'] is False and original['phase']=='failed'
assert original['error'].endswith('AssertionError\n') and "probe.get('color_range')" in original['error']
assert not original.get('finalization_error') and js(D/'int8-ffn-face480-review-v1.log.lease.json')['returncode']==1
sources=dict(original['sources']);sources.update({str(p):h for p,h in pins.items()})
for p in (Path(__file__),HERE/'Run-Int8Face480FinishV1.cmd',HERE/'audit_int8_face480_finish_v1.py',FF,FP):sources[str(p)]=sha(p)
OUT.mkdir()
report=dict(scope=__doc__,passed=False,phase='authenticating_stored_evidence',sources=sources,exact_gate=original['exact_gate'],
    original_run=dict(report=str(OLD/'validation.json'),sha256=sha(OLD/'validation.json'),returncode=1,error=original['error']),
    no_model_execution=True,no_gpu_execution=True,no_reencoding=True,human_review='pending',new_quality_approved=False,
    candidate_promoted=False,original_failed_report_preserved=True,decoded_frames=[],
    original_video=dict(path=str(oldvideo),sha256=sha(oldvideo),bytes=oldvideo.stat().st_size))

def save():
    p=OUT/'progress.tmp';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');p.replace(OUT/'validation.json')

def probe(path,packets=False):
    fields='packet=pts,dts,duration' if packets else 'stream=width,height,avg_frame_rate,nb_frames,duration,time_base,pix_fmt,color_space,color_range,color_transfer,color_primaries'
    command=[str(FP),'-v','error','-select_streams','v:0']
    if packets:command+=['-show_packets']
    command+=['-show_entries',fields,'-of','json',str(path)]
    return json.loads(subprocess.check_output(command,creationflags=0x08000000,timeout=30))

def color_atom(path):
    data=Path(path).read_bytes();i=data.find(b'colrnclx')
    assert i>=4 and data.find(b'colrnclx',i+1)<0 and int.from_bytes(data[i-4:i],'big')==19
    return dict(offset=i-4,primaries=int.from_bytes(data[i+8:i+10],'big'),
        transfer=int.from_bytes(data[i+10:i+12],'big'),matrix=int.from_bytes(data[i+12:i+14],'big'),full_range=bool(data[i+14]&128))

def vui(path):
    command=[str(FF),'-hide_banner','-v','info','-i',str(path),'-map','0:v:0','-c:v','copy',
        '-bsf:v','trace_headers','-frames:v','1','-f','null','-']
    run=subprocess.run(command,capture_output=True,creationflags=0x08000000,timeout=30)
    assert run.returncode==0
    text=run.stderr.decode('utf-8',errors='replace');values={}
    for key in ('colour_primaries','transfer_characteristics','matrix_coefficients','video_full_range_flag'):
        found={int(v) for v in re.findall(r'\b'+key+r'\s+[01]+\s+=\s+(\d+)',text)}
        assert len(found)==1,(key,found);values[key]=found.pop()
    return values

processes=[];errors=[];save()
try:
    for group in (sources,original['exact_gate']):
        for p,h in group.items():assert sha(p)==h,p
    metas={}
    for i,row in enumerate(original['frames']):
        assert row['frame']==i and row['reset']==(i==0) and row['inputs_unchanged'] and row['held_outputs_unchanged']
        for name in ('selected','int8_p4'):
            run=row['runs'][name];meta=run['low']
            assert run['next_seed']==i+1 and run['private_history_matches_low']
            assert meta['shape']==[256,256,3] and meta['dtype']=='<f2'
            metas[meta['path']]=meta
        meta=row['motion'];metas[meta['path']]=meta
    assert len(original['frames'])==243 and len(original['images'])==7
    for p,meta in metas.items():assert sha(p)==meta['sha256'] and Path(p).stat().st_size==meta['stored_bytes'],p
    for p,h in original['images'].items():assert sha(p)==h,p
    assert original['scaler_identity_and_zero_motion_passed'] and original['model_input_face_crop'] is False
    assert original['face_roi']==[256,16,640,400] and original['face_enlargement']==2
    for name,graphs in original['graphs'].items():
        assert len(graphs)==2 and sum(g['replays'] for g in graphs)==246
        assert all(g['persistent_inputs_and_output_verified_outside_pool'] for g in graphs)
    assert original['candidate']['registered_packed_tensors']==40
    for resources in (original['candidate']['ffn_resources'],original['candidate']['capture']['resources'],original['selected_resources']):
        assert resources and all(v['spills']==0 for v in resources.values())
    report['stored_inference_audit']=dict(passed=True,frames=243,array_files_checked=len(metas),images=7,
        graph_replays_per_route=246,graph_metadata_reached_after_final_model_checks=True,
        scope='Frozen original inference execution evidence, including final reset/constant/held-output checks before metadata assertion; no fresh model execution')
    for key in ('frames','images','graphs','candidate','selected_resources','panel_order','face_roi','face_enlargement','source_geometry','display_geometry','scaler'):
        report[key]=original[key]
    before=probe(oldvideo)['streams'][0];oldatom=color_atom(oldvideo);oldvui=vui(oldvideo)
    assert (oldatom['primaries'],oldatom['transfer'],oldatom['matrix'],oldatom['full_range'])==(2,2,1,False)
    assert oldvui==dict(colour_primaries=1,transfer_characteristics=1,matrix_coefficients=1,video_full_range_flag=0)
    report.update(original_probe=before,diagnosis=dict(mp4_nclx=oldatom,h264_vui=oldvui,
        issue='H264 SPS already BT709 but MP4 nclx primaries and transfer unspecified'),phase='remuxing')
    command=[str(FF),'-hide_banner','-v','error','-i',str(oldvideo),'-map','0:v:0','-c:v','copy',
        '-bsf:v','h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0',
        '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv',
        '-movflags','+faststart+write_colr','-n',str(video)]
    save();result=subprocess.run(command,capture_output=True,creationflags=0x08000000,timeout=60)
    report['remux']=dict(command=command,returncode=result.returncode,stderr=result.stderr.decode(errors='replace'))
    assert result.returncode==0,report['remux']
    after=probe(video)['streams'][0];newatom=color_atom(video);newvui=vui(video)
    assert (newatom['primaries'],newatom['transfer'],newatom['matrix'],newatom['full_range'])==(1,1,1,False)
    assert newvui==oldvui
    for s in (before,after):
        assert (s['width'],s['height'],s['pix_fmt'],s['avg_frame_rate'],int(s['nb_frames']))==(2592,1352,'yuv420p','24/1',243)
        assert abs(float(s['duration'])-10.125)<.001 and s['color_space']=='bt709' and s['color_range']=='tv'
    assert all(after.get(k)=='bt709' for k in ('color_space','color_transfer','color_primaries'))
    packets=probe(oldvideo,True)['packets'];assert len(packets)==243 and packets==probe(video,True)['packets']
    assert before['time_base']==after['time_base']
    report.update(probe=after,corrected_mp4_nclx=newatom,corrected_h264_vui=newvui,all_packet_timestamps_equal=True,
        video=dict(path=str(video),sha256=sha(video),bytes=video.stat().st_size),phase='comparing_decoded_frames')
    save();print('Comparing all243 YUV frames after metadata-only remux; CPU only',flush=True)
    for label,path in (('original',oldvideo),('repaired',video)):
        err=(OUT/(label+'-decode.stderr.txt')).open('wb');errors.append(err)
        cmd=[str(FF),'-v','error','-threads','2','-i',str(path),'-map','0:v:0','-an','-sn','-dn',
            '-fps_mode','passthrough','-pix_fmt','yuv420p','-c:v','rawvideo','-threads','1','-f','rawvideo','pipe:1']
        processes.append(subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err,creationflags=0x08000000))
    size=2592*1352*3//2
    def read_frame(p):
        parts=[];n=size
        while n:
            b=p.stdout.read(n)
            if not b:raise EOFError('Incomplete decoded frame')
            parts.append(b);n-=len(b)
        return b''.join(parts)
    for i in range(243):
        a,b=[read_frame(p) for p in processes];assert a==b,('Remux changed decoded pixels',i)
        report['decoded_frames'].append(dict(frame=i,byte_equal=True,bytes=size,sha256=hashlib.sha256(a).hexdigest()))
        if i%48==0 or i==242:save();print(json.dumps(dict(decoded_frame=i,byte_equal=True)),flush=True)
    for p in processes:assert p.stdout.read(1)==b'' and p.wait(timeout=30)==0
    report.update(passed=True,phase='completed',all243_decoded_frames_byte_equal=True,decoded_bytes_compared_per_video=size*243,
        original_files_unchanged=True,frames_completed=243)
except BaseException:
    report.update(passed=False,phase='failed',error=traceback.format_exc());raise
finally:
    for p in processes:
        if p.poll() is None:p.kill();p.wait()
    for f in errors:f.close()
    try:
        for group in (sources,original['exact_gate']):
            for p,h in group.items():assert sha(p)==h,p
    except BaseException:
        report.update(passed=False,phase='failed',finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=True,frames=243,all_decoded_bytes_equal=True,video=report['video'],human_review='pending')),flush=True)
