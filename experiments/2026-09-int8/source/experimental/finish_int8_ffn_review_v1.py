"""CPU-only recovery of an already completed NR inference/video run.

The original run remains failed at its missing color_transfer metadata check.
Main has remuxed a NEW copy with explicit BT709 VUI/container tags, without
re-encoding. Authenticate the frozen inference evidence, verify all metadata,
then directly compare every decoded YUV byte and all packet timestamps. No
torch/triton/model import, inference rerun, or change to original artifacts.
"""
import hashlib,json,math,statistics,subprocess,sys,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=Path('D:/Codex-NR-Experiments/nr-b580/reference/results')
OLD=D/'int8-ffn-residual1080-review-v1';OUT=D/'int8-ffn-residual1080-finish-v1'
RESULT=OUT/'validation.json';assert OUT.is_dir() and not RESULT.exists()
old_video=OLD/'comparison-full-59.94fps.mp4';video=OUT/old_video.name
ffmpeg=HERE.parent.parent/'xess-tools/work/d3d12-media-pipeline/deps/ffmpeg-lgpl-shared-9.0/ffmpeg-n9.0-latest-win64-lgpl-shared-9.0/bin/ffmpeg.exe'
ffprobe=ffmpeg.with_name('ffprobe.exe')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pinned={
    OLD/'validation.json':'b8e9e4726f18ec675a740e867bc497330bd4791c63d68a01852b59aeaba64fe4',
    D/'int8-ffn-residual1080-review-v1.log':'a0624a6ee59e97691ed36e2cdc63b20f7942c6d55c0ad12636322e7146fca277',
    D/'int8-ffn-residual1080-review-v1.log.lease.json':'f806fd72b4e1013ef2f1c713eb251852eda86d824d9fe1cacc24d8c5f29f1c55',
    old_video:'b93debf5f8ab17ea8b2d59a997b7b1d5fabce12520ffeb8dfb85bc64e39204d3',
    OUT/'remux-receipt.json':'a42916d4b2dc8e9c3063c6fdf5221c6373f06618125dbf65417c5f2e3076cd94',
}
for p,h in pinned.items():assert sha(p)==h,p
receipt=js(OUT/'remux-receipt.json')
assert receipt['exit_code']==0 and receipt['source']['sha256']==sha(old_video)
assert receipt['output']['sha256']==sha(video) and receipt['output']['bytes']==video.stat().st_size
assert receipt['tool']['sha256']==sha(ffmpeg)
original=js(OLD/'validation.json');lease=js(D/'int8-ffn-residual1080-review-v1.log.lease.json')
assert lease['returncode']==1 and original['passed'] is False and original['phase']=='failed'
assert original['error'].endswith("KeyError: 'color_transfer'\n") and not original.get('finalization_error')
sources=dict(original['sources']);sources.update({str(p):h for p,h in pinned.items()})
for p in (Path(__file__),HERE/'Run-Int8FfnReviewFinishV1.cmd',video,ffmpeg,ffprobe):
    value=sha(p)
    if str(p) in sources:assert sources[str(p)]==value
    sources[str(p)]=value
report=dict(scope=__doc__,passed=False,phase='initialized',sources=sources,exact_gate=original['exact_gate'],
    original_run=dict(path=str(OLD/'validation.json'),sha256=pinned[OLD/'validation.json'],returncode=1,
        failed_stage='video color metadata validation',error=original['error']),
    no_model_execution=True,no_gpu_execution=True,no_reencoding=True,original_files_unchanged=False,
    human_review='pending',new_quality_approved=False,candidate_promoted=False,complete_migration=False,
    video=dict(path=str(video),sha256=sha(video),bytes=video.stat().st_size),
    original_video=dict(path=str(old_video),sha256=sha(old_video),bytes=old_video.stat().st_size),
    decoded_frames=[],inference_evidence='Checks inherited from frozen execution, independently audited as stored evidence; no model reexecution')


def save():
    temp=OUT/'progress.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(RESULT)


def probe(path,packets=False):
    fields='packet=pts,dts,duration' if packets else 'stream=codec_name,width,height,pix_fmt,time_base,avg_frame_rate,nb_frames,duration,color_space,color_range,color_transfer,color_primaries'
    command=[str(ffprobe),'-v','error','-select_streams','v:0']
    if packets:command+=['-show_packets']
    command+=['-show_entries',fields,'-of','json',str(path)]
    return json.loads(subprocess.check_output(command,creationflags=0x08000000))


processes=[];error_files=[];save()
try:
    for group in (sources,original['exact_gate']):
        for p,h in group.items():assert sha(p)==h,p
    metas={}
    def collect(value):
        if isinstance(value,dict):
            if {'path','sha256','stored_bytes'}<=value.keys():
                p=value['path']
                if p in metas:assert metas[p]['sha256']==value['sha256']
                metas[p]=value
            for item in value.values():collect(item)
        elif isinstance(value,list):
            for item in value:collect(item)
    collect(original)
    for p,meta in metas.items():assert sha(p)==meta['sha256'] and Path(p).stat().st_size==meta['stored_bytes'],p
    for name,h in original['images'].items():assert sha(OLD/name)==h
    assert len(original['images'])==7 and len(original['frames'])==390
    for index,row in enumerate(original['frames']):
        assert row['frame']==index and row['reset']==(index==0)
        assert row['selected_matches_approved_low_and_full'] and row['inputs_unchanged'] and row['held_outputs_unchanged']
        for name in ('selected','int8_p4'):
            assert row['runs'][name]['next_seed']==index+1 and row['runs'][name]['private_history_matches_low']
        assert row['candidate_low']['raw_sha256']==row['runs']['int8_p4']['low_raw_sha256']
        assert row['selected_low']['raw_sha256']==row['runs']['selected']['low_raw_sha256']
    assert original['paired']['passed']
    for name,rows in original['paired']['runs'].items():
        assert len(rows)==39 and {(r['round'],r['frame']) for r in rows}=={(k,i) for k in range(3) for i in range(13)}
        assert all(r['reset']==(r['frame'] in (0,12)) and r['full_and_low_match_own_reference'] and r['private_history_matches_low'] for r in rows)
        for mode in ('all','temporal'):
            chosen=[r for r in rows if mode=='all' or not r['reset']]
            assert statistics.mean(r['host_ms'] for r in chosen)==original['paired']['summary'][name][mode]['mean_host_ms']
        for i in range(13):
            for key in ('full_raw_sha256','low_raw_sha256'):assert len({r[key] for r in rows if r['frame']==i})==1
    for name,graphs in original['graphs'].items():
        assert len(graphs)==2 and sum(g['replays'] for g in graphs)==432
        assert all(g['persistent_inputs_and_output_verified_outside_pool'] for g in graphs)
    assert original['candidate']['registered_packed_tensors']==40
    def spills(value):
        if isinstance(value,dict):
            if 'spills' in value:assert value['spills']==0
            for item in value.values():spills(item)
        elif isinstance(value,list):
            for item in value:spills(item)
    spills(original)
    report['stored_inference_audit']=dict(passed=True,frames=390,paired_rows_per_variant=39,
        graph_replays_per_variant=432,array_files_checked=len(metas),image_files_checked=7,
        graph_metadata_reached_after_final_model_checks=True)
    report['remux_receipt']=receipt
    report['paired']=original['paired'];report['graphs']=original['graphs']
    report['images']={str(OLD/name):h for name,h in original['images'].items()}
    before=probe(old_video)['streams'][0];after=probe(video)['streams'][0]
    for s in (before,after):
        assert (s['width'],s['height'],s['pix_fmt'],s['avg_frame_rate'],int(s['nb_frames']))==(3840,1152,'yuv420p','60000/1001',390)
        assert abs(float(s['duration'])-6.5065)<.02
        assert s['color_space']=='bt709' and s['color_range']=='tv'
    assert all(after.get(k)=='bt709' for k in ('color_space','color_transfer','color_primaries')),after
    old_packets=probe(old_video,True)['packets'];new_packets=probe(video,True)['packets']
    assert old_packets==new_packets and len(new_packets)==390 and before['time_base']==after['time_base']
    report.update(phase='comparing_decoded_frames',original_probe=before,probe=after,all_packet_timestamps_equal=True)
    save();print('Comparing 390 decoded YUV frames, CPU only',flush=True)
    for label,path in (('original',old_video),('repaired',video)):
        errors=(OUT/(label+'-decode.stderr.txt')).open('wb');error_files.append(errors)
        command=[str(ffmpeg),'-v','error','-threads','2','-i',str(path),'-map','0:v:0','-an','-sn','-dn',
            '-fps_mode','passthrough','-pix_fmt','yuv420p','-c:v','rawvideo','-threads','1','-f','rawvideo','pipe:1']
        processes.append(subprocess.Popen(command,stdout=subprocess.PIPE,stderr=errors,creationflags=0x08000000))
    frame_size=3840*1152*3//2
    def read_exact(process):
        parts=[];remaining=frame_size
        while remaining:
            part=process.stdout.read(remaining)
            if not part:raise EOFError('Incomplete decoded frame')
            parts.append(part);remaining-=len(part)
        return b''.join(parts)
    for index in range(390):
        a,b=(read_exact(p) for p in processes)
        assert a==b,('Decoded pixels changed after metadata-only remux',index)
        report['decoded_frames'].append(dict(frame=index,bytes=frame_size,byte_equal=True,sha256=hashlib.sha256(a).hexdigest()))
        if index%60==0 or index==389:save();print(json.dumps(dict(decoded_frame=index,byte_equal=True)),flush=True)
    for p in processes:
        assert p.stdout.read(1)==b'' and p.wait(timeout=30)==0
    report.update(passed=True,phase='completed',all390_decoded_frames_byte_equal=True,
        decoded_bytes_compared_per_video=390*frame_size,original_files_unchanged=True,
        recovery='New color-tagged stream-copy artifact and separate completion receipt; original failed report retained')
except BaseException:
    report.update(phase='failed',error=traceback.format_exc());raise
finally:
    for p in processes:
        if p.poll() is None:p.kill();p.wait()
    for f in error_files:f.close()
    try:
        for group in (sources,original['exact_gate']):
            for p,h in group.items():assert sha(p)==h,p
    except BaseException:
        report.update(passed=False,phase='failed',finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=report['passed'],decoded_frames=len(report['decoded_frames']),video=report['video'],no_model_execution=True)),flush=True)
