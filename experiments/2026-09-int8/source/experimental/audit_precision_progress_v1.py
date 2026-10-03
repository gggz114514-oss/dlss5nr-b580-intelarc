"""Independent saved-data checks for cache timings and the complete long clip."""
import argparse,hashlib,io,json,math,statistics,subprocess,zlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--scope',choices=['cache','long'],required=True);args=p.parse_args()
HERE=Path(__file__).resolve().parent
EXACT=HERE.parent.parent/'nr-b580'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=DREF/('results/static-cache-full-480-v2' if args.scope=='cache' else 'results/long-precision-480-v1')
path=OUT/'validation.json'
report=json.loads(path.read_text(encoding='utf-8'))
assert report['passed'] and not report['complete_migration']
assert not (OUT/'saved-audit-v1.json').exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,digest in report['sources'].items():assert sha(name)==digest
verified={}

def read(meta):
    target=Path(meta['path'])
    assert target.is_relative_to(DREF/'experimental/immutable-artifacts-v1')
    data=target.read_bytes()
    assert hashlib.sha256(data).hexdigest()==meta['sha256'] and len(data)==meta['stored_bytes']
    if meta.get('format')=='npy+zlib':
        data=zlib.decompress(data)
        assert hashlib.sha256(data).hexdigest()==meta['npy_sha256']
    value=np.load(io.BytesIO(data),allow_pickle=False)
    assert not value.dtype.hasobject and list(value.shape)==meta['shape'] and value.dtype.str==meta['dtype']
    assert value.nbytes==meta['raw_bytes'] and hashlib.sha256(value.tobytes()).hexdigest()==meta['raw_sha256']
    verified[str(target)]=meta['stored_bytes']
    return value

audit=dict(scope=args.scope,report_sha256=sha(path),auditor_sha256=sha(Path(__file__)),passed=False,complete_migration=False)
if args.scope=='cache':
    prior=json.loads(Path(report['prior_report']).read_text(encoding='utf-8'))
    assert len(report['warmup'])==8
    means={}
    for mode,rows in report['runs'].items():
        assert len(rows)==26
        for index,row in enumerate(rows):
            i=index%13
            assert row['round']==index//13 and row['frame']==i
            actual=read(row['actual']);private=read(row['private'])
            old_mode='int8_dense' if mode=='int8_cached' else mode
            expected=read(prior['runs'][old_mode][i]['actual'])
            assert actual.tobytes()==expected.tobytes()==private.tobytes()
            assert row['rgb_sha256']==hashlib.sha256(actual.astype('f4').tobytes()).hexdigest()
            assert row['reset']==(i in (0,12)) and row['next_seed']==(1 if row['reset'] else i+1)
            if mode=='int8_cached':assert row['cache_counts']=={'hit':1262}
        means[mode]=statistics.mean(row['seconds'] for row in rows)
        assert means[mode]==report['matching_mean_seconds'][mode]
    assert report['cache_speedup']==means['int8_dense']/means['int8_cached']
    audit.update(passed=True,full_calls_verified=104,all_output_and_history_bytes_equal_prior=True,matching_mean_seconds=means,cache_speedup=report['cache_speedup'],cache_time_reduction=1-means['int8_cached']/means['int8_dense'])
else:
    import cv2
    cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
    assert len(report['frames'])==243 and report['frames_completed']==243 and report['fps']==24
    assert not report['native_capture_for_this_sequence']
    ffmpeg=EXACT.parent/'xess-tools/work/d3d12-media-pipeline/deps/ffmpeg-lgpl-shared-9.0/ffmpeg-n9.0-latest-win64-lgpl-shared-9.0/bin/ffmpeg.exe'
    process=subprocess.Popen([str(ffmpeg),'-v','error','-i',report['source'],'-vf',report['decode_filter'],'-frames:v','243','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
    previous=None;previous_error={};yy,xx=np.indices((480,864),dtype='f4')
    def input_frame():
        data=bytearray()
        while len(data)<864*480*3:
            part=process.stdout.read(864*480*3-len(data))
            if not part:raise EOFError('Input sequence truncated during independent reread')
            data.extend(part)
        return np.frombuffer(data,dtype='u1').reshape(480,864,3)
    try:
        for i,frame in enumerate(report['frames']):
            assert frame['frame']==frame['source_frame']==i and frame['source_seconds']==i/24 and frame['reset']==(i==0)
            pixels=input_frame()
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==frame['input_rgb8_sha256']
            gray=cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
            recomputed=np.zeros((480,864,2),dtype='f2') if previous is None else cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray,previous,None).astype('f2')
            previous=gray.copy()
            motion=read(frame['motion'])
            assert motion.tobytes()==recomputed.tobytes()
            fx=xx+motion[...,0].astype('f4');fy=yy+motion[...,1].astype('f4')
            valid=(fx>=1)&(fx<863)&(fy>=1)&(fy<479)
            assert float(valid.mean())==frame['flow_valid_fraction']
            outputs={}
            for mode,row in frame['runs'].items():
                value=read(row['output'])
                assert value.shape==(480,864,3) and value.dtype==np.dtype('f2') and np.isfinite(value).all()
                assert row['private']==row['output']
                assert hashlib.sha256(value.astype('f4').tobytes()).hexdigest()==row['rgb32f_sha256']
                assert row['next_seed']==i+1
                outputs[mode]=value.astype('f4')
            for mode in ('fp16_xmx','int8_cached'):
                row=frame['runs'][mode]
                delta=outputs[mode]-outputs['baseline']
                mse=float(np.mean(delta.astype('f8')**2))
                assert abs(mse-row['mse'])<1e-15
                if mse:assert abs(-10*math.log10(mse)-row['psnr_db'])<1e-9
                else:assert row['psnr_db'] is None
                if mode in previous_error and valid.any():
                    warped=cv2.remap(previous_error[mode],fx,fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                    metric=float(np.abs(delta-warped)[valid].mean())
                    assert abs(metric-row['flow_compensated_error_change_mae'])<1e-9
                previous_error[mode]=delta.copy()
            if i%48==0:print(f'Reread+flow+metrics: {i}/243',flush=True)
        assert process.stdout.read()==b'' and process.wait(timeout=30)==0
    finally:
        if process.poll() is None:process.kill();process.wait()
    video=OUT/'review/comparison-full-24fps.mp4'
    assert sha(video)==report['video_sha256']
    probe=json.loads(subprocess.check_output([str(ffmpeg.with_name('ffprobe.exe')),'-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(video)],creationflags=0x08000000))['streams'][0]
    assert (probe['width'],probe['height'],probe['avg_frame_rate'],int(probe['nb_read_frames']))==(1728,1056,'24/1',243)
    assert abs(float(probe['duration'])-10.125)<.001
    audit.update(passed=True,full_outputs_verified=729,source_frames_redecoded=243,flows_recomputed_and_byte_equal=243,metrics_recomputed=True,video_probe=probe,native_capture_for_this_sequence=False,reference='B580 exact35',human_review='pending_long_clip',quality=report['quality'])
audit.update(unique_arrays_verified=len(verified),unique_stored_bytes=sum(verified.values()))
(OUT/'saved-audit-v1.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2),flush=True)
