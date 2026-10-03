"""Entire user 480p clip, independent exact/FP16/cached-W8A8 histories, 24fps review.

The reference for this NEW long stream is the existing B580 exact backend, NOT
a new NVIDIA capture. All complete half RGB/history and half motion bytes are
retained with lossless compression. Encoded video is only a visual presentation.
"""
import hashlib,json,os,statistics,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
EXACT=ROOT.parent/'nr-b580'
R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=DREF/'results/long-precision-480-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path.insert(0,str(R))
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha
gate=authenticate_main()
cache_report_path=DREF/'results/static-cache-full-480-v2/validation.json'
cache_report=js(cache_report_path)
assert cache_report['passed'] and cache_report['all_byte_equal_prior']
assert all(sha(Path(p))==h for p,h in cache_report['sources'].items())
source=R/'inputs/visual-qa-01/clip480.mp4'
assert sha(source)=='e0d0776bb01dce7e64dab8c1b365dccaf4aba97bf6620297d73e5718f5dec107'
ffmpeg=ROOT.parent/'xess-tools/work/d3d12-media-pipeline/deps/ffmpeg-lgpl-shared-9.0/ffmpeg-n9.0-latest-win64-lgpl-shared-9.0/bin/ffmpeg.exe'
ffprobe=ffmpeg.with_name('ffprobe.exe')
probe=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames,duration','-of','json',str(source)],creationflags=0x08000000))
stream=probe['streams'][0]
assert (stream['width'],stream['height'],stream['avg_frame_rate'],int(stream['nb_frames']))==(864,480,'24/1',243)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from skimage.metrics import structural_similarity
import imageio_ffmpeg
import torch
sys.path.insert(0,str(ROOT/'backend'))
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from static_weight_cache_v2 import CachedFastMatrices
import compressed_arrays_v1 as arrays
for path in (ROOT/'backend/nr_backend').glob('*.py'):assert path.read_bytes()==(EXACT/'backend/nr_backend'/path.name).read_bytes()
cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(2)
encoder=Path(imageio_ffmpeg.get_ffmpeg_exe())
paths=[Path(__file__),HERE/'Run-LongPrecisionV1.cmd',HERE/'static_weight_cache_v2.py',HERE/'static_weight_cache_v1.py',HERE/'fast_matrices_v3.py',HERE/'compressed_arrays_v1.py',HERE/'immutable_artifacts_v1.py',cache_report_path,source,ffmpeg,ffprobe,encoder,*sorted((ROOT/'backend/nr_backend').glob('*.py'))]
frozen={str(p):sha(p) for p in paths}
OUT.mkdir(parents=True)
(OUT/'probe.json').write_text(json.dumps(probe,indent=2)+'\n',encoding='utf-8')
review=OUT/'review';review.mkdir()
modes=('baseline','fp16_xmx','int8_cached')
report=dict(scope=__doc__,sources=frozen,exact_gate=gate,source=str(source),fps=24,expected_frames=243,duration_seconds=243/24,dimension='864x480',native_capture_for_this_sequence=False,reference='B580 exact35 backend with its OWN previous output; default controls, SDR, zero depth',history='Independent state per mode, seed0/reset only at first frame, then 242 consecutive frames; final reset reproduction checked separately',flow='OpenCV DIS MEDIUM current-to-previous displacement; OpenCL disabled, two CPU threads; prequantized FP16 at model input; no engine motion/depth',frames=[],passed=False,complete_migration=False,promoted=False,human_review='pending_long_clip',timing_scope='Diagnostic synchronized model calls, uploads/flow/IO excluded. Video encoder may run on CPU concurrently. Use the separate prewarmed cache benchmark for performance claims.',decode_filter='scale=in_color_matrix=smpte170m:in_range=tv:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp,format=rgb24')
processes=[];models={};experiment=CachedFastMatrices()
first_hash={};previous_error={};previous_gray=None
yy,xx=np.indices((480,864),dtype='f4')
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)

def save_report():
    temp=OUT/'progress.tmp'
    temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(OUT/'validation.json')

def read_frame(pipe):
    parts=[];remaining=864*480*3
    while remaining:
        part=pipe.read(remaining)
        if not part:raise EOFError('Incomplete decoded frame')
        parts.append(part);remaining-=len(part)
    return np.frombuffer(b''.join(parts),dtype='u1').reshape(480,864,3).copy()

def flow_for(rgb,previous):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    flow=np.zeros((480,864,2),dtype='f4') if previous is None else cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray,previous,None)
    assert np.isfinite(flow).all() and np.abs(flow).max()<=65504
    return gray,flow.astype('f2')

def grid(frame,original,outputs):
    labels=['原始画面','B580 精确后端（本片对照）','B580 FP16 XMX','B580 INT8（静态权重缓存）']
    pictures=[original]+[np.rint(np.clip(outputs[mode].astype('f4'),0,1)*255).astype('u1') for mode in modes]
    canvas=Image.new('RGB',(1728,1056),(19,24,32));draw=ImageDraw.Draw(canvas)
    for j,(label,picture) in enumerate(zip(labels,pictures)):
        x=(j%2)*864;y=(j//2)*528
        draw.text((x+9,y+1),label,font=font,fill='white')
        draw.text((x+9,y+27),f'原速回放 24 fps  |  {frame/24:.2f} / 10.125 秒  |  帧 {frame:03d}/242',font=small,fill=(182,194,210))
        canvas.paste(Image.fromarray(picture),(x,y+48))
    return canvas

def quality_summary(mode):
    rows=[frame['runs'][mode] for frame in report['frames']]
    mse=statistics.mean(row['mse'] for row in rows)
    finite_psnr=[row['psnr_db'] for row in rows if row['psnr_db'] is not None]
    return dict(aggregate_psnr_db=None if mse==0 else float(-10*np.log10(mse)),worst_psnr_db=min(finite_psnr) if finite_psnr else None,identical_frames=sum(row['mse']==0 for row in rows),mean_ssim_sampled=statistics.mean(row['ssim'] for row in rows if 'ssim' in row))

try:
    # Verify the flow convention on a known translation before processing media.
    rng=np.random.default_rng(90611)
    a=cv2.GaussianBlur(rng.integers(0,256,(192,256),dtype='u1'),(5,5),.8)
    b=cv2.warpAffine(a,np.array([[1,0,5],[0,1,-3]],dtype='f4'),(256,192),borderMode=cv2.BORDER_REFLECT_101)
    test=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(b,a,None)
    error=float(np.median(np.linalg.norm(test[32:-32,32:-32]-np.array([-5,3]),axis=2)))
    assert error<.5
    report['flow_direction_endpoint_error']=error
    decoder=subprocess.Popen([str(ffmpeg),'-v','error','-i',str(source),'-vf',report['decode_filter'],'-frames:v','243','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
    processes.append(decoder)
    warm_pixels=[read_frame(decoder.stdout),read_frame(decoder.stdout)]
    warm_gray,warm_flow0=flow_for(warm_pixels[0],None)
    _,warm_flow1=flow_for(warm_pixels[1],warm_gray)
    for mode in modes:
        models[mode]=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    with torch.inference_mode(),experiment.installed():
        experiment.static_cache.prepack(models['int8_cached'])
        report['cache_entries']=len(experiment.static_cache.entries)
        report['cache_bytes']=experiment.static_cache.packed_bytes
        for mode in modes:
            experiment.select(mode)
            for j,flow in enumerate((warm_flow0,warm_flow1)):
                rgb=torch.from_numpy(warm_pixels[j].astype('f4')/255).to('xpu');motion=torch.from_numpy(flow).to('xpu')
                with use_arithmetic_backend('triton'):models[mode](rgb,motion,reset=j==0)
                torch.xpu.synchronize()
            models[mode].reset()
        video_path=review/'comparison-full-24fps.mp4'
        writer=subprocess.Popen([str(encoder),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1728x1056','-r','24','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','12','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(video_path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
        processes.append(writer)
        for i in range(243):
            pixels=warm_pixels[i] if i<2 else read_frame(decoder.stdout)
            gray,flow=flow_for(pixels,previous_gray);previous_gray=gray
            frame=dict(frame=i,source_frame=i,source_seconds=i/24,input_rgb8_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),motion=arrays.save(flow),reset=i==0,runs={})
            outputs={}
            rgb=torch.from_numpy(pixels.astype('f4')/255).to('xpu');motion=torch.from_numpy(flow).to('xpu')
            offset=i%3;order=modes[offset:]+modes[:offset]
            frame['execution_order']=list(order)
            for mode in order:
                model=models[mode];experiment.select(mode)
                torch.xpu.synchronize();started=time.perf_counter()
                with use_arithmetic_backend('triton') as dispatch:
                    result=model(rgb,motion,reset=i==0)
                    torch.xpu.synchronize()
                seconds=time.perf_counter()-started
                actual=result.cpu().numpy()
                private=model._previous.cpu().numpy()
                assert actual.dtype==np.dtype('f2') and private.tobytes()==actual.tobytes() and np.isfinite(actual).all()
                outputs[mode]=actual.copy()
                meta=arrays.save(actual)
                row=dict(seconds=seconds,output=meta,private=meta,rgb32f_sha256=hashlib.sha256(actual.astype('f4').tobytes()).hexdigest(),next_seed=model.next_seed,matrix_calls=experiment.calls.copy(),dispatches=dict(dispatch))
                assert model.next_seed==i+1
                if mode=='int8_cached':assert experiment.calls.get('packed_weight_hit')==experiment.calls['int8_dense'] and not experiment.calls.get('packed_weight_miss')
                if i==0:first_hash[mode]=meta['raw_sha256']
                result.zero_();assert model._previous.cpu().numpy().tobytes()==private.tobytes()
                frame['runs'][mode]=row
            assert rgb.cpu().numpy().tobytes()==(pixels.astype('f4')/255).tobytes() and motion.cpu().numpy().tobytes()==flow.tobytes()
            expected=outputs['baseline'].astype('f4')
            fx=xx+flow[...,0].astype('f4');fy=yy+flow[...,1].astype('f4')
            valid=(fx>=1)&(fx<863)&(fy>=1)&(fy<479)
            frame['flow_valid_fraction']=float(valid.mean())
            for mode in ('fp16_xmx','int8_cached'):
                difference=outputs[mode].astype('f4')-expected
                mse=float(np.mean(difference.astype('f8')**2))
                row=frame['runs'][mode]
                row.update(mse=mse,psnr_db=None if mse==0 else float(-10*np.log10(mse)),mae=float(np.mean(np.abs(difference))),max_abs=float(np.abs(difference).max()))
                if i%24==0 or i==242:row['ssim']=float(structural_similarity(expected,outputs[mode].astype('f4'),data_range=1,channel_axis=2))
                if mode in previous_error and valid.any():
                    warped=cv2.remap(previous_error[mode],fx,fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                    row['flow_compensated_error_change_mae']=float(np.abs(difference-warped)[valid].mean())
                previous_error[mode]=difference.copy()
            canvas=grid(i,pixels,outputs)
            writer.stdin.write(np.asarray(canvas).tobytes())
            if i in (0,48,96,144,192,242):canvas.save(review/f'comparison-frame{i:03d}.png')
            report['frames'].append(frame)
            if i%12==0 or i==242:
                save_report()
                print(json.dumps(dict(frame=i,total=243,psnr={mode:frame['runs'][mode]['psnr_db'] for mode in ('fp16_xmx','int8_cached')},cache_hits=frame['runs']['int8_cached']['matrix_calls'].get('packed_weight_hit'))),flush=True)
        assert decoder.stdout.read()==b''
        assert decoder.wait(timeout=30)==0,decoder.stderr.read().decode(errors='replace')
        writer.stdin.close()
        assert writer.wait(timeout=60)==0,writer.stderr.read().decode(errors='replace')
        for mode,model in models.items():
            experiment.select(mode)
            rgb=torch.from_numpy(warm_pixels[0].astype('f4')/255).to('xpu');motion=torch.from_numpy(warm_flow0).to('xpu')
            with use_arithmetic_backend('triton'):value=model(rgb,motion,reset=True)
            actual=value.cpu().numpy()
            assert hashlib.sha256(actual.tobytes()).hexdigest()==first_hash[mode] and model.next_seed==1
        report.update(passed=True,frames_completed=243,independent_history=True,reset_reproduces_first_frame=True,caller_ownership_guards_passed=True,unique_input_images=len({f['input_rgb8_sha256'] for f in report['frames']}),video_sha256=sha(video_path),diagnostic_mean_model_seconds={mode:statistics.mean(f['runs'][mode]['seconds'] for f in report['frames']) for mode in modes},quality={mode:quality_summary(mode) for mode in ('fp16_xmx','int8_cached')})
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    for process in processes:
        if process.poll() is None:process.kill();process.wait()
    assert all(sha(Path(p))==h for p,h in frozen.items())
    authenticate_main();save_report()
print(json.dumps({k:report[k] for k in ('passed','frames_completed','quality')}),flush=True)
