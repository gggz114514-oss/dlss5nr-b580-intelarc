"""Validate both NR256 histories over390 frames and render a complete review.

Previous reduced FP16 and batched branches must match complete low NR, full
FP32 composition, dispatch and private history on every frame. Full-size exact
and FP16 panels reuse existing authenticated references. Neither is a new
native4060 capture. Reduced quality is pending the user's explicit review.

To limit disk growth, save only lossless low NR plus full output hashes, and
stream complete display frames into H264. Timings exclude encoding/validation/
motion estimation/upload, and are diagnostic because IO occurs between frames.
Use the separate paired pipeline benchmark for speed comparisons.
"""
import hashlib,json,os,statistics,subprocess,sys,time,traceback,urllib.request
from contextlib import ExitStack,contextmanager
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=D/'results/batched-residual-long1080-review-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-sm89-v1')
sys.path[:0]=[str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
paired_path=D/'results/batched-branches-residual256-v1/validation.json'
assert sha(paired_path)=='e0627317e6b1abdf15b25dca116ab93b3ce7bcad38e6bfbca9f117630dcecb42'
paired=js(paired_path)
assert paired['passed'] and js(paired_path.parent.with_suffix('.log.lease.json'))['returncode']==0
assert all(sha(p)==h for p,h in paired['sources'].items())
paired_audit_path=paired_path.with_name('saved-audit-v1.json');paired_audit=js(paired_audit_path)
assert paired_audit['passed'] and paired_audit['report_sha256']==sha(paired_path)
refs={};frozen=dict(paired['sources'])
for mode in ('baseline','fp16_xmx'):
    p=D/f'results/long-precision-1080-{mode}-v1/validation.json'
    r=js(p);audit_path=p.with_name('saved-audit-v1.json');audit=js(audit_path)
    assert r['passed'] and r['frames_completed']==len(r['frames'])==390
    assert audit['passed'] and audit['report_sha256']==sha(p)
    assert all(sha(q)==h for q,h in r['sources'].items())
    refs[mode]=r;frozen[str(p)]=sha(p);frozen[str(audit_path)]=sha(audit_path)
prior=refs['fp16_xmx'];source=Path(prior['source'])
assert refs['baseline']['source']==str(source) and refs['baseline']['decode_filter']==prior['decode_filter']
ffmpeg=ROOT.parent/'xess-tools/work/d3d12-media-pipeline/deps/ffmpeg-lgpl-shared-9.0/ffmpeg-n9.0-latest-win64-lgpl-shared-9.0/bin/ffmpeg.exe'
ffprobe=ffmpeg.with_name('ffprobe.exe')
assert sha(source)==prior['sources'][str(source)] and sha(ffmpeg)==prior['sources'][str(ffmpeg)]
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from fused_branched_pairs_v2 import FusedPairs
from batched_branched_mlp_v2 import FusedBatched
from fused_split_ffwd_v2 import FusedSplit
from fused_c32_mlp_lut_v1 import FusedC32
from cubic_lut_constant_v1 import register,Constant,TABLE
from fused_vit_projection_v3 import FusedVitProjection
from fused_swin_scheduling_v1 import FusedSwin
from c32_chunk_layout_v2 import ChunkedHeadLayout
from fused_dynamic_front_v1 import FusedFront
from graph_front_v6 import GraphFront
from graph_history_warp_v2 import GraphHistoryWarp
from residual_scale_v1 import ResidualScale
from shared_model_buffers_v1 import share_identical_buffers
from serial_graph_workspace_v1 import share_before_capture
import compressed_arrays_v1 as arrays
encoder=Path(imageio_ffmpeg.get_ffmpeg_exe());font_path=Path('C:/Windows/Fonts/msyh.ttc')
for p in (Path(__file__),HERE/'Run-BatchedResidualLong1080ReviewV1.cmd',paired_path,paired_audit_path,
          source,ffmpeg,ffprobe,encoder,font_path):frozen[str(p)]=sha(p)
OUT.mkdir()
video=OUT/'comparison-full-59.94fps.mp4'
report=dict(scope=__doc__,sources=frozen,exact_gate=gate,passed=False,frames=[],complete_migration=False,
            human_review='pending',new_quality_approved=False,native_capture_for_this_sequence=False,
            dimension='1920x1080',fps='60000/1001',duration_seconds=6.5065,
            video_scope='All390 source frames. Four unscaled1920x1080 panels. Offline original-speed playback.',
            panel_order=['original','existing B580 exact','existing full-size B580 FP16','new NR256 residual FP16'],
            display_conversion='Clip RGB to[0,1], round toRGB8; H264 CRF12 BT709 limited yuv420p.',
            new_lossless_full_frame_arrays=False,paired_pipeline_timing=paired_audit['temporal_only'])
models={};providers={};adapters={};warps={};contexts={};held={}
decoder=writer=None
font=ImageFont.truetype(str(font_path),34);small=ImageFont.truetype(str(font_path),23)
labels=['原始画面（1080p）','B580 精确后端 · 已有参考',
        'B580 全尺寸 FP16 · 已有参考','本轮快速版 · NR256 + 残差重建']
first_low=None;first_full_digest=None

def save():
    p=OUT/'progress.tmp';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    p.replace(OUT/'validation.json')

def digest(value):return hashlib.sha256(value.tobytes()).hexdigest()

def read_frame():
    parts=[];remaining=1080*1920*3
    while remaining:
        part=decoder.stdout.read(remaining)
        if not part:raise EOFError('Incomplete source frame')
        parts.append(part);remaining-=len(part)
    return np.frombuffer(b''.join(parts),dtype='u1').reshape(1080,1920,3).copy()

@contextmanager
def installed(name):
    with ExitStack() as stack:
        for component in contexts[name]:stack.enter_context(component.installed())
        yield

def run(name,rgb,motion,reset):
    with installed(name):
        torch.xpu.synchronize();started=time.perf_counter()
        with use_arithmetic_backend('triton') as dispatch:
            canvas,flow=scaler.prepare(rgb,motion)
            low=models[name](canvas,flow,reset=reset)
            output=scaler.composite(rgb,canvas,low)
            torch.xpu.synchronize()
        seconds=time.perf_counter()-started
    effective=dict(dispatch);assert effective.pop('xpu_graph_replay')==1
    for key,count in adapters[name].last_entry.dispatch.items():
        if key!='backend':effective[key]=effective.get(key,0)+count
    return output,low,seconds,effective

def grid(i,pictures):
    canvas=Image.new('RGB',(3840,2304),(19,24,32));draw=ImageDraw.Draw(canvas)
    for j,pixels in enumerate(pictures):
        x=(j%2)*1920;y=(j//2)*1152
        draw.text((x+14,y+1),labels[j],font=font,fill='white')
        note=f'帧 {i:03d}/389 · {i*1001/60000:.3f}/6.507 秒 · 离线生成，原速 59.94 fps 回放'
        draw.text((x+14,y+42),note,font=small,fill=(182,194,210))
        canvas.paste(Image.fromarray(pixels),(x,y+72))
    return canvas

try:
    torch.set_num_threads(2);scaler=ResidualScale(256);report['residual_scale']=scaler.metadata()
    for name in ('previous','batched'):
        model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',
            EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
        register(model)
        if models:report['shared_constants']=share_identical_buffers(model,models['previous'])
        models[name]=model;provider=StridedMatrices();provider.select('fp16_xmx');providers[name]=provider
        adapters[name]=GraphFront(model,arithmetic=provider);warps[name]=GraphHistoryWarp(model)
        pairs=FusedPairs(model,provider) if name=='previous' else FusedBatched(model,provider,workload='small')
        contexts[name]=[provider,FusedSwin(provider),ChunkedHeadLayout(model,provider),adapters[name],pairs,
                        FusedSplit(model,provider),FusedC32(model,provider),FusedVitProjection(model,provider),
                        FusedFront(model),warps[name]]
    report['shared_transient_pool']=share_before_capture(adapters.values())
    decoder=subprocess.Popen([str(ffmpeg),'-v','error','-threads','2','-i',str(source),'-vf',prior['decode_filter'],
        '-frames:v','390','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
    warm=[read_frame(),read_frame()]
    with torch.inference_mode():
        for name in models:
            for i,pixels in enumerate(warm):
                rgb=torch.from_numpy(pixels.astype('f4')/255).to('xpu')
                motion=torch.from_numpy(arrays.load(prior['frames'][i]['motion'])).to('xpu')
                run(name,rgb,motion,i==0)
            models[name].reset()
        print('Prewarmed both independent reduced histories',flush=True)
        writer=subprocess.Popen([str(encoder),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','3840x2304',
            '-r','60000/1001','-i','-','-an','-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
            '-c:v','libx264','-threads','4','-preset','fast','-crf','12','-color_primaries','bt709','-color_trc','bt709',
            '-colorspace','bt709','-color_range','tv','-movflags','+faststart','-n',str(video)],
            stdin=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
        for i,old in enumerate(prior['frames']):
            pixels=warm[i] if i<2 else read_frame();assert digest(pixels)==old['input_rgb8_sha256']
            flow=arrays.load(old['motion']);rgb=torch.from_numpy(pixels.astype('f4')/255).to('xpu')
            motion=torch.from_numpy(flow).to('xpu');values={};rows={}
            order=['previous','batched'] if i%2==0 else ['batched','previous']
            for name in order:
                value,low,seconds,dispatch=run(name,rgb,motion,i==0)
                a=value.cpu().numpy();b=low.cpu().numpy();private=models[name]._previous.cpu().numpy()
                assert a.shape==(1080,1920,3) and a.dtype==np.dtype('f4') and np.isfinite(a).all()
                assert b.shape==(256,256,3) and b.dtype==np.dtype('f2') and np.isfinite(b).all()
                assert private.tobytes()==b.tobytes() and models[name].next_seed==i+1
                values[name]=(a,b)
                rows[name]=dict(seconds=seconds,full_raw_sha256=digest(a),low_raw_sha256=digest(b),
                                private_byte_equal_low=True,next_seed=models[name].next_seed,effective_dispatch=dispatch)
                if name not in held:held[name]=(value,a.tobytes())
                else:
                    value.zero_();low.zero_()
                    assert models[name]._previous.cpu().numpy().tobytes()==private.tobytes()
            assert all(values['previous'][n].tobytes()==values['batched'][n].tobytes() for n in (0,1)),i
            assert rows['previous']['effective_dispatch']==rows['batched']['effective_dispatch']
            for value,data in held.values():assert value.cpu().numpy().tobytes()==data
            assert rgb.cpu().numpy().tobytes()==(pixels.astype('f4')/255).tobytes()
            assert motion.cpu().numpy().tobytes()==flow.tobytes()
            a,b=values['batched'];low_meta=arrays.save(b)
            if i==0:first_low=b.tobytes();first_full_digest=digest(a)
            pictures=[pixels];reference_meta={}
            for mode in ('baseline','fp16_xmx'):
                ref=refs[mode]['frames'][i]
                assert ref['input_rgb8_sha256']==digest(pixels) and ref['motion']==old['motion']
                reference=arrays.load(ref['output']);assert reference.shape==a.shape and np.isfinite(reference).all()
                pictures.append(np.rint(np.clip(reference.astype('f4'),0,1)*255).astype('u1'))
                reference_meta[mode]=ref['output']
            pictures.append(np.rint(np.clip(a,0,1)*255).astype('u1'))
            canvas=grid(i,pictures);writer.stdin.write(np.asarray(canvas).tobytes())
            if i in (0,65,130,195,260,325,389):canvas.save(OUT/f'frame{i:03d}-full.png')
            report['frames'].append(dict(frame=i,reset=i==0,input_rgb8_sha256=digest(pixels),motion=old['motion'],
                low_nr=low_meta,full_byte_equal_previous=True,low_byte_equal_previous=True,runs=rows,
                reference_outputs=reference_meta,presentation_rgb_sha256=digest(np.asarray(canvas))))
            if i%30==0 or i==389:save();print(json.dumps(dict(frame=i,total=390,all_reduced_bytes_equal=True,rendered=True)),flush=True)
        assert decoder.stdout.read()==b''
        assert decoder.wait(timeout=30)==0,decoder.stderr.read().decode(errors='replace')
        writer.stdin.close();assert writer.wait(timeout=60)==0,writer.stderr.read().decode(errors='replace')
        rgb=torch.from_numpy(warm[0].astype('f4')/255).to('xpu')
        motion=torch.from_numpy(arrays.load(prior['frames'][0]['motion'])).to('xpu')
        for name in models:
            value,low,_,_=run(name,rgb,motion,True)
            assert low.cpu().numpy().tobytes()==first_low and digest(value.cpu().numpy())==first_full_digest
            assert models[name].next_seed==1 and len(adapters[name].entries)==2 and adapters[name].replays==393
            assert Constant(models[name]).require().cpu().numpy().tobytes()==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        report['graphs']={name:a.metadata() for name,a in adapters.items()}
    probe=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration,color_space,color_range,color_transfer,color_primaries',
        '-of','json',str(video)],creationflags=0x08000000))
    s=probe['streams'][0]
    assert (s['width'],s['height'],s['avg_frame_rate'],int(s['nb_read_frames']))==(3840,2304,'60000/1001',390)
    assert s['color_space']==s['color_transfer']==s['color_primaries']=='bt709' and s['color_range']=='tv'
    assert abs(float(s['duration'])-6.5065)<.02
    report.update(passed=True,frames_completed=390,all_reduced_outputs_byte_equal=True,independent_history=True,
                  reset_reproduces_first_frame=True,caller_ownership_guards_passed=True,held_outputs_survive_replay=True,
                  lut_bytes_unchanged=True,probe=probe,video=dict(path=str(video),sha256=sha(video),bytes=video.stat().st_size),
                  images={p.name:sha(p) for p in sorted(OUT.glob('*.png'))},
                  diagnostic_temporal_mean_ms={name:statistics.mean(f['runs'][name]['seconds'] for f in report['frames'][1:])*1000 for name in models})
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for a in warps.values():a.close()
    for a in adapters.values():a.close()
    for p in (decoder,writer):
        if p is not None and p.poll() is None:p.kill();p.wait()
    assert all(sha(p)==h for p,h in frozen.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],frames=report.get('frames_completed'),video=report.get('video'))),flush=True)
