"""Full 243-frame face review: source / selected NR256 FP16 / continuous INT8.

Preserve complete-frame inputs, original 24 fps and independent NR histories.
The fixed face crop is taken from finished outputs and enlarged 2x with nearest
sampling for inspection. No face processing, sharpening or input ROI shortcut.
This is a quality review, not a benchmark or new RTX4060 reference capture.
"""
from layout_crop_validation_env_v1 import *
import subprocess,traceback
OUT=D/'results/int8-ffn-face480-review-v1';assert not OUT.exists()
base=receipt(D/'results/long-precision-480-v1/validation.json','858b4dd646c3be61263bc3fac6fabe400912686af672861638619c63be3be8a3')
nr=receipt(D/'results/int8-ffn-nr-v1/validation.json','288e23a5a846c038f59a1496d865342d563f814415bd4b936b91c87dd74afc17')
cpu=receipt(D/'experimental/int8-ffn-segment-cpu-v1/validation.json','1f1bdd147c386245dad5701407f65f5ac663a31a36647c4aeae6e11c9b799130')
import numpy as np
import torch,triton,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from nr256_selected_stack_v4 import Stack
from int8_ffn_nr_stack_v1 import Stack as CandidateStack
from serial_graph_workspace_v1 import share_before_capture
from nr_backend.execution import use_arithmetic_backend
from face480_residual_scale_v1 import Face480Scale
from cubic_lut_constant_v1 import Constant,TABLE
import compressed_arrays_v1 as arrays
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
source=Path(base['source']);ffmpeg=Path(next(p for p in base['sources'] if Path(p).name=='ffmpeg.exe'))
ffprobe=ffmpeg.with_name('ffprobe.exe');encoder=Path(imageio_ffmpeg.get_ffmpeg_exe());font_path=Path('C:/Windows/Fonts/msyh.ttc')
for p in (Path(__file__),HERE/'Run-Int8Face480ReviewV1.cmd',HERE/'face480_residual_scale_v1.py',
          HERE/'audit_int8_face480_v1.py',source,ffmpeg,ffprobe,encoder,font_path):sources[str(p)]=sha(p)
assert sha(source)=='e0d0776bb01dce7e64dab8c1b365dccaf4aba97bf6620297d73e5718f5dec107'
assert len(base['frames'])==243
OUT.mkdir();video=OUT/'comparison-full-and-face-24fps.mp4'
W,H,N=864,480,243
ROI=(256,16,640,400);VW,VH=2592,1352
report=dict(scope=__doc__,passed=False,phase='initialized',sources=sources,exact_gate=gate,
    source=str(source),decode_filter=base['decode_filter'],fps='24/1',duration_seconds=10.125,
    source_geometry=[W,H],display_geometry=[VW,VH],face_roi=list(ROI),face_enlargement=2,
    face_sampling='nearest; identical fixed region on all columns, after full-frame inference',
    panel_order=['source','selected_NR256_FP16','continuous_INT8_FFN_NR256'],
    reference='Current selected FP16 NR256 residual route on this source; not previously approved NR256 face output or a new4060 capture',
    flow='Reused authenticated complete-frame current-to-previous DIS from prior 480p run; no new flow estimation',
    human_review='pending',new_quality_approved=False,candidate_promoted=False,complete_migration=False,
    performance_benchmark=False,queue_check=queue_check,frames=[],images={},model_input_face_crop=False)

def save():
    t=OUT/'progress.tmp';t.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(OUT/'validation.json')

raw=lambda t:t.cpu().numpy().tobytes()
digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
stacks={};held={};processes=[];error_files=[];save()
font=ImageFont.truetype(str(font_path),25);small=ImageFont.truetype(str(font_path),18)

def read_frame(pipe):
    data=bytearray()
    while len(data)<W*H*3:
        p=pipe.read(W*H*3-len(data))
        if not p:raise EOFError('Incomplete source frame')
        data.extend(p)
    return np.frombuffer(data,dtype='u1').reshape(H,W,3).copy()

def run(name,rgb,motion,reset):
    s=stacks[name];before=s.rewrite.scopes
    with s.installed(),use_arithmetic_backend('triton') as dispatch:
        canvas,flow=scaler.prepare(rgb,motion)
        low=s.model(canvas,flow,reset=reset);output=scaler.composite(rgb,canvas,low)
        torch.xpu.synchronize()
    assert dispatch.get('xpu_graph_replay')==1
    if before>=6:assert s.rewrite.scopes==before
    s.rewrite.verify_restored()
    return output,low

def present(index,pixels,values):
    pictures=[pixels]+[np.rint(np.clip(values[n][0],0,1)*255).astype('u1') for n in ('selected','int8_p4')]
    labels=['原视频','当前路线：NR256 FP16','新候选：连续 INT8 FFN']
    canvas=Image.new('RGB',(VW,VH),(19,24,32));draw=ImageDraw.Draw(canvas)
    for j,(picture,label) in enumerate(zip(pictures,labels)):
        x=j*W;im=Image.fromarray(picture)
        draw.text((x+10,0),label,font=font,fill='white')
        draw.text((x+10,32),f'{index/24:.3f}/10.125 秒 · 帧 {index:03d}/242 · 离线生成，原速回放',font=small,fill=(182,194,210))
        canvas.paste(im,(x,56))
        draw.text((x+10,544),'同一区域放大 2 倍 · 仅放大输出，不裁模型输入',font=small,fill=(182,194,210))
        canvas.paste(im.crop(ROI).resize((768,768),Image.Resampling.NEAREST),(x+48,584))
    return canvas

try:
    torch.set_num_threads(2)
    scales=[arrays.load(next(c for c in cpu['cases'] if c['name']==f'vit.{i}.ffn' and c['margin']==1.25)['hidden_scale']) for i in range(8)]
    scaler=Face480Scale();report['scaler']=scaler.metadata()
    stacks['selected']=Stack(EXACT)
    stacks['int8_p4']=CandidateStack(EXACT,hidden_scales=scales,share_with=stacks['selected'])
    report['shared_transient_pool']=share_before_capture([s.graph for s in stacks.values()])
    report['runtime']=dict(torch=torch.__version__,triton=triton.__version__,device=torch.xpu.get_device_name())
    sources.update(stacks['int8_p4'].rewrite.fork.sources)
    constants=[hashlib.sha256(raw(t)).hexdigest() for row in stacks['int8_p4'].int8_vit.packed for t in row]
    tables=[hashlib.sha256(raw(t)).hexdigest() for values in scaler.tables.values() for t in values]
    def child(label,command,**pipes):
        err=(OUT/(label+'.stderr.txt')).open('wb');error_files.append(err)
        p=subprocess.Popen(command,stderr=err,creationflags=0x08000000,**pipes);processes.append(p);return p
    decoder=child('decode',[str(ffmpeg),'-v','error','-threads','2','-i',str(source),'-vf',base['decode_filter'],
        '-frames:v',str(N),'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
    warm=[read_frame(decoder.stdout) for _ in range(2)]
    with torch.inference_mode():
        for name in ('int8_p4','selected'):
            for i,pixels in enumerate(warm):
                assert digest(pixels)==base['frames'][i]['input_rgb8_sha256']
                rgb=torch.from_numpy(pixels.astype('f4')/255).to('xpu')
                motion=torch.from_numpy(arrays.load(base['frames'][i]['motion'])).to('xpu')
                out,low=run(name,rgb,motion,i==0)
                assert torch.isfinite(out).all() and torch.isfinite(low).all()
            stacks[name].model.reset()
        # Verify the geometry adapter's identity residual and zero-motion rules.
        zero=torch.zeros((H,W,2),dtype=torch.float16,device='xpu')
        canvas,zero_low=scaler.prepare(rgb,zero)
        identity=scaler.composite(rgb,canvas,canvas)
        assert raw(identity)==raw(rgb) and torch.count_nonzero(zero_low)==0
        report['scaler_identity_and_zero_motion_passed']=True
        writer=child('encode',[str(encoder),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{VW}x{VH}',
            '-r','24','-i','-','-an','-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
            '-c:v','libx264','-threads','4','-preset','fast','-crf','12','-x264-params','colorprim=bt709:transfer=bt709:colormatrix=bt709:fullrange=off',
            '-bsf:v','h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0',
            '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv',
            '-movflags','+faststart+write_colr','-n',str(video)],stdin=subprocess.PIPE)
        report['phase']='rendering';save();first={}
        for i,old in enumerate(base['frames']):
            pixels=warm[i] if i<2 else read_frame(decoder.stdout)
            assert digest(pixels)==old['input_rgb8_sha256']
            flow=arrays.load(old['motion']);rgb_cpu=pixels.astype('f4')/255
            rgb,motion=torch.from_numpy(rgb_cpu).to('xpu'),torch.from_numpy(flow).to('xpu')
            values={};runs={}
            for name in (('selected','int8_p4') if i%2==0 else ('int8_p4','selected')):
                output,low=run(name,rgb,motion,i==0)
                a,b=output.cpu().numpy(),low.cpu().numpy();s=stacks[name]
                assert a.shape==(H,W,3) and a.dtype==np.float32 and np.isfinite(a).all()
                assert b.shape==(256,256,3) and b.dtype==np.float16 and np.isfinite(b).all()
                assert raw(s.model._previous)==b.tobytes() and s.model.next_seed==i+1
                values[name]=(a,b)
                runs[name]=dict(full_raw_sha256=digest(a),low=arrays.save(b),next_seed=i+1,private_history_matches_low=True)
                if i==0:first[name]=(digest(a),b.tobytes())
                if i in (0,120):held[name,i]=(output,low,a.tobytes(),b.tobytes())
                else:
                    output.zero_();low.zero_();assert raw(s.model._previous)==b.tobytes()
            for output,low,a_bytes,b_bytes in held.values():assert raw(output)==a_bytes and raw(low)==b_bytes
            assert raw(rgb)==rgb_cpu.tobytes() and raw(motion)==flow.tobytes()
            difference=values['int8_p4'][0]-values['selected'][0]
            x0,y0,x1,y1=ROI;face=difference[y0:y1,x0:x1]
            grid=present(i,pixels,values);writer.stdin.write(np.asarray(grid).tobytes())
            if i in (0,48,72,96,144,192,242):
                p=OUT/f'frame{i:03d}-full-and-face.png';grid.save(p);report['images'][str(p)]=sha(p)
            report['frames'].append(dict(frame=i,reset=i==0,input_rgb8_sha256=digest(pixels),motion=old['motion'],runs=runs,
                inputs_unchanged=True,held_outputs_unchanged=True,presentation_rgb_sha256=digest(np.asarray(grid)),
                full_mae=float(np.abs(difference).mean(dtype='f8')),face_roi_mae=float(np.abs(face).mean(dtype='f8'))))
            if i%24==0 or i==N-1:save();print(json.dumps(dict(frame=i,total=N,rendered=True)),flush=True)
        assert decoder.stdout.read()==b'' and decoder.wait(timeout=30)==0
        writer.stdin.close();assert writer.wait(timeout=60)==0
        rgb=torch.from_numpy(warm[0].astype('f4')/255).to('xpu');motion=torch.from_numpy(arrays.load(base['frames'][0]['motion'])).to('xpu')
        for name,s in stacks.items():
            output,low=run(name,rgb,motion,True)
            assert digest(output.cpu().numpy())==first[name][0] and raw(low)==first[name][1]
            assert s.model.next_seed==1 and len(s.graph.entries)==2 and s.graph.replays==246
            assert len(s.rewrite.builds)==s.rewrite.scopes==6
            counts=(651,180,395,215) if name=='selected' else (643,172,395,223)
            assert all(tuple(b[k] for k in ('triton_calls','standalone_fp8','quantization_calls','elided_fp8'))==counts for b in s.rewrite.builds)
            assert raw(Constant(s.model).require())==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
            with s.installed():s.graph._validate()
        assert constants==[hashlib.sha256(raw(t)).hexdigest() for row in stacks['int8_p4'].int8_vit.packed for t in row]
        assert tables==[hashlib.sha256(raw(t)).hexdigest() for values in scaler.tables.values() for t in values]
        for output,low,a_bytes,b_bytes in held.values():assert raw(output)==a_bytes and raw(low)==b_bytes
        report['candidate']=stacks['int8_p4'].metadata();report['selected_resources']=stacks['selected'].rewrite.metadata()['resources']
        for resources in (report['candidate']['ffn_resources'],report['candidate']['capture']['resources'],report['selected_resources']):
            assert resources and all(v['spills']==0 for v in resources.values())
        report['graphs']={n:s.graph.metadata() for n,s in stacks.items()}
    report['phase']='validating_video';save()
    probe=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration,color_space,color_range,color_transfer,color_primaries',
        '-of','json',str(video)],creationflags=0x08000000))['streams'][0]
    assert (probe['width'],probe['height'],probe['avg_frame_rate'],int(probe['nb_read_frames']))==(VW,VH,'24/1',N)
    assert abs(float(probe['duration'])-10.125)<.001
    assert all(probe.get(k)=='bt709' for k in ('color_space','color_transfer','color_primaries')) and probe.get('color_range')=='tv'
    report.update(passed=True,phase='completed',frames_completed=N,independent_histories=True,reset_reproduces_first_frame=True,
        inputs_constants_held_outputs_unchanged=True,probe=probe,video=dict(path=str(video),sha256=sha(video),bytes=video.stat().st_size))
except BaseException:
    report.update(passed=False,phase='failed',error=traceback.format_exc());raise
finally:
    try:
        for s in stacks.values():s.close()
        for p in processes:
            if p.poll() is None:p.kill();p.wait()
        for f in error_files:f.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False,phase='failed',finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=report['passed'],frames=report.get('frames_completed'),video=report.get('video'),human_review='pending')),flush=True)
