"""Paired residual pipeline screen, then 390-frame INT8 visual review.

Phase 1: 13 authenticated real-motion frames, three alternating paired rounds;
resident-input timings include downscale, full NR256, private history, residual
composition and completion. Phase 2: uninterrupted 390-frame source video with
independent selected/INT8 histories and full-resolution side-by-side H264.
Decode, flow estimation, upload, display and JIT are excluded from phase 1.
Phase 2 timings are diagnostic because decode, verification and encoding occur
between calls. New quantization is pending human review, never auto-promoted.
"""
from layout_crop_validation_env_v1 import *
import statistics,subprocess,time,traceback
OUT=D/'results/int8-ffn-residual1080-review-v1';assert not OUT.exists()
nr=receipt(D/'results/int8-ffn-nr-v1/validation.json',
    '288e23a5a846c038f59a1496d865342d563f814415bd4b936b91c87dd74afc17')
assert nr['full_nr_call_test'] and not nr['video_test']
paired=receipt(D/'results/layout-crop-residual256-v1/validation.json',
    'b0d5b7e984ca530d36459b3c20651b3989f37719aae57bcee3bb6eb64e265fe8')
prior=receipt(D/'results/batched-residual-long1080-review-v2/validation.json',
    'd968f89ab835187d050a8b6805d9a11739492b954199ad090f1ab16f1f151751')
full=receipt(D/'results/long-precision-1080-fp16_xmx-v1/validation.json',
    'ff4f46387ae73190794158dd3af48c5170a9809c82f44bc00802a7c7cf6f840c')
cpu=receipt(D/'experimental/int8-ffn-segment-cpu-v1/validation.json',
    '1f1bdd147c386245dad5701407f65f5ac663a31a36647c4aeae6e11c9b799130')
review_path=D/'results/batched-residual-long1080-review-v2/user-review-v1.json'
assert sha(review_path)=='309f8f85b756ce43354a75a56278f51666d96584826c6baa15201b9c2590fdf0'
assert js(review_path)['nr256_residual_visually_accepted']
inputs=R/'inputs/flow-full-1920x1080-v3';manifest_path=inputs/'manifest.json'
manifest=js(manifest_path)
assert sha(manifest_path)==paired['sources'][str(manifest_path)]
source=Path(full['source'])
ffmpeg=Path(next(p for p in full['sources'] if Path(p).name=='ffmpeg.exe'))
ffprobe=ffmpeg.with_name('ffprobe.exe')
assert sha(source)==full['sources'][str(source)] and sha(ffmpeg)==full['sources'][str(ffmpeg)]
for p in (Path(__file__),HERE/'Run-Int8FfnResidual1080ReviewV1.cmd',review_path,manifest_path):sources[str(p)]=sha(p)
import numpy as np
import torch,triton,cv2,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from nr256_selected_stack_v4 import Stack
from int8_ffn_nr_stack_v1 import Stack as CandidateStack
from nr_backend.execution import use_arithmetic_backend
from serial_graph_workspace_v1 import share_before_capture
from residual_scale_v1 import ResidualScale
from cubic_lut_constant_v1 import Constant,TABLE
import compressed_arrays_v1 as arrays
import int8_ffn_segment_oracle_v1 as oracle
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
encoder=Path(imageio_ffmpeg.get_ffmpeg_exe());font_path=Path('C:/Windows/Fonts/msyh.ttc')
for p in (encoder,font_path,ffprobe):
    assert sha(p)==prior['sources'][str(p)]
    sources[str(p)]=sha(p)
OUT.mkdir()
video=OUT/'comparison-full-59.94fps.mp4'
report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,phase='initialized',
    complete_migration=False,candidate_promoted=False,new_quantization=True,human_review='pending',
    new_quality_approved=False,queue_check=queue_check,source=str(source),decode_filter=full['decode_filter'],
    fps='60000/1001',duration_seconds=6.5065,panel_order=['approved_NR256_FP16','continuous_INT8_FFN_NR256'],
    video_scope='390 complete source frames; two unscaled 1920x1080 panels, original-speed offline playback',
    display_conversion='Clip SDR RGB to [0,1], round to RGB8; H264 CRF12 BT709 limited yuv420p',
    temporal_metric_scope='Current-to-previous stored DIS warp of INT8-minus-selected error; in-bounds pixels, no occlusion ground truth; diagnostic only',
    new_lossless_full_frame_arrays=False,paired={'runs':{'selected':[],'int8_p4':[]}},frames=[],images={},
    human_review_reference=dict(path=str(review_path),sha256=sha(review_path)))


def save():
    temp=OUT/'progress.tmp'
    temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(OUT/'validation.json')


raw=lambda t:t.cpu().numpy().tobytes()
digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
stacks={};held={};decoder=writer=None;save()
font=ImageFont.truetype(str(font_path),34);small=ImageFont.truetype(str(font_path),23)


def run(name,rgb,motion,reset):
    s=stacks[name];before=s.rewrite.scopes
    with s.installed(),use_arithmetic_backend('triton') as dispatch:
        torch.xpu.synchronize();start=time.perf_counter()
        canvas,flow=scaler.prepare(rgb,motion)
        low=s.model(canvas,flow,reset=reset)
        output=scaler.composite(rgb,canvas,low)
        torch.xpu.synchronize();ms=(time.perf_counter()-start)*1000
    effective=dict(dispatch);assert effective.pop('xpu_graph_replay')==1
    for key,count in s.graph.last_entry.dispatch.items():
        if key!='backend':effective[key]=effective.get(key,0)+count
    if before>=6:assert s.rewrite.scopes==before,'Unexpected body rebuild'
    s.rewrite.verify_restored()
    return output,low,ms,effective


def checked_outputs(name,output,low,seed):
    a,b=output.cpu().numpy(),low.cpu().numpy();s=stacks[name]
    assert a.shape==(1080,1920,3) and a.dtype==np.dtype('f4') and np.isfinite(a).all()
    assert b.shape==(256,256,3) and b.dtype==np.dtype('f2') and np.isfinite(b).all()
    assert raw(s.model._previous)==b.tobytes() and s.model.next_seed==seed
    return a,b


def small_metrics(actual,reference):
    difference=actual-reference
    mse=float(np.mean(np.square(difference,dtype=np.float64)))
    energy=float(np.mean(np.square(reference,dtype=np.float64)))
    return dict(rmse=float(np.sqrt(mse)),relative_rmse=float(np.sqrt(mse/max(energy,1e-30))),
        mae=float(np.mean(np.abs(difference),dtype=np.float64)),max_abs=float(np.max(np.abs(difference))))


def load_pair(index):
    spec=manifest['frames'][index];paths=[inputs/spec['file'],inputs/spec['motion_file']]
    for p in paths:
        assert sha(p)==paired['sources'][str(p)]
        sources[str(p)]=sha(p)
    pixels=np.asarray(Image.open(paths[0]).convert('RGB'),dtype='f4')/255
    flow=np.fromfile(paths[1],'<f4').reshape(1080,1920,2)
    return pixels,flow,torch.from_numpy(pixels).to('xpu'),torch.from_numpy(flow).to('xpu')


def read_frame():
    parts=[];remaining=1080*1920*3
    while remaining:
        data=decoder.stdout.read(remaining)
        if not data:raise EOFError('Incomplete source frame')
        parts.append(data);remaining-=len(data)
    return np.frombuffer(b''.join(parts),dtype='u1').reshape(1080,1920,3).copy()


def grid(index,values):
    canvas=Image.new('RGB',(3840,1152),(19,24,32));draw=ImageDraw.Draw(canvas)
    for column,(name,label) in enumerate((('selected','已通过审核：NR256 FP16'),('int8_p4','新候选：NR256 连续 INT8 FFN'))):
        x=column*1920
        draw.text((x+14,1),label,font=font,fill='white')
        draw.text((x+14,42),f'帧 {index:03d}/389 · {index*1001/60000:.3f}/6.507 秒 · 离线生成，原速回放',font=small,fill=(182,194,210))
        pixels=np.rint(np.clip(values[name][0],0,1)*255).astype('u1')
        canvas.paste(Image.fromarray(pixels),(x,72))
    return canvas


try:
    torch.set_num_threads(2);cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
    assert len(prior['frames'])==len(full['frames'])==390 and len(manifest['frames'])==13
    expected={r['frame']:r for r in paired['runs']['layout_crop'] if r['round']==0};assert len(expected)==13
    assert all(bool(spec['reset'])==expected[i]['reset'] for i,spec in enumerate(manifest['frames']))
    scales=[arrays.load(next(c for c in cpu['cases'] if c['name']==f'vit.{i}.ffn' and c['margin']==1.25)['hidden_scale']) for i in range(8)]
    scaler=ResidualScale(256);report['residual_scale']=scaler.metadata()
    stacks['selected']=Stack(EXACT)
    stacks['int8_p4']=CandidateStack(EXACT,hidden_scales=scales,share_with=stacks['selected'])
    s=stacks['int8_p4'];constants=[hashlib.sha256(raw(t)).hexdigest() for row in s.int8_vit.packed for t in row]
    scale_constants=[hashlib.sha256(raw(t)).hexdigest() for values in scaler.tables.values() for t in values]
    report['shared_constants']=s.shared
    report['shared_transient_pool']=share_before_capture([s.graph for s in stacks.values()])
    report['runtime']=dict(torch=torch.__version__,triton=triton.__version__,device=torch.xpu.get_device_name())
    sources.update(s.rewrite.fork.sources)
    dispatch_reference={};paired_reference={}
    with torch.inference_mode():
        for name in ('int8_p4','selected'):
            for index in (0,1):
                _,_,rgb,motion=load_pair(index)
                output,low,_,_=run(name,rgb,motion,index==0)
                a,b=checked_outputs(name,output,low,index+1)
                if name=='selected':
                    assert a.tobytes()==arrays.load(expected[index]['output']).tobytes()
                    assert b.tobytes()==arrays.load(expected[index]['low_nr']).tobytes()
            stacks[name].model.reset()
        report.update(phase='paired_timing');save()
        print('sampling_started: 13 real-motion frames x 3 paired residual rounds',flush=True)
        for repeat in range(3):
            for index in range(13):
                pixels,flow,rgb,motion=load_pair(index)
                reset=bool(manifest['frames'][index]['reset']);seed=expected[index]['next_seed']
                order=['selected','int8_p4'] if (repeat+index)%2==0 else ['int8_p4','selected']
                values={}
                for name in order:
                    output,low,ms,dispatch=run(name,rgb,motion,reset)
                    a,b=checked_outputs(name,output,low,seed)
                    mode='reset' if reset else 'temporal'
                    key=(name,mode)
                    if key not in dispatch_reference:dispatch_reference[key]=dispatch
                    assert dispatch==dispatch_reference[key]
                    if name=='selected':
                        assert a.tobytes()==arrays.load(expected[index]['output']).tobytes()
                        assert b.tobytes()==arrays.load(expected[index]['low_nr']).tobytes()
                    elif repeat==0:paired_reference[index]=(digest(a),arrays.save(b))
                    else:
                        assert digest(a)==paired_reference[index][0]
                        assert b.tobytes()==arrays.load(paired_reference[index][1]).tobytes()
                    values[name]=(a,b)
                    report['paired']['runs'][name].append(dict(round=repeat,frame=index,reset=reset,order=order,
                        host_ms=ms,full_raw_sha256=digest(a),low_raw_sha256=digest(b),next_seed=seed,
                        full_and_low_match_own_reference=True,private_history_matches_low=True))
                    output.zero_();low.zero_();assert raw(stacks[name].model._previous)==b.tobytes()
                assert raw(rgb)==pixels.tobytes() and raw(motion)==flow.tobytes()
                if repeat==0:
                    report['paired'].setdefault('samples',[]).append(dict(frame=index,
                        selected_output=expected[index]['output'],selected_low=expected[index]['low_nr'],
                        candidate_low=paired_reference[index][1],full_error=small_metrics(values['int8_p4'][0],values['selected'][0]),
                        low_error=oracle.error_metrics(values['int8_p4'][1],values['selected'][1])))
            save();print('Completed residual paired round '+str(repeat),flush=True)
        report['paired']['summary']={n:{mode:dict(mean_host_ms=statistics.mean(r['host_ms'] for r in rows if mode=='all' or not r['reset']),
            round_mean_host_ms=[statistics.mean(r['host_ms'] for r in rows if r['round']==k and (mode=='all' or not r['reset'])) for k in range(3)])
            for mode in ('all','temporal')} for n,rows in report['paired']['runs'].items()}
        report['paired']['change_percent']={mode:(report['paired']['summary']['int8_p4'][mode]['mean_host_ms']/report['paired']['summary']['selected'][mode]['mean_host_ms']-1)*100 for mode in ('all','temporal')}
        report['paired']['passed']=True;report.update(phase='video');save()
        print(json.dumps(dict(paired_passed=True,change_percent=report['paired']['change_percent'])),flush=True)
        for s in stacks.values():s.model.reset()
        decoder=subprocess.Popen([str(ffmpeg),'-v','error','-threads','2','-i',str(source),'-vf',full['decode_filter'],
            '-frames:v','390','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
        writer=subprocess.Popen([str(encoder),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','3840x1152',
            '-r','60000/1001','-i','-','-an','-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
            '-c:v','libx264','-threads','4','-preset','fast','-crf','12','-color_primaries','bt709','-color_trc','bt709',
            '-colorspace','bt709','-color_range','tv','-movflags','+faststart','-n',str(video)],
            stdin=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
        yy,xx=np.indices((1080,1920),dtype='f4');previous_error=None
        first_pixels=None;first_output={}
        for index,old in enumerate(prior['frames']):
            pixels=read_frame();assert digest(pixels)==old['input_rgb8_sha256']==full['frames'][index]['input_rgb8_sha256']
            assert old['motion']==full['frames'][index]['motion']
            flow=arrays.load(old['motion']);rgb_cpu=pixels.astype('f4')/255
            rgb,motion=torch.from_numpy(rgb_cpu).to('xpu'),torch.from_numpy(flow).to('xpu')
            values={};runs={};order=['selected','int8_p4'] if index%2==0 else ['int8_p4','selected']
            for name in order:
                output,low,ms,dispatch=run(name,rgb,motion,index==0)
                a,b=checked_outputs(name,output,low,index+1)
                assert dispatch==dispatch_reference[name,'reset' if index==0 else 'temporal']
                if name=='selected':
                    assert b.tobytes()==arrays.load(old['low_nr']).tobytes(),('approved low NR',index)
                    assert digest(a)==old['runs']['batched']['full_raw_sha256'],('approved full composite',index)
                values[name]=(a,b)
                runs[name]=dict(diagnostic_host_ms=ms,full_raw_sha256=digest(a),low_raw_sha256=digest(b),
                    private_history_matches_low=True,next_seed=index+1)
                if index in (0,120):held[name,index]=(output,low,a.tobytes(),b.tobytes())
                else:
                    output.zero_();low.zero_();assert raw(stacks[name].model._previous)==b.tobytes()
            for output,low,a_bytes,b_bytes in held.values():assert raw(output)==a_bytes and raw(low)==b_bytes
            assert raw(rgb)==rgb_cpu.tobytes() and raw(motion)==flow.tobytes()
            a,b=values['int8_p4'];reference=values['selected'][0]
            error=a-reference;quality=small_metrics(a,reference)
            fx,fy=xx+flow[...,0],yy+flow[...,1]
            valid=(fx>=1)&(fx<1919)&(fy>=1)&(fy<1079)
            quality['flow_valid_fraction']=float(valid.mean())
            if previous_error is not None and valid.any():
                warped=cv2.remap(previous_error,fx,fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                quality['flow_compensated_error_change_mae']=float(np.mean(np.abs(error-warped)[valid],dtype=np.float64))
            previous_error=error
            canvas=grid(index,values);writer.stdin.write(np.asarray(canvas).tobytes())
            if index in (0,65,130,195,260,325,389):
                image_path=OUT/f'frame{index:03d}-full.png';canvas.save(image_path)
                report['images'][image_path.name]=sha(image_path)
            if index==0:
                first_pixels=pixels.copy();first_output={name:(digest(v[0]),v[1].tobytes()) for name,v in values.items()}
            report['frames'].append(dict(frame=index,reset=index==0,input_rgb8_sha256=digest(pixels),motion=old['motion'],
                selected_low=old['low_nr'],candidate_low=arrays.save(b),runs=runs,
                selected_matches_approved_low_and_full=True,inputs_unchanged=True,held_outputs_unchanged=True,
                full_error=quality,low_error=oracle.error_metrics(b,values['selected'][1]),
                presentation_rgb_sha256=digest(np.asarray(canvas))))
            if index%30==0 or index==389:
                save();print(json.dumps(dict(frame=index,total=390,rendered=True,selected_matches_approved=True)),flush=True)
        assert decoder.stdout.read()==b''
        assert decoder.wait(timeout=30)==0,decoder.stderr.read().decode(errors='replace')
        writer.stdin.close()
        assert writer.wait(timeout=60)==0,writer.stderr.read().decode(errors='replace')
        rgb=torch.from_numpy(first_pixels.astype('f4')/255).to('xpu')
        motion=torch.from_numpy(arrays.load(prior['frames'][0]['motion'])).to('xpu')
        for name,s in stacks.items():
            output,low,_,_=run(name,rgb,motion,True)
            assert digest(output.cpu().numpy())==first_output[name][0] and raw(low)==first_output[name][1]
            assert s.model.next_seed==1 and len(s.graph.entries)==2 and s.graph.replays==432
            assert len(s.rewrite.builds)==s.rewrite.scopes==6
            expected_counts=(651,180,395,215) if name=='selected' else (643,172,395,223)
            assert all(tuple(build[k] for k in ('triton_calls','standalone_fp8','quantization_calls','elided_fp8'))==expected_counts for build in s.rewrite.builds)
            assert raw(Constant(s.model).require())==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
            with s.installed():s.graph._validate()
        for output,low,a_bytes,b_bytes in held.values():assert raw(output)==a_bytes and raw(low)==b_bytes
        s=stacks['int8_p4'];s.call_guard.verify_restored()
        assert constants==[hashlib.sha256(raw(t)).hexdigest() for row in s.int8_vit.packed for t in row]
        assert scale_constants==[hashlib.sha256(raw(t)).hexdigest() for values in scaler.tables.values() for t in values]
        report['candidate']=s.metadata();report['selected_resources']=stacks['selected'].rewrite.metadata()['resources']
        for resources in (report['candidate']['ffn_resources'],report['candidate']['capture']['resources'],report['selected_resources']):
            assert resources and all(v['spills']==0 for v in resources.values())
        report['graphs']={name:s.graph.metadata() for name,s in stacks.items()}
    report.update(phase='validating_video');save()
    probe=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration,color_space,color_range,color_transfer,color_primaries',
        '-of','json',str(video)],creationflags=0x08000000))
    stream=probe['streams'][0]
    assert (stream['width'],stream['height'],stream['avg_frame_rate'],int(stream['nb_read_frames']))==(3840,1152,'60000/1001',390)
    assert abs(float(stream['duration'])-6.5065)<.02
    assert stream['color_space']==stream['color_transfer']==stream['color_primaries']=='bt709' and stream['color_range']=='tv'
    report.update(passed=True,phase='completed',frames_completed=390,independent_uninterrupted_history=True,
        caller_ownership_passed=True,held_outputs_unchanged=True,reset_reproduces_first_frame=True,
        inputs_lut_int8_scale_constants_unchanged=True,selected_all390_match_approved=True,
        no_dependency_swaps_on_steady_replay=True,video=dict(path=str(video),sha256=sha(video),bytes=video.stat().st_size),probe=probe)
except BaseException:
    report.update(phase='failed',error=traceback.format_exc());raise
finally:
    try:
        for s in stacks.values():s.close()
        for process in (decoder,writer):
            if process is not None and process.poll() is None:process.kill();process.wait()
        finalize_sources()
    except BaseException:
        report.update(passed=False,phase='failed',finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=report['passed'],frames=report.get('frames_completed'),
    paired_change_percent=report['paired'].get('change_percent'),video=report.get('video'),human_review='pending')),flush=True)
