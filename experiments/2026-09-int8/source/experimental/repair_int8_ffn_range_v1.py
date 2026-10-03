"""Static-range repair: calibration, paired timing, full face review and held-out probes.

Calibration: face frames48/144/242 using their frozen INT8 histories, plus car
samples0/5/10 as independent resets. Use all64 tokens in every FFN, fixed1.25
margin, no fit to face96/192 or to evaluation errors. Validation frames from
these same two clips are correlated; this is not unseen-video generalization.
"""
from layout_crop_validation_env_v1 import *
import gc, statistics, subprocess, time, traceback
OUT = D/'results/int8-ffn-range-repair-v1'
assert not OUT.exists()
face = receipt(D/'results/int8-ffn-face480-finish-v1/validation.json', 'b9590f1d70607ca0e3a20869f85ee7149af08ca0a9d4c8d005138e4bb1751e24')
full = receipt(D/'results/long-precision-480-v1/validation.json', '858b4dd646c3be61263bc3fac6fabe400912686af672861638619c63be3be8a3')
cpu = receipt(D/'experimental/int8-ffn-segment-cpu-v1/validation.json', '1f1bdd147c386245dad5701407f65f5ac663a31a36647c4aeae6e11c9b799130')
paired = receipt(D/'results/layout-crop-residual256-v1/validation.json', 'b0d5b7e984ca530d36459b3c20651b3989f37719aae57bcee3bb6eb64e265fe8')
old_car = receipt(D/'results/int8-ffn-residual1080-finish-v1/validation.json', 'c5169b23fba361a581ab5e087ad7fb49bc4f21f69b3b923af9f16c7b988ef269')
import numpy as np
import torch, triton, cv2, imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from nr256_selected_stack_v4 import Stack as SelectedStack
from int8_ffn_nr_stack_v1 import Stack as OldStack
from int8_ffn_calibrated_stack_v1 import Stack as RepairStack, fit_scales
from serial_graph_workspace_v1 import share_before_capture
from nr_backend.execution import use_arithmetic_backend
from face480_residual_scale_v1 import Face480Scale
from residual_scale_v1 import ResidualScale
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
import int8_ffn_segment_oracle_v1 as oracle
from nr_review_video_finalize_v1 import read_exact, finalize

assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
FACE_CAL = [48, 144, 242]; CAR_CAL = [0, 5, 10]; TARGETS = [96, 192]
NAMES = ('selected', 'original_int8', 'repaired_int8')
car_inputs = R/'inputs/flow-full-1920x1080-v3'
manifest_path = car_inputs/'manifest.json'; manifest = js(manifest_path)
assert sha(manifest_path) == paired['sources'][str(manifest_path)]
expected = {r['frame']: r for r in paired['runs']['layout_crop'] if r['round'] == 0}
old_expected = {r['frame']: r for r in old_car['paired']['runs']['int8_p4'] if r['round'] == 0}
assert len(expected) == len(old_expected) == len(manifest['frames']) == 13
source = Path(full['source'])
ffmpeg = Path(next(p for p in full['sources'] if Path(p).name == 'ffmpeg.exe'))
encoder = Path(imageio_ffmpeg.get_ffmpeg_exe()); font_path = Path('C:/Windows/Fonts/msyh.ttc')
for p in (Path(__file__), HERE/'int8_ffn_calibrated_stack_v1.py', HERE/'nr_review_video_finalize_v1.py',
          HERE/'audit_int8_ffn_range_repair_v1.py', HERE/'Run-Int8FfnRangeRepairV1.cmd',
          manifest_path, source, ffmpeg, ffmpeg.with_name('ffprobe.exe'), encoder, font_path):
    sources[str(p)] = sha(p)
OUT.mkdir()
report = dict(scope=__doc__, passed=False, phase='initializing', sources=sources, exact_gate=gate,
    queue_check=queue_check, candidate_promoted=False, new_quality_approved=False, human_review='pending',
    default_unchanged=True, calibration=dict(face_frames=FACE_CAL, car_samples=CAR_CAL, margin=1.25, samples=[], layers=[]),
    heldout_targets=TARGETS, heldout_same_face_clip=[i for i in range(243) if i not in FACE_CAL],
    paired=dict(runs={n: [] for n in NAMES}), frames=[], images={}, heldout_probes=[],
    speed_scope='Resident 1080p prepare + NR256 + private history + residual + host completion; excludes decode, flow, upload, encode, JIT',
    roi_scope='Fixed384x384 face display region includes background; not a segmented skin metric',
    temporal_scope='DIS-warped change of error relative to frozen FP16, in-bounds pixels, no occlusion truth',
    face_reference='Frozen authenticated FP16 and original INT8 lows reconstructed with the identical residual scaler',
    display_geometry=[2592,1352], panel_order=list(NAMES), source=str(source), source_geometry=[864,480])
stacks = {}; cal = None; processes = []; files = []; held = {}
digest = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
raw = lambda t: t.cpu().numpy().tobytes()
Y = np.array([.2126,.7152,.0722], dtype='f8')


def save():
    p = OUT/'progress.tmp'
    p.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    p.replace(OUT/'validation.json')


def metrics(actual, reference, roi=False):
    if roi:
        actual, reference = actual[16:400,256:640], reference[16:400,256:640]
    d = actual.astype('f8')-reference.astype('f8'); mean = d.mean((0,1))
    return dict(rgb_rmse_8bit=float(np.sqrt(np.mean(d*d))*255), mae_8bit=float(np.abs(d).mean()*255),
                mean_delta_RGB_8bit=(mean*255).tolist(), mean_delta_Yprime_8bit=float(mean@Y*255),
                mean_delta_blue_minus_red_8bit=float((mean[2]-mean[0])*255))


def face_pixels(i):
    p = D/'results/int8-ffn-face480-review-v1'/f'frame{i:03d}-full-and-face.png'
    assert sha(p) == face['images'][str(p)]
    pixels = np.asarray(Image.open(p).convert('RGB'))[56:536,:864].copy()
    assert digest(pixels) == face['frames'][i]['input_rgb8_sha256']
    return pixels


def car_pair(i):
    spec = manifest['frames'][i]; p, f = car_inputs/spec['file'], car_inputs/spec['motion_file']
    for path in (p, f):
        assert sha(path) == paired['sources'][str(path)]
        sources[str(path)] = sha(path)
    return np.asarray(Image.open(p).convert('RGB')).astype('f4')/255, np.fromfile(f,'<f4').reshape(1080,1920,2)


def run(s, scaler, rgb, motion, reset):
    scopes, entries = s.rewrite.scopes, len(s.graph.entries)
    with s.installed(), use_arithmetic_backend('triton') as dispatch:
        torch.xpu.synchronize(); start = time.perf_counter()
        canvas, flow = scaler.prepare(rgb, motion)
        low = s.model(canvas, flow, reset=reset)
        output = scaler.composite(rgb, canvas, low)
        torch.xpu.synchronize(); elapsed = (time.perf_counter()-start)*1000
    assert dispatch.get('xpu_graph_replay') == 1
    assert len(s.graph.entries) > entries or s.rewrite.scopes == scopes
    s.rewrite.verify_restored()
    return output, low, elapsed


def captured_ffns(s, low):
    """Only eager diagnostic work; never attach CPU probes during graph capture."""
    captured = {}; scope = s.int8_vit
    assert scope.int8_probe is None
    inputs = {k: None if t is None else t.clone() for k,t in s.graph.last_entry.inputs.items()}
    before = {k: None if t is None else raw(t) for k,t in inputs.items()}
    history, hbytes, seed = s.model._previous, raw(s.model._previous), s.model.next_seed
    def probe(i, x, mlp, output, qh):
        assert i not in captured and qh is not None
        captured[i] = tuple(t.cpu().numpy().copy() for t in (x,mlp,qh))
    scope.int8_probe = probe
    try:
        with s.installed(), body.installed(), use_arithmetic_backend('triton'):
            eager = body.forward_front(s.model, **inputs, sigmoid=s.model.sigmoid,
                blend_scale=s.model.blend_scale, return_float32=False)
            torch.xpu.synchronize()
        assert raw(eager) == raw(low) and set(captured) == set(range(8))
        assert s.model._previous is history and raw(history) == hbytes and s.model.next_seed == seed
        assert before == {k: None if t is None else raw(t) for k,t in inputs.items()}
        return captured
    finally:
        scope.int8_probe = None
        s.rewrite.verify_restored()


def check_ffns(s, captured, scales, save_inputs=False):
    rows = []; hidden_values = []
    for i, (x,mlp,qh) in sorted(captured.items()):
        b = s.model.vit[i]
        we,wc,skip = [t.cpu().numpy() for t in (b.expand,b.contract,b.ffn_skip)]
        hidden,_ = oracle.expansion(x,we)
        initial = (x.astype('f4')*skip.astype('f4')).astype('f2')
        _,expected_mlp,detail = oracle.contraction(hidden,scales[i],wc,initial)
        assert expected_mlp.tobytes() == mlp.tobytes() and detail['qh'].tobytes() == qh.tobytes(), ('CPU/GPU FFN',i)
        unit = np.abs(hidden.astype('f4')/scales[i])
        row = dict(block=i,cpu_matches_gpu=True,hidden_clip_fraction=float((unit>127).mean()),
                   hidden_max_range_multiple=float(unit.max()/127),mlp_raw_sha256=digest(mlp),qh_raw_sha256=digest(qh))
        if save_inputs:
            row['input'] = arrays.save(x)
        rows.append(row); hidden_values.append(hidden)
    return rows, hidden_values


def child(label, command, **pipes):
    err = (OUT/(label+'.stderr.txt')).open('wb'); files.append(err)
    p = subprocess.Popen(command, stderr=err, creationflags=0x08000000, **pipes)
    processes.append(p); return p


save()
try:
    torch.set_num_threads(2); cv2.setNumThreads(2); cv2.ocl.setUseOpenCL(False)
    old_scales = [arrays.load(next(c for c in cpu['cases'] if c['name']==f'vit.{i}.ffn' and c['margin']==1.25)['hidden_scale']) for i in range(8)]
    face_scaler, car_scaler = Face480Scale(), ResidualScale(256)
    report['scalers'] = dict(face=face_scaler.metadata(), car=car_scaler.metadata())
    cal = OldStack(EXACT, hidden_scales=old_scales)
    maxima = [np.zeros((1,4096),dtype='f4') for _ in range(8)]
    calibration_hidden = []
    report['phase'] = 'collecting_calibration'; save()
    with torch.inference_mode():
        for scene, indices in [('face',FACE_CAL),('car',CAR_CAL)]:
            for index in indices:
                if scene == 'face':
                    pixels = face_pixels(index); rgb_cpu = pixels.astype('f4')/255
                    motion_cpu = arrays.load(face['frames'][index]['motion']); scaler = face_scaler
                    previous = arrays.load(face['frames'][index-1]['runs']['int8_p4']['low'])
                    cal.model.reset(); cal.model._previous = torch.from_numpy(previous.copy()).to('xpu'); cal.model._next_seed = index
                else:
                    rgb_cpu,motion_cpu = car_pair(index); scaler = car_scaler
                rgb,motion = torch.from_numpy(rgb_cpu).to('xpu'),torch.from_numpy(motion_cpu).to('xpu')
                out,low,_ = run(cal,scaler,rgb,motion,scene=='car')
                if scene == 'face':
                    assert digest(out.cpu().numpy()) == face['frames'][index]['runs']['int8_p4']['full_raw_sha256']
                    assert raw(low) == arrays.load(face['frames'][index]['runs']['int8_p4']['low']).tobytes()
                captured = captured_ffns(cal,low)
                rows, hidden = check_ffns(cal,captured,old_scales,save_inputs=True)
                for i,h in enumerate(hidden):
                    maxima[i] = np.maximum(maxima[i],np.abs(h.astype('f4')).max(axis=0,keepdims=True))
                calibration_hidden.append(hidden)
                report['calibration']['samples'].append(dict(scene=scene,frame=index,reset=scene=='car',
                    input_rgb_float_sha256=digest(rgb_cpu),full_raw_sha256=digest(out.cpu().numpy()),
                    reviewed_face_reproduced=scene=='face',eager_matches_graph=True,ffns=rows))
                assert raw(rgb)==rgb_cpu.tobytes() and raw(motion)==motion_cpu.tobytes()
                save(); print(json.dumps(dict(calibration_scene=scene,frame=index,ffns_checked=8)),flush=True)
    scales = fit_scales(old_scales,maxima)
    for i,(old,new,maximum) in enumerate(zip(old_scales,scales,maxima)):
        ratio = new/old
        clips = [float((np.abs(h[i].astype('f4')/new)>127).mean()) for h in calibration_hidden]
        assert max(clips) == 0
        report['calibration']['layers'].append(dict(block=i,old_scale=arrays.save(old),hidden_scale=arrays.save(new),
            observed_maximum=arrays.save(maximum),changed_channels=int(np.count_nonzero(old!=new)),
            range_ratio_percentiles=np.percentile(ratio,[0,50,90,99,100]).tolist(),calibration_clip_fractions_after=clips))
    sources.update(cal.rewrite.fork.sources)
    cal.close(); cal = None
    del captured, hidden, calibration_hidden, out, low
    gc.collect(); torch.xpu.empty_cache()
    report['phase']='constructing_repair'; save()
    stacks['selected']=SelectedStack(EXACT)
    stacks['original_int8']=OldStack(EXACT,hidden_scales=old_scales,share_with=stacks['selected'])
    stacks['repaired_int8']=RepairStack(EXACT,hidden_scales=scales,share_with=stacks['selected'])
    report['shared_transient_pool']=share_before_capture([s.graph for s in stacks.values()])
    constants={n:[digest(t.cpu().numpy()) for row in s.int8_vit.packed for t in row] for n,s in stacks.items() if n!='selected'}
    tables={n:[digest(t.cpu().numpy()) for row in sc.tables.values() for t in row] for n,sc in [('face',face_scaler),('car',car_scaler)]}
    report['runtime']=dict(torch=torch.__version__,triton=triton.__version__,device=torch.xpu.get_device_name())
    sources.update(stacks['repaired_int8'].rewrite.fork.sources)
    with torch.inference_mode():
        resident=[]
        for i in range(13):
            a,m=car_pair(i);resident.append((a,m,torch.from_numpy(a).to('xpu'),torch.from_numpy(m).to('xpu')))
        for s in stacks.values():
            for i in (0,1):
                _,_,rgb,motion=resident[i];run(s,car_scaler,rgb,motion,i==0)
            s.model.reset()
        report['phase']='paired_timing';save()
        repeats={}
        for repeat in range(3):
            for i,(a,m,rgb,motion) in enumerate(resident):
                reset=bool(manifest['frames'][i]['reset']); seed=expected[i]['next_seed']
                shift=(repeat+i)%3; order=NAMES[shift:]+NAMES[:shift]; values={}
                for name in order:
                    s=stacks[name];out,low,ms=run(s,car_scaler,rgb,motion,reset)
                    o,b=out.cpu().numpy(),low.cpu().numpy()
                    assert np.isfinite(o).all() and np.isfinite(b).all()
                    assert raw(s.model._previous)==b.tobytes() and s.model.next_seed==seed
                    pair=(digest(o),digest(b));key=(name,i)
                    if repeat==0:repeats[key]=pair
                    else:assert repeats[key]==pair
                    if name=='selected':assert pair==(expected[i]['output']['raw_sha256'],expected[i]['low_nr']['raw_sha256'])
                    if name=='original_int8':assert pair==(old_expected[i]['full_raw_sha256'],old_expected[i]['low_raw_sha256'])
                    values[name]=o.copy()
                    report['paired']['runs'][name].append(dict(round=repeat,frame=i,reset=reset,order=list(order),host_ms=ms,
                        full_raw_sha256=pair[0],low_raw_sha256=pair[1],next_seed=seed,own_repeat_equal=True,private_history_matches_low=True))
                    out.zero_();low.zero_();assert raw(s.model._previous)==b.tobytes()
                if repeat==0:
                    report['paired'].setdefault('quality',[]).append(dict(frame=i,calibration_frame=i in CAR_CAL,
                        original=metrics(values['original_int8'],values['selected']),repaired=metrics(values['repaired_int8'],values['selected'])))
                assert raw(rgb)==a.tobytes() and raw(motion)==m.tobytes()
            save();print(json.dumps(dict(paired_round=repeat,completed=True)),flush=True)
        report['paired']['summary']={n:{mode:dict(mean_host_ms=statistics.mean(r['host_ms'] for r in rows if mode=='all' or not r['reset']),
            round_mean_host_ms=[statistics.mean(r['host_ms'] for r in rows if r['round']==k and (mode=='all' or not r['reset'])) for k in range(3)])
            for mode in ('all','temporal')} for n,rows in report['paired']['runs'].items()}
        report['paired']['passed']=True
        for s in stacks.values():s.model.reset()
        del resident
        report['phase']='face_video';save()
        decoder=child('decode',[str(ffmpeg),'-v','error','-threads','2','-i',str(source),'-vf',full['decode_filter'],
            '-frames:v','243','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
        encoded=OUT/'encoded-before-finalization.mp4'
        writer=child('encode',[str(encoder),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','2592x1352','-r','24','-i','-',
            '-an','-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
            '-c:v','libx264','-threads','4','-preset','fast','-crf','12',
            '-x264-params','colorprim=bt709:transfer=bt709:colormatrix=bt709:fullrange=off',
            '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv',
            '-movflags','+faststart+write_colr','-n',str(encoded)],stdin=subprocess.PIPE)
        font=ImageFont.truetype(str(font_path),25);small=ImageFont.truetype(str(font_path),18)
        yy,xx=np.indices((480,864),dtype='f4');previous_errors={};first=None
        for i in range(243):
            pixels=np.frombuffer(read_exact(decoder.stdout,480*864*3),dtype='u1').reshape(480,864,3).copy()
            assert digest(pixels)==full['frames'][i]['input_rgb8_sha256']==face['frames'][i]['input_rgb8_sha256']
            a=pixels.astype('f4')/255;m=arrays.load(face['frames'][i]['motion'])
            rgb,motion=torch.from_numpy(a).to('xpu'),torch.from_numpy(m).to('xpu')
            canvas,_=face_scaler.prepare(rgb,motion); values={}
            for name,old_name in [('selected','selected'),('original_int8','int8_p4')]:
                b=arrays.load(face['frames'][i]['runs'][old_name]['low'])
                value=face_scaler.composite(rgb,canvas,torch.from_numpy(b).to('xpu')).cpu().numpy()
                assert digest(value)==face['frames'][i]['runs'][old_name]['full_raw_sha256']
                values[name]=value
            s=stacks['repaired_int8'];out,low,_=run(s,face_scaler,rgb,motion,i==0)
            value,b=out.cpu().numpy(),low.cpu().numpy()
            assert value.shape==(480,864,3) and value.dtype==np.float32 and np.isfinite(value).all()
            assert b.shape==(256,256,3) and b.dtype==np.float16 and np.isfinite(b).all()
            assert raw(s.model._previous)==b.tobytes() and s.model.next_seed==i+1
            values['repaired_int8']=value
            row=dict(frame=i,reset=i==0,calibration_frame=i in FACE_CAL,input_rgb8_sha256=digest(pixels),motion=face['frames'][i]['motion'],
                frozen_references_reproduced=True,low=arrays.save(b),full_raw_sha256=digest(value),next_seed=i+1,
                private_history_matches_low=True,metrics={})
            fx,fy=xx+m[...,0],yy+m[...,1];valid=(fx>=1)&(fx<863)&(fy>=1)&(fy<479)
            for name in ('original_int8','repaired_int8'):
                error=values[name]-values['selected']
                q=dict(full=metrics(values[name],values['selected']),roi=metrics(values[name],values['selected'],True))
                if i:
                    warped=cv2.remap(previous_errors[name],fx,fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                    diff=error-warped
                    q['temporal_error_mae_8bit']=float(np.abs(diff[valid]).mean(dtype='f8')*255)
                previous_errors[name]=error.copy();row['metrics'][name]=q
            if i==0:first=(pixels.copy(),digest(value),b.tobytes())
            if i in (0,120):held[i]=(out,low,value.tobytes(),b.tobytes())
            else:out.zero_();low.zero_();assert raw(s.model._previous)==b.tobytes()
            for ot,lt,ob,lb in held.values():assert raw(ot)==ob and raw(lt)==lb
            assert raw(rgb)==a.tobytes() and raw(motion)==m.tobytes()
            row.update(inputs_unchanged=True,held_outputs_unchanged=True)
            grid=Image.new('RGB',(2592,1352),(19,24,32));draw=ImageDraw.Draw(grid)
            for j,(name,label) in enumerate(zip(NAMES,['NR256 FP16 参考','原 INT8','修复候选：多帧范围校准'])):
                x=j*864;im=Image.fromarray(np.rint(np.clip(values[name],0,1)*255).astype('u1'))
                draw.text((x+10,0),label,font=font,fill='white')
                draw.text((x+10,32),f'帧 {i:03d}/242 · {i/24:.3f}/10.125 秒 · 离线生成，原速回放',font=small,fill=(182,194,210))
                grid.paste(im,(x,56));draw.text((x+10,544),'同一区域放大 2 倍 · 模型处理完整帧',font=small,fill=(182,194,210))
                grid.paste(im.crop((256,16,640,400)).resize((768,768),Image.Resampling.NEAREST),(x+48,584))
            writer.stdin.write(np.asarray(grid).tobytes())
            if i in (0,48,72,96,144,192,242):
                p=OUT/f'frame{i:03d}-full-and-face.png';grid.save(p);report['images'][str(p)]=sha(p)
            report['frames'].append(row)
            if i%24==0 or i==242:save();print(json.dumps(dict(face_frame=i,total=243)),flush=True)
        assert decoder.stdout.read(1)==b'' and decoder.wait(timeout=30)==0
        writer.stdin.close();assert writer.wait(timeout=60)==0
        rgb=torch.from_numpy(first[0].astype('f4')/255).to('xpu');motion=torch.from_numpy(arrays.load(face['frames'][0]['motion'])).to('xpu')
        out,low,_=run(stacks['repaired_int8'],face_scaler,rgb,motion,True)
        assert digest(out.cpu().numpy())==first[1] and raw(low)==first[2]
        for name,s in stacks.items():
            assert len(s.graph.entries)==2 and s.graph.replays==(285 if name=='repaired_int8' else 41)
            assert len(s.rewrite.builds)==s.rewrite.scopes==6
            with s.installed():s.graph._validate()
        report['capture_before_diagnostics']={n:s.rewrite.metadata() for n,s in stacks.items()}
        assert stacks['original_int8'].int8_vit.ffn_resources==stacks['repaired_int8'].int8_vit.ffn_resources
        report['same_int8_compiled_kernels_and_resources']=True
        report['phase']='heldout_probes';save()
        for index in TARGETS:
            s=stacks['repaired_int8'];previous=arrays.load(report['frames'][index-1]['low'])
            s.model.reset();s.model._previous=torch.from_numpy(previous.copy()).to('xpu');s.model._next_seed=index
            a=face_pixels(index).astype('f4')/255;m=arrays.load(face['frames'][index]['motion'])
            out,low,_=run(s,face_scaler,torch.from_numpy(a).to('xpu'),torch.from_numpy(m).to('xpu'),False)
            assert digest(out.cpu().numpy())==report['frames'][index]['full_raw_sha256']
            assert raw(low)==arrays.load(report['frames'][index]['low']).tobytes()
            captured=captured_ffns(s,low);rows,_=check_ffns(s,captured,scales,save_inputs=True)
            report['heldout_probes'].append(dict(frame=index,own_video_output_reproduced=True,eager_matches_graph=True,ffns=rows))
            save();print(json.dumps(dict(heldout_probe=index,ffns_checked=8)),flush=True)
        for name,s in stacks.items():
            if name!='selected':assert constants[name]==[digest(t.cpu().numpy()) for row in s.int8_vit.packed for t in row]
            with s.installed():s.graph._validate()
        for n,sc in [('face',face_scaler),('car',car_scaler)]:
            assert tables[n]==[digest(t.cpu().numpy()) for row in sc.tables.values() for t in row]
        for ot,lt,ob,lb in held.values():assert raw(ot)==ob and raw(lt)==lb
        report['candidate']=stacks['repaired_int8'].metadata()
        for resources in (report['candidate']['ffn_resources'],report['candidate']['capture']['resources']):
            assert resources and all(v['spills']==0 for v in resources.values())
        report['graphs']={n:s.graph.metadata() for n,s in stacks.items()}
    report['quality_summary']={subset:{name:dict(
        mean_roi_rgb_rmse_8bit=statistics.mean(r['metrics'][name]['roi']['rgb_rmse_8bit'] for r in report['frames'] if subset=='all' or not r['calibration_frame']),
        mean_full_rgb_rmse_8bit=statistics.mean(r['metrics'][name]['full']['rgb_rmse_8bit'] for r in report['frames'] if subset=='all' or not r['calibration_frame']),
        mean_temporal_error_mae_8bit=statistics.mean(r['metrics'][name]['temporal_error_mae_8bit'] for r in report['frames'] if r['frame'] and (subset=='all' or not r['calibration_frame'])))
        for name in ('original_int8','repaired_int8')} for subset in ('all','heldout')}
    report['phase']='video_finalization';save()
    print('Finalizing MP4 metadata and checking all243 decoded frames; no model rerun',flush=True)
    report['video_finalization']=finalize(ffmpeg,encoded,OUT/'comparison-full-and-face-24fps.mp4',2592,1352,243,'24/1',10.125)
    report['video']=report['video_finalization']['video']
    report.update(passed=True,phase='completed',frames_completed=243,all48_calibration_and16_heldout_ffns_match_cpu=True,
        all_frozen_face_references_reproduced=True,reset_reproduces_first_frame=True,inputs_constants_histories_held_outputs_checked=True)
except BaseException:
    report.update(passed=False,phase='failed',error=traceback.format_exc());raise
finally:
    try:
        if cal is not None:cal.close()
        for s in stacks.values():s.close()
        for p in processes:
            if p.poll() is None:p.kill();p.wait()
        for f in files:f.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False,phase='failed',finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=True,frames=243,video=report['video'],quality=report['quality_summary'],human_review='pending')),flush=True)
