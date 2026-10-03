"""390 uninterrupted 1080p frames using combined layout/crop NR256 residual reconstruction.

Every low NR byte and every complete composite SHA256 is checked against the
visually accepted sequence. No new video or full-frame dumps. Decode and disk
IO make these timings diagnostic only; use paired complete-call benchmarks.
"""
from layout_crop_validation_env_v1 import *
import subprocess
import time
import traceback
OUT = D / 'results/layout-crop-long1080-v1'
assert not OUT.exists()
suite_path=D/'results/layout-crop-validation-suite-v1/validation.json'
suite=js(suite_path)
assert suite['scope']=='serial_layout_crop_validation' and len(suite['phases'])==1
phase=suite['phases'][0]
assert phase['name']=='residual' and phase['returncode']==0 and phase['passed']
paired_path=D/'results/layout-crop-residual256-v1/validation.json'
assert Path(phase['result']['path'])==paired_path
paired=receipt(paired_path,phase['result']['sha256'])
assert paired['full_outputs_verified']==78
# The suite manifest remains live until both children finish; pin its immutable
# completed phase into this report, never hash the still-changing parent file.
paired_phase=dict(phase)
prior = receipt(D / 'results/batched-residual-long1080-review-v2/validation.json',
    'd968f89ab835187d050a8b6805d9a11739492b954199ad090f1ab16f1f151751')
full = receipt(D / 'results/long-precision-1080-fp16_xmx-v1/validation.json',
    'ff4f46387ae73190794158dd3af48c5170a9809c82f44bc00802a7c7cf6f840c')
review_path = D / 'results/batched-residual-long1080-review-v2/user-review-v1.json'
assert sha(review_path) == '309f8f85b756ce43354a75a56278f51666d96584826c6baa15201b9c2590fdf0'
review = js(review_path)
assert review['nr256_residual_visually_accepted']
assert review['inference_report']['sha256'] == 'd968f89ab835187d050a8b6805d9a11739492b954199ad090f1ab16f1f151751'
source = Path(full['source'])
ffmpeg = ROOT.parent / 'xess-tools/work/d3d12-media-pipeline/deps/ffmpeg-lgpl-shared-9.0/ffmpeg-n9.0-latest-win64-lgpl-shared-9.0/bin/ffmpeg.exe'
assert sha(source) == full['sources'][str(source)] and sha(ffmpeg) == full['sources'][str(ffmpeg)]
for p in (Path(__file__), HERE / 'Run-LayoutCropValidationSuiteV1.cmd', HERE / 'run_layout_crop_validation_suite_v1.py',review_path):
    sources[str(p)] = sha(p)
import numpy as np
import torch
import triton
from layout_crop_stack_v1 import Stack
from nr_backend.execution import use_arithmetic_backend
from residual_scale_v1 import ResidualScale
from cubic_lut_constant_v1 import Constant,TABLE
import compressed_arrays_v1 as arrays
assert Path(triton.__file__).is_relative_to(TOOLCHAIN / 'site')
OUT.mkdir()
report = dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,complete_migration=False,
    candidate_promoted=False,frames=[],source=str(source),decode_filter=full['decode_filter'],
    paired_phase=paired_phase,queue_check=queue_check,logical_dispatch_delta={},
    human_review_reference=dict(path=str(review_path),sha256=sha(review_path)),
    new_quality_change=False,new_video_encoded=False,new_output_arrays=False)
save = lambda: (OUT / 'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
raw = lambda t: t.cpu().numpy().tobytes()
digest = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
stack = decoder = None
held = {}

def read_frame():
    parts,remaining = [],1080*1920*3
    while remaining:
        part = decoder.stdout.read(remaining)
        if not part:
            raise EOFError('Incomplete source frame')
        parts.append(part);remaining -= len(part)
    return np.frombuffer(b''.join(parts),dtype='u1').reshape(1080,1920,3).copy()

def run(rgb,motion,reset):
    with stack.installed(),use_arithmetic_backend('triton') as dispatch:
        torch.xpu.synchronize();start = time.perf_counter()
        canvas,flow = scaler.prepare(rgb,motion)
        low = stack.model(canvas,flow,reset=reset)
        output = scaler.composite(rgb,canvas,low)
        torch.xpu.synchronize();seconds = time.perf_counter()-start
    effective = dict(dispatch)
    assert effective.pop('xpu_graph_replay') == 1
    for key,count in stack.graph.last_entry.dispatch.items():
        if key != 'backend':
            effective[key] = effective.get(key,0)+count
    return output,low,seconds,effective

try:
    torch.set_num_threads(2)
    assert len(prior['frames']) == len(full['frames']) == 390
    scaler = ResidualScale(256)
    stack = Stack(EXACT)
    sources.update(stack.rewrite.fork.sources)
    report['runtime'] = dict(torch=torch.__version__,triton=triton.__version__,device=torch.xpu.get_device_name())
    decoder = subprocess.Popen([str(ffmpeg),'-v','error','-threads','2','-i',str(source),
        '-vf',full['decode_filter'],'-frames:v','390','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
    warm = [read_frame(),read_frame()]
    with torch.inference_mode():
        for i,pixels in enumerate(warm):
            rgb = torch.from_numpy(pixels.astype('f4')/255).to('xpu')
            motion = torch.from_numpy(arrays.load(prior['frames'][i]['motion'])).to('xpu')
            value,low,_,_ = run(rgb,motion,i==0)
            assert raw(low) == arrays.load(prior['frames'][i]['low_nr']).tobytes()
            assert digest(value.cpu().numpy()) == prior['frames'][i]['runs']['batched']['full_raw_sha256']
        stack.model.reset()
        print('Prewarmed candidate; checking 390 uninterrupted frames',flush=True)
        for i,old in enumerate(prior['frames']):
            pixels = warm[i] if i < 2 else read_frame()
            assert digest(pixels) == old['input_rgb8_sha256'] == full['frames'][i]['input_rgb8_sha256']
            assert old['motion'] == full['frames'][i]['motion']
            flow,expected = arrays.load(old['motion']),arrays.load(old['low_nr'])
            rgb_cpu = pixels.astype('f4')/255
            rgb,motion = torch.from_numpy(rgb_cpu).to('xpu'),torch.from_numpy(flow).to('xpu')
            value,low,seconds,dispatch = run(rgb,motion,i==0)
            a,b = value.cpu().numpy(),low.cpu().numpy()
            private = raw(stack.model._previous)
            assert a.shape == (1080,1920,3) and a.dtype == np.dtype('f4') and np.isfinite(a).all()
            assert b.shape == expected.shape == (256,256,3) and b.dtype == expected.dtype == np.dtype('f2') and np.isfinite(b).all()
            assert b.tobytes() == expected.tobytes(), ('Low NR',i)
            assert digest(a) == old['runs']['batched']['full_raw_sha256'], ('Composite',i)
            assert digest(b) == old['runs']['batched']['low_raw_sha256']
            assert private == b.tobytes() and stack.model.next_seed == i+1
            prior_dispatch=old['runs']['batched']['effective_dispatch']
            delta={key:dispatch.get(key,0)-prior_dispatch.get(key,0)
                for key in dispatch.keys()|prior_dispatch.keys() if dispatch.get(key)!=prior_dispatch.get(key)}
            mode='reset' if i==0 else 'temporal'
            assert delta==paired['logical_dispatch_delta'][mode]
            report['logical_dispatch_delta'][mode]=delta
            if i in (0,120):
                held[i] = value,low,a.tobytes(),b.tobytes()
            else:
                value.zero_();low.zero_()
                assert raw(stack.model._previous) == private
            for v,l,av,bl in held.values():
                assert raw(v) == av and raw(l) == bl
            assert raw(rgb) == rgb_cpu.tobytes() and raw(motion) == flow.tobytes()
            report['frames'].append(dict(frame=i,reset=i==0,input_rgb8_sha256=digest(pixels),motion=old['motion'],
                low_nr=old['low_nr'],low_raw_sha256=digest(b),full_raw_sha256=digest(a),seconds=seconds,
                low_bytes_equal=True,full_hash_equal=True,private_byte_equal_low=True,next_seed=stack.model.next_seed,
                effective_dispatch=dispatch,held_outputs_unchanged=True,inputs_unchanged=True))
            if i%30 == 0 or i == 389:
                save();print(json.dumps(dict(frame=i,total=390,all_outputs_match=True)),flush=True)
        assert decoder.stdout.read() == b'' and decoder.wait(timeout=30) == 0
        rgb = torch.from_numpy(warm[0].astype('f4')/255).to('xpu')
        motion = torch.from_numpy(arrays.load(prior['frames'][0]['motion'])).to('xpu')
        value,low,_,_ = run(rgb,motion,True)
        assert raw(low) == held[0][3] and raw(value) == held[0][2]
        for v,l,av,bl in held.values():
            assert raw(v) == av and raw(l) == bl
        assert stack.model.next_seed == 1 and len(stack.graph.entries) == 2 and stack.graph.replays == 393
        assert raw(Constant(stack.model).require()) == np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        report['builds'] = stack.rewrite.builds
        assert len(report['builds']) == 6
        report['candidate'] = candidate_gates(stack)
        report.update(passed=True,frames_completed=390,graphs=stack.graph.metadata(),graph_replays=stack.graph.replays,
            all_outputs_match_approved_sequence=True,independent_uninterrupted_history=True,
            reset_reproduces_first_frame=True,caller_ownership_passed=True,held_outputs_unchanged=True,
            lut_bytes_unchanged=True,no_dependency_swaps_on_steady_replay=True)
except BaseException:
    report['error'] = traceback.format_exc()
    raise
finally:
    try:
        if stack is not None:
            stack.close()
        if decoder is not None and decoder.poll() is None:
            decoder.kill();decoder.wait()
        finalize_sources()
    except BaseException:
        report.update(passed=False,finalization_error=traceback.format_exc())
        raise
    finally:
        save()
print(json.dumps(dict(passed=report['passed'],frames=report.get('frames_completed'),new_video=False)),flush=True)
