"""Replay all243 approved frames with fused INT8 and compare complete output bytes.

Reuse authenticated recorded FP16 motion and complete approved output artifacts.
No new encode or image copy when bytes match. Own uninterrupted temporal state.
"""
import hashlib,json,os,statistics,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=DREF/'results/fused-long-480-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha
gate=authenticate_main()
full_path=DREF/'results/fused-full-480-v1/validation.json'
full=js(full_path);assert full['passed'] and full['fusion_speedup']>1
assert all(sha(Path(p))==h for p,h in full['sources'].items())
prior_path=DREF/'results/long-precision-480-v1/validation.json'
assert sha(prior_path)=='858b4dd646c3be61263bc3fac6fabe400912686af672861638619c63be3be8a3'
prior=js(prior_path);assert prior['passed'] and prior['frames_completed']==243
review_path=prior_path.with_name('user-review-v1.json');review=js(review_path)
assert review['int8_cached_visually_accepted'] and review['byte_identical_follow_up_outputs_may_reuse_this_visual_review']
assert review['report_sha256']==sha(prior_path)
source=Path(prior['source']);assert sha(source)==prior['sources'][str(source)]
ffmpeg=ROOT.parent/'xess-tools/work/d3d12-media-pipeline/deps/ffmpeg-lgpl-shared-9.0/ffmpeg-n9.0-latest-win64-lgpl-shared-9.0/bin/ffmpeg.exe'
assert sha(ffmpeg)==prior['sources'][str(ffmpeg)]
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
import torch
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from fused_cached_matrices_v1 import FusedCachedMatrices
import compressed_arrays_v1 as arrays
paths=[Path(__file__),HERE/'Run-FusedLongV1.cmd',HERE/'fused_cached_matrices_v1.py',HERE/'fused_activation_int8_v1.py',HERE/'static_weight_cache_v2.py',HERE/'static_weight_cache_v1.py',HERE/'fast_matrices_v3.py',HERE/'compressed_arrays_v1.py',HERE/'immutable_artifacts_v1.py',full_path,prior_path,review_path,source,ffmpeg,*sorted((ROOT/'backend/nr_backend').glob('*.py'))]
frozen={str(p):sha(p) for p in paths}
for p in (ROOT/'backend/nr_backend').glob('*.py'):assert p.read_bytes()==(EXACT/'backend/nr_backend'/p.name).read_bytes()
OUT.mkdir()
report=dict(scope=__doc__,sources=frozen,exact_gate=gate,passed=False,frames=[],expected_frames=243,dimension='864x480',fps=24,duration_seconds=10.125,native_capture_for_this_sequence=False,comparison='Previously approved cached INT8; no change to its mathematical result',timing_scope='Diagnostic model plus synchronization only. For paired performance use fused-full-480-v1.',complete_migration=False,new_video_encoded=False,new_output_artifact_bytes=0)
decoder=None;experiment=FusedCachedMatrices()
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def read_frame():
    parts=[];remaining=864*480*3
    while remaining:
        part=decoder.stdout.read(remaining)
        if not part:raise EOFError('Incomplete decoded frame')
        parts.append(part);remaining-=len(part)
    return np.frombuffer(b''.join(parts),dtype='u1').reshape(480,864,3).copy()
def tensors(pixels,flow):return torch.from_numpy(pixels.astype('f4')/255).to('xpu'),torch.from_numpy(flow).to('xpu')
try:
    torch.set_num_threads(2)
    decoder=subprocess.Popen([str(ffmpeg),'-v','error','-i',str(source),'-vf',prior['decode_filter'],'-frames:v','243','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
    warm_pixels=[read_frame(),read_frame()]
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    with torch.inference_mode(),experiment.installed():
        experiment.static_cache.prepack(model);experiment.select('int8_fused')
        for i,pixels in enumerate(warm_pixels):
            flow=arrays.load(prior['frames'][i]['motion']);rgb,motion=tensors(pixels,flow)
            with use_arithmetic_backend('triton'):model(rgb,motion,reset=i==0)
            torch.xpu.synchronize()
        model.reset();print('Prewarmed fused full243 replay',flush=True)
        for i,old_frame in enumerate(prior['frames']):
            pixels=warm_pixels[i] if i<2 else read_frame()
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==old_frame['input_rgb8_sha256']
            flow=arrays.load(old_frame['motion']);rgb,motion=tensors(pixels,flow)
            experiment.select('int8_fused')
            torch.xpu.synchronize();started=time.perf_counter()
            with use_arithmetic_backend('triton') as dispatch:
                value=model(rgb,motion,reset=i==0);torch.xpu.synchronize()
            seconds=time.perf_counter()-started
            actual=value.cpu().numpy();private=model._previous.cpu().numpy()
            old=old_frame['runs']['int8_cached'];meta=old['output'];expected=arrays.load(meta)
            equal=actual.tobytes()==expected.tobytes()
            row=dict(frame=i,reset=i==0,seconds=seconds,byte_equal_approved=equal,output=meta if equal else arrays.save(actual),private_byte_equal_output=private.tobytes()==actual.tobytes(),motion=old_frame['motion'],input_rgb8_sha256=old_frame['input_rgb8_sha256'],next_seed=model.next_seed,dispatches=dict(dispatch),matrix_calls=experiment.calls.copy(),matrix_shapes=experiment.shapes.copy())
            report['frames'].append(row)
            assert actual.dtype==np.dtype('f2') and equal and row['private_byte_equal_output'],i
            assert np.isfinite(actual).all() and model.next_seed==i+1 and dict(dispatch)==old['dispatches']
            assert experiment.calls.get('packed_weight_hit')==experiment.calls['int8_dense'] and not experiment.calls.get('packed_weight_miss') and experiment.calls.get('int8_fused',0)>0
            value.zero_();assert model._previous.cpu().numpy().tobytes()==private.tobytes()
            assert rgb.cpu().numpy().tobytes()==(pixels.astype('f4')/255).tobytes() and motion.cpu().numpy().tobytes()==flow.tobytes()
            if i%12==0 or i==242:
                save();print(json.dumps(dict(frame=i,all_bytes_equal=True,fused_calls=experiment.calls['int8_fused'])),flush=True)
        assert decoder.stdout.read()==b''
        assert decoder.wait(timeout=30)==0,decoder.stderr.read().decode(errors='replace')
        rgb,motion=tensors(warm_pixels[0],arrays.load(prior['frames'][0]['motion']))
        experiment.select('int8_fused')
        with use_arithmetic_backend('triton'):value=model(rgb,motion,reset=True)
        assert value.cpu().numpy().tobytes()==arrays.load(prior['frames'][0]['runs']['int8_cached']['output']).tobytes() and model.next_seed==1
        report.update(passed=True,frames_completed=243,all_outputs_byte_equal_approved=True,independent_history=True,reset_reproduces_first_frame=True,caller_ownership_guards_passed=True,human_review_reused_from=str(review_path),human_review_quote=review['user_response'],diagnostic_mean_model_seconds=statistics.mean(row['seconds'] for row in report['frames']))
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    if decoder is not None and decoder.poll() is None:decoder.kill();decoder.wait()
    assert all(sha(Path(p))==h for p,h in frozen.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],frames_completed=report.get('frames_completed'),new_output_artifact_bytes=0)),flush=True)
