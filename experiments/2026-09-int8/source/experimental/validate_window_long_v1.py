"""Replay243 approved frames with strided attention and fused window permutations.

Reuse authenticated recorded FP16 motion and complete approved output artifacts.
No new encode or image copy when bytes match. Own uninterrupted temporal state.
"""
import argparse,hashlib,json,os,statistics,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=('int8_fused_v2','fp16_xmx'))
MODE=parser.parse_args().mode
OLD_MODE='int8_cached' if MODE=='int8_fused_v2' else 'fp16_xmx'
OUT=DREF/f'results/{MODE}-window-layout-long-480-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha
gate=authenticate_main()
shared_guard_path=DREF/'results/graph-body-480-v6/validation.json'
shared_guard=js(shared_guard_path);assert shared_guard['passed']
assert all(sha(Path(p))==h for p,h in shared_guard['sources'].items())
full_path=DREF/'results/window-layout-full-480-v1/validation.json'
full=js(full_path);assert full['passed'] and full['window_layout_speedup'][MODE]>1
assert all(sha(Path(p))==h for p,h in full['sources'].items())
prior_path=DREF/'results/long-precision-480-v1/validation.json'
assert sha(prior_path)=='858b4dd646c3be61263bc3fac6fabe400912686af672861638619c63be3be8a3'
prior=js(prior_path);assert prior['passed'] and prior['frames_completed']==243
review_path=prior_path.with_name('user-review-v1.json');review=js(review_path)
assert review[OLD_MODE+'_visually_accepted'] and review['byte_identical_follow_up_outputs_may_reuse_this_visual_review']
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
from strided_batched_v2 import StridedMatrices as FusedCachedMatrices
from window_layout_v1 import WindowLayout
from swin_scheduling_v1 import SwinScheduling
from graph_front_v6 import GraphFront
import compressed_arrays_v1 as arrays
paths=[Path(__file__),HERE/'Run-WindowLongV1.cmd',HERE/'window_layout_v1.py',HERE/'strided_batched_v2.py',HERE/'strided_batched_v1.py',HERE/'graph_front_v6.py',HERE/'graph_front_v5.py',HERE/'graph_front_v4.py',HERE/'graph_front_v3.py',HERE/'graph_front_v2.py',HERE/'graph_front_v1.py',HERE/'capture_body_v1.py',HERE/'swin_scheduling_v1.py',HERE/'fused_cached_matrices_v2.py',HERE/'fused_cached_matrices_v1.py',HERE/'fused_activation_int8_v1.py',HERE/'static_weight_cache_v2.py',HERE/'static_weight_cache_v1.py',HERE/'fast_matrices_v3.py',HERE/'compressed_arrays_v1.py',HERE/'immutable_artifacts_v1.py',shared_guard_path,full_path,prior_path,review_path,source,ffmpeg,*sorted((ROOT/'backend/nr_backend').glob('*.py'))]
frozen={str(p):sha(p) for p in paths}
for p in (ROOT/'backend/nr_backend').glob('*.py'):assert p.read_bytes()==(EXACT/'backend/nr_backend'/p.name).read_bytes()
OUT.mkdir()
report=dict(scope=__doc__,sources=frozen,exact_gate=gate,passed=False,frames=[],expected_frames=243,dimension='864x480',fps=24,duration_seconds=10.125,native_capture_for_this_sequence=False,mode=MODE,comparison='Previously approved '+OLD_MODE+'; identical output expected',timing_scope='Diagnostic model plus synchronization only. Includes dynamic front, graph copies, completion wait and private history. For paired performance use graph-full-480-v3.',complete_migration=False,new_video_encoded=False,new_output_artifact_bytes=0)
decoder=None;adapter=None;held=[];experiment=FusedCachedMatrices();scheduler=SwinScheduling(enabled=True);layout=WindowLayout()
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
    experiment.select(MODE);adapter=GraphFront(model,arithmetic=experiment)
    with torch.inference_mode(),experiment.installed(),scheduler.installed(),layout.installed(),adapter.installed():
        if MODE=='int8_fused_v2':experiment.static_cache.prepack(model)
        experiment.select(MODE);scheduler.counts={}
        for i,pixels in enumerate(warm_pixels):
            flow=arrays.load(prior['frames'][i]['motion']);rgb,motion=tensors(pixels,flow)
            with use_arithmetic_backend('triton'):model(rgb,motion,reset=i==0)
            torch.xpu.synchronize()
        model.reset();print('Prewarmed '+MODE+' graphed full243 replay',flush=True)
        for i,old_frame in enumerate(prior['frames']):
            pixels=warm_pixels[i] if i<2 else read_frame()
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==old_frame['input_rgb8_sha256']
            flow=arrays.load(old_frame['motion']);rgb,motion=tensors(pixels,flow)
            experiment.select(MODE);scheduler.counts={}
            torch.xpu.synchronize();torch.xpu.reset_peak_memory_stats();started=time.perf_counter()
            with use_arithmetic_backend('triton') as dispatch:
                value=model(rgb,motion,reset=i==0);torch.xpu.synchronize()
            seconds=time.perf_counter()-started
            actual=value.cpu().numpy();private=model._previous.cpu().numpy()
            old=old_frame['runs'][OLD_MODE];meta=old['output'];expected=arrays.load(meta)
            equal=actual.tobytes()==expected.tobytes()
            effective=dict(dispatch);assert effective.pop('xpu_graph_replay')==1
            for key,count in adapter.last_entry.dispatch.items():
                if key!='backend':effective[key]=effective.get(key,0)+count
            row=dict(frame=i,reset=i==0,seconds=seconds,byte_equal_approved=equal,output=meta if equal else arrays.save(actual),private_byte_equal_output=private.tobytes()==actual.tobytes(),motion=old_frame['motion'],input_rgb8_sha256=old_frame['input_rgb8_sha256'],next_seed=model.next_seed,runtime_dispatch=dict(dispatch),captured_dispatch=adapter.last_entry.dispatch,effective_dispatch=effective,graph_replays=adapter.replays,peak_allocated=torch.xpu.max_memory_allocated(),peak_reserved=torch.xpu.max_memory_reserved(),reserved=torch.xpu.memory_reserved())
            report['frames'].append(row)
            assert actual.dtype==np.dtype('f2') and equal and row['private_byte_equal_output'],i
            assert np.isfinite(actual).all() and model.next_seed==i+1 and effective==old['dispatches']
            assert adapter.owned_packed_bytes==(147185664 if MODE=='int8_fused_v2' else 0)
            for old_value,old_bytes in held:assert old_value.cpu().numpy().tobytes()==old_bytes
            if i in (0,120):held.append((value,actual.tobytes()))
            else:value.zero_();assert model._previous.cpu().numpy().tobytes()==private.tobytes()
            assert rgb.cpu().numpy().tobytes()==(pixels.astype('f4')/255).tobytes() and motion.cpu().numpy().tobytes()==flow.tobytes()
            if i%12==0 or i==242:
                report['graphs']=adapter.metadata();save();print(json.dumps(dict(frame=i,all_bytes_equal=True,mode=MODE,graph_replays=adapter.replays)),flush=True)
        assert decoder.stdout.read()==b''
        assert decoder.wait(timeout=30)==0,decoder.stderr.read().decode(errors='replace')
        rgb,motion=tensors(warm_pixels[0],arrays.load(prior['frames'][0]['motion']))
        experiment.select(MODE);scheduler.counts={}
        with use_arithmetic_backend('triton'):value=model(rgb,motion,reset=True)
        assert value.cpu().numpy().tobytes()==arrays.load(prior['frames'][0]['runs'][OLD_MODE]['output']).tobytes() and model.next_seed==1
        assert len(adapter.entries)==2 and adapter.replays==246
        assert len({str(e.graph.pool()) for e in adapter.entries.values()})==1
        for old_value,old_bytes in held:assert old_value.cpu().numpy().tobytes()==old_bytes
        report.update(passed=True,frames_completed=243,held_outputs_survive_replay=True,graphs=adapter.metadata(),owned_packed_bytes=adapter.owned_packed_bytes,all_outputs_byte_equal_approved=True,independent_history=True,reset_reproduces_first_frame=True,caller_ownership_guards_passed=True,human_review_reused_from=str(review_path),human_review_quote=review['user_response'],peak_allocated_bytes=max(row['peak_allocated'] for row in report['frames']),peak_reserved_bytes=max(row['peak_reserved'] for row in report['frames']),graph_pool_bytes=sum(segment['total_size'] for segment in torch.xpu.memory_snapshot(adapter.capture_pool)),diagnostic_mean_model_seconds=statistics.mean(row['seconds'] for row in report['frames']))
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    if adapter is not None:adapter.close()
    if decoder is not None and decoder.poll() is None:decoder.kill();decoder.wait()
    assert all(sha(Path(p))==h for p,h in frozen.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],frames_completed=report.get('frames_completed'),new_output_artifact_bytes=0)),flush=True)
