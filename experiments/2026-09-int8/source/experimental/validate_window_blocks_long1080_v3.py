"""Validate isolated3.8/K8 plus all existing fusions with native cubic, fused native-half ViT64 attention and provenance-driven FP8 graph rewrite plus joint packed-window blocks over the previously approved 390-frame NR256 sequence.

Compare every complete low NR tensor and complete FP32 composite hash with the
approved run. Keep an independent uninterrupted candidate history. Reuse saved
references without another full-frame dump or encode. This is a correctness run;
IO between frames makes its timings diagnostic, not the paired speed benchmark.
"""
import hashlib,json,os,statistics,subprocess,sys,time,traceback,urllib.request
from contextlib import ExitStack
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=D/'results/window-blocks-long1080-v3';assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
sys.path[:0]=[str(TOOLCHAIN/'site'),str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
provision_path=TOOLCHAIN/'provision-v1.json'
assert sha(provision_path)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision_path)['files'].items())
paired_path=D/'results/window-blocks-residual256-v3/validation.json'
prior_path=D/'results/batched-residual-long1080-review-v2/validation.json'
review_path=prior_path.with_name('user-review-v1.json')
full_path=D/'results/long-precision-1080-fp16_xmx-v1/validation.json'
assert sha(paired_path)=='c29c1a67b2e1d8314b0f8169a79c7ea51aa33ffbebecbb4078d23dbbf6316586'
assert sha(prior_path)=='d968f89ab835187d050a8b6805d9a11739492b954199ad090f1ab16f1f151751'
assert sha(review_path)=='309f8f85b756ce43354a75a56278f51666d96584826c6baa15201b9c2590fdf0'
assert sha(full_path)=='ff4f46387ae73190794158dd3af48c5170a9809c82f44bc00802a7c7cf6f840c'
paired=js(paired_path);prior=js(prior_path);full=js(full_path);review=js(review_path)
assert review['nr256_residual_visually_accepted'] and review['inference_report']['sha256']==sha(prior_path)
assert prior['passed'] and full['passed'] and len(prior['frames'])==len(full['frames'])==390
assert paired['passed'] and js(paired_path.parent.with_suffix('.log.lease.json'))['returncode']==0
frozen={}
for p in (paired_path,prior_path,full_path):
    record=js(p);audit_path=p.with_name('saved-audit-v1.json');audit=js(audit_path)
    assert audit['passed'] and audit['report_sha256']==sha(p)
    assert all(sha(q)==h for q,h in record['sources'].items())
    frozen.update(record['sources']);frozen[str(p)]=sha(p);frozen[str(audit_path)]=sha(audit_path)
source=Path(full['source'])
ffmpeg=ROOT.parent/'xess-tools/work/d3d12-media-pipeline/deps/ffmpeg-lgpl-shared-9.0/ffmpeg-n9.0-latest-win64-lgpl-shared-9.0/bin/ffmpeg.exe'
assert sha(source)==full['sources'][str(source)] and sha(ffmpeg)==full['sources'][str(ffmpeg)]
for p in (Path(__file__),HERE/'Run-WindowBlocksLong1080V3.cmd',review_path):frozen[str(p)]=sha(p)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site') and triton.__version__.startswith('3.8.0')
from current_dense_tiled_provider_v1 import CurrentDenseTiledMatrices as K8TiledMatrices
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from native_cubic_adapters_v1 import FusedBatched,FusedSplit,FusedC32
from cubic_lut_constant_v1 import register,Constant,TABLE
from fused_vit_projection_v3 import FusedVitProjection
from fused_swin_native_half_v1 import FusedSwin
from native_half_head_layout_v1 import HeadLayout as ProjectedHeadLayout
from fused_dynamic_front_v1 import FusedFront
from graph_front_v6 import GraphFront
from fused_graph_history_warp_v2 import FusedHistoryWarp as GraphHistoryWarp
import compressed_arrays_v1 as arrays
from residual_scale_v1 import ResidualScale
from fused_vit_attention64_adapter_v2 import FusedVitQKV
from fp8_graph_rewrite_v1 import FP8GraphRewrite
from window_blocks_v3 import WindowBlocks
OUT.mkdir()
report=dict(scope=__doc__,sources=frozen,exact_gate=gate,passed=False,frames=[],complete_migration=False,
            source=str(source),decode_filter=full['decode_filter'],dimension='1920x1080',fps='60000/1001',nr_input=[256,256],source_canvas=[1920,1080],
            human_review_reference=dict(path=str(review_path),sha256=sha(review_path)),
            new_quality_change=False,new_video_encoded=False,new_output_arrays=False,
            comparison='Complete low tensor bytes and SHA256 of every byte in complete FP32 composite')
model=adapter=warp=decoder=None;held={}
report['runtime']=dict(triton_version=triton.__version__,triton_file=triton.__file__,torch_version=torch.__version__,isolated=True)
digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest()

def save():
    p=OUT/'progress.tmp';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    p.replace(OUT/'validation.json')

def read_frame():
    parts=[];remaining=1080*1920*3
    while remaining:
        part=decoder.stdout.read(remaining)
        if not part:raise EOFError('Incomplete source frame')
        parts.append(part);remaining-=len(part)
    return np.frombuffer(b''.join(parts),dtype='u1').reshape(1080,1920,3).copy()

def run(rgb,motion,reset):
    with ExitStack() as stack:
        for component in components:stack.enter_context(component.installed())
        torch.xpu.synchronize();started=time.perf_counter()
        with use_arithmetic_backend('triton') as dispatch:
            canvas,flow=scaler.prepare(rgb,motion)
            low=model(canvas,flow,reset=reset)
            output=scaler.composite(rgb,canvas,low)
            torch.xpu.synchronize()
        seconds=time.perf_counter()-started
    effective=dict(dispatch);assert effective.pop('xpu_graph_replay')==1
    for key,count in adapter.last_entry.dispatch.items():
        if key!='backend':effective[key]=effective.get(key,0)+count
    return output,low,seconds,effective

try:
    torch.set_num_threads(2);scaler=ResidualScale(256);report['residual_scale']=scaler.metadata()
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',
        EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    register(model);provider=K8TiledMatrices(model);provider.select('fp16_xmx')
    adapter=GraphFront(model,arithmetic=provider);warp=GraphHistoryWarp(model)
    components=[provider,FusedSwin(provider),ProjectedHeadLayout(model,provider),adapter,
        FusedBatched(model,provider,workload='small'),FusedSplit(model,provider),FusedC32(model,provider),
        FusedVitProjection(model,provider),FusedFront(model),warp,FusedVitQKV(model,provider)]
    rewrite=FP8GraphRewrite(model,provider);components.insert(-1,rewrite)
    block_layout=WindowBlocks(model,provider,components[2]);components.insert(-1,block_layout)
    decoder=subprocess.Popen([str(ffmpeg),'-v','error','-threads','2','-i',str(source),'-vf',full['decode_filter'],
        '-frames:v','390','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=0x08000000)
    warm=[read_frame(),read_frame()]
    with torch.inference_mode():
        for i,pixels in enumerate(warm):
            rgb=torch.from_numpy(pixels.astype('f4')/255).to('xpu')
            motion=torch.from_numpy(arrays.load(prior['frames'][i]['motion'])).to('xpu')
            value,low,_,_=run(rgb,motion,i==0)
            assert low.cpu().numpy().tobytes()==arrays.load(prior['frames'][i]['low_nr']).tobytes()
            assert digest(value.cpu().numpy())==prior['frames'][i]['runs']['batched']['full_raw_sha256']
        model.reset();print('Prewarmed candidate; validating390 uninterrupted frames',flush=True)
        for i,old in enumerate(prior['frames']):
            pixels=warm[i] if i<2 else read_frame();assert digest(pixels)==old['input_rgb8_sha256']==full['frames'][i]['input_rgb8_sha256']
            assert old['motion']==full['frames'][i]['motion']
            flow=arrays.load(old['motion']);expected=arrays.load(old['low_nr'])
            rgb_cpu=pixels.astype('f4')/255;rgb=torch.from_numpy(rgb_cpu).to('xpu');motion=torch.from_numpy(flow).to('xpu')
            value,low,seconds,dispatch=run(rgb,motion,i==0)
            a=value.cpu().numpy();b=low.cpu().numpy();private=model._previous.cpu().numpy()
            assert a.shape==(1080,1920,3) and a.dtype==np.dtype('f4') and np.isfinite(a).all()
            assert b.shape==expected.shape==(256,256,3) and b.dtype==expected.dtype==np.dtype('f2') and np.isfinite(b).all()
            assert b.tobytes()==expected.tobytes(),f'Low NR mismatch frame{i}'
            assert digest(a)==old['runs']['batched']['full_raw_sha256'],f'Composite mismatch frame{i}'
            assert digest(b)==old['runs']['batched']['low_raw_sha256']
            assert private.tobytes()==b.tobytes() and model.next_seed==i+1
            assert dispatch==old['runs']['batched']['effective_dispatch']
            if i in (0,120):held[i]=(value,low,a.tobytes(),b.tobytes())
            else:
                value.zero_();low.zero_()
                assert model._previous.cpu().numpy().tobytes()==private.tobytes()
            for v,l,av,bl in held.values():
                assert v.cpu().numpy().tobytes()==av and l.cpu().numpy().tobytes()==bl
            assert rgb.cpu().numpy().tobytes()==rgb_cpu.tobytes() and motion.cpu().numpy().tobytes()==flow.tobytes()
            report['frames'].append(dict(frame=i,reset=i==0,input_rgb8_sha256=digest(pixels),motion=old['motion'],
                low_nr=old['low_nr'],low_raw_sha256=digest(b),full_raw_sha256=digest(a),seconds=seconds,
                low_bytes_equal=True,full_hash_equal=True,private_byte_equal_low=True,next_seed=model.next_seed,
                effective_dispatch=dispatch,held_outputs_unchanged=True,inputs_unchanged=True))
            if i%30==0 or i==389:save();print(json.dumps(dict(frame=i,total=390,all_outputs_match=True)),flush=True)
        assert decoder.stdout.read()==b'' and decoder.wait(timeout=30)==0
        rgb=torch.from_numpy(warm[0].astype('f4')/255).to('xpu')
        motion=torch.from_numpy(arrays.load(prior['frames'][0]['motion'])).to('xpu')
        value,low,_,_=run(rgb,motion,True)
        assert low.cpu().numpy().tobytes()==held[0][3] and value.cpu().numpy().tobytes()==held[0][2]
        for v,l,av,bl in held.values():assert v.cpu().numpy().tobytes()==av and l.cpu().numpy().tobytes()==bl
        assert model.next_seed==1 and len(adapter.entries)==2 and adapter.replays==393
        assert Constant(model).require().cpu().numpy().tobytes()==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        assert set(provider.k8_calls)=={'pre','post'} and provider.k8_calls['pre']==provider.k8_calls['post']>0
        report['k8_calls']=dict(provider.k8_calls)
        assert warp.fused_builds==1
        report['fused_history_builds']=warp.fused_builds
        assert components[-1].calls>0
        report['vit_qkv_calls']=components[-1].calls
        report['fused_vit_attention_calls']=components[-1].attention_calls
        assert report['fused_vit_attention_calls']==report['vit_qkv_calls']==48
        assert len(provider.tiled_calls)==17 and sum(provider.tiled_calls.values())==498
        report['dense_tiles_calls']=dict(provider.tiled_calls)
        report['native_cubic_calls']={f:dict(components[i].native_calls) for f,i in [('batched',4),('split',5),('c32',6)]}
        assert report['native_cubic_calls']==dict(c32={'102400':6,'25600':12,'28224':12,'26880':24,'107584':6},batched={'576x256':96},split={'144x512':96})
        report['window_block_calls']=block_layout.calls;assert len(block_layout.calls)==216
        report['quantization_graph_builds']=rewrite.builds
        assert len(rewrite.builds)==6 and all(b['elided_fp8']==231 and b['triton_calls']==683 and b['standalone_fp8']==196 and b['quantization_calls']==427 for b in rewrite.builds)
        report.update(passed=True,frames_completed=390,graphs=adapter.metadata(),graph_replays=adapter.replays,
            all_outputs_match_approved_sequence=True,independent_uninterrupted_history=True,
            reset_reproduces_first_frame=True,caller_ownership_guards_passed=True,held_outputs_survive_replay=True,
            lut_bytes_unchanged=True,diagnostic_temporal_mean_ms=statistics.mean(f['seconds'] for f in report['frames'][1:])*1000)
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    if warp is not None:warp.close()
    if adapter is not None:adapter.close()
    if decoder is not None and decoder.poll() is None:decoder.kill();decoder.wait()
    assert all(sha(p)==h for p,h in frozen.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],frames=report.get('frames_completed'),new_video=False)),flush=True)
