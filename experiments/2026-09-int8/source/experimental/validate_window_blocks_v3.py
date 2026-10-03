"""Validate packed-window to cropped-output Swin blocks and complete NR frames.

The reusable stack is compared in full against the approved first two frames.
Keep scalar operation metadata and the escaping final result for conservative
producer/consumer rewrites. No performance timing or additional frame dumps.
"""
import hashlib,json,os,sys,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/window-blocks-v3';assert not OUT.exists()
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(TOOLCHAIN/'site'),str(R),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
provision=TOOLCHAIN/'provision-v1.json'
assert sha(provision)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision)['files'].items())
prior_path=D/'results/residual-scale-fp16_xmx-256-v1/validation.json'
assert sha(prior_path)=='056251b4e2a839fcc44845d9646d2cd8ae1e9ed7f99fd1858dc7e3d51e3e50f9'
prior=js(prior_path)
pipeline_path=D/'results/fp8-graph-residual256-v1/validation.json'
assert sha(pipeline_path)=='7a45f2bdf93e9cfc45778df3703cf37172c1cb84667459af178b877c52c28af7'
pipeline=js(pipeline_path);assert pipeline['passed'] and js(pipeline_path.parent.with_suffix('.log.lease.json'))['returncode']==0
assert js(pipeline_path.with_name('saved-audit-v1.json'))['passed']
assert all(sha(p)==h for p,h in pipeline['sources'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
from PIL import Image
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site') and triton.__version__.startswith('3.8.0')
from nr256_selected_stack_v1 import Stack
from window_blocks_v3 import WindowBlocks
from immutable_artifacts_v1 import put
import nr_backend.multihead_block as blocks
from spill_preflight_v1 import select
from window_block_projection_v1 import _project as known_spilling_project
import capture_body_v1 as body
from nr_backend.execution import use_arithmetic_backend
from cubic_lut_constant_v1 import Constant,TABLE
from residual_scale_v1 import ResidualScale
import compressed_arrays_v1 as arrays
inputs=R/'inputs/flow-full-1920x1080-v3';manifest_path=inputs/'manifest.json';manifest=js(manifest_path)
sources=dict(pipeline['sources'])
for p in (Path(__file__),HERE/'Run-WindowBlocksV3.cmd',HERE/'nr256_selected_stack_v1.py',HERE/'window_blocks_v1.py',HERE/'window_blocks_v3.py',HERE/'window_block_projection_v1.py',HERE/'window_block_projection_v3.py',HERE/'window_block_attention_v3.py',HERE/'spill_preflight_v1.py',provision,prior_path,pipeline_path,manifest_path):sources[str(p)]=sha(p)
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,complete_migration=False,frames=[],timing_measured=False)
stack=None
raw=lambda t:t.cpu().numpy().tobytes()
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
try:
    torch.set_num_threads(2);stack=Stack(EXACT);scaler=ResidualScale(256)
    adapter=WindowBlocks(stack.model,stack.provider,stack.components[2]);stack.components.append(adapter)
    assert len(adapter.modules)==36;report['block_proofs']=[]
    with torch.inference_mode(),stack.installed(),use_arithmetic_backend('triton'):
        for i in (0,1):
            spec=manifest['frames'][i]
            pixels=np.asarray(Image.open(inputs/spec['file']).convert('RGB'),dtype='f4')/255
            flow=np.fromfile(inputs/spec['motion_file'],'<f4').reshape(1080,1920,2)
            rgb=torch.from_numpy(pixels).to('xpu');motion=torch.from_numpy(flow).to('xpu')
            low_rgb,low_motion=scaler.prepare(rgb,motion)
            low=stack.model(low_rgb,low_motion,reset=i==0);output=scaler.composite(rgb,low_rgb,low)
            expected=prior['runs'][i]
            assert raw(low)==arrays.load(expected['low_nr']).tobytes()
            assert raw(output)==arrays.load(expected['output']).tobytes()
            assert raw(stack.model._previous)==raw(low) and stack.model.next_seed==i+1
            assert raw(rgb)==pixels.tobytes() and raw(motion)==flow.tobytes()
            report['frames'].append(dict(frame=i,low_nr=expected['low_nr'],output=expected['output'],byte_equal=True,history_equal=True,next_seed=i+1,inputs_unchanged=True))
    report['builds']=list(stack.rewrite.builds)
    assert len(report['builds'])==6
    report['normal_block_calls']=list(adapter.calls)
    assert len(adapter.calls)==216
    assert all(b['triton_calls']==683 and b['standalone_fp8']==196 and b['elided_fp8']==231 for b in report['builds'])
    private=raw(stack.model._previous);seed=stack.model.next_seed
    def probe(module,features,packed,mlp,result,kernel,info):
        attention_hw=blocks.quantize_fp8(module.attention(mlp))
        hp,wp,c=mlp.shape;heads=c//32
        expected_packed=attention_hw.reshape(hp//8,8,wp//8,8,heads,32).permute(4,0,2,1,3,5).reshape(heads,hp//8,wp//8,64,32)[...,module.attention.pixel_order,:].contiguous()
        assert raw(packed)==raw(expected_packed),('Packed quantization mismatch',info)
        if 'spill_guard' not in report and (hp,wp,c)==(80,80,64):
            scratch=torch.full_like(result,1.25);before=raw(scratch);sy,sx=module.window_shift;h,w=features.shape[:2]
            args_for=lambda config:(packed,module.output_weight,mlp,module.skip_scale,module.attention.pixel_order,scratch,hp,wp,h,w,c,sy,sx,*config)
            grid_for=lambda config:(triton.cdiv(hp*wp,config[0]),triton.cdiv(c,config[1]))
            chosen,compiled,selection=select(known_spilling_project,[(64,64),(16,32)],args_for,grid_for,num_warps=4,enable_fp_fusion=False)
            assert chosen==(16,32) and selection['attempts'][0]['spills']==832 and selection['attempts'][1]['spills']==0
            assert raw(scratch)==before
            report['spill_guard']=dict(**selection,no_gpu_dispatch=True,sentinel_output_unchanged=True)
        expected=adapter.original(module,features)
        assert raw(result)==raw(expected),('Block output mismatch',info)
        assert result.is_contiguous() and result.storage_offset()==0
        report['block_proofs'].append(dict(**info,input=arrays.save(features.cpu().numpy()),output=arrays.save(result.cpu().numpy()),
            all_bytes_equal=True,unquantized_crop_equal=True,packed_bytes_equal=True,packed=arrays.save(packed.cpu().numpy()),attention_llir=put(adapter.last_attention_kernel.asm['llir'].encode('utf-8'),'llir'),attention_spills=adapter.last_attention_kernel.n_spills,llir=put(kernel.asm['llir'].encode('utf-8'),'llir'),spills=getattr(kernel,'n_spills',None)))
    adapter.probe=probe
    with torch.inference_mode(),stack.installed(),body.installed(),use_arithmetic_backend('triton'):
        for entry in stack.graph.entries.values():
            value=body.forward_front(stack.model,**entry.inputs,sigmoid=stack.model.sigmoid,blend_scale=stack.model.blend_scale,return_float32=False)
            assert raw(value)==raw(entry.output)
    assert len(report['block_proofs'])==72 and 'spill_guard' in report
    adapter.probe=None
    report['shape_guards']=[]
    with torch.inference_mode(),stack.installed(),use_arithmetic_backend('triton'):
        modules=[m for m in stack.model.modules() if id(m) in adapter.modules]
        for c in (64,128,256):
            module=next(m for m in modules if m.channels==c and m.window_shift==(4,4))
            for kind in ('zero','strided'):
                backing=torch.zeros((10,24,c),device='xpu',dtype=torch.float16)
                if kind=='strided':backing[:]=torch.arange(c,device='xpu',dtype=torch.float16)*.003
                features=backing[:,:12] if kind=='zero' else backing[:,::2]
                before=len(adapter.calls);expected=adapter.original(module,features);actual=adapter.apply(module,features)
                assert raw(actual)==raw(expected) and len(adapter.calls)==before+1
                report['shape_guards'].append(dict(channels=c,kind=kind,shape=list(features.shape),all_bytes_equal=True))
    report['history_and_seed_unchanged_by_block_validation']=True
    assert raw(stack.model._previous)==private and stack.model.next_seed==seed
    assert raw(Constant(stack.model).require())==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
    with stack.installed():stack.graph._validate()
    report.update(passed=True,lut_unchanged=True,graph_metadata=stack.graph.metadata(),runtime=dict(triton=triton.__version__,torch=torch.__version__))
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    if stack is not None:stack.close()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],block_proofs=len(report.get('block_proofs',[])),body_counts=[(b['triton_calls'],b['standalone_fp8'],b['elided_fp8']) for b in report.get('builds',[])]),indent=2),flush=True)
