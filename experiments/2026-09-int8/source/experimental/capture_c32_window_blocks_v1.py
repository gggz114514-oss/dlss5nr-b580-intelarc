"""Capture complete C32 block inputs and outputs from the approved NR256 body.

Saved temporal frame181 operands, not a newly simulated history. No standalone
block result is accepted unless the complete body still matches its frozen bytes.
"""
import hashlib,json,os,sys,traceback,urllib.request
from contextlib import ExitStack
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/c32-window-blocks-v1'
assert not OUT.exists();os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-sm89-v1')
sys.path[:0]=[str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main();sources={};records={}
for key,folder,h in [
    ('native','experimental/native-half-attention-v1','4ee4b5efa75fb8a78310e2824b1631980568c7dd7f128ab8fc8055d635507a1b'),
    ('body','experimental/small-branched-mlp-operands-v1','0c3ee88b397831266e6359e5eb24db5d77f3279f5bd9c67f9bda8ae11df4c11b')]:
    p=D/folder/'validation.json';assert sha(p)==h;r=js(p)
    assert r['passed'] and js(p.parent.with_suffix('.log.lease.json'))['returncode']==0
    assert all(sha(q)==h for q,h in r['sources'].items())
    records[key]=r;sources.update(r['sources']);sources[str(p)]=sha(p)
for p in [Path(__file__),HERE/'Run-CaptureC32WindowBlocksV1.cmd']:sources[str(p)]=sha(p)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
from nr_backend.temporal import MotionNR
from nr_backend.c32_block import C32SwinBlock
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from batched_branched_mlp_v2 import FusedBatched
from fused_split_ffwd_v2 import FusedSplit
from fused_c32_mlp_lut_v1 import FusedC32
from cubic_lut_constant_v1 import register,Constant,TABLE
from fused_vit_projection_v3 import FusedVitProjection
from native_half_head_layout_v1 import HeadLayout
from fused_swin_native_half_v1 import FusedSwin
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,cases=[],
    source_frame=181,body_inputs=records['body']['body_inputs'],expected_output=records['body']['expected_output'],
    raw_bytes=0,maximum_raw_bytes=128*2**20,complete_migration=False)
raw=lambda t:t.cpu().numpy().tobytes()
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
overrides=[]
try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut=register(model);constant=Constant(model);provider=StridedMatrices();provider.select('fp16_xmx')
    modules={n:m for n,m in model.named_modules() if type(m) is C32SwinBlock};seen=set()
    inputs={n:torch.from_numpy(arrays.load(m)).to('xpu') for n,m in report['body_inputs'].items()}
    def capture(name,module,original,features):
        assert name not in seen;seen.add(name)
        # Snapshot logical dispatch delta without changing the enclosing context.
        before=dict(dispatch);value=original(features)
        delta={k:v-before.get(k,0) for k,v in dispatch.items() if k!='backend' and v-before.get(k,0)}
        tensors=dict(features=features,expansion=module.mlp.expansion,contraction=module.mlp.contraction,
            mlp_scale=module.mlp.skip_scale,qkv=module.attention.front.qkv,q_scale=module.attention.front.scale,
            order=module.attention.pixel_order,bias=module.attention.bias,projection=module.output_weight,
            skip_scale=module.skip_scale,output=value)
        size=sum(t.numel()*t.element_size() for t in tensors.values())
        assert report['raw_bytes']+size<=report['maximum_raw_bytes']
        meta={}
        for n,t in tensors.items():
            a=t.cpu().numpy();assert np.isfinite(a).all();meta[n]=arrays.save(a)
        report['raw_bytes']+=size
        report['cases'].append(dict(name=name,shape=list(features.shape),shift=list(module.window_shift),
            input_stride=list(features.stride()),output_stride=list(value.stride()),arrays=meta,dispatch=delta))
        print(json.dumps(dict(name=name,shape=list(features.shape),shift=list(module.window_shift),dispatch=delta)),flush=True)
        return value
    for name,module in modules.items():
        assert 'forward_unquantized' not in module.__dict__
        original=module.forward_unquantized
        replacement=lambda x,n=name,m=module,f=original:capture(n,m,f,x)
        module.forward_unquantized=replacement;overrides.append((module,replacement))
    components=[FusedBatched(model,provider,workload='small'),FusedSplit(model,provider),FusedC32(model,provider),
                FusedVitProjection(model,provider),HeadLayout(model,provider),FusedSwin(provider)]
    with torch.inference_mode(),provider.installed(),body.installed(),ExitStack() as stack:
        for c in components:stack.enter_context(c.installed())
        with use_arithmetic_backend('triton') as dispatch:
            value=body.forward_front(model,**inputs,sigmoid=model.sigmoid,blend_scale=model.blend_scale,return_float32=False)
    assert raw(value)==arrays.load(report['expected_output']).tobytes()
    assert dict(dispatch)==records['body']['dispatch']
    assert seen==set(modules) and len(seen)==7
    assert constant.require() is lut and raw(lut)==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
    assert all(raw(t)==arrays.load(report['body_inputs'][n]).tobytes() for n,t in inputs.items())
    report.update(passed=True,entire_body_byte_equal=True,all_modules_captured=True,
        inputs_and_lut_unchanged=True,dispatch=dict(dispatch))
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for m,f in reversed(overrides):assert m.forward_unquantized is f;del m.forward_unquantized
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']),raw_bytes=report['raw_bytes'])),flush=True)
