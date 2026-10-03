"""Collect real pre/post K8 matrices, then compare exact integer launch layouts.

The complete original pre/post stages must match their authenticated full output.
Only the existing exact K8 tiled kernel is used; no FP16 approximation is added.
Standalone replay timings include a complete owned output copy and completion.
They are diagnostics, not full NR speed. Finite actual frame181 operands only.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from contextlib import ExitStack
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/k8-tiling38-v1';assert not OUT.exists()
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(TOOLCHAIN/'site'),str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main();prior_path=D/'experimental/triton38-body-stages-v1/validation.json'
assert sha(prior_path)=='619714ceafb097336fcd8e8a69b7d8de1e560d0bac42ff947da5af0076895ff2'
prior=js(prior_path);assert prior['passed'] and js(prior_path.parent.with_suffix('.log.lease.json'))['returncode']==0
sources=dict(prior['sources']);sources[str(prior_path)]=sha(prior_path)
provision_path=TOOLCHAIN/'provision-v1.json'
assert sha(provision_path)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision_path)['files'].items())
for p in (Path(__file__),HERE/'Run-K8Tiling38V1.cmd'):sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site') and triton.__version__.startswith('3.8.0')
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from nr_backend.triton_math import _tiled_dot,fused_dot
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
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,cases=[],stage_checks=[],complete_migration=False)
raw=lambda t:t.cpu().numpy().tobytes();graphs=[];captures=[]
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
class Collector(StridedMatrices):
    def dense(self,a,w,*,chunk_k,initial=None,**kwargs):
        out=super().dense(a,w,chunk_k=chunk_k,initial=initial,**kwargs)
        if chunk_k==8:
            values=dict(a=a,w=w,initial=initial,output=out)
            meta={n:None if t is None else arrays.save(t.cpu().numpy()) for n,t in values.items()}
            row=dict(name=stage_name,arrays=meta,shape=[a.numel()//w.shape[0],*w.shape],candidates=[])
            report['cases'].append(row);captures.append((row,{n:None if t is None else t.clone() for n,t in values.items()}))
        return out
def benchmark(functions,expected):
    stream=torch.xpu.Stream();pool=torch.xpu.graph_pool_handle();entries=[]
    for label,fn in functions:
        with stream:
            for _ in range(2):warm=fn()
        torch.xpu.synchronize();owned=torch.empty_like(warm);del warm;g=torch.xpu.XPUGraph()
        with torch.xpu.graph(g,stream=stream,pool=pool):temporary=fn();owned.copy_(temporary)
        del temporary;graphs.append(g)
        assert not any(s['address']<=owned.data_ptr()<s['address']+s['total_size'] for s in torch.xpu.memory_snapshot(pool))
        entries.append((g,owned))
    samples=[[] for _ in entries];orders=[]
    for r in range(5):
        order=list(range(len(entries)));order=order[r%len(order):]+order[:r%len(order)];orders.append(order)
        for i in order:
            g,out=entries[i];torch.xpu.synchronize();started=time.perf_counter()
            for _ in range(10):g.replay()
            torch.xpu.synchronize();samples[i].append((time.perf_counter()-started)/10)
            assert raw(out)==expected
    for g,_ in entries:g.reset();graphs.remove(g)
    return dict(labels=[l for l,_ in functions],samples_seconds=samples,median_seconds=[statistics.median(s) for s in samples],
        orders=orders,graph_replays_per_variant=50,full_output_readbacks_per_variant=5)
try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut=register(model);constant=Constant(model);provider=Collector();provider.select('fp16_xmx')
    components=[FusedBatched(model,provider,workload='small'),FusedSplit(model,provider),FusedC32(model,provider),
        FusedVitProjection(model,provider),HeadLayout(model,provider),FusedSwin(provider)]
    with torch.inference_mode(),provider.installed(),body.installed(),ExitStack() as stack:
        for c in components:stack.enter_context(c.installed())
        for stage_name in ('pre','post'):
            row=next(s for s in prior['stages'] if s['name']==stage_name)
            args=[None if m is None else torch.from_numpy(arrays.load(m)).to('xpu') for m in row['inputs']]
            with use_arithmetic_backend('triton'):
                if stage_name=='pre':result=model.pre.forward_features_outputs(*args)
                else:
                    features,skip,rgb,previous,reciprocal=args
                    result=(model.post(features,skip,rgb,previous=previous,history_reciprocal=reciprocal,
                        sigmoid=model.sigmoid,blend_scale=model.blend_scale,return_float32=False),)
            assert [raw(t) for t in result]==[arrays.load(m).tobytes() for m in row['outputs']]
            report['stage_checks'].append(dict(name=stage_name,complete_outputs_byte_equal=True))
            print('Captured verified '+stage_name,flush=True)
    assert len(captures)==2 and [c['name'] for c,_ in captures]==['pre','post']
    with torch.inference_mode():
        for row,tensors in captures:
            a,w,initial=tensors['a'],tensors['w'],tensors['initial'];expected=raw(tensors['output'])
            functions=[('previous',lambda:fused_dot(a,w,chunk_k=8,initial=initial))]
            assert raw(functions[0][1]())==expected
            configs=([(2,32,1),(4,32,1),(8,32,1),(8,32,4)] if row['name']=='pre'
                else [(4,8,1),(8,8,1),(16,8,1),(16,8,4)])
            for bm,bn,warps in configs:
                fn=lambda bm=bm,bn=bn,warps=warps:_tiled_dot(a,w,chunk_k=8,initial=initial,bm=bm,bn=bn,warps=warps)
                value=fn().cpu().numpy();target=tensors['output'].cpu().numpy();equal=value.tobytes()==expected
                label=f'bm{bm}-bn{bn}-w{warps}'
                row['candidates'].append(dict(label=label,byte_equal=equal,different_half_words=int(np.count_nonzero(value.view('u2')!=target.view('u2')))))
                save();assert equal,(row['name'],label)
                functions.append((label,fn))
            row['timing']=benchmark(functions,expected);save()
            assert all(t is None or raw(t)==arrays.load(row['arrays'][n]).tobytes() for n,t in tensors.items())
            print(json.dumps(dict(name=row['name'],median_ms=dict(zip(row['timing']['labels'],[s*1000 for s in row['timing']['median_seconds']])))),flush=True)
        assert constant.require() is lut and raw(lut)==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        assert model._previous is None and model.next_seed==0
        report.update(passed=True,inputs_and_lut_unchanged=True,history_not_advanced=True)
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for g in graphs:g.reset()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'])),flush=True)
