"""Gate the complete current NR256 body under Triton3.8 before adding C32 tails.

The previous and fused variants use identical authenticated temporal frame181
body inputs. They must each match the frozen complete output from Triton3.7.2.
Static body replay is not independent history evolution or complete NR timing.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from contextlib import ExitStack,nullcontext
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/c32-tail-body38-v1'
assert not OUT.exists();os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
sys.path[:0]=[str(TOOLCHAIN/'site'),str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main();sources={};records={}
provision_path=TOOLCHAIN/'provision-v1.json'
assert sha(provision_path)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
provision=js(provision_path)
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in provision['files'].items())
for key,folder,digest in [
    ('primitive','c32-tail-triton38-complete-v1','df5640c358b105b4b90abaaa3556463d4ca55a2173c1e2a513a1f5d656a1a508'),
    ('capture','c32-window-blocks-v2','b99326949733fe839431341cc743ea5903abda8f95ef3ca795b23ed0f2ca08cd')]:
    path=D/'experimental'/folder/'validation.json';assert sha(path)==digest;r=js(path)
    assert r['passed'] and js(path.parent.with_suffix('.log.lease.json'))['returncode']==0
    assert all(sha(p)==h for p,h in r['sources'].items())
    sources.update(r['sources']);sources[str(path)]=digest;records[key]=r
for p in [Path(__file__),HERE/'Run-C32TailBody38V1.cmd',HERE/'c32_tail_adapter_v1.py',provision_path]:sources[str(p)]=sha(p)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site') and triton.__version__.startswith('3.8.0')
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from batched_branched_mlp_v2 import FusedBatched
from fused_split_ffwd_v2 import FusedSplit
from fused_c32_mlp_lut_v1 import FusedC32
from cubic_lut_constant_v1 import register,Constant,TABLE,BUFFER
from fused_vit_projection_v3 import FusedVitProjection
from native_half_head_layout_v1 import HeadLayout
from fused_swin_native_half_v1 import FusedSwin
from c32_tail_adapter_v1 import C32Tail
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,body=[],
    body_inputs=records['capture']['body_inputs'],expected_output=records['capture']['expected_output'],
    expected_dispatch=records['capture']['dispatch'],complete_migration=False,
    runtime=dict(triton_version=triton.__version__,triton_file=triton.__file__,torch_version=torch.__version__,isolated=True))
raw=lambda t:t.cpu().numpy().tobytes();graphs=[]
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


def check(value,label):
    actual=value.cpu().numpy();expected=arrays.load(report['expected_output'])
    equal=actual.tobytes()==expected.tobytes()
    row=dict(name=label,byte_equal=equal,output_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
    if not equal:
        row.update(actual=arrays.save(actual),different_half_words=int(np.count_nonzero(actual.view('u2')!=expected.view('u2'))),
            max_abs=float(np.max(np.abs(actual.astype('f4')-expected.astype('f4')))))
    report['body'].append(row);save();assert equal,(label,row)


def benchmark(functions):
    pool=torch.xpu.graph_pool_handle();stream=torch.xpu.Stream();entries=[]
    expected=arrays.load(report['expected_output']).tobytes()
    for name,fn in functions:
        with stream:
            for _ in range(2):warm=fn()
        torch.xpu.synchronize();owned=torch.empty_like(warm);del warm;g=torch.xpu.XPUGraph()
        with torch.xpu.graph(g,stream=stream,pool=pool):
            temporary=fn();owned.copy_(temporary)
        del temporary
        assert not any(s['address']<=owned.data_ptr()<s['address']+s['total_size'] for s in torch.xpu.memory_snapshot(pool))
        graphs.append(g);entries.append((g,owned));print('Captured complete body '+name,flush=True)
    samples=[[] for _ in entries];orders=[]
    for r in range(5):
        order=[0,1] if r%2==0 else [1,0];orders.append(order)
        for i in order:
            g,owned=entries[i];torch.xpu.synchronize();started=time.perf_counter()
            for _ in range(10):g.replay()
            torch.xpu.synchronize();samples[i].append((time.perf_counter()-started)/10)
            assert raw(owned)==expected
        print(json.dumps(dict(round=r,body_ms=[s[-1]*1000 for s in samples])),flush=True)
    for g,_ in entries:g.reset();graphs.remove(g)
    return dict(labels=[n for n,_ in functions],samples_seconds=samples,
        median_seconds=[statistics.median(s) for s in samples],orders=orders,complete_replay_outputs_verified=100)


try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut=register(model);constant=Constant(model);provider=StridedMatrices();provider.select('fp16_xmx')
    tail=C32Tail(model,provider)
    inputs={n:torch.from_numpy(arrays.load(m)).to('xpu') for n,m in report['body_inputs'].items()}
    components=[FusedBatched(model,provider,workload='small'),FusedSplit(model,provider),FusedC32(model,provider),
        FusedVitProjection(model,provider),HeadLayout(model,provider),FusedSwin(provider)]
    with torch.inference_mode(),provider.installed(),body.installed(),ExitStack() as stack:
        for c in components:stack.enter_context(c.installed())
        print('Cold compile: original complete body with isolated Triton3.8',flush=True)
        with use_arithmetic_backend('triton') as dispatch:
            value=model._forward_front(**inputs,sigmoid=model.sigmoid,blend_scale=model.blend_scale,return_float32=False,
                progress=lambda label:print('Baseline stage complete: '+label,flush=True))
        assert dict(dispatch)==report['expected_dispatch'];check(value,'previous_cold_with_progress')
        def call(enabled):
            with tail.installed() if enabled else nullcontext():
                before=dict(tail.calls)
                with use_arithmetic_backend('triton') as dispatch:
                    value=body.forward_front(model,**inputs,sigmoid=model.sigmoid,blend_scale=model.blend_scale,return_float32=False)
                assert dict(dispatch)==report['expected_dispatch']
                if enabled:assert {k:v-before.get(k,0) for k,v in tail.calls.items()}=={n:1 for n in tail.names.values()}
            assert all('forward_unquantized' not in m.__dict__ for m in tail.modules)
            return value
        check(call(False),'previous');check(call(True),'tail')
        report['tail_modules']=list(tail.names.values());save()
        report['timing']=benchmark([('previous',lambda:call(False)),('tail',lambda:call(True))])
        # Reject overlapping adapters and restore method ownership after errors.
        with tail.installed():
            try:
                with tail.installed():raise AssertionError('Nested adapter accepted')
            except ValueError:pass
        try:
            with tail.installed():raise LookupError('owned scope probe')
        except LookupError:pass
        assert all('forward_unquantized' not in m.__dict__ for m in tail.modules)
        replacement=lut.clone();setattr(model,BUFFER,replacement)
        try:
            try:tail.constant.require();raise AssertionError('Changed constant accepted')
            except RuntimeError:pass
        finally:setattr(model,BUFFER,lut)
        assert constant.require() is lut and tail.constant.require() is lut
        assert raw(lut)==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        assert all(raw(t)==arrays.load(report['body_inputs'][n]).tobytes() for n,t in inputs.items())
        assert model._previous is None and model._next_seed==0
        report.update(passed=True,inputs_and_lut_unchanged=True,adapter_ownership_guards=True,
            constant_replacement_rejected=True,body_did_not_advance_history=True)
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for g in graphs:g.reset()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],body_median_ms=[s*1000 for s in report['timing']['median_seconds']])),flush=True)
