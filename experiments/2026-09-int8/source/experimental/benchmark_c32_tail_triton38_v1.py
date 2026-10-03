"""C32 attention/projection tail fusion against current FP16 complete block outputs.

Only encoder.0.0 at160x160 with an isolated newer compiler; initial speed screen. Both implementations compile with that runtime and must match the frozen3.7.2 output.
Timing is graph replay with completion and an owned full-output copy, not a
complete NR inference. Output byte equality is required before timing a kernel.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from contextlib import ExitStack
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/c32-tail-triton38-screen-v1'
assert not OUT.exists();os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(EXACT/'reference'),str(ROOT/'backend')]
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
sys.path.insert(0,str(TOOLCHAIN/'site'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
provision=js(TOOLCHAIN/'provision-v1.json')
assert provision['wheel_sha256']=='c363a2c6e5b0450015a18237fe58d830ab1d80f995947a470917f87ae26b37d8'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in provision['files'].items())
p=D/'experimental/c32-window-blocks-v2/validation.json'
assert sha(p)=='b99326949733fe839431341cc743ea5903abda8f95ef3ca795b23ed0f2ca08cd'
prior=js(p);assert prior['passed'] and js(p.parent.with_suffix('.log.lease.json'))['returncode']==0
sources=dict(prior['sources']);sources[str(p)]=sha(p)
for p in [Path(__file__),HERE/'Run-C32TailTriton38V1.cmd',HERE/'fused_c32_attention_projection_v1.py',HERE/'prepare_triton38_isolated_v1.py',TOOLCHAIN/'provision-v1.json',TOOLCHAIN/'site/triton/_C/libtriton.pyd',TOOLCHAIN/'site/triton/backends/intel/compiler.py']:sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
assert triton.__version__.startswith('3.8.0')
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from fused_c32_mlp_lut_v1 import FusedC32
from cubic_lut_constant_v1 import register,Constant,TABLE
from native_half_head_layout_v1 import HeadLayout
from fused_swin_native_half_v1 import FusedSwin
from fused_c32_attention_projection_v1 import forward
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,cases=[],complete_migration=False)
report['runtime']=dict(triton_version=triton.__version__,triton_file=triton.__file__,torch_version=torch.__version__,isolated=True,default_compiler_pipeline=True)
graphs=[];raw=lambda t:t.cpu().numpy().tobytes()
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


def time_functions(functions,expected):
    pool=torch.xpu.graph_pool_handle();stream=torch.xpu.Stream();entries=[]
    for fn in functions:
        with stream:
            for _ in range(2):warm=fn()
        torch.xpu.synchronize();owned=torch.empty_like(warm);del warm
        g=torch.xpu.XPUGraph()
        with torch.xpu.graph(g,stream=stream,pool=pool):
            temporary=fn();owned.copy_(temporary)
        del temporary
        assert not any(s['address']<=owned.data_ptr()<s['address']+s['total_size'] for s in torch.xpu.memory_snapshot(pool))
        graphs.append(g);entries.append((g,owned))
    samples=[[] for _ in entries];orders=[]
    for r in range(5):
        order=list(range(len(entries)));i=r%len(order);order=order[i:]+order[:i];orders.append(order)
        for i in order:
            g,owned=entries[i];torch.xpu.synchronize();started=time.perf_counter()
            for _ in range(10):g.replay()
            torch.xpu.synchronize();samples[i].append((time.perf_counter()-started)/10)
            assert raw(owned)==expected
    for g,_ in entries:g.reset();graphs.remove(g)
    return dict(samples_seconds=samples,median_seconds=[statistics.median(s) for s in samples],orders=orders)


try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut=register(model);constant=Constant(model);provider=StridedMatrices();provider.select('fp16_xmx');modules=dict(model.named_modules())
    cases=[dict(c,synthetic=False) for c in prior['cases'][:1]]
    for i,(h,w) in enumerate([]):
        c=prior['cases'][i];meta=dict(c['arrays']);a=arrays.load(meta['features'])
        meta['features']=arrays.save(np.tile(a,((h+159)//160,(w+159)//160,1))[:h,:w].copy());meta.pop('output')
        cases.append(dict(c,name=c['name']+f'/{h}x{w}',module_name=c['name'],arrays=meta,shape=[h,w,32],synthetic=True))
    with torch.inference_mode(),provider.installed(),body.installed(),ExitStack() as stack:
        for c in [FusedC32(model,provider),HeadLayout(model,provider),FusedSwin(provider)]:stack.enter_context(c.installed())
        for c in cases:
            module=modules[c.get('module_name',c['name'])];meta=c['arrays']
            tensors={n:torch.from_numpy(arrays.load(m)).to('xpu') for n,m in meta.items() if n!='output'}
            x=tensors['features']
            with use_arithmetic_backend('triton') as dispatch:value=module.forward_unquantized(x)
            expected=raw(value)
            if not c['synthetic']:
                assert expected==arrays.load(meta['output']).tobytes()
                assert {k:v for k,v in dispatch.items() if k!='backend' and v}==c['dispatch']
            else:meta['output']=arrays.save(value.cpu().numpy())
            args=[tensors[n] for n in ('features','expansion','contraction','mlp_scale','qkv','q_scale','order','bias','projection','skip_scale')]+[lut]
            row=dict(name=c['name'],synthetic=c['synthetic'],arrays=meta,shape=c['shape'],shift=c['shift'],baseline_dispatch=dict(dispatch),candidates=[])
            report['cases'].append(row)
            functions=[];labels=['previous']
            def baseline():
                with use_arithmetic_backend('triton'):return module.forward_unquantized(x)
            functions.append(baseline)
            for warps,stages in ((4,1),(8,1)):
                value,kernel=forward(*args,shift=c['shift'],warps=warps,stages=stages)
                actual=raw(value);a=value.cpu().numpy();e=np.frombuffer(expected,dtype=np.float16).reshape(a.shape)
                equal=actual==expected
                candidate=dict(warps=warps,stages=stages,byte_equal=equal,
                    different_half_words=int(np.count_nonzero(a.view('u2')!=e.view('u2'))),
                    max_abs=float(np.max(np.abs(a.astype('f4')-e.astype('f4')))),spills=[k.n_spills for k in kernel],
                    output_sha256=hashlib.sha256(actual).hexdigest(),
                    ir=[{k:artifacts.text(v,k) for k,v in part.asm.items() if isinstance(v,str) and k in ('ttgir','llir')} for part in kernel])
                row['candidates'].append(candidate);save()
                print(json.dumps(dict(name=c['name'],warps=warps,stages=stages,byte_equal=equal,different=candidate['different_half_words'],spills=[k.n_spills for k in kernel])),flush=True)
                if equal:
                    functions.append(lambda warps=warps,stages=stages:forward(*args,shift=c['shift'],warps=warps,stages=stages)[0])
                    labels.append(f'w{warps}s{stages}')
            if len(functions)>1:
                row['timing']=time_functions(functions,expected);row['timing']['labels']=labels
                print(json.dumps(dict(name=c['name'],labels=labels,median_ms=[s*1000 for s in row['timing']['median_seconds']])),flush=True)
            assert all(raw(t)==arrays.load(meta[n]).tobytes() for n,t in tensors.items());save()
        assert constant.require() is lut and raw(lut)==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        report.update(passed=all(c['byte_equal'] for r in report['cases'] for c in r['candidates']),inputs_and_lut_unchanged=True)
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for g in graphs:g.reset()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
assert report['passed']
print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']))),flush=True)
