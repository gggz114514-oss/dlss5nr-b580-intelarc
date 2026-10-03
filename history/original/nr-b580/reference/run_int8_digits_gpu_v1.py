"""Bounded K16 INT8 experiment: 14-worker compile, then exact local A/B."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

BASE=Path(__file__).resolve().parents[2]
PRODUCT=BASE/'nr-b580-int8/product'
DATA=Path('D:/Codex-NR-Experiments/nr-b580')
CACHE=DATA/'reference/triton-cache-c32-triton38-v1'
from precompile_reuse_v2 import worker_init,package


def collect(path,cases,sources):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode,FakeTensor
    from triton.runtime.jit import JITFunction,compute_cache_key,serialize_specialization_data
    from triton.runtime import driver
    from triton import knobs
    from int8_digits_gpu_v1.kernel import forward
    from int8_digits_gpu_v1.fixtures import pack
    from nr_backend.triton_math import _tiled_dot
    rows={}
    class Pointer:
        def __init__(self,t):self.dtype=t.dtype;self.ptr=256+int(t.storage_offset())*t.element_size()
        def data_ptr(self):return self.ptr
    original=JITFunction.run
    def capture(fn,*args,grid,warmup=False,**kwargs):
        assert all(not isinstance(v,torch.Tensor) or isinstance(v,FakeTensor) for v in args)
        args=tuple(Pointer(v) if isinstance(v,FakeTensor) else v for v in args)
        kwargs['debug']=kwargs.get('debug',fn.debug) or knobs.runtime.debug
        kwargs['instrumentation_mode']=knobs.compilation.instrumentation_mode
        _,keys,target,backend,binder=fn.device_caches[driver.active.get_current_device()]
        bound,specialization,options=binder(*args,**kwargs)
        key=compute_cache_key(keys,specialization,options)
        options,signature,constants,attrs=fn._pack_args(backend,kwargs,bound,specialization,options)
        data=serialize_specialization_data(f'{fn.fn.__module__}.{fn.fn.__qualname__}',signature,constants,attrs,options,key,target)
        identity=hashlib.sha256(data.encode()).hexdigest()
        rows[identity]=dict(id=identity,module=fn.fn.__module__,qualname=fn.fn.__qualname__,specialization=data,source_hash=fn.cache_key)
    try:
        JITFunction.run=capture
        for name,a,w,i in cases:
            arrays=(a,w,*pack(w),i)
            with FakeTensorMode():
                tensors=[torch.empty(v.shape,dtype=torch.from_numpy(v).dtype,device='xpu') for v in arrays]
                _tiled_dot(tensors[0],tensors[1],initial=tensors[-1])
                for debug in (False,True):forward(*tensors,debug=debug)
    finally:JITFunction.run=original
    package.write(path,dict(mode='exact',gpu_dispatches=0,shape_only=True,numerical_validation=False,
                           source_files=sources,kernels=list(rows.values()),target=driver.active.get_current_target().__dict__))


def test(out,cases,sources):
    import numpy as np
    import torch
    from int8_digits_gpu_v1.kernel import forward
    from int8_digits_gpu_v1.fixtures import pack
    from nr_backend.triton_math import _tiled_dot
    from triton.runtime.cache import FileCacheManager
    report=dict(passed=False,scope='Real archived operands; newly recomputed exact K16 baseline. Not full model acceptance.',
                sources=sources,cases=[],dynamic_conversion_included=True,static_weight_pack_excluded=True,per_product_remainder_included=True,four_int8_dots_per_k16=True)
    graphs=[]
    original_put=FileCacheManager.put
    def reject(*a,**kw):raise RuntimeError('Missing compiled specialization; use 14-worker compile phase')
    try:
        FileCacheManager.put=reject
        with torch.inference_mode():
            for name,a,w,i in cases:
                args=[torch.from_numpy(v).to('xpu') for v in (a,w,*pack(w),i)]
                expected=_tiled_dot(args[0],args[1],initial=args[-1])
                actual,flags,kernel=forward(*args,debug=True)
                raw=lambda t:t.cpu().contiguous().numpy().tobytes()
                identical=raw(actual)==raw(expected)
                fast=int(flags.cpu().sum().item());groups=flags.numel()
                row=dict(name=name,shape=[a.shape[0],a.shape[1],w.shape[1]],byte_equal=identical,
                         int8_groups=fast,total_groups=groups,hit_percent=100*fast/groups,samples=[],
                         operand_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in (a,w,i)],
                         debug_resources=dict(spills=kernel.n_spills,registers=kernel.n_regs))
                report['cases'].append(row)
                if not identical:raise AssertionError(f'{name}: exact K16 byte mismatch')
                if fast!=groups:raise AssertionError('Digit kernel did not mark every group')
                # Archive compiler IR for confirming that the intended integer
                # matrix instruction was selected; never infer this from dtype.
                for kind,ir in kernel.asm.items():
                    if kind in ('llir','ttgir') and isinstance(ir,str):
                        path=out/(hashlib.sha256(name.encode()).hexdigest()[:12]+'.'+kind)
                        path.write_text(ir,encoding='utf-8')
                stream,pool=torch.xpu.Stream(),torch.xpu.graph_pool_handle()
                def capture(fn):
                    with stream:
                        for _ in range(2):warm=fn()
                    torch.xpu.synchronize()
                    output=torch.empty_like(warm)
                    g=torch.xpu.XPUGraph();graphs.append(g)
                    with torch.xpu.graph(g,stream=stream,pool=pool):output.copy_(fn())
                    g.replay();torch.xpu.synchronize()
                    assert raw(output)==raw(expected)
                    return g,output
                workloads=[capture(lambda:_tiled_dot(args[0],args[1],initial=args[-1])),capture(lambda:forward(*args,debug=False)[0])]
                for iteration in range(8):
                    times={}
                    for index in ((0,1) if iteration%2==0 else (1,0)):
                        g,output=workloads[index]
                        torch.xpu.synchronize();start=time.perf_counter()
                        for _ in range(10):g.replay()
                        torch.xpu.synchronize()
                        times[index]=(time.perf_counter()-start)*100
                        assert raw(output)==raw(expected)
                    row['samples'].append(dict(warmup=iteration<2,baseline_ms=times[0],candidate_ms=times[1]))
                row['baseline_ms']=statistics.median(s['baseline_ms'] for s in row['samples'][2:])
                row['candidate_ms']=statistics.median(s['candidate_ms'] for s in row['samples'][2:])
                row['reduction_percent']=(1-row['candidate_ms']/row['baseline_ms'])*100
                for g,_ in workloads:g.reset();graphs.remove(g)
                print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
        if any(package.sha(path)!=digest for path,digest in sources.items()):raise RuntimeError('Sources changed')
        report['passed']=True
    except BaseException:
        report['error']=traceback.format_exc()
        raise
    finally:
        for g in graphs:g.reset()
        FileCacheManager.put=original_put
        (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['compile','test'],required=True)
    p.add_argument('--workers',type=int,choices=[14],default=14)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    worker_init(CACHE)
    from int8_digits_gpu_v1.fixtures import cases,REPORT
    sources={str(p):package.sha(p) for p in [Path(__file__),REPORT,PRODUCT/'int8_exact_probe_v1/fixtures.py',BASE/'nr-b580/backend/nr_backend/triton_math.py',*list((PRODUCT/'int8_digits_gpu_v1').glob('*.py'))]}
    fixtures=cases()
    if args.phase=='compile':
        catalog=args.out/'catalog.json'
        collect(catalog,fixtures,sources)
        package.build(catalog,CACHE,args.out/'package.json',args.workers)
    else:test(args.out,fixtures,sources)


if __name__=='__main__':main()
