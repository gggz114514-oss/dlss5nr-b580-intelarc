"""Shape-only collection of the diagnostic adapter, then offline compilation.

Fake resource/scalar values exist exclusively during collection. The numerical
runner retains real resource checks, cache-only loading and byte acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

BASE=Path(__file__).resolve().parents[2]
PRODUCT=BASE/'nr-b580-int8/product'
sys.path.insert(0,str(PRODUCT/'precompile'))
import exact_kernel_package_v2 as package


def source_manifest(modules):
    sources={};virtual=[]
    for module in modules:
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename)
        # Torch creates modules with names such as '_ops.py', not disk files.
        # Never interpret those relative labels against the project directory.
        if not path.is_absolute():
            virtual.append(dict(module=getattr(module,'__name__',''),file=filename))
            continue
        path=path.resolve()
        if path.suffix!='.py' or not path.is_relative_to(BASE):continue
        if not path.is_file():raise FileNotFoundError(f'Loaded project source missing: {path}')
        sources[str(path)]=package.sha(path)
    if not sources:raise RuntimeError('Empty project source manifest')
    return sources,virtual


def worker_init(cache):
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.path.insert(0,str(PRODUCT/'comfy'))
    from runtime_environment import isolate
    global handles
    handles=isolate()
    from fast_cached_runtime_v1 import bootstrap
    bootstrap()
    os.environ['TRITON_CACHE_DIR']=str(cache)


def collect(stack,installed,out):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode,FakeTensor
    from triton.runtime.jit import JITFunction,compute_cache_key,serialize_specialization_data
    from triton.runtime import driver
    from triton import knobs
    from nr_backend.execution import use_arithmetic_backend
    target=driver.active.get_current_target()
    rows={};stubs={}
    original=(JITFunction.run,FakeTensor.__bool__,torch.xpu.synchronize)
    class Pointer:
        def __init__(self,t):self.dtype=t.dtype;self.ptr=256+int(t.storage_offset())*t.element_size()
        def data_ptr(self):return self.ptr
    def capture(fn,*args,grid,warmup=False,**kwargs):
        proxy=lambda v:Pointer(v) if isinstance(v,torch.Tensor) else v
        # Real constants may be read during fake tracing; this hook never launches.
        args=tuple(proxy(v) for v in args);kwargs={k:proxy(v) for k,v in kwargs.items()}
        kwargs['debug']=kwargs.get('debug',fn.debug) or knobs.runtime.debug
        kwargs['instrumentation_mode']=knobs.compilation.instrumentation_mode
        _,keys,tgt,backend,binder=fn.device_caches[driver.active.get_current_device()]
        bound,spec,options=binder(*args,**kwargs)
        key=compute_cache_key(keys,spec,options)
        options,signature,constants,attrs=fn._pack_args(backend,kwargs,bound,spec,options)
        data=serialize_specialization_data(f'{fn.fn.__module__}.{fn.fn.__qualname__}',signature,constants,attrs,options,key,tgt)
        identity=hashlib.sha256(data.encode()).hexdigest()
        rows[identity]=dict(id=identity,module=fn.fn.__module__,qualname=fn.fn.__qualname__,specialization=data,source_hash=fn.cache_key)
        if identity not in stubs:
            stubs[identity]=SimpleNamespace(hash=identity,n_spills=0,n_regs=0,metadata=SimpleNamespace(shared=0),_init_handles=lambda:None)
        return stubs[identity]
    def fake_bool(t):
        if t.numel()!=1 or t.dtype!=torch.bool:raise RuntimeError('Unexpected fake scalar dependency')
        return True
    cases=[]
    try:
        with installed():
            JITFunction.run,FakeTensor.__bool__,torch.xpu.synchronize=capture,fake_bool,lambda *a,**kw:None
            with FakeTensorMode(allow_non_fake_inputs=True),torch.inference_mode(),use_arithmetic_backend('triton'):
                for h,w in ((256,256),(480,864)):
                    stack.model.reset()
                    rgb=torch.empty((h,w,3),device='xpu');motion=torch.empty((h,w,2),device='xpu')
                    for reset in (True,False):
                        result=stack.model(rgb,motion,reset=reset)
                        assert isinstance(result,FakeTensor) and result.shape==(h,w,3)
                        cases.append(dict(size=[h,w],reset=reset,kernels=len(rows)))
                        print(json.dumps(cases[-1]),flush=True)
            JITFunction.run,FakeTensor.__bool__,torch.xpu.synchronize=original
    finally:
        JITFunction.run,FakeTensor.__bool__,torch.xpu.synchronize=original
        stack.model.reset()
    # Preserve collected signatures before producing auxiliary source metadata.
    package.write(out/'specializations.json',dict(kernels=list(rows.values()),cases=cases,gpu_dispatches=0))
    sources,virtual=source_manifest(list(sys.modules.values()))
    # 'exact' is the existing compiler catalog schema selector, not a numerical claim.
    package.write(out/'catalog.json',dict(mode='exact',semantic_mode='fast-diagnostic',gpu_dispatches=0,
        source_files=sources,virtual_modules=virtual,kernels=list(rows.values()),target=target.__dict__))
    return dict(passed=True,shape_only=True,numerical_validation=False,gpu_dispatches=0,cases=cases)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();package.compile_worker_init=worker_init
    package.build(a.catalog,Path('D:/Codex-NR-Experiments/nr-b580/reference/triton-cache-c32-triton38-v1'),a.out,14)
