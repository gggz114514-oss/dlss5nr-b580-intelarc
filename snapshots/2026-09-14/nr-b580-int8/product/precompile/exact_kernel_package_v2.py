"""Separate shape collection, bounded parallel compilation, and cache-only loading.

The collector uses FakeTensor shapes, never numerical outputs. Only this isolated
collection process replaces fake scalar guards and synchronization. Real model
arithmetic and validation files are not modified. Coverage is checked at runtime:
an unlisted specialization is an error, never an implicit compilation.
"""
import argparse
import concurrent.futures as futures
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

FAST=Path(__file__).resolve().parents[2]
EXACT=FAST.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580')
SITE=D/'reference/toolchains/triton-xpu-3.8.0-git1e2d42a0/site'
SIZES=((256,256),(512,512),(864,480),(1920,1080),(2559,1439))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def bootstrap(cache):
    sys.path[:0]=[str(SITE),str(EXACT/'backend'),str(FAST/'product')]
    os.environ['TRITON_CACHE_DIR']=str(Path(cache).resolve())
    import torch
    torch.set_num_threads(1)
    return torch


def write(path,value):
    path=Path(path)
    temp=path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def resolve(row):
    value=importlib.import_module(row['module'])
    for part in row['qualname'].split('.'):
        value=getattr(value,part)
    return value


def collect(out,cache):
    torch=bootstrap(cache)
    from torch._subclasses.fake_tensor import FakeTensorMode,FakeTensor
    from triton.runtime.jit import JITFunction,compute_cache_key,serialize_specialization_data
    from triton.runtime import driver
    from triton import knobs
    from nr_backend.temporal import MotionNR
    from nr_backend.execution import use_arithmetic_backend
    # Device description/host helper initialization only; no NR GPU dispatch.
    target=driver.active.get_current_target()
    backend_sources={str(p):sha(p) for p in (EXACT/'backend/nr_backend').glob('*.py')}
    stage=D/'reference/experimental/attention-weights-stage-v1/manifest.json'
    expected=js(stage)['staged_sources']
    assert {Path(p).name:h for p,h in backend_sources.items()}==expected
    assets=EXACT/'model-assets'
    print('Loading exact model on CPU for shape collection',flush=True)
    model=MotionNR.from_assets(assets/'sf-v2/WEIGHTS_HT.bin',assets/'noise-sm89-v2',assets/'sigmoid-sm89-v1').eval()
    rows={}
    calls=0
    scalar_guards=0
    original_run=JITFunction.run
    original_bool=FakeTensor.__bool__
    original_sync=torch.xpu.synchronize

    class Pointer:
        def __init__(self,t):
            self.dtype=t.dtype
            self._ptr=256+int(t.storage_offset())*t.element_size()
        def data_ptr(self):return self._ptr

    def capture(fn,*args,grid,warmup=False,**kwargs):
        nonlocal calls
        assert all(not isinstance(a,torch.Tensor) or isinstance(a,FakeTensor) for a in args)
        proxy=lambda v:Pointer(v) if isinstance(v,FakeTensor) else v
        args=tuple(proxy(v) for v in args)
        kwargs={k:proxy(v) for k,v in kwargs.items()}
        kwargs['debug']=kwargs.get('debug',fn.debug) or knobs.runtime.debug
        kwargs['instrumentation_mode']=knobs.compilation.instrumentation_mode
        _,key_cache,tgt,backend,binder=fn.device_caches[driver.active.get_current_device()]
        bound,specialization,options=binder(*args,**kwargs)
        key=compute_cache_key(key_cache,specialization,options)
        options,signature,constants,attrs=fn._pack_args(backend,kwargs,bound,specialization,options)
        full=f'{fn.fn.__module__}.{fn.fn.__qualname__}'
        data=serialize_specialization_data(full,signature,constants,attrs,options,key,tgt)
        identity=hashlib.sha256(data.encode()).hexdigest()
        rows[identity]=dict(id=identity,module=fn.fn.__module__,qualname=fn.fn.__qualname__,
            specialization=data,source_hash=fn.cache_key)
        calls+=1
        return None

    def shape_guard(t):
        nonlocal scalar_guards
        assert t.numel()==1 and t.dtype==torch.bool, 'Data-dependent non-boolean scalar in shape trace'
        scalar_guards+=1
        return True

    traces=[]
    try:
        JITFunction.run=capture
        FakeTensor.__bool__=shape_guard
        torch.xpu.synchronize=lambda *a,**kw:None
        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            model=model.to('xpu')
            with torch.inference_mode(),use_arithmetic_backend('triton'):
                for w,h in SIZES:
                    model.reset()
                    rgb=torch.empty((h,w,3),dtype=torch.float32,device='xpu')
                    mv=torch.empty((h,w,2),dtype=torch.float32,device='xpu')
                    for reset in (True,False):
                        before=calls
                        value=model(rgb,mv,reset=reset)
                        assert isinstance(value,FakeTensor) and tuple(value.shape)==(h,w,3)
                        traces.append(dict(width=w,height=h,reset=reset,calls=calls-before))
                        print(json.dumps(dict(phase='collect',width=w,height=h,reset=reset,
                            calls=calls,unique=len(rows))),flush=True)
    finally:
        JITFunction.run=original_run
        FakeTensor.__bool__=original_bool
        torch.xpu.synchronize=original_sync
    assert all(sha(p)==h for p,h in backend_sources.items())
    result=dict(schema=1,mode='exact',shape_only=True,numerical_validation=False,gpu_dispatches=0,
        target=target.__dict__,source_files=backend_sources,collector_sha256=sha(__file__),
        cases=traces,fake_boolean_guards=scalar_guards,calls=calls,kernels=list(rows.values()))
    write(out,result)
    print(json.dumps(dict(phase='collected',unique=len(rows),calls=calls)),flush=True)


def compile_worker_init(cache):
    bootstrap(cache)


def runtime_fingerprint():
    import torch
    from triton.runtime.cache import triton_key
    return dict(python=sys.version,torch=torch.__version__,triton_key=triton_key())


def compile_one(row):
    started=time.perf_counter()
    fn=resolve(row)
    assert fn.cache_key==row['source_hash']
    kernel=fn.preload(row['specialization'])
    # Compilation returns a CompiledKernel. Do not access function/run, which
    # would load/execute it through the device driver.
    group=kernel.metadata_group
    return dict(id=row['id'],module=row['module'],qualname=row['qualname'],
        seconds=time.perf_counter()-started,worker_pid=os.getpid(),
        runtime=runtime_fingerprint(),
        files={str(Path(p).resolve()):sha(p) for p in group.values()})


def build(catalog,cache,out,workers):
    data=js(catalog)
    assert data['mode']=='exact' and data['gpu_dispatches']==0
    assert all(sha(p)==h for p,h in data['source_files'].items())
    assert len({v['id'] for v in data['kernels']})==len(data['kernels'])
    report=dict(passed=False,phase='compiling',catalog=str(Path(catalog).resolve()),
        catalog_sha256=sha(catalog),cache=str(Path(cache).resolve()),workers=workers,
        expected=len(data['kernels']),completed=[],gpu_dispatches=0)
    write(out,report)
    # Processes avoid depending on whether each compiler pass releases the GIL.
    pool=futures.ProcessPoolExecutor(max_workers=workers,initializer=compile_worker_init,initargs=(cache,))
    jobs={pool.submit(compile_one,row):row for row in data['kernels']}
    try:
        for future in futures.as_completed(jobs):
            report['completed'].append(future.result())
            write(out,report)
            print(json.dumps(dict(phase='compiling',completed=len(report['completed']),
                total=report['expected'],last=report['completed'][-1]['qualname'])),flush=True)
    except BaseException as error:
        report.update(phase='failed',error=repr(error))
        write(out,report)
        for future in jobs:future.cancel()
        for process in tuple(pool._processes.values()):
            if process.is_alive():process.terminate()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    pool.shutdown()
    assert len(report['completed'])==report['expected']
    report['runtime']=report['completed'][0]['runtime']
    assert all(row['runtime']==report['runtime'] for row in report['completed'])
    report.update(passed=True,phase='completed')
    write(out,report)


class CacheOnly:
    """Load a completed local package, then prohibit new Triton specializations.

Requires matching backend/toolchain/device and installed host helpers. This is
not yet a portable, cross-driver release installer. Device-driver initialization
may still occur; the guard forbids Triton's source compilation during inference.
"""
    def __init__(self,package):
        self.package=js(package)
        self.old_hook=None
        self.changed_functions=[]
        self.disk_hits=0

    def __enter__(self):
        p=self.package
        assert p['passed'] and p['phase']=='completed'
        assert sha(p['catalog'])==p['catalog_sha256']
        data=js(p['catalog'])
        assert all(sha(path)==h for path,h in data['source_files'].items())
        bootstrap(p['cache'])
        assert runtime_fingerprint()==p['runtime'],'Precompiled package runtime version mismatch'
        from triton import knobs
        from triton.runtime import driver
        from triton.compiler.compiler import make_backend,ASTSource
        from triton.runtime.cache import get_cache_key,get_cache_manager
        from triton._C.libtriton import get_cache_invalidating_env_vars
        assert driver.active.get_current_target().__dict__==data['target']
        for row in p['completed']:
            assert all(sha(path)==h for path,h in row['files'].items())
        def require_disk_hit(**args):
            # All requested specializations must belong to the frozen catalog.
            spec=args['compile']['specialization_data']
            assert hashlib.sha256(spec.encode()).hexdigest() in ids,'Kernel absent from precompiled package'
            # The known artifact files have already been verified; prevent an
            # actual cache miss from invoking a compiler through an altered key.
            return False
        ids={row['id'] for row in data['kernels']}
        self.old_hook=knobs.runtime.jit_cache_hook
        assert not knobs.compilation.always_compile and not knobs.compilation.override
        knobs.runtime.jit_cache_hook=require_disk_hit
        # Install a disk-only compile wrapper while resolving preload entries.
        # A package is incomplete if original compile would not return a cache hit.
        import triton.compiler.compiler as compiler
        original_compile=compiler.compile
        def disk_only(src,target=None,options=None,**kw):
            target=target or driver.active.get_current_target()
            backend=make_backend(target)
            options=backend.parse_options(options or {})
            env=kw.get('_env_vars') or get_cache_invalidating_env_vars()
            key=get_cache_key(src,backend,options,env)
            hashed=hashlib.sha256(key.encode()).hexdigest()
            group=get_cache_manager(hashed).get_group(src.name[:150]+'.json')
            if not group:raise RuntimeError('Precompiled artifact missing; runtime compilation forbidden')
            self.disk_hits+=1
            return original_compile(src,target=target,options=options.__dict__,**kw)
        try:
            for row in data['kernels']:
                fn=resolve(row)
                _=fn.device_caches[driver.active.get_current_device()]
                self.changed_functions.append((fn,fn.compile))
                fn.compile=disk_only
                fn.preload(row['specialization'])
        except BaseException:
            knobs.runtime.jit_cache_hook=self.old_hook
            for fn,previous in reversed(self.changed_functions):fn.compile=previous
            self.changed_functions.clear()
            raise
        def reject_new(**kwargs):
            raise RuntimeError('Unloaded NR kernel configuration; prepare the kernel package before running')
        knobs.runtime.jit_cache_hook=reject_new
        return self

    def __exit__(self,*args):
        from triton import knobs
        knobs.runtime.jit_cache_hook=self.old_hook
        for fn,previous in reversed(self.changed_functions):fn.compile=previous
        self.changed_functions.clear()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('collect','build','verify-load'))
    p.add_argument('--cache',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--catalog')
    p.add_argument('--package')
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    if a.command=='collect':collect(a.out,a.cache)
    elif a.command=='build':
        assert a.catalog and 1<=a.workers<=16
        build(a.catalog,a.cache,a.out,a.workers)
    else:
        assert a.package
        with CacheOnly(a.package) as loaded:
            expected=loaded.package['expected']
            assert loaded.disk_hits==expected
            from triton import knobs
            try:knobs.runtime.jit_cache_hook()
            except RuntimeError as e:assert 'prepare the kernel package' in str(e)
            else:raise AssertionError('Runtime compilation guard inactive')
            write(a.out,dict(passed=True,package_sha256=sha(a.package),
                disk_only_preloads=loaded.disk_hits,runtime_new_kernel_rejected=True,
                gpu_dispatches=0,numerical_validation=False))
        print(json.dumps(dict(phase='loaded_without_compilation',kernels=expected)),flush=True)
