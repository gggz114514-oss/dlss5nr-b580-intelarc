"""Collect exact candidate shapes, then compile with 14 independent workers.

Fake scalar/sync replacements exist only in shape collection. Never validates
numerics, never changes real inference guards, and never dispatches NR kernels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
PRODUCT = BASE / 'nr-b580-int8/product'
DATA = Path('D:/Codex-NR-Experiments/nr-b580')
SITE = DATA / 'reference/toolchains/triton-xpu-3.8.0-git1e2d42a0/site'
sys.path.insert(0, str(PRODUCT / 'precompile'))
import exact_kernel_package_v2 as package


def worker_init(cache):
    # Fourteen compiler processes, each without an extra BLAS thread pool.
    os.environ.update(PYTHONUTF8='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    sys.path.insert(0, str(PRODUCT / 'comfy'))
    from runtime_environment import isolate
    global _handles
    _handles = isolate()
    package.bootstrap(cache)


package.compile_worker_init = worker_init


def collect(path, cache, sizes, variants, include_primitives=False):
    worker_init(cache)
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
    from triton.runtime.jit import JITFunction, compute_cache_key, serialize_specialization_data
    from triton.runtime import driver
    from triton import knobs
    from nr_backend.controlled_temporal import ControlledMotionNR, NRControls
    from nr_backend.execution import use_arithmetic_backend
    from nr_exact_all_candidate_v1 import Session as All
    from nr_exact_controls_candidate_v1 import Session as Baseline
    from reuse_v2.session import Session as New
    from exact_pipeline_v1.swin import SwinScheduling
    target = driver.active.get_current_target()
    source_files = list((BASE / 'nr-b580/backend/nr_backend').glob('*.py'))
    source_files += list(PRODUCT.glob('nr_exact*.py'))
    for folder in ('branched_exact_v1','reuse_v2','exact_pipeline_v1','exact_decoder_v1','exact_shortfp8_v1','exact_cubic_v1','exact_all_cubic_v1'):
        source_files += list((PRODUCT / folder).glob('*.py'))
    sources = {str(p):package.sha(p) for p in source_files}
    original_run, original_bool, original_sync = JITFunction.run, FakeTensor.__bool__, torch.xpu.synchronize
    rows, cases = {}, []
    calls = 0
    class Pointer:
        def __init__(self, tensor):
            self.dtype = tensor.dtype
            self.ptr = 256+int(tensor.storage_offset())*tensor.element_size()
        def data_ptr(self):
            return self.ptr
    def capture(fn, *args, grid, warmup=False, **kwargs):
        nonlocal calls
        assert all(not isinstance(v,torch.Tensor) or isinstance(v,FakeTensor) for v in args)
        def proxy(v):
            return Pointer(v) if isinstance(v,FakeTensor) else v
        args = tuple(proxy(v) for v in args)
        kwargs = {k:proxy(v) for k,v in kwargs.items()}
        kwargs['debug'] = kwargs.get('debug',fn.debug) or knobs.runtime.debug
        kwargs['instrumentation_mode'] = knobs.compilation.instrumentation_mode
        _, key_cache, tgt, backend, binder = fn.device_caches[driver.active.get_current_device()]
        bound, specialization, options = binder(*args,**kwargs)
        key = compute_cache_key(key_cache,specialization,options)
        options,signature,constants,attrs = fn._pack_args(backend,kwargs,bound,specialization,options)
        full = f'{fn.fn.__module__}.{fn.fn.__qualname__}'
        data = serialize_specialization_data(full,signature,constants,attrs,options,key,tgt)
        identity = hashlib.sha256(data.encode()).hexdigest()
        rows[identity] = dict(id=identity,module=fn.fn.__module__,qualname=fn.fn.__qualname__,specialization=data,source_hash=fn.cache_key)
        calls += 1
        return None
    def shape_guard(tensor):
        if tensor.numel()!=1 or tensor.dtype!=torch.bool:
            raise RuntimeError('Unexpected data-dependent shape guard')
        return True
    assets = BASE / 'nr-b580/model-assets'
    # Cover raw-private/style branches as well as default body signatures.
    models = [ControlledMotionNR.from_assets(assets/'sf-v2/WEIGHTS_HT.bin',assets/'noise-sm89-v2',
              assets/'sigmoid-sm89-v1', controls=NRControls(style=style,intensity=intensity),
              style_directory=assets/'style-sm89-v1').eval() for style,intensity in ((0,1),(0,.5),(1,1),(2,1))]
    try:
        JITFunction.run, FakeTensor.__bool__, torch.xpu.synchronize = capture, shape_guard, lambda *a,**kw:None
        with FakeTensorMode(allow_non_fake_inputs=True), torch.inference_mode():
            for variant in variants:
                cls = Baseline if variant=='baseline' else All if variant=='v1' else New
                if variant=='branched':
                    from branched_exact_v1.session import Session as cls
                if cls is not Baseline:
                    cls.enabled = All.enabled - {'graph'}
                if cls is New:
                    cls.variant = frozenset(variant.split(','))
                for model in models:
                    model = model.to('xpu')
                    session = cls(model,torch,use_arithmetic_backend)
                    session.diagnostics = False
                    for h,w in sizes:
                        session.reset()
                        rgb = torch.empty((h,w,3),device='xpu',dtype=torch.float32)
                        motion = torch.empty((h,w,2),device='xpu',dtype=torch.float32)
                        for reset in (True,False):
                            before = calls
                            with SwinScheduling(enabled=True).installed():
                                output = session.process(rgb,motion,reset=reset)
                            assert isinstance(output.color,FakeTensor) and output.color.shape==(h,w,3)
                            cases.append(dict(variant=variant,size=[h,w],style=model.controls.style,
                                              intensity=model.controls.intensity,reset=reset,calls=calls-before))
                        print(json.dumps(dict(phase='collect',variant=variant,size=[h,w],unique=len(rows))),flush=True)
                    session.close()
            if include_primitives:
                if variants==['branched']:
                    from branched_exact_v1.primitive import check
                else:
                    from reuse_v2.primitive import check
                check(shape_only=True)
    finally:
        JITFunction.run,FakeTensor.__bool__,torch.xpu.synchronize = original_run,original_bool,original_sync
    if any(package.sha(p)!=digest for p,digest in sources.items()):
        raise RuntimeError('Candidate sources changed during collection')
    package.write(path,dict(schema=1,mode='exact',shape_only=True,numerical_validation=False,gpu_dispatches=0,
                           target=target.__dict__,source_files=sources,cases=cases,calls=calls,kernels=list(rows.values())))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--sizes',default='256x256')
    p.add_argument('--variants',default='dataflow;compact;dataflow,compact;dataflow,compact,tile')
    p.add_argument('--workers',type=int,default=14,choices=[14])
    p.add_argument('--include-primitives',action='store_true')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    cache=DATA/'reference/triton-cache-c32-triton38-v1'
    sizes=[tuple(map(int,item.split('x'))) for item in args.sizes.split(',')]
    catalog=args.out/'catalog.json'
    collect(catalog,cache,sizes,args.variants.split(';'),args.include_primitives)
    package.build(catalog,cache,args.out/'package.json',args.workers)


if __name__=='__main__':
    main()
