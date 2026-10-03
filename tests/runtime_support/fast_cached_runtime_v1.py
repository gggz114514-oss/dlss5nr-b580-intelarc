"""Reuse reviewed fast kernel artifacts; never compile a missing kernel in inference."""
import hashlib,json,os,sys
from pathlib import Path
FAST=Path(__file__).resolve().parent
EXACT=FAST/'exact'
D=FAST/'data'
CACHE=D/'fast-cache'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bootstrap():
    sys.path[:0]=[str(FAST/'toolchain'),
        str(FAST/'fast/backend'),str(FAST/'modules')]
    os.environ['TRITON_CACHE_DIR']=str(CACHE)
    from portable_cache import install_helpers
    install_helpers()

class DiskOnly:
    def __init__(self):
        self.rows={};self.changed={};self.hits=0;self.pending=None
    def __enter__(self):
        from triton import knobs
        from triton.runtime import driver
        from triton.compiler.compiler import make_backend
        from triton.runtime.cache import get_cache_key,get_cache_manager
        from triton._C.libtriton import get_cache_invalidating_env_vars
        assert not knobs.compilation.always_compile and not knobs.compilation.override
        self.old_hook=knobs.runtime.jit_cache_hook
        def hook(**args):
            fn=args['fn'].jit_function
            spec=args['compile']['specialization_data']
            identity=hashlib.sha256(spec.encode()).hexdigest()
            self.pending=dict(id=identity,module=fn.fn.__module__,qualname=fn.fn.__qualname__,
                              specialization=spec,source_hash=fn.cache_key)
            if fn not in self.changed:
                original=fn.compile;self.changed[fn]=original
                def disk_only(src,target=None,options=None,**kw):
                    target=target or driver.active.get_current_target()
                    from portable_cache import fast_cache_options
                    backend=make_backend(target);parsed=backend.parse_options(fast_cache_options(options))
                    env=kw.get('_env_vars') or get_cache_invalidating_env_vars()
                    hashed=hashlib.sha256(get_cache_key(src,backend,parsed,env).encode()).hexdigest()
                    group=get_cache_manager(hashed).get_group(src.name[:150]+'.json')
                    if not group:raise RuntimeError('Missing fast artifact; prepare offline before video processing: '+src.name)
                    kernel=original(src,target=target,options=parsed.__dict__,**kw)
                    self.hits+=1
                    self.rows[self.pending['id']]=dict(self.pending,files={str(Path(p).resolve()):sha(p) for p in kernel.metadata_group.values()})
                    return kernel
                fn.compile=disk_only
            return False
        knobs.runtime.jit_cache_hook=hook
        return self
    def __exit__(self,*args):
        from triton import knobs
        knobs.runtime.jit_cache_hook=self.old_hook
        for fn,original in self.changed.items():fn.compile=original

