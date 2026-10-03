"""Exact offline compiler receipts and the actual runtime DiskOnly API.

Imports of Triton occur only inside __enter__, called by the explicit GPU child.
No pool is started, and no missing specialization can be compiled in readonly.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
import hashlib
from pathlib import Path
import release_contracts as c
from portable_runtime import report_value

def compiler_value(value):
    """Actual JIT constexpr/type descriptors; never converts runtime counters."""
    kind = type(value).__name__
    if kind == 'dtype': return dict(dtype=str(value))
    if kind == 'constexpr': return dict(constexpr=compiler_value(value.value))
    if kind == 'JITFunction': return dict(jit_function=value.module+':'+value.fn.__qualname__, source_hash=value.cache_key)
    if isinstance(value,dict):
        return report_value({k:compiler_value(v) for k,v in value.items()})
    if isinstance(value,(tuple,list)): return [compiler_value(v) for v in value]
    return report_value(value)

class CachePhase(AbstractContextManager):
    def __init__(self, bootstrap, *, readonly, prepared=None):
        self.bootstrap, self.readonly = bootstrap, readonly
        self.prepared = prepared or {}
        self.changed, self.rows = {}, {}
        self.hits, self.write_attempts = 0, 0
        self.disk = None

    def __enter__(self):
        from triton import knobs
        from triton.runtime import driver, cache
        from triton.compiler.compiler import make_backend
        from triton._C.libtriton import get_cache_invalidating_env_vars
        from portable_cache import fast_cache_options
        c.require(not knobs.compilation.always_compile and not knobs.compilation.override,
                  'forced compiler/override is forbidden')
        c.require(knobs.cache.manager_class in (None, cache.FileCacheManager), 'foreign cache manager')
        self.knobs, self.cache = knobs, cache
        self.old_hook = knobs.runtime.jit_cache_hook
        self.old_put, self.old_group = cache.FileCacheManager.put, cache.FileCacheManager.put_group
        if self.readonly:
            self.disk = self.bootstrap.DiskOnly()
            self.disk.__enter__()
            def deny(*args, **kwargs):
                self.write_attempts += 1
                raise RuntimeError('readonly cache write attempted; explicit offline precompile required')
            cache.FileCacheManager.put = deny
            cache.FileCacheManager.put_group = deny
        delegate = knobs.runtime.jit_cache_hook if self.readonly else None
        def hook(**args):
            # Execution is serial. The real DiskOnly hook wraps each fn.compile;
            # this wrapper records the ACTUAL ASTSource/key/object it receives.
            if delegate is not None:
                delegate(**args)
            jit = args['fn'].jit_function
            if jit not in self.changed:
                original = jit.compile
                self.changed[jit] = original
                def compile_bound(src, target=None, options=None, **kw):
                    target = target or driver.active.get_current_target()
                    parsed = make_backend(target).parse_options(fast_cache_options(options))
                    env = kw.get('_env_vars') or get_cache_invalidating_env_vars()
                    key = hashlib.sha256(cache.get_cache_key(src, make_backend(target), parsed, env).encode()).hexdigest()
                    if self.readonly:
                        c.require(key in self.prepared, 'unprepared compiler key: ' + src.name)
                    kernel = original(src, target=target, options=vars(parsed), **kw)
                    row = dict(compiler_key=key, source_name=src.name,
                        module=jit.fn.__module__, qualname=jit.fn.__qualname__, source_hash=jit.cache_key,
                        loaded_kernel_hash=kernel.hash,
                        loaded_binary_sha256=hashlib.sha256(kernel.kernel).hexdigest(),
                        signature=compiler_value(src.signature), constants=compiler_value(src.constants),
                        options=compiler_value(vars(parsed)), target=compiler_value(target), env=compiler_value(env),
                        metadata_files={str(Path(p).resolve()):c.sha(p) for p in kernel.metadata_group.values()})
                    self.accept_receipt(key, row)
                    return kernel
                jit.compile = compile_bound
            return False
        knobs.runtime.jit_cache_hook = hook
        return self

    def accept_receipt(self, key, row):
        c.require(row['loaded_binary_sha256'] and row['metadata_files'], 'actual compiled binary/group receipt absent')
        if self.readonly:
            c.require(row == self.prepared[key], 'readonly compiler/binary/source receipt drift: ' + key)
        if key in self.rows:
            c.require(self.rows[key] == row, 'same compiler key changed object metadata')
        self.rows[key] = row
        self.hits += 1

    def __exit__(self, *args):
        try:
            for jit, original in self.changed.items():
                jit.compile = original
            if self.disk is not None:
                self.disk.__exit__(*args)
        finally:
            self.knobs.runtime.jit_cache_hook = self.old_hook
            self.cache.FileCacheManager.put, self.cache.FileCacheManager.put_group = self.old_put, self.old_group

    def snapshot(self):
        return dict(policy='actual bootstrap.DiskOnly + exact receipts + write denial' if self.readonly else
                    'offline real compiler calls; no readonly results accepted yet',
                    actual_compiler_keys=self.rows, hits=self.hits,
                    actual_disk_only_hits=None if self.disk is None else self.disk.hits,
                    readonly_write_attempts=self.write_attempts,
                    parallel_compilers=False)
