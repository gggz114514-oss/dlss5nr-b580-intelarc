"""Real compiler-key receipts; readonly mode delegates the unmodified runtime DiskOnly.

All Triton imports happen only in __enter__, called by future Luna GPU children.
No cache key is known in advance, no executable is moved/rekeyed and no G cache is copied.
"""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from copy import deepcopy
import contextlib
import hashlib
from importlib.machinery import SourceFileLoader
import json
import os
from pathlib import Path
import sys
from types import CodeType, FunctionType, ModuleType
from threading import RLock, get_ident
import runner_common as c

_CONTEXTMANAGER = contextlib.contextmanager
_CONTEXT_HELPER = next(code for code in _CONTEXTMANAGER.__code__.co_consts
                       if type(code) is CodeType and code.co_name == 'helper')

def json_value(value):
    if value is None or type(value) in (str, bool, int, float): return value
    if is_dataclass(value): return json_value(asdict(value))
    if isinstance(value, (list, tuple)): return [json_value(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value): return {k:json_value(v) for k,v in value.items()}
        return [dict(key=json_value(k), value=json_value(v)) for k,v in sorted(value.items(), key=lambda pair:repr(pair[0]))]
    if hasattr(value, '__dict__'): return json_value(vars(value))
    return str(value)


def constant_value(value):
    # Same normalization used by the pinned jit.serialize_specialization_data.
    kind = value.__class__.__name__
    if kind == 'dtype': return str(value)
    if kind == 'constexpr': return {'constexpr':json_value(value.value)}
    if kind == 'JITFunction': return {'jit_function':value.module+':'+value.fn.__qualname__}
    return json_value(value)


def binding_key(jit, signature, constants, attrs, target, options):
    constants = {(jit.arg_names.index(k),) if isinstance(k,str) else k:constant_value(v)
                 for k,v in constants.items()}
    value = dict(signature=json_value(signature), constants=json_value(constants), attrs=json_value(attrs or {}),
                 target=json_value(target), options=json_value(options))
    return jit, json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

class CachePolicy:
    def __init__(self, *, precompile, bootstrap, expected_keys=None, source_tree=None, progress=None,
                 cpu_compile_workers=None, compile_memory_budget_gib=32, compile_reserve_gib=8):
        self.precompile, self.bootstrap = precompile, bootstrap
        self.expected_keys = expected_keys or {}
        self.changed, self.rows, self.misses = {}, {}, []
        self.readonly_write_attempts, self.hits = 0, 0
        self.source_tree = None if source_tree is None else Path(source_tree).resolve(strict=True)
        self.progress = progress or (lambda *args, **kwargs: None)
        self.owner_objects = None
        self.lock = RLock(); self.key_locks = {}; self.specializations = {}
        self.owner_thread = get_ident(); self.compile_session = None
        self.cpu_compile_workers = int(os.environ.get('B580_CPU_COMPILE_WORKERS','8')) if cpu_compile_workers is None else cpu_compile_workers
        self.compile_memory_budget_gib, self.compile_reserve_gib = compile_memory_budget_gib, compile_reserve_gib
        self.active_compilers, self.peak_active_compilers = 0, 0
        # PHASE1 uses its complete immutable 225-file manifest, plus only the
        # frozen runtime sources/assets needed by this pinned v4 environment.
        self.allowed_files = c.allowed_source_map()

    def source_identity(self, jit):
        fn = jit.fn
        path = str(Path(fn.__code__.co_filename).resolve(strict=True))
        module = sys.modules.get(fn.__module__)
        c.require(path in self.allowed_files and c.sha(path) == self.allowed_files[path], 'JIT source not frozen: ' + path)
        registered = (module is not None and fn.__globals__ is vars(module) and
                      str(Path(module.__file__).resolve()) == path)
        if registered:
            return dict(path=path, sha256=self.allowed_files[path], module=fn.__module__, qualname=fn.__qualname__, source_hash=jit.cache_key)
        link = self.k8_owner_link(jit, path)
        return dict(path=path, sha256=self.allowed_files[path], module=fn.__module__, qualname=fn.__qualname__,
                    source_hash=jit.cache_key, owner_link=link)

    def k8_owner_link(self, jit, path):
        """Only the frozen combo's existing unregistered attention object."""
        fn = jit.fn
        name = 'post_attention_fusion_combo_local_v1'
        c.require(self.source_tree == c.source_root('PARENT_OFF').resolve() and
                  fn.__module__ == name and fn.__qualname__ == '_attention_project', 'JIT is a cloned/foreign source owner')
        parent = sys.modules.get('post_attention_k8_combined_v1')
        c.require(type(parent) is ModuleType and parent.__name__ == 'post_attention_k8_combined_v1', 'Missing registered k8 combo owner')
        parent_path = self.source_tree / 'game/post_attention_k8_combined_v1.py'
        child_path = parent_path.with_name('post_attention_fusion_v1.py')
        c.require(Path(parent.__file__).resolve() == parent_path and str(parent_path) in self.allowed_files and
                  c.sha(parent_path) == self.allowed_files[str(parent_path)] and Path(path) == child_path,
                  'Foreign k8 combo source tree')
        wrapper = parent.installed
        original = getattr(wrapper, '__wrapped__', None)
        c.require(contextlib is sys.modules.get('contextlib') and contextlib.contextmanager is _CONTEXTMANAGER and
                  _CONTEXTMANAGER.__globals__ is vars(contextlib) and
                  Path(_CONTEXTMANAGER.__code__.co_filename).resolve() == Path(contextlib.__file__).resolve(),
                  'Foreign stdlib contextmanager')
        c.require(type(wrapper) is FunctionType and wrapper.__code__ is _CONTEXT_HELPER and
                  wrapper.__globals__ is vars(contextlib) and wrapper.__module__ == parent.__name__ and
                  wrapper.__code__.co_freevars == ('func',) and wrapper.__closure__ is not None and
                  len(wrapper.__closure__) == 1 and wrapper.__closure__[0].cell_contents is original,
                  'Foreign k8 contextmanager wrapper/closure')
        c.require(type(original) is FunctionType and original.__module__ == parent.__name__ and
                  original.__name__ == original.__qualname__ == 'installed' and original.__code__.co_flags & 0x20 and
                  original.__globals__ is vars(parent) and Path(original.__code__.co_filename).resolve() == parent_path,
                  'Foreign k8 original generator/factory')
        attention, spec = parent.attention, parent._SPEC
        c.require(type(attention) is ModuleType and sys.modules.get(name) is None and
                  attention.__name__ == spec.name == name and attention.__spec__ is spec and
                  type(spec.loader) is SourceFileLoader and attention.__loader__ is spec.loader and
                  spec.loader.name == name and Path(spec.loader.path).resolve() == child_path and
                  Path(spec.origin).resolve() == Path(attention.__file__).resolve() == child_path and
                  Path(parent._ATTENTION_SOURCE).resolve() == child_path,
                  'Foreign k8 private module/spec/loader')
        c.require(fn.__globals__ is vars(attention) and attention._attention_project is jit and
                  attention._attention_project.fn is fn, 'Copied k8 globals or replaced actual JIT')
        objects = (parent, wrapper, original, attention, spec, spec.loader, jit, fn)
        if self.owner_objects is None: self.owner_objects = objects
        c.require(all(a is b for a,b in zip(self.owner_objects, objects)), 'K8 owner link changed within cache scope')
        return dict(kind='registered_k8_combo_private_attention', parent_module=parent.__name__,
                    parent_path=str(parent_path), parent_sha256=self.allowed_files[str(parent_path)],
                    module_attribute='attention', spec_attribute='_SPEC', source_attribute='_ATTENTION_SOURCE',
                    spec_name=name, loader='importlib.machinery.SourceFileLoader',
                    parent_factory='contextlib.contextmanager.<locals>.helper->__wrapped__:installed',
                    wrapper_closure='func is __wrapped__ original generator',
                    child_path=str(child_path), child_sha256=self.allowed_files[path], kernel_attribute='_attention_project')

    @staticmethod
    def same_k8_alias(left, right):
        """One physical key may have canonical and exact private owner proofs."""
        a, b = left['source'], right['source']
        if {a['module'],b['module']} != {'post_attention_fusion_v1','post_attention_fusion_combo_local_v1'}: return False
        private = a if 'owner_link' in a else b
        if private.get('owner_link',{}).get('kind') != 'registered_k8_combo_private_attention': return False
        if any(a.get(k) != b.get(k) for k in ('path','sha256','qualname','source_hash')) or a['qualname'] != '_attention_project': return False
        if any(left.get(k) != right.get(k) for k in left.keys() | right.keys()
               if k not in ('source','specialization_data','source_variants')): return False
        x, y = json.loads(left['specialization_data']), json.loads(right['specialization_data'])
        if x.pop('name',None) != a['module']+'.'+a['qualname'] or y.pop('name',None) != b['module']+'.'+b['qualname']: return False
        return x == y

    def record_receipt(self, key, receipt):
        with self.lock:
            return self._record_receipt_locked(key, receipt)

    def _record_receipt_locked(self, key, receipt):
        earlier = self.rows.get(key)
        if earlier is None:
            self.rows[key] = receipt; return
        variants = earlier.get('source_variants', [dict(source=earlier['source'],specialization_data=earlier['specialization_data'])])
        current = dict(source=receipt['source'],specialization_data=receipt['specialization_data'])
        base = {k:v for k,v in earlier.items() if k != 'source_variants'}
        if current in variants:
            matched = dict(base, **current)
            c.require(receipt == matched, 'One compiler key describes different actual binaries/sources')
        else:
            c.require(self.same_k8_alias(base, receipt), 'One compiler key describes different actual binaries/sources')
            c.require(len(variants) == 1, 'Unexpected third source owner for one compiler key')
            earlier['source_variants'] = variants + [current]

    def describe(self, jit, src, target, options, kw):
        c.require(target is not None, 'Compiler target must be captured by the actual caller-thread JIT')
        actual_target = target
        identity = binding_key(jit, src.signature, src.constants, getattr(src,'attrs',None),
                               actual_target, options or {})
        with self.lock:
            binding = self.specializations.get(identity)
            c.require(binding is not None, 'Unbound actual ASTSource specialization; no last-JIT-slot fallback')
            binding = dict(binding)
        c.require(jit.fn is binding['fn'] and jit.fn.__code__ is binding['code'] and
                  jit.cache_key == binding['source']['source_hash'], 'JIT owner changed after actual specialization binding')
        backend = self.make_backend(actual_target)
        parsed = backend.parse_options(self.fast_options(options))
        env = kw.get('_env_vars') or self.env_vars()
        c.require(json_value(env) == binding['env'], 'Compiler environment changed after caller-thread binding')
        key = hashlib.sha256(self.get_key(src, backend, parsed, env).encode()).hexdigest()
        value = dict(source=deepcopy(binding['source']), compiler_key=key,
                     signature=json_value(src.signature), constants=json_value(src.constants),
                     attrs=json_value(getattr(src, 'attrs', None)), target=json_value(actual_target),
                     options=json_value(vars(parsed)), env=json_value(env), src_name=src.name,
                     specialization_data=binding['specialization_data'])
        return value, actual_target, parsed, key

    def authorize(self, value, group):
        key = value['compiler_key']
        if self.precompile:
            if not group: self.misses.append(value)
        else:
            c.require(bool(group), 'Actual runtime DiskOnly cache group missing: ' + value['src_name'])
            c.require(key in self.expected_keys, 'Specialization was not observed in separate precompile: ' + key)
            expected = self.expected_keys[key]
            for name in ('source', 'signature', 'constants', 'attrs', 'target', 'options', 'env', 'src_name'):
                variants = expected.get('source_variants', [dict(source=expected['source'])])
                matches = any(value['source'] == v['source'] for v in variants) if name == 'source' else value[name] == expected[name]
                c.require(matches, 'Actual compiler identity differs: ' + name)

    def __enter__(self):
        from triton import knobs
        from triton.runtime import driver
        from triton.compiler.compiler import make_backend
        from triton.runtime import cache
        from triton._C.libtriton import get_cache_invalidating_env_vars
        from portable_cache import fast_cache_options
        c.require(not knobs.compilation.always_compile and not knobs.compilation.override, 'Forced compilation/override forbidden')
        c.require(knobs.cache.manager_class in (None, cache.FileCacheManager), 'Foreign cache manager forbidden')
        self.knobs, self.driver, self.make_backend = knobs, driver, make_backend
        self.get_key, self.manager = cache.get_cache_key, cache.get_cache_manager
        self.env_vars, self.fast_options = get_cache_invalidating_env_vars, fast_cache_options
        from parallel_compile_720_v1 import ColdCompileSession
        parallel_module = sys.modules['parallel_compile_720_v1']
        parallel_path = Path(parallel_module.__file__).resolve(strict=True)
        c.require(self.source_tree is not None and parallel_path == self.source_tree/'game/parallel_compile_720_v1.py' and
                  str(parallel_path) in self.allowed_files and c.sha(parallel_path) == self.allowed_files[str(parallel_path)],
                  'Parallel compile controller is not in the frozen source inventory')
        self.compile_session = ColdCompileSession(precompile=self.precompile, workers=self.cpu_compile_workers,
            memory_budget_gib=self.compile_memory_budget_gib, reserve_gib=self.compile_reserve_gib,
            progress=self.progress)
        self.original_hook = knobs.runtime.jit_cache_hook
        self.disk = None
        self.specializations = {}
        self.cache_class = cache.FileCacheManager
        self.original_put = self.cache_class.put
        self.original_put_group = self.cache_class.put_group
        if not self.precompile:
            self.disk = self.bootstrap.DiskOnly()
            self.disk.__enter__()
            delegate_hook = knobs.runtime.jit_cache_hook
            def deny_write(*args, **kwargs):
                with self.lock: self.readonly_write_attempts += 1
                raise RuntimeError('Readonly Triton cache write attempted; prepare this specialization separately')
            self.cache_class.put = deny_write
            self.cache_class.put_group = deny_write
        else:
            delegate_hook = None
        def hook(**args):
            c.require(get_ident() == self.owner_thread, 'Actual JIT warmup/binding must remain on the single GPU caller thread')
            jit = args['fn'].jit_function
            specialization = args['compile']['specialization_data']
            data = json.loads(specialization)
            identity = binding_key(jit, args['compile']['signature'], args['compile']['constants'],
                args['compile']['configs'][0], data['target'], data['options'])
            source = self.source_identity(jit)
            binding = dict(specialization_data=specialization,source=source,fn=jit.fn,code=jit.fn.__code__,
                           env=json_value(self.env_vars()))
            with self.lock:
                previous = self.specializations.get(identity)
                c.require(previous is None or previous == binding, 'Actual specialization/source binding changed')
                self.specializations[identity] = binding
            if delegate_hook is not None: delegate_hook(**args)
            if jit not in self.changed:
                original = jit.compile
                self.changed[jit] = original
                def compiled(src, target=None, options=None, **kw):
                    self.emit_progress('compiler_identity_begin', module=jit.fn.__module__, kernel=jit.fn.__qualname__)
                    value, actual_target, parsed, key = self.describe(jit, src, target, options, kw)
                    with self.lock: key_lock = self.key_locks.setdefault(key, RLock())
                    with key_lock:
                        return self._compile_bound(original,src,actual_target,parsed,key,value,kw)
                jit.compile = compiled
            return False
        knobs.runtime.jit_cache_hook = hook
        self.compile_session.__enter__()
        return self

    def emit_progress(self, *args, **kwargs):
        with self.lock: self.progress(*args, **kwargs)

    def _compile_bound(self, original, src, actual_target, parsed, key, value, kw):
        group = self.manager(key).get_group(src.name[:150] + '.json')
        with self.lock: self.authorize(value, group)
        self.emit_progress('compiler_load_begin', kernel=src.name, compiler_key=key, cache_group_present=bool(group))
        # In readonly mode this calls the unchanged DiskOnly wrapper,
        # which calculates/checks the actual runtime group again.
        with self.lock:
            self.active_compilers += 1
            self.peak_active_compilers = max(self.peak_active_compilers,self.active_compilers)
        try:
            kernel = original(src, target=actual_target, options=vars(parsed), **kw)
        finally:
            with self.lock: self.active_compilers -= 1
        files = {str(Path(p).resolve()):c.sha(p) for p in kernel.metadata_group.values()}
        receipt = dict(value, loaded_kernel_hash=kernel.hash, metadata_files=files,
                       loaded_binary_sha256=hashlib.sha256(kernel.kernel).hexdigest())
        if not self.precompile:
            expected = self.expected_keys[key]
            c.require(receipt['loaded_kernel_hash'] == expected['loaded_kernel_hash'] and
                      receipt['loaded_binary_sha256'] == expected['loaded_binary_sha256'] and
                      files == expected['metadata_files'], 'Prepared executable/group bytes differ')
        with self.lock:
            self.record_receipt(key, receipt); self.hits += 1
            hits = self.hits
        self.emit_progress('compiler_load_done', kernel=src.name, compiler_key=key, hits=hits)
        return kernel

    def __exit__(self, *args):
        try:
            if self.compile_session is not None: self.compile_session.__exit__(*args)
        finally:
            self.knobs.runtime.jit_cache_hook = self.original_hook
            for jit, original in self.changed.items(): jit.compile = original
            self.cache_class.put, self.cache_class.put_group = self.original_put, self.original_put_group
            try:
                if self.disk is not None: self.disk.__exit__(*args)
            finally:
                self.knobs.runtime.jit_cache_hook = self.original_hook
                self.owner_objects = None

    def snapshot(self):
        with self.lock:
            return deepcopy(dict(policy='separate_actual_source_precompile' if self.precompile else 'unmodified_runtime_DiskOnly_plus_precompile_identity_and_write_denial',
                    actual_compiler_keys=self.rows, misses=self.misses, hits=self.hits,
                    readonly_write_attempts=self.readonly_write_attempts,
                    original_disk_only_hits=None if self.disk is None else self.disk.hits,
                    new_candidate_kernels=self.misses if self.precompile else [],
                    peak_active_compiler_calls=self.peak_active_compilers,
                    specialization_bindings=len(self.specializations),
                    parallel_compile=None if self.compile_session is None else self.compile_session.snapshot()))
