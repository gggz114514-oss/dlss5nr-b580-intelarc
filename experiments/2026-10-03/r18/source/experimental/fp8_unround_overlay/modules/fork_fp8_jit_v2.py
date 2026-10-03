"""Isolated JIT dependency forks for an exhaustive-equivalent leaf experiment.

No installed file or existing JIT object's source/global snapshot is rewritten.
Candidate JIT functions have private globals and their own in-memory caches;
module bindings are swapped only in this serial experimental context. Original
qualified names retain the already reviewed dataflow contracts. This is not a
production concurrent patching API or a general optimizer.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import sys
import types
from triton.runtime.jit import JITFunction, _triton_jit_function_registry
from nr_backend.triton_fp8 import _round_fp8_half
from short_fp8_v2 import round_half


class Fork:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.original = _round_fp8_half
        self.memo = {self.original: round_half}
        self.active = set()
        self.bindings = []
        self.sources = {}
        self.forks = []
        # Snapshot before touching any module binding; the candidate helper and
        # this coordinator are excluded so the old/new scalar control stays old.
        modules = []
        seen_modules = set()
        for module in list(sys.modules.values()):
            path = getattr(module, '__file__', None)
            if not path or id(module) in seen_modules or module.__name__ in (__name__, 'short_fp8_v1', 'short_fp8_v2', '__main__', '__mp_main__'):
                continue
            path = Path(path).resolve()
            if path.is_relative_to(self.root):
                modules.append(module)
                seen_modules.add(id(module))
        for module in modules:
            for name, value in list(vars(module).items()):
                if isinstance(value, JITFunction):
                    replacement = self.fork(value)
                    if replacement is not value:
                        self.bindings.append((module, name, value, replacement))
        assert self.forks and self.bindings
        assert len({(id(m), n) for m, n, _, _ in self.bindings}) == len(self.bindings)

    def fork(self, original):
        if original in self.memo:
            return self.memo[original]
        if original in self.active:
            raise RuntimeError('Recursive JIT dependency is not supported')
        self.active.add(original)
        fn = original.fn
        globals_copy = dict(fn.__globals__)
        changes = []
        for name in set(fn.__code__.co_names):
            dependency = globals_copy.get(name)
            if isinstance(dependency, JITFunction):
                candidate = self.fork(dependency)
                if candidate is not dependency:
                    globals_copy[name] = candidate
                    changes.append(name)
        self.active.remove(original)
        if not changes:
            self.memo[original] = original
            return original
        if original.pre_run_hooks:
            raise RuntimeError('Unreviewed pre-run hooks')
        copied = types.FunctionType(fn.__code__, globals_copy, fn.__name__, fn.__defaults__, fn.__closure__)
        copied.__kwdefaults__ = fn.__kwdefaults__
        copied.__annotations__ = dict(fn.__annotations__)
        copied.__module__ = fn.__module__
        copied.__qualname__ = fn.__qualname__
        key = f'{fn.__module__}:{fn.__qualname__}'
        registered = _triton_jit_function_registry.get(key)
        try:
            candidate = JITFunction(copied, version=original.version,
                do_not_specialize=original.do_not_specialize,
                do_not_specialize_on_alignment=original.do_not_specialize_on_alignment,
                debug=original.debug, noinline=original.noinline, repr=original._repr,
                launch_metadata=original.launch_metadata)
        finally:
            if registered is None:
                _triton_jit_function_registry.pop(key, None)
            else:
                _triton_jit_function_registry[key] = registered
        assert candidate.src == original.src and candidate.arg_names == original.arg_names
        assert candidate.cache_key != original.cache_key
        path = Path(inspect.getsourcefile(fn)).resolve()
        self.sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.memo[original] = candidate
        self.forks.append(dict(name=f'{fn.__module__}.{fn.__name__}', dependencies=sorted(changes),
            original_cache_key=original.cache_key, candidate_cache_key=candidate.cache_key))
        return candidate

    @contextmanager
    def installed(self):
        for module, name, old, new in self.bindings:
            assert getattr(module, name) is old
        for module, name, old, new in self.bindings:
            setattr(module, name, new)
        try:
            yield self
        finally:
            mismatches = []
            for module, name, old, new in reversed(self.bindings):
                if getattr(module, name) is not new:
                    mismatches.append(f'{module.__name__}.{name}')
                setattr(module, name, old)
            assert not mismatches, mismatches

    def verify_restored(self):
        assert all(getattr(module, name) is old for module, name, old, _ in self.bindings)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in self.sources.items())
