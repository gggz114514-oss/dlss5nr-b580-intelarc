"""CPU-only regressions for alias cleanup and the changed-kernel spill gate.

These synthetic gate cases check dispatch control, not actual GPU resources.
Real zero-spill resources and outputs are tested by the complete-call runners.
"""
from short_fp8_validation_env_v1 import *
import importlib
import types
from triton.compiler.compiler import CompiledKernel
from triton.runtime.jit import _triton_jit_function_registry
from fork_fp8_jit_v2 import Fork
from short_fp8_graph_v1 import ShortFP8Graph,PRELOAD
import nr_backend.triton_fp8 as fp8
OUT = D / 'experimental/short-fp8-framework-v1'
assert not OUT.exists()
sources[str(Path(__file__))] = sha(__file__)
for name in PRELOAD:
    importlib.import_module(name)
alias = types.ModuleType('short_fp8_test_alias')
alias.__file__ = str(HERE / 'short_fp8_v2.py')
alias.leaf = fp8._round_fp8_half
sys.modules[alias.__name__] = alias
sys.modules['short_fp8_test_alias_second_name'] = alias
registry = dict(_triton_jit_function_registry)
fork = Fork(ROOT)
assert registry == _triton_jit_function_registry
sources.update(fork.sources)
assert sum(m is alias and n == 'leaf' for m,n,_,_ in fork.bindings) == 1
original = alias.leaf
try:
    try:
        with fork.installed():
            assert alias.leaf is fork.memo[original] and alias.leaf is not original
            raise ValueError('Intentional body-build failure')
    except ValueError as error:
        assert str(error) == 'Intentional body-build failure'
    fork.verify_restored()
    try:
        with fork.installed():
            alias.leaf = original
    except AssertionError as error:
        assert 'short_fp8_test_alias.leaf' in str(error)
    else:
        raise AssertionError('Binding interference was not detected')
    fork.verify_restored()
finally:
    del sys.modules[alias.__name__]
    del sys.modules['short_fp8_test_alias_second_name']

candidate,unchanged = object(),object()
adapter = object.__new__(ShortFP8Graph)
adapter.changed = {candidate}
adapter.resources = {}
dispatched,cases = [],[]
original_metadata = CompiledKernel.launch_metadata
def dispatch(kernel,grid,stream,*args):
    dispatched.append(kernel.hash)
    return 'dispatched'
CompiledKernel.launch_metadata = dispatch
try:
    with adapter.resource_gate():
        for label,changed,spill,accept in (
            ('changed_zero',True,0,True),('changed_spill',True,448,False),
            ('changed_unknown',True,None,False),('unchanged_existing_spill',False,448,True)):
            fn = candidate if changed else unchanged
            # The real path needs a qualified name; provide a synthetic callable shell.
            fn = types.SimpleNamespace(fn=types.SimpleNamespace(__module__='test',__name__=label))
            # Identity-based membership without requiring a hashable SimpleNamespace.
            class Function:
                pass
            identity = Function();identity.fn = fn.fn
            if changed:
                adapter.changed.add(identity)
            kernel = types.SimpleNamespace(src=types.SimpleNamespace(fn=identity),hash=label,
                n_spills=spill,n_regs=0,metadata=types.SimpleNamespace(shared=0),_init_handles=lambda:None)
            before = len(dispatched)
            try:
                value = CompiledKernel.launch_metadata(kernel,None,None)
            except RuntimeError:
                assert not accept and len(dispatched) == before
            else:
                assert accept and value == 'dispatched' and len(dispatched) == before+1
            cases.append(dict(case=label,accepted=accept,synthetic=True))
    assert CompiledKernel.launch_metadata is dispatch
finally:
    CompiledKernel.launch_metadata = original_metadata
assert _triton_jit_function_registry == registry
finalize_sources()
OUT.mkdir()
report = dict(passed=True,scope=__doc__,sources=sources,cases=cases,
    module_alias_deduplicated=True,exception_cleanup_restores_every_binding=True,
    interference_detected_and_all_bindings_restored=True,registry_unchanged=True,no_gpu_execution=True)
(OUT / 'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))
