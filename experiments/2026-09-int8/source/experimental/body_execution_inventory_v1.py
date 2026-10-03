"""Non-timing inventory of Triton launch requests and ATen operations.

Use only for an isolated eager equivalent of the graph body. ATen operations
include allocation and views and must not be reported as GPU kernel counts.
Output byte counts describe logical tensors, not measured memory traffic.
No native profiler or device-side callbacks are installed.
"""
from collections import Counter
from contextlib import contextmanager
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from triton.compiler.compiler import CompiledKernel


def tensors(value):
    if isinstance(value,torch.Tensor):
        yield value
    elif isinstance(value,(list,tuple)):
        for item in value:yield from tensors(item)
    elif isinstance(value,dict):
        for item in value.values():yield from tensors(item)


def description(t):
    return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),
                logical_bytes=t.numel()*t.element_size())


class Inventory(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.events=[]

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        result=func(*args,**kwargs)
        inputs=[t for t in tensors((args,kwargs)) if t.device.type=='xpu']
        outputs=[t for t in tensors(result) if t.device.type=='xpu']
        if inputs or outputs:
            addresses={t.untyped_storage().data_ptr() for t in inputs}
            self.events.append(dict(kind='aten',name=str(func),inputs=[description(t) for t in inputs],
                outputs=[dict(**description(t),aliases_argument_storage=t.untyped_storage().data_ptr() in addresses) for t in outputs]))
        return result

    @contextmanager
    def installed(self):
        original=CompiledKernel.launch_metadata
        def observe(kernel,grid,stream,*args):
            fn=getattr(getattr(kernel.src,'fn',None),'fn',None)
            self.events.append(dict(kind='triton',name=f'{getattr(fn,"__module__","unknown")}.{getattr(fn,"__name__",kernel.name)}',
                grid=[int(n) for n in grid],arguments=[description(t) for t in tensors(args) if t.device.type=='xpu']))
            return original(kernel,grid,stream,*args)
        CompiledKernel.launch_metadata=observe
        try:
            with self:yield self
        finally:
            assert CompiledKernel.launch_metadata is observe
            CompiledKernel.launch_metadata=original

    def summary(self):
        result={}
        for kind in ('triton','aten'):
            counts=Counter(e['name'] for e in self.events if e['kind']==kind)
            result[kind]=dict(count=sum(counts.values()),by_name=dict(counts.most_common()))
        result['aten_output_logical_bytes']=sum(t['logical_bytes'] for e in self.events if e['kind']=='aten' for t in e['outputs'])
        result['aten_argument_alias_output_logical_bytes']=sum(t['logical_bytes'] for e in self.events if e['kind']=='aten' for t in e['outputs'] if t['aliases_argument_storage'])
        return result
