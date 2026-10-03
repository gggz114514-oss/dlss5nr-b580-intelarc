"""Owned immutable-model graph adapter around the complete NR network body.

MotionNR still validates inputs, computes new seed/noise/warp/front and commits
private history after success. Reset and temporal bodies have separate graphs.
Outputs are cloned before returning so later replay cannot overwrite caller data.
Experimental single-owner scope; not a concurrent production backend.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import time
import torch
import nr_backend.triton_math as arithmetic_functions
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch,use_arithmetic_backend
import capture_body_v1 as body

def descriptor(t):
    return None if t is None else (tuple(t.shape),t.dtype,t.device.type,t.device.index)

class GraphFront:
    def __init__(self,model,*,arithmetic):
        self.model=model;self.arithmetic=arithmetic;self.original=model._forward_front
        self.signature=self._signature();self.constants=self._constants()
        self.entries={};self.replays=0;self.closed=False;self.last_entry=None

    def _signature(self):
        a=self.arithmetic
        return (a.mode,a.use_fused,a.expanded,a.use_view_cache)

    def _constants(self):
        result=[]
        for name,t in self.model.named_buffers():
            if name=='_previous':continue
            # An explicit immutable session rejects ordinary mutations and buffer
            # replacement. Inference tensors without version counters are refused.
            result.append((name,id(t),t.data_ptr(),descriptor(t),tuple(t.stride()),t._version))
        return tuple(result)

    def _validate(self):
        if self.closed:raise RuntimeError('Graph session is closed')
        if self._signature()!=self.signature:raise RuntimeError('Arithmetic changed: create a new graph session')
        if self._constants()!=self.constants:raise RuntimeError('Model constants changed: create a new graph session')
        if getattr(arithmetic_functions.fused_dot,'__self__',None) is not self.arithmetic:
            raise RuntimeError('Graph session requires its own installed arithmetic provider')

    def _build(self,key,inputs,options):
        started=time.perf_counter()
        static={name:None if t is None else t.clone(memory_format=torch.contiguous_format) for name,t in inputs.items()}
        stream=torch.xpu.Stream();torch.xpu.synchronize()
        def compute():return body.forward_front(self.model,**static,**options)
        with body.installed():
            with stream:
                for _ in range(2):
                    with use_arithmetic_backend('triton'):warm=compute()
            torch.xpu.synchronize();del warm
            graph=torch.xpu.XPUGraph()
            with use_arithmetic_backend('triton') as captured_dispatch:
                with torch.xpu.graph(graph,stream=stream):output=compute()
        entry=SimpleNamespace(graph=graph,stream=stream,inputs=static,output=output,dispatch=dict(captured_dispatch),replays=0,build_seconds=time.perf_counter()-started)
        self.entries[key]=entry
        return entry

    def forward(self,rgb,front,*,progress=None,previous=None,sigmoid=None,blend_scale=None,history_reciprocal=None,return_float32=False):
        if progress is not None or current_arithmetic_backend()!='triton':
            return self.original(rgb,front,progress=progress,previous=previous,sigmoid=sigmoid,blend_scale=blend_scale,history_reciprocal=history_reciprocal,return_float32=return_float32)
        self._validate()
        if sigmoid is not self.model.sigmoid or blend_scale is not self.model.blend_scale:
            raise ValueError('Graph session expects its owned sigmoid/blend constants')
        inputs=dict(rgb=rgb,front=front,previous=previous,history_reciprocal=history_reciprocal)
        key=tuple(descriptor(t) for t in inputs.values())+(bool(return_float32),)
        options=dict(sigmoid=sigmoid,blend_scale=blend_scale,return_float32=return_float32)
        entry=self.entries.get(key)
        if entry is None:entry=self._build(key,inputs,options)
        for name,t in inputs.items():
            if t is not None and t is not entry.inputs[name]:entry.inputs[name].copy_(t)
        entry.graph.replay();entry.replays+=1;self.replays+=1
        self.last_entry=entry
        record_arithmetic_dispatch('xpu_graph_replay')
        return entry.output.clone()

    def metadata(self):
        return [dict(key=str(key),build_seconds=e.build_seconds,replays=e.replays,captured_dispatch=e.dispatch,input_bytes=sum(t.numel()*t.element_size() for t in e.inputs.values() if t is not None)) for key,e in self.entries.items()]

    @contextmanager
    def installed(self):
        if self.closed:raise RuntimeError('Graph session is closed')
        had='_forward_front' in self.model.__dict__;previous=self.model.__dict__.get('_forward_front')
        replacement=self.forward;self.model._forward_front=replacement
        try:yield self
        finally:
            assert self.model._forward_front is replacement
            if had:self.model._forward_front=previous
            else:del self.model._forward_front

    def close(self):
        if self.closed:return
        torch.xpu.synchronize()
        for entry in self.entries.values():entry.graph.reset()
        self.entries.clear();self.closed=True
