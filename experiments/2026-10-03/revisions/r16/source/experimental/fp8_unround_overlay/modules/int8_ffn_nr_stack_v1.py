"""Experimental complete NR256 session with owned, immutable INT8 FFN constants.

Rebuild the unused v6 graph adapter after registering packed weights/scales so
its ordinary model signature protects every captured constant. Only serialized
NR256 Triton calls are supported. Progress callbacks are rejected before work:
silently falling back to the old FP16 FFN would change the session's arithmetic.
This module does not change the selected default or claim video validation.
"""
from contextlib import contextmanager
import torch
from nr_backend.execution import current_arithmetic_backend
from nr256_selected_stack_v4 import Stack as SelectedStack
from graph_front_v6 import GraphFront
from int8_ffn_body_scope_v1 import Int8VitLayout

NAMES=('expand_int8','expand_scale','hidden_scale','contract_int8','contract_scale')


class CallGuard:
    def __init__(self,stack):
        self.stack=stack;self.model=stack.model
        self.original=self.model.forward
        assert 'forward' not in self.model.__dict__

    def validate(self):
        stack=self.stack;scope=stack.int8_vit
        if stack.rewrite.vit_layout is not scope or not scope.enabled:
            raise RuntimeError('INT8 FFN route changed: create a new session')
        scope.validate_constants()
        for index,row in enumerate(stack.model._int8_ffn_buffers):
            if any(getattr(row,name) is not tensor for name,tensor in zip(NAMES,scope.packed[index])):
                raise RuntimeError('INT8 FFN registered constants changed: create a new session')

    def forward(self,rgb,motion,*,reset=False,progress=None):
        if progress is not None:
            raise ValueError('Experimental INT8 NR256 session does not support progress callbacks')
        if current_arithmetic_backend()!='triton' or self.stack.provider.mode!='fp16_xmx':
            raise RuntimeError('Experimental INT8 NR256 session requires its selected Triton provider')
        if tuple(rgb.shape)!=(256,256,3) or rgb.device.type!='xpu':
            raise ValueError('Experimental INT8 session requires XPU NR256 HWC RGB')
        self.validate()
        return self.original(rgb,motion,reset=reset,progress=None)

    @contextmanager
    def installed(self):
        assert 'forward' not in self.model.__dict__ and self.model.forward==self.original
        replacement=self.forward;self.model.forward=replacement
        try:yield self
        finally:
            valid=self.model.forward is replacement
            del self.model.forward
            assert valid,'INT8 public call scope interference'

    def verify_restored(self):
        assert 'forward' not in self.model.__dict__ and self.model.forward==self.original
        self.validate()


class Stack(SelectedStack):
    def __init__(self,exact_root,*,hidden_scales,share_with=None):
        super().__init__(exact_root,share_with=share_with)
        previous=self.graph
        assert not previous.entries and previous.replays==0 and previous._graph_cache is None
        scope=Int8VitLayout(self.model,self.provider,hidden_scales)
        registry=torch.nn.ModuleList()
        for values in scope.packed:
            row=torch.nn.Module()
            for name,tensor in zip(NAMES,values):row.register_buffer(name,tensor,persistent=False)
            registry.append(row)
        assert not hasattr(self.model,'_int8_ffn_buffers')
        self.model.add_module('_int8_ffn_buffers',registry)
        # No captured graph exists yet; the replacement signature includes all 40
        # packed tensors. Original skip constants are already model buffers.
        previous.close()
        self.graph=GraphFront(self.model,arithmetic=self.provider)
        self.components[self.components.index(previous)]=self.graph
        self.int8_vit=scope;self.rewrite.vit_layout=scope
        self.call_guard=CallGuard(self);self.components.append(self.call_guard)
        registered={name for name,*_ in self.graph.constants if name.startswith('_int8_ffn_buffers.')}
        assert len(registered)==40

    def metadata(self):
        return dict(quantization='continuous_ffn_int8_p4_margin1.25',nr_size=[256,256],
            registered_packed_tensors=40,
            packed_bytes=sum(t.numel()*t.element_size() for row in self.int8_vit.packed for t in row[:5]),
            ffn_resources=self.int8_vit.ffn_resources,ffn_selections=self.int8_vit.selections,
            capture=self.rewrite.metadata(),graphs=self.graph.metadata(),
            progress_policy='explicitly rejected before model execution',serialized_only=True)

    def close(self):
        self.call_guard.verify_restored()
        super().close()
