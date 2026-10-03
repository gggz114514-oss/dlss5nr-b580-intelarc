"""Eager ablation using the same fence-free body that graph capture uses."""
from contextlib import contextmanager
import capture_body_v1 as body

class QueuedFront:
    def __init__(self,model):self.model=model;self.original=model._forward_front
    def forward(self,rgb,front,*,progress=None,**kwargs):
        if progress is not None:return self.original(rgb,front,progress=progress,**kwargs)
        return body.forward_front(self.model,rgb,front,**kwargs)
    @contextmanager
    def installed(self):
        had='_forward_front' in self.model.__dict__;previous=self.model.__dict__.get('_forward_front')
        replacement=self.forward;self.model._forward_front=replacement
        try:
            with body.installed():yield self
        finally:
            assert self.model._forward_front is replacement
            if had:self.model._forward_front=previous
            else:del self.model._forward_front
