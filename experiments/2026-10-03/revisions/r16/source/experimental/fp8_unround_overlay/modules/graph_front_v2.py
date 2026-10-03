"""Preserve the original completed-body boundary before MotionNR commits history."""
import torch
from graph_front_v1 import GraphFront as Base

class GraphFront(Base):
    def _complete(self):
        # Original ResetNR._forward_front waits at mark('RGB') before returning.
        # Deferred device errors must surface before MotionNR replaces _previous
        # or advances its seed. Only submitted XPU work is involved here.
        torch.xpu.synchronize(self.model.blend_scale.device)

    def forward(self,*args,**kwargs):
        before=self.replays
        value=super().forward(*args,**kwargs)
        if self.replays!=before:self._complete()
        return value
