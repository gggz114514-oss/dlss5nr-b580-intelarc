"""Use the fused five-tap builder inside the original history ownership/guards.

Base apply retains the table identity/version check, status readback before
returning, caller-owned cloned outputs and close/scope behavior. Other tensor
contracts use the original builder. No model state is committed by this adapter.
"""
from types import SimpleNamespace
import torch
from graph_history_warp_v2 import GraphHistoryWarp as Base
from fused_square_history_v1 import forward


class FusedHistoryWarp(Base):
    def __init__(self,model,*,block=64,warps=4):
        if block not in (32,64,128) or warps not in (4,8):raise ValueError('Unsupported fused history launch')
        super().__init__(model);self.options=dict(block=block,warps=warps);self.fused_builds=0

    def _build(self,image,motion):
        if image.dtype!=torch.float16 or motion.dtype not in (torch.float16,torch.float32) or not image.is_contiguous() or not motion.is_contiguous():
            return super()._build(image,motion)
        static_image,static_motion=image.clone(),motion.clone()
        def compute():
            (numerator,reciprocal,flags),_=forward(static_image,static_motion,self.table.values,**self.options)
            return numerator,reciprocal,flags.all()
        stream=torch.xpu.Stream();torch.xpu.synchronize()
        with stream:
            for _ in range(2):warm=compute()
        torch.xpu.synchronize();outputs=tuple(torch.empty_like(t) for t in warm);del warm
        graph=torch.xpu.XPUGraph()
        with torch.xpu.graph(graph,stream=stream):
            temporary=compute()
            for output,value in zip(outputs,temporary):output.copy_(value)
        del temporary
        segments=torch.xpu.memory_snapshot(graph.pool())
        assert not any(s['address']<=t.data_ptr()<s['address']+s['total_size'] for s in segments for t in (static_image,static_motion,*outputs))
        self.fused_builds+=1
        return SimpleNamespace(graph=graph,stream=stream,image=static_image,motion=static_motion,outputs=outputs)
