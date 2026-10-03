"""Keep scalar FMA operands resident before capture; original arithmetic remains."""
import torch
import nr_backend.sampling as sampling
from graph_history_warp_v1 import GraphHistoryWarp as Base


class GraphHistoryWarp(Base):
    def _build(self, image, motion):
        original = sampling.fma32
        scalars = {}

        def operand(value):
            if isinstance(value, torch.Tensor):
                return torch.as_tensor(value,device=image.device,dtype=torch.float32)
            key = (type(value), value)
            if key not in scalars:
                if torch.xpu.is_current_stream_capturing():
                    raise RuntimeError('New scalar appeared after history-warp warmup')
                scalars[key] = torch.as_tensor(value,device=image.device,dtype=torch.float32)
            return scalars[key]

        def fma(a, b, c):
            return torch.addcmul(operand(c), a.float(), operand(b))

        sampling.fma32 = fma
        try:
            entry = super()._build(image, motion)
            entry.scalar_operands = scalars
            return entry
        finally:
            assert sampling.fma32 is fma
            sampling.fma32 = original
