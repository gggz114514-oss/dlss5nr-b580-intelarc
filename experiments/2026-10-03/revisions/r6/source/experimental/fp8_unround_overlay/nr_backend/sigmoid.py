"""Complete half-input scalar sigmoid domain measured on the SM89 reference."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from torch import Tensor,nn

SIGMOID_SHA256='394394a5258bad437495d68076be75d8413fa0ed5b752400e3947c332437b850'


class NativeSigmoidTable(nn.Module):
    """256 KiB scalar table; contains no image, history or model activations."""
    def __init__(self,values: Tensor):
        super().__init__()
        if values.dtype!=torch.float32 or tuple(values.shape)!=(65536,):
            raise ValueError('Expected complete 16-bit sigmoid domain')
        self.register_buffer('values',values)

    @classmethod
    def from_directory(cls,path: str|Path):
        data=(Path(path)/'sigmoid.f32.bin').read_bytes()
        if len(data)!=65536*4 or hashlib.sha256(data).hexdigest()!=SIGMOID_SHA256:
            raise ValueError('Native sigmoid scalar table hash mismatch')
        return cls(torch.frombuffer(bytearray(data),dtype=torch.float32).clone())

    def forward(self,logit: Tensor) -> Tensor:
        if logit.device!=self.values.device:raise ValueError('Logit and sigmoid table must share a device')
        bits=logit.half().contiguous().view(torch.int16).int()&0xffff
        return self.values[bits.long()]
