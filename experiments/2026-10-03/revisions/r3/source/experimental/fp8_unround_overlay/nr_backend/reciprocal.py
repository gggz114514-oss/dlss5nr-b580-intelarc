"""Complete native reciprocal scalar interval for the observed history filter."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from torch import Tensor,nn

START_BITS=0x3f780000
END_BITS=0x3f940000
RECIPROCAL_SHA256='22af04fd9ac7f1a0345e6f3fbbb0060aeff3e8a1373516872b4772e7926386f6'


class NativeReciprocalTable(nn.Module):
    """Every FP32 input in [0.96875,1.15625); no frame-dependent data."""
    def __init__(self,values: Tensor):
        super().__init__()
        if values.dtype!=torch.float32 or tuple(values.shape)!=(END_BITS-START_BITS,):raise ValueError('Expected complete bounded reciprocal domain')
        self.register_buffer('values',values)

    @classmethod
    def from_directory(cls,path: str|Path):
        data=(Path(path)/'reciprocal.f32.bin').read_bytes()
        if len(data)!=4*(END_BITS-START_BITS)or hashlib.sha256(data).hexdigest()!=RECIPROCAL_SHA256:raise ValueError('Native reciprocal scalar table hash mismatch')
        return cls(torch.frombuffer(bytearray(data),dtype=torch.float32).clone())

    def forward(self,value: Tensor) -> Tensor:
        if value.device!=self.values.device:raise ValueError('Scalar input and reciprocal table must share a device')
        index=(value.float().contiguous().view(torch.int32)-START_BITS).long()
        if not bool(((index>=0)&(index<len(self.values))).all()):raise ValueError('Input outside the validated native reciprocal interval')
        return self.values[index]


class NativeDimensionReciprocalTable(nn.Module):
    """Original MUFU.RCP for every integer dimension from 1 through 4096."""
    SHA256='672e6b8767c976a003406e4bdd1a903a7c2299da60341645d61969f37d5e21bf'
    def __init__(self,values: Tensor):
        super().__init__()
        if values.dtype!=torch.float32 or tuple(values.shape)!=(4096,):raise ValueError('Expected complete dimension reciprocal table')
        self.register_buffer('values',values)

    @classmethod
    def from_directory(cls,path: str|Path):
        data=(Path(path)/'reciprocal.f32.bin').read_bytes()
        if len(data)!=16384 or hashlib.sha256(data).hexdigest()!=cls.SHA256:raise ValueError('Dimension reciprocal table hash mismatch')
        return cls(torch.frombuffer(bytearray(data),dtype=torch.float32).clone())

    def forward(self,dimension: int) -> Tensor:
        if not isinstance(dimension,int)or not 1<=dimension<=4096:raise ValueError('Expected integer dimension in 1..4096')
        return self.values[dimension-1]
