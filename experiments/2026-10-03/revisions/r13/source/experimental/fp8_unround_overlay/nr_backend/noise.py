"""Portable lookup implementation of the complete native scalar noise domain."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
from torch import Tensor,nn
from .front import _uniform_indices


class NativeNoiseTable(nn.Module):
    """Three 2^24 FP32 tables; no image-specific values or seed-specific cache.

    Native radius and cycle sin/cos are enumerated once on the reference GPU.
    Runtime hashing, table lookup and final products execute on the input GPU.
    This 192 MiB correctness path is pending size/performance optimization.
    """
    def __init__(self,radius: Tensor,sine: Tensor,cosine: Tensor):
        super().__init__()
        for name,value in [('radius',radius),('sine',sine),('cosine',cosine)]:
            if value.dtype!=torch.float32 or tuple(value.shape)!=(1<<24,):raise ValueError('Expected complete 24-bit FP32 table')
            self.register_buffer(name,value)

    @classmethod
    def from_directory(cls,path: str|Path):
        path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
        if manifest['domain_count']!=1<<24 or manifest['schema']!=1:raise ValueError('Unknown scalar table contract')
        tables=[]
        for name in ('radius.f32.bin','sin.f32.bin','cos.f32.bin'):
            file=path/name;data=file.read_bytes()
            if len(data)!=1<<26 or hashlib.sha256(data).hexdigest()!=manifest['files'][name]:raise ValueError('Scalar table hash mismatch')
            tables.append(torch.frombuffer(bytearray(data),dtype=torch.float32).clone())
        return cls(*tables)

    def forward(self,x: Tensor,y: Tensor,seed: int) -> Tensor:
        if x.device!=self.radius.device or y.device!=x.device:raise ValueError('Noise coordinates and table must share a device')
        a,b,c,d=_uniform_indices(x,y,seed)
        ra,rc=self.radius[a],self.radius[c]
        return torch.stack((rc*self.cosine[d],rc*self.sine[d],ra*self.cosine[b]),dim=-1).half()
