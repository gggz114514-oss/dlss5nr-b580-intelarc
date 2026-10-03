"""C512 split-Swin research implementation; check captured boundaries before use."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from .pre_mlp import cubic_quantize,_decode_weights,quantize_fp8 as q
from .tensor_math import sm89_f16_dot as dot,cubic_activation
from .multihead_block import MultiHeadAttention


class SplitFeedForward(nn.Module):
    """Full C512 projection followed by eight independent C64->256->64 groups."""
    def __init__(self,record: bytes):
        super().__init__()
        if len(record)!=524288:raise ValueError('Unexpected split FFWD record size')
        self.register_buffer('linear',_decode_weights(record[:262144],512,512))
        self.register_buffer('expand',torch.stack([_decode_weights(record[0x40000+i*16384:0x40000+(i+1)*16384],64,256)for i in range(8)]))
        self.register_buffer('reduce',torch.stack([_decode_weights(record[0x60000+i*16384:0x60000+(i+1)*16384],256,64)for i in range(8)]))

    def forward(self,features: Tensor) -> Tensor:
        z=q(dot(q(features),self.linear,chunk_k=16));groups=[]
        for i in range(8):
            hidden=cubic_quantize(dot(z[...,i*64:(i+1)*64],self.expand[i],chunk_k=16))
            groups.append(q(dot(hidden,self.reduce[i],chunk_k=16)))
        return torch.cat(groups,dim=-1)


class SplitProjection(nn.Module):
    """C512 projection accumulated into a scaled FP8 residual."""
    def __init__(self,record: bytes):
        super().__init__()
        if len(record)!=263168:raise ValueError('Unexpected split projection record size')
        self.register_buffer('weight',_decode_weights(record[:262144],512,512))
        self.register_buffer('skip_scale',torch.frombuffer(bytearray(record[262144:]),dtype=torch.float16).clone())

    def forward_unquantized(self,features: Tensor,residual: Tensor) -> Tensor:
        return dot(q(features),self.weight,chunk_k=16,initial=(q(residual)*self.skip_scale).half())

    def forward(self,features: Tensor,residual: Tensor) -> Tensor:return q(self.forward_unquantized(features,residual))


class SplitSwinBlock(nn.Module):
    """C512 boundaries with optional padded pooling and C1024 final projection."""
    def __init__(self,records: list[bytes], *, window_shift: tuple[int,int]=(0,0)):
        super().__init__()
        if len(records)not in(4,5)or len(records[2])!=917568:raise ValueError('Unexpected split records')
        if len(records)==5 and len(records[4])!=524304:raise ValueError('Unexpected final-head record')
        if len(window_shift)!=2 or any(s not in (0,4)for s in window_shift):raise ValueError('Expected shifts zero or four')
        self.window_shift=tuple(window_shift)
        self.ffwd=SplitFeedForward(records[0]);self.ffwd_projection=SplitProjection(records[1])
        self.attention=MultiHeadAttention(records[2],512,qkv_offset=0,bias_offset=786432,scale_offset=917504)
        self.projection=SplitProjection(records[3])
        self.register_buffer('final_weight',_decode_weights(records[4][:524288],512,1024)if len(records)==5 else None)

    def forward_boundaries(self,features: Tensor) -> tuple[Tensor,...]:
        if features.ndim!=3 or features.shape[-1]!=512:raise ValueError('Expected HWC512')
        height,width=features.shape[:2]
        if min(height,width)==0 or height%4 or width%4:raise ValueError('Expected dimensions aligned to four')
        ffwd=self.ffwd(features);mlp=self.ffwd_projection(ffwd,features);sy,sx=self.window_shift
        padded=torch.nn.functional.pad(mlp,(0,0,sx,(-width-sx)%8,sy,(-height-sy)%8))
        attended=q(self.attention(padded)[sy:sy+height,sx:sx+width])
        full=self.projection.forward_unquantized(attended,mlp);output=q(full)
        if self.final_weight is None:return ffwd,mlp,attended,output
        top=(full[0::2,0::2]+full[0::2,1::2]).half();bottom=(full[1::2,0::2]+full[1::2,1::2]).half()
        pool=q(((top+bottom).half()*.25).half())
        pool=torch.nn.functional.pad(pool,(0,0,0,(-pool.shape[1])%4,0,(-pool.shape[0])%4))
        final=q(dot(pool,self.final_weight,chunk_k=16))
        return ffwd,mlp,attended,output,pool,final

    def forward(self,features: Tensor) -> Tensor:return self.forward_boundaries(features)[-1]
