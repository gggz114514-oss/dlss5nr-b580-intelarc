"""C32 encoder Swin blocks, reconstructed from original launch boundaries."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from .pre_mlp import C32MLP,_decode_weights,quantize_fp8
from .unround_policy import round_c32 as quantize_fp8
from .attention import C32AttentionCore
from .tensor_math import sm89_f16_dot


class C32SwinBlock(nn.Module):
    """Observed C32 encoder contract; logical HWC32 tensors at FP8 boundaries.

    window_shift is (y,x), either zero or four. Shifted windows start before
    the image, using zero extension on both edges. This is not cyclic rolling.
    Validated size is 160x160 from the pinned input256/pad320 reference.
    Optional downsampling returns H/2 W/2 C64. No temporal state or XMX tuning.
    """
    def __init__(self,record: bytes, *, window_shift: tuple[int,int]=(0,0), downsample: bool=False):
        super().__init__()
        if len(record)!=(22720 if downsample else 20672):raise ValueError('Wrong pinned C32 record size')
        if len(window_shift)!=2 or any(s not in (0,4) for s in window_shift):raise ValueError('Expected shifts zero or four')
        self.window_shift=tuple(window_shift)
        self.mlp=C32MLP(record,skip_scale_offset=0x2010)
        self.attention=C32AttentionCore(record,qkv_offset=0x2060,bias_offset=0x2c60,scale_offset=0x4c60)
        self.register_buffer('output_weight',_decode_weights(record[0x4c70:0x5070],32,32))
        self.register_buffer('skip_scale',torch.frombuffer(bytearray(record[0x5070:0x50b0]),dtype=torch.float16).clone())
        self.register_buffer('down_weight',_decode_weights(record[0x50b0:0x58b0],32,64) if downsample else None)

    def forward_unquantized(self,features: Tensor) -> Tensor:
        if features.ndim!=3 or features.shape[-1]!=32 or features.shape[0]%8 or features.shape[1]%8:
            raise ValueError('Expected HWC32 with positive dimensions divisible by eight')
        height,width=features.shape[:2]
        if min(height,width)==0:raise ValueError('Empty C32 features')
        sy,sx=self.window_shift
        x=quantize_fp8(features)
        if sy or sx:x=torch.nn.functional.pad(x,(0,0,sx,sx,sy,sy))
        mlp=self.mlp.forward_unquantized(x)
        attended=self.attention(quantize_fp8(mlp))
        result=sm89_f16_dot(quantize_fp8(attended),self.output_weight,chunk_k=16,initial=(mlp*self.skip_scale).half())
        return result[sy:sy+height,sx:sx+width]

    def forward(self,features: Tensor) -> Tensor:
        return quantize_fp8(self.forward_unquantized(features))

    def forward_outputs(self,features: Tensor) -> tuple[Tensor,Tensor|None]:
        full=self.forward_unquantized(features);skip=quantize_fp8(full)
        if self.down_weight is None:return skip,None
        top=(full[0::2,0::2]+full[0::2,1::2]).half()
        bottom=(full[1::2,0::2]+full[1::2,1::2]).half()
        pooled=quantize_fp8(((top+bottom).half()*0.25).half())
        down=quantize_fp8(sm89_f16_dot(pooled,self.down_weight,chunk_k=16))
        return skip,down
