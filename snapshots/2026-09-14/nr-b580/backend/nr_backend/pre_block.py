"""Pre skip and downsample computation with native-reference arithmetic."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from .front import reset_front_features,project_front_features
from .pre_mlp import PreMLP,_decode_weights,quantize_fp8
from .attention import PreAttentionCore
from .tensor_math import sm89_f16_dot


class PreBlock(nn.Module):
    """Reference-math implementation of the pinned pre block's skip output.

    forward_features accepts computed front features. The complete RGB executor
    supplies its exact scalar noise table there. The convenience RGB methods
    below retain the analytic noise approximation for diagnostic comparisons.
    """
    def __init__(self,record: bytes):
        super().__init__()
        if len(record)!=21696:raise ValueError('Expected pinned pre record')
        front=torch.empty((16,32),dtype=torch.float16)
        lane=torch.arange(32);g,t=lane//4,lane%4
        for tile,offset in enumerate((0x2010,0x2210)):
            physical=torch.frombuffer(bytearray(record[offset:offset+512]),dtype=torch.float16).reshape(32,8)
            for frag in range(2):
                for i in range(4):front[2*t+i%2+8*(i//2),tile*16+g+8*frag]=physical[lane,frag*4+i]
        self.register_buffer('front_weight',front)
        self.mlp=PreMLP(record)
        self.attention=PreAttentionCore(record)
        self.register_buffer('output_weight',_decode_weights(record[0x5070:0x5470],32,32))
        self.register_buffer('skip_scale',torch.frombuffer(bytearray(record[0x5470:0x54b0]),dtype=torch.float16).clone())

    def forward_features_unquantized(self,features: Tensor) -> Tensor:
        projected=project_front_features(features,self.front_weight)
        mlp=self.mlp.forward_unquantized(projected)
        attended=self.attention(quantize_fp8(mlp))
        return sm89_f16_dot(quantize_fp8(attended),self.output_weight,chunk_k=16,initial=(mlp*self.skip_scale).half())

    def forward_features(self,features: Tensor) -> Tensor:
        return quantize_fp8(self.forward_features_unquantized(features))

    def forward_features_outputs(self,features: Tensor) -> tuple[Tensor,Tensor]:
        """Return (skip HWC32, down H/2 W/2 C32), both at their FP8 boundaries."""
        full=self.forward_features_unquantized(features)
        top=(full[0::2,0::2]+full[0::2,1::2]).half()
        bottom=(full[1::2,0::2]+full[1::2,1::2]).half()
        # Pool the unquantized projection. Pooling the FP8 skip loses evidence.
        down=((top+bottom).half()*0.25).half()
        return quantize_fp8(full),quantize_fp8(down)

    def forward(self,rgb: Tensor, *, padded_size: tuple[int,int], seed: int=0) -> Tensor:
        return self.forward_features(reset_front_features(rgb,padded_size=padded_size,seed=seed))

    def forward_outputs(self,rgb: Tensor, *, padded_size: tuple[int,int], seed: int=0) -> tuple[Tensor,Tensor]:
        return self.forward_features_outputs(reset_front_features(rgb,padded_size=padded_size,seed=seed))
