"""Reset post block and RGB surface conversion with native size boundaries."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from .c32_block import C32SwinBlock
from .pre_mlp import quantize_fp8 as q,quantize_fp8_grid as qg
from .tensor_math import sm89_f16_dot as dot,half_fma
from .sampling import fma32


def store_unit_half_rz(value: Tensor) -> Tensor:
    """Observed half surface store: saturate RGB, round toward zero."""
    return store_half_rz(value.float().clamp(0,1))


def store_half_rz(value: Tensor) -> Tensor:
    """Signed half store toward zero; temporal overshoot must remain intact."""
    value=value.float();rounded=value.half()
    return (rounded.contiguous().view(torch.int16)-(rounded.float().abs()>value.abs()).to(torch.int16)).view(torch.float16)


class ResetPostBlock(nn.Module):
    """Pinned reset path, gain1/32, no history blend.

    Computation retains unquantized merged/MLP/post features where the fused
    kernel does. The final half head matrix is consumed by HMMA K8 math.
    """
    def __init__(self,record: bytes):
        super().__init__()
        if len(record)!=21808:raise ValueError('Unexpected post record')
        body=record[:8272]+bytes(16)+record[8400:20768]+bytes(16)
        self.body=C32SwinBlock(body,window_shift=(4,4))
        self.register_buffer('input_weight',torch.frombuffer(bytearray(record[8272:8336]),dtype=torch.float16).clone())
        self.register_buffer('input_skip_scale',torch.frombuffer(bytearray(record[8336:8400]),dtype=torch.float16).clone())
        physical=torch.frombuffer(bytearray(record[20784:]),dtype=torch.float16).reshape(2,32,8)
        weight=torch.zeros((32,8),dtype=torch.float16)
        for kt in range(2):
            for lane in range(32):
                for j in range(4):weight[kt*16+2*(lane%4)+j%2+8*(j//2),lane//4]=physical[kt,lane,j]
        self.register_buffer('head_weight',weight)

    def forward_head(self,features: Tensor,skip: Tensor) -> Tensor:
        """Return all eight head channels, including the temporal blend logit."""
        if features.ndim!=3 or skip.ndim!=3 or features.shape[-1]!=32 or skip.shape[-1]!=32:raise ValueError('Expected HWC32 features and skip')
        ph,pw=skip.shape[:2]
        if tuple(features.shape[:2])!=(ph//2,pw//2)or ph%8 or pw%8:raise ValueError('Expected 2x skip dimensions aligned to eight')
        expanded=(qg(features).repeat_interleave(2,0).repeat_interleave(2,1)*self.input_weight).half()
        merged=half_fma(qg(skip),self.input_skip_scale,expanded)
        padded=torch.nn.functional.pad(merged,(0,0,4,4,4,4))
        mlp=self.body.mlp.forward_unquantized(padded);attended=self.body.attention(q(mlp))
        full=dot(q(attended),self.body.output_weight,chunk_k=16,initial=(mlp*self.body.skip_scale).half())[4:4+ph,4:4+pw]
        return dot(full,self.head_weight,chunk_k=8)

    def forward(self,features: Tensor,skip: Tensor,rgb: Tensor,*,previous: Tensor|None=None,sigmoid=None,blend_scale: Tensor|None=None,history_reciprocal: Tensor|None=None,return_float32: bool=False) -> Tensor:
        if rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('Expected HWC3 RGB')
        h,w=rgb.shape[:2]
        if not(0<h<=skip.shape[0] and 0<w<=skip.shape[1]):raise ValueError('RGB must fit within the skip canvas')
        if previous is not None and (previous.shape!=rgb.shape or previous.device!=rgb.device or sigmoid is None or blend_scale is None):
            raise ValueError('Temporal post requires matching history, sigmoid and blend scale')
        if history_reciprocal is not None and (previous is None or history_reciprocal.shape!=rgb.shape[:2]or history_reciprocal.device!=rgb.device):
            raise ValueError('History normalization must match the RGB canvas and device')
        head=self.forward_head(features,skip)[:h,:w]
        # Texture input is half; both factors are exact powers of two.
        value=(head[...,:3].float()*.03125+(rgb.half().float()*.125-.0625))*8+.5
        if previous is not None:
            value=value.clamp(0,1)
            alpha=(sigmoid(head[...,3])*blend_scale.float()).clamp(0,1)[...,None]
            # Native post fuses the five-tap normalization with base subtraction.
            # Rounding the normalized history first changes rare final half stores.
            delta=previous.float()-value if history_reciprocal is None else fma32(previous,history_reciprocal[...,None],-value)
            value=fma32(delta,alpha,value)
            return value if return_float32 else store_half_rz(value)
        return value.clamp(0,1)if return_float32 else store_unit_half_rz(value)
