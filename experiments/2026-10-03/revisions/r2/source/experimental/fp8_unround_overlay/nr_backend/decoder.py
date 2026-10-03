"""Decoder transitions recovered from native kernels and reference boundaries."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from .pre_mlp import _decode_weights,quantize_fp8 as q
from .unround_policy import round_activation
from .vit_block import split_k_projection
from .tensor_math import sm89_f16_dot as dot,half_fma
from .multihead_block import MultiHeadSwinBlock
from .c32_block import C32SwinBlock


class DecoderInputUpsample(nn.Module):
    """C1024->C512 split-K projection, nearest 2x expansion and fused skip.

    The skip determines the crop of the padded projection. Native output uses
    one half rounding after scale*skip+projection, then the FP8 boundary.
    """
    def __init__(self,record: bytes):
        super().__init__()
        if len(record)!=525312:raise ValueError('Unexpected decoder input record')
        self.register_buffer('weight',_decode_weights(record[:524288],1024,512))
        self.register_buffer('skip_scale',torch.frombuffer(bytearray(record[524288:]),dtype=torch.float16).clone())

    def forward(self,features: Tensor,skip: Tensor) -> Tensor:
        if features.ndim!=3 or features.shape[-1]!=1024 or skip.ndim!=3 or skip.shape[-1]!=512:raise ValueError('Expected HWC1024 input and HWC512 skip')
        h,w=skip.shape[:2]
        if min(h,w)==0 or h>2*features.shape[0] or w>2*features.shape[1]:raise ValueError('Skip must fit the expanded projection')
        initial=torch.zeros((*features.shape[:2],512),device=features.device,dtype=torch.float16)
        projected=split_k_projection(round_activation("c512",features),self.weight,initial)
        expanded=projected.repeat_interleave(2,0).repeat_interleave(2,1)[:h,:w]
        return round_activation("c512",half_fma(round_activation("c512",skip),self.skip_scale,expanded))


class DecoderUpsampleSwin(nn.Module):
    """Nearest upsample and skip fusion followed by a C32/64/128/256 block.

    The transition matrix follows the MLP matrices, not the record prefix.
    Repack unchanged body bytes to its existing parser; inserted bytes are
    alignment only and are not interpreted as weights.
    """
    def __init__(self,record: bytes,channels: int,*,window_shift: tuple[int,int]=(0,0)):
        super().__init__()
        sizes={32:22784,64:70048,128:230176,256:820784}
        if channels not in sizes or len(record)!=sizes[channels]:raise ValueError('Unexpected upsample record')
        self.channels=channels
        end=8192 if channels==32 else (channels//32)*(channels*128+4096+32*channels)
        upend=end+2*channels*channels
        if channels==32:
            scale_at=upend+96;qkv_at=upend+160
            body=record[:end]+record[upend:upend+80]+bytes(16)+record[qkv_at:]
            self.body=C32SwinBlock(body,window_shift=window_shift)
        else:
            scale_at=upend+channels*2;qkv_at=upend+channels*4
            body=record[:end]+bytes(16)+record[upend:scale_at]+bytes(16)+record[qkv_at:]
            self.body=MultiHeadSwinBlock(body,channels,window_shift=window_shift)
        self.register_buffer('weight',_decode_weights(record[end:upend],2*channels,channels))
        self.register_buffer('input_skip_scale',torch.frombuffer(bytearray(record[scale_at:scale_at+2*channels]),dtype=torch.float16).clone())

    def forward(self,features: Tensor,skip: Tensor) -> Tensor:
        if features.ndim!=3 or features.shape[-1]!=2*self.channels or skip.ndim!=3 or skip.shape[-1]!=self.channels:raise ValueError('Unexpected upsample tensor shapes')
        h,w=skip.shape[:2]
        if min(h,w)==0 or h>2*features.shape[0] or w>2*features.shape[1]:raise ValueError('Skip must fit the expanded projection')
        family = "c32" if self.channels == 32 else f"c{self.channels}"
        projected=dot(round_activation(family,features),self.weight,chunk_k=16)
        expanded=projected.repeat_interleave(2,0).repeat_interleave(2,1)[:h,:w]
        merged=half_fma(round_activation(family,skip),self.input_skip_scale,expanded)
        if self.channels!=32:return self.body(round_activation(family,merged))
        # Like pre, this C32 fused kernel retains the unquantized merge for
        # its MLP residual, while matrix operands cross the FP8 boundary.
        sy,sx=self.body.window_shift
        padded=torch.nn.functional.pad(merged,(0,0,sx,(-w-sx)%8,sy,(-h-sy)%8))
        mlp=self.body.mlp.forward_unquantized(padded)
        attended=self.body.attention(round_activation(family,mlp))
        value=dot(round_activation(family,attended),self.body.output_weight,chunk_k=16,initial=(mlp*self.body.skip_scale).half())
        return round_activation(family,value[sy:sy+h,sx:sx+w])
