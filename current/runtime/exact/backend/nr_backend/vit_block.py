"""Native-reference ViT math for observed image-size contracts."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from .pre_mlp import cubic_quantize,_decode_weights,quantize_fp8 as q
from .tensor_math import sm89_f16_dot as dot,sm89_f16_batched_dot,cubic_activation,half_fma
from .attention import normalize_c32,attention_row_sum64
from .execution import current_arithmetic_backend,record_arithmetic_dispatch


def split_k_projection(features: Tensor,weight: Tensor,initial: Tensor,parts: int=4) -> Tensor:
    """Independent K partitions; skip in partition zero, sequential half merge."""
    size=features.shape[-1]//parts
    if size*parts!=features.shape[-1]:raise ValueError('K must divide into native partitions')
    result=dot(features[...,:size],weight[:size],chunk_k=16,initial=initial)
    for i in range(1,parts):result=(result+dot(features[...,i*size:(i+1)*size],weight[i*size:(i+1)*size],chunk_k=16)).half()
    return result


def vit_exponential(scores: Tensor) -> Tensor:
    if current_arithmetic_backend() == 'triton' and scores.device.type == 'xpu':
        from .triton_attention_exp import exponential
        result = exponential(scores, vit=True)
        record_arithmetic_dispatch('attention_exp_vit')
        return result
    affine=half_fma(scores,.08953857421875,1.708984375).clamp(1.439453125,1.9775390625)
    bits=affine.contiguous().view(torch.int16).int()&0xffff
    return (((bits<<4)+0x4000)&0xffff).to(torch.int16).contiguous().view(torch.float16)


def vit_attention(query: Tensor,key: Tensor,value: Tensor) -> Tensor:
    """Native 64-key streaming reductions, including padded-key correction.

    Missing keys contribute exp(0) to the rounded denominator before a final
    subtraction. Masking them before the reduction changes half rounding.
    Numerator accumulation continues through successive 64-key chunks.
    """
    tokens=key.shape[-2];padding=(-tokens)%64
    key=torch.nn.functional.pad(key,(0,0,0,padding))
    value=torch.nn.functional.pad(value,(0,0,0,padding))
    e=vit_exponential(sm89_f16_batched_dot(query,key.transpose(-1,-2)))
    numerator=sm89_f16_batched_dot(q(e),value)
    total=attention_row_sum64(e[...,:64])
    for start in range(64,e.shape[-1],64):total=(total+attention_row_sum64(e[...,start:start+64])).half()
    if padding:
        correction=(vit_exponential(torch.zeros((),device=e.device)).float()*padding).half()
        total=(total-correction).half()
    reciprocal=total.clamp(min=0.00006198883056640625).float().reciprocal().half()
    return q((numerator*reciprocal).half())


class VitBlock(nn.Module):
    """Five original ViT calls, exposing seven computed output boundaries.

    The observed 1D attention kernel does not read its two-byte scalar record;
    its exponent coefficient is an instruction literal. Keep the record as
    metadata rather than injecting an unsupported extra multiplication.
    """
    def __init__(self,records: list[bytes]):
        super().__init__()
        if list(map(len,records))!=[4194320,4196352,3145856,2,1050624]:raise ValueError('Unexpected ViT records')
        self.register_buffer('expand',_decode_weights(records[0][:4194304],1024,4096))
        self.register_buffer('contract',_decode_weights(records[1][:4194304],4096,1024))
        self.register_buffer('ffn_skip',torch.frombuffer(bytearray(records[1][4194304:]),dtype=torch.float16).clone())
        self.register_buffer('qkv_weight',_decode_weights(records[2][128:],1024,3072))
        self.register_buffer('query_scale',torch.frombuffer(bytearray(records[2][:128]),dtype=torch.float32).half().clone())
        self.register_buffer('attention_record_scalar',torch.frombuffer(bytearray(records[3]),dtype=torch.float16).clone())
        self.register_buffer('projection',_decode_weights(records[4][:1048576],1024,1024))
        self.register_buffer('attn_skip',torch.frombuffer(bytearray(records[4][1048576:]),dtype=torch.float16).clone())

    def forward_boundaries(self,features: Tensor) -> tuple[Tensor,...]:
        if features.ndim!=2 or features.shape[0]not in(64,96,128,640,960)or features.shape[1]!=1024:raise ValueError('Expected observed 64/96/128/640/960 token contract, C1024')
        tokens=features.shape[0]
        x=q(features);hidden=cubic_quantize(dot(x,self.expand,chunk_k=16))
        mlp=q(split_k_projection(hidden,self.contract,(x*self.ffn_skip).half()))
        z=(dot(mlp[:,:512],self.qkv_weight[:512],chunk_k=16)+dot(mlp[:,512:],self.qkv_weight[512:],chunk_k=16)).half().reshape(tokens,32,3,32)
        query=q((normalize_c32(z[:,:,0])*5.65625).half()*self.query_scale[None,:,None])
        key=q(normalize_c32(z[:,:,1]));value=q(z[:,:,2])
        Q,K,V=query.transpose(0,1),key.transpose(0,1),value.transpose(0,1)
        attended=vit_attention(Q,K,V).transpose(0,1).reshape(tokens,1024)
        output=q(split_k_projection(attended,self.projection,(mlp*self.attn_skip).half()))
        return hidden,mlp,query,key,value,attended,output

    def forward(self,features: Tensor) -> Tensor:return self.forward_boundaries(features)[-1]
