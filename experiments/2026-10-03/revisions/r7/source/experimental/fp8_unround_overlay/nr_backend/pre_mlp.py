"""First pre-block MLP, reconstructed against native SM89 boundaries.

Input/output NHWC32. This excludes front feature generation, attention, pooling,
all later blocks, and temporal state. The implementation prioritizes auditability.
"""
from __future__ import annotations
import torch
from torch import Tensor,nn
from .tensor_math import sm89_f16_dot,cubic_activation
from .execution import current_arithmetic_backend,record_arithmetic_dispatch


def _decode_weights(data: bytes,k: int,n: int) -> Tensor:
    if len(data)!=k*n or k%32 or n%16:raise ValueError('Invalid K32/N16 tile payload')
    # Physical axes: K tile, N tile, lane group, lane thread, N fragment,
    # K register, byte high, byte low. Reorder these exact integer axes.
    physical=torch.frombuffer(bytearray(data),dtype=torch.uint8).reshape(k//32,n//16,8,4,2,2,2,2)
    output=physical.permute(0,5,6,3,7,1,4,2).reshape(k,n)
    decoded=output.view(torch.float8_e4m3fn).to(torch.float16)
    if not torch.isfinite(decoded).all():raise ValueError('Nonfinite weight in observed finite pre-MLP contract')
    return decoded


def quantize_fp8(x: Tensor) -> Tensor:
    """Observed saturating E4M3 boundary, retained as FP16 for portable arithmetic.

    NaN is deliberately not silently replaced. Current finite-input validation
    does not establish the native handling of exceptional values.
    """
    if current_arithmetic_backend()=='triton' and x.device.type=='xpu':
        from .triton_fp8 import quantize_fp8 as fused_quantize
        result=fused_quantize(x)
        record_arithmetic_dispatch('fp8')
        return result
    source=x.to(torch.float16).clamp(-448,448)
    decoded=source.to(torch.float8_e4m3fn).to(torch.float16)
    # The current XPU FP8 conversion drops an existing negative-zero sign.
    # Native SATFINITE preserves it (visible in the ViT cubic expansion).
    # Restore only zero signs; NaNs and nonzero numeric results stay intact.
    bits=decoded.contiguous().view(torch.int16)
    signed_zero=bits | (source.contiguous().view(torch.int16)&-32768)
    return torch.where(decoded==0,signed_zero,bits).contiguous().view(torch.float16)



def cubic_quantize(x: Tensor) -> Tensor:
    """Original cubic followed by E4M3, with explicit optional XPU fusion."""
    if current_arithmetic_backend()=='triton' and x.device.type=='xpu':
        from .triton_cubic_fp8 import direct_cubic_fp8
        result=direct_cubic_fp8(x)
        record_arithmetic_dispatch('cubic_fp8')
        return result
    return quantize_fp8(cubic_activation(x))


class C32MLP(nn.Module):
    """C32 expansion + cubic + contraction with a scaled skip accumulator."""
    def __init__(self,record: bytes, *, skip_scale_offset: int):
        super().__init__()
        self.rounding_family = "c32"
        if skip_scale_offset<8192 or len(record)<skip_scale_offset+64:
            raise ValueError('Truncated C32 MLP weight record')
        self.register_buffer('expansion',_decode_weights(record[:4096],32,128))
        self.register_buffer('contraction',_decode_weights(record[4096:8192],128,32))
        self.register_buffer('skip_scale',torch.frombuffer(bytearray(record[skip_scale_offset:skip_scale_offset+64]),dtype=torch.float16).clone())

    def forward_unquantized(self,projected: Tensor) -> Tensor:
        if projected.shape[-1]!=32 or projected.device!=self.expansion.device:
            raise ValueError('Expected projected features ending in C32 on the module device')
        # MLP rows are independent. Retain only one bounded expansion/cubic
        # workspace instead of several full-resolution C128 intermediates.
        if projected.numel()>32768*32:
            flat=projected.reshape(-1,32);result=torch.empty_like(flat,dtype=torch.float16)
            for start in range(0,len(flat),32768):
                result[start:start+32768]=self.forward_unquantized(flat[start:start+32768])
                if projected.device.type=='xpu' and (start//32768)%8==7:torch.xpu.synchronize(projected.device)
            return result.reshape(projected.shape)
        projected=projected.to(torch.float16)
        from .unround_policy import round_activation
        expanded=sm89_f16_dot(round_activation(self.rounding_family,projected),self.expansion,chunk_k=16)
        from .unround_policy import cubic_activation_family
        hidden=cubic_activation_family(self.rounding_family,expanded)
        initial=(projected*self.skip_scale).to(torch.float16)
        contracted=sm89_f16_dot(hidden,self.contraction,chunk_k=16,initial=initial)
        return contracted

    def forward(self,projected: Tensor) -> Tensor:
        from .unround_policy import round_activation
        return round_activation(self.rounding_family,self.forward_unquantized(projected))


class PreMLP(C32MLP):
    """Pinned pre record's MLP; its 1024-byte front matrix precedes skip scale."""
    def __init__(self,record: bytes):
        if len(record)!=21696:raise ValueError('Expected the pinned 21696-byte pre record')
        super().__init__(record,skip_scale_offset=0x2410)
        self.rounding_family = "pre"
