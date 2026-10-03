"""First pre-block MLP, reconstructed against native SM89 boundaries.

Input/output NHWC32. This excludes front feature generation, attention, pooling,
all later blocks, and temporal state. The implementation prioritizes auditability.
"""
from __future__ import annotations
import os
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


_FP8FAST_OFF=os.environ.get('NR_FP8FAST','on').strip().lower() in ('','off','none','0','false')


def quantize_fp8(x: Tensor) -> Tensor:
    """Observed saturating E4M3 boundary, retained as FP16 for portable arithmetic.

    NaN is deliberately not silently replaced. Current finite-input validation
    does not establish the native handling of exceptional values.
    """
    if current_arithmetic_backend()=='triton' and x.device.type=='xpu':
        if _FP8FAST_OFF:
            from .triton_fp8 import quantize_fp8 as fused_quantize
            result=fused_quantize(x)
        else:
            # 内联 `triton_fp8.quantize_fp8` 的等价形态：同算术、同 grid、同内核，
            # 但省掉一次跨模块 Python 调用帧（实测 0.63 µs/次，技能 183）。
            # 设备守卫不必重复：外层 `x.device.type=='xpu'` 已经把它挡住了。
            from .triton_fp8 import _kernel as fp8_kernel
            source=x.half().contiguous()
            count=source.numel()
            result=torch.empty_like(source)
            if count:
                fp8_kernel[((count+511)>>9,)](source,result,count,512,enable_fp_fusion=False)
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


_GRID_GUARD_OFF=os.environ.get('NR_QUANTTRIM_GUARD','on').strip().lower() in ('','off','none','0','false')


def quantize_fp8_grid(x: Tensor) -> Tensor:
    """`quantize_fp8` for values **already on the E4M3FN grid**.

    Measured (`reference/quanttrim_v1/RESULT.md`): at the block-input / residual /
    skip call sites the incoming tensor is already the output of an upstream
    `quantize_fp8` (possibly through grid-preserving ops), so the conversion is a
    value-level no-op -- 191 of 383 calls per frame returned a bit-identical tensor
    across 16 frames, at 13 call sites, with no intermediate shares.

    Short-circuiting still records the `fp8` arithmetic dispatch, so the ledger is
    unchanged, and skips one allocation plus one kernel launch per call. The two
    cheap preconditions that were measured to hold are re-checked here; anything
    unexpected falls back to the real conversion. `NR_QUANTTRIM_GUARD=off` restores
    the pre-landing form.
    """
    if _GRID_GUARD_OFF or x.dtype!=torch.float16 or not x.is_contiguous():
        return quantize_fp8(x)
    record_arithmetic_dispatch('fp8')
    return x



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
        expanded=sm89_f16_dot(quantize_fp8(projected),self.expansion,chunk_k=16)
        hidden=cubic_quantize(expanded)
        initial=(projected*self.skip_scale).to(torch.float16)
        contracted=sm89_f16_dot(hidden,self.contraction,chunk_k=16,initial=initial)
        return contracted

    def forward(self,projected: Tensor) -> Tensor:
        return quantize_fp8(self.forward_unquantized(projected))


class PreMLP(C32MLP):
    """Pinned pre record's MLP; its 1024-byte front matrix precedes skip scale."""
    def __init__(self,record: bytes):
        if len(record)!=21696:raise ValueError('Expected the pinned 21696-byte pre record')
        super().__init__(record,skip_scale_offset=0x2410)
_PAD_GUARD_OFF=os.environ.get('NR_PADGUARD','on').strip().lower() in ('','off','none','0','false')


def pad_or_identity(x: Tensor,pad) -> Tensor:
    """`F.pad` 的**全零元组短路**。

    pad 元组全零 ⇒ 语义恒等，但仍付一次 dispatch + 一次分配 + 一次真复制
    （`reference/opcensus_v1/RESULT.md`：host 实测 5.54 µs/次，
    对比"什么都不做"的地板价 0.02 µs）。短路后直接返回 `x`。

    等价性：`F.pad` 对全零元组返回的是 `x` 的**逐元素复制**，值与连续性都与 `x` 相同
    （调用点传进来的 `x` 都是 `quantize_fp8` / `quantize_fp8_grid` 的输出，
    按定义连续）⇒ 返回 `x` 本身在值与布局上都一致。

    本条由 `padguard_land_v1` 的**配对逐字节门**背书（on/off 两臂逐帧数组相等）。
    `NR_PADGUARD=off` 逐字复现落地前的形态。
    """
    if _PAD_GUARD_OFF or any(pad):
        return torch.nn.functional.pad(x,pad)
    return x
