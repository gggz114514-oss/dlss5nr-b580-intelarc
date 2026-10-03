"""Auditable SM89 F16-accumulator math; correctness path, not a fast XMX kernel.

The shared-exponent rule is experimentally derived from this SM89 model kernel.
Validation reports define its tested scope; it is not a universal CUDA guarantee.
"""
from __future__ import annotations
import torch
from torch import Tensor
from .execution import current_arithmetic_backend,record_arithmetic_dispatch


def half_fma(a: Tensor,b: Tensor|float,c: Tensor|float) -> Tensor:
    """Fused finite-half multiply/add with a single round to FP16.

    A half product is exact in FP32. TwoSum retains the part lost by its FP32
    addition; that residual decides an FP16 midpoint without requiring FP64.
    Casting a CPU FP64 tensor to half also goes through FP32 and cannot serve
    as the oracle for this case. Nonfinite sums keep the ordinary cast result.
    """
    if current_arithmetic_backend()=='triton':
        if a.device.type!='xpu':raise ValueError('Triton arithmetic requires XPU tensors')
        from .triton_half_fma import fused_half_fma
        result=fused_half_fma(a,b,c)
        record_arithmetic_dispatch('half_fma')
        return result
    a,b,c=torch.broadcast_tensors(a.half(),torch.as_tensor(b,device=a.device,dtype=torch.float16),torch.as_tensor(c,device=a.device,dtype=torch.float16))
    limit=1024*1024
    if a.numel()<=limit:return _half_fma_core(a,b,c)
    # Each element is independent. Bound intermediates in full-resolution MLPs
    # and attention; large broadcasts otherwise need several GB of scratch.
    shape=a.shape;af,bf,cf=a.reshape(-1),b.reshape(-1),c.reshape(-1)
    result=torch.empty_like(af)
    for start in range(0,af.numel(),limit):
        sl=slice(start,start+limit);result[sl]=_half_fma_core(af[sl],bf[sl],cf[sl])
    return result.reshape(shape)


def _half_fma_core(a: Tensor,b: Tensor,c: Tensor) -> Tensor:
    product=a.float()*b.float()
    addend=c.float()
    summed=product+addend
    virtual=summed-product
    residual=(product-(summed-virtual))+(addend-virtual)
    rounded=summed.half();bits=rounded.contiguous().view(torch.int16).to(torch.int32)&0xffff
    # The finite-to-infinity midpoint is between 65504 and the virtual 65536.
    rounded_float=torch.where(torch.isinf(rounded),torch.copysign(torch.full_like(summed,65536),summed),rounded.float())
    negative=(bits&0x8000)!=0
    adjacent_bits=bits+torch.where((summed>rounded_float)!=negative,1,-1)
    adjacent=adjacent_bits.to(torch.int16).contiguous().view(torch.float16)
    adjacent_float=torch.where(torch.isinf(adjacent),torch.copysign(torch.full_like(summed,65536),summed),adjacent.float())
    midpoint=(rounded_float+adjacent_float)*.5
    correction=(summed==midpoint)&(residual!=0)&((summed>rounded_float)==(residual>0))
    return torch.where(correction,adjacent,rounded)


def _exponent(x: Tensor) -> Tensor:
    # Inputs are FP16/E4M3 values promoted exactly to normal FP32 (or zero).
    bits=x.contiguous().view(torch.int32)
    exponent=((bits>>23)&255)-127
    return torch.where(x==0,-1000,exponent)


def _fp8_operand_exponent(x: Tensor) -> Tensor:
    # E4M3 subnormal operands retain exponent-field -6 during tensor-core
    # alignment; renormalizing their numerical value to -7/-8/-9 is incorrect.
    return torch.where(x==0,-1000,_exponent(x).clamp(min=-6))


def _scaled_integer_to_half(value: Tensor, scale_exponent: Tensor) -> Tensor:
    """Round integer * 2**scale_exponent once, avoiding an FP32 double round."""
    magnitude=value.abs()
    lead=_exponent(magnitude.float()).clamp(0,62)
    lead=lead-(magnitude<(torch.ones_like(magnitude)<<lead)).to(torch.int32)
    lead=torch.where(magnitude==0,0,lead)
    exponent=lead+scale_exponent
    shift=torch.maximum(exponent-10,torch.full_like(exponent,-24))-scale_exponent
    rs=shift.clamp(0,62)
    quotient=magnitude>>rs
    remainder=magnitude-(quotient<<rs)
    midpoint=torch.ones_like(magnitude)<<(rs-1).clamp(0,61)
    increment=(rs>0)&((remainder>midpoint)|((remainder==midpoint)&((quotient&1)!=0)))
    rounded=quotient+increment.to(torch.int64)
    rounded=torch.where(shift<0,magnitude<<(-shift).clamp(0,62),rounded)
    bits=((exponent+14).clamp(min=0).to(torch.int64)*1024+rounded).clamp(max=0x7c00)
    bits=torch.where(magnitude==0,0,bits) | ((value<0).to(torch.int64)<<15)
    return bits.to(torch.int16).contiguous().view(torch.float16)


def sm89_f16_dot(a: Tensor, weight: Tensor, *, chunk_k: int, rows_per_batch: int=2048, initial: Tensor|None=None) -> Tensor:
    """Logical [...,K] @ [K,N] with observed SM89 half-accumulator rounding.

    Within each K group, align products and the previous accumulator toward zero
    at the largest *sum of operand exponents*, retaining 24 fraction bits for
    HMMA16 (K8 groups) or 13 for QMMA32 (K16 groups). Sum aligned integers, then
    round once to FP16. No host tensor copies. Batching bounds temporary storage.
    The latter requires already decoded E4M3 operands. General CUDA equivalence
    outside the captured finite input range remains unproven.
    """
    if a.device!=weight.device or weight.ndim!=2 or a.shape[-1]!=weight.shape[0]:
        raise ValueError('Expected compatible matrices on one device')
    if chunk_k not in (8,16) or a.shape[-1]%chunk_k or rows_per_batch<1:
        raise ValueError('Expected a whole number of observed K8/K16 groups')
    if initial is not None and (initial.device!=a.device or initial.shape!=(*a.shape[:-1],weight.shape[1])):
        raise ValueError('Initial accumulator must match output shape and device')
    if current_arithmetic_backend()=='triton':
        if a.device.type!='xpu':raise ValueError('Triton arithmetic requires XPU tensors')
        from .triton_math import fused_dot
        result=fused_dot(a,weight,chunk_k=chunk_k,initial=initial)
        record_arithmetic_dispatch('dense')
        return result
    af=a.to(torch.float16).float().reshape(-1,a.shape[-1])
    wf=weight.to(torch.float16).float()
    operand_exponent=_exponent if chunk_k==8 else _fp8_operand_exponent
    ew=operand_exponent(wf)
    fraction_bits=24 if chunk_k==8 else 13
    output=torch.empty((af.shape[0],wf.shape[1]),device=a.device,dtype=torch.float16)
    initial_flat=None if initial is None else initial.to(torch.float16).float().reshape(output.shape)
    for row in range(0,af.shape[0],rows_per_batch):
        aa=af[row:row+rows_per_batch];ea=operand_exponent(aa)
        acc=(torch.zeros((aa.shape[0],wf.shape[1]),device=a.device,dtype=torch.float32)
             if initial_flat is None else initial_flat[row:row+aa.shape[0]])
        for start in range(0,wf.shape[0],chunk_k):
            sl=slice(start,start+chunk_k)
            exponent=torch.maximum((ea[:,sl,None]+ew[None,sl,:]).amax(dim=1),_exponent(acc))
            # All-zero dots use a harmless normal scale rather than overflowing.
            exponent=exponent.clamp(-50,50)
            scale=torch.ldexp(torch.ones_like(acc),fraction_bits-exponent)
            products=aa[:,sl,None]*wf[None,sl,:]
            aligned=(products*scale[:,None,:]).to(torch.int64).sum(dim=1)+(acc*scale).to(torch.int64)
            acc=_scaled_integer_to_half(aligned,exponent-fraction_bits).float()
        output[row:row+aa.shape[0]]=acc.to(torch.float16)
    record_arithmetic_dispatch('dense')
    return output.reshape(*a.shape[:-1],wf.shape[1])


def cubic_activation(x: Tensor) -> Tensor:
    """Native clamp/HFMA2/HMUL2 order, with each fused half result rounded once."""
    x=x.to(torch.float16)
    t=x.clamp(-4,4)
    p=half_fma(-t.abs(),0.055908203125,0.447265625)
    v=half_fma(t,p,0.89453125)
    return (x*v).to(torch.float16)


def sm89_f16_batched_dot(a: Tensor, weight: Tensor, *, initial: Tensor|None=None, batches_per_step: int=32) -> Tensor:
    """QMMA32 shared-exponent path for per-window FP8 matrices [...,M,K] @ [...,K,N]."""
    if a.ndim<3 or weight.ndim!=a.ndim or a.shape[:-2]!=weight.shape[:-2] or a.shape[-1]!=weight.shape[-2]:
        raise ValueError('Expected matching matrix batches')
    if a.device!=weight.device or a.shape[-1]%32 or batches_per_step<1:
        raise ValueError('Expected whole K32 instructions on one device')
    shape=(*a.shape[:-2],a.shape[-2],weight.shape[-1])
    if initial is not None and (initial.shape!=shape or initial.device!=a.device):
        raise ValueError('Initial accumulator must match output')
    if current_arithmetic_backend()=='triton':
        if a.device.type!='xpu':raise ValueError('Triton arithmetic requires XPU tensors')
        from .triton_math import fused_batched_dot
        result=fused_batched_dot(a,weight,initial=initial,batches_per_step=batches_per_step)
        record_arithmetic_dispatch('batched')
        return result
    aa=a.half().float().reshape(-1,a.shape[-2],a.shape[-1])
    ww=weight.half().float().reshape(-1,weight.shape[-2],weight.shape[-1])
    output=torch.empty((aa.shape[0],aa.shape[1],ww.shape[2]),dtype=torch.float16,device=a.device)
    init=None if initial is None else initial.half().float().reshape(output.shape)
    # Bound broadcast product temporaries for full-frame ViT attention. Batch
    # and query rows are independent; each output keeps its original K order.
    product_budget=2*1024*1024
    batches_per_step=min(batches_per_step,max(1,product_budget//(aa.shape[1]*ww.shape[2]*16)))
    rows_per_step=max(1,product_budget//(batches_per_step*ww.shape[2]*16))
    for batch in range(0,aa.shape[0],batches_per_step):
        wv=ww[batch:batch+batches_per_step];ew=_fp8_operand_exponent(wv)
        for row in range(0,aa.shape[1],rows_per_step):
            av=aa[batch:batch+batches_per_step,row:row+rows_per_step];ea=_fp8_operand_exponent(av)
            acc=torch.zeros((av.shape[0],av.shape[1],wv.shape[2]),device=a.device) if init is None else init[batch:batch+av.shape[0],row:row+av.shape[1]]
            for k in range(0,av.shape[-1],16):
                sl=slice(k,k+16)
                ex=torch.maximum((ea[...,sl,None]+ew[:,None,sl,:]).amax(dim=-2),_exponent(acc)).clamp(-50,50)
                scale=torch.ldexp(torch.ones_like(acc),13-ex)
                prod=av[...,sl,None]*wv[:,None,sl,:]
                value=(prod*scale[...,None,:]).to(torch.int64).sum(-2)+(acc*scale).to(torch.int64)
                acc=_scaled_integer_to_half(value,ex-13).float()
            output[batch:batch+av.shape[0],row:row+av.shape[1]]=acc.half()
    record_arithmetic_dispatch('batched')
    return output.reshape(shape)
