"""Reset-frame front producer reconstructed from the independently replayed SM89 kernel.

Observed contract: SDR RGB16F, no temporal/depth/motion textures, fixed captured
style/strength controls. Output is NHWC16 FP16, before the first projection.
The optional NativeNoiseTable supplies the complete native scalar domain for
the three noise channels. Without it, portable transcendental approximations
remain experimental. The RGB executor always supplies the validated table.
"""
from __future__ import annotations
import torch
from torch import Tensor

MASK32 = 0xffffffff

def _permute(value: Tensor) -> Tensor:
    return (((value >> ((value >> 28) + 4)) ^ value) * 0x108ef2d9) & MASK32

def _uniform_indices(x: Tensor, y: Tensor, seed: int) -> list[Tensor]:
    value = ((x * 0x8da6b343) & MASK32) ^ ((y * 0xd8163841) & MASK32)
    value = value ^ ((seed * 0x9e3779b9) & MASK32) ^ 0x243f6a88
    value = _permute(value)
    value = value ^ (value >> 22)
    result=[]
    for multiplier,increment in ((0xcaa5b80d,0x21dd796b),(0x83232c31,0x3463e0ac),
                                  (0x2c9277b5,0xac564b05),(0xfa6dc5f9,0x4712a88e)):
        item = _permute((value * multiplier + increment) & MASK32)
        result.append((item >> 30) ^ (item >> 8))
    return result


def _uniforms(x: Tensor, y: Tensor, seed: int) -> list[Tensor]:
    return [(i+1).to(torch.float32)*(2**-24)for i in _uniform_indices(x,y,seed)]

def _positive_multiply_rz(a: Tensor, b_bits: int) -> Tensor:
    """Exact positive normal FP32 multiply toward zero, using integer significands.

    The captured angle range stays normal. Avoids FP64, which is unsuitable for
    the B580 execution path. This helper is not a general-purpose float multiply.
    """
    bits=a.contiguous().view(torch.int32).to(torch.int64)
    mantissa=(bits & 0x7fffff) | 0x800000
    product=mantissa * ((b_bits & 0x7fffff) | 0x800000)
    high=product >= (1 << 47)
    exponent=((bits >> 23) & 255) + ((b_bits >> 23) & 255) - 127 + high.to(torch.int64)
    rounded_mantissa=product >> (23+high.to(torch.int64))
    output=((exponent << 23) | (rounded_mantissa & 0x7fffff)).to(torch.int32)
    return output.contiguous().view(torch.float32)

def _cycle(uniform: Tensor) -> Tensor:
    radians=uniform * 6.2831854820251464844
    return _positive_multiply_rz(radians,0x3e22f983)

def _radius(uniform: Tensor) -> Tensor:
    return torch.sqrt((torch.log2(uniform) * 0.69314718246459960938) * -2.0)

def reset_front_features(rgb: Tensor, *, padded_size: tuple[int,int], seed: int=0,noise_source=None) -> Tensor:
    """Return NHWC16 reset-front features on the input device.

    rgb is HWC3/4 sRGB proxy data; the function quantizes its input to FP16 as the
    reference texture does. padded_size is (height,width) from the recorded model
    contract, not inferred from an unverified padding rule. No CPU frame transfer.
    """
    if rgb.ndim!=3 or rgb.shape[-1] not in (3,4) or not rgb.is_floating_point():
        raise ValueError('Expected floating point HWC RGB/RGBA input')
    height,width=rgb.shape[:2]
    ph,pw=padded_size
    if not (height<=ph<=2*height-1 and width<=pw<=2*width-1):
        raise ValueError('Only the observed single-reflection padding range is supported')
    if not 0<=seed<=MASK32: raise ValueError('Seed must be uint32')
    y,x=torch.meshgrid(torch.arange(ph,device=rgb.device,dtype=torch.int64),
                       torch.arange(pw,device=rgb.device,dtype=torch.int64),indexing='ij')
    if noise_source is None:
        ua,ub,uc,ud=_uniforms(x,y,seed)
        ra,rc=_radius(ua),_radius(uc)
        ab,ad=_cycle(ub)*6.283185307179586,_cycle(ud)*6.283185307179586
        noise=torch.stack((rc*torch.cos(ad),rc*torch.sin(ad),ra*torch.cos(ab)),dim=-1).to(torch.float16)
    else:noise=noise_source(x,y,seed)
    sx=torch.where(x<width,x,2*width-2-x)
    sy=torch.where(y<height,y,2*height-2-y)
    sample=rgb.to(torch.float16)[sy,sx,:3]
    scaled=((sample-0.5).to(torch.float16)*0.125).to(torch.float16)
    output=torch.zeros((ph,pw,16),device=rgb.device,dtype=torch.float16)
    output[...,:3]=noise
    output[...,3]=1
    output[...,4:7]=scaled
    output[...,7:10]=scaled
    output[...,11:13]=1
    output[...,13:15]=-1
    return output

def project_front_features(features: Tensor, weight: Tensor, *, reference_math: bool=True) -> Tensor:
    """Observed SM89 HMMA16 front projection, including its intermediate FP16 round.

    weight is logical K16 x N32, decoded from the two serialized front tiles.
    A conventional single FP32-accumulating dot omits the rounding after K0..7.
    The default uses the validated shared-exponent implementation. The optional
    ordinary matmul path retains measured small deviations and is experimental.
    """
    if features.shape[-1]!=16 or weight.shape!=(16,32):
        raise ValueError('Expected NHWC16 features and logical 16x32 projection weights')
    if features.device!=weight.device:raise ValueError('Features and weights must share a device')
    if reference_math:
        from .tensor_math import sm89_f16_dot
        return sm89_f16_dot(features,weight,chunk_k=8)
    features=features.to(torch.float16);weight=weight.to(torch.float16)
    first=(features[...,:8] @ weight[:8]).to(torch.float16)
    second=features[...,8:].float() @ weight[8:].float()
    return (first.float()+second).to(torch.float16)


def zero_motion_front_features(rgb: Tensor, previous: Tensor, *, padded_size: tuple[int,int], seed: int, noise_source) -> Tensor:
    """Observed temporal front: previous final NR RGB, with zero motion/depth.

    History sampling uses the same reflected coordinates as current RGB. This
    contract deliberately does not approximate nonzero motion-vector sampling.
    """
    if previous.shape!=rgb.shape or previous.device!=rgb.device:
        raise ValueError('History and current RGB must share shape and device')
    output=reset_front_features(rgb,padded_size=padded_size,seed=seed,noise_source=noise_source)
    h,w=rgb.shape[:2];ph,pw=padded_size
    y=torch.arange(ph,device=rgb.device);x=torch.arange(pw,device=rgb.device)
    y=torch.where(y<h,y,2*h-2-y);x=torch.where(x<w,x,2*w-2-x)
    sample=previous.half()[y[:,None],x[None,:]]
    output[...,7:10]=((sample-.5).half()*.125).half()
    return output
