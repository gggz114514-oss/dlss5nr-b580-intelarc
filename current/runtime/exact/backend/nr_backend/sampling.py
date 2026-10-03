"""Measured SM89 half-texture and five-tap history sampling, correctness path.

The recorded power-of-two square, zero-depth tests define the coordinate contract.
This is not a claim about every CUDA texture format or sampler configuration.
"""
from __future__ import annotations
import torch
from torch import Tensor
from .tensor_math import _exponent


def fma32(a: Tensor,b: Tensor|float,c: Tensor|float) -> Tensor:
    """FP32 fused arithmetic on XPU; CPU double is a bounded diagnostic fallback."""
    a=a.float();b=torch.as_tensor(b,device=a.device,dtype=torch.float32);c=torch.as_tensor(c,device=a.device,dtype=torch.float32)
    if a.device.type=='xpu':return torch.addcmul(c,a,b)
    return (a.double()*b.double()+c.double()).float()


def _integer_half_away(value: Tensor,scale_exponent: Tensor) -> Tensor:
    """Exact signed fixed-point -> half, nearest with ties away from zero."""
    magnitude=value.abs();lead=_exponent(magnitude.float()).clamp(0,62)
    lead=lead-(magnitude<(torch.ones_like(magnitude)<<lead)).int();lead=torch.where(magnitude==0,0,lead)
    exponent=lead+scale_exponent
    shift=torch.maximum(exponent-10,torch.full_like(exponent,-24))-scale_exponent;rs=shift.clamp(0,62)
    quotient=magnitude>>rs;remainder=magnitude-(quotient<<rs);midpoint=torch.ones_like(magnitude)<<(rs-1).clamp(0,61)
    rounded=quotient+((rs>0)&(remainder>=midpoint)).long()
    rounded=torch.where(shift<0,magnitude<<(-shift).clamp(0,62),rounded)
    bits=((exponent+14).clamp(min=0).long()*1024+rounded).clamp(max=0x7c00)
    bits=torch.where(magnitude==0,0,bits)|((value<0).long()<<15)
    return bits.to(torch.int16).contiguous().view(torch.float16)


def half_texture_linear(image: Tensor,x: Tensor,y: Tensor) -> Tensor:
    """Pixel-center coordinates, clamp addressing, measured half texture math.

    Fractions and the bilinear cross weight use eight fractional bits. Derive
    the other three weights by subtraction. Active texels align toward zero
    with 14 fractional significand bits at their largest exponent. The weighted
    sum rounds to half with ties away from zero, then expands to FP32.
    """
    h,w=image.shape[:2];x=x.float().clamp(.5,w-.5)-.5;y=y.float().clamp(.5,h-.5)-.5
    ix=x.floor().long();iy=y.floor().long()
    ax=torch.floor((x-ix.float())*256+.5).long();ay=torch.floor((y-iy.float())*256+.5).long()
    return _half_texture_counts(image,ix*256+ax,iy*256+ay)


def _half_texture_counts(image: Tensor,cx: Tensor,cy: Tensor) -> Tensor:
    """Half interpolation from measured integer 1/256 pixel coefficients."""
    h,w=image.shape[:2];cx=cx.clamp(0,(w-1)*256);cy=cy.clamp(0,(h-1)*256)
    ix=cx>>8;iy=cy>>8;jx=(ix+1).clamp(max=w-1);jy=(iy+1).clamp(max=h-1);ax=cx&255;ay=cy&255
    cross=(ax*ay+128)>>8
    weights=torch.stack((256-ax-ay+cross,ax-cross,ay-cross,cross),-1)[...,None,:]
    pixels=torch.stack((image[iy,ix],image[iy,jx],image[jy,ix],image[jy,jx]),-1).half().float()
    exponent=torch.where((pixels!=0)&(weights!=0),_exponent(pixels.abs()),-1000).amax(-1).clamp(min=-24)
    scale=torch.ldexp(torch.ones_like(exponent,dtype=torch.float32),14-exponent)[...,None]
    aligned=(pixels*scale).long();summed=(aligned*weights).sum(-1)
    value=_integer_half_away(summed,exponent-22)
    negative_zero=(summed==0)&((weights==0)|torch.signbit(pixels)).all(-1)
    bits=value.contiguous().view(torch.int16)|(negative_zero.to(torch.int16)<<15)
    return bits.contiguous().view(torch.float16).float()


def half_texture_normalized(image: Tensor,u: Tensor,v: Tensor) -> Tensor:
    """Measured normalized-coordinate conversion: floor to 21 fractional bits.

    Multiply by texture extent in integer arithmetic and round coefficients to
    1/256 with ties upward. Clamping occurs before texel addressing.
    """
    h,w=image.shape[:2]
    nx=torch.floor(u.float()*(2**21)).long();ny=torch.floor(v.float()*(2**21)).long()
    return _half_texture_counts(image,((nx*w+4096)>>13)-128,((ny*h+4096)>>13)-128)


def _axis_weights(position: Tensor,reciprocal_source=None) -> tuple[Tensor,...]:
    base=(position-.5).floor();center=base+.5;t=(position-center).clamp(0,1)
    return _axis_weights_fraction(center,t,reciprocal_source)


def _axis_weights_fraction(center: Tensor,t: Tensor,reciprocal_source=None) -> tuple[Tensor,...]:
    t2=t*t;t3=t*t2
    w0=fma32(t+t3,-.5,t2);w1=fma32(t3,1.5,-(t2*2.5))+1;w3=(t3-t2)*.5
    w2=((1-w0)-w1)-w3;group=w1+w2
    inverse=group.reciprocal()if reciprocal_source is None else reciprocal_source(group)
    middle=fma32(w2,inverse,center)
    return center,w0,group,w3,middle


def history_sample_five(image: Tensor,x: Tensor,y: Tensor,*,return_components: bool=False,reciprocal_source=None) -> Tensor|tuple[Tensor,Tensor]:
    """Native cross of five bilinear taps, including its accumulation order."""
    bx,x0,gx,x3,mx=_axis_weights(x,reciprocal_source);by,y0,gy,y3,my=_axis_weights(y,reciprocal_source)
    return _sample_five_axes(image,(bx,x0,gx,x3,mx),(by,y0,gy,y3,my),return_components=return_components,reciprocal_source=reciprocal_source)


def _sample_five_axes(image: Tensor,axis_x,axis_y,*,return_components=False,reciprocal_source=None,normalized_scale=None):
    bx,x0,gx,x3,mx=axis_x;by,y0,gy,y3,my=axis_y
    weights=(x0*gy,y0*gx,gx*gy,y3*gx,x3*gy)
    coordinates=((bx-1,my),(mx,by-1),(mx,my),(mx,by+2),(bx+2,my))
    total=((weights[0]+weights[1])+weights[2])+weights[3];total=total+weights[4]
    def sample(x,y):
        if normalized_scale is None:return half_texture_linear(image,x,y)
        h,w=image.shape[:2];iw,ih=normalized_scale
        # Preserve model-normalized -> region -> backing texture normalization.
        u=fma32(x.clamp(.5,w-.5)*iw,w,0)*(1/w)
        v=fma32(y.clamp(.5,h-.5)*ih,h,0)*(1/h)
        return half_texture_normalized(image,u,v)
    value=sample(*coordinates[1])*weights[1][...,None]
    for i in(0,2,3,4):value=fma32(sample(*coordinates[i]),weights[i][...,None],value)
    reciprocal=total.reciprocal()if reciprocal_source is None else reciprocal_source(total)
    if return_components:return value,reciprocal
    return value*reciprocal[...,None]


def warp_history_normalized(image: Tensor,motion: Tensor,*,return_components=False,reciprocal_source,dimension_reciprocal):
    """Default zero-origin region mapping, preserving native normalized FMAs."""
    h,w=image.shape[:2]
    if image.ndim!=3 or image.shape[-1]!=3 or motion.shape!=(h,w,2)or image.device!=motion.device:raise ValueError('Expected matching RGB history and RG pixel motion')
    if dimension_reciprocal is None:raise ValueError('Normalized history requires native dimension reciprocals')
    iw,ih=dimension_reciprocal(w),dimension_reciprocal(h)
    if iw.device!=image.device:raise ValueError('History and dimension reciprocal table must share a device')
    y,x=torch.meshgrid(torch.arange(h,device=image.device,dtype=torch.float32),torch.arange(w,device=image.device,dtype=torch.float32),indexing='ij')
    mv=motion.half().float();u=fma32(mv[...,0],1/w,(x+.5)*iw);v=fma32(mv[...,1],1/h,(y+.5)*ih)
    def axis(norm,extent):
        center=fma32(norm,extent,-.5).floor()+.5
        return _axis_weights_fraction(center,fma32(norm,extent,-center).clamp(0,1),reciprocal_source)
    return _sample_five_axes(image,axis(u,w),axis(v,h),return_components=return_components,reciprocal_source=reciprocal_source,normalized_scale=(iw,ih))


def warp_history_square(image: Tensor,motion: Tensor,*,return_components: bool=False,reciprocal_source=None) -> Tensor|tuple[Tensor,Tensor]:
    """Current-to-previous pixel motion for the observed 256/512 square inputs.

    Power-of-two normalized texture coordinates retain the observed pixel-space
    arithmetic. Non-power-of-two sizes require separate native validation.
    """
    if image.ndim!=3 or image.shape[-1]!=3 or tuple(image.shape[:2])not in((256,256),(512,512))or motion.shape!=(*image.shape[:2],2)or image.device!=motion.device:
        raise ValueError('Expected same-device 256/512-square RGB history and RG pixel motion')
    h,w=image.shape[:2]
    y,x=torch.meshgrid(torch.arange(h,device=image.device,dtype=torch.float32),torch.arange(w,device=image.device,dtype=torch.float32),indexing='ij')
    mv=motion.half().float()
    return history_sample_five(image,x+.5+mv[...,0],y+.5+mv[...,1],return_components=return_components,reciprocal_source=reciprocal_source)


def warp_history_256(image: Tensor,motion: Tensor,*,return_components: bool=False,reciprocal_source=None) -> Tensor|tuple[Tensor,Tensor]:
    """Compatibility helper retaining the strict 256-square contract."""
    if image.shape!=(256,256,3)or motion.shape!=(256,256,2):raise ValueError('Expected 256x256 RGB history and RG pixel motion')
    return warp_history_square(image,motion,return_components=return_components,reciprocal_source=reciprocal_source)
