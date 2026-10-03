"""Measured SM89 half-texture and five-tap history sampling, correctness path.

The recorded power-of-two square, zero-depth tests define the coordinate contract.
This is not a claim about every CUDA texture format or sampler configuration.
"""
from __future__ import annotations
import os
import torch
from torch import Tensor
from .tensor_math import _exponent


_SCALAR_TENSORS={}
_SCALAR_CACHE_LIMIT=4096
_SCALAR_STATS={'hits':0,'misses':0}
_SCALAR_CACHE_ON=os.environ.get('NR_FMA_SCALAR_CACHE','on').strip().lower() not in ('','off','none','0')
_ALLMERGE_GUARD_ON=os.environ.get('NR_ALLMERGE_GUARD','on').strip().lower() not in ('','off','none','0')
_MOTION_SCOPE_ON=os.environ.get('NR_MOTION_SCOPE','on').strip().lower() not in ('','off','none','0')
_MOTION_SCOPE_TOL=float(os.environ.get('NR_MOTION_SCOPE_TOL') or '0.00390625')


def _scalar_tensor(value,device):
    """Cached scalar -> device tensor. Equal scalars give bit-identical tensors."""
    key=(str(device),type(value).__name__,value)
    cached=_SCALAR_TENSORS.get(key)
    if cached is None:
        if len(_SCALAR_TENSORS)>=_SCALAR_CACHE_LIMIT:_SCALAR_TENSORS.clear()
        cached=_SCALAR_TENSORS[key]=torch.as_tensor(value,device=device,dtype=torch.float32)
        _SCALAR_STATS['misses']+=1
    else:
        _SCALAR_STATS['hits']+=1
    return cached


def _fma32_cached(a: Tensor,b: Tensor|float,c: Tensor|float) -> Tensor:
    """`fma32` with the scalar -> device conversion memoized.

    A Python scalar reaches the device through a host-to-device copy, which on XPU
    blocks the host until the queue reaches it. Call sites repeat the same literals
    every frame, so the copy is paid once and equal scalars then reuse a
    bit-identical zero-dim tensor -- `addcmul` therefore sees identical inputs.
    `NR_FMA_SCALAR_CACHE=off` selects `_fma32_plain` instead.
    """
    a=a.float();device=a.device
    b=_scalar_tensor(b,device) if type(b) in (float,int,bool) else torch.as_tensor(b,device=device,dtype=torch.float32)
    c=_scalar_tensor(c,device) if type(c) in (float,int,bool) else torch.as_tensor(c,device=device,dtype=torch.float32)
    if device.type=='xpu':return torch.addcmul(c,a,b)
    return (a.double()*b.double()+c.double()).float()


def _fma32_plain(a: Tensor,b: Tensor|float,c: Tensor|float) -> Tensor:
    """FP32 fused arithmetic on XPU; CPU double is a bounded diagnostic fallback."""
    a=a.float();b=torch.as_tensor(b,device=a.device,dtype=torch.float32);c=torch.as_tensor(c,device=a.device,dtype=torch.float32)
    if a.device.type=='xpu':return torch.addcmul(c,a,b)
    return (a.double()*b.double()+c.double()).float()


fma32=_fma32_cached if _SCALAR_CACHE_ON else _fma32_plain


def scalar_cache_stats():
    """Scalar-conversion cache counters; `hits` is the number of copies avoided."""
    return dict(_SCALAR_STATS,enabled=_SCALAR_CACHE_ON,entries=len(_SCALAR_TENSORS),
                limit=_SCALAR_CACHE_LIMIT)


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
    """Half interpolation from measured integer 1/256 pixel coefficients.

    Integer coordinates zero every fractional weight, which collapses the
    shared-exponent fixed-point stack onto a single tapped texel; that texel is
    returned directly. Any fractional coordinate keeps the measured path.

    The integer test fuses both axes into one reduction (``(ax==0) & (ay==0)``):
    logically identical to two separate ``.all()`` calls, but one device
    reduction and one host readback instead of two. ``NR_ALLMERGE_GUARD=off``
    restores the two-reduction form so the two can be A/B'd in place.
    """
    h,w=image.shape[:2];cx=cx.clamp(0,(w-1)*256);cy=cy.clamp(0,(h-1)*256)
    hit=(bool((((cx&255)==0)&((cy&255)==0)).all())if _ALLMERGE_GUARD_ON
         else bool(((cx&255)==0).all())and bool(((cy&255)==0).all()))
    if hit:return image[cy>>8,cx>>8].half().float()
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
    """Native cross of five bilinear taps, including its accumulation order.

    Integer motion zeroes every fractional axis weight, so the five taps collapse
    onto the centre tap ``coordinates[2] == (mx, my)`` and ``total`` is exactly 1
    (whose reciprocal is exactly 1.0). The gate tests all six axis terms against
    the sampler's own 1/256-pixel fixed-point grid in one reduction; below that
    grid the sampler's own quantisation already snaps the coordinate to the texel
    centre. Measured residual is 2.5749e-05 px, a 150x margin.

    This is **not** bit-identical: the neighbouring tap still carries a ~1.7e-05
    weight, so the short circuit is an O(1e-5) perturbation of the output.
    Correctness therefore rests on the frame metrics, not on the algebra.
    ``NR_MOTION_SCOPE=off`` restores the five-tap form so the two can be A/B'd in
    place; ``NR_MOTION_SCOPE_TOL`` (default 1/256) sets the gate.
    """
    bx,x0,gx,x3,mx=axis_x;by,y0,gy,y3,my=axis_y
    if _MOTION_SCOPE_ON and bool(((x0.abs()<=_MOTION_SCOPE_TOL)&(x3.abs()<=_MOTION_SCOPE_TOL)&((gx-1).abs()<=_MOTION_SCOPE_TOL)&(y0.abs()<=_MOTION_SCOPE_TOL)&(y3.abs()<=_MOTION_SCOPE_TOL)&((gy-1).abs()<=_MOTION_SCOPE_TOL)).all()):
        if normalized_scale is None:value=half_texture_linear(image,mx,my)
        else:
            h,w=image.shape[:2];iw,ih=normalized_scale
            u=fma32(mx.clamp(.5,w-.5)*iw,w,0)*(1/w)
            v=fma32(my.clamp(.5,h-.5)*ih,h,0)*(1/h)
            value=half_texture_normalized(image,u,v)
        ones=torch.ones_like(gx)
        if return_components:return value,ones
        return value*ones[...,None]
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
