"""Fused square five-tap history sampling with the pinned native table/math.

Keep float32 FMA order, integer1/256 bilinear weights, active-texel14-bit
alignment, signed ties-away half conversion, negative zero, and all three
reciprocal-domain flags. Original image half conversion occurs before sampling.
This first experiment accepts contiguous half256/512 RGB and half/float motion.
"""
import torch
import triton
import triton.language as tl
from nr_backend.reciprocal import START_BITS,END_BITS


@triton.jit
def exponent(value):
    return tl.where(value==0,-1000,((value.to(tl.int32,bitcast=True)>>23)&255)-127)


@triton.jit
def integer_half_away(value,scale_exponent):
    magnitude=tl.abs(value)
    lead=tl.minimum(tl.maximum(exponent(magnitude.to(tl.float32)),0),62)
    one=tl.full((),1,tl.int64)
    lead=lead-(magnitude<(one<<lead)).to(tl.int32)
    lead=tl.where(magnitude==0,0,lead)
    e=lead+scale_exponent
    shift=tl.maximum(e-10,-24)-scale_exponent
    rs=tl.minimum(tl.maximum(shift,0),62)
    quotient=magnitude>>rs;remainder=magnitude-(quotient<<rs)
    midpoint=one<<tl.minimum(tl.maximum(rs-1,0),61)
    rounded=quotient+((rs>0)&(remainder>=midpoint)).to(tl.int64)
    rounded=tl.where(shift<0,magnitude<<tl.minimum(tl.maximum(-shift,0),62),rounded)
    bits=tl.minimum(tl.maximum(e+14,0).to(tl.int64)*1024+rounded,0x7c00)
    return tl.where(magnitude==0,0,bits)|((value<0).to(tl.int64)<<15)


@triton.jit
def reciprocal(value,TABLE,SIZE:tl.constexpr):
    index=value.to(tl.int32,bitcast=True)-0x3f780000
    valid=(index>=0)&(index<SIZE)
    result=tl.load(TABLE+tl.minimum(tl.maximum(index,0),SIZE-1))
    return result,valid


@triton.jit
def axis(position,TABLE,SIZE:tl.constexpr):
    center=tl.floor(position-.5)+.5
    t=tl.minimum(tl.maximum(position-center,0.),1.)
    t2=t*t;t3=t*t2
    w0=tl.fma(t+t3,-.5,t2)
    w1=tl.fma(t3,1.5,-(t2*2.5))+1.
    w3=(t3-t2)*.5
    w2=((1.-w0)-w1)-w3
    group=w1+w2
    inverse,valid=reciprocal(group,TABLE,SIZE)
    middle=tl.fma(w2,inverse,center)
    return center,w0,group,w3,middle,valid


@triton.jit
def sample(IMAGE,x,y,mask,H:tl.constexpr,W:tl.constexpr,C:tl.constexpr):
    # Clamp first, then round each fraction upward at a half coefficient step.
    x=tl.minimum(tl.maximum(x,.5),W-.5)-.5
    y=tl.minimum(tl.maximum(y,.5),H-.5)-.5
    ix=tl.floor(x).to(tl.int64);iy=tl.floor(y).to(tl.int64)
    ax=tl.floor((x-ix.to(tl.float32))*256.+.5).to(tl.int64)
    ay=tl.floor((y-iy.to(tl.float32))*256.+.5).to(tl.int64)
    cx=tl.minimum(tl.maximum(ix*256+ax,0),(W-1)*256)
    cy=tl.minimum(tl.maximum(iy*256+ay,0),(H-1)*256)
    ix=cx>>8;iy=cy>>8;jx=tl.minimum(ix+1,W-1);jy=tl.minimum(iy+1,H-1)
    ax=cx&255;ay=cy&255;cross=(ax*ay+128)>>8
    w0=256-ax-ay+cross;w1=ax-cross;w2=ay-cross;w3=cross
    channel=tl.arange(0,C)
    valid=mask[:,None]&(channel[None,:]<3)
    p0=tl.load(IMAGE+(iy*W+ix)[:,None]*3+channel[None,:],valid,other=0).to(tl.float32)
    p1=tl.load(IMAGE+(iy*W+jx)[:,None]*3+channel[None,:],valid,other=0).to(tl.float32)
    p2=tl.load(IMAGE+(jy*W+ix)[:,None]*3+channel[None,:],valid,other=0).to(tl.float32)
    p3=tl.load(IMAGE+(jy*W+jx)[:,None]*3+channel[None,:],valid,other=0).to(tl.float32)
    e0=tl.where((p0!=0)&(w0[:,None]!=0),exponent(tl.abs(p0)),-1000)
    e1=tl.where((p1!=0)&(w1[:,None]!=0),exponent(tl.abs(p1)),-1000)
    e2=tl.where((p2!=0)&(w2[:,None]!=0),exponent(tl.abs(p2)),-1000)
    e3=tl.where((p3!=0)&(w3[:,None]!=0),exponent(tl.abs(p3)),-1000)
    e=tl.maximum(tl.maximum(tl.maximum(e0,e1),tl.maximum(e2,e3)),-24)
    scale=((127+14-e)<<23).to(tl.float32,bitcast=True)
    total=(p0*scale).to(tl.int64)*w0[:,None]+(p1*scale).to(tl.int64)*w1[:,None]
    total=total+(p2*scale).to(tl.int64)*w2[:,None]+(p3*scale).to(tl.int64)*w3[:,None]
    bits=integer_half_away(total,e-22)
    nz=(total==0)&((w0[:,None]==0)|(p0.to(tl.int32,bitcast=True)<0))
    nz=nz&((w1[:,None]==0)|(p1.to(tl.int32,bitcast=True)<0))
    nz=nz&((w2[:,None]==0)|(p2.to(tl.int32,bitcast=True)<0))
    nz=nz&((w3[:,None]==0)|(p3.to(tl.int32,bitcast=True)<0))
    bits=bits|(nz.to(tl.int64)<<15)
    return bits.to(tl.uint16).to(tl.float16,bitcast=True).to(tl.float32)


@triton.jit
def _kernel(IMAGE,MOTION,TABLE,NUMERATOR,RECIPROCAL,VALID,H:tl.constexpr,W:tl.constexpr,SIZE:tl.constexpr,B:tl.constexpr):
    pixel=tl.program_id(0)*B+tl.arange(0,B);valid=pixel<H*W
    x=(pixel%W).to(tl.float32);y=(pixel//W).to(tl.float32)
    mx=tl.load(MOTION+pixel*2,valid,other=0).to(tl.float16).to(tl.float32)
    my=tl.load(MOTION+pixel*2+1,valid,other=0).to(tl.float16).to(tl.float32)
    bx,x0,gx,x3,tx,valid_x=axis(x+.5+mx,TABLE,SIZE)
    by,y0,gy,y3,ty,valid_y=axis(y+.5+my,TABLE,SIZE)
    w0=x0*gy;w1=y0*gx;w2=gx*gy;w3=y3*gx;w4=x3*gy
    total=((w0+w1)+w2)+w3;total=total+w4
    rcp,valid_total=reciprocal(total,TABLE,SIZE)
    # First multiply, then four explicit float32 FMAs in native tap order.
    value=sample(IMAGE,tx,by-1.,valid,H,W,4)*w1[:,None]
    value=tl.fma(sample(IMAGE,bx-1.,ty,valid,H,W,4),w0[:,None],value)
    value=tl.fma(sample(IMAGE,tx,ty,valid,H,W,4),w2[:,None],value)
    value=tl.fma(sample(IMAGE,tx,by+2.,valid,H,W,4),w3[:,None],value)
    value=tl.fma(sample(IMAGE,bx+2.,ty,valid,H,W,4),w4[:,None],value)
    channel=tl.arange(0,4)
    tl.store(NUMERATOR+pixel[:,None]*3+channel[None,:],value,valid[:,None]&(channel[None,:]<3))
    tl.store(RECIPROCAL+pixel,rcp,valid)
    tl.store(VALID+tl.program_id(0),tl.sum((valid&~(valid_x&valid_y&valid_total)).to(tl.int32),0)==0)


def forward(image,motion,table,*,block=64,warps=4):
    if image.device.type!='xpu' or image.dtype!=torch.float16 or not image.is_contiguous():
        raise ValueError('Expected contiguous half XPU history')
    if tuple(image.shape) not in ((256,256,3),(512,512,3)):
        raise ValueError('Expected the validated square history contract')
    h,w=image.shape[:2]
    if motion.shape!=(h,w,2) or motion.device!=image.device or motion.dtype not in (torch.float16,torch.float32) or not motion.is_contiguous():
        raise ValueError('Expected matching contiguous half/float motion')
    if table.shape!=(END_BITS-START_BITS,) or table.dtype!=torch.float32 or table.device!=image.device or not table.is_contiguous():
        raise ValueError('Expected complete native reciprocal interval')
    if block not in (32,64,128) or warps not in (4,8):raise ValueError('Unsupported sampling launch')
    numerator=torch.empty((h,w,3),device=image.device,dtype=torch.float32)
    result=torch.empty((h,w),device=image.device,dtype=torch.float32)
    flags=torch.empty((triton.cdiv(h*w,block),),device=image.device,dtype=torch.bool)
    kernel=_kernel[(triton.cdiv(h*w,block),)](image,motion,table,numerator,result,flags,h,w,len(table),block,
        num_warps=warps,num_stages=1,enable_fp_fusion=False)
    return (numerator,result,flags),kernel
