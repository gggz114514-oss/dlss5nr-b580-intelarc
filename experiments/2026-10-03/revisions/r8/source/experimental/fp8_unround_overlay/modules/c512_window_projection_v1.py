"""Read FP8 half C512 head/windows directly into the original HWC projection.

Same output pixels, K32 dot order and half scaled residual as the selected fast
provider. No attention unpack, cropped contiguous copy or scaled skip tensor.
The producer must supply the already rounded FP8 attention boundary. Return
unquantized half full: encoder pooling needs it before output FP8 rounding.
"""
import torch
import triton
import triton.language as tl
from spill_preflight_v1 import select


@triton.jit
def _project(X,W,RESIDUAL,SCALE,INVERSE,OUT,HP:tl.constexpr,WP:tl.constexpr,
             H:tl.constexpr,WIDTH:tl.constexpr,SY:tl.constexpr,SX:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,32)
    y=rows//WIDTH+SY
    x=rows%WIDTH+SX
    window=(y//8)*(WP//8)+x//8
    local=tl.load(INVERSE+(y%8)*8+x%8).to(tl.int32)
    total=tl.full((BM,BN),0,tl.float32)
    for head in range(16):
        av=tl.load(X+(head*(HP//8)*(WP//8)+window[:,None])*2048
                     +local[:,None]*32+kk[None,:],rows[:,None]<H*WIDTH,other=0)
        wv=tl.load(W+(head*32+kk[:,None])*512+cols[None,:],cols[None,:]<512,other=0)
        total=tl.dot(av,wv,total,out_dtype=tl.float32)
    offset=rows[:,None]*512+cols[None,:]
    valid=(rows[:,None]<H*WIDTH)&(cols[None,:]<512)
    r=tl.load(RESIDUAL+offset,valid,other=0)
    s=tl.load(SCALE+cols,cols<512,other=0)
    initial=(r.to(tl.float32)*s[None,:].to(tl.float32)).to(tl.float16)
    result=(total+initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT+offset,result,valid)


def forward(windows,weight,residual,scale,inverse,*,shift):
    if windows.ndim!=5 or windows.shape[0]!=16 or tuple(windows.shape[-2:])!=(64,32):
        raise ValueError('Expected C512 head,row,col,64,32 FP8 half attention')
    _,rows,cols,_,_=windows.shape
    hp,wp=rows*8,cols*8
    if residual.ndim!=3 or residual.shape[-1]!=512 or weight.shape!=(512,512) or scale.shape!=(512,):
        raise ValueError('Unexpected C512 projection operands')
    height,width=residual.shape[:2]
    sy,sx=shift
    if (sy not in (0,4) or sx not in (0,4) or min(height,width)<=0
            or height%4 or width%4 or height+sy>hp or width+sx>wp):
        raise ValueError('Invalid C512 window region')
    for t in (windows,weight,residual,scale):
        if t.device.type!='xpu' or t.device!=windows.device or t.dtype!=torch.float16 or not t.is_contiguous():
            raise ValueError('Expected contiguous half XPU operands')
    if inverse.shape!=(64,) or inverse.dtype!=torch.int64 or inverse.device!=windows.device or not inverse.is_contiguous():
        raise ValueError('Expected owned int64 inverse pixel order')
    out=torch.empty_like(residual)
    # Match the selected C512 dense BM16/BN32 rather than introduce a tile sweep.
    config=(16,32)
    args=(windows,weight,residual,scale,inverse,out,hp,wp,height,width,sy,sx,*config)
    grid=(triton.cdiv(height*width,16),16)
    _,compiled,selection=select(_project,[config],lambda _:args,lambda _:grid,
        num_warps=4,enable_fp_fusion=False)
    kernel=_project[grid](*args,num_warps=4,enable_fp_fusion=False)
    assert kernel is compiled
    return out,kernel,selection
