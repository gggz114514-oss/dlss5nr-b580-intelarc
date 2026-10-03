"""Consume native head/window/pixel layout directly and scatter cropped HWC output.

The matrix rows are independent pixels in native window order. The K32 reduction
order is unchanged. The producer already quantized attention operands; round the
scaled half residual separately, add it to the FP32 dot, then round once to half.
No unpacked attention, standalone FP8 tensor, or scaled residual is materialized.
"""
import torch
import triton
import triton.language as tl
from spill_preflight_v1 import select

@triton.jit
def _project(X,W,RESIDUAL,SCALE,ORDER,OUT,HP:tl.constexpr,WP:tl.constexpr,
             H:tl.constexpr,WIDTH:tl.constexpr,C:tl.constexpr,SY:tl.constexpr,SX:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,32)
    window=rows//64;local=rows%64
    physical=tl.load(ORDER+local).to(tl.int32)
    y=(window//(WP//8))*8+physical//8
    x=(window%(WP//8))*8+physical%8
    pixel=y*WP+x
    total=tl.full((BM,BN),0,tl.float32)
    for head in range(C//32):
        av=tl.load(X+(head*(HP//8)*(WP//8)+window[:,None])*2048+local[:,None]*32+kk[None,:],rows[:,None]<HP*WP,other=0)
        wv=tl.load(W+(head*32+kk[:,None])*C+cols[None,:],cols[None,:]<C,other=0)
        total=tl.dot(av,wv,total,out_dtype=tl.float32)
    r=tl.load(RESIDUAL+pixel[:,None]*C+cols[None,:],(rows[:,None]<HP*WP)&(cols[None,:]<C),other=0)
    s=tl.load(SCALE+cols,cols<C,other=0)
    initial=(r.to(tl.float32)*s[None,:].to(tl.float32)).to(tl.float16)
    result=(total+initial.to(tl.float32)).to(tl.float16)
    oy=y-SY;ox=x-SX
    valid=(rows[:,None]<HP*WP)&(cols[None,:]<C)&(oy[:,None]>=0)&(oy[:,None]<H)&(ox[:,None]>=0)&(ox[:,None]<WIDTH)
    tl.store(OUT+(oy[:,None]*WIDTH+ox[:,None])*C+cols[None,:],result,valid)

def forward(windows,weight,residual,scale,order,*,height,width,shift,tile):
    if windows.ndim!=5 or tuple(windows.shape[-2:])!=(64,32):raise ValueError('Expected head,row,col,64,32 attention output')
    heads,rows,cols=windows.shape[:3];channels=heads*32;hp,wp=rows*8,cols*8
    if channels not in (64,128,256):raise ValueError('Unsupported multihead block channel count')
    if weight.shape!=(channels,channels) or residual.shape!=(hp,wp,channels) or scale.shape!=(channels,):raise ValueError('Unexpected projection/residual shape')
    for t in (windows,weight,residual,scale):
        if t.device.type!='xpu' or t.device!=windows.device or t.dtype!=torch.float16 or not t.is_contiguous():raise ValueError('Expected contiguous half XPU operands')
    if order.shape!=(64,) or order.dtype!=torch.int64 or order.device!=windows.device or not order.is_contiguous():raise ValueError('Expected owned int64 pixel order')
    sy,sx=shift
    if sy not in (0,4) or sx not in (0,4) or min(height,width)<=0 or height+sy>hp or width+sx>wp:raise ValueError('Invalid output crop')
    bm,bn,bk,warps=tile
    if bm not in (16,32,64) or bn not in (32,64,128) or bk!=32 or warps!=4:raise ValueError('Unreviewed matrix tile')
    out=torch.empty((height,width,channels),device=windows.device,dtype=torch.float16)
    configs=[(bm,bn)]
    if (bm,bn)!=(16,32):configs.append((16,32))
    args_for=lambda config:(windows,weight,residual,scale,order,out,hp,wp,height,width,channels,sy,sx,*config)
    grid_for=lambda config:(triton.cdiv(hp*wp,config[0]),triton.cdiv(channels,config[1]))
    selected,compiled,selection=select(_project,configs,args_for,grid_for,num_warps=warps,enable_fp_fusion=False)
    kernel=_project[grid_for(selected)](*args_for(selected),num_warps=warps,enable_fp_fusion=False)
    assert kernel is compiled
    return out,kernel,selection
