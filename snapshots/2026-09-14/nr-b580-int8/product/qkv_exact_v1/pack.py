"""Original normalization/FP8 boundaries stored directly in head/window order."""
import torch
import triton
import triton.language as tl
from nr_backend.triton_attention_normalize import _nan_left, rsqrt_half_clamped
from nr_backend.triton_cubic_fp8 import _half_fma_value
from exact_shortfp8_v1.fp8 import _round_fp8_half

@triton.jit
def _pack(X,Y,SCALE,INV,ROWS:tl.constexpr,H:tl.constexpr,W:tl.constexpr,HEADS:tl.constexpr,BR:tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    lanes = tl.arange(0, 8)
    kind=tl.program_id(1)
    offsets = (rows[:, None]*3+kind)*32+lanes[None,:]
    valid = rows[:, None] < ROWS
    x0 = tl.load(X + offsets, valid, other=0)
    x8 = tl.load(X + offsets + 8, valid, other=0)
    x16 = tl.load(X + offsets + 16, valid, other=0)
    x24 = tl.load(X + offsets + 24, valid, other=0)
    if kind<2:
        a = (x16.to(tl.float32) * x16.to(tl.float32)).to(tl.float16)
        a = _nan_left(_half_fma_value(x0, x0, a), x0, a)
        b = (x24.to(tl.float32) * x24.to(tl.float32)).to(tl.float16)
        b = _nan_left(_half_fma_value(x8, x8, b), x8, b)
        total = _nan_left((a.to(tl.float32) + b.to(tl.float32)).to(tl.float16), a, b)
        for mask in tl.static_range(3):
            xor_mask = 4 >> mask
            other = tl.gather(total, tl.broadcast_to((lanes ^ xor_mask)[None, :], (BR, 8)), 1)
            total = _nan_left((total.to(tl.float32) + other.to(tl.float32)).to(tl.float16), total, other)
        denominator = tl.gather(total, tl.full((BR, 1), 0, tl.int32), 1)
        scale = rsqrt_half_clamped(denominator)
        x0=_nan_left((x0.to(tl.float32)*scale.to(tl.float32)).to(tl.float16),x0,scale)
        x8=_nan_left((x8.to(tl.float32)*scale.to(tl.float32)).to(tl.float16),x8,scale)
        x16=_nan_left((x16.to(tl.float32)*scale.to(tl.float32)).to(tl.float16),x16,scale)
        x24=_nan_left((x24.to(tl.float32)*scale.to(tl.float32)).to(tl.float16),x24,scale)
    head=rows%HEADS
    token=rows//HEADS
    yy,xx=token//W,token%W
    pixel=tl.load(INV+(yy%8)*8+xx%8,rows<ROWS,other=0)
    out=((kind*HEADS+head)*(H//8)*(W//8)+(yy//8)*(W//8)+xx//8)*64+pixel
    offsets=out[:,None]*32+lanes[None,:]
    if kind==0:
        factor=tl.load(SCALE+head,rows<ROWS,other=0)[:,None]
        x0=_nan_left((x0.to(tl.float32)*factor.to(tl.float32)).to(tl.float16),x0,factor)
        x8=_nan_left((x8.to(tl.float32)*factor.to(tl.float32)).to(tl.float16),x8,factor)
        x16=_nan_left((x16.to(tl.float32)*factor.to(tl.float32)).to(tl.float16),x16,factor)
        x24=_nan_left((x24.to(tl.float32)*factor.to(tl.float32)).to(tl.float16),x24,factor)
    tl.store(Y+offsets,_round_fp8_half(x0),valid)
    tl.store(Y+offsets+8,_round_fp8_half(x8),valid)
    tl.store(Y+offsets+16,_round_fp8_half(x16),valid)
    tl.store(Y+offsets+24,_round_fp8_half(x24),valid)


def pack(z,scale,inverse,*,diagnostics=False):
    h,w,heads,three,lanes=z.shape
    if h%8 or w%8 or three!=3 or lanes!=32 or z.dtype!=torch.float16 or z.device.type!='xpu' or not z.is_contiguous():
        raise ValueError('Expected contiguous exact QKV in whole windows')
    output=torch.empty((3,heads,h//8,w//8,64,32),device=z.device,dtype=z.dtype)
    kernel=_pack[(triton.cdiv(h*w*heads,16),3)](z,output,scale,inverse,h*w*heads,h,w,heads,16,num_warps=4,enable_fp_fusion=False)
    resource=None
    if diagnostics:
        resource=dict(hash=kernel.hash,name=kernel.name,spills=kernel.n_spills,registers=kernel.n_regs)
        if kernel.n_spills!=0:raise RuntimeError(f'QKV pack spills: {kernel.n_spills}')
    return (output[0],output[1],output[2]),resource
