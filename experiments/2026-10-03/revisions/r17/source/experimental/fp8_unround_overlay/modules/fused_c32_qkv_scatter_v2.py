"""Normalize/scale/FP8 a <=32768-row projected C32 chunk into its full windows.

Input is contiguous [rows,96]. START is the global row, independent of window or
scanline boundaries. The model owns inverse permutation and disjoint Q/K/V outputs.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_attention_normalize import _nan_left, rsqrt_half_clamped
from nr_backend.triton_cubic_fp8 import _half_fma_value
from nr_backend.triton_fp8 import _round_fp8_half


@triton.jit
def _scatter(Z, SCALE, INVERSE, Q, K, V, START, COUNT: tl.constexpr,
             W: tl.constexpr, COLS: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    family = tl.program_id(1)
    lane = tl.arange(0, 8)
    global_row = START + r
    y, x = global_row // W, global_row % W
    pixel = tl.load(INVERSE + (y % 8) * 8 + x % 8).to(tl.int32)
    window = (y // 8) * COLS + x // 8
    off = r[:, None] * 96 + family * 32 + lane[None, :]
    valid = r[:, None] < COUNT
    x0 = tl.load(Z + off, valid, other=0)
    x8 = tl.load(Z + off + 8, valid, other=0)
    x16 = tl.load(Z + off + 16, valid, other=0)
    x24 = tl.load(Z + off + 24, valid, other=0)
    if family < 2:
        a = (x16.to(tl.float32) * x16.to(tl.float32)).to(tl.float16)
        a = _nan_left(_half_fma_value(x0, x0, a), x0, a)
        b = (x24.to(tl.float32) * x24.to(tl.float32)).to(tl.float16)
        b = _nan_left(_half_fma_value(x8, x8, b), x8, b)
        total = _nan_left((a.to(tl.float32) + b.to(tl.float32)).to(tl.float16), a, b)
        for mask in tl.static_range(3):
            other = tl.gather(total, tl.broadcast_to((lane ^ (4 >> mask))[None, :], (BR, 8)), 1)
            total = _nan_left((total.to(tl.float32) + other.to(tl.float32)).to(tl.float16), total, other)
        denominator = tl.gather(total, tl.full((BR, 1), 0, tl.int32), 1)
        scale = rsqrt_half_clamped(denominator)
        x0 = _nan_left((x0.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x0, scale)
        x8 = _nan_left((x8.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x8, scale)
        x16 = _nan_left((x16.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x16, scale)
        x24 = _nan_left((x24.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x24, scale)
        if family == 0:
            q_scale = tl.load(SCALE)
            x0 = _nan_left((x0.to(tl.float32) * q_scale.to(tl.float32)).to(tl.float16), x0, q_scale)
            x8 = _nan_left((x8.to(tl.float32) * q_scale.to(tl.float32)).to(tl.float16), x8, q_scale)
            x16 = _nan_left((x16.to(tl.float32) * q_scale.to(tl.float32)).to(tl.float16), x16, q_scale)
            x24 = _nan_left((x24.to(tl.float32) * q_scale.to(tl.float32)).to(tl.float16), x24, q_scale)
    output = tl.where(family == 0, Q, tl.where(family == 1, K, V))
    dest = (window * 64 + pixel)[:, None] * 32 + lane[None, :]
    tl.store(output + dest, _round_fp8_half(x0), valid)
    tl.store(output + dest + 8, _round_fp8_half(x8), valid)
    tl.store(output + dest + 16, _round_fp8_half(x16), valid)
    tl.store(output + dest + 24, _round_fp8_half(x24), valid)


def scatter(z,scale,inverse,outputs,start,*,rows=16):
    if z.ndim!=2 or z.shape[1]!=96 or not 0<z.shape[0]<=32768:
        raise ValueError('Expected a nonempty C32 projection chunk no larger than32768 rows')
    if z.device.type!='xpu' or z.dtype!=torch.float16 or not z.is_contiguous():
        raise ValueError('Expected contiguous half XPU projected chunk')
    if scale.device!=z.device or scale.dtype!=torch.float16 or scale.shape!=(1,) or not scale.is_contiguous():
        raise ValueError('Expected one owned half query scale')
    if inverse.device!=z.device or inverse.dtype!=torch.int64 or inverse.shape!=(64,) or not inverse.is_contiguous():
        raise ValueError('Expected owned int64 inverse pixel permutation')
    if len(outputs)!=3:raise ValueError('Expected three disjoint owned output tensors')
    shape=outputs[0].shape
    if len(shape)!=4 or shape[-2:]!=(64,32) or min(shape)<=0:
        raise ValueError('Expected row,col,64,32 output windows')
    for t in outputs:
        if t.shape!=shape or t.device!=z.device or t.dtype!=torch.float16 or not t.is_contiguous():
            raise ValueError('Expected matching contiguous half XPU outputs')
    height,width=shape[0]*8,shape[1]*8
    if not isinstance(start,int) or isinstance(start,bool) or start<0 or start+z.shape[0]>height*width:
        raise ValueError('Chunk lies outside output canvas')
    if rows not in (8,16,32,64):raise ValueError('Unsupported row tile')
    return _scatter[(triton.cdiv(z.shape[0],rows),3)](
        z,scale,inverse,*outputs,start,z.shape[0],width,shape[1],rows,num_warps=4,enable_fp_fusion=False)
