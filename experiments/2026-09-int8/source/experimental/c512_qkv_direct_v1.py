"""Experimental direct C512 QKV dot -> half normalization -> FP8 windows.

BN32/BK32, no projected Z buffer. Owned outputs need BOTH pad and dense calls.
Same normalization arithmetic as compact_c512_qkv_pack_v1; no INT8 conversion.
Fixed NR256 geometry and the validated native four-quadrant pixel permutation.
"""
import triton
import triton.language as tl
from nr_backend.triton_attention_normalize import _nan_left, rsqrt_half_clamped
from native_half_attention_fma_v1 import half_fma_attention as _half_fma_value
from short_fp8_v2 import round_half as _round_fp8_half


@triton.jit
def _finish(x0, x8, x16, x24, SCALE, head, family, BR: tl.constexpr):
    lane = tl.arange(0, 8)
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
            scale = tl.load(SCALE + head, head < 16, other=0)[:, None]
            x0 = _nan_left((x0.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x0, scale)
            x8 = _nan_left((x8.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x8, scale)
            x16 = _nan_left((x16.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x16, scale)
            x24 = _nan_left((x24.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x24, scale)
    return _round_fp8_half(x0), _round_fp8_half(x8), _round_fp8_half(x16), _round_fp8_half(x24)


@triton.jit
def _dense_pack(X, W, SCALE, Q, K, V, SY: tl.constexpr, SX: tl.constexpr, BM: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    group = tl.program_id(1)
    col = group*32 + tl.arange(0, 32)
    kk = tl.arange(0, 32)
    total = tl.full((BM, 32), 0, tl.float32)
    for block in range(16):
        k = block*32 + kk
        x = tl.load(X + row[:, None]*512 + k[None, :], row[:, None] < 144, other=0)
        w = tl.load(W + k[:, None]*1536 + col[None, :])
        total = tl.dot(x, w, total, out_dtype=tl.float32)
    # Preserve the original intermediate half store boundary in registers.
    value = total.to(tl.float16)
    lane = tl.arange(0, 8)
    index = tl.broadcast_to(lane[None, :], (BM, 8))
    x0 = tl.gather(value, index, 1)
    x8 = tl.gather(value, index+8, 1)
    x16 = tl.gather(value, index+16, 1)
    x24 = tl.gather(value, index+24, 1)
    head = tl.full((BM,), 0, tl.int32) + group//3
    family = group % 3
    x0, x8, x16, x24 = _finish(x0, x8, x16, x24, SCALE, head, family, BM)
    y, x = row//12 + SY, row%12 + SX
    window = y//8*2 + x//8
    pixel = ((y%8)//4*2 + (x%8)//4)*16 + (y%4)*4 + x%4
    dest = ((head*4 + window)*64 + pixel)[:, None]*32 + lane[None, :]
    output = tl.where(family == 0, Q, tl.where(family == 1, K, V))
    valid = row[:, None] < 144
    tl.store(output + dest, x0, valid)
    tl.store(output + dest+8, x8, valid)
    tl.store(output + dest+16, x16, valid)
    tl.store(output + dest+24, x24, valid)


@triton.jit
def _pad(SCALE, ORDER, Q, K, V, SY: tl.constexpr, SX: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0)*BR + tl.arange(0, BR)
    family = tl.program_id(1)
    pixel, window, head = r%64, (r//64)%4, r//256
    physical = tl.load(ORDER + pixel)
    iy = window//2*8 + physical//8 - SY
    ix = window%2*8 + physical%8 - SX
    inside = (iy >= 0) & (iy < 12) & (ix >= 0) & (ix < 12)
    zero = tl.full((BR, 8), 0, tl.float16)
    x0, x8, x16, x24 = _finish(zero, zero, zero, zero, SCALE, head, family, BR)
    lane = tl.arange(0, 8)
    dest = r[:, None]*32 + lane[None, :]
    output = tl.where(family == 0, Q, tl.where(family == 1, K, V))
    valid = ((r < 4096) & ~inside)[:, None]
    tl.store(output + dest, x0, valid)
    tl.store(output + dest+8, x8, valid)
    tl.store(output + dest+16, x16, valid)
    tl.store(output + dest+24, x24, valid)


def geometry(shift):
    sy, sx = shift
    assert sy in (0, 4) and sx in (0, 4)
    return [((i//12+sy)//8*2+(i%12+sx)//8,
             (((i//12+sy)%8)//4*2+((i%12+sx)%8)//4)*16
             +((i//12+sy)%4)*4+(i%12+sx)%4) for i in range(144)]
