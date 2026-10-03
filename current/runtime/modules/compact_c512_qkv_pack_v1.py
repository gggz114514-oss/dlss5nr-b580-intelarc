"""Pack compact 12x12 C512 QKV into unchanged complete 16x16 attention windows.

Only source addressing changes. The full half normalization, scale, XOR sum
and FP8 stores match the selected native pack, including all padding rows.
"""
import triton
import triton.language as tl
from nr_backend.triton_attention_normalize import _nan_left, rsqrt_half_clamped
from native_half_attention_fma_v1 import half_fma_attention as _half_fma_value
from short_fp8_v2 import round_half as _round_fp8_half


@triton.jit
def _pack(Z, SCALE, ORDER, Q, K, V, SY: tl.constexpr, SX: tl.constexpr, BR: tl.constexpr):
    COUNT: tl.constexpr = 4096
    HEADS: tl.constexpr = 16
    ROWS: tl.constexpr = 2
    COLS: tl.constexpr = 2
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    family = tl.program_id(1)
    lane = tl.arange(0, 8)
    pixel = r % 64
    window = r // 64
    col = window % COLS
    row = (window // COLS) % ROWS
    head = window // (COLS * ROWS)
    physical = tl.load(ORDER + pixel).to(tl.int32)
    y, x = row * 8 + physical // 8, col * 8 + physical % 8
    iy, ix = y - SY, x - SX
    inside = (iy >= 0) & (iy < 12) & (ix >= 0) & (ix < 12)
    offset = (((iy * 12 + ix) * HEADS + head) * 3 + family) * 32
    off = offset[:, None] + lane[None, :]
    valid = r[:, None] < COUNT
    source_valid = valid & inside[:, None]
    # The old projection computes +0 from padded zeros and finite weights.
    # Retain all normalization/scale/FP8 operations even for those zero rows.
    x0 = tl.load(Z + off, source_valid, other=0)
    x8 = tl.load(Z + off + 8, source_valid, other=0)
    x16 = tl.load(Z + off + 16, source_valid, other=0)
    x24 = tl.load(Z + off + 24, source_valid, other=0)
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
            scale = tl.load(SCALE + head, r < COUNT, other=0)[:, None]
            x0 = _nan_left((x0.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x0, scale)
            x8 = _nan_left((x8.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x8, scale)
            x16 = _nan_left((x16.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x16, scale)
            x24 = _nan_left((x24.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x24, scale)
    output = tl.where(family == 0, Q, tl.where(family == 1, K, V))
    dest = r[:, None] * 32 + lane[None, :]
    tl.store(output + dest, _round_fp8_half(x0), valid)
    tl.store(output + dest + 8, _round_fp8_half(x8), valid)
    tl.store(output + dest + 16, _round_fp8_half(x16), valid)
    tl.store(output + dest + 24, _round_fp8_half(x24), valid)
