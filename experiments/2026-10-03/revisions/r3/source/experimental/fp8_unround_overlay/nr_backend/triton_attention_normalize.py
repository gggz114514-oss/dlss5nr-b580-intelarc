"""C32 normalization with the original half squares/FMA/XOR tree in one kernel."""
import torch
import triton
import triton.language as tl
from .triton_cubic_fp8 import _half_fma_value


@triton.jit
def _nan_left(result, left, right):
    # PyTorch XPU binary operations keep the left NaN, then the right NaN.
    # Explicit integer selection prevents reassociation from changing payloads.
    lbits = left.to(tl.uint16, bitcast=True).to(tl.int32)
    rbits = right.to(tl.uint16, bitcast=True).to(tl.int32)
    bits = result.to(tl.uint16, bitcast=True).to(tl.int32)
    bits = tl.where((rbits & 0x7fff) > 0x7c00, rbits | 0x200, bits)
    bits = tl.where((lbits & 0x7fff) > 0x7c00, lbits | 0x200, bits)
    return bits.to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _normalize(X, Y, ROWS: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    lanes = tl.arange(0, 8)
    offsets = rows[:, None] * 32 + lanes[None, :]
    valid = rows[:, None] < ROWS
    x0 = tl.load(X + offsets, valid, other=0)
    x8 = tl.load(X + offsets + 8, valid, other=0)
    x16 = tl.load(X + offsets + 16, valid, other=0)
    x24 = tl.load(X + offsets + 24, valid, other=0)
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
    tl.store(Y + offsets, _nan_left((x0.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x0, scale), valid)
    tl.store(Y + offsets + 8, _nan_left((x8.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x8, scale), valid)
    tl.store(Y + offsets + 16, _nan_left((x16.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x16, scale), valid)
    tl.store(Y + offsets + 24, _nan_left((x24.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x24, scale), valid)


def normalize_c32(x, *, rows=16, warps=4):
    if x.device.type != 'xpu' or x.shape[-1] != 32:
        raise ValueError('C32 XPU vectors required')
    source = x.half().contiguous()
    output = torch.empty_like(source)
    count = source.numel() // 32
    if count:
        _normalize[(triton.cdiv(count, rows),)](source, output, count, rows, num_warps=warps, enable_fp_fusion=False)
    return output


@triton.jit
def rsqrt_half_clamped(x):
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
    value = tl.maximum(x.to(tl.float32), 6.198883056640625e-05)
    result = tl.rsqrt(value).to(tl.float16)
    output_bits = result.to(tl.uint16, bitcast=True).to(tl.int32)
    output_bits = tl.where(bits & 32767 > 31744, bits | 512, output_bits)
    return output_bits.to(tl.uint16).to(tl.float16, bitcast=True)
