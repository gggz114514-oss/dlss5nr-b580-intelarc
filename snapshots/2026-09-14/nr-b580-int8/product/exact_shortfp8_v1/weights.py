"""Original ordered 64-key half reduction, optionally reciprocal/multiply/E4M3."""
import torch
import triton
import triton.language as tl
from nr_backend.triton_attention_normalize import _nan_left
from .fp8 import _round_fp8_half


@triton.jit
def reciprocal_half_clamped(x):
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
    value = tl.maximum(x.to(tl.float32), 0.00006198883056640625)
    result = tl.div_rn(tl.full((), 1.0, tl.float32), value).to(tl.float16)
    output_bits = result.to(tl.uint16, bitcast=True).to(tl.int32)
    output_bits = tl.where((bits & 0x7fff) > 0x7c00, bits | 0x200, output_bits)
    return output_bits.to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _add_half(a, b):
    return _nan_left((a.to(tl.float32) + b.to(tl.float32)).to(tl.float16), a, b)


@triton.jit
def _weights(X, Y, ROWS: tl.constexpr, BR: tl.constexpr, NORMALIZE: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    lanes = tl.arange(0, 8)
    offsets = rows[:, None] * 64 + lanes[None, :]
    valid = rows[:, None] < ROWS
    v0 = tl.load(X + offsets, valid, other=0)
    v8 = tl.load(X + offsets + 8, valid, other=0)
    v16 = tl.load(X + offsets + 16, valid, other=0)
    v24 = tl.load(X + offsets + 24, valid, other=0)
    v32 = tl.load(X + offsets + 32, valid, other=0)
    v40 = tl.load(X + offsets + 40, valid, other=0)
    v48 = tl.load(X + offsets + 48, valid, other=0)
    v56 = tl.load(X + offsets + 56, valid, other=0)
    p0 = _add_half(v0, v8)
    p1 = _add_half(v16, v24)
    p2 = _add_half(v32, v40)
    p3 = _add_half(v48, v56)
    partial = _add_half(_add_half(_add_half(p0, p1), p2), p3)
    small = tl.arange(0, 2)
    t0 = tl.gather(partial, tl.broadcast_to(small[None, :], (BR, 2)), 1)
    t2 = tl.gather(partial, tl.broadcast_to((small + 2)[None, :], (BR, 2)), 1)
    t4 = tl.gather(partial, tl.broadcast_to((small + 4)[None, :], (BR, 2)), 1)
    t6 = tl.gather(partial, tl.broadcast_to((small + 6)[None, :], (BR, 2)), 1)
    total = _add_half(_add_half(_add_half(t0, t2), t4), t6)
    a = tl.gather(total, tl.full((BR, 1), 0, tl.int32), 1)
    b = tl.gather(total, tl.full((BR, 1), 1, tl.int32), 1)
    denominator = _add_half(a, b)
    if NORMALIZE:
        reciprocal = reciprocal_half_clamped(denominator)
        values = (v0, v8, v16, v24, v32, v40, v48, v56)
        for segment in tl.static_range(8):
            x = values[segment]
            product = _nan_left((x.to(tl.float32) * reciprocal.to(tl.float32)).to(tl.float16), x, reciprocal)
            tl.store(Y + offsets + segment * 8, _round_fp8_half(product), valid)
    else:
        tl.store(Y + rows[:, None], denominator, valid)


def _run(x, normalize, rows, warps):
    if x.shape[-1] != 64:
        raise ValueError('Expected 64 key weights')
    if x.device.type != 'xpu':
        raise ValueError('Attention weight fusion requires XPU tensors')
    source = x.half().contiguous()
    shape = source.shape if normalize else (*source.shape[:-1], 1)
    output = torch.empty(shape, dtype=torch.float16, device=x.device)
    count = source.numel() // 64
    if count:
        _weights[(triton.cdiv(count, rows),)](source, output, count, rows, normalize,
            num_warps=warps, enable_fp_fusion=False)
    return output


def row_sum64(x, *, rows=32, warps=4):
    return _run(x, False, rows, warps)


def normalize_weights(x, *, rows=16, warps=1):
    return _run(x, True, rows, warps)
