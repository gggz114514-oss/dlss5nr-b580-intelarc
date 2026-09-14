"""Original K16 integer projection directly reading head/window layouts."""
import torch
import triton
import triton.language as tl
from nr_backend.triton_math import _exponent, _scaled_integer_to_half
from nr_backend.triton_fp8 import _round_fp8_half


@triton.jit
def _project(A, W, INIT, INV, OUT, M: tl.constexpr, N: tl.constexpr,
             TOTAL_K: tl.constexpr, START: tl.constexpr, COUNT: tl.constexpr,
             INITIALIZED: tl.constexpr, WINDOW: tl.constexpr, WIDTH: tl.constexpr,
             WR: tl.constexpr, WC: tl.constexpr, SY: tl.constexpr, SX: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.program_id(1)*BN + tl.arange(0, BN)
    group = tl.arange(0, 16)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    offset = rows[:, None]*N + cols[None, :]
    if INITIALIZED:
        acc = tl.load(INIT + offset, valid, other=0).to(tl.float32)
    else:
        acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(START, START + COUNT, 16):
        channel = start + group
        if WINDOW:
            yy, xx = rows // WIDTH + SY, rows % WIDTH + SX
            pixel = tl.load(INV + (yy % 8)*8 + xx % 8, rows < M, other=0)
            window = (yy // 8)*WC + xx // 8
            address = ((channel[None, :]//32 * WR*WC + window[:, None])*64 + pixel[:, None])*32 + channel[None, :] % 32
        else:
            address = ((channel[None, :]//32)*M + rows[:, None])*32 + channel[None, :] % 32
        av = _round_fp8_half(tl.load(A + address, rows[:, None] < M, other=0)).to(tl.float32)
        wv = tl.load(W + channel[:, None]*N + cols[None, :], cols[None, :] < N, other=0).to(tl.float32)
        ea, ew = _exponent(av), _exponent(wv)
        ea = tl.where(av == 0, -1000, tl.maximum(ea, -6))
        ew = tl.where(wv == 0, -1000, tl.maximum(ew, -6))
        exponent = tl.minimum(tl.maximum(tl.maximum(tl.max(ea[:, :, None]+ew[None, :, :], 1), _exponent(acc)), -50), 50)
        scale = ((127+13-exponent) << 23).to(tl.float32, bitcast=True)
        product = av[:, :, None]*wv[None, :, :]
        summed = tl.sum((product*scale[:, None, :]).to(tl.int32), 1)+(acc*scale).to(tl.int32)
        acc = _scaled_integer_to_half(summed, exponent-13).to(tl.float32)
    tl.store(OUT + offset, acc.to(tl.float16), valid)


def project(packed, weight, initial, *, geometry=None, inverse=None, parts=1):
    if packed.device.type != 'xpu' or packed.dtype != torch.float16 or not packed.is_contiguous():
        raise ValueError('Expected contiguous head/window half input')
    total, channels = weight.shape
    if total % (parts*16) or not weight.is_contiguous():
        raise ValueError('Whole K16 partitions and contiguous original weights required')
    if geometry is None:
        heads, tokens, lanes = packed.shape
        if heads*lanes != total or lanes != 32 or parts != 4:
            raise ValueError('Expected native ViT head layout and four partitions')
        shape = (tokens, channels)
        width, wr, wc, sy, sx = 1, 1, 1, 0, 0
    else:
        height, width, sy, sx = geometry
        heads, wr, wc, pixels, lanes = packed.shape
        if heads*32 != total or pixels != 64 or lanes != 32 or parts != 1:
            raise ValueError('Expected native Swin window layout')
        shape = (height, width, channels)
        tokens = height*width
    if initial.shape != shape or initial.device != packed.device or initial.dtype != torch.float16:
        raise ValueError('Initial residual geometry mismatch')
    result = None
    for part in range(parts):
        out = torch.empty(shape, device=packed.device, dtype=torch.float16)
        _project[(triton.cdiv(tokens, 4), triton.cdiv(channels, 32))](
            packed, weight, initial.contiguous(), packed if inverse is None else inverse, out,
            tokens, channels, total, part*(total//parts), total//parts, part == 0,
            geometry is not None, width, wr, wc, sy, sx, 4, 32,
            num_warps=1, enable_fp_fusion=False)
        result = out if result is None else (result + out).half()
    return result
